"""Command-line entry point. Mirrors the Rust `src/args.rs` flags."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .dataset import Dataset, load_corpus
from .eval import evaluate, gen_problems, make_corpus
from .grpo import GrpoConfig, grpo_train
from .model import GptConfig
from .rft import rft_train
from .tokenizer import BpeTokenizer, CharTokenizer
from .train import build_model, generate_text, train_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="smolgpt", description="A smol GPT model")
    p.add_argument(
        "-m", "--model-type", choices=["gpt", "bigram", "ngram"], default="gpt"
    )
    p.add_argument("--ngram-order", type=int, default=3, help="n for the n-gram model")
    p.add_argument(
        "-k", "--tokenizer", choices=["char", "bpe"], default="bpe",
        help="How to tokenize the corpus (default: bpe)",
    )
    p.add_argument("--vocab-size", type=int, default=1024, help="Target BPE vocab size")
    p.add_argument(
        "-d", "--dataset-path", default="../data/tinyshakespeare.txt", type=Path
    )
    p.add_argument("-e", "--epochs", type=int, default=100)
    p.add_argument("-p", "--model-path", type=Path, default=None)
    # Modes
    p.add_argument("--train", action="store_true", help="Train the model")
    p.add_argument("--generate", action="store_true", help="Generate from the model")
    p.add_argument("--eval", action="store_true", help="Run the arithmetic eval harness")
    p.add_argument("--rft", action="store_true", help="Rejection-sampling fine-tuning")
    p.add_argument("--grpo", action="store_true", help="GRPO post-training")
    p.add_argument("--grpo-mode", choices=["lite", "full"], default="lite")
    p.add_argument("--rft-rounds", type=int, default=20)
    p.add_argument("--grpo-rounds", type=int, default=30)
    p.add_argument("--group-size", type=int, default=8, help="GRPO group size")
    # Arithmetic task (for eval/rft/grpo, or arithmetic pre-training)
    p.add_argument(
        "--arithmetic", action="store_true",
        help="Use a synthetic arithmetic corpus instead of --dataset-path",
    )
    p.add_argument("--arith-min", type=int, default=0)
    p.add_argument("--arith-max", type=int, default=99)
    p.add_argument("--arith-ops", default="+")
    p.add_argument("--arith-samples", type=int, default=20000)
    p.add_argument("--arith-eval-samples", type=int, default=200)
    # GPT architecture (must match between train and later load).
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def default_model_path(model_type: str, tokenizer: str, arithmetic: bool) -> Path:
    # Keep char/BPE (and arithmetic vs text) models separate — incompatible vocabs.
    suffix = "arith" if arithmetic else tokenizer
    return Path(f"{model_type}-{suffix}.pt")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    modes = [args.train, args.generate, args.eval, args.rft, args.grpo]
    if not any(modes):
        raise SystemExit("Specify at least one of --train/--generate/--eval/--rft/--grpo")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = None
    if args.seed is not None:
        torch.manual_seed(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # --- Corpus + tokenizer ---
    if args.arithmetic or args.eval or args.rft or args.grpo:
        train_problems = gen_problems(
            args.arith_samples, args.arith_min, args.arith_max, args.arith_ops,
            seed=args.seed,
        )
        corpus = make_corpus(train_problems)
        print(f"Arithmetic corpus: {len(train_problems)} problems")
        arithmetic = True
    else:
        corpus = load_corpus(args.dataset_path)
        arithmetic = False

    if args.tokenizer == "char":
        tokenizer = CharTokenizer(corpus)
    else:
        tokenizer = BpeTokenizer.train(corpus, args.vocab_size)
    print(f"Tokenizer: {args.tokenizer}, vocab size: {tokenizer.vocab_size}")

    model_path = args.model_path or default_model_path(
        args.model_type, args.tokenizer, arithmetic
    )

    config = GptConfig(
        block_size=args.block_size, n_embd=args.n_embd,
        n_head=args.n_head, n_layer=args.n_layer,
    )
    model = build_model(
        args.model_type, tokenizer.vocab_size, config, device, args.ngram_order
    )
    block_size = model.block_size

    if Path(model_path).exists():
        print(f"Loading {args.model_type} model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Creating new {args.model_type} model")

    # --- Base training ---
    if args.train:
        data = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
        print(f"Encoded corpus: {data.shape[0]} tokens")
        dataset = Dataset(data, train_ratio=0.9)
        train_model(
            model, dataset, model_path, args.epochs, args.num_batches,
            block_size, args.lr, device, generator,
        )

    # --- Post-training (never overwrites the base checkpoint) ---
    def eval_problems():
        return gen_problems(
            args.arith_eval_samples, args.arith_min, args.arith_max,
            args.arith_ops, seed=(args.seed + 1) if args.seed is not None else None,
        )

    if args.rft:
        rft_path = model_path.with_name(model_path.stem + "-rft.pt")
        rft_train(model, tokenizer, eval_problems(), rft_path, rounds=args.rft_rounds,
                  lr=args.lr, device=device, generator=generator)

    if args.grpo:
        suffix = "-grpo-full.pt" if args.grpo_mode == "full" else "-grpo.pt"
        grpo_path = model_path.with_name(model_path.stem + suffix)
        grpo_train(
            model, tokenizer, eval_problems(), grpo_path,
            GrpoConfig(mode=args.grpo_mode, lr=args.lr, rounds=args.grpo_rounds,
                       group_size=args.group_size),
            device, generator,
        )

    if args.eval:
        res = evaluate(model, tokenizer, eval_problems(), device)
        print(f"\nArithmetic eval: {res.correct}/{res.total} = {res.accuracy:.1%}")
        for prompt, expected, got in res.samples[:10]:
            mark = "✓" if expected == got else "✗"
            print(f"  {mark} {prompt}{got}  (expected {expected})")

    if args.generate:
        if not args.train and not Path(model_path).exists():
            raise SystemExit(
                f"No trained checkpoint at {model_path}; run with --train first "
                "(generating from a random model would only produce garbage)."
            )
        print(f"Generating from {args.model_type} model ({model_path})")
        text = generate_text(model, tokenizer, 500, device, generator)
        print(f"Generated text: {text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
