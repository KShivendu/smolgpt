# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SmolGPT: a from-scratch GPT/bigram/n-gram language model implementation in Rust using `candle` (not PyTorch/tokenizers/tiktoken — everything, including the BPE tokenizer, model registry, and eval harness, is hand-rolled). It trains on Tiny Shakespeare or synthetic arithmetic corpora, and includes RFT and GRPO post-training loops plus a local web UI for browsing trained models.

## Commands

```sh
cargo build -r                          # release build
cargo run -r -- --train --generate      # train + sample from a fresh model
cargo test                              # run all tests (unit tests live inline in src/*.rs via #[cfg(test)])
cargo test <name>                       # run a single test by name (supports rstest cases)
cargo run -r -- --serve                 # local web UI at 127.0.0.1:8080 (browse models, run evals, REPL)

# Generate synthetic arithmetic corpora (separate example binary)
cargo run --release --example gen_arithmetic -- --output data/arithmetic-add.txt --samples 200000 --min 0 --max 99 --ops + --max-result 99
```

CLI flags are defined in `src/args.rs` (clap derive) — check there before guessing a flag name; most non-obvious flags (`--eval-mode`, `--grpo-mode`, `--mask-loss`, `--ngram-order`, etc.) have detailed doc comments explaining semantics and interactions.

Model architecture flags (`--block-size`, `--hidden-size`, `--num-heads`, `--num-blocks`) must match between the training run that produced a `.bin` file and any later `--generate`/`--eval`/`--rft`/`--grpo` run that loads it.

## Architecture

**Pipeline**: `main.rs` → `args::parse_args()` → `train::do_training(args)`, which dispatches to one of several mutually-exclusive modes (clap `ArgGroup("mode")`): `--train`, `--generate`, `--eval`, `--rft`, `--grpo`, `--serve`.

**Models** (`src/model/`): `LanguageModel` is an enum (not a trait) over `BigramLM`, `Gpt`, `NgramLM` (`mod.rs`). `NgramLM` is a strict generalization of `BigramLM` (order 2 == bigram) that conditions on the previous `N-1` tokens via a composite-key embedding table. `Gpt` (`gpt.rs`) implements the transformer blocks; `--num-heads` can be a single value (broadcast) or a comma-separated per-block list, resolved by `resolve_heads_schedule`.

**Tokenizers** (`tokenizer.rs`): `SimpleTokenizer` (char-level) or `BpeTokenizer` (byte-level BPE trained on the corpus, vocab size configurable via `--vocab-size`, floor 256). Selected via `-k char|bpe`.

**Dataset** (`dataset.rs`): `Dataset` holds tokenized train/validation tensors plus an optional per-position loss mask (`train_mask`) used only by the experimental `--mask-loss` feature for arithmetic corpora — it's aligned to char tokenization only, not BPE.

**Post-training loops**:
- `rft.rs` — Rejection-sampling Fine-Tuning: sample completions, keep exact-correct ones, SFT on the winners, repeat. Reward is a filter only, never enters the gradient. Saves to a `-rft` variant path, never overwrites the base checkpoint.
- `grpo.rs` — Group Relative Policy Optimization. `GrpoMode::Lite` is REINFORCE + group-relative advantage (no clipping/KL); `GrpoMode::Full` adds PPO-style importance-ratio clipping, KL-to-reference penalty, and K mini-epochs per group. Saves to a `-grpo`/`-grpo-full` variant path.

**Eval** (`eval.rs`): greedy-decoding correctness harness for arithmetic problems. `EvalMode::Smart` (default) derives eval operand range from the training corpus at registration time; `EvalMode::Legacy` uses the pre-existing fixed-range/no-filter behavior — exists specifically to bisect regressions against `Smart`.

**Registry** (`registry.rs`): SQLite-backed (`smolgpt.db`, bundled rusqlite — pinned to `libsqlite3-sys 0.37` because 0.38+ needs unstable `cfg_select!`). Every `--train` run auto-registers its model; `--serve` reads from here. On first `--serve` run, if the DB is empty and `models.toml` exists, it's imported as a seed (legacy format, now superseded by the DB).

**Serve** (`serve.rs`): axum web UI + JSON API for browsing models/datasets and running evals/REPL generation from the browser. Heavy work (eval, tokenizer build, model load, inference) runs via `spawn_blocking`; concurrent identical eval requests get HTTP 409 rather than double-computing. Route list is documented in the file's module doc comment.

## Repo conventions

- Trained model checkpoints (`*.bin`), `smolgpt.db*`, `*.eval.json`, `*.bak`, and generated arithmetic corpora (`data/arithmetic*.txt`) are gitignored — they're regenerable from training/eval runs and not meant to be committed. Don't treat stray `.bin`/`.db`/`.bak` files in the repo root as meaningful state to preserve.
- Non-obvious design decisions (why a flag exists, why a default was chosen, why one code path was preserved alongside another) are explained in doc comments at the point of definition — read those before changing behavior, since many encode a deliberate trade-off or a past regression.
- Tests are inline `#[cfg(test)]` modules in the same file as the code they cover (see `dataset.rs`, `eval.rs`, `model/gpt.rs`, `model/mod.rs`, `model/ngram.rs`, `registry.rs`, `tokenizer.rs`), using `rstest` for parameterized cases.
