# SmolGPT (Python / PyTorch)

A PyTorch port of the Rust [smolgpt](../). Same idea — a from-scratch
GPT/bigram language model with char and byte-level BPE tokenizers, trained on
Tiny Shakespeare.

## Setup

```sh
cd python
uv sync
```

## Usage

```sh
# Train a GPT with BPE (the default), then sample from it
uv run smolgpt --train --generate --epochs 3000

# Character-level bigram
uv run smolgpt -m bigram -k char --train --generate

# Generate from a previously trained checkpoint
uv run smolgpt --generate            # errors if no checkpoint exists yet
```

Checkpoints are saved per model+tokenizer (e.g. `gpt-bpe.pt`, `bigram-char.pt`).
GPT architecture flags (`--block-size`, `--n-embd`, `--n-head`, `--n-layer`)
must match between the training run and any later `--generate` that loads it.

## Layout

| File | Role |
|---|---|
| `smolgpt/tokenizer.py` | `CharTokenizer`, `BpeTokenizer` (byte-level BPE) |
| `smolgpt/dataset.py` | corpus load + random batching |
| `smolgpt/model.py` | `BigramLM`, `Gpt` (transformer), `generate` |
| `smolgpt/train.py` | training loop |
| `smolgpt/cli.py` | argparse CLI (mirrors the Rust flags) |

## Tests

```sh
uv run pytest
```

## Scope

This is the **core** port (models, tokenizers, dataset, train/generate). The
eval harness, RFT/GRPO post-training, registry, and web UI from the Rust version
are not ported yet.
