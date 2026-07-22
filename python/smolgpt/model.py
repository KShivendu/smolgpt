"""Models: BigramLM and GPT. Ports the Rust `src/model/`.

The GPT mirrors the Rust architecture: token + position embeddings, a stack of
pre-norm transformer blocks (multi-head causal self-attention + a 4x ReLU MLP),
then a linear head back to vocab size. There is no final layer-norm before the
head, matching the Rust implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GptConfig:
    block_size: int = 32
    vocab_size: int = 65
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, c = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * c**-0.5  # (B, T, T)
        weights = weights.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        return weights @ v  # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-norm transformer block with residual connections."""

    def __init__(self, n_head: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Gpt(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(config.n_head, config.n_embd, config.block_size, config.dropout)
                for _ in range(config.n_layer)
            ]
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        tok = self.token_embeddings(idx)  # (B, T, n_embd)
        pos = self.position_embeddings(torch.arange(t, device=idx.device))  # (T, n_embd)
        x = tok + pos
        x = self.blocks(x)
        return self.lm_head(x)  # (B, T, vocab_size)

    @property
    def block_size(self) -> int:
        return self.config.block_size


class BigramLM(nn.Module):
    """Each token directly predicts the next via a lookup table."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)
        # Bigram has no real context window; 1 is enough for generation.
        self._block_size = 1

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(idx)  # (B, T, vocab_size)

    @property
    def block_size(self) -> int:
        return self._block_size


class NgramLM(nn.Module):
    """Strict generalization of the bigram: conditions on the previous ``order-1``
    tokens via a composite-key embedding table (order 2 == bigram).

    The composite key packs the ``order-1`` context tokens into a single index in
    ``[0, vocab_size ** (order-1))``, so the table has that many rows. This grows
    exponentially with order — fine for the small char vocab it was designed for,
    but a footgun for large BPE vocabularies (guarded below).
    """

    MAX_TABLE_ROWS = 50_000_000

    def __init__(self, vocab_size: int, order: int):
        super().__init__()
        if order < 2:
            raise ValueError("order must be >= 2")
        self.vocab_size = vocab_size
        self.order = order
        self.ctx = order - 1
        rows = vocab_size**self.ctx
        if rows > self.MAX_TABLE_ROWS:
            raise ValueError(
                f"n-gram table would need {rows} rows (vocab={vocab_size}, "
                f"order={order}); too large. Use a smaller order or char vocab."
            )
        self.table = nn.Embedding(rows, vocab_size)
        self._block_size = self.ctx

    def _composite_keys(self, idx: torch.Tensor) -> torch.Tensor:
        # For output position t, the context is the ctx tokens immediately
        # before t; left-pad with 0 so early positions are still defined.
        b, t = idx.shape
        padded = F.pad(idx, (self.ctx, 0))  # (B, T + ctx)
        keys = torch.zeros_like(idx)
        for j in range(self.ctx):
            keys = keys * self.vocab_size + padded[:, j : j + t]
        return keys

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.table(self._composite_keys(idx))  # (B, T, vocab_size)

    @property
    def block_size(self) -> int:
        return self._block_size


@torch.no_grad()
def greedy_generate(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: torch.device | str = "cpu",
    stop_id: int | None = None,
) -> list[int]:
    """Greedy (argmax) decoding from a prompt — used by the eval harness."""
    model.eval()
    block_size = model.block_size
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])
        next_id = int(logits[0, -1, :].argmax())
        out.append(next_id)
        if stop_id is not None and next_id == stop_id:
            break
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
    return out


@torch.no_grad()
def generate(
    model: nn.Module,
    max_new_tokens: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> list[int]:
    """Autoregressively sample ``max_new_tokens`` token ids, seeded with 0."""
    model.eval()
    block_size = model.block_size
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)  # start with <BOS>=0
    for _ in range(max_new_tokens - 1):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]  # (1, vocab_size)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()
