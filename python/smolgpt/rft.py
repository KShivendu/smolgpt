"""Rejection-sampling Fine-Tuning (RFT). Ports `src/rft.rs`.

Each round: sample completions for a batch of arithmetic prompts, KEEP only the
exact-correct ones, and SFT the model on those winners. The reward is a filter
only — it never enters the gradient. Saves to a ``-rft`` checkpoint, never
overwriting the base.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from .eval import Problem
from .tokenizer import Tokenizer


@torch.no_grad()
def sample_completion(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    stop_id: int | None,
    temperature: float,
    device: str,
    generator: torch.Generator | None = None,
) -> list[int]:
    model.eval()
    block_size = model.block_size
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])[0, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1, generator=generator))
        out.append(next_id)
        if stop_id is not None and next_id == stop_id:
            break
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
    return out


def _sft_step(
    model,
    optimizer,
    sequences: list[list[int]],
    block_size: int,
    device: str,
) -> float:
    """One supervised next-token step over padded winner sequences."""
    model.train()
    # Right-pad to a common length; ignore padded targets in the loss.
    maxlen = min(block_size, max(len(s) for s in sequences))
    xs, ys = [], []
    for s in sequences:
        s = s[: maxlen + 1]
        x = s[:-1] if len(s) > 1 else s
        y = s[1:] if len(s) > 1 else s
        pad = maxlen - len(x)
        xs.append(x + [0] * pad)
        ys.append(y + [-100] * pad)  # -100 = ignore_index
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    logits = model(x)
    b, t, c = logits.shape
    loss = F.cross_entropy(logits.view(b * t, c), y.reshape(b * t), ignore_index=-100)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


def rft_train(
    model,
    tokenizer: Tokenizer,
    problems: list[Problem],
    model_path: str | Path,
    rounds: int = 20,
    samples_per_prompt: int = 4,
    temperature: float = 1.0,
    lr: float = 1e-3,
    max_answer_len: int = 8,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> None:
    stop_ids = tokenizer.encode("\n")
    stop_id = stop_ids[0] if stop_ids else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    block_size = model.block_size

    for r in range(1, rounds + 1):
        winners: list[list[int]] = []
        attempts = 0
        for p in problems:
            prompt_ids = tokenizer.encode(p.prompt)
            for _ in range(samples_per_prompt):
                attempts += 1
                comp = sample_completion(
                    model, prompt_ids, max_answer_len, stop_id,
                    temperature, device, generator,
                )
                got = tokenizer.decode(comp).split("\n", 1)[0].strip()
                if got == p.answer:  # reward = filter only
                    winners.append(tokenizer.encode(p.line()))
        acc = len(winners) / attempts if attempts else 0.0
        if winners:
            loss = _sft_step(model, optimizer, winners, block_size, device)
            print(
                f"RFT round {r}/{rounds}: kept {len(winners)}/{attempts} "
                f"({acc:.1%}), SFT loss = {loss:.4f}"
            )
        else:
            print(f"RFT round {r}/{rounds}: kept 0/{attempts} — no winners to train on")
        torch.save(model.state_dict(), model_path)
    print(f"RFT model saved to {model_path}")
