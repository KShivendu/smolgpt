"""Training and generation loops."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .dataset import Dataset
from .model import BigramLM, Gpt, GptConfig, NgramLM, generate


def train_model(
    model: torch.nn.Module,
    dataset: Dataset,
    model_path: str | Path,
    epochs: int,
    num_batches: int,
    block_size: int,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    start = time.time()
    for epoch in range(1, epochs + 1):
        x, y = dataset.get_random_batches(
            block_size, num_batches, "train", device, generator
        )
        logits = model(x)
        b, t, c = logits.shape
        loss = F.cross_entropy(logits.view(b * t, c), y.reshape(b * t))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}/{epochs}: Loss = {loss.item()}")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print(f"Training completed in {time.time() - start:.2f}s")


def build_model(
    model_type: str,
    vocab_size: int,
    config: GptConfig,
    device: torch.device | str,
    ngram_order: int = 3,
) -> torch.nn.Module:
    if model_type == "gpt":
        config.vocab_size = vocab_size
        model = Gpt(config)
    elif model_type == "bigram":
        model = BigramLM(vocab_size)
    elif model_type == "ngram":
        model = NgramLM(vocab_size, ngram_order)
    else:
        raise ValueError(f"unknown model type: {model_type}")
    return model.to(device)


def generate_text(model, tokenizer, max_new_tokens, device, generator=None) -> str:
    ids = generate(model, max_new_tokens, device, generator)
    return tokenizer.decode(ids)
