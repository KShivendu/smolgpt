"""Corpus loading and batching. Ports the Rust `src/dataset.rs`."""

from __future__ import annotations

from pathlib import Path

import torch


def load_corpus(path: str | Path, show_sample: bool = False) -> str:
    text = Path(path).read_text(encoding="utf-8")
    print(f"Length of the dataset: {len(text)}")
    if show_sample:
        print("First 1000 characters of the dataset:")
        print(text[:1000])
    return text


class Dataset:
    """Holds tokenized train/validation tensors and yields random batches."""

    def __init__(self, data: torch.Tensor, train_ratio: float = 0.9) -> None:
        n = data.shape[0]
        split = int(n * train_ratio)
        self.train_data = data[:split]
        self.val_data = data[split:]
        self.train_size = split
        self.val_size = n - split

    def get_random_batches(
        self,
        block_size: int,
        num_batches: int,
        split: str = "train",
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) each of shape (num_batches, block_size).

        y is x shifted by one position (next-token targets).
        """
        data = self.train_data if split == "train" else self.val_data
        high = data.shape[0] - block_size
        if high <= 0:
            raise ValueError("block_size exceeds dataset size")
        ix = torch.randint(high, (num_batches,), generator=generator)
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)
