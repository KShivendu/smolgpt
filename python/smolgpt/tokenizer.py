"""Tokenizers: character-level and byte-level BPE.

Ports the Rust `src/tokenizer.rs`. Both tokenizers expose the same interface:
``encode(str) -> list[int]``, ``decode(list[int]) -> str``, and ``vocab_size``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...

    @property
    def vocab_size(self) -> int: ...


class CharTokenizer:
    """One token per character (vocab = distinct chars in the corpus, sorted)."""

    def __init__(self, corpus: str) -> None:
        self.charset: list[str] = sorted(set(corpus))
        self._index: dict[str, int] = {c: i for i, c in enumerate(self.charset)}

    def encode(self, text: str) -> list[int]:
        out = []
        for c in text:
            idx = self._index.get(c)
            if idx is None:
                # Mirror the Rust behavior: fall back to token 0 (a real,
                # arbitrary char) but surface it loudly rather than silently
                # corrupting the input.
                print(
                    f"warning: CharTokenizer.encode: char {c!r} not in trained "
                    f"charset, substituting token 0 ({self.charset[0]!r})"
                )
                idx = 0
            out.append(idx)
        return out

    def decode(self, tokens: list[int]) -> str:
        return "".join(
            self.charset[t] if 0 <= t < len(self.charset) else " " for t in tokens
        )

    @property
    def vocab_size(self) -> int:
        return len(self.charset)


def _pretokenize(text: str) -> Iterator[str]:
    """Split text into maximal runs of whitespace vs. non-whitespace.

    Merges are confined within a chunk so the tokenizer never crosses word
    boundaries (e.g. "Hello, world" -> ["Hello,", " ", "world"]).
    """
    start = 0
    prev_ws: bool | None = None
    for i, ch in enumerate(text):
        ws = ch.isspace()
        if prev_ws is not None and prev_ws != ws:
            yield text[start:i]
            start = i
        prev_ws = ws
    if start < len(text):
        yield text[start:]


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of ``pair`` in ``ids`` with ``new_id``."""
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i + 1 < n and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BpeTokenizer:
    """Byte-level Byte-Pair Encoding, GPT-2 family.

    Training starts from the 256 raw bytes and greedily merges the most
    frequent adjacent pair into a new token until ``target_vocab_size`` is
    reached (or no pair repeats). The base vocab covers every byte, so any
    UTF-8 input round-trips with no ``<unk>``.
    """

    def __init__(self, ranks: dict[tuple[int, int], int], vocab: list[bytes]) -> None:
        self.ranks = ranks
        self.vocab = vocab

    @classmethod
    def train(cls, corpus: str, target_vocab_size: int) -> "BpeTokenizer":
        vocab: list[bytes] = [bytes([b]) for b in range(256)]
        ranks: dict[tuple[int, int], int] = {}

        # {word -> frequency} over pre-tokenized chunks; work on unique words so
        # training scales with the corpus's vocabulary, not its length.
        word_freqs: dict[tuple[int, ...], int] = defaultdict(int)
        for chunk in _pretokenize(corpus):
            word_freqs[tuple(chunk.encode("utf-8"))] += 1
        words: list[list[int]] = [list(w) for w in word_freqs]
        freqs: list[int] = list(word_freqs.values())

        num_merges = max(0, target_vocab_size - 256)
        for i in range(num_merges):
            counts: dict[tuple[int, int], int] = defaultdict(int)
            for word, freq in zip(words, freqs):
                for a, b in zip(word, word[1:]):
                    counts[(a, b)] += freq
            if not counts:
                break
            # Most frequent pair; tie-break by pair value for determinism
            # (matches the Rust `.then_with(|| b.0.cmp(a.0))`: on a tie, prefer
            # the smaller pair).
            pair = max(counts, key=lambda p: (counts[p], (-p[0], -p[1])))
            if counts[pair] < 2:
                break
            new_id = 256 + i
            ranks[pair] = new_id
            vocab.append(vocab[pair[0]] + vocab[pair[1]])
            words = [_merge_pair(w, pair, new_id) for w in words]

        return cls(ranks, vocab)

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while True:
            best_pair: tuple[int, int] | None = None
            best_rank = None
            for a, b in zip(ids, ids[1:]):
                rank = self.ranks.get((a, b))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = (a, b)
            if best_pair is None:
                break
            ids = _merge_pair(ids, best_pair, best_rank)  # best_rank == new id
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in _pretokenize(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, tokens: list[int]) -> str:
        data = b"".join(
            self.vocab[t] if 0 <= t < len(self.vocab) else b"" for t in tokens
        )
        return data.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
