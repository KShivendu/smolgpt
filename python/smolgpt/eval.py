"""Greedy-decoding correctness harness for arithmetic. Ports `src/eval.rs`.

Problems look like ``"12+7="`` (prompt) with expected completion ``"19\n"``.
The model greedily decodes from the prompt; we compare the decoded answer
(up to the newline) against the true result.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .model import greedy_generate

OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}


@dataclass
class Problem:
    a: int
    b: int
    op: str

    @property
    def result(self) -> int:
        return OPS[self.op](self.a, self.b)

    @property
    def prompt(self) -> str:
        return f"{self.a}{self.op}{self.b}="

    @property
    def answer(self) -> str:
        return str(self.result)

    def line(self) -> str:
        return f"{self.prompt}{self.answer}\n"


def gen_problems(
    n: int, lo: int, hi: int, ops: str = "+", seed: int | None = None
) -> list[Problem]:
    rng = random.Random(seed)
    op_list = list(ops)
    return [
        Problem(rng.randint(lo, hi), rng.randint(lo, hi), rng.choice(op_list))
        for _ in range(n)
    ]


def make_corpus(problems: list[Problem]) -> str:
    return "".join(p.line() for p in problems)


@dataclass
class EvalResult:
    total: int
    correct: int
    samples: list[tuple[str, str, str]]  # (prompt, expected, got)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def evaluate(
    model,
    tokenizer,
    problems: list[Problem],
    device: str = "cpu",
    max_answer_len: int = 8,
    keep_samples: int = 10,
) -> EvalResult:
    """Greedy-decode each problem's answer and score exact-match correctness."""
    newline_ids = tokenizer.encode("\n")
    stop_id = newline_ids[0] if newline_ids else None
    correct = 0
    samples: list[tuple[str, str, str]] = []
    for p in problems:
        out_ids = greedy_generate(
            model, tokenizer.encode(p.prompt), max_answer_len, device, stop_id
        )
        got = tokenizer.decode(out_ids).split("\n", 1)[0].strip()
        if got == p.answer:
            correct += 1
        if len(samples) < keep_samples:
            samples.append((p.prompt, p.answer, got))
    return EvalResult(total=len(problems), correct=correct, samples=samples)
