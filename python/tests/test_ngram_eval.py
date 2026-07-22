import torch

from smolgpt.eval import Problem, gen_problems, make_corpus
from smolgpt.model import NgramLM, greedy_generate


def test_ngram_order2_equals_bigram_shape():
    model = NgramLM(vocab_size=10, order=2)
    assert model.block_size == 1
    x = torch.zeros((2, 4), dtype=torch.long)
    assert model(x).shape == (2, 4, 10)


def test_ngram_order3():
    model = NgramLM(vocab_size=8, order=3)
    assert model.block_size == 2
    x = torch.randint(0, 8, (3, 6))
    assert model(x).shape == (3, 6, 8)


def test_ngram_table_guard():
    import pytest

    with pytest.raises(ValueError):
        NgramLM(vocab_size=1024, order=5)  # 1024**4 rows -> too large


def test_greedy_generate_length():
    model = NgramLM(vocab_size=10, order=2)
    out = greedy_generate(model, prompt_ids=[1, 2], max_new_tokens=5, device="cpu")
    assert len(out) <= 5


def test_problem_arithmetic():
    p = Problem(12, 7, "+")
    assert p.result == 19
    assert p.prompt == "12+7="
    assert p.answer == "19"
    assert p.line() == "12+7=19\n"


def test_gen_problems_deterministic():
    a = gen_problems(50, 0, 99, "+", seed=1)
    b = gen_problems(50, 0, 99, "+", seed=1)
    assert [(p.a, p.b, p.op) for p in a] == [(p.a, p.b, p.op) for p in b]


def test_make_corpus():
    probs = [Problem(1, 2, "+"), Problem(3, 4, "+")]
    assert make_corpus(probs) == "1+2=3\n3+4=7\n"
