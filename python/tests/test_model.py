import torch

from smolgpt.dataset import Dataset
from smolgpt.model import BigramLM, Gpt, GptConfig, generate


def test_gpt_forward_shape():
    cfg = GptConfig(block_size=8, vocab_size=40, n_embd=32, n_head=4, n_layer=2)
    model = Gpt(cfg)
    x = torch.zeros((2, 8), dtype=torch.long)
    logits = model(x)
    assert logits.shape == (2, 8, 40)


def test_bigram_forward_shape():
    model = BigramLM(vocab_size=50)
    x = torch.zeros((3, 5), dtype=torch.long)
    assert model(x).shape == (3, 5, 50)


def test_generate_length():
    model = BigramLM(vocab_size=10)
    ids = generate(model, max_new_tokens=20, device="cpu")
    assert len(ids) == 20
    assert all(0 <= i < 10 for i in ids)


def test_dataset_batches():
    data = torch.arange(100)
    ds = Dataset(data, train_ratio=0.9)
    assert ds.train_size == 90 and ds.val_size == 10
    x, y = ds.get_random_batches(block_size=8, num_batches=4, split="train")
    assert x.shape == (4, 8) and y.shape == (4, 8)
    # y is x shifted by one
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_gpt_trains_one_step():
    cfg = GptConfig(block_size=8, vocab_size=20, n_embd=32, n_head=4, n_layer=2)
    model = Gpt(cfg)
    data = torch.randint(0, 20, (500,))
    ds = Dataset(data)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = ds.get_random_batches(8, 16, "train")
    logits = model(x)
    b, t, c = logits.shape
    loss = torch.nn.functional.cross_entropy(logits.view(b * t, c), y.reshape(b * t))
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
