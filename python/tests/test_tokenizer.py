from smolgpt.tokenizer import BpeTokenizer, CharTokenizer


def test_char_roundtrip():
    tok = CharTokenizer("Hello, world!")
    assert tok.decode(tok.encode("Hee")) == "Hee"
    assert tok.vocab_size == len(sorted(set("Hello, world!")))


def test_char_empty():
    assert CharTokenizer("").encode("") == []
    assert CharTokenizer("abc").decode([]) == ""


def test_bpe_roundtrip():
    tok = BpeTokenizer.train("the cat sat on the mat. the cat ran.", 300)
    for q in ["the cat", "hello world!", "", "the the the"]:
        assert tok.decode(tok.encode(q)) == q


def test_bpe_roundtrip_unseen_bytes():
    tok = BpeTokenizer.train("aaaa bbbb aaaa", 270)
    q = "totally unseen — café 🚀"
    assert tok.decode(tok.encode(q)) == q


def test_bpe_compresses():
    corpus = "abcabcabcabcabcabc"
    tok = BpeTokenizer.train(corpus, 300)
    assert len(tok.encode(corpus)) < len(corpus)


def test_bpe_vocab_grows():
    tok = BpeTokenizer.train("the cat sat on the mat", 300)
    assert 256 < tok.vocab_size <= 300
