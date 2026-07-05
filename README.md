# SmolGPT

```sh
git clone https://github.com/kshivendu/smolgpt
cd smolgpt
cargo run -r -- --train --generate

# You can install the binary for ease of use (saved in ~/.cargo/bin)
cargo install --path . --profile release
smolgpt --help
smolgpt --epochs 1000 --train --generate
```

## Output:

BigramLM saturates around 2.5 cross-entropy loss for [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

```console
$ smolgpt -e 4000 --train --generate

...
Epoch 4000/4000: Loss = 2.5262532
Training completed in 23.88s
Generated text:
LI mar th n--O:' ISTh oues,
I sealyo be.ME:nd bl3BENOistreanspontharatheQ;
IUSTheathansthad:ndang,IOueDWMr st anYCAy hen wharieZ.
ulluliewLicit 'lind Mat
Be at mMrofse wseresand gird ban
ARGUju en IO,
Prd he e nadstrebye,
Asttkee theso n u ' 3.ve? ase o ICO:


DVINGqullstind at pqures see sas3qk ou,ve wYe Ge hrond Wie, Prdofoun IN re tand,-nar hatJOMXENunn,h:
Tho tor tok le s spo g, t! hd sttsheand tte s &W$jetollowouraBE is
ICHen nfur herothyo wTringju, besurtly bavermy
Vad iYVI f my w,
ThDUEN
```

## Tokenizers: characters vs BPE

By default the model tokenizes one token per character. You can instead train a
byte-level BPE tokenizer (GPT-2 family) on the corpus:

```sh
smolgpt -m bigram -k char --train --generate                    # character-level
smolgpt -m bigram -k bpe --vocab-size 1024 --train --generate   # BPE (subword)
```

Comparison on Tiny Shakespeare (bigram model, 200 epochs):

| | char | BPE-1024 |
|---|---|---|
| vocab size | 65 | 1024 |
| corpus length | 1,115,394 tokens | 589,653 tokens |
| bytes / token | 1.00 | 1.89 |
| final loss | 4.350 | 7.085 |
| **loss / byte (normalized)** | **6.28 bits/byte** | **5.40 bits/byte** |

Takeaways:

- **Sequences are ~half as long** (1.89 bytes/token), so a fixed context window
  sees ~2× more text — the main reason real LLMs use subword tokens.
- Raw cross-entropy is higher for BPE but not comparable directly: it's a
  1024-way vs 65-way softmax (random baselines 6.93 vs 4.17 nats). Normalized to
  **bits-per-byte**, BPE already edges out char even on this trivial bigram.
- Qualitatively the jump is clearest. Char generation is letter-soup, while BPE
  emits whole words and Shakespeare speaker tags even with no attention:

  ```console
  # char
  NuxZNQjH3wVh!IW3FoSqnzFtaqEkWMwvBaFirvqvBJceHT!vVO&c'jK?&TbW...

  # BPE-1024
  ...theyhead arm... ROMEO: true ...think... LADY ...crown... PETRUCHIO:
  ...daughter... mother And... JULIET: ...noble... KING ...Warwick... GLOUCESTER:
  ```

Check out [roadmap](./roadmap.md)

## Useful links

- https://www.youtube.com/watch?v=kCc8FmEb1nY
- https://github.com/jeroenvlek/gpt-from-scratch-rs
- https://github.com/keyvank/femtoGPT
- https://github.com/Murattut/RustGpt
- https://github.com/nerdai/llms-from-scratch-rs
