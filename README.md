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

From BigramLM:

```console
$ smolgpt -e 3500 --train --generate

Epoch 3200/3200: Loss = 2.5841358
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

## Useful links

- https://www.youtube.com/watch?v=kCc8FmEb1nY
- https://github.com/jeroenvlek/gpt-from-scratch-rs
- https://github.com/keyvank/femtoGPT
- https://github.com/Murattut/RustGpt
- https://github.com/nerdai/llms-from-scratch-rs
