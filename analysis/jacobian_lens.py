"""Jacobian-lens interpretability analysis for any `Gpt`-type smolgpt model.

Reproduces Anthropic's Jacobian lens (Kramar/Templeton et al., "Verbalizable
Representations Form a Global Workspace in Language Models", transformer-circuits.pub
2026; reference implementation: github.com/anthropics/jacobian-lens) against this
project's char-level addition GPT, trained via candle-core.

What the lens computes (ground truth, from the reference repo's
jlens/fitting.py and jlens/lens.py docstrings):

    lens_l(h) = unembed( J_l @ h ),   J_l = E[ dh_final/dh_l ]

`h_l` is the residual stream at (transformer-block-output) layer `l`; `h_final`
is the residual at the last block (== input to the LM head, since this
architecture -- like the reference's target models -- has no separate readout
step beyond the head). `J_l` is fit once per model by averaging the local
input-output Jacobian over many (prompt, source-position) pairs, using
one-hot cotangents at the target layer summed over every "current-and-future"
target position and averaged over source positions (fitting.jacobian_for_prompt).
`unembed` is the model's own embedding-matrix-derived readout.

This script:
  1. Reimplements the exact candle forward pass (embeddings -> N pre-norm
     residual transformer blocks -> lm_head, no final LayerNorm) in torch,
     loading the trained weights directly from the candle-native safetensors
     `.bin` file. Supports a per-block attention-head SCHEDULE (e.g.
     `1,1,4,4`), not just a uniform head count, mirroring
     `src/model/gpt.rs`'s `heads_schedule`/`validate_heads_schedule`.
  2. Fits J_l at FINE (intra-block) granularity: not just block boundaries,
     but also each block's post-attention/pre-MLP checkpoint (`attn_output`
     in `src/model/gpt.rs`'s `forward_with_training` -- the residual value
     right after the attention sub-block's own residual add, before that
     block's MLP runs). For an N-block model this is `2*N+1` checkpoints
     (embeddings, then an attn/full-output pair per block) instead of just
     `N+1` block boundaries -- every checkpoint except h_final itself is
     fit against the last block's output, averaged over the corpus's facts,
     faithfully reproducing the reference estimator at this finer grain.
     This directly answers, per block, "does the ATTENTION step already do
     the work, or does the MLP do it?" instead of only being able to say
     that the answer changed somewhere across a whole block.
  3. Applies the fitted lens at the "=" position of every fact, and reports
     when (which layer) the correct answer digit becomes the lens's top
     prediction.
  4. Extension beyond the vanilla lens (clearly labeled as such): a
     per-example, per-source-POSITION exact local Jacobian (not corpus
     averaged) of the answer-token logit w.r.t. every position's residual at
     every layer, giving a position x layer sensitivity heatmap per fact --
     answering "does this model's answer depend on both operands, or just the
     '=' sign", and letting us compare +0/+1 facts against arbitrary ones.
  5. A second, independent extension (also clearly labeled): a layer-by-layer
     embedding/activation visualization. The token-embedding matrix
     (`vocab_size` x `hidden_size`) and, for every layer boundary (embeddings,
     post-block0, ..., post-block(N-1)), every token position's residual
     vector across the whole fact corpus are each projected to 2D via PCA
     (always) and UMAP (if the `umap-learn` package is importable; `null`
     otherwise, with `have_umap: false` in the JSON so callers can render a
     "PCA-only" note instead of a broken toggle). This reuses the exact same
     forward pass already computed for the Jacobian fitting above -- no
     extra model invocations, just also keeping the per-layer residuals
     around. Each layer's projection is fit independently and then
     Procrustes-aligned into one consistent running frame across the layer
     sequence (see `procrustes_align`/`align_layer_sequence`) so a UI
     animating point positions across layers shows real movement of the
     representation, not artifacts of PCA's arbitrary axis sign/ordering or
     UMAP's arbitrary orientation between independent fits.

Fully parameterized -- no hardcoded model/corpus/architecture. Run as:

    python3 analysis/jacobian_lens.py \
        --model-path mask-test-4blocks-2heads.bin \
        --dataset-path data/arithmetic-add-1digit.txt \
        --block-size 16 --hidden-size 16 --num-heads 2 --num-blocks 4 \
        --vocab-size 13 --output-dir analysis/output/mask-test-4blocks-2heads

`--num-heads` accepts either a single integer (applied uniformly to every
block, e.g. `2`) or a comma-separated per-block schedule with exactly
`--num-blocks` entries (e.g. `1,1,4,4`), exactly like the Rust CLI's
`--num-heads` flag.

Needs torch, numpy, matplotlib, safetensors -- see requirements printed on
import error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
except ImportError as e:  # pragma: no cover - environment guard
    sys.stderr.write(
        f"jacobian_lens.py: missing dependency ({e}). Install with:\n"
        "  pip install torch numpy matplotlib safetensors\n"
    )
    sys.exit(2)

# UMAP is optional: the embedding-visualization extension (see module doc,
# point 5) degrades gracefully to PCA-only when `umap-learn` isn't installed,
# rather than failing the whole script over one extra, heavier dependency.
try:
    import umap  # noqa: F401
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False

# t-SNE (sklearn) is likewise optional, following the exact same
# degrade-gracefully pattern as UMAP above: if scikit-learn isn't installed,
# the embedding-viz JSON just omits `tsne` (`have_tsne: false`) instead of
# failing the whole script over one more heavy, non-essential dependency.
try:
    from sklearn.manifold import TSNE  # noqa: F401
    HAVE_TSNE = True
except ImportError:
    HAVE_TSNE = False

torch.manual_seed(0)

FACT_RE = re.compile(r"^(-?\d+)([+\-])(-?\d+)=(-?\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="Path to the candle-saved .bin (safetensors) checkpoint")
    p.add_argument("--dataset-path", required=True, help="Path to the training corpus (a op b=c per line)")
    p.add_argument("--block-size", type=int, required=True)
    p.add_argument("--hidden-size", type=int, required=True)
    p.add_argument(
        "--num-heads",
        required=True,
        help="Single int (uniform) or comma-separated per-block schedule, e.g. '1,1,4,4'",
    )
    p.add_argument("--num-blocks", type=int, required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--output-dir", required=True, help="Directory to write results.json + PNG plots into")
    return p.parse_args()


def resolve_heads_schedule(num_heads_arg: str, num_blocks: int, embed_dims: int) -> list[int]:
    """Mirror `model::resolve_heads_schedule` / `validate_heads_schedule` in
    src/model/gpt.rs: a bare integer is broadcast to every block; a
    comma-separated list must have exactly `num_blocks` entries. Each entry
    must individually divide `embed_dims`."""
    parts = [s.strip() for s in num_heads_arg.split(",") if s.strip() != ""]
    values = [int(s) for s in parts]
    if len(values) == 1:
        schedule = values * num_blocks
    elif len(values) == num_blocks:
        schedule = values
    else:
        raise ValueError(
            f"--num-heads has {len(values)} entries but --num-blocks is {num_blocks}; "
            f"pass a single number (applied to every block) or exactly {num_blocks} entries"
        )
    for i, h in enumerate(schedule):
        if h <= 0 or embed_dims % h != 0:
            raise ValueError(f"hidden_size ({embed_dims}) must be divisible by num_heads ({h}) at block {i}")
    return schedule


# --------------------------------------------------------------------------
# Tokenizer: SimpleTokenizer::new sorts the unique chars of the corpus.
# --------------------------------------------------------------------------
def build_charset(corpus: str) -> list[str]:
    return sorted(set(corpus))


def encode(charset: list[str], text: str) -> list[int]:
    return [charset.index(c) for c in text]


# --------------------------------------------------------------------------
# Faithful reimplementation of src/model/gpt.rs's forward_with_training
# (inference mode: dropout is a no-op either way). Supports a per-block head
# count (`heads_schedule[i]` heads in block i), not just a uniform count.
# --------------------------------------------------------------------------
class Head:
    def __init__(self, w: dict, prefix: str, head_size: int):
        # candle `linear_no_bias`: weight shape (out, in); y = x @ W^T
        self.Wk = w[f"{prefix}.key.weight"]
        self.Wq = w[f"{prefix}.query.weight"]
        self.Wv = w[f"{prefix}.value.weight"]
        self.head_size = head_size

    def __call__(self, x: torch.Tensor, return_att: bool = False):
        # x: (T, C)
        T = x.shape[0]
        k = x @ self.Wk.T  # (T, head_size)
        q = x @ self.Wq.T
        v = x @ self.Wv.T
        att = (q @ k.T) * (self.head_size ** -0.5)  # (T, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v  # (T, head_size)
        return (out, att) if return_att else out


class Block:
    def __init__(self, w: dict, i: int, embed_dims: int, num_heads: int):
        p = f"blocks.block_{i}"
        head_size = embed_dims // num_heads
        self.heads = [Head(w, f"{p}.attn.head_{h}", head_size) for h in range(num_heads)]
        self.proj_w = w[f"{p}.attn.proj.weight"]
        self.proj_b = w[f"{p}.attn.proj.bias"]
        self.ln1_w = w[f"{p}.ln1.weight"]
        self.ln1_b = w[f"{p}.ln1.bias"]
        self.ln2_w = w[f"{p}.ln2.weight"]
        self.ln2_b = w[f"{p}.ln2.bias"]
        self.fc1_w = w[f"{p}.ffwd.fc1.weight"]
        self.fc1_b = w[f"{p}.ffwd.fc1.bias"]
        self.fc2_w = w[f"{p}.ffwd.fc2.weight"]
        self.fc2_b = w[f"{p}.ffwd.fc2.bias"]

    def ln(self, x, w, b, eps=1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + eps) * w + b

    def attn_part(self, x: torch.Tensor, return_att: bool = False):
        """`x + self_attn(ln1(x))` -- the intermediate residual value AFTER
        the attention sub-block's residual add but BEFORE the MLP sub-block,
        exactly matching `src/model/gpt.rs`'s `forward_with_training`:
        `attn_output = input.broadcast_add(&self_attn)?`. Exposed as its own
        method (rather than inlined in `__call__`) so it can be captured as
        an extra lens checkpoint between block boundaries -- see
        `TinyGpt.forward_all_layers`."""
        ln1 = self.ln(x, self.ln1_w, self.ln1_b)
        if return_att:
            head_outs, atts = zip(*[h(ln1, return_att=True) for h in self.heads])
            heads_out = torch.cat(head_outs, dim=-1)
        else:
            heads_out = torch.cat([h(ln1) for h in self.heads], dim=-1)  # (T, hidden)
        attn_out = heads_out @ self.proj_w.T + self.proj_b
        attn_output = x + attn_out
        if return_att:
            return attn_output, atts
        return attn_output

    def mlp_part(self, attn_output: torch.Tensor) -> torch.Tensor:
        """`attn_output + feedforward(ln2(attn_output))` -- completes the
        block given its own `attn_part` output (or an externally supplied
        stand-in, e.g. when resuming the forward pass from a mid-block lens
        checkpoint in `TinyGpt.run_from_layer`)."""
        ln2 = self.ln(attn_output, self.ln2_w, self.ln2_b)
        ff = F.relu(ln2 @ self.fc1_w.T + self.fc1_b) @ self.fc2_w.T + self.fc2_b
        return attn_output + ff

    def __call__(self, x: torch.Tensor, return_att: bool = False):
        if return_att:
            attn_output, atts = self.attn_part(x, return_att=True)
            out = self.mlp_part(attn_output)
            return out, atts  # atts: tuple of (T,T) per head
        attn_output = self.attn_part(x)
        return self.mlp_part(attn_output)


class TinyGpt:
    """Lens checkpoints, at FINE (intra-block) granularity: layer 0 is the
    combined embeddings; for each block `k` (0 <= k < num_blocks), layer
    `2*k+1` is that block's `attn_output` (post-attention-residual-add,
    pre-MLP -- see `Block.attn_part`) and layer `2*k+2` is that block's full
    output (post-MLP-residual-add, i.e. the old block-boundary checkpoint).
    Total checkpoints: `2*num_blocks + 1`. The last checkpoint (index
    `2*num_blocks`) is h_final, fed directly to lm_head (no final LayerNorm
    in this architecture) -- identical to the last block's full output, just
    named for clarity at the call sites that treat it as the lens target.

    This lets the lens ask a question the old block-boundary-only checkpoints
    couldn't: for a given block, does its OWN attention step already produce
    the answer, or does that block's MLP do the work? (`layer_label` below
    renders the attn checkpoints as `post-block{k}-attn` to make this
    explicit in every report/plot.)"""

    def __init__(self, weights_path: Path, hidden: int, heads_schedule: list[int], num_blocks: int, block_size: int):
        w = load_file(str(weights_path))
        self.w = {k: v.float() for k, v in w.items()}
        self.hidden = hidden
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.total_layers = 2 * num_blocks + 1
        self.tok_emb = self.w["token_embeddings"]  # (vocab, hidden)
        self.pos_emb = self.w["position_embeddings"]  # (block_size, hidden)
        self.blocks = [Block(self.w, i, hidden, heads_schedule[i]) for i in range(num_blocks)]
        self.lm_head_w = self.w["lm_head.weight"]  # (vocab, hidden)
        self.lm_head_b = self.w["lm_head.bias"]

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[0]
        return self.tok_emb[ids] + self.pos_emb[:T]

    def run_from_layer(self, h: torch.Tensor, start_layer: int) -> torch.Tensor:
        """Resume the forward pass from ANY of the `2*num_blocks+1` lens
        checkpoints (not just a block boundary) and return h_final.
        `start_layer == 0` means `h` is the embeddings; otherwise `h` is
        either a block's `attn_output` (odd `start_layer`) or a block's full
        output (even `start_layer`), per the indexing documented on the
        class."""
        if start_layer == 0:
            x = h
            for b in self.blocks:
                x = b(x)
            return x
        idx0 = start_layer - 1
        k = idx0 // 2
        is_attn_checkpoint = (idx0 % 2 == 0)
        x = self.blocks[k].mlp_part(h) if is_attn_checkpoint else h
        for b in self.blocks[k + 1:]:
            x = b(x)
        return x

    def forward_all_layers(self, ids: torch.Tensor) -> list[torch.Tensor]:
        """Returns all `2*num_blocks+1` checkpoint residuals (each (T,hidden)),
        in the order documented on the class."""
        layers = [self.embed(ids)]
        x = layers[0]
        for b in self.blocks:
            attn_output = b.attn_part(x)
            layers.append(attn_output)
            x = b.mlp_part(attn_output)
            layers.append(x)
        return layers

    def forward_with_attention(self, ids: torch.Tensor):
        """Returns (final_layers, attentions) where attentions[block_idx][head_idx]
        is a (T,T) attention-weight matrix. Not part of the Jacobian lens --
        plain attention-pattern inspection, used only as corroborating
        evidence for where in the network information moves."""
        x = self.embed(ids)
        attentions = []
        for b in self.blocks:
            x, atts = b(x, return_att=True)
            attentions.append(atts)
        return x, attentions

    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.lm_head_w.T + self.lm_head_b

    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        layers = self.forward_all_layers(ids)
        return self.unembed(layers[-1])


def layer_label(idx: int, num_blocks: int) -> str:
    """Human-readable name for lens-checkpoint `idx` in the fine-grained
    `2*num_blocks+1` scheme (see `TinyGpt`'s doc): `"embeddings"` for 0,
    `"post-block{k}-attn"` for the post-attention/pre-MLP checkpoint of block
    `k`, `"post-block{k}"` for block `k`'s full output (including the final
    checkpoint, index `2*num_blocks`, which is h_final == the last block's
    full output)."""
    if idx == 0:
        return "embeddings"
    k = (idx - 1) // 2
    is_attn_checkpoint = (idx - 1) % 2 == 0
    return f"post-block{k}-attn" if is_attn_checkpoint else f"post-block{k}"


# --------------------------------------------------------------------------
# Data: facts `a op b=c` parsed generically (op in {+, -}), any operand width.
# --------------------------------------------------------------------------
def load_facts(corpus_path: Path) -> list[tuple[int, str, int, int, str]]:
    """Returns (a, op, b, c, line)."""
    facts = []
    for line in corpus_path.read_text().splitlines():
        line = line.strip()
        m = FACT_RE.match(line)
        if not m:
            continue
        a, op, b, c = m.groups()
        facts.append((int(a), op, int(b), int(c), line))
    return facts


# --------------------------------------------------------------------------
# Step 1: faithful reproduction of jlens.fitting.jacobian_for_prompt / fit(),
# specialized to our architecture (every checkpoint except the last -- i.e.
# indices 0..2*num_blocks-1 in the fine-grained scheme, see `TinyGpt`'s doc --
# is a source candidate; target = index 2*num_blocks == h_final).
# --------------------------------------------------------------------------
def jacobian_for_prompt(model: TinyGpt, ids: torch.Tensor, source_layers: list[int]):
    """Returns {layer: (hidden,hidden) Jacobian summed-over-future-targets,
    averaged-over-source-positions}, matching fitting.jacobian_for_prompt's
    estimator (skip_first=0 here -- these sequences are short, no
    attention-sink region to exclude)."""
    T = ids.shape[0]
    hidden = model.hidden
    valid_positions = list(range(T - 1))  # exclude final position (no target)

    layers = model.forward_all_layers(ids)
    # Re-run forward starting from each source layer with requires_grad, so we
    # get an independent autograd graph per source layer (cheap at this scale;
    # the reference implementation instead reuses one retained graph across
    # dim-batched backward passes for efficiency at LLM scale -- not needed
    # here given these tiny models).
    out = {}
    for l in source_layers:
        h = layers[l].detach().clone().requires_grad_(True)
        h_final = model.run_from_layer(h, start_layer=l)  # (T, hidden)
        J = torch.zeros(hidden, hidden)
        for dim in range(hidden):
            cotangent = torch.zeros_like(h_final)
            for p in valid_positions:
                for target_p in range(p, T):
                    cotangent[target_p, dim] = 1.0
            grad, = torch.autograd.grad(h_final, h, grad_outputs=cotangent, retain_graph=(dim < hidden - 1))
            # grad[p] = sum_{p'>=p} d h_final[p', dim] / d h[p]; average over p
            rows = grad[valid_positions, :]  # (n_valid, hidden)
            J[dim, :] = rows.mean(dim=0)
        out[l] = J
    return out


def fit_lens(model: TinyGpt, facts: list[tuple[int, str, int, int, str]], charset: list[str]):
    # Every checkpoint EXCEPT the last (h_final, index `2*num_blocks`, which
    # IS the fitting target -- fitting a layer against itself would just be
    # the identity map) -- see `TinyGpt`'s doc for what each index means.
    source_layers = list(range(model.total_layers - 1))
    hidden = model.hidden
    J_sum = {l: torch.zeros(hidden, hidden) for l in source_layers}
    n = 0
    for a, op, b, c, line in facts:
        ids = torch.tensor(encode(charset, line + "\n"), dtype=torch.long)
        per_prompt = jacobian_for_prompt(model, ids, source_layers)
        for l in source_layers:
            J_sum[l] += per_prompt[l]
        n += 1
    return {l: J_sum[l] / n for l in source_layers}


# --------------------------------------------------------------------------
# Step 2: extension -- exact local per-position Jacobian (not corpus
# averaged) of the answer position's h_final w.r.t. every SOURCE position's
# residual, at every layer. Gives a genuine layer x position sensitivity map
# per individual fact.
# --------------------------------------------------------------------------
def local_position_jacobian(model: TinyGpt, ids: torch.Tensor, answer_pos: int, source_layers: list[int]):
    """Returns {layer: (hidden, T, hidden)} -- d h_final[answer_pos, :] / d h_l[:, :]."""
    hidden = model.hidden
    layers = model.forward_all_layers(ids)
    out = {}
    for l in source_layers:
        h = layers[l].detach().clone().requires_grad_(True)
        h_final = model.run_from_layer(h, start_layer=l)  # (T, hidden)
        target = h_final[answer_pos]  # (hidden,)
        rows = []
        for dim in range(hidden):
            grad, = torch.autograd.grad(target[dim], h, retain_graph=(dim < hidden - 1))
            rows.append(grad)  # (T, hidden)
        out[l] = torch.stack(rows, dim=0)  # (hidden_out, T, hidden_in)
    return out


# --------------------------------------------------------------------------
# Extension 2: layer-by-layer embedding/activation visualization (PCA + UMAP
# 2D projections). See module doc, point 5.
# --------------------------------------------------------------------------
def pca_2d(x: "np.ndarray") -> "np.ndarray":
    """Project `x` (N, D) to 2D via PCA (mean-center + truncated SVD). Pure
    numpy -- no extra dependency beyond what the rest of this script already
    needs. Degenerate inputs (N < 2 or D < 2) are padded with a zero column
    so callers always get an (N, 2) array back."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n < 2:
        return np.zeros((n, 2))
    mean = x.mean(axis=0)
    xc = x - mean
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(2, vt.shape[0])
    proj = xc @ vt[:k].T
    if k < 2:
        proj = np.hstack([proj, np.zeros((n, 2 - k))])
    return proj


def umap_2d(x: "np.ndarray") -> "np.ndarray | None":
    """Project `x` (N, D) to 2D via UMAP, or `None` if UMAP isn't installed
    or the projection fails (e.g. too few points for the requested
    neighborhood size) -- callers fall back to PCA-only in that case rather
    than failing the whole analysis over one plot."""
    if not HAVE_UMAP:
        return None
    n = x.shape[0]
    if n < 4:
        return None
    try:
        n_neighbors = max(2, min(15, n - 1))
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, n_jobs=1)
        return reducer.fit_transform(np.asarray(x, dtype=np.float64))
    except Exception as e:  # pragma: no cover - defensive
        sys.stderr.write(f"WARNING: UMAP projection failed ({e}); falling back to PCA-only for this layer.\n")
        return None


def tsne_perplexity(n: int) -> int:
    """Perplexity scaling that stays safe for tiny sample counts -- sklearn's
    `TSNE` requires `perplexity < n_samples` (it errors otherwise), and
    perplexity values close to `n_samples` produce degenerate/unstable
    embeddings well before that hard limit. The WE layer for the primary test
    model (`mask-test-4blocks-2heads`) has only `vocab_size == 13` points, so
    this can't just use sklearn's default of 30. `(n - 1) // 3` keeps
    perplexity comfortably below `n_samples` (roughly "a third of the
    points"), floored at 2 (sklearn's minimum sane value) and capped at 30
    (sklearn's own default, fine once there are enough points that the exact
    value stops mattering much)."""
    return min(30, max(2, (n - 1) // 3))


def tsne_2d(x: "np.ndarray") -> "np.ndarray | None":
    """Project `x` (N, D) to 2D via t-SNE, or `None` if scikit-learn isn't
    installed or the projection fails -- same graceful-degradation contract
    as `umap_2d`. Unlike PCA/UMAP, t-SNE has no notion of out-of-sample
    projection or a reusable basis; every layer gets its own independent fit,
    exactly like UMAP, and is then Procrustes-aligned into the running frame
    by the same `align_layer_sequence` call as PCA/UMAP (see
    `compute_embedding_viz`)."""
    if not HAVE_TSNE:
        return None
    n = x.shape[0]
    if n < 4:
        # t-SNE on a near-degenerate handful of points isn't meaningful (and
        # perplexity can't be made small enough to be safe below this) --
        # same threshold UMAP uses for the same reason.
        return None
    try:
        perplexity = tsne_perplexity(n)
        reducer = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        )
        return reducer.fit_transform(np.asarray(x, dtype=np.float64))
    except Exception as e:  # pragma: no cover - defensive
        sys.stderr.write(f"WARNING: t-SNE projection failed ({e}); omitting t-SNE for this layer.\n")
        return None


def procrustes_align(source: "np.ndarray", target: "np.ndarray", allow_scaling: bool = False) -> "np.ndarray":
    """Orthogonal Procrustes: rotate/reflect (and optionally scale) `source`
    (N, 2) to best match `target` (N, 2), assuming `source[i]` and
    `target[i]` are the SAME underlying point (same token/fact/position,
    just at a different layer) -- exactly the case here, since every layer's
    projection is over the same ordered list of (fact, position) pairs. This
    exists because PCA's principal components can flip sign or swap between
    two INDEPENDENTLY fit layers (and UMAP has essentially no consistent
    orientation across separate runs at all) -- naively lerping raw
    per-layer coordinates for an animation would partly show arbitrary basis
    differences, not real movement of the representation. Solved via SVD
    (Schonemann 1966): center both point sets, then the optimal rotation
    R minimizing ||target_c - source_c @ R||_F is `U @ Vt` where
    `U, S, Vt = svd(source_c.T @ target_c)`. Returns `source` remapped into
    `target`'s frame (rotated/reflected, and translated to `target`'s
    centroid)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_c = source - src_mean
    tgt_c = target - tgt_mean
    m = src_c.T @ tgt_c  # (2, 2)
    u, s, vt = np.linalg.svd(m)
    r = u @ vt  # optimal orthogonal (rotation or reflection) matrix
    aligned = src_c @ r
    if allow_scaling:
        denom = np.sum(src_c ** 2)
        if denom > 1e-12:
            scale = s.sum() / denom
            aligned = aligned * scale
    return aligned + tgt_mean


def align_layer_sequence(coords_by_layer: list) -> list:
    """Chains `procrustes_align` consecutively across a sequence of per-layer
    (N, 2) coordinate arrays (or `None` for a layer where the projection is
    unavailable, e.g. UMAP too unstable/uninstalled) so a full glide through
    every layer moves in one consistent frame: layer 1 is aligned to layer
    0's (unaligned, taken as the reference) frame, layer 2 is aligned to
    layer 1's ALREADY-ALIGNED frame, and so on -- each alignment builds on
    the previous one rather than every layer independently chasing layer 0,
    which would let small pairwise misalignments accumulate differently than
    a true running frame. `None` entries pass through unchanged and do not
    update the running reference (so a single missing layer in the middle
    doesn't break the chain for the layers after it)."""
    aligned = []
    prev = None
    for coords in coords_by_layer:
        if coords is None:
            aligned.append(None)
            continue
        arr = np.asarray(coords, dtype=np.float64)
        if prev is None:
            aligned.append(arr)
        else:
            aligned.append(procrustes_align(arr, prev))
        prev = aligned[-1]
    return aligned


def pca_fit_errors(vecs: list, max_k: int = 4) -> list | None:
    """% of total variance UNEXPLAINED when the token's cloud is approximated
    by its best-fit k-dimensional affine subspace (PCA truncation), for
    k=0..max_k. k=0 = a single point (the centroid); k=1 = a line; k=2 = a
    plane; etc. `None` if fewer than 2 occurrences (no variance to speak of)."""
    X = np.stack(vecs)  # (n, hidden_size) -- RAW un-projected vectors, not 2D-projected
    n = X.shape[0]
    if n < 2:
        return None
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (n - 1)
    eigvals = np.clip(np.linalg.eigvalsh(cov)[::-1], 0, None)
    total = eigvals.sum()
    if total < 1e-12:
        return [0.0] * (max_k + 1)
    errors = []
    for k in range(0, max_k + 1):
        kk = min(k, len(eigvals))
        unexplained = eigvals[kk:].sum()
        errors.append(float(unexplained / total))
    return errors


def covariance_ellipse_2d(points_2d: list) -> dict | None:
    """Fit a 2-std-dev covariance ellipse to a token's 2D (already
    Procrustes-aligned) points, SVG-ready: {cx, cy, rx, ry, angle_deg}.
    `rx`/`ry` are `sqrt(eigenvalue) * 2` (2 std devs) along each principal
    axis; `angle_deg` is the rotation of the major axis (the eigenvector for
    the LARGER eigenvalue). Returns `None` for n<2; a zero-radius ellipse
    (rather than crashing) for degenerate/near-zero covariance -- some tokens
    (e.g. '='/'+' at the embedding layer) have EXACTLY zero variance."""
    P = np.asarray(points_2d, dtype=np.float64)
    n = P.shape[0]
    if n < 2:
        return None
    mean = P.mean(axis=0)
    Pc = P - mean
    cov = (Pc.T @ Pc) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    eigvals = np.clip(eigvals, 0, None)
    # eigh returns ascending eigenvalues -- index 1 is the larger one.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    rx, ry = (2.0 * np.sqrt(eigvals)).tolist()
    major = eigvecs[:, 0]
    angle_deg = float(np.degrees(np.arctan2(major[1], major[0])))
    return {
        "cx": float(mean[0]), "cy": float(mean[1]),
        "rx": float(rx), "ry": float(ry),
        "angle_deg": angle_deg,
    }


def display_char(c: str) -> str:
    """Human-readable label for a token character -- newline (the answer
    terminator) is unreadable as a raw JSON string in a UI label, so it's
    shown as an explicit escape."""
    return "\\n" if c == "\n" else c


def compute_embedding_viz(model: TinyGpt, charset: list[str], facts: list[tuple[int, str, int, int, str]]):
    """Returns the JSON-serializable `embedding_viz` payload: the WE
    (token-embedding) layer's 2D projection, plus one 2D projection per
    layer boundary (embeddings, post-block0, ..., post-block(N-1)) of every
    token position's residual vector across the whole fact corpus. Reuses
    `forward_all_layers` (the same per-fact forward pass the Jacobian fitting
    above already runs) rather than a separate extraction pass.

    Also exports each point's RAW (un-projected, `hidden_size`-dimensional)
    vector under `"raw_vectors"` per layer -- the PCA/UMAP scatter only shows
    2 of `hidden_size` dimensions' worth of variance, so the UI's radar/spider
    chart (rendered for a single selected point) reads directly from these
    instead of the lossy 2D projection. `raw_vector_max_abs` is ONE global
    scalar (max absolute value over EVERY layer and EVERY point, computed up
    front) for the radar chart's radius scale -- deliberately NOT recomputed
    per layer/point, since a per-frame rescale would make a token's shape
    visually change just because the axis stretched, not because the
    underlying values actually moved (see the module doc / caller for the
    full rationale)."""
    we_vectors = model.tok_emb.detach().numpy()  # (vocab, hidden)
    we_labels = [display_char(c) for c in charset]
    we_layer = {
        "pca": pca_2d(we_vectors).tolist(),
        "umap": (lambda u: u.tolist() if u is not None else None)(umap_2d(we_vectors)),
        "tsne": (lambda t: t.tolist() if t is not None else None)(tsne_2d(we_vectors)),
        "labels": we_labels,
        # Raw (un-projected) hidden_size-dim vector per token -- exported here
        # for the SAME reason as each layer's `raw_vectors` below: the UI's
        # client-side force-directed ("spring") layout builds a k-nearest-
        # neighbor graph from true Euclidean distance in this raw space (not
        # from any 2D projection), and the WE-layer scatter offers that mode
        # too, so it needs its own raw vectors independent of the per-layer
        # ones (this is the embedding TABLE, not a per-fact-position
        # residual -- a different, smaller population).
        "raw_vectors": [[round(float(v), 5) for v in vec] for vec in we_vectors],
    }

    num_layers = model.total_layers  # embeddings + attn/full-output pair per block
    layer_vectors: list[list["np.ndarray"]] = [[] for _ in range(num_layers)]
    layer_point_labels: list[list[str]] = [[] for _ in range(num_layers)]
    # Which fact/position each flat-list point belongs to, plus that fact's
    # full prompt string -- IDENTICAL across every layer (same facts, same
    # order, same per-fact prompt length each time), so this is computed
    # once here rather than duplicated inside the `for li` loop below, and
    # exported as a top-level (not per-layer) key.
    point_fact_idx: list[int] = []
    point_pos_in_fact: list[int] = []
    point_prompts: list[str] = []
    for fact_idx, (a, op, b, c, line) in enumerate(facts):
        prompt = f"{a}{op}{b}="
        ids = torch.tensor(encode(charset, prompt), dtype=torch.long)
        layers = model.forward_all_layers(ids)
        for li, layer_h in enumerate(layers):
            for pos in range(layer_h.shape[0]):
                layer_vectors[li].append(layer_h[pos].detach().numpy())
                layer_point_labels[li].append(display_char(prompt[pos]))
        for pos in range(len(prompt)):
            point_fact_idx.append(fact_idx)
            point_pos_in_fact.append(pos)
            point_prompts.append(prompt)

    # Fit each layer's projection INDEPENDENTLY first (the raw, unaligned
    # fits), then Procrustes-align the sequence into one consistent running
    # frame (see `align_layer_sequence`'s doc for why this matters -- PCA
    # axes can flip/swap and UMAP has no consistent orientation at all
    # between independent fits, so raw coordinates would make a layer-to-layer
    # animation partly show arbitrary basis changes, not real movement).
    raw_pca = [pca_2d(np.stack(layer_vectors[li])) for li in range(num_layers)]
    raw_umap = [umap_2d(np.stack(layer_vectors[li])) for li in range(num_layers)]
    raw_tsne = [tsne_2d(np.stack(layer_vectors[li])) for li in range(num_layers)]
    aligned_pca = align_layer_sequence(raw_pca)
    aligned_umap = align_layer_sequence(raw_umap)
    aligned_tsne = align_layer_sequence(raw_tsne)

    # One fixed radar-chart scale, over EVERY layer and EVERY point --
    # rounded up slightly (1%) so the most extreme value doesn't sit exactly
    # on the outer boundary with no margin.
    global_max_abs = max(
        float(np.max(np.abs(np.stack(layer_vectors[li])))) for li in range(num_layers)
    )
    global_max_abs = global_max_abs * 1.01 if global_max_abs > 0 else 1.0

    layers_out = []
    for li in range(num_layers):
        label = layer_label(li, model.num_blocks)
        # Rounded to 5 decimals -- full float64 repr would meaningfully
        # bloat the JSON for no visualization-relevant precision gain.
        raw_vectors = [
            [round(float(v), 5) for v in vec] for vec in layer_vectors[li]
        ]

        # Per-token shape summary: PCA-fit-error curve over RAW vectors, plus
        # a 2D covariance ellipse in each already-aligned projection's
        # coordinate frame (so the ellipse visually matches that frame's
        # rendered dots). Group by token label; skip n<2 tokens.
        labels_arr = np.asarray(layer_point_labels[li])
        by_label_idx: dict[str, "np.ndarray"] = {
            lbl: np.where(labels_arr == lbl)[0] for lbl in dict.fromkeys(layer_point_labels[li])
        }
        shape_fit_errors = {}
        for lbl, idxs in by_label_idx.items():
            if len(idxs) < 2:
                continue
            errs = pca_fit_errors([layer_vectors[li][i] for i in idxs])
            if errs is not None:
                shape_fit_errors[lbl] = [round(e, 5) for e in errs]

        def ellipses_for(aligned_coords):
            if aligned_coords is None:
                return None
            out = {}
            for lbl, idxs in by_label_idx.items():
                if len(idxs) < 2:
                    continue
                ell = covariance_ellipse_2d([aligned_coords[i] for i in idxs])
                if ell is not None:
                    out[lbl] = ell
            return out

        pca_ellipses = ellipses_for(aligned_pca[li])
        umap_ellipses = ellipses_for(aligned_umap[li])
        tsne_ellipses = ellipses_for(aligned_tsne[li])

        layers_out.append({
            "layer": li,
            "label": label,
            # Procrustes-aligned coordinates -- what the UI animates.
            "pca": aligned_pca[li].tolist(),
            "umap": aligned_umap[li].tolist() if aligned_umap[li] is not None else None,
            "tsne": aligned_tsne[li].tolist() if aligned_tsne[li] is not None else None,
            # Independently-fit coordinates, kept for reference/debugging --
            # NOT what the UI's layer-to-layer glide should use.
            "pca_raw": raw_pca[li].tolist(),
            "umap_raw": raw_umap[li].tolist() if raw_umap[li] is not None else None,
            "tsne_raw": raw_tsne[li].tolist() if raw_tsne[li] is not None else None,
            "point_labels": layer_point_labels[li],
            # Raw hidden_size-dim vector per point, for the radar chart.
            "raw_vectors": raw_vectors,
            # Per-token PCA-reconstruction-error curve (k=0..4) over RAW
            # vectors, and per-projection covariance ellipses in aligned 2D
            # coordinates -- see `pca_fit_errors`/`covariance_ellipse_2d`.
            "shape_fit_errors": shape_fit_errors,
            "pca_ellipses": pca_ellipses,
            "umap_ellipses": umap_ellipses,
            "tsne_ellipses": tsne_ellipses,
        })

    return {
        "have_umap": HAVE_UMAP,
        "have_tsne": HAVE_TSNE,
        "alignment": "procrustes",
        "we_layer": we_layer,
        "layers": layers_out,
        "raw_vector_max_abs": global_max_abs,
        # Fact/position metadata for each flat-list point index -- same
        # length/order as any single layer's `raw_vectors`/`point_labels`,
        # and invariant across layers (unlike those, which are per-layer
        # because the VECTORS differ by layer), so these live at the top
        # level instead of being duplicated inside every per-layer dict.
        "point_fact_idx": point_fact_idx,
        "point_pos_in_fact": point_pos_in_fact,
        "point_prompts": point_prompts,
    }


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    corpus_path = Path(args.dataset_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    heads_schedule = resolve_heads_schedule(args.num_heads, args.num_blocks, args.hidden_size)

    corpus = corpus_path.read_text()
    charset = build_charset(corpus)
    if len(charset) != args.vocab_size:
        sys.stderr.write(
            f"WARNING: corpus charset has {len(charset)} distinct characters but "
            f"--vocab-size was {args.vocab_size}; proceeding with the corpus-derived "
            f"charset (the model's embedding table must match it exactly).\n"
        )
    model = TinyGpt(model_path, args.hidden_size, heads_schedule, args.num_blocks, args.block_size)
    facts = load_facts(corpus_path)
    if not facts:
        sys.stderr.write(f"No 'a op b=c' facts found in {corpus_path}; nothing to analyze.\n")
        sys.exit(1)
    print(f"Loaded {len(facts)} facts, charset={charset}, heads_schedule={heads_schedule}")

    # ---- ground-truth behavioral eval: greedy decode a+b= -> does it match c?
    correctness = {}
    for a, op, b, c, line in facts:
        prompt = f"{a}{op}{b}="
        ids = torch.tensor(encode(charset, prompt), dtype=torch.long)
        logits = model.logits(ids)
        pred = int(logits[-1].argmax())
        correctness[(a, op, b)] = (pred == charset.index(str(c)))
    n_correct = sum(correctness.values())
    print(f"Greedy exact-match accuracy (this reimplementation): {n_correct}/{len(facts)}")

    # =====================================================================
    # PART A: faithful Jacobian lens (corpus-fitted J_l), applied at the "="
    # position for every fact.
    # =====================================================================
    print("\nFitting J_l over the corpus (this may take a while for larger models)...")
    J = fit_lens(model, facts, charset)
    for l in sorted(J):
        print(f"  ||J_{l}||_F = {J[l].norm().item():.3f}")

    rows = []
    for a, op, b, c, line in facts:
        prompt = f"{a}{op}{b}="
        ids = torch.tensor(encode(charset, prompt), dtype=torch.long)
        layers = model.forward_all_layers(ids)
        eq_pos = len(prompt) - 1  # position of "="
        true_digit_tok = charset.index(str(c))
        row = {"a": a, "op": op, "b": b, "c": c, "correct": correctness[(a, op, b)]}
        for l in sorted(J):
            h = layers[l][eq_pos]
            transported = J[l] @ h
            lens_logits = model.unembed(transported.unsqueeze(0)).squeeze(0)
            top1 = int(lens_logits.argmax())
            rank_of_true = int((lens_logits > lens_logits[true_digit_tok]).sum())  # 0 = top
            row[f"lens_l{l}_top1"] = charset[top1]
            row[f"lens_l{l}_rank_true"] = rank_of_true
        # also the true final-layer readout (should reproduce the model's own top1 exactly)
        h_final = layers[model.total_layers - 1][eq_pos]
        final_logits = model.unembed(h_final.unsqueeze(0)).squeeze(0)
        row["model_top1"] = charset[int(final_logits.argmax())]
        row["final_rank_true"] = int((final_logits > final_logits[true_digit_tok]).sum())
        rows.append(row)

    # Layer-by-layer "does the lens already know the answer" accuracy -- now
    # at FINE (intra-block) granularity: every `post-block{k}-attn` checkpoint
    # is included alongside the block-boundary `post-block{k}` checkpoints, so
    # this directly answers "does this block's attention step already do the
    # work, or does its MLP do it?" for every block, not just qualitatively.
    print("\nLens top-1 == true digit, by layer (at the '=' position):")
    layer_acc = {}
    for l in sorted(J):
        acc = sum(1 for r in rows if r[f"lens_l{l}_top1"] == str(r["c"])) / len(rows)
        layer_acc[l] = acc
        label = layer_label(l, model.num_blocks)
        print(f"  layer {l} ({label:>18}): lens-top1-correct = {acc:.1%}")
    print(f"  layer {model.total_layers - 1} ({'h_final':>18}, true output): "
          f"{sum(1 for r in rows if r['model_top1'] == str(r['c'])) / len(rows):.1%}  "
          f"(cross-check: should equal the {n_correct}/{len(facts)} greedy accuracy above)")

    # Split correct vs incorrect facts, and "+0"/"+1" vs arbitrary.
    def is_successor(a, b):
        return b in (0, 1) or a in (0, 1)

    groups = {
        "correct": [r for r in rows if r["correct"]],
        "incorrect": [r for r in rows if not r["correct"]],
        "successor (+0/+1)": [r for r in rows if is_successor(r["a"], r["b"])],
        "arbitrary": [r for r in rows if not is_successor(r["a"], r["b"])],
    }
    print("\nMean rank of the TRUE digit in the lens's ranking (0 = top1), by layer and group:")
    print(f"{'group':<20}" + "".join(f"L{l:>7}" for l in sorted(J)) + "  h_final")
    group_layer_rank = {}
    for name, grp in groups.items():
        if not grp:
            continue
        line_out = f"{name:<20}"
        group_layer_rank[name] = {}
        for l in sorted(J):
            mean_rank = sum(r[f"lens_l{l}_rank_true"] for r in grp) / len(grp)
            group_layer_rank[name][l] = mean_rank
            line_out += f"{mean_rank:8.2f}"
        final_mean_rank = sum(r["final_rank_true"] for r in grp) / len(grp)
        group_layer_rank[name][model.total_layers - 1] = final_mean_rank
        line_out += f"{final_mean_rank:9.2f}"
        print(line_out)

    # =====================================================================
    # PART B (extension): per-fact, per-source-position local Jacobian
    # sensitivity heatmap -- which POSITION (a, op, b, '=') does the answer
    # prediction actually depend on, at each layer? Facts can have differing
    # prompt lengths (e.g. multi-digit operands); we restrict this heatmap to
    # the MODAL prompt length so every row is directly comparable, and note
    # how many facts were excluded.
    # =====================================================================
    print("\nComputing per-fact position x layer sensitivity heatmaps (extension)...")
    # Same fine-grained (intra-block) checkpoint list as the Jacobian-lens
    # fitting above -- every checkpoint except h_final itself.
    source_layers = list(range(model.total_layers - 1))
    length_counts = Counter(len(f"{a}{op}{b}=") for a, op, b, c, _ in facts)
    modal_len = length_counts.most_common(1)[0][0]
    excluded = sum(cnt for ln, cnt in length_counts.items() if ln != modal_len)
    facts_for_heat = [(a, op, b, c, line) for a, op, b, c, line in facts if len(f"{a}{op}{b}=") == modal_len]
    sample_a, sample_op, sample_b, _, _ = facts_for_heat[0]
    sample_prompt = f"{sample_a}{sample_op}{sample_b}="
    # Structural labels (not the literal digits of one sample fact): every
    # char before the operator is "a", the operator itself, every char
    # between the operator and "=" is "b", and the trailing "=".
    op_idx = len(str(sample_a))
    position_labels = []
    for i, ch in enumerate(sample_prompt):
        if i < op_idx:
            position_labels.append("a")
        elif i == op_idx:
            position_labels.append(ch)  # '+' or '-'
        elif i == len(sample_prompt) - 1:
            position_labels.append("=")
        else:
            position_labels.append("b")
    n_pos = len(position_labels)
    sens_by_group = {name: np.zeros((len(source_layers), n_pos)) for name in groups}
    sens_count = {name: 0 for name in groups}
    per_fact_sens = {}
    for a, op, b, c, line in facts_for_heat:
        prompt = f"{a}{op}{b}="
        ids = torch.tensor(encode(charset, prompt), dtype=torch.long)
        eq_pos = len(prompt) - 1
        J_local = local_position_jacobian(model, ids, eq_pos, source_layers)
        heat = np.zeros((len(source_layers), n_pos))
        for li, l in enumerate(source_layers):
            block = J_local[l]  # (hidden_out, T, hidden_in)
            for p in range(n_pos):
                heat[li, p] = block[:, p, :].norm().item()
        per_fact_sens[(a, b)] = heat
        # accumulate into groups using correctness/successor flags computed above
        if correctness[(a, op, b)]:
            sens_by_group["correct"] += heat
            sens_count["correct"] += 1
        else:
            sens_by_group["incorrect"] += heat
            sens_count["incorrect"] += 1
        if is_successor(a, b):
            sens_by_group["successor (+0/+1)"] += heat
            sens_count["successor (+0/+1)"] += 1
        else:
            sens_by_group["arbitrary"] += heat
            sens_count["arbitrary"] += 1

    for name in groups:
        if sens_count[name] > 0:
            sens_by_group[name] /= sens_count[name]

    # Normalize each heatmap so rows (layers) sum to 1 -- makes the SHAPE of
    # the dependency (which positions matter) comparable across groups even
    # if overall gradient magnitude differs.
    def row_normalize(h):
        s = h.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        return h / s

    print(f"\n(Position heatmap covers the {modal_len}-char-prompt subset: "
          f"{len(facts_for_heat)}/{len(facts)} facts; {excluded} facts with a different prompt length excluded.)")
    print("Mean position-sensitivity (Frobenius norm of dh_final(=pos)/dh_l(pos), row-normalized "
          f"so each layer sums to 1 across positions {''.join(position_labels)}):")
    for name in ["correct", "incorrect", "successor (+0/+1)", "arbitrary"]:
        h = row_normalize(sens_by_group[name])
        print(f"\n  {name} (n={sens_count[name]}):")
        print(f"    {'layer':<20}" + "".join(f"{p:>8}" for p in position_labels))
        for li, l in enumerate(source_layers):
            label = layer_label(l, model.num_blocks)
            print(f"    {label:<20}" + "".join(f"{h[li,p]:8.3f}" for p in range(n_pos)))

    # =====================================================================
    # PART C (corroborating, non-Jacobian evidence): raw attention weight
    # from the '=' query position to the first two operand positions, per
    # block/head, averaged over all (modal-length) facts.
    # =====================================================================
    n_heads_max = max(heads_schedule)
    print("\nAttention weight FROM '=' TO operand positions, per block/head (mean over facts):")
    att_to_a = np.zeros((model.num_blocks, n_heads_max))
    att_to_b = np.zeros((model.num_blocks, n_heads_max))
    att_counted = np.zeros((model.num_blocks, n_heads_max))
    b_pos = 2 if n_pos > 2 else min(2, n_pos - 1)
    for a, op, b, c, line in facts_for_heat:
        prompt = f"{a}{op}{b}="
        ids = torch.tensor(encode(charset, prompt), dtype=torch.long)
        eq_pos = len(prompt) - 1
        _, attentions = model.forward_with_attention(ids)
        for bi, atts in enumerate(attentions):
            for hi, att in enumerate(atts):
                att_to_a[bi, hi] += att[eq_pos, 0].item()
                att_to_b[bi, hi] += att[eq_pos, b_pos].item()
                att_counted[bi, hi] += 1
    att_counted[att_counted == 0] = 1
    att_to_a /= att_counted
    att_to_b /= att_counted
    print(f"  {'block':<8}{'head':<8}{'att->pos0':<12}{'att->pos' + str(b_pos):<12}")
    for bi in range(model.num_blocks):
        for hi in range(heads_schedule[bi]):
            print(f"  {bi:<8}{hi:<8}{att_to_a[bi,hi]:<12.3f}{att_to_b[bi,hi]:<12.3f}")

    # =====================================================================
    # Save plots
    # =====================================================================
    plot_lens_layer_accuracy(out_dir, layer_acc, n_correct, len(facts), model.num_blocks)
    plot_group_rank(out_dir, group_layer_rank, source_layers, model.num_blocks)
    plot_sensitivity_heatmaps(out_dir, sens_by_group, sens_count, position_labels, source_layers, model.num_blocks)
    plot_attention(out_dir, att_to_a, att_to_b, heads_schedule, model.num_blocks, b_pos)

    viz_backends = "PCA" + (" + UMAP" if HAVE_UMAP else "") + (" + t-SNE" if HAVE_TSNE else "")
    print(f"\nComputing layer-by-layer embedding/activation visualization ({viz_backends})...")
    embedding_viz = compute_embedding_viz(model, charset, facts)

    # Save raw numbers for the report.
    dump = {
        "model_path": str(model_path),
        "dataset_path": str(corpus_path),
        "hidden_size": args.hidden_size,
        "heads_schedule": heads_schedule,
        "num_blocks": args.num_blocks,
        "block_size": args.block_size,
        "vocab_size": args.vocab_size,
        "n_facts": len(facts),
        "greedy_accuracy": n_correct / len(facts),
        "layer_lens_top1_accuracy": layer_acc,
        "group_layer_mean_rank": group_layer_rank,
        "group_sensitivity_row_normalized": {
            name: row_normalize(sens_by_group[name]).tolist() for name in groups
        },
        "group_counts": sens_count,
        "position_labels": position_labels,
        "position_heatmap_excluded_facts": excluded,
        "attention_eq_to_pos0": att_to_a.tolist(),
        "attention_eq_to_pos_b": att_to_b.tolist(),
        "embedding_viz": embedding_viz,
        "plots": [
            "layer_accuracy.png",
            "group_rank.png",
            "sensitivity_heatmaps.png",
            "attention_routing.png",
        ],
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nWrote {out_dir / 'results.json'}")


def plot_lens_layer_accuracy(out_dir, layer_acc, n_correct, n_total, num_blocks):
    fig, ax = plt.subplots(figsize=(max(6, len(layer_acc) * 0.85), 4))
    layers = sorted(layer_acc)
    labels = [layer_label(l, num_blocks) for l in layers] + ["h_final"]
    accs = [layer_acc[l] for l in layers] + [n_correct / n_total]
    ax.plot(labels, accs, marker="o", linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("lens top-1 == true answer digit")
    ax.set_title("Jacobian lens: when does the correct digit become the top prediction?\n(read out at the '=' position; fine-grained -- includes post-attn/pre-MLP checkpoints)")
    ax.tick_params(axis="x", labelrotation=60)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "layer_accuracy.png", dpi=150)
    plt.close(fig)


def plot_group_rank(out_dir, group_layer_rank, source_layers, num_blocks):
    fig, ax = plt.subplots(figsize=(max(6, (len(source_layers) + 1) * 0.85), 4))
    final_layer_idx = source_layers[-1] + 1  # h_final, one past the last source checkpoint
    labels = [layer_label(l, num_blocks) for l in source_layers] + ["h_final"]
    for name, per_layer in group_layer_rank.items():
        ys = [per_layer[l] for l in source_layers] + [per_layer[final_layer_idx]]
        ax.plot(labels, ys, marker="o", label=name)
    ax.set_ylabel("mean rank of true digit (0 = top1)")
    ax.set_title("Jacobian lens: rank of the correct digit by layer, by fact group")
    ax.tick_params(axis="x", labelrotation=60)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "group_rank.png", dpi=150)
    plt.close(fig)


def plot_sensitivity_heatmaps(out_dir, sens_by_group, sens_count, position_labels, source_layers, num_blocks):
    def row_normalize(h):
        s = h.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        return h / s

    names = ["correct", "incorrect", "successor (+0/+1)", "arbitrary"]
    n_pos = len(position_labels)
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4 + len(source_layers) * 0.15), sharey=True)
    layer_labels = [layer_label(l, num_blocks) for l in source_layers]
    im = None
    for ax, name in zip(axes, names):
        h = row_normalize(sens_by_group[name])
        im = ax.imshow(h, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(n_pos))
        ax.set_xticklabels(position_labels)
        ax.set_yticks(range(len(layer_labels)))
        ax.set_yticklabels(layer_labels)
        ax.set_title(f"{name}\n(n={sens_count[name]})")
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                ax.text(j, i, f"{h[i,j]:.2f}", ha="center", va="center",
                        color="white" if h[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="share of ||dh_final(=)/dh_l|| per row")
    fig.suptitle("Position sensitivity of the answer prediction, by layer (row-normalized)", y=1.08)
    fig.savefig(out_dir / "sensitivity_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention(out_dir, att_to_a, att_to_b, heads_schedule, num_blocks, b_pos):
    n_heads_max = max(heads_schedule)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    im = None
    for ax, data, title in zip(axes, [att_to_a, att_to_b], ["attention '=' -> pos0", f"attention '=' -> pos{b_pos}"]):
        im = ax.imshow(data, vmin=0, vmax=max(att_to_a.max(), att_to_b.max(), 1e-6), cmap="cividis", aspect="auto")
        ax.set_xticks(range(n_heads_max))
        ax.set_xticklabels([f"head{h}" for h in range(n_heads_max)])
        ax.set_yticks(range(num_blocks))
        ax.set_yticklabels([f"block{b}" for b in range(num_blocks)])
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(heads_schedule[i]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if data[i, j] < data.max() * 0.6 else "black")
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean attention weight")
    fig.suptitle("Raw attention from '=' to operand positions, by block/head\n(corroborating evidence, not part of the Jacobian lens)", y=1.12)
    fig.savefig(out_dir / "attention_routing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
