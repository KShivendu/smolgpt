use clap::{ArgGroup, Parser};
use std::path::PathBuf;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum, Debug)]
pub enum ModelType {
    Gpt,
    Bigram,
    /// N-gram model (see `model::ngram::NgramLM`): a faithful generalization
    /// of `Bigram` that conditions on the previous `--ngram-order - 1`
    /// tokens instead of just 1. `--ngram-order 2` is bigram-equivalent.
    Ngram,
}

#[derive(Copy, Clone, PartialEq, Eq, clap::ValueEnum, Debug)]
pub enum TokenizerType {
    /// One token per character (vocab = distinct chars in the corpus).
    Char,
    /// Byte-level Byte-Pair Encoding trained on the corpus.
    Bpe,
}

/// GRPO variant: `Lite` = REINFORCE with group-relative advantage (single
/// on-policy step, no ratio/clip/KL); `Full` = PPO-style with importance
/// ratio, clipping, KL-to-reference penalty, and K mini-epochs per group.
/// `Lite` is the default so existing behavior is unchanged.
#[derive(Copy, Clone, PartialEq, Eq, clap::ValueEnum, Debug)]
pub enum GrpoMode {
    Lite,
    Full,
}

/// Eval-range behavior mode. See `Args::eval_mode` doc for the full
/// description. `Copy` so it threads through `AppState` without an `Arc`.
/// Derives `clap::ValueEnum` so `--eval-mode smart|legacy` parses via the
/// same machinery as `ModelType` and `TokenizerType`.
#[derive(Copy, Clone, PartialEq, Eq, clap::ValueEnum, Debug)]
pub enum EvalMode {
    /// Derive `eval_min`/`eval_max` from the training corpus at registration
    /// when the CLI flags are omitted, and filter `latest_eval` by the
    /// model's current range so stale rows from an old range are hidden.
    Smart,
    /// Pre-smart behavior: stored CLI range (default 0–999) and plain
    /// latest-by-timestamp eval caching with no range filter.
    Legacy,
}

#[derive(Parser, Debug)]
#[clap(name = "smolgpt", about = "A smol GPT model")]
#[clap(group(ArgGroup::new("mode").multiple(true).required(true)))]
pub struct Args {
    /// The path to the model file.
    #[clap(short = 'p', long)]
    pub model_path: Option<PathBuf>,

    #[clap(short = 'm', value_enum, default_value_t = ModelType::Gpt)]
    pub model_type: ModelType,

    /// How to tokenize the corpus.
    #[clap(short = 'k', long, value_enum, default_value_t = TokenizerType::Bpe)]
    pub tokenizer: TokenizerType,

    /// Target vocabulary size for the BPE tokenizer (>= 256, ignored for char).
    /// BPE starts from the 256 raw bytes, so anything below that floor would
    /// silently clamp to 0 learned merges (a 256-token vocab) instead of
    /// erroring on the value the user actually asked for.
    #[clap(long, default_value = "1024", value_parser = parse_vocab_size)]
    pub vocab_size: usize,

    /// The path to the dataset file.
    #[clap(short, long, default_value = "data/tinyshakespeare.txt")]
    pub dataset_path: PathBuf,

    /// Number of training epochs.
    #[clap(short, long, default_value = "100")]
    pub epochs: usize,

    /// Early-stopping patience: max epochs the smoothed training loss can go
    /// without improving by more than `--min-delta` before training halts. 0
    /// disables early stopping (run all `--epochs`). The smoothed loss is a
    /// rolling mean over the last 20 epochs, so per-epoch loss noise doesn't
    /// trip the counter prematurely. On by default.
    #[clap(long, default_value_t = 200)]
    pub patience: usize,

    /// Minimum smoothed-loss improvement required to reset the early-stopping
    /// counter. Smaller values are more sensitive (stop later); larger values
    /// stop sooner. Ignored when `--patience 0`.
    #[clap(long, default_value_t = 0.001)]
    pub min_delta: f32,

    /// Disable dropout for `--train` (regular SFT). Dropout regularizes
    /// against overfitting a large/diverse corpus, but for a small,
    /// fully-memorizable one (e.g. a few dozen arithmetic facts) it caps how
    /// low the training loss can go and prevents the model from reaching the
    /// confident weights needed to get every training example right — pass
    /// this when you actually want the model to memorize its training set.
    /// Off by default so existing behavior (dropout on) is unchanged.
    #[clap(long, default_value = "false")]
    pub no_dropout: bool,

    /// EXPERIMENTAL: fixed AdamW learning rate for `--train`'s SFT loop, used
    /// to test whether candle's default (0.001, fixed for the whole run) is
    /// too large to converge past the loss-plateau seen on tiny
    /// fully-memorizable corpora (e.g. 1-digit arithmetic). No schedule/decay
    /// — just a different constant.
    #[clap(long, default_value_t = 0.001)]
    pub lr: f64,

    /// EXPERIMENTAL: compute `--train`'s SFT cross-entropy loss only over
    /// "answer" token positions (the result digit(s) + trailing newline of
    /// each `a op b=c` line) instead of uniformly over every token in the
    /// sampled window. Only meaningful with `--tokenizer char` on an
    /// arithmetic-shaped corpus (see `dataset::compute_answer_mask`) — for
    /// any other corpus/tokenizer this silently falls back to the unmasked
    /// loss (mask defaults to all-1.0). Off by default.
    #[clap(long, default_value = "false")]
    pub mask_loss: bool,

    /// EXPERIMENTAL (Hypothesis B, train/inference format-mismatch test):
    /// sample `--train`'s SFT training windows ONLY from true `"a op b="`
    /// fact boundaries (offset 0, or immediately after a `\n`) instead of
    /// uniformly over the whole encoded corpus. Motivation: with uniform
    /// sampling, only ~1-in-6 to 1-in-7 training windows happen to start at
    /// a real fact boundary — the rest start mid-fact, with "position 0" of
    /// the window landing on an arbitrary character — while GRPO/RFT
    /// sampling and `--eval` always present a clean, complete prompt
    /// starting fresh at position 0. This flag removes that mismatch for
    /// SFT. Only meaningful with `--tokenizer char` on an arithmetic-shaped
    /// corpus (same assumption as `--mask-loss`/`dataset::compute_answer_mask`)
    /// — for any other corpus/tokenizer this silently falls back to the
    /// original uniform sampling. Off by default so existing training
    /// behavior/models are completely unaffected.
    #[clap(long, default_value = "false")]
    pub aligned_windows: bool,

    /// EXPERIMENTAL (Experiment A, init-scale ablation): stdev used for
    /// `Init::Randn` when constructing a FRESH Gpt model, applied
    /// consistently to every weight matrix (token/position embeddings,
    /// attention Q/K/V/proj, MLP fc1/fc2, and `lm_head` when untied) — not
    /// just the embedding tables, since mixing a small embedding-only init
    /// with an unrelated default everywhere else would introduce a scale
    /// mismatch rather than test the intended hypothesis. Defaults to `1.0`,
    /// which is a special sentinel (see `model::gpt::DEFAULT_INIT_STD`) that
    /// reproduces today's exact fresh-init scheme byte-for-byte (Kaiming-normal
    /// weights for the attention/MLP `Linear` layers via candle_nn's own
    /// `linear`/`linear_no_bias`, `Randn(0, 1.0)` for the embedding tables) —
    /// so omitting this flag changes NOTHING about existing training runs.
    /// Only meaningful when constructing a NEW model (`--train` without an
    /// existing file at `--model-path`); ignored when loading an existing
    /// `.bin` (the saved weights are used as-is regardless of what
    /// `--init-std` was passed at load time).
    #[clap(long, default_value_t = 1.0)]
    pub init_std: f32,

    /// EXPERIMENTAL (init-gain ablation, follow-up to Experiment A): override
    /// the GAIN constant used in candle's own Kaiming-Normal weight init
    /// (`std = gain / sqrt(fan_in)`) for every `Linear` layer built via
    /// `model::gpt::build_linear` (attention Q/K/V/proj, MLP fc1/fc2, and
    /// `lm_head` when untied — NOT the embedding tables, which are governed
    /// by `--init-std` directly, and NOT `Linear` layers whose weights come
    /// from `--init-std` instead of candle's default Kaiming-Normal — see
    /// below). candle's own default is a hardcoded "textbook" ReLU gain of
    /// √2≈1.414 (`Init::DEFAULT_KAIMING_NORMAL`); PyTorch's default `Linear`
    /// init uses a much smaller gain of √(1/3)≈0.577. On a from-scratch
    /// PyTorch replica of this repo's tiny (`hidden-size 16, num-heads 2,
    /// num-blocks 4`) architecture, swapping to the smaller PyTorch-default
    /// gain measurably raised final training accuracy on a trivial
    /// single-digit-arithmetic memorization task versus candle's larger
    /// default gain — this flag lets that hypothesis be tested directly in
    /// the real Rust/candle model rather than only in the Python replica.
    /// Unset (`None`, the CLI default) leaves candle's built-in gain (√2)
    /// completely untouched — omitting this flag changes NOTHING about
    /// existing training runs. Precedence versus `--init-std`: `--init-std`
    /// is the more direct override (it sets the absolute stdev, not just the
    /// gain feeding into a fan-in-scaled formula), so if `--init-std` is set
    /// away from its own sentinel (`1.0`), it wins and `--init-gain` is
    /// ignored — the two flags are meant to be used one at a time, not
    /// combined. Only meaningful when constructing a NEW model (same
    /// ignored-on-load rule as `--init-std`).
    #[clap(long)]
    pub init_gain: Option<f64>,

    /// EXPERIMENTAL (Experiment B, weight tying): make `lm_head` reuse
    /// `token_embeddings`' weight tensor directly (transposed at matmul time
    /// by `Linear::forward`, same as any other `Linear`) instead of
    /// allocating an independent `(vocab_size, hidden_size)` matrix. A
    /// separate small `lm_head.bias` is still kept. Off by default so
    /// existing models/behavior are completely unaffected. MUST be passed
    /// consistently between the run that created a model (`--train`) and any
    /// later run that loads it (`--eval`/`--rft`/`--grpo`/`--quantize`/
    /// `--generate` with `-p` pointing at that file) — like `--num-heads`/
    /// `--hidden-size`, this describes the saved architecture, not a
    /// runtime-only toggle, and passing the wrong value will fail to load
    /// (untied-but-actually-tied) or silently skip loading `lm_head`'s real
    /// trained weights (tied-but-actually-untied).
    #[clap(long, default_value = "false")]
    pub tie_embeddings: bool,

    /// Random seed for batch sampling and token generation (reproducible runs).
    /// Note: fresh model init uses candle's CPU RNG which cannot be seeded in
    /// candle-core 0.9.1, so full reproducibility requires a saved model on
    /// disk. If unset, OS entropy is used.
    #[clap(long)]
    pub seed: Option<u64>,

    /// Whether to train the model.
    #[clap(short, long, default_value = "false", group = "mode")]
    pub train: bool,

    /// Whether to generate from the model
    #[clap(short, long, default_value = "false", group = "mode")]
    pub generate: bool,

    /// Run greedy-decoding eval on held-out arithmetic problems.
    #[clap(long, default_value = "false", group = "mode")]
    pub eval: bool,

    /// Number of held-out problems to evaluate.
    #[clap(long, default_value_t = 200)]
    pub eval_samples: usize,

    /// Inclusive lower bound for eval operands. If omitted, `smart` mode
    /// derives the bound from the training corpus's actual operand range at
    /// registration time (so a 1-digit model is evaluated on 1-digit operands
    /// even without `--eval-min 0 --eval-max 9`); `legacy` mode falls back to
    /// 0. Explicitly passing this flag overrides corpus-derived values in
    /// both modes.
    #[clap(long)]
    pub eval_min: Option<i64>,

    /// Inclusive upper bound for eval operands. If omitted, `smart` mode
    /// derives the bound from the training corpus's actual operand range at
    /// registration time; `legacy` mode falls back to 999. Explicitly passing
    /// this flag overrides corpus-derived values in both modes.
    #[clap(long)]
    pub eval_max: Option<i64>,

    /// Eval behavior mode.
    ///
    /// - `smart` (default): at registration, derive `eval_min`/`eval_max` from
    ///   the training corpus's actual operand range when `--eval-min`/
    ///   `--eval-max` aren't passed. The web UI's `cached_eval` is filtered by
    ///   the model's current range, so a range fix instantly hides stale eval
    ///   rows from the UI without deleting them.
    ///
    /// - `legacy`: revert to the pre-smart behavior. Stored CLI range (default
    ///   0–999) is used at registration when the flags are omitted, and the
    ///   UI's `cached_eval` is the newest eval row by `run_at` with no range
    ///   filter. Use this to bisect a regression if `smart` mode misbehaves.
    #[clap(long, value_enum, default_value_t = EvalMode::Smart)]
    pub eval_mode: EvalMode,

    /// Comma-separated operators for `--eval` (e.g. `+` or `+,-`). Default
    /// `+,-`. Use `+` to eval addition only. Mirrors `--grpo-ops`/`--rft-ops`.
    #[clap(long, default_value = "+,-")]
    pub eval_ops: String,

    /// Run RFT (Rejection sampling Fine-Tuning): sample -> filter by correctness -> SFT on winners -> repeat.
    #[clap(long, default_value = "false", group = "mode")]
    pub rft: bool,

    /// Run GRPO-lite (group-relative policy gradient): sample G completions per
    /// prompt -> reward each (correct/wrong) -> policy-gradient step that
    /// pushes UP correct completions and DOWN wrong ones. Unlike RFT, wrong
    /// answers carry corrective gradient signal. No ratio clipping / KL
    /// (the "lite" in the name). Requires a pretrained model (`-p`).
    #[clap(long, default_value = "false", group = "mode")]
    pub grpo: bool,

    /// Number of GRPO rounds.
    #[clap(long, default_value_t = 3)]
    pub grpo_rounds: usize,

    /// Number of prompts per GRPO round (one optimizer step per prompt).
    #[clap(long, default_value_t = 500)]
    pub grpo_prompts: usize,

    /// Group size G: completions sampled per prompt for advantage estimation.
    #[clap(long, default_value_t = 8)]
    pub grpo_group: usize,

    /// Sampling temperature for GRPO completion generation.
    #[clap(long, default_value_t = 1.0)]
    pub grpo_temperature: f32,

    /// AdamW learning rate for GRPO policy-gradient steps.
    #[clap(long, default_value_t = 1e-3)]
    pub grpo_lr: f64,

    /// Inclusive lower bound for GRPO prompt operands.
    #[clap(long, default_value_t = 0)]
    pub grpo_min: i64,

    /// Inclusive upper bound for GRPO prompt operands.
    #[clap(long, default_value_t = 999)]
    pub grpo_max: i64,

    /// Comma-separated operators for GRPO prompt generation + per-round eval,
    /// e.g. `+` or `+,-`. Default `+,-`.
    #[clap(long, default_value = "+,-")]
    pub grpo_ops: String,

    /// GRPO variant: `lite` (REINFORCE + group-relative advantage, single
    /// on-policy step, no clipping/KL) or `full` (PPO-style: importance
    /// ratio + clipping + KL-to-reference + K mini-epochs per group). `lite`
    /// is the default so existing behavior is unchanged.
    #[clap(long, value_enum, default_value_t = GrpoMode::Lite)]
    pub grpo_mode: GrpoMode,

    /// PPO clipping epsilon for `--grpo-mode full`: the ratio
    /// `exp(logp_theta - old_logp)` is clamped to `[1-eps, 1+eps]` before the
    /// surrogate loss. Ignored by `lite`. Default 0.2 (standard PPO value).
    #[clap(long, default_value_t = 0.2)]
    pub grpo_clip_eps: f64,

    /// KL penalty coefficient for `--grpo-mode full`: the total loss is
    /// `policy_loss + beta * KL(logp_theta || logp_ref)`. Ignored by `lite`.
    /// Default 0.04 (a moderate penalty; tune down if the policy moves too
    /// sluggishly, up if it drifts too far from the reference).
    #[clap(long, default_value_t = 0.04)]
    pub grpo_kl_beta: f64,

    /// Number of mini-epochs per group for `--grpo-mode full`: each group's
    /// cached completions are re-used for K optimizer steps (recomputing
    /// `logp_theta` each step, reusing cached `old_logp`/`ref_logp`).
    /// Ignored by `lite`. Default 1 (purely on-policy, like lite but with
    /// ratio/clip/KL); 2-4 is typical for PPO-style reuse.
    #[clap(long, default_value_t = 1)]
    pub grpo_epochs: usize,

    /// Post-training INT8 quantization for storage: load an existing model
    /// (`-p <model>.bin`), quantize every trainable tensor to int8 (per-tensor
    /// symmetric scale, see `crate::quantize`'s module doc), and write it as a
    /// new variant `<stem>-quant.bin`, registered in `smolgpt.db` linked to
    /// the base model via `base_model_id` (same "derive a variant, never
    /// mutate the base" pattern `--rft`/`--grpo` use). Forward passes still
    /// run in f32 (weights are dequantized on load) — this only shrinks the
    /// on-disk file (~4x smaller: int8 is 1 byte vs f32's 4 bytes, modulo
    /// small header/scale overhead).
    #[clap(long, default_value = "false", group = "mode")]
    pub quantize: bool,

    /// Start a local web UI to browse models, datasets, and run evals.
    #[clap(long, default_value = "false", group = "mode")]
    pub serve: bool,

    /// "Compiled" precompute: after `--train` finishes (Gpt-type models
    /// only), automatically run the Jacobian-lens interpretability analysis
    /// (see `crate::jacobian_lens` / `analysis/jacobian_lens.py`) and cache
    /// the result in `smolgpt.db`, the same way `--serve`'s Jacobian tab
    /// would on-demand -- so the result is already there, no UI click
    /// needed, the moment training finishes. Ignored (with a printed note,
    /// not an error) for `-m bigram`/`-m ngram`, which have no transformer
    /// layers to lens through. Off by default since it shells out to a
    /// separate Python/torch process and is noticeably slower than the rest
    /// of `--train`.
    #[clap(long, default_value = "false")]
    pub jacobian_lens: bool,

    /// Port for the --serve web UI.
    #[clap(long, default_value_t = 8080)]
    pub port: u16,

    /// Host to bind the --serve web UI to.
    #[clap(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Number of RFT rounds.
    #[clap(long, default_value_t = 3)]
    pub rft_rounds: usize,

    /// Number of prompts per round.
    #[clap(long, default_value_t = 1000)]
    pub rft_prompts: usize,

    /// Completions sampled per prompt (the "K" in RFT).
    #[clap(long, default_value_t = 16)]
    pub rft_samples: usize,

    /// Sampling temperature for completion generation.
    #[clap(long, default_value_t = 1.0)]
    pub rft_temperature: f32,

    /// SFT epochs per round (on the winners corpus).
    #[clap(long, default_value_t = 100)]
    pub rft_epochs: usize,

    /// Inclusive lower bound for RFT prompt operands.
    #[clap(long, default_value_t = 0)]
    pub rft_min: i64,

    /// Inclusive upper bound for RFT prompt operands.
    #[clap(long, default_value_t = 999)]
    pub rft_max: i64,

    /// Comma-separated operators for RFT prompt generation + per-round eval,
    /// e.g. `+` or `+,-`. Default `+,-`.
    #[clap(long, default_value = "+,-")]
    pub rft_ops: String,

    /// Context window size (tokens). Must match the saved model when loading.
    #[clap(long, default_value_t = 16)]
    pub block_size: usize,

    /// N-gram order (`N`) for `-m ngram`: the model conditions its
    /// prediction on the previous `N - 1` tokens (a composite key into an
    /// `Embedding(vocab_size^(N-1), vocab_size)` table — see
    /// `model::ngram::NgramLM`). `N = 2` is bigram-equivalent (conditions on
    /// just the current/most-recent token, matching `-m bigram` exactly).
    /// Ignored for `-m gpt`/`-m bigram`. When `-m ngram` is used, this value
    /// OVERRIDES `--block-size`: the effective context length used to build/
    /// load the model (and stored in the registry) is `ngram_order - 1`, so
    /// `--block-size` doesn't need to be set separately for ngram runs.
    #[clap(long, default_value_t = 2)]
    pub ngram_order: usize,

    /// Hidden / embedding dimension. Must match the saved model when loading.
    #[clap(long, default_value_t = 16)]
    pub hidden_size: usize,

    /// Number of attention heads. Either a single number, applied uniformly
    /// to every transformer block (e.g. `--num-heads 4`, today's behavior),
    /// or a comma-separated list with exactly `--num-blocks` entries, one
    /// per block, for a non-uniform architecture (e.g. `--num-heads
    /// 1,2,4,8` for a 4-block model: block 0 gets 1 head, block 1 gets 2,
    /// etc). Each entry must individually divide `--hidden-size` evenly.
    /// Must match the saved model when loading (same rule for both forms).
    /// `value_delimiter = ','` makes clap split a single `--num-heads`
    /// occurrence on commas into this `Vec` (a bare `--num-heads 4` yields
    /// the single-element `[4]`); see `model::resolve_heads_schedule` for
    /// how the `[4]` vs `[h0..hN]` cases are told apart and applied.
    #[clap(long, default_value = "4", value_delimiter = ',')]
    pub num_heads: Vec<usize>,

    /// Number of transformer blocks. Must match the saved model when loading.
    #[clap(long, default_value_t = 2)]
    pub num_blocks: usize,

    /// Minibatch size: number of random `block_size`-token windows stacked into
    /// a single forward/backward/optimizer step. In this codebase one epoch is
    /// exactly one optimizer step on `num_batches` random windows, so this is
    /// effectively the minibatch width (bigger = less noisy gradient, more
    /// compute per step). Total training compute scales as
    /// `epochs * num_batches * block_size` token predictions.
    #[clap(long, default_value_t = 64)]
    pub num_batches: usize,
}

pub fn parse_args() -> Args {
    Args::parse()
}

/// `--vocab-size` validator: must parse as a `usize` and be >= 256 (BPE's
/// base vocabulary is the 256 raw bytes, so anything below that floor can't
/// express even the un-merged base vocabulary).
fn parse_vocab_size(s: &str) -> Result<usize, String> {
    let v: usize = s.parse().map_err(|e| format!("`{s}` is not a valid number: {e}"))?;
    if v < 256 {
        return Err(format!("--vocab-size must be >= 256 (BPE's base vocabulary is 256 raw bytes), got {v}"));
    }
    Ok(v)
}
