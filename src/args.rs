use clap::{ArgGroup, Parser};
use std::path::PathBuf;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum, Debug)]
pub enum ModelType {
    Gpt,
    Bigram,
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

    /// Start a local web UI to browse models, datasets, and run evals.
    #[clap(long, default_value = "false", group = "mode")]
    pub serve: bool,

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

    /// Hidden / embedding dimension. Must match the saved model when loading.
    #[clap(long, default_value_t = 16)]
    pub hidden_size: usize,

    /// Number of attention heads. Must match the saved model when loading.
    #[clap(long, default_value_t = 4)]
    pub num_heads: usize,

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
