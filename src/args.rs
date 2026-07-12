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
    #[clap(long, default_value = "1024")]
    pub vocab_size: usize,

    /// The path to the dataset file.
    #[clap(short, long, default_value = "data/tinyshakespeare.txt")]
    pub dataset_path: PathBuf,

    /// The path to the dataset file.
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

    /// Inclusive lower bound for eval operands.
    #[clap(long, default_value_t = 0)]
    pub eval_min: i64,

    /// Inclusive upper bound for eval operands.
    #[clap(long, default_value_t = 999)]
    pub eval_max: i64,

    /// Run RFT (Rejection sampling Fine-Tuning): sample -> filter by correctness -> SFT on winners -> repeat.
    #[clap(long, default_value = "false", group = "mode")]
    pub rft: bool,

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
}

pub fn parse_args() -> Args {
    Args::parse()
}
