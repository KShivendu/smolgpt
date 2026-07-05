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
    #[clap(short = 'k', long, value_enum, default_value_t = TokenizerType::Char)]
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

    /// Whether to train the model.
    #[clap(short, long, default_value = "false", group = "mode")]
    pub train: bool,

    /// Whether to generate from the model
    #[clap(short, long, default_value = "false", group = "mode")]
    pub generate: bool,
}

pub fn parse_args() -> Args {
    Args::parse()
}
