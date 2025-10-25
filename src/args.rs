use clap::{ArgGroup, Parser};
use std::path::PathBuf;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum, Debug)]
pub enum ModelType {
    Gpt,
    Bigram,
}

#[derive(Parser, Debug)]
#[clap(name = "smolgpt", about = "A smol GPT model")]
#[clap(group(ArgGroup::new("mode").multiple(true).required(true)))]
pub struct Args {
    /// The path to the model file.
    #[clap(short = 'p', long, default_value = "model.bin")]
    pub model_path: PathBuf,

    #[clap(short = 'm', value_enum, default_value_t = ModelType::Gpt)]
    pub model_type: ModelType,

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
