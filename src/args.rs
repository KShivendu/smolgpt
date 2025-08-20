use std::path::PathBuf;

use clap::{ArgGroup, Parser};

#[derive(Parser, Debug)]
#[clap(name = "smolgpt", about = "A smol GPT model")]
#[clap(group(ArgGroup::new("mode").multiple(true).required(true)))]
pub struct Args {
    /// The path to the model file.
    #[clap(short, long, default_value = "model.bin")]
    pub model_path: PathBuf,

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
