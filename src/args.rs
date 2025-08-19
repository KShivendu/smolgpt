use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[clap(name = "smolgpt", about = "A smol GPT model")]
pub struct Args {
    /// The path to the model file.
    #[clap(short, long, default_value = "model.bin")]
    pub model_path: PathBuf,

    /// The path to the dataset file.
    #[clap(short, long, default_value = "data/tinyshakespeare.txt")]
    pub dataset_path: PathBuf,
}

pub fn parse_args() -> Args {
    Args::parse()
}
