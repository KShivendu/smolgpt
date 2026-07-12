mod args;
mod dataset;
mod debugging;
mod eval;
mod error;
mod model;
mod rft;
mod serve;
mod tokenizer;
mod train;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = args::parse_args();

    train::do_training(args)?;

    Ok(())
}
