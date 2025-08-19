mod args;
mod dataset;
mod error;
mod model;
mod tokenizer;
mod train;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args::Args {
        dataset_path,
        model_path: _,
    } = args::parse_args();

    train::do_training(dataset_path)?;

    Ok(())
}
