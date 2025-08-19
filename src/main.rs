mod args;
mod dataset;
mod tokenizer;

use tokenizer::{SimpleTokenizer, Tokenizer};

use candle_core::{Device, IndexOp, Shape, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args::Args {
        dataset_path,
        model_path: _,
    } = args::parse_args();

    let corpus = dataset::load_corpus(&dataset_path, false);
    let tokenizer = SimpleTokenizer::new(&corpus);

    let device = Device::Cpu;

    let encoded_text = tokenizer.encode(&corpus);
    let data = Tensor::from_vec(
        encoded_text.clone(),
        Shape::from(encoded_text.len()),
        &device,
    )?;

    println!(
        "Encoded text tensor shape: {:?}; dtype {:?}",
        data.shape(),
        data.dtype()
    );

    println!("First 10 values: {:?}", data.i(..10));

    Ok(())
}
