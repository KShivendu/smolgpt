use crate::{
    dataset::{self, Dataset, DatasetType},
    error::SmolError,
    tokenizer::{SimpleTokenizer, Tokenizer},
};
use candle_core::{Device, Shape, Tensor};
use std::path::PathBuf;

pub fn do_training(dataset_path: PathBuf) -> Result<(), SmolError> {
    let corpus = dataset::load_corpus(&dataset_path, false);
    let tokenizer = SimpleTokenizer::new(&corpus);
    let device = Device::Cpu;

    let encoded_corpus = tokenizer.encode(&corpus);
    let encoded_corpus_len = encoded_corpus.len();
    let data = Tensor::from_vec(encoded_corpus, Shape::from(encoded_corpus_len), &device)?;

    println!(
        "Encoded text tensor shape: {:?}; dtype {:?}",
        data.shape(),
        data.dtype()
    );

    let mut dataset = Dataset::new(data, 0.9)?;

    let block_size = 8_usize;
    let num_batches = 1_usize;

    let first_block_size = dataset.get_batch(DatasetType::Training, 0, block_size)?;
    println!("First block of size {}: {:?}", block_size, first_block_size);

    let (x_batch, y_batch) =
        dataset.get_random_batches(DatasetType::Training, block_size, num_batches)?;
    println!("Random batch: X: {:?}, Y: {:?}", x_batch, y_batch);

    Ok(())
}
