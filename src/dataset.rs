use candle_core::{IndexOp, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::path::PathBuf;

use crate::error::SmolError;

pub fn load_corpus(path: &PathBuf, show_sample: bool) -> String {
    let text = std::fs::read_to_string(path).expect("Failed to read dataset file");

    println!("Length of the dataset: {}", text.len());

    if show_sample {
        println!("First 1000 characters of the dataset:");
        println!("{}", &text[..1000]);
    }

    text
}

pub struct Dataset {
    pub train_data: Tensor,
    pub train_size: usize,
    pub validation_data: Tensor,
    pub validation_size: usize,
    pub rng: StdRng,
}

#[derive(Clone)]
pub enum DatasetType {
    Training,
    #[expect(dead_code)]
    Validation,
}

impl Dataset {
    #[allow(dead_code)]
    pub fn new(data: Tensor, train_ratio: f64) -> Result<Self, SmolError> {
        Self::with_rng(data, train_ratio, StdRng::from_os_rng())
    }

    /// Build a `Dataset` with a specific RNG (e.g. seeded for reproducibility).
    pub fn with_rng(data: Tensor, train_ratio: f64, rng: StdRng) -> Result<Self, SmolError> {
        let data_size = *data.shape().dims().first().unwrap();

        let train_size = (data_size as f64 * train_ratio) as usize;
        let training_data = data.i(..train_size)?;
        let validation_data = data.i(train_size..)?;

        Ok(Dataset {
            train_data: training_data,
            train_size,
            validation_data,
            validation_size: data_size - train_size,
            rng,
        })
    }

    /// Get a random batches of data from the dataset.
    pub fn get_random_batches(
        &mut self,
        r#type: DatasetType,
        block_size: usize,
        num_batches: usize,
    ) -> Result<(Tensor, Tensor), SmolError> {
        let total_size = match r#type {
            DatasetType::Training => self.train_size,
            DatasetType::Validation => self.validation_size,
        };

        let random_indices: Vec<usize> = (0..num_batches)
            .map(|_| self.rng.random_range(0..total_size - block_size))
            .collect();

        let rows = random_indices
            .iter()
            .map(|&i| self.get_batch(r#type.clone(), i, block_size))
            .collect::<Result<Vec<_>, _>>()?;

        // FIXME: This is too much cloning. We can do this in one shot
        let stacked_x = Tensor::stack(&rows.iter().map(|(x, _)| x.clone()).collect::<Vec<_>>(), 0)?;
        let stacked_y = Tensor::stack(&rows.iter().map(|(_, y)| y.clone()).collect::<Vec<_>>(), 0)?;

        Ok((stacked_x, stacked_y))
    }

    /// Get a batch of data starting from a specific index.
    ///
    /// Returns x & y tensor. Each containing `batch_size` number of blocks, each of size `block_size`.
    pub fn get_batch(
        &self,
        r#type: DatasetType,
        start_index: usize,
        block_size: usize,
    ) -> Result<(Tensor, Tensor), SmolError> {
        let (data, total_size) = match r#type {
            DatasetType::Training => (&self.train_data, self.train_size),
            DatasetType::Validation => (&self.validation_data, self.validation_size),
        };

        if start_index + block_size > total_size {
            return Err(SmolError::dataset_error("Batch size exceeds dataset size"));
        }

        let x_range = start_index..(start_index + block_size).min(total_size - 1);
        let y_range = start_index + 1..(start_index + block_size + 1).min(total_size);

        let x = data.i(x_range)?;
        let y = data.i(y_range)?;

        Ok((x, y))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Shape;

    use super::*;

    #[test]
    fn test_dataset() {
        let device = candle_core::Device::Cpu;
        let encoded_corpus: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let data = Tensor::from_vec(encoded_corpus, Shape::from(10), &device).unwrap();
        let mut dataset = Dataset::new(data, 0.8).unwrap();

        assert_eq!(dataset.train_size, 8);
        assert_eq!(dataset.validation_size, 2);

        let (x_batch, y_batch) = dataset.get_batch(DatasetType::Training, 0, 4).unwrap();
        assert_eq!(x_batch.shape(), &Shape::from(4));
        assert_eq!(y_batch.shape(), &Shape::from(4));

        let (x_batch, y_batch) = dataset.get_random_batches(DatasetType::Training, 4, 2).unwrap();
        assert_eq!(x_batch.shape(), &Shape::from_dims(&[2, 4]));
        assert_eq!(y_batch.shape(), &Shape::from(&[2, 4]));
    }
}
