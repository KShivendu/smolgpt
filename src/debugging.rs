use crate::{
    dataset::{Dataset, DatasetType},
    error::SmolResult,
};
use candle_core::IndexOp;

#[allow(dead_code)]
/// Debugging and playing around with a dataset.
pub fn debug_dataset(dataset: &mut Dataset) -> SmolResult<()> {
    let block_size = 8_usize;
    let num_batches = 1_usize;

    let first_block_size = dataset.get_batch(DatasetType::Training, 0, block_size)?;
    println!("First block of size {}: {:?}", block_size, first_block_size);

    let (x_batch, y_batch) =
        dataset.get_random_batches(DatasetType::Training, block_size, num_batches)?;
    println!("Random batch: X: {:?}, Y: {:?}", x_batch, y_batch);

    for b in 0..num_batches {
        for t in 0..block_size {
            let x = x_batch.i(b)?.i(0..t + 1)?;
            let y = y_batch.i(b)?.i(t)?;
            println!("When input is {x:?} the target is {y:?}");
        }
    }

    Ok(())
}
