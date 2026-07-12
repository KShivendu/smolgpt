use candle_core::{IndexOp, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::path::PathBuf;

use crate::error::{SmolError, SmolResult};

pub fn load_corpus(path: &PathBuf, show_sample: bool) -> SmolResult<String> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        SmolError::dataset_error(&format!("Failed to read dataset file {path:?}: {e}"))
    })?;

    println!("Length of the dataset: {}", text.len());

    if show_sample {
        // Clamp to the corpus length and snap back to the nearest char
        // boundary so this never panics on a short or non-ASCII corpus.
        let end = (0..=text.len().min(1000))
            .rev()
            .find(|&i| text.is_char_boundary(i))
            .unwrap_or(0);
        println!("First 1000 characters of the dataset:");
        println!("{}", &text[..end]);
    }

    Ok(text)
}

/// Scan an arithmetic corpus (`a op b=c` lines, op ∈ {+,-}) and return the
/// inclusive `(min, max)` operand range across every parseable line. Returns
/// `None` if the corpus has no parseable lines (e.g. tinyshakespeare, an empty
/// file, or a corpus with a non-arithmetic shape).
///
/// Each non-empty line is parsed by:
///   1. Splitting on `=` — the LHS is `a op b`.
///   2. Finding the operator: if the LHS starts with `-` (negative `a`), skip
///      that sign and look for the next `+`/`-`; otherwise the first `+`/`-` is
///      the operator.
///   3. Parsing `a` and `b` as `i64` and folding them into the running
///      min/max.
///
/// Used by `registry::ModelRecord::from_training` to derive a model's eval
/// range from the corpus it was trained on (smart mode), so a single-digit
/// arithmetic model doesn't get evaluated on 3-digit operands.
pub fn operand_range(corpus: &str) -> Option<(i64, i64)> {
    let mut min_val: Option<i64> = None;
    let mut max_val: Option<i64> = None;
    for raw_line in corpus.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        // Skip unparseable lines (e.g. blank-ish lines, prose, anything that
        // isn't `a op b=c`) instead of bailing the whole scan — a corpus with
        // a header comment or a stray prose line should still yield the range
        // of its arithmetic lines.
        let Some((a, b, _op)) = parse_operand_pair(line) else {
            continue;
        };
        for v in [a, b] {
            min_val = Some(match min_val {
                Some(m) => m.min(v),
                None => v,
            });
            max_val = Some(match max_val {
                Some(m) => m.max(v),
                None => v,
            });
        }
    }
    Some((min_val?, max_val?))
}

/// Parse one `a op b=c` line into `(a, b, op)`. Returns `None` on any parse
/// failure so `operand_range`/`operators_present` can skip malformed lines
/// instead of bailing the whole scan.
fn parse_operand_pair(line: &str) -> Option<(i64, i64, char)> {
    let lhs = line.split('=').next()?;
    // If LHS starts with '-', the leading char is the sign of `a`, not the
    // operator. Skip it before searching for the operator so we don't mistake
    // the sign for subtraction.
    let start = if lhs.starts_with('-') { 1 } else { 0 };
    let op_offset = lhs[start..].find(|c| c == '+' || c == '-')?;
    let op = lhs[start + op_offset..].chars().next()?;
    let a_str = &lhs[..start + op_offset];
    let b_str = &lhs[start + op_offset + 1..];
    let a: i64 = a_str.parse().ok()?;
    let b: i64 = b_str.parse().ok()?;
    Some((a, b, op))
}

/// Scan an arithmetic corpus and return the distinct operators (`+`/`-`) that
/// actually appear in it, e.g. `"+"` for an addition-only corpus or `"+,-"`
/// for a mixed one. Returns `None` if the corpus has no parseable arithmetic
/// lines (mirrors `operand_range`'s `None` case).
///
/// Used by `--serve`'s eval endpoint so a model trained on an addition-only
/// corpus (whose char-tokenizer charset may not even contain `-`) isn't
/// evaluated on subtraction problems it never saw and can't tokenize.
pub fn operators_present(corpus: &str) -> Option<String> {
    let mut seen_plus = false;
    let mut seen_minus = false;
    for raw_line in corpus.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((_, _, op)) = parse_operand_pair(line) else {
            continue;
        };
        match op {
            '+' => seen_plus = true,
            '-' => seen_minus = true,
            _ => {}
        }
    }
    match (seen_plus, seen_minus) {
        (false, false) => None,
        (true, false) => Some("+".to_string()),
        (false, true) => Some("-".to_string()),
        (true, true) => Some("+,-".to_string()),
    }
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

        // A window starting at index `i` needs tokens `i..=i+block_size` (the
        // `+1` is for the y-shift), so the last valid start index is
        // `total_size - block_size - 1`; the exclusive range below needs
        // `total_size - block_size` to be > 0, i.e. `total_size > block_size`.
        if total_size <= block_size {
            return Err(SmolError::dataset_error(&format!(
                "dataset split has {total_size} tokens, too few for block_size {block_size} (need > block_size)"
            )));
        }

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

        // `y` is `x` shifted by one token, so it needs one token past `x`'s
        // window: the last token read is at index `start_index + block_size`,
        // which must be a valid index (`< total_size`).
        if start_index + block_size >= total_size {
            return Err(SmolError::dataset_error("Batch size exceeds dataset size"));
        }

        let x_range = start_index..start_index + block_size;
        let y_range = start_index + 1..start_index + block_size + 1;

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

    #[test]
    fn test_operand_range_single_digit() {
        // Inline 1-digit arithmetic corpus — min=0, max=9.
        let corpus = "3+4=7\n9+9=18\n0+0=0\n5-2=3\n8+1=9\n";
        assert_eq!(operand_range(corpus), Some((0, 9)));
    }

    #[test]
    fn test_operand_range_three_digit_with_negatives() {
        // Sample lines from data/arithmetic.txt. The corpus format is
        // `a op b = c` where `a` can be negative but `b` is always
        // non-negative (no `+-` or `--` lines). For `-265-985=-1250` the
        // operands are a=-265 and b=985 (not -985 — that's the result `c`'s
        // sign). The full arithmetic.txt range is [-999, 999]; this sample
        // hits [-265, 996].
        let corpus = "-265-985=-1250\n468+756=1224\n-5+618=613\n942+996=1938\n";
        let (min, max) = operand_range(corpus).unwrap();
        assert_eq!(min, -265);
        assert_eq!(max, 996);
    }

    #[test]
    fn test_operand_range_ignores_blank_and_garbage_lines() {
        // Empty lines and non-arithmetic lines are skipped; only the two
        // valid lines contribute (0..9).
        let corpus = "\nhello world\n3+4=7\n  \n9-2=7\n";
        assert_eq!(operand_range(corpus), Some((2, 9)));
    }

    #[test]
    fn test_operand_range_no_parseable_lines() {
        // tinyshakespeare or any non-arithmetic corpus → None.
        assert_eq!(operand_range("To be, or not to be\nThat is the question\n"), None);
        assert_eq!(operand_range(""), None);
    }
}
