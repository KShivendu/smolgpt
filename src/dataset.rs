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
    /// EXPERIMENTAL (loss-masking test): per-char f32 mask aligned 1:1 with
    /// the *char-tokenized* corpus (only valid when `TokenizerType::Char` is
    /// used — a BPE token doesn't correspond to one corpus char, so this mask
    /// would be misaligned there). `1.0` marks a position whose target token
    /// is part of an arithmetic line's answer digits or its trailing newline;
    /// `0.0` marks everything else (operands, operator, `=`). `None` when no
    /// mask was computed (the default / non-arithmetic-corpus / BPE case).
    /// Rough, single-purpose scaffolding for testing the loss-masking
    /// hypothesis — not meant as a general masking mechanism.
    pub train_mask: Option<Tensor>,
    pub validation_mask: Option<Tensor>,
    /// EXPERIMENTAL (Hypothesis B / `--aligned-windows`): fact-boundary
    /// offsets (see `compute_fact_boundaries`) that fall within the TRAIN
    /// split (`offset < train_size`). `None` unless `--aligned-windows` was
    /// requested (default off — existing training behavior/models are
    /// completely unaffected). When `Some`, `get_random_batches`/
    /// `get_random_batches_masked` sample window starts ONLY from this list
    /// (instead of uniformly over `0..total_size - block_size`) so every
    /// training window starts exactly at a true `"a op b="` fact boundary,
    /// matching what GRPO/RFT sampling and `--eval` always present at
    /// inference time.
    pub train_aligned_starts: Option<Vec<usize>>,
}

/// Scan an arithmetic corpus (`a op b=c` lines) and return a per-char f32
/// mask the same length as `corpus.chars().count()`: `1.0` for a char that is
/// part of the answer `c` or the line's trailing `\n`, `0.0` otherwise
/// (operands, operator, `=`). Lines with no `=` are left all-zero. Only
/// meaningful when paired with a char tokenizer (1 char == 1 token), which is
/// what this experiment uses.
pub fn compute_answer_mask(corpus: &str) -> Vec<f32> {
    let total_chars = corpus.chars().count();
    let mut mask = vec![0.0f32; total_chars];
    let mut char_offset = 0usize;
    for raw_line in corpus.split_inclusive('\n') {
        let has_newline = raw_line.ends_with('\n');
        let line = if has_newline { &raw_line[..raw_line.len() - 1] } else { raw_line };
        let line_char_len = line.chars().count();
        if let Some(eq_byte_pos) = line.find('=') {
            let eq_char_idx = line[..eq_byte_pos].chars().count();
            for i in (eq_char_idx + 1)..line_char_len {
                mask[char_offset + i] = 1.0;
            }
            if has_newline {
                let nl_idx = char_offset + line_char_len;
                if nl_idx < mask.len() {
                    mask[nl_idx] = 1.0;
                }
            }
        }
        char_offset += line_char_len + if has_newline { 1 } else { 0 };
    }
    mask
}

/// EXPERIMENTAL (Hypothesis B / `--aligned-windows`): scan a char-tokenized
/// arithmetic corpus and return every char-offset that is a true "fact
/// boundary" — offset `0` (the very start of the corpus), plus the offset of
/// the char immediately following each `\n`. These are the only offsets at
/// which a training window can start such that "position 0" of that window
/// really is the start of an `"a op b="` fact, rather than some arbitrary
/// mid-fact character.
///
/// Only meaningful when paired with a char tokenizer (1 char == 1 token),
/// same assumption `compute_answer_mask` makes. The last entry (the offset
/// right after the corpus's final `\n`, if the corpus ends in one) may equal
/// the corpus length and is filtered out by callers that need a valid window
/// start (e.g. `Dataset::get_random_batches`'s `total_size - block_size`
/// bound already excludes it).
pub fn compute_fact_boundaries(corpus: &str) -> Vec<usize> {
    let mut boundaries = vec![0usize];
    let mut idx = 0usize;
    for ch in corpus.chars() {
        idx += 1;
        if ch == '\n' {
            boundaries.push(idx);
        }
    }
    boundaries
}

/// Sample `count` window-start offsets from `0..total_size - block_size`
/// (exclusive upper bound): if `aligned_pool` is `Some` and non-empty, starts
/// are drawn only from that pool (fact-boundary-aligned sampling); otherwise
/// falls back to uniform sampling over the whole valid range. Shared by
/// `Dataset::sample_start_indices` (actual training) and
/// `sample_example_windows` (UI-facing example reconstruction for `--serve`'s
/// Samples tab) so both paths use IDENTICAL sampling semantics — the UI never
/// reimplements this logic ad hoc.
pub fn sample_window_starts(
    rng: &mut StdRng,
    total_size: usize,
    block_size: usize,
    aligned_pool: Option<&[usize]>,
    count: usize,
) -> Vec<usize> {
    match aligned_pool {
        Some(pool) if !pool.is_empty() => {
            (0..count).map(|_| pool[rng.random_range(0..pool.len())]).collect()
        }
        _ => (0..count).map(|_| rng.random_range(0..total_size - block_size)).collect(),
    }
}

/// Fixed seed for `sample_example_windows`'s RNG, so the "Samples" tab shows
/// stable example windows across page reloads/requests instead of a fresh
/// random set every time (there's no training-fidelity reason to reseed on
/// every call — this is a read-only UI reconstruction, not a training step).
const EXAMPLE_WINDOWS_SEED: u64 = 0x53_4d_4f_4c_47_50_54; // "SMOLGPT" in hex-ish, arbitrary constant

/// Reconstruct `count` representative SFT training windows as raw text, given
/// a corpus, `block_size`, and whether `--aligned-windows` was used, for
/// `--serve`'s Samples tab. Reuses `sample_window_starts` (the exact function
/// `Dataset::get_random_batches` calls during real training) so what's shown
/// is genuinely faithful to what training does, rather than a separately
/// reimplemented approximation. Only meaningful for a char tokenizer (1 char
/// == 1 token, same assumption `compute_answer_mask`/`compute_fact_boundaries`
/// make) — this operates directly on corpus chars, not encoded tokens, since
/// the UI just needs representative raw text, not exact token boundaries.
///
/// Returns an empty `Vec` if the corpus has too few chars for a `block_size`
/// window (mirrors `get_random_batches`'s "too few tokens" guard, but returns
/// empty here instead of erroring — this is a best-effort UI helper).
pub fn sample_example_windows(
    corpus: &str,
    block_size: usize,
    aligned: bool,
    count: usize,
) -> Vec<String> {
    let chars: Vec<char> = corpus.chars().collect();
    let total_size = chars.len();
    if total_size <= block_size || count == 0 {
        return Vec::new();
    }

    let mut rng = StdRng::seed_from_u64(EXAMPLE_WINDOWS_SEED);
    let aligned_pool: Option<Vec<usize>> = if aligned {
        let boundaries = compute_fact_boundaries(corpus);
        Some(
            boundaries
                .into_iter()
                .filter(|&i| i < total_size - block_size)
                .collect(),
        )
    } else {
        None
    };

    let starts = sample_window_starts(&mut rng, total_size, block_size, aligned_pool.as_deref(), count);
    starts
        .into_iter()
        .map(|s| chars[s..s + block_size].iter().collect())
        .collect()
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
        Self::with_rng_and_mask(data, None, train_ratio, rng)
    }

    /// Same as `with_rng`, plus an optional full-corpus `answer_mask` (e.g.
    /// from `compute_answer_mask`) that gets split train/validation the same
    /// way as `data`. EXPERIMENTAL: scaffolding for the loss-masking test —
    /// see `train_mask`'s doc comment. Delegates to `with_rng_and_mask_aligned`
    /// with no aligned-window starts (regular uniform sampling).
    pub fn with_rng_and_mask(
        data: Tensor,
        answer_mask: Option<Vec<f32>>,
        train_ratio: f64,
        rng: StdRng,
    ) -> Result<Self, SmolError> {
        Self::with_rng_and_mask_aligned(data, answer_mask, None, train_ratio, rng)
    }

    /// Same as `with_rng_and_mask`, plus an optional full-corpus
    /// `fact_boundaries` (e.g. from `compute_fact_boundaries`) — see
    /// `train_aligned_starts`'s doc comment. Only the entries that fall
    /// within the train split (`offset < train_size`) are kept; the rest
    /// belong to the validation split and aren't needed since only training
    /// sampling is aligned by `--aligned-windows`.
    pub fn with_rng_and_mask_aligned(
        data: Tensor,
        answer_mask: Option<Vec<f32>>,
        fact_boundaries: Option<Vec<usize>>,
        train_ratio: f64,
        rng: StdRng,
    ) -> Result<Self, SmolError> {
        let data_size = *data.shape().dims().first().unwrap();

        let train_size = (data_size as f64 * train_ratio) as usize;
        let training_data = data.i(..train_size)?;
        let validation_data = data.i(train_size..)?;

        let (train_mask, validation_mask) = match answer_mask {
            Some(m) => {
                if m.len() != data_size {
                    return Err(SmolError::dataset_error(&format!(
                        "answer_mask length {} != data length {data_size}",
                        m.len()
                    )));
                }
                let device = data.device();
                let full = Tensor::from_vec(m, data_size, device)?;
                (Some(full.i(..train_size)?), Some(full.i(train_size..)?))
            }
            None => (None, None),
        };

        let train_aligned_starts = fact_boundaries.map(|boundaries| {
            boundaries
                .into_iter()
                .filter(|&i| i < train_size)
                .collect::<Vec<usize>>()
        });

        Ok(Dataset {
            train_data: training_data,
            train_size,
            validation_data,
            validation_size: data_size - train_size,
            rng,
            train_mask,
            validation_mask,
            train_aligned_starts,
        })
    }

    /// Sample `num_batches` window-start indices for `r#type`/`block_size`.
    ///
    /// EXPERIMENTAL (Hypothesis B / `--aligned-windows`): when this is a
    /// `DatasetType::Training` sample AND `train_aligned_starts` is `Some`
    /// (non-empty after filtering to starts that leave room for a full
    /// `block_size` window), starts are sampled ONLY from that list — i.e.
    /// every window begins exactly at a true `"a op b="` fact boundary. In
    /// every other case (validation split, or no aligned starts configured,
    /// or the filtered list is empty), falls back to the original uniform
    /// `0..total_size - block_size` sampling so existing behavior is
    /// unchanged whenever `--aligned-windows` isn't in play.
    fn sample_start_indices(
        &mut self,
        r#type: &DatasetType,
        total_size: usize,
        block_size: usize,
        num_batches: usize,
    ) -> Result<Vec<usize>, SmolError> {
        let aligned_pool: Option<Vec<usize>> = match r#type {
            DatasetType::Training => self.train_aligned_starts.as_ref().map(|starts| {
                starts
                    .iter()
                    .copied()
                    .filter(|&i| i < total_size - block_size)
                    .collect::<Vec<usize>>()
            }),
            DatasetType::Validation => None,
        };

        Ok(sample_window_starts(
            &mut self.rng,
            total_size,
            block_size,
            aligned_pool.as_deref(),
            num_batches,
        ))
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

        let random_indices: Vec<usize> =
            self.sample_start_indices(&r#type, total_size, block_size, num_batches)?;

        let rows = random_indices
            .iter()
            .map(|&i| self.get_batch(r#type.clone(), i, block_size))
            .collect::<Result<Vec<_>, _>>()?;

        // FIXME: This is too much cloning. We can do this in one shot
        let stacked_x = Tensor::stack(&rows.iter().map(|(x, _)| x.clone()).collect::<Vec<_>>(), 0)?;
        let stacked_y = Tensor::stack(&rows.iter().map(|(_, y)| y.clone()).collect::<Vec<_>>(), 0)?;

        Ok((stacked_x, stacked_y))
    }

    /// Same as `get_random_batches`, plus a stacked f32 mask tensor (shape
    /// matching `y`) built from `train_mask`/`validation_mask`. If no mask was
    /// set on this `Dataset` (`with_rng` / no `answer_mask` given), the
    /// returned mask is all-1.0 (i.e. equivalent to no masking) so callers
    /// can use this method unconditionally without special-casing.
    /// EXPERIMENTAL: see `train_mask`'s doc comment.
    pub fn get_random_batches_masked(
        &mut self,
        r#type: DatasetType,
        block_size: usize,
        num_batches: usize,
    ) -> Result<(Tensor, Tensor, Tensor), SmolError> {
        let total_size = match r#type {
            DatasetType::Training => self.train_size,
            DatasetType::Validation => self.validation_size,
        };

        if total_size <= block_size {
            return Err(SmolError::dataset_error(&format!(
                "dataset split has {total_size} tokens, too few for block_size {block_size} (need > block_size)"
            )));
        }

        let random_indices: Vec<usize> =
            self.sample_start_indices(&r#type, total_size, block_size, num_batches)?;

        let rows = random_indices
            .iter()
            .map(|&i| self.get_batch_masked(r#type.clone(), i, block_size))
            .collect::<Result<Vec<_>, _>>()?;

        let stacked_x = Tensor::stack(&rows.iter().map(|(x, _, _)| x.clone()).collect::<Vec<_>>(), 0)?;
        let stacked_y = Tensor::stack(&rows.iter().map(|(_, y, _)| y.clone()).collect::<Vec<_>>(), 0)?;
        let stacked_mask = Tensor::stack(&rows.iter().map(|(_, _, m)| m.clone()).collect::<Vec<_>>(), 0)?;

        Ok((stacked_x, stacked_y, stacked_mask))
    }

    /// Same as `get_batch`, plus the mask slice for `y`'s range (all-1.0 if
    /// this `Dataset` has no mask set). EXPERIMENTAL: see `train_mask`'s doc
    /// comment.
    pub fn get_batch_masked(
        &self,
        r#type: DatasetType,
        start_index: usize,
        block_size: usize,
    ) -> Result<(Tensor, Tensor, Tensor), SmolError> {
        let (data, total_size, mask_data) = match r#type {
            DatasetType::Training => (&self.train_data, self.train_size, &self.train_mask),
            DatasetType::Validation => (&self.validation_data, self.validation_size, &self.validation_mask),
        };

        if start_index + block_size >= total_size {
            return Err(SmolError::dataset_error("Batch size exceeds dataset size"));
        }

        let x_range = start_index..start_index + block_size;
        let y_range = start_index + 1..start_index + block_size + 1;

        let x = data.i(x_range)?;
        let y = data.i(y_range.clone())?;
        let mask = match mask_data {
            Some(m) => m.i(y_range)?,
            None => Tensor::ones(block_size, candle_core::DType::F32, data.device())?,
        };

        Ok((x, y, mask))
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
    fn test_compute_fact_boundaries() {
        // "3+4=7\n9+9=18\n0+0=0\n" -> boundaries at 0 (start), and right after
        // each '\n'. Line lengths (in chars): "3+4=7\n" = 6, "9+9=18\n" = 7,
        // "0+0=0\n" = 6. So boundaries: 0, 6, 13, 19.
        let corpus = "3+4=7\n9+9=18\n0+0=0\n";
        assert_eq!(compute_fact_boundaries(corpus), vec![0, 6, 13, 19]);
    }

    #[test]
    fn test_aligned_windows_sampling_only_starts_at_fact_boundaries() {
        // EXPERIMENTAL (Hypothesis B / --aligned-windows) verification: when
        // `train_aligned_starts` is configured, every start index sampled by
        // `get_random_batches`/`get_random_batches_masked` must be a true
        // fact boundary (0 or right after a '\n'), never a mid-fact offset —
        // this is the whole point of the flag, so assert it directly rather
        // than trusting the implementation.
        let device = candle_core::Device::Cpu;
        // Repeat a tiny 2-fact corpus many times so there's plenty of room
        // for block_size=4 windows within the train split.
        let corpus = "1+1=2\n3+3=6\n".repeat(20);
        let boundaries = compute_fact_boundaries(&corpus);
        let encoded: Vec<u32> = corpus.bytes().map(|b| b as u32).collect();
        let len = encoded.len();
        let data = Tensor::from_vec(encoded, Shape::from(len), &device).unwrap();
        let rng = StdRng::seed_from_u64(42);

        let mut dataset =
            Dataset::with_rng_and_mask_aligned(data, None, Some(boundaries.clone()), 0.9, rng)
                .unwrap();
        assert!(
            dataset.train_aligned_starts.is_some(),
            "aligned starts should be populated when fact_boundaries is Some"
        );

        let block_size = 4;
        let (x_batch, _y_batch) = dataset
            .get_random_batches(DatasetType::Training, block_size, 200)
            .unwrap();
        // Recover the actual sampled start offsets by checking every row's
        // first token against the corpus bytes at each valid boundary offset
        // (a window starting at a fact boundary must have `corpus.bytes()[i]`
        // equal to the row's first byte, and — decisively — must be a member
        // of `boundaries`). We assert more directly: every possible non-
        // boundary start (i.e. every index NOT in boundaries, within the
        // valid range) would produce a DIFFERENT first byte pattern in
        // general, but the cleanest direct check is to inspect the training
        // split for which offsets are legal and confirm none outside that
        // set were used, via a manual re-sampling probe using the same RNG
        // seed on a fresh unaligned dataset for contrast below.
        let train_len = dataset.train_size;
        let valid_boundaries: Vec<usize> = boundaries
            .iter()
            .copied()
            .filter(|&i| i < train_len - block_size)
            .collect();
        assert!(!valid_boundaries.is_empty());

        // Cross-check: every row's tokens must exactly match the corpus
        // bytes starting at SOME valid boundary offset (not just any offset).
        let x_rows = x_batch.to_vec2::<u32>().unwrap();
        for row in &x_rows {
            let matches_some_boundary = valid_boundaries.iter().any(|&start| {
                let expected: Vec<u32> =
                    corpus.bytes().skip(start).take(block_size).map(|b| b as u32).collect();
                *row == expected
            });
            assert!(
                matches_some_boundary,
                "sampled window {row:?} does not start at any fact-boundary offset {valid_boundaries:?}"
            );
        }
    }

    #[test]
    fn test_sample_example_windows_unaligned_mixes_mid_fact_starts() {
        // Unaligned (default) sampling should, over enough draws, include at
        // least one window whose position-0 char is NOT a fact-boundary char
        // (i.e. not the char immediately following a '\n', nor corpus start)
        // -- this is the exact "position 0 lands mid-fact" behavior
        // `--aligned-windows` exists to fix.
        let corpus = "1+1=2\n3+3=6\n5+5=10\n".repeat(20);
        let boundaries = compute_fact_boundaries(&corpus);
        let chars: Vec<char> = corpus.chars().collect();
        let boundary_chars: std::collections::HashSet<char> =
            boundaries.iter().filter(|&&i| i < chars.len()).map(|&i| chars[i]).collect();
        let windows = sample_example_windows(&corpus, 4, false, 50);
        assert_eq!(windows.len(), 50);
        let any_mid_fact = windows.iter().any(|w| {
            let first = w.chars().next().unwrap();
            !boundary_chars.contains(&first)
        });
        assert!(any_mid_fact, "expected at least one mid-fact-start window under unaligned sampling");
    }

    #[test]
    fn test_sample_example_windows_aligned_always_starts_at_fact_boundary() {
        // Aligned sampling must NEVER start a window mid-fact: every window's
        // text must exactly match the corpus starting at some fact-boundary
        // offset.
        let corpus = "1+1=2\n3+3=6\n5+5=10\n".repeat(20);
        let block_size = 4;
        let boundaries = compute_fact_boundaries(&corpus);
        let total = corpus.chars().count();
        let valid_boundaries: Vec<usize> =
            boundaries.into_iter().filter(|&i| i < total - block_size).collect();
        assert!(!valid_boundaries.is_empty());

        let chars: Vec<char> = corpus.chars().collect();
        let windows = sample_example_windows(&corpus, block_size, true, 50);
        assert_eq!(windows.len(), 50);
        for w in &windows {
            let matches_some_boundary = valid_boundaries.iter().any(|&start| {
                let expected: String = chars[start..start + block_size].iter().collect();
                *w == expected
            });
            assert!(matches_some_boundary, "aligned window {w:?} did not start at a fact boundary");
        }
    }

    #[test]
    fn test_sample_example_windows_empty_for_short_corpus() {
        // Corpus shorter than block_size => no windows possible, return empty
        // rather than panicking or erroring (best-effort UI helper).
        assert_eq!(sample_example_windows("ab", 4, false, 5), Vec::<String>::new());
        assert_eq!(sample_example_windows("", 4, true, 5), Vec::<String>::new());
    }

    #[test]
    fn test_sample_example_windows_is_deterministic_across_calls() {
        // Fixed seed means repeated calls (e.g. across page reloads in
        // --serve) return identical output, not a fresh random set each time.
        let corpus = "1+1=2\n3+3=6\n5+5=10\n".repeat(20);
        let a = sample_example_windows(&corpus, 4, true, 8);
        let b = sample_example_windows(&corpus, 4, true, 8);
        assert_eq!(a, b);
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
