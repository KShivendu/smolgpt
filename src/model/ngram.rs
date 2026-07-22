use crate::error::SmolResult;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Init, Module, VarBuilder, VarMap};
use std::path::PathBuf;

/// N-gram language model: a faithful generalization of `BigramLM`.
///
/// `BigramLM` is `Embedding(vocab_size, vocab_size)` indexed by the single
/// immediately-preceding token. `NgramLM` is
/// `Embedding(vocab_size^(n-1), vocab_size)` indexed by a composite key built
/// from the previous `n-1` tokens (base-`vocab_size` positional encoding,
/// oldest token most significant): for a position predicting the NEXT token,
/// the `n-1` conditioning tokens are the current token and its `n-2`
/// predecessors (so `n=2` reduces exactly to `BigramLM`'s "index by the
/// current token" behavior).
///
/// Positions near the start of a sequence that don't have `n-1` real
/// preceding tokens are padded with sentinel token id `0` — the same
/// convention `LanguageModel::generate` already uses for "no prior context"
/// (it seeds generation with `generated_ids.push(0); // <BOS>`).
///
/// This keeps the same "literal frequency/embedding table" design as
/// `BigramLM` (not a concatenated-embeddings feedforward net): the whole
/// model is one embedding table, so training a small vocab/context is
/// tractable (e.g. `vocab_size=13`, `n=5` is `13^4 = 28,561` rows, ~371K
/// entries total).
pub struct NgramLM {
    pub token_embedding: Embedding,
    pub var_map: VarMap,
    pub vocab_size: usize,
    /// N-gram order: predictions condition on the previous `n - 1` tokens.
    /// `n == 2` is bigram-equivalent.
    pub n: usize,
}

impl NgramLM {
    /// Number of conditioning tokens (`n - 1`). This is `NgramLM`'s real,
    /// meaningful block-size/context-length value (unlike `BigramLM`, which
    /// has no such concept).
    pub fn context_len(&self) -> usize {
        self.n - 1
    }

    /// Number of rows in the embedding table: `vocab_size ^ (n - 1)`.
    #[allow(dead_code)]
    pub fn num_keys(&self) -> usize {
        self.vocab_size.pow(self.context_len() as u32)
    }

    /// `n` must be >= 2 (an n-gram needs at least one conditioning token;
    /// `n == 1` would be a unigram with no context at all, which isn't a
    /// useful "language model" and isn't what this type is for).
    pub fn new(vocab_size: usize, n: usize, device: &Device) -> SmolResult<Self> {
        assert!(n >= 2, "NgramLM requires n >= 2 (got n = {n})");
        let context_len = n - 1;
        let num_keys = vocab_size.pow(context_len as u32);
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let embeddings = var_builder.get_with_hints(
            (num_keys, vocab_size),
            "embeddings",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        let token_embedding = Embedding::new(embeddings, vocab_size);

        Ok(NgramLM {
            token_embedding,
            var_map,
            vocab_size,
            n,
        })
    }

    pub fn save(&self, path: &PathBuf) -> SmolResult<()> {
        self.var_map.save(path)?;
        Ok(())
    }

    /// See `Gpt::save_quantized`'s doc / `crate::quantize`'s module doc.
    pub fn save_quantized(&self, path: &PathBuf) -> SmolResult<()> {
        crate::quantize::save_var_map_quantized(&self.var_map, path)
    }

    pub fn load(path: &PathBuf, vocab_size: usize, n: usize, device: &Device) -> SmolResult<Self> {
        assert!(n >= 2, "NgramLM requires n >= 2 (got n = {n})");
        let context_len = n - 1;
        let num_keys = vocab_size.pow(context_len as u32);
        let mut var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let embeddings = var_builder.get_with_hints(
            (num_keys, vocab_size),
            "embeddings",
            Init::Const(0.0),
        )?;
        crate::quantize::load_into_var_map(&mut var_map, path, device)?;

        let token_embedding = Embedding::new(embeddings, vocab_size);

        Ok(NgramLM {
            token_embedding,
            var_map,
            vocab_size,
            n,
        })
    }

    /// Build the composite lookup keys for every position in `xs` (shape
    /// `(batch, time)`, raw token ids). For position `i` in a row, the key
    /// encodes tokens `row[i-(context_len-1)] ..= row[i]` (oldest token most
    /// significant, `row[i]` itself least significant — so `n=2`'s single
    /// conditioning token is just `row[i]`, matching `BigramLM`). Positions
    /// before the start of the row (no real token there) are padded with
    /// sentinel id `0`.
    fn compute_keys(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let ids: Vec<Vec<u32>> = xs.to_vec2()?;
        let context_len = self.context_len();
        let vocab = self.vocab_size as u64;
        let (batch, time) = xs.shape().dims2()?;
        let mut keys: Vec<u32> = Vec::with_capacity(batch * time);
        for row in &ids {
            let len = row.len();
            for i in 0..len {
                let mut key: u64 = 0;
                for j in 0..context_len {
                    // j=0 -> oldest conditioning token (most significant digit);
                    // j=context_len-1 -> row[i] itself (least significant digit).
                    let offset = (context_len - 1 - j) as i64;
                    let pos = i as i64 - offset;
                    let tok = if pos >= 0 { row[pos as usize] as u64 } else { 0u64 };
                    key = key * vocab + tok;
                }
                keys.push(key as u32);
            }
        }
        Tensor::from_vec(keys, (batch, time), xs.device())
    }
}

impl Module for NgramLM {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let keys = self.compute_keys(xs)?;
        self.token_embedding.forward(&keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;
    use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW};
    use temp_dir::TempDir;

    #[test]
    fn test_ngram_new_shapes() {
        let device = Device::Cpu;
        let vocab_size = 13;
        for n in [2usize, 3, 4, 5] {
            let model = NgramLM::new(vocab_size, n, &device).unwrap();
            assert_eq!(model.context_len(), n - 1);
            assert_eq!(model.num_keys(), vocab_size.pow((n - 1) as u32));
        }
    }

    #[test]
    fn test_ngram_forward_shape() {
        let device = Device::Cpu;
        let vocab_size = 13;
        let n = 4;
        let model = NgramLM::new(vocab_size, n, &device).unwrap();
        let batch = 2;
        let time = 6;
        let ids: Vec<u32> = (0..(batch * time) as u32).map(|v| v % vocab_size as u32).collect();
        let input = Tensor::from_vec(ids, (batch, time), &device).unwrap();
        let logits = model.forward(&input).unwrap();
        assert_eq!(logits.shape().dims3().unwrap(), (batch, time, vocab_size));
    }

    /// `n=2` must reduce to `BigramLM`'s exact indexing convention: the
    /// lookup key at position `i` is just `row[i]` (no history beyond the
    /// current token). Verified by checking `compute_keys` directly against
    /// a hand-built row.
    #[test]
    fn test_ngram_order_2_matches_bigram_indexing() {
        let device = Device::Cpu;
        let vocab_size = 13;
        let model = NgramLM::new(vocab_size, 2, &device).unwrap();
        let row = vec![3u32, 7, 1, 9];
        let input = Tensor::from_vec(row.clone(), (1, row.len()), &device).unwrap();
        let keys = model.compute_keys(&input).unwrap();
        let keys_vec: Vec<u32> = keys.to_vec2::<u32>().unwrap().into_iter().next().unwrap();
        assert_eq!(keys_vec, row, "n=2 keys must equal the raw token ids (BigramLM's own indexing)");
    }

    /// For `n=3` (context_len=2), the key at position `i` should be
    /// `row[i-1] * vocab_size + row[i]` (with a 0 pad for i=0's missing
    /// predecessor).
    #[test]
    fn test_ngram_order_3_composite_key() {
        let device = Device::Cpu;
        let vocab_size = 13;
        let model = NgramLM::new(vocab_size, 3, &device).unwrap();
        let row = vec![3u32, 7, 1, 9];
        let input = Tensor::from_vec(row.clone(), (1, row.len()), &device).unwrap();
        let keys = model.compute_keys(&input).unwrap();
        let keys_vec: Vec<u32> = keys.to_vec2::<u32>().unwrap().into_iter().next().unwrap();
        let expected = vec![
            0 * vocab_size as u32 + 3, // i=0: no predecessor -> pad 0
            3 * vocab_size as u32 + 7, // i=1: (row[0], row[1])
            7 * vocab_size as u32 + 1, // i=2: (row[1], row[2])
            1 * vocab_size as u32 + 9, // i=3: (row[2], row[3])
        ];
        assert_eq!(keys_vec, expected);
    }

    #[test]
    fn test_ngram_save_load_round_trip() {
        let device = Device::Cpu;
        let vocab_size = 13;
        let n = 4;
        let model = NgramLM::new(vocab_size, n, &device).unwrap();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("ngram.bin");
        model.save(&path).unwrap();

        let loaded = NgramLM::load(&path, vocab_size, n, &device).unwrap();
        assert_eq!(loaded.vocab_size, model.vocab_size);
        assert_eq!(loaded.n, model.n);

        let original = model.token_embedding.embeddings().to_vec2::<f32>().unwrap();
        let reloaded = loaded.token_embedding.embeddings().to_vec2::<f32>().unwrap();
        assert_eq!(original, reloaded);
    }

    /// Directly-verifiable correctness check: train a tiny NgramLM on a
    /// deterministic synthetic sequence where the next token is a
    /// deterministic function of the previous `n-1` tokens, and confirm the
    /// model's argmax prediction matches the known-correct next token after
    /// training converges.
    ///
    /// Sequence: repeated cycle `[1, 2, 3, 1, 2, 3, ...]` with n=3 (context
    /// of 2). After token pair (1,2) the correct next token is always 3;
    /// after (2,3) it's always 1; after (3,1) it's always 2. This is exactly
    /// the kind of context BigramLM (n=2) cannot learn (predicting from a
    /// single token is ambiguous here — e.g. after `1` alone, next could be
    /// `2` immediately following the *first* `1`, but the point of this test
    /// is n=3 CAN learn the deterministic 2-token-context rule).
    #[test]
    fn test_ngram_learns_deterministic_next_token() {
        let device = Device::Cpu;
        let vocab_size = 4; // tokens 0..3; 0 unused/padding sentinel
        let n = 3;
        let model = NgramLM::new(vocab_size, n, &device).unwrap();

        // Build many repeats of the cycle so the composite-key rows get
        // plenty of gradient signal.
        let cycle = [1u32, 2, 3];
        let repeats = 200;
        let mut seq: Vec<u32> = Vec::with_capacity(repeats * cycle.len());
        for _ in 0..repeats {
            seq.extend_from_slice(&cycle);
        }
        // x = seq[..-1], y = seq[1..] (standard next-token shift).
        let x: Vec<u32> = seq[..seq.len() - 1].to_vec();
        let y: Vec<u32> = seq[1..].to_vec();
        let len = x.len();
        let x_t = Tensor::from_vec(x, (1, len), &device).unwrap();
        let y_t = Tensor::from_vec(y, (len,), &device).unwrap();

        let mut params = ParamsAdamW::default();
        params.lr = 0.1;
        let mut optimizer = AdamW::new(model.var_map.all_vars(), params).unwrap();

        for _ in 0..300 {
            let logits = model.forward(&x_t).unwrap();
            let (b, t, c) = logits.shape().dims3().unwrap();
            let loss = loss::cross_entropy(
                &logits.reshape((b * t, c)).unwrap(),
                &y_t.reshape((b * t,)).unwrap(),
            )
            .unwrap();
            let grads = loss.backward().unwrap();
            optimizer.step(&grads).unwrap();
        }

        // After training, predicting from context (1,2) should argmax to 3;
        // (2,3) -> 1; (3,1) -> 2.
        let cases: [(Vec<u32>, u32); 3] = [
            (vec![1, 2], 3),
            (vec![2, 3], 1),
            (vec![3, 1], 2),
        ];
        for (ctx, expected_next) in cases {
            let input = Tensor::from_vec(ctx.clone(), (1, ctx.len()), &device).unwrap();
            let logits = model.forward(&input).unwrap();
            let last = logits.i((0, ctx.len() - 1, ..)).unwrap();
            let pred: u32 = last.argmax(candle_core::D::Minus1).unwrap().to_scalar().unwrap();
            assert_eq!(
                pred, expected_next,
                "context {:?} should predict {expected_next}, got {pred}",
                ctx
            );
        }
    }
}
