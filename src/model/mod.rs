mod bigramlm;
mod gpt;

use crate::{
    args::ModelType,
    dataset::{Dataset, DatasetType},
    error::{SmolError, SmolResult},
    model::{bigramlm::BigramLM, gpt::Gpt},
};
use candle_core::{Device, IndexOp, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, loss, ops::softmax};
use rand::{
    distr::{Distribution, weighted::WeightedIndex},
    Rng,
};

pub enum LanguageModel {
    BigramLM(BigramLM),
    Gpt(Gpt),
}

impl LanguageModel {
    pub fn new_bigram(vocab_size: usize, device: &Device) -> SmolResult<Self> {
        let model = BigramLM::new(vocab_size, device)?;
        Ok(LanguageModel::BigramLM(model))
    }

    pub fn new_gpt(
        block_size: usize,
        vocab_size: usize,
        embed_dims: usize,
        num_heads: usize,
        num_blocks: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        let model = Gpt::new(block_size, vocab_size, embed_dims, num_heads, num_blocks, device)?;
        Ok(LanguageModel::Gpt(model))
    }

    pub fn new(
        model_type: ModelType,
        block_size: usize,
        vocab_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_blocks: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        match model_type {
            ModelType::Gpt => {
                Self::new_gpt(block_size, vocab_size, hidden_size, num_heads, num_blocks, device)
            }
            ModelType::Bigram => Self::new_bigram(vocab_size, device),
        }
    }

    #[allow(dead_code)]
    pub fn train(
        &self,
        dataset: &mut Dataset,
        model_path: &std::path::PathBuf,
        num_epochs: usize,
        num_batches: usize,
    ) -> SmolResult<()> {
        // Backward-compat wrapper: dropout on, early stopping off. Callers that
        // want early stopping (e.g. `--train` via `train.rs`) call
        // `train_with_dropout` directly with CLI-supplied patience/min_delta.
        self.train_with_dropout(dataset, model_path, num_epochs, num_batches, true, 0, 0.0)
    }

    /// Same as `train` but with an explicit `use_dropout` flag and early-stopping
    /// knobs. RFT passes `false` for `use_dropout` so the SFT step is
    /// deterministic across runs with the same `--seed` — candle-nn 0.9.1's
    /// `Dropout` uses `Tensor::rand`, which draws from candle's CPU RNG that
    /// cannot be seeded (see the note in `train.rs`), so dropout-on SFT is
    /// non-reproducible. The `--train` path keeps `use_dropout = true` (regular
    /// SFT with dropout regularization).
    ///
    /// Early stopping: when `patience > 0`, the per-epoch loss is smoothed via
    /// a rolling mean over the last `SMOOTH_WINDOW` epochs; if the smoothed
    /// loss fails to drop by more than `min_delta` for `patience` consecutive
    /// epochs, training halts (with a final save). This catches the loss
    /// plateau we observed on the 1-digit arithmetic run, where ~430 of the
    /// last epochs bought no measurable improvement. `patience == 0` disables
    /// early stopping (run all `num_epochs`).
    pub fn train_with_dropout(
        &self,
        dataset: &mut Dataset,
        model_path: &std::path::PathBuf,
        num_epochs: usize,
        num_batches: usize,
        use_dropout: bool,
        patience: usize,
        min_delta: f32,
    ) -> SmolResult<()> {
        const SMOOTH_WINDOW: usize = 20;
        let block_size = self.get_block_size();
        let mut optimizer = AdamW::new(self.get_var_map().all_vars(), ParamsAdamW::default())?;

        let mut best_smoothed: f32 = f32::MAX;
        let mut epochs_no_improve: usize = 0;
        let mut recent_losses: std::collections::VecDeque<f32> =
            std::collections::VecDeque::with_capacity(SMOOTH_WINDOW);

        for epoch in 0..num_epochs {
            let (stacked_x, stacked_y) =
                dataset.get_random_batches(DatasetType::Training, block_size, num_batches)?;
            // Use the training-mode forward so dropout is active during
            // training. The inference forward (via `&dyn Module`) is a no-op
            // for dropout, which is what we want during `generate`.
            let logits = self.forward_train_with_flag(&stacked_x, use_dropout)?;
            // Batch size -> Number of sequences processed in parallel
            // Time size -> Number of tokens in each sequence (context length)
            // Channel size -> The dimension of each token's representation (here vocab size)
            let (batch_size, time_size, channel_size) = logits.shape().dims3()?;
            let loss = loss::cross_entropy(
                &logits.reshape((batch_size * time_size, channel_size))?,
                &stacked_y.reshape((batch_size * time_size,))?,
            )?;
            // Looks like params.zero_grad()?; is not required in candle because gradients are accumulated externally
            let grads = loss.backward()?;
            optimizer.step(&grads)?;

            let epoch_loss = loss.to_scalar::<f32>()?;
            println!(
                "Epoch {}/{}: Loss = {}",
                epoch + 1,
                num_epochs,
                epoch_loss
            );

            // Save every 10 epochs
            if (epoch + 1) % 10 == 0 {
                self.save(model_path)?;
                println!("Model saved at epoch {}", epoch + 1);
            }

            // Early stopping: compare a rolling-mean loss against the best
            // seen so far. The window damps per-epoch noise (the 1-digit run
            // bounced +-0.03 around its plateau); min_delta prevents
            // sub-threshold drift from resetting the counter.
            if patience > 0 {
                recent_losses.push_back(epoch_loss);
                if recent_losses.len() > SMOOTH_WINDOW {
                    recent_losses.pop_front();
                }
                let smoothed =
                    recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
                if best_smoothed - smoothed > min_delta {
                    best_smoothed = smoothed;
                    epochs_no_improve = 0;
                } else {
                    epochs_no_improve += 1;
                }
                if epochs_no_improve >= patience {
                    println!(
                        "Early stopping at epoch {}/{}: no improvement for {} \
                         epochs (best smoothed loss {:.6}, window={}, min_delta={})",
                        epoch + 1,
                        num_epochs,
                        epochs_no_improve,
                        best_smoothed,
                        SMOOTH_WINDOW,
                        min_delta
                    );
                    break;
                }
            }
        }

        // Final save after training (also runs after an early-stop break).
        self.save(model_path)?;
        println!("Model saved after final epoch");

        Ok(())
    }

    #[cfg(test)]
    /// Returns the embeddings as a 2D vector for testing purposes
    fn get_embeddings_vec2d(&self) -> SmolResult<Vec<Vec<f32>>> {
        let embedding_table = match self {
            LanguageModel::BigramLM(model) => model.token_embedding.clone(),
            LanguageModel::Gpt(model) => model.token_embeddings.clone(),
        };
        let flattened = embedding_table.embeddings().to_vec2::<f32>()?;
        Ok(flattened)
    }

    pub fn generate(
        &self,
        max_new_tokens: usize,
        rng: &mut impl Rng,
        device: &candle_core::Device,
    ) -> SmolResult<Vec<u32>> {
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new_tokens);
        generated_ids.push(0); // Start with a token, e.g., <BOS>
        let model = self.get_model();
        let block_size = self.get_block_size();
        for _ in 1..max_new_tokens {
            // Truncate generated_ids to the model's block size
            let truncated_ids: Vec<u32> = generated_ids
                .iter()
                .skip(0.max(generated_ids.len().saturating_sub(block_size)))
                .cloned()
                .collect();
            let truncated_len = truncated_ids.len();

            let logits = model.forward(&Tensor::from_vec(
                truncated_ids,
                (1, truncated_len),
                device,
            )?)?;
            let most_recent_logits = logits.i((0, truncated_len - 1, ..))?;
            let probabilities = softmax(&most_recent_logits, 0)?;
            let vec = probabilities.to_vec1()?;
            let next_token = sample_multinomial(rng, &vec)?;
            generated_ids.push(next_token);
        }

        Ok(generated_ids)
    }

    /// Greedy (argmax) decoding from an arbitrary prompt. Unlike `generate`,
    /// this starts from the supplied `prompt` instead of a BOS token, uses no
    /// RNG (pure argmax over the vocab dimension at each step), and stops early
    /// when the model emits `stop_token`. Returns only the *newly generated*
    /// tokens (the prompt is excluded from the returned vec).
    ///
    /// The `max_new_tokens` cap prevents runaway generation; callers typically
    /// pass the model's block size, which is a safe upper bound on answer
    /// length. Inputs are truncated to the model's block size on each step so
    /// the GPT position-embedding table is never indexed out of range.
    pub fn generate_greedy_from_prompt(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
        stop_token: u32,
        device: &Device,
    ) -> SmolResult<Vec<u32>> {
        let block_size = self.get_block_size();
        let mut generated: Vec<u32> = prompt.to_vec();
        let mut new_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            // Truncate to the model's block size (GPT position-embedding limit).
            let start = 0.max(generated.len().saturating_sub(block_size));
            let ctx: Vec<u32> = generated[start..].to_vec();
            let ctx_len = ctx.len();

            let input = Tensor::from_vec(ctx, (1, ctx_len), device)?;
            let logits = match self {
                LanguageModel::Gpt(model) => model.forward_with_training(&input, false)?,
                // `get_model` returns `&dyn Module`, so `forward` is callable
                // via the trait object without importing `Module` into scope.
                LanguageModel::BigramLM(_) => self.get_model().forward(&input)?,
            };

            // Argmax over the vocab dimension at the last position.
            let last_logits = logits.i((0, ctx_len - 1, ..))?;
            let next_token_tensor = last_logits.argmax(D::Minus1)?;
            let next_token: u32 = next_token_tensor.to_scalar::<u32>()?;

            generated.push(next_token);
            new_tokens.push(next_token);

            if next_token == stop_token {
                break;
            }
        }

        Ok(new_tokens)
    }

    /// Temperature-sampling completion from an arbitrary prompt. Like
    /// `generate_greedy_from_prompt`, but instead of argmax the next token is
    /// sampled from a temperature-scaled softmax distribution over the
    /// vocabulary. This is the exploration primitive RFT needs: at T > 0 the
    /// model produces diverse completions, some of which may be correct even
    /// when the greedy decoding is wrong.
    ///
    /// `temperature <= 0` falls back to greedy (argmax) so callers don't have
    /// to special-case degenerate temperatures. The running sequence is
    /// truncated to the model's block size on each step (same as the greedy
    /// variant). Returns only the *newly generated* tokens (prompt excluded).
    pub fn sample_from_prompt(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
        stop_token: u32,
        temperature: f32,
        rng: &mut impl Rng,
        device: &Device,
    ) -> SmolResult<Vec<u32>> {
        let block_size = self.get_block_size();
        let mut generated: Vec<u32> = prompt.to_vec();
        let mut new_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            // Truncate to the model's block size (GPT position-embedding limit).
            let start = 0.max(generated.len().saturating_sub(block_size));
            let ctx: Vec<u32> = generated[start..].to_vec();
            let ctx_len = ctx.len();

            let input = Tensor::from_vec(ctx, (1, ctx_len), device)?;
            let logits = match self {
                LanguageModel::Gpt(model) => model.forward_with_training(&input, false)?,
                LanguageModel::BigramLM(_) => self.get_model().forward(&input)?,
            };

            let last_logits = logits.i((0, ctx_len - 1, ..))?;

            // temperature <= 0 -> greedy fallback (argmax), so a degenerate
            // temperature can't crash the sampler.
            let next_token: u32 = if temperature <= 0.0 {
                last_logits.argmax(D::Minus1)?.to_scalar::<u32>()?
            } else {
                // Scale logits by 1/T via `affine` (candle-core 0.9.1 doesn't
                // implement `Tensor / f32`; only `Tensor / f64` is wired up to
                // `affine(1/v, 0)`).
                let scaled = last_logits.affine(1.0 / temperature as f64, 0.0)?;
                let probs = softmax(&scaled, 0)?;
                let probs_vec = probs.to_vec1()?;
                sample_multinomial(rng, &probs_vec)?
            };

            generated.push(next_token);
            new_tokens.push(next_token);

            if next_token == stop_token {
                break;
            }
        }

        Ok(new_tokens)
    }

    pub fn save(&self, path: &std::path::PathBuf) -> SmolResult<()> {
        match self {
            LanguageModel::BigramLM(model) => model.save(path),
            LanguageModel::Gpt(model) => model.save(path),
        }
    }

    /// Training-mode forward with an explicit dropout flag. RFT passes
    /// `false` for deterministic SFT (candle-nn 0.9.1's dropout uses candle's
    /// unseedable CPU RNG); `--train` passes `true` for regular dropout
    /// regularization. `BigramLM` has no dropout so the flag is a no-op there.
    fn forward_train_with_flag(&self, xs: &Tensor, use_dropout: bool) -> SmolResult<Tensor> {
        match self {
            LanguageModel::Gpt(model) => {
                Ok(model.forward_with_training(xs, use_dropout)?)
            }
            LanguageModel::BigramLM(model) => {
                use candle_nn::Module;
                Ok(model.forward(xs)?)
            }
        }
    }

    pub fn load(
        model_type: ModelType,
        path: &std::path::PathBuf,
        block_size: usize,
        vocab_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_blocks: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        match model_type {
            ModelType::Gpt => Self::load_gpt(
                path,
                block_size,
                vocab_size,
                hidden_size,
                num_heads,
                num_blocks,
                device,
            ),
            ModelType::Bigram => Self::load_bigram(path, vocab_size, device),
        }
    }

    pub fn load_bigram(
        path: &std::path::PathBuf,
        vocab_size: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        let model = BigramLM::load(path, vocab_size, device)?;
        Ok(LanguageModel::BigramLM(model))
    }

    pub fn load_gpt(
        path: &std::path::PathBuf,
        block_size: usize,
        vocab_size: usize,
        embed_dims: usize,
        num_heads: usize,
        num_blocks: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        let model = Gpt::load(
            path,
            block_size,
            vocab_size,
            embed_dims,
            num_heads,
            num_blocks,
            device,
        )?;
        Ok(LanguageModel::Gpt(model))
    }

    fn get_model(&self) -> &dyn candle_nn::Module {
        match self {
            LanguageModel::BigramLM(model) => model,
            LanguageModel::Gpt(model) => model,
        }
    }

    fn get_var_map(&self) -> &candle_nn::VarMap {
        match self {
            LanguageModel::BigramLM(model) => &model.var_map,
            LanguageModel::Gpt(model) => &model.var_map,
        }
    }

    fn get_block_size(&self) -> usize {
        match self {
            LanguageModel::BigramLM(model) => model.vocab_size, // Bigram model does have block size concept
            LanguageModel::Gpt(model) => model.block_size,
        }
    }
}

pub fn sample_multinomial(rng: &mut impl Rng, prs: &Vec<f32>) -> SmolResult<u32> {
    let distribution = WeightedIndex::new(prs)
        .map_err(|e| SmolError::custom_error(&format!("Failed to create distribution: {}", e)))?;
    let next_token = distribution.sample(rng) as u32;
    Ok(next_token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use temp_dir::TempDir;

    #[rstest]
    #[case("bigramlm")]
    #[case("gpt")]
    fn test_lm_save_load_preserves_weights(#[case] model_type: &str) {
        let device = Device::Cpu;
        let vocab_size = 100;
        let hidden_size = 64;
        let block_size = 16;
        let num_heads = 4;
        let num_blocks = 2;

        let model = match model_type {
            "bigramlm" => LanguageModel::new_bigram(vocab_size, &device).unwrap(),
            "gpt" => LanguageModel::new_gpt(
                block_size,
                vocab_size,
                hidden_size,
                num_heads,
                num_blocks,
                &device,
            )
            .unwrap(),
            _ => panic!("Unknown model type {model_type}"),
        };

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.pt");
        model.save(&path).unwrap();

        let loaded_model = match model_type {
            "bigramlm" => LanguageModel::load_bigram(&path, vocab_size, &device).unwrap(),
            "gpt" => LanguageModel::load_gpt(
                &path,
                block_size,
                vocab_size,
                hidden_size,
                num_heads,
                num_blocks,
                &device,
            )
            .unwrap(),
            _ => panic!("Unknown model type {model_type}"),
        };

        assert_eq!(model.get_block_size(), loaded_model.get_block_size());

        let original_embeddings = model.get_embeddings_vec2d().unwrap();
        let loaded_embeddings = loaded_model.get_embeddings_vec2d().unwrap();

        assert_eq!(original_embeddings, loaded_embeddings);
    }
}
