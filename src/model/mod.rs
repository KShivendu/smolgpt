mod bigramlm;
mod gpt;
mod ngram;

use crate::{
    args::ModelType,
    dataset::{Dataset, DatasetType},
    error::{SmolError, SmolResult},
    model::{bigramlm::BigramLM, gpt::Gpt, ngram::NgramLM},
};
use candle_core::{Device, IndexOp, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, loss, ops::softmax};
use rand::{
    distr::{Distribution, weighted::WeightedIndex},
    Rng,
};

/// Result of a training run, returned by `train_with_dropout` so callers
/// (e.g. `train.rs` auto-registration) can record the actual number of epochs
/// run + whether early stopping fired in the model registry note. `final_loss`
/// is the last per-epoch cross-entropy loss seen (0.0 if no epochs ran);
/// `losses` is the full per-epoch loss series (empty if no epochs ran).
/// `final_loss` is kept in sync with `losses.last()` — i.e. it equals the last
/// entry of `losses` when the series is non-empty, and 0.0 when it is.
#[derive(Debug, Clone)]
pub struct TrainOutcome {
    pub epochs_run: usize,
    pub early_stopped: bool,
    #[allow(dead_code)]
    pub final_loss: f32,
    /// Per-epoch cross-entropy loss series, in epoch order (index 0 = epoch 1).
    /// Empty when `num_epochs == 0` or the loop never entered. Stored as `f32`
    /// (the dtype `loss.to_scalar::<f32>()` yields); the JSON blob in the
    /// `trainings` table keeps full precision — UI sparkline downsamples in JS.
    pub losses: Vec<f32>,
}

impl TrainOutcome {
    /// A "no training happened" outcome: 0 epochs, nothing early-stopped, no
    /// loss recorded. Used to register a model row (e.g. at an `--eval`-only
    /// or `--rft`/`--grpo` start) before any SFT epoch has actually run.
    pub fn placeholder() -> TrainOutcome {
        TrainOutcome {
            epochs_run: 0,
            early_stopped: false,
            final_loss: 0.0,
            losses: Vec::new(),
        }
    }
}

pub enum LanguageModel {
    BigramLM(BigramLM),
    Gpt(Gpt),
    Ngram(NgramLM),
}

impl LanguageModel {
    pub fn new_bigram(vocab_size: usize, device: &Device) -> SmolResult<Self> {
        let model = BigramLM::new(vocab_size, device)?;
        Ok(LanguageModel::BigramLM(model))
    }

    /// `context_len` is `N - 1` (the number of preceding tokens the model
    /// conditions on). Callers pass this via the (possibly overridden, see
    /// `train.rs`) `block_size` value, since for `NgramLM` `block_size` IS
    /// the real, meaningful context length (unlike `BigramLM`, which has no
    /// such concept — see `get_block_size`'s doc).
    pub fn new_ngram(vocab_size: usize, context_len: usize, device: &Device) -> SmolResult<Self> {
        let n = context_len + 1;
        let model = NgramLM::new(vocab_size, n, device)?;
        Ok(LanguageModel::Ngram(model))
    }

    /// `num_heads` is the raw `--num-heads` value: either a single entry
    /// (broadcast uniformly to every block, today's behavior) or exactly
    /// `num_blocks` entries (one per block, for a non-uniform architecture).
    /// See `resolve_heads_schedule` for the exact resolution rule/errors.
    /// `init_std`/`tie_embeddings` are EXPERIMENTAL knobs (see `Gpt::new`'s
    /// doc): controlling the fresh-init weight stdev and whether `lm_head`
    /// reuses `token_embeddings`, respectively. Both default to the
    /// unchanged-behavior sentinel at every existing call site (`1.0`,
    /// `false`). `init_gain` (EXPERIMENTAL, `--init-gain`) is a further knob,
    /// only meaningful when `init_std` is at its sentinel: `None` (the
    /// default) leaves candle's own Kaiming-Normal gain (√2) untouched;
    /// `Some(gain)` substitutes `gain` for it. See `Gpt::new`/`build_linear`'s
    /// docs for the full precedence rule between `init_std` and `init_gain`.
    pub fn new_gpt(
        block_size: usize,
        vocab_size: usize,
        embed_dims: usize,
        num_heads: &[usize],
        num_blocks: usize,
        init_std: f32,
        init_gain: Option<f64>,
        tie_embeddings: bool,
        device: &Device,
    ) -> SmolResult<Self> {
        let heads_schedule = resolve_heads_schedule(num_heads, num_blocks)?;
        let model = Gpt::new(
            block_size,
            vocab_size,
            embed_dims,
            &heads_schedule,
            num_blocks,
            init_std,
            init_gain,
            tie_embeddings,
            device,
        )?;
        Ok(LanguageModel::Gpt(model))
    }

    /// `init_std`/`init_gain`/`tie_embeddings`: see `new_gpt`'s doc. Ignored
    /// by `Bigram`/`Ngram` (neither has an init-scale or embedding-tying
    /// concept).
    pub fn new(
        model_type: ModelType,
        block_size: usize,
        vocab_size: usize,
        hidden_size: usize,
        num_heads: &[usize],
        num_blocks: usize,
        init_std: f32,
        init_gain: Option<f64>,
        tie_embeddings: bool,
        device: &Device,
    ) -> SmolResult<Self> {
        match model_type {
            ModelType::Gpt => Self::new_gpt(
                block_size,
                vocab_size,
                hidden_size,
                num_heads,
                num_blocks,
                init_std,
                init_gain,
                tie_embeddings,
                device,
            ),
            ModelType::Bigram => Self::new_bigram(vocab_size, device),
            ModelType::Ngram => Self::new_ngram(vocab_size, block_size, device),
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
        // The outcome (epochs run, early-stop flag) is discarded here. No
        // on_checkpoint callback — the wrapper is used by tests that don't
        // need live progress upserts.
        self.train_with_dropout(
            dataset,
            model_path,
            num_epochs,
            num_batches,
            true,
            0,
            0.0,
            None,
            0.001,
            false,
            None,
        )?;
        Ok(())
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
    ///
    /// `on_checkpoint`: when `Some(cb)`, the callback is invoked at each
    /// checkpoint save (every 10 epochs + the final save) with a `TrainOutcome`
    /// snapshot of the current `{epochs_run, early_stopped, final_loss, losses}`
    /// state. `train.rs`'s `--train` branch fills this with a closure that
    /// upserts the partial loss trajectory into the `trainings` table so the
    /// web UI shows live training progress (reload the page mid-training to
    /// see the latest checkpoint). RFT's internal SFT sub-loop passes `None`
    /// — its per-round upserts happen at the round granularity via the
    /// `on_round` callback on `run_rft`/`run_grpo`, not per-SFT-checkpoint.
    ///
    /// `on_best_loss`: when `Some(cb)`, fired whenever the smoothed loss (the
    /// same rolling-mean-over-`SMOOTH_WINDOW`-epochs value used for early
    /// stopping) hits a NEW best-so-far value, subject to the throttle in
    /// `should_snapshot` (see its doc for the exact policy/numbers). Called
    /// as `cb(epoch_number_1_indexed, smoothed_loss, self)` — `self` is
    /// passed through so the caller can run an exhaustive eval grid against
    /// the model's CURRENT (mid-training) weights without this module
    /// needing to know about tokenizers/eval-grids/the registry at all.
    /// `train.rs`'s `--train` branch uses this to snapshot the exhaustive
    /// eval grid into the `checkpoint_grids` table over the course of
    /// training, so a follow-up UI can animate "how did the grid change as
    /// training progressed". RFT's internal SFT sub-loop and the plain
    /// `train` wrapper both pass `None`.
    pub fn train_with_dropout(
        &self,
        dataset: &mut Dataset,
        model_path: &std::path::PathBuf,
        num_epochs: usize,
        num_batches: usize,
        use_dropout: bool,
        patience: usize,
        min_delta: f32,
        mut on_checkpoint: Option<&mut dyn FnMut(&TrainOutcome)>,
        // EXPERIMENTAL knobs added to test two convergence hypotheses on the
        // 1-digit-arithmetic SFT plateau: (1) a fixed learning rate other
        // than candle's AdamW default of 1e-3, and (2) computing the loss
        // only over "answer" token positions (see `dataset::compute_answer_mask`)
        // instead of uniformly over every token in the window. Both default
        // to old behavior at existing call sites (`lr=0.001`, `mask_loss=false`).
        lr: f64,
        mask_loss: bool,
        mut on_best_loss: Option<&mut dyn FnMut(usize, f32, &LanguageModel)>,
    ) -> SmolResult<TrainOutcome> {
        const SMOOTH_WINDOW: usize = 20;
        let block_size = self.get_block_size();
        let mut adamw_params = ParamsAdamW::default();
        adamw_params.lr = lr;
        let mut optimizer = AdamW::new(self.get_var_map().all_vars(), adamw_params)?;

        let mut best_smoothed: f32 = f32::MAX;
        let mut epochs_no_improve: usize = 0;
        let mut recent_losses: std::collections::VecDeque<f32> =
            std::collections::VecDeque::with_capacity(SMOOTH_WINDOW);

        // Checkpoint-grid snapshot throttle state (see `should_snapshot`'s
        // doc for the policy). Tracked independently of `best_smoothed`
        // above (which is early-stopping's own "best" and is only updated
        // when `patience > 0`) so the snapshot trigger works regardless of
        // whether early stopping is enabled.
        let mut best_snapshot_loss: f32 = f32::MAX;
        let mut last_snapshot_epoch: Option<usize> = None;
        let mut last_snapshot_loss: f32 = f32::MAX;

        // Tracked so `train.rs` can record them in the model registry note.
        let mut epochs_run: usize = 0;
        let mut early_stopped: bool = false;
        let mut final_loss: f32 = 0.0;
        // Per-epoch cross-entropy loss series, surfaced to the UI sparkline via
        // the `trainings` table. Pre-allocated to `num_epochs` since the loop
        // may break early via `patience` (in which case `losses.len() ==
        // epochs_run < num_epochs`).
        let mut losses: Vec<f32> = Vec::with_capacity(num_epochs);

        for epoch in 0..num_epochs {
            // EXPERIMENTAL branch: when `mask_loss` is set, pull the parallel
            // per-token answer mask alongside x/y (see
            // `dataset::compute_answer_mask` / `get_random_batches_masked`);
            // otherwise use the original unmasked path unchanged.
            let (stacked_x, stacked_y, stacked_mask) = if mask_loss {
                let (x, y, m) = dataset.get_random_batches_masked(
                    DatasetType::Training,
                    block_size,
                    num_batches,
                )?;
                (x, y, Some(m))
            } else {
                let (x, y) =
                    dataset.get_random_batches(DatasetType::Training, block_size, num_batches)?;
                (x, y, None)
            };
            // Use the training-mode forward so dropout is active during
            // training. The inference forward (via `&dyn Module`) is a no-op
            // for dropout, which is what we want during `generate`.
            let logits = self.forward_train_with_flag(&stacked_x, use_dropout)?;
            // Batch size -> Number of sequences processed in parallel
            // Time size -> Number of tokens in each sequence (context length)
            // Channel size -> The dimension of each token's representation (here vocab size)
            let (batch_size, time_size, channel_size) = logits.shape().dims3()?;
            let loss = match stacked_mask {
                None => loss::cross_entropy(
                    &logits.reshape((batch_size * time_size, channel_size))?,
                    &stacked_y.reshape((batch_size * time_size,))?,
                )?,
                Some(mask) => {
                    // Masked cross-entropy, computed by hand since
                    // candle_nn::loss::cross_entropy always averages over
                    // every position. Only "answer" positions (mask == 1.0)
                    // contribute to the loss/gradient.
                    let flat_logits =
                        logits.reshape((batch_size * time_size, channel_size))?;
                    let flat_targets = stacked_y.reshape((batch_size * time_size,))?;
                    let flat_mask = mask.reshape((batch_size * time_size,))?;
                    let log_probs = candle_nn::ops::log_softmax(&flat_logits, D::Minus1)?;
                    // Per-row negative log-likelihood of the true next token:
                    // gather the log-prob at the target index, then negate.
                    let picked = log_probs
                        .gather(&flat_targets.unsqueeze(1)?, 1)?
                        .squeeze(1)?;
                    let nll = picked.neg()?;
                    let masked_nll = (nll.clone() * &flat_mask)?;
                    let mask_sum = flat_mask.sum_all()?;
                    // Guard against an all-zero mask (e.g. a window with no
                    // answer position in it) so we don't divide by zero;
                    // falls back to the unmasked mean loss for that batch.
                    let mask_sum_scalar = mask_sum.to_scalar::<f32>()?;
                    if mask_sum_scalar > 0.0 {
                        (masked_nll.sum_all()? / mask_sum)?
                    } else {
                        nll.mean_all()?
                    }
                }
            };
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

            final_loss = epoch_loss;
            losses.push(epoch_loss);
            epochs_run = epoch + 1;

            // Save every 10 epochs and fire the on_checkpoint callback so the
            // web UI's `trainings` row updates in place with the partial loss
            // trajectory (live progress: reload the page mid-training to see
            // the latest checkpoint).
            if (epoch + 1) % 10 == 0 {
                self.save(model_path)?;
                println!("Model saved at epoch {}", epoch + 1);
                if let Some(cb) = on_checkpoint.as_mut() {
                    cb(&TrainOutcome {
                        epochs_run,
                        early_stopped: false, // still running
                        final_loss: epoch_loss,
                        losses: losses.clone(),
                    });
                }
            }

            // Rolling-mean smoothing over the last SMOOTH_WINDOW epochs.
            // Computed unconditionally (not gated on `patience > 0`) because
            // both early stopping AND the checkpoint-grid snapshot trigger
            // below need it.
            recent_losses.push_back(epoch_loss);
            if recent_losses.len() > SMOOTH_WINDOW {
                recent_losses.pop_front();
            }
            let smoothed = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;

            // Loss-improvement-triggered checkpoint-grid snapshot: fire
            // `on_best_loss` whenever the smoothed loss hits a NEW best-ever
            // value AND the throttle in `should_snapshot` allows it. Runs
            // regardless of `patience` (early stopping may be disabled
            // entirely, but snapshotting should still work).
            if let Some(cb) = on_best_loss.as_mut() {
                if smoothed < best_snapshot_loss {
                    best_snapshot_loss = smoothed;
                    if should_snapshot(epoch + 1, smoothed, last_snapshot_epoch, last_snapshot_loss)
                    {
                        cb(epoch + 1, smoothed, self);
                        last_snapshot_epoch = Some(epoch + 1);
                        last_snapshot_loss = smoothed;
                    }
                }
            }

            // Early stopping: compare a rolling-mean loss against the best
            // seen so far. The window damps per-epoch noise (the 1-digit run
            // bounced +-0.03 around its plateau); min_delta prevents
            // sub-threshold drift from resetting the counter.
            if patience > 0 {
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
                    early_stopped = true;
                    break;
                }
            }
        }

        // Final save after training (also runs after an early-stop break).
        self.save(model_path)?;
        println!("Model saved after final epoch");
        // Fire the final on_checkpoint with the complete outcome so the UI's
        // trainings row reflects the final epochs_run / early_stopped / loss
        // series. This is the authoritative final upsert; earlier per-checkpoint
        // upserts are superseded by this one.
        if let Some(cb) = on_checkpoint.as_mut() {
            cb(&TrainOutcome {
                epochs_run,
                early_stopped,
                final_loss,
                losses: losses.clone(),
            });
        }

        Ok(TrainOutcome {
            epochs_run,
            early_stopped,
            final_loss,
            losses,
        })
    }

    #[cfg(test)]
    /// Returns the embeddings as a 2D vector for testing purposes
    fn get_embeddings_vec2d(&self) -> SmolResult<Vec<Vec<f32>>> {
        let embedding_table = match self {
            LanguageModel::BigramLM(model) => model.token_embedding.clone(),
            LanguageModel::Gpt(model) => model.token_embeddings.clone(),
            LanguageModel::Ngram(model) => model.token_embedding.clone(),
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
                LanguageModel::BigramLM(_) | LanguageModel::Ngram(_) => {
                    self.get_model().forward(&input)?
                }
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
                LanguageModel::BigramLM(_) | LanguageModel::Ngram(_) => {
                    self.get_model().forward(&input)?
                }
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
            LanguageModel::Ngram(model) => model.save(path),
        }
    }

    /// Post-training INT8 quantization for storage (see `crate::quantize`'s
    /// module doc for the scheme). Writes a quantized copy of `self`'s
    /// weights to `path`; `self` is left unmodified. `--quantize` (see
    /// `train.rs`) is the CLI entry point that calls this on an already
    /// loaded base model to produce a `-quant.bin` variant.
    pub fn save_quantized(&self, path: &std::path::PathBuf) -> SmolResult<()> {
        match self {
            LanguageModel::BigramLM(model) => model.save_quantized(path),
            LanguageModel::Gpt(model) => model.save_quantized(path),
            LanguageModel::Ngram(model) => model.save_quantized(path),
        }
    }

    /// Build an AdamW optimizer over all of the model's trainable variables,
    /// with a caller-specified learning rate. GRPO needs to own the optimizer
    /// across many small policy-gradient steps (one per prompt), so it can't
    /// reuse `train_with_dropout`'s internal optimizer.
    pub fn make_optimizer(&self, lr: f64) -> SmolResult<AdamW> {
        let mut params = ParamsAdamW::default();
        params.lr = lr;
        Ok(AdamW::new(self.get_var_map().all_vars(), params)?)
    }

    /// GRPO-lite policy-gradient step for ONE prompt's group of `G` sampled
    /// completions.
    ///
    /// `completions[i]` is the i-th sampled continuation of `prompt` (token ids,
    /// prompt excluded). `rewards[i]` is its scalar reward (1.0 for a correct
    /// arithmetic answer, 0.0 otherwise). Computes group-relative advantages
    /// `a_i = (r_i - mean_r) / (std_r + eps)`, then the policy-gradient loss
    /// `L = -mean_i(a_i * logp(completion_i | prompt))`, backprops it, and takes
    /// an optimizer step. Returns the scalar loss for logging.
    ///
    /// This is "GRPO-lite": group-relative advantages (the part of GRPO that
    /// removes the need for a learned value function) but NO PPO-style ratio
    /// clipping and NO KL penalty to a reference model. The gradient pushes UP
    /// the logprob of above-average (correct) completions and DOWN the logprob
    /// of below-average (wrong) ones — so wrong answers carry corrective signal
    /// that RFT discards. A group with uniform reward (all correct or all
    /// wrong) yields zero advantage everywhere → no update (nothing to learn
    /// from a tied group), and `grpo_step` returns 0.0 without stepping.
    pub fn grpo_step(
        &self,
        prompt: &[u32],
        completions: &[Vec<u32>],
        rewards: &[f32],
        optimizer: &mut AdamW,
        device: &Device,
    ) -> SmolResult<f32> {
        let block_size = self.get_block_size();
        let g = completions.len();
        if g == 0 || rewards.len() != g {
            return Ok(0.0);
        }

        // Group-relative advantage. std==0 (uniform reward) → all-zero
        // advantages → no gradient signal → we skip the step entirely.
        let eps = 1e-6f32;
        let mean_r = rewards.iter().sum::<f32>() / g as f32;
        let var = rewards.iter().map(|r| (r - mean_r).powi(2)).sum::<f32>() / g as f32;
        let std_r = var.sqrt();
        if std_r < eps {
            return Ok(0.0);
        }
        let advantages: Vec<f64> = rewards
            .iter()
            .map(|r| ((r - mean_r) / (std_r + eps)) as f64)
            .collect();

        // Accumulate per-completion policy-gradient loss terms. Each term is
        // `-a_i * logp(completion_i | prompt)`. logp is computed by forwarding
        // `[prompt, completion]` and taking log-softmax of the logits that
        // predict the completion tokens. cross_entropy(logits, targets) returns
        // `-(1/L) * logp`, so `loss_i = a_i * L * ce` (the negation cancels).
        //
        // We reshape the logits slice to a rank-2 `(L, V)` tensor (cross_entropy
        // requires rank 2). `.squeeze(0)` can't be used for that because a
        // single-token completion (L=1) would over-squeeze to rank 1.
        let mut terms: Vec<Tensor> = Vec::with_capacity(g);
        for (comp, &adv) in completions.iter().zip(advantages.iter()) {
            if comp.is_empty() || adv.abs() < 1e-12 {
                continue;
            }
            let l = comp.len();
            // full = prompt ++ completion, truncated to the last block_size
            // tokens (GPT position-embedding limit). The completion lives at
            // the tail, so truncation only ever drops prompt prefix.
            let mut full: Vec<u32> = Vec::with_capacity(prompt.len() + l);
            full.extend_from_slice(prompt);
            full.extend_from_slice(comp);
            let start = 0.max(full.len().saturating_sub(block_size));
            let ctx: Vec<u32> = full[start..].to_vec();
            let ctx_len = ctx.len();
            if ctx_len <= l {
                // No prompt context survived truncation → logp given the prompt
                // is undefined. Skip.
                continue;
            }
            // Completion occupies the last `l` positions of `ctx`; token t_j is
            // predicted by the logit at position (ctx_len - l - 1 + j).
            let pred_start = ctx_len - l - 1;
            let pred_end = ctx_len - 1;
            let input = Tensor::from_vec(ctx, (1, ctx_len), device)?;
            let logits = self.forward_train_with_flag(&input, false)?;
            let v_dim = logits.shape().dims3()?.2;
            let pred_logits = logits
                .i((0, pred_start..pred_end, ..))?
                .reshape((l, v_dim))?;
            let targets = Tensor::from_vec(comp.to_vec(), (l,), device)?;
            let ce = loss::cross_entropy(&pred_logits, &targets)?; // scalar
            // loss_i = -adv * logp = -adv * (-l * ce) = adv * l * ce
            let loss_i = ce.affine(adv * l as f64, 0.0)?;
            terms.push(loss_i);
        }

        if terms.is_empty() {
            return Ok(0.0);
        }
        let stacked = Tensor::stack(&terms, 0)?;
        let loss = stacked.mean_all()?;
        let grads = loss.backward()?;
        optimizer.step(&grads)?;
        Ok(loss.to_scalar::<f32>()?)
    }

    /// Scalar sum of log-probs of `completion` tokens given `prompt`, under
    /// this model, as a plain `f64` — no gradient graph is consumed (the
    /// caller never calls `backward()` on this value). Used by GRPO-full to
    /// cache `old_logp` (under the sampling policy) and the frozen
    /// reference's `ref_logp` ONCE per group, before the K mini-epochs.
    ///
    /// Reuses the exact logp logic from `grpo_step`: forward
    /// `[prompt ++ completion]` truncated to the last `block_size` tokens,
    /// slice the logits predicting the completion, `cross_entropy` →
    /// `ce = -(1/L)·logp` → `logp = -L·ce`. Returns `0.0` if `completion` is
    /// empty or no prompt context survives truncation (the same guard as
    /// `grpo_step`, so callers can distinguish "skipped" from a real 0.0
    /// the same way they do for the lite step).
    pub fn completion_logp_scalar(
        &self,
        prompt: &[u32],
        completion: &[u32],
        device: &Device,
    ) -> SmolResult<f64> {
        let block_size = self.get_block_size();
        let l = completion.len();
        if l == 0 {
            return Ok(0.0);
        }
        let mut full: Vec<u32> = Vec::with_capacity(prompt.len() + l);
        full.extend_from_slice(prompt);
        full.extend_from_slice(completion);
        let start = 0.max(full.len().saturating_sub(block_size));
        let ctx: Vec<u32> = full[start..].to_vec();
        let ctx_len = ctx.len();
        if ctx_len <= l {
            // No prompt context survived truncation → logp given the prompt
            // is undefined. Match grpo_step's skip guard.
            return Ok(0.0);
        }
        let pred_start = ctx_len - l - 1;
        let pred_end = ctx_len - 1;
        let input = Tensor::from_vec(ctx, (1, ctx_len), device)?;
        let logits = self.forward_train_with_flag(&input, false)?;
        let v_dim = logits.shape().dims3()?.2;
        let pred_logits = logits
            .i((0, pred_start..pred_end, ..))?
            .reshape((l, v_dim))?;
        let targets = Tensor::from_vec(completion.to_vec(), (l,), device)?;
        let ce = loss::cross_entropy(&pred_logits, &targets)?; // scalar: -(1/L)·logp
        let ce_f32 = ce.to_scalar::<f32>()?;
        // logp = -L * ce
        Ok(-(l as f64) * (ce_f32 as f64))
    }

    /// Save `self` to `path`, then load it back as a NEW independent
    /// `LanguageModel` with a fresh `VarMap`. This is the frozen reference
    /// policy for GRPO-full: it shares NO state with `self`'s `VarMap`, so
    /// the optimizer never touches it, and `ref_logp` values computed under
    /// it stay constant across the K mini-epochs.
    ///
    /// The arch is recovered via the `get_*` getters so the caller doesn't
    /// have to thread the constructor params back through. The temp file is
    /// the caller's responsibility to clean up (GRPO-full uses a `Drop`
    /// guard in `run_grpo`).
    pub fn snapshot(
        &self,
        path: &std::path::PathBuf,
        device: &Device,
    ) -> SmolResult<LanguageModel> {
        self.save(path)?;
        // Use the FULL per-block schedule (not `get_num_heads`'s collapsed
        // representative value) so a non-uniform architecture reloads with
        // the exact same per-block shapes, not a uniform approximation.
        let heads_schedule = self.get_heads_schedule();
        LanguageModel::load(
            self.get_model_type(),
            path,
            self.get_block_size(),
            self.get_vocab_size(),
            self.get_hidden_size(),
            &heads_schedule,
            self.get_num_blocks(),
            self.get_tie_embeddings(),
            device,
        )
    }

    /// GRPO-full (PPO-style) policy step for ONE prompt's group of `G`
    /// sampled completions, over `k_epochs` mini-epochs.
    ///
    /// `advantages[i]` is the group-relative advantage of completion `i`
    /// (computed once from the rewards at sampling time — the caller does
    /// this, since the rewards/rewards-std are known before the K epochs).
    /// `old_logps[i]` is `logp(completion_i | prompt)` under the SAMPLING
    /// policy (cached once before the K epochs via
    /// `completion_logp_scalar`). `ref_logps[i]` is the same under the
    /// FROZEN reference policy (cached once, never updated). Only
    /// `logp_theta` (under the live, updated policy) is recomputed each
    /// mini-epoch.
    ///
    /// Per mini-epoch, per completion (with non-zero advantage, non-empty
    /// completion, surviving prompt context):
    ///   ratio   = exp(logp_theta - old_logp)
    ///   clipped = clamp(ratio, 1-eps, 1+eps)
    ///   surr1   = ratio * adv
    ///   surr2   = clipped * adv
    ///   pol_loss = -min(surr1, surr2)
    ///   kl       = logp_theta - ref_logp
    ///   kl_loss  = beta * kl
    ///   term     = pol_loss + kl_loss
    /// Stack terms over completions, `.mean_all()`, `.backward()`,
    /// `optimizer.step()`. Returns the mean loss of the LAST mini-epoch as
    /// `f32`. Returns `0.0` (no step) if `completions` is empty or every
    /// advantage is ~0 (uniform-reward group — nothing to learn from a
    /// tied group, same guard as `grpo_step`).
    pub fn grpo_step_full(
        &self,
        prompt: &[u32],
        completions: &[Vec<u32>],
        advantages: &[f64],
        old_logps: &[f64],
        ref_logps: &[f64],
        clip_eps: f64,
        kl_beta: f64,
        k_epochs: usize,
        optimizer: &mut AdamW,
        device: &Device,
    ) -> SmolResult<f32> {
        let block_size = self.get_block_size();
        let g = completions.len();
        if g == 0
            || advantages.len() != g
            || old_logps.len() != g
            || ref_logps.len() != g
            || k_epochs == 0
        {
            return Ok(0.0);
        }
        // Uniform-reward guard: every advantage ~0 → no signal, skip the
        // whole group (matches grpo_step's std≈0 skip).
        let all_zero_adv = advantages.iter().all(|&a| a.abs() < 1e-12);
        if all_zero_adv {
            return Ok(0.0);
        }

        let lo = 1.0 - clip_eps;
        let hi = 1.0 + clip_eps;
        let mut last_loss: f32 = 0.0;

        for _ in 0..k_epochs {
            let mut terms: Vec<Tensor> = Vec::with_capacity(g);
            for (i, comp) in completions.iter().enumerate() {
                let adv_i = advantages[i];
                if comp.is_empty() || adv_i.abs() < 1e-12 {
                    continue;
                }
                let l = comp.len();
                let mut full: Vec<u32> = Vec::with_capacity(prompt.len() + l);
                full.extend_from_slice(prompt);
                full.extend_from_slice(comp);
                let start = 0.max(full.len().saturating_sub(block_size));
                let ctx: Vec<u32> = full[start..].to_vec();
                let ctx_len = ctx.len();
                if ctx_len <= l {
                    // No prompt context survived truncation → skip this
                    // completion (can't define logp given the prompt).
                    continue;
                }
                let pred_start = ctx_len - l - 1;
                let pred_end = ctx_len - 1;
                let input = Tensor::from_vec(ctx, (1, ctx_len), device)?;
                let logits = self.forward_train_with_flag(&input, false)?;
                let v_dim = logits.shape().dims3()?.2;
                let pred_logits = logits
                    .i((0, pred_start..pred_end, ..))?
                    .reshape((l, v_dim))?;
                let targets = Tensor::from_vec(comp.to_vec(), (l,), device)?;
                let ce = loss::cross_entropy(&pred_logits, &targets)?; // -(1/L)·logp
                // logp_theta = -L * ce (differentiable scalar tensor).
                let logp_theta = ce.affine(-(l as f64), 0.0)?;
                // ratio = exp(logp_theta - old_logp)
                let ratio = logp_theta.affine(1.0, -old_logps[i])?.exp()?;
                // candle-core 0.9.1's `Tensor::clamp(min, max)` accepts f64
                // scalars directly via the `TensorOrScalar` trait, so no
                // need to build scalar lo/hi tensors manually.
                let clipped = ratio.clamp(lo, hi)?;
                // surr1 = ratio * adv, surr2 = clipped * adv
                let surr1 = ratio.affine(adv_i, 0.0)?;
                let surr2 = clipped.affine(adv_i, 0.0)?;
                // pol_loss = -min(surr1, surr2)
                let pol_loss = surr1.minimum(&surr2)?.affine(-1.0, 0.0)?;
                // kl = logp_theta - ref_logp ; kl_loss = beta * kl
                let kl = logp_theta.affine(1.0, -ref_logps[i])?;
                let kl_loss = kl.affine(kl_beta, 0.0)?;
                // term = pol_loss + kl_loss (both scalar tensors; broadcast_add
                // handles the rank-0/rank-1 mix cross_entropy may produce).
                let term = pol_loss.broadcast_add(&kl_loss)?;
                terms.push(term);
            }

            if terms.is_empty() {
                // No completion was eligible this mini-epoch (all skipped via
                // the empty-completion / no-context guard). No step. The
                // last_loss stays at whatever the previous epoch produced
                // (0.0 if this is the first epoch).
                continue;
            }
            let stacked = Tensor::stack(&terms, 0)?;
            let loss = stacked.mean_all()?;
            let grads = loss.backward()?;
            optimizer.step(&grads)?;
            last_loss = loss.to_scalar::<f32>()?;
        }

        Ok(last_loss)
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
            LanguageModel::Ngram(model) => {
                use candle_nn::Module;
                Ok(model.forward(xs)?)
            }
        }
    }

    /// `num_heads` is the raw `--num-heads` value (see `new_gpt`'s doc for
    /// the uniform-vs-per-block resolution rule). `tie_embeddings` MUST match
    /// what the saved model was actually trained/constructed with (see
    /// `Gpt::load`'s doc) — ignored for `Bigram`/`Ngram`.
    pub fn load(
        model_type: ModelType,
        path: &std::path::PathBuf,
        block_size: usize,
        vocab_size: usize,
        hidden_size: usize,
        num_heads: &[usize],
        num_blocks: usize,
        tie_embeddings: bool,
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
                tie_embeddings,
                device,
            ),
            ModelType::Bigram => Self::load_bigram(path, vocab_size, device),
            ModelType::Ngram => Self::load_ngram(path, block_size, vocab_size, device),
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

    /// `context_len` is `N - 1`; see `new_ngram`'s doc for why it arrives via
    /// the (possibly `train.rs`-overridden) `block_size` parameter.
    pub fn load_ngram(
        path: &std::path::PathBuf,
        context_len: usize,
        vocab_size: usize,
        device: &Device,
    ) -> SmolResult<Self> {
        let n = context_len + 1;
        let model = NgramLM::load(path, vocab_size, n, device)?;
        Ok(LanguageModel::Ngram(model))
    }

    pub fn load_gpt(
        path: &std::path::PathBuf,
        block_size: usize,
        vocab_size: usize,
        embed_dims: usize,
        num_heads: &[usize],
        num_blocks: usize,
        tie_embeddings: bool,
        device: &Device,
    ) -> SmolResult<Self> {
        let heads_schedule = resolve_heads_schedule(num_heads, num_blocks)?;
        let model = Gpt::load(
            path,
            block_size,
            vocab_size,
            embed_dims,
            &heads_schedule,
            num_blocks,
            tie_embeddings,
            device,
        )?;
        Ok(LanguageModel::Gpt(model))
    }

    fn get_model(&self) -> &dyn candle_nn::Module {
        match self {
            LanguageModel::BigramLM(model) => model,
            LanguageModel::Gpt(model) => model,
            LanguageModel::Ngram(model) => model,
        }
    }

    fn get_var_map(&self) -> &candle_nn::VarMap {
        match self {
            LanguageModel::BigramLM(model) => &model.var_map,
            LanguageModel::Gpt(model) => &model.var_map,
            LanguageModel::Ngram(model) => &model.var_map,
        }
    }

    fn get_block_size(&self) -> usize {
        match self {
            // BigramLM has no block-size/context-length concept (each
            // prediction only looks at the single most recent token), so
            // there's no real value to return here. `vocab_size` is used as a
            // stand-in upper bound for the truncation logic in `generate`/
            // `generate_greedy_from_prompt`/`sample_from_prompt`/`grpo_step`
            // (all of which call `get_block_size` to cap context length)
            // purely so those call sites don't need a `BigramLM` special
            // case; it is not meaningful as a context length.
            LanguageModel::BigramLM(model) => model.vocab_size,
            LanguageModel::Gpt(model) => model.block_size,
            // Unlike BigramLM, NgramLM DOES have a real, meaningful context
            // length: N-1 conditioning tokens. Return it as the actual
            // block-size value (not a stand-in), so truncation logic in
            // `generate`/`generate_greedy_from_prompt`/`sample_from_prompt`/
            // `grpo_step` behaves correctly for ngram — a window longer than
            // N-1 tokens is truncated to just the tokens that matter for the
            // next-token key anyway.
            LanguageModel::Ngram(model) => model.context_len(),
        }
    }

    /// Vocab size = embedding-table row count. Needed by `snapshot` to
    /// re-load a frozen reference copy with the same arch. BigramLM stores
    /// this as a field; Gpt stores it (see the `Gpt` struct).
    pub fn get_vocab_size(&self) -> usize {
        match self {
            LanguageModel::BigramLM(model) => model.vocab_size,
            LanguageModel::Gpt(model) => model.vocab_size,
            LanguageModel::Ngram(model) => model.vocab_size,
        }
    }

    /// Hidden / embedding dimension. BigramLM has no concept of a hidden dim
    /// (its embedding table is square `vocab x vocab`), so it returns 0 —
    /// `load_bigram` ignores this value anyway. Gpt stores it as a field.
    /// NgramLM likewise has no hidden-dim concept (a single lookup table),
    /// so it also returns 0.
    pub fn get_hidden_size(&self) -> usize {
        match self {
            LanguageModel::BigramLM(_) => 0,
            LanguageModel::Gpt(model) => model.embed_dims,
            LanguageModel::Ngram(_) => 0,
        }
    }

    /// A single representative head count: the minimum entry of the
    /// per-block `heads_schedule` (equal to the common value when the
    /// architecture is uniform, which is the overwhelmingly common case).
    /// Used where only a scalar makes sense — e.g. the registry's single
    /// `num_heads` column, which predates per-block schedules and isn't
    /// worth migrating for this. `get_heads_schedule` below is the actual
    /// source of truth for reconstructing a model's exact per-block shapes.
    /// Kept `pub` as a small, independently useful API (mirrors
    /// `get_num_blocks`/`get_hidden_size`); only exercised directly by tests
    /// right now, hence `allow(dead_code)` for non-test builds.
    #[allow(dead_code)]
    pub fn get_num_heads(&self) -> usize {
        match self {
            LanguageModel::BigramLM(_) => 0,
            LanguageModel::Gpt(model) => {
                model.heads_schedule.iter().copied().min().unwrap_or(0)
            }
            LanguageModel::Ngram(_) => 0,
        }
    }

    /// Full per-block head-count schedule, length == `get_num_blocks()`.
    /// This is the source of truth `snapshot` uses to reload a frozen
    /// reference copy with the EXACT same per-block shapes, whether the
    /// architecture is uniform or not (unlike `get_num_heads`, which
    /// collapses the schedule to a single representative value). Empty for
    /// `BigramLM` (no heads concept).
    pub fn get_heads_schedule(&self) -> Vec<usize> {
        match self {
            LanguageModel::BigramLM(_) => Vec::new(),
            LanguageModel::Gpt(model) => model.heads_schedule.clone(),
            LanguageModel::Ngram(_) => Vec::new(),
        }
    }

    /// EXPERIMENTAL (Experiment B): whether `lm_head` is tied to
    /// `token_embeddings` (see `Gpt::new`'s doc). `false` for
    /// `Bigram`/`Ngram` (neither has an `lm_head`/embedding-tying concept).
    /// Used by `snapshot` so a frozen reference copy of a tied model reloads
    /// with the same wiring instead of defaulting to untied.
    pub fn get_tie_embeddings(&self) -> bool {
        match self {
            LanguageModel::BigramLM(_) => false,
            LanguageModel::Gpt(model) => model.tie_embeddings,
            LanguageModel::Ngram(_) => false,
        }
    }

    pub fn get_num_blocks(&self) -> usize {
        match self {
            LanguageModel::BigramLM(_) => 0,
            LanguageModel::Gpt(model) => model.num_blocks,
            LanguageModel::Ngram(_) => 0,
        }
    }

    /// Which architecture this is. Used by `snapshot` to dispatch to the
    /// right `load_*` constructor when re-loading the frozen reference.
    pub fn get_model_type(&self) -> ModelType {
        match self {
            LanguageModel::BigramLM(_) => ModelType::Bigram,
            LanguageModel::Gpt(_) => ModelType::Gpt,
            LanguageModel::Ngram(_) => ModelType::Ngram,
        }
    }
}

/// Minimum epoch gap between checkpoint-grid snapshots (`on_best_loss`
/// firings). Early in training, the smoothed loss can hit a new best-ever
/// value on nearly every single epoch — a naive "snapshot on every new best"
/// policy over a run of a few thousand epochs would compute an exhaustive
/// eval grid (a full greedy-decode forward pass per cell) thousands of times,
/// which is wasteful even for a small 10x10 grid and would badly hurt
/// larger-range models. 25 (slightly more than `SMOOTH_WINDOW`'s 20, so
/// consecutive snapshots reflect genuinely different smoothing windows
/// rather than heavily-overlapping ones) bounds a 4000-epoch run to at most
/// 4000/25 = 160 snapshots, and a 10000-epoch run to at most 400 — both
/// comfortably cheap for this project's ~7K-param models and small corpora.
pub const MIN_SNAPSHOT_GAP_EPOCHS: usize = 25;

/// Minimum relative improvement (over the LAST STORED snapshot's smoothed
/// loss) required for a new best-ever smoothed loss to actually fire another
/// checkpoint-grid snapshot, even once `MIN_SNAPSHOT_GAP_EPOCHS` has
/// elapsed. Guards against "new best by a rounding-error amount" re-firing
/// every eligible epoch once the gap has passed and the loss is essentially
/// flat (e.g. deep into a long plateau). 0.5% is small enough to still catch
/// every meaningful step down the loss curve (the interesting frames for the
/// eventual grid-animation UI) but large enough to skip noise-level
/// "improvements".
pub const MIN_SNAPSHOT_REL_IMPROVEMENT: f32 = 0.005;

/// Decide whether a new best-ever smoothed loss (already confirmed by the
/// caller: `smoothed < best_snapshot_loss`) should actually fire a
/// checkpoint-grid snapshot, given the throttle state. Extracted as a pure,
/// side-effect-free function (rather than inlined in the training loop) so
/// the throttle policy is unit-testable without spinning up a model/dataset.
///
/// Policy (see the constants' docs for the exact numbers and reasoning):
/// - The very first snapshot (`last_snapshot_epoch == None`) always fires —
///   we always want a frame at/near the start of training for the animation.
/// - After that, a new best only fires a snapshot if BOTH:
///   1. at least `MIN_SNAPSHOT_GAP_EPOCHS` epochs have elapsed since the
///      last snapshot, AND
///   2. the new smoothed loss is at least `MIN_SNAPSHOT_REL_IMPROVEMENT`
///      relatively better than the last snapshot's smoothed loss.
fn should_snapshot(
    epoch: usize,
    smoothed: f32,
    last_snapshot_epoch: Option<usize>,
    last_snapshot_loss: f32,
) -> bool {
    let Some(last_epoch) = last_snapshot_epoch else {
        return true;
    };
    let gap_ok = epoch.saturating_sub(last_epoch) >= MIN_SNAPSHOT_GAP_EPOCHS;
    // `last_snapshot_loss` is always finite here (it was set the last time
    // `last_snapshot_epoch` was set), but guard the denominator anyway so a
    // pathological ~0 loss can't divide-by-zero into NaN/inf.
    let denom = last_snapshot_loss.abs().max(1e-9);
    let rel_improve_ok = (last_snapshot_loss - smoothed) / denom >= MIN_SNAPSHOT_REL_IMPROVEMENT;
    gap_ok && rel_improve_ok
}

/// Resolve the raw `--num-heads` CLI value into a full per-block head-count
/// schedule of length `num_blocks`.
///
/// - `num_heads.len() == 1`: uniform architecture (today's behavior) —
///   broadcast that single value to every block.
/// - `num_heads.len() == num_blocks`: an explicit per-block schedule, used
///   as-is.
/// - anything else: a clear error naming both lengths, since silently
///   truncating/padding would build the wrong architecture without the
///   caller noticing until a shape mismatch (or worse, a silently-wrong
///   model) surfaced much later.
///
/// Per-block divisibility against `hidden_size` is NOT checked here — that's
/// `Gpt::new`/`Gpt::load`'s job (via `validate_heads_schedule`), since this
/// function also runs for `BigramLM` (via `LanguageModel::new`/`load`'s
/// dispatch), which has no hidden-size/head-count concept at all.
pub fn resolve_heads_schedule(num_heads: &[usize], num_blocks: usize) -> SmolResult<Vec<usize>> {
    match num_heads.len() {
        1 => Ok(vec![num_heads[0]; num_blocks]),
        n if n == num_blocks => Ok(num_heads.to_vec()),
        n => Err(SmolError::invalid_argument(&format!(
            "--num-heads has {n} entries but num_blocks is {num_blocks}; pass either a \
             single number (applied to every block) or a comma-separated list with \
             exactly {num_blocks} entries, e.g. --num-heads 1,2,4,8"
        ))),
    }
}

pub fn sample_multinomial(rng: &mut impl Rng, prs: &[f32]) -> SmolResult<u32> {
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

    #[test]
    fn test_should_snapshot_always_fires_first_time() {
        // No prior snapshot -> always fires, regardless of loss/epoch value.
        assert!(should_snapshot(1, 100.0, None, f32::MAX));
        assert!(should_snapshot(1, 0.0001, None, f32::MAX));
    }

    #[test]
    fn test_should_snapshot_blocks_on_epoch_gap() {
        // Big relative improvement, but not enough epochs have elapsed.
        let last_epoch = 100;
        let last_loss = 1.0;
        assert!(!should_snapshot(
            last_epoch + MIN_SNAPSHOT_GAP_EPOCHS - 1,
            0.1,
            Some(last_epoch),
            last_loss
        ));
        // Exactly the gap threshold -> allowed (gap check is >=).
        assert!(should_snapshot(
            last_epoch + MIN_SNAPSHOT_GAP_EPOCHS,
            0.1,
            Some(last_epoch),
            last_loss
        ));
    }

    #[test]
    fn test_should_snapshot_blocks_on_insufficient_relative_improvement() {
        let last_epoch = 100;
        let next_epoch = last_epoch + MIN_SNAPSHOT_GAP_EPOCHS;
        let last_loss = 1.0;
        // 0.1% improvement < the 0.5% threshold -> blocked.
        assert!(!should_snapshot(next_epoch, 0.999, Some(last_epoch), last_loss));
        // 1% improvement >= the 0.5% threshold -> allowed.
        assert!(should_snapshot(next_epoch, 0.99, Some(last_epoch), last_loss));
    }

    #[test]
    fn test_should_snapshot_requires_both_conditions() {
        let last_epoch = 100;
        let last_loss = 1.0;
        // Gap satisfied but improvement not -> blocked.
        assert!(!should_snapshot(
            last_epoch + MIN_SNAPSHOT_GAP_EPOCHS,
            0.999,
            Some(last_epoch),
            last_loss
        ));
        // Improvement satisfied but gap not -> blocked.
        assert!(!should_snapshot(last_epoch + 1, 0.5, Some(last_epoch), last_loss));
        // Both satisfied -> fires.
        assert!(should_snapshot(
            last_epoch + MIN_SNAPSHOT_GAP_EPOCHS,
            0.5,
            Some(last_epoch),
            last_loss
        ));
    }

    #[rstest]
    #[case("bigramlm")]
    #[case("gpt")]
    #[case("ngram")]
    fn test_lm_save_load_preserves_weights(#[case] model_type: &str) {
        let device = Device::Cpu;
        let vocab_size = 100;
        let hidden_size = 64;
        let block_size = 16;
        let num_heads = [4];
        let num_blocks = 2;

        let model = match model_type {
            "bigramlm" => LanguageModel::new_bigram(vocab_size, &device).unwrap(),
            "gpt" => LanguageModel::new_gpt(
                block_size,
                vocab_size,
                hidden_size,
                &num_heads,
                num_blocks,
                1.0,
                None,
                false,
                &device,
            )
            .unwrap(),
            // A small context_len (2), independent of the shared block_size=16
            // used by the gpt/bigram cases above: NgramLM's embedding table is
            // vocab_size^context_len rows, so context_len=16 with vocab_size=100
            // would be astronomically large.
            "ngram" => LanguageModel::new_ngram(vocab_size, 2, &device).unwrap(),
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
                &num_heads,
                num_blocks,
                false,
                &device,
            )
            .unwrap(),
            "ngram" => LanguageModel::load_ngram(&path, 2, vocab_size, &device).unwrap(),
            _ => panic!("Unknown model type {model_type}"),
        };

        assert_eq!(model.get_block_size(), loaded_model.get_block_size());

        let original_embeddings = model.get_embeddings_vec2d().unwrap();
        let loaded_embeddings = loaded_model.get_embeddings_vec2d().unwrap();

        assert_eq!(original_embeddings, loaded_embeddings);
    }

    /// `completion_logp_scalar` must return `0.0` for an empty completion
    /// (the documented skip guard) and a finite, non-positive value for a
    /// non-empty completion whose prompt context survives truncation.
    /// A sum of log-probs is always ≤ 0, so the non-empty case is bounded
    /// above by 0.0; we also assert it's finite so NaN/inf would fail.
    #[rstest]
    #[case("bigramlm")]
    #[case("gpt")]
    #[case("ngram")]
    fn test_completion_logp_scalar(#[case] model_type: &str) {
        let device = Device::Cpu;
        let vocab_size = 100;
        let hidden_size = 64;
        let block_size = 16;
        let num_heads = [4];
        let num_blocks = 2;

        let model = match model_type {
            "bigramlm" => LanguageModel::new_bigram(vocab_size, &device).unwrap(),
            "gpt" => LanguageModel::new_gpt(
                block_size,
                vocab_size,
                hidden_size,
                &num_heads,
                num_blocks,
                1.0,
                None,
                false,
                &device,
            )
            .unwrap(),
            // Small context_len (2), independent of the shared block_size=16
            // used by gpt (see test_lm_save_load_preserves_weights's comment).
            "ngram" => LanguageModel::new_ngram(vocab_size, 2, &device).unwrap(),
            _ => panic!("Unknown model type {model_type}"),
        };

        // Empty completion → 0.0 (skip guard).
        let empty = model
            .completion_logp_scalar(&[1, 2, 3], &[], &device)
            .unwrap();
        assert_eq!(empty, 0.0, "empty completion must return 0.0");

        // Non-empty completion with surviving prompt context → finite,
        // non-positive (sum of log-probs ≤ 0).
        let logp = model
            .completion_logp_scalar(&[1, 2, 3], &[4, 5], &device)
            .unwrap();
        assert!(
            logp.is_finite(),
            "non-empty completion logp must be finite, got {logp}"
        );
        assert!(
            logp <= 0.0,
            "sum of log-probs must be <= 0.0, got {logp}"
        );
    }

    /// `grpo_step_full` must (a) return `0.0` and take no step when every
    /// advantage is ~0 (uniform-reward group, the same guard as the lite
    /// step), and (b) return a finite loss when advantages are non-zero,
    /// running K mini-epochs end-to-end without panicking. The KL term
    /// uses the same model as both reference and sampling policy here
    /// (so `ref_logp == old_logp`), which is enough to exercise the math.
    #[rstest]
    #[case("bigramlm")]
    #[case("gpt")]
    #[case("ngram")]
    fn test_grpo_step_full_guard_and_step(#[case] model_type: &str) {
        let device = Device::Cpu;
        let vocab_size = 100;
        let hidden_size = 64;
        let block_size = 16;
        let num_heads = [4];
        let num_blocks = 2;

        let model = match model_type {
            "bigramlm" => LanguageModel::new_bigram(vocab_size, &device).unwrap(),
            "gpt" => LanguageModel::new_gpt(
                block_size,
                vocab_size,
                hidden_size,
                &num_heads,
                num_blocks,
                1.0,
                None,
                false,
                &device,
            )
            .unwrap(),
            // Small context_len (2), independent of the shared block_size=16
            // used by gpt (see test_lm_save_load_preserves_weights's comment).
            "ngram" => LanguageModel::new_ngram(vocab_size, 2, &device).unwrap(),
            _ => panic!("Unknown model type {model_type}"),
        };

        let prompt = vec![1u32, 2, 3];
        let completions = vec![vec![4u32, 5], vec![6, 7]];

        // Uniform-advantage guard: all-zero advantages → no step, 0.0.
        let mut opt = model.make_optimizer(1e-3).unwrap();
        let loss = model
            .grpo_step_full(
                &prompt,
                &completions,
                &[0.0, 0.0],
                &[0.0, 0.0],
                &[0.0, 0.0],
                0.2,
                0.04,
                2,
                &mut opt,
                &device,
            )
            .unwrap();
        assert_eq!(loss, 0.0, "uniform-advantage group must skip (return 0.0)");

        // Non-uniform advantages → runs K=2 mini-epochs, returns a finite
        // loss. old/ref logps cached from the same model (so they're equal
        // at epoch 0, making the KL term = logp_theta - old_logp = the
        // log-ratio, which is a well-behaved finite value).
        let mut opt = model.make_optimizer(1e-3).unwrap();
        let old0 = model.completion_logp_scalar(&prompt, &completions[0], &device).unwrap();
        let old1 = model.completion_logp_scalar(&prompt, &completions[1], &device).unwrap();
        let loss = model
            .grpo_step_full(
                &prompt,
                &completions,
                &[1.0, -1.0],
                &[old0, old1],
                &[old0, old1],
                0.2,
                0.04,
                2,
                &mut opt,
                &device,
            )
            .unwrap();
        assert!(loss.is_finite(), "grpo_step_full loss must be finite, got {loss}");
    }

    /// `snapshot` must produce a fully independent `LanguageModel`: same
    /// arch (so its `get_*` getters match), same weights (so a logp
    /// computed under it equals the original), but a separate `VarMap` so
    /// an optimizer step on the original does NOT move the snapshot.
    #[rstest]
    #[case("bigramlm")]
    #[case("gpt")]
    #[case("ngram")]
    fn test_snapshot_is_independent_copy(#[case] model_type: &str) {
        let device = Device::Cpu;
        let vocab_size = 100;
        let hidden_size = 64;
        let block_size = 16;
        let num_heads = [4];
        let num_blocks = 2;

        let model = match model_type {
            "bigramlm" => LanguageModel::new_bigram(vocab_size, &device).unwrap(),
            "gpt" => LanguageModel::new_gpt(
                block_size,
                vocab_size,
                hidden_size,
                &num_heads,
                num_blocks,
                1.0,
                None,
                false,
                &device,
            )
            .unwrap(),
            // Small context_len (2), independent of the shared block_size=16
            // used by gpt (see test_lm_save_load_preserves_weights's comment).
            "ngram" => LanguageModel::new_ngram(vocab_size, 2, &device).unwrap(),
            _ => panic!("Unknown model type {model_type}"),
        };

        let dir = TempDir::new().unwrap();
        let snap_path = dir.path().join("snap.bin");
        let reference = model.snapshot(&snap_path, &device).unwrap();

        // Same arch getters.
        assert_eq!(model.get_vocab_size(), reference.get_vocab_size());
        assert_eq!(model.get_block_size(), reference.get_block_size());
        assert_eq!(model.get_model_type(), reference.get_model_type());
        if matches!(model, LanguageModel::Gpt(_)) {
            assert_eq!(model.get_hidden_size(), reference.get_hidden_size());
            assert_eq!(model.get_num_heads(), reference.get_num_heads());
            assert_eq!(model.get_num_blocks(), reference.get_num_blocks());
        }

        // Same weights → same logp on a sample completion.
        let prompt = vec![1u32, 2, 3];
        let comp = vec![4u32, 5];
        let logp_before = model.completion_logp_scalar(&prompt, &comp, &device).unwrap();
        let ref_logp = reference.completion_logp_scalar(&prompt, &comp, &device).unwrap();
        assert!(
            (logp_before - ref_logp).abs() < 1e-5,
            "snapshot must reproduce the original's logp, got {logp_before} vs {ref_logp}"
        );

        // Take an optimizer step on the original (non-uniform advantages)
        // and confirm the reference's logp is unchanged → independent
        // VarMap.
        let mut opt = model.make_optimizer(1e-3).unwrap();
        let _ = model
            .grpo_step_full(
                &prompt,
                &[comp.clone()],
                &[1.0],
                &[logp_before],
                &[ref_logp],
                0.2,
                0.04,
                1,
                &mut opt,
                &device,
            )
            .unwrap();
        let ref_logp_after = reference.completion_logp_scalar(&prompt, &comp, &device).unwrap();
        assert!(
            (ref_logp - ref_logp_after).abs() < 1e-5,
            "snapshot must be unaffected by optimizer steps on the original: \
             ref_logp {ref_logp} -> {ref_logp_after}"
        );
    }
}
