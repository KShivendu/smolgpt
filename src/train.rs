use std::time::Instant;

use crate::{
    args::{Args, TokenizerType},
    dataset::{self, Dataset},
    error::SmolError,
    model::LanguageModel,
    tokenizer::{BpeTokenizer, SimpleTokenizer, Tokenizer},
};
use candle_core::{Device, Shape, Tensor};
use rand::{rngs::StdRng, SeedableRng};

pub fn do_training(args: Args) -> Result<(), SmolError> {
    let Args {
        dataset_path,
        model_path,
        epochs,
        train,
        generate,
        eval,
        eval_samples,
        eval_min,
        eval_max,
        eval_mode,
        eval_ops,
        rft,
        rft_rounds,
        rft_prompts,
        rft_samples,
        rft_temperature,
        rft_epochs,
        rft_min,
        rft_max,
        rft_ops,
        grpo,
        grpo_rounds,
        grpo_prompts,
        grpo_group,
        grpo_temperature,
        grpo_lr,
        grpo_min,
        grpo_max,
        grpo_ops,
        grpo_mode,
        grpo_clip_eps,
        grpo_kl_beta,
        grpo_epochs,
        quantize,
        jacobian_lens: run_jacobian_lens_flag,
        patience,
        min_delta,
        no_dropout,
        lr,
        mask_loss,
        aligned_windows,
        init_std,
        init_gain,
        tie_embeddings,
        model_type,
        tokenizer: tokenizer_type,
        vocab_size: target_vocab_size,
        seed,
        block_size,
        ngram_order,
        hidden_size,
        num_heads,
        num_blocks,
        num_batches,
        serve,
        port,
        host,
    } = args;

    // For `-m ngram`, `--ngram-order` (N) OVERRIDES `--block-size`: NgramLM's
    // real, meaningful context length is `N - 1` (unlike BigramLM, which has
    // no such concept), so `block_size` from here on IS that context length
    // for every downstream use (model construction/load, the registry's
    // `block_size` column, checkpoint-grid truncation, etc). Gpt/Bigram are
    // unaffected.
    let block_size = if matches!(model_type, crate::args::ModelType::Ngram) {
        ngram_order.saturating_sub(1).max(1)
    } else {
        block_size
    };

    if !train && !generate && !eval && !rft && !grpo && !quantize && !serve {
        return Err(SmolError::invalid_argument(
            "Either --train, --generate, --eval, --rft, --grpo, --quantize, or --serve must be specified",
        ));
    }

    // --serve manages its own per-request model/tokenizer loading from
    // models.toml, so it neither needs nor benefits from the corpus/dataset/
    // model preamble below. Bail out early before the corpus read.
    if serve {
        crate::serve::run_serve(&host, port, eval_mode)?;
        return Ok(());
    }

    let corpus = dataset::load_corpus(&dataset_path, false)?;
    let device = Device::Cpu;

    let mut rng: StdRng = match seed {
        Some(s) => {
            println!("Using seeded RNG (seed = {s})");
            // NOTE: this seeds batch sampling (Dataset) and token sampling
            // (generate). Fresh model init still uses candle's CPU RNG, which
            // cannot be seeded in candle-core 0.9.1. For fully reproducible
            // runs, train once to save the model, then `--generate` loads it
            // from disk under the seed.
            StdRng::seed_from_u64(s)
        }
        None => StdRng::from_os_rng(),
    };

    let tokenizer: Box<dyn Tokenizer<u32>> = match tokenizer_type {
        TokenizerType::Char => Box::new(SimpleTokenizer::new(&corpus)),
        TokenizerType::Bpe => Box::new(BpeTokenizer::train(&corpus, target_vocab_size)),
    };
    println!(
        "Tokenizer: {:?}, vocab size: {}",
        tokenizer_type,
        tokenizer.vocab_size()
    );

    // Keep char- and BPE-trained models in separate files: their vocabularies
    // (and therefore embedding tables) are incompatible.
    let model_path = model_path.unwrap_or_else(|| {
        let suffix = match tokenizer_type {
            TokenizerType::Char => "char",
            TokenizerType::Bpe => "bpe",
        };
        match model_type {
            crate::args::ModelType::Gpt => format!("gpt-{suffix}.bin").into(),
            crate::args::ModelType::Bigram => format!("bigram-{suffix}.bin").into(),
            crate::args::ModelType::Ngram => format!("ngram-{suffix}.bin").into(),
        }
    });

    let vocab_size = tokenizer.vocab_size();

    // Eval-only / RFT-only need just the tokenizer (built above) + a loaded
    // model. Skip the encoded-corpus tensor and Dataset construction in those
    // cases to keep them fast and avoid requiring the corpus to be encoded.
    // (The tokenizer still needs the corpus *string* for vocab scanning, which
    // is why `corpus` is loaded unconditionally above.) `--serve` is handled
    // earlier and never reaches this point, so it isn't mentioned here.
    let only_eval = eval && !train && !generate && !rft && !grpo && !quantize;
    let only_rft = rft && !train && !generate && !eval && !grpo && !quantize;
    let only_grpo = grpo && !train && !generate && !eval && !rft && !quantize;
    let only_quantize = quantize && !train && !generate && !eval && !rft && !grpo;

    let mut dataset: Option<Dataset> = None;
    if !only_eval && !only_rft && !only_grpo && !only_quantize {
        let encoded_corpus = tokenizer.encode(&corpus);
        let encoded_corpus_len = encoded_corpus.len();
        let data = Tensor::from_vec(encoded_corpus, Shape::from(encoded_corpus_len), &device)?;
        println!(
            "Encoded text tensor shape: {:?}; dtype {:?}",
            data.shape(),
            data.dtype()
        );
        // EXPERIMENTAL: only build the answer mask when `--mask-loss` is
        // requested and the tokenizer is char (1 token == 1 corpus char, the
        // alignment `compute_answer_mask` assumes). BPE would misalign, so
        // silently skip (train_with_dropout falls back to unmasked loss when
        // no mask is present).
        let answer_mask = if mask_loss && matches!(tokenizer_type, TokenizerType::Char) {
            Some(dataset::compute_answer_mask(&corpus))
        } else {
            None
        };
        // EXPERIMENTAL (Hypothesis B): only meaningful for a char tokenizer
        // (1 char == 1 token, same alignment assumption as `--mask-loss`'s
        // answer mask); silently falls back to `None` (regular uniform
        // sampling) otherwise.
        let fact_boundaries = if aligned_windows && matches!(tokenizer_type, TokenizerType::Char) {
            Some(dataset::compute_fact_boundaries(&corpus))
        } else {
            None
        };
        dataset = Some(Dataset::with_rng_and_mask_aligned(
            data,
            answer_mask,
            fact_boundaries,
            0.9,
            rng.clone(),
        )?);
    }

    let num_batches = num_batches;

    // --eval / --rft / --grpo never train from scratch, so the model file must
    // already exist on disk. (RFT does SFT *on the winners*, but it must start
    // from a pretrained model — there's no point sampling completions from a
    // freshly initialized model. GRPO likewise needs a pretrained policy to
    // sample completions worth scoring.)
    if (eval || rft || grpo || quantize) && !model_path.exists() {
        return Err(SmolError::invalid_argument(&format!(
            "{} requires an existing model file at {}; train first",
            if quantize {
                "--quantize"
            } else if grpo {
                "--grpo"
            } else if rft {
                "--rft"
            } else {
                "--eval"
            },
            model_path.display()
        )));
    }

    let model = if model_path.exists() {
        println!("Loading {model_type:?} model from {}", model_path.display());
        LanguageModel::load(
            model_type,
            &model_path,
            block_size,
            vocab_size,
            hidden_size,
            &num_heads,
            num_blocks,
            tie_embeddings,
            &device,
        )?
    } else {
        println!("Creating new {model_type:?} model");
        LanguageModel::new(
            model_type,
            block_size,
            vocab_size,
            hidden_size,
            &num_heads,
            num_blocks,
            init_std,
            init_gain,
            tie_embeddings,
            &device,
        )?
    };

    // Resolve the raw `--num-heads` CLI value (single number or
    // comma-separated per-block list) into the full per-block schedule, for
    // the registry note / TrainingMeta below. The model construction above
    // already validated this (via `Gpt::new`/`Gpt::load`), so this should
    // never actually error here for a GPT model that just loaded/built
    // successfully; BigramLM ignores num_heads entirely, so resolve against
    // its own num_blocks (0) rather than the GPT one when applicable.
    let heads_schedule_for_meta = match model_type {
        crate::args::ModelType::Gpt => {
            crate::model::resolve_heads_schedule(&num_heads, num_blocks)?
        }
        crate::args::ModelType::Bigram | crate::args::ModelType::Ngram => Vec::new(),
    };

    // Built once so both the --train (auto-register) and --eval (record_eval
    // + maybe register) branches can use it without duplicating the literal.
    // Borrows `dataset_path` + `model_path` (both live till end of fn).
    // `base_model_id` is None here (the common meta is for the base model
    // itself); `--rft`/`--grpo` build a separate meta with the variant path
    // and `base_model_id = Some(base_id)`.
    let training_meta = crate::registry::TrainingMeta {
        model_type,
        tokenizer: tokenizer_type,
        block_size,
        hidden_size,
        heads_schedule: &heads_schedule_for_meta,
        num_blocks,
        aligned_windows,
        dataset_path: &dataset_path,
        model_path: &model_path,
        actual_vocab_size: vocab_size,
        eval_min,
        eval_max,
        eval_samples,
        eval_mode,
        seed,
        base_model_id: None,
    };

    if train {
        let dataset = dataset
            .as_mut()
            .expect("dataset must be built when --train is set");
        let now = Instant::now();

        // Open the registry ONCE and reuse it for the start-registration, the
        // per-checkpoint upserts (via the on_checkpoint closure), and the
        // final post-training re-register. Best-effort: a registry failure
        // prints a warning but never aborts the training run — training
        // proceeds without live upserts and the rest of the function
        // (--eval/--generate) still runs.
        let reg = crate::registry::Registry::open();

        // Register the model at training START with a placeholder outcome so
        // the card appears in the UI the moment training begins. The
        // post-training register_model below upserts with the final outcome
        // (register_model is an upsert keyed on id, so the start row is
        // overwritten in place — evals/created_at survive).
        let model_id_for_cb: Option<String> = match &reg {
            Ok(r) => {
                let placeholder = crate::model::TrainOutcome::placeholder();
                let start_rec =
                    crate::registry::ModelRecord::from_training(&training_meta, &placeholder);
                let id = start_rec.id.clone();
                match r.register_model(&start_rec) {
                    Ok(()) => println!(
                        "Registered model {} in smolgpt.db at training start",
                        start_rec.id
                    ),
                    Err(e) => eprintln!(
                        "[train] WARNING: failed to register model at start in smolgpt.db: {e}"
                    ),
                }
                Some(id)
            }
            Err(e) => {
                eprintln!("[train] WARNING: failed to open smolgpt.db: {e}");
                None
            }
        };

        // Resolve the grid's operand range once (same corpus-derivation rule
        // `--eval` uses) so the checkpoint-grid snapshots and the final Grid
        // tab cache describe the same range. Only enable snapshotting when
        // the range fits `eval::MAX_GRID_AXIS` — a wider-range corpus would
        // make each mid-training exhaustive grid expensive (and is out of
        // scope here: this task's corpus is [0,9], comfortably 10x10).
        let (grid_min, grid_max) = crate::registry::resolve_eval_range(&training_meta);
        let grid_axis = grid_max - grid_min + 1;
        let checkpoint_grids_enabled = grid_axis > 0 && grid_axis <= crate::eval::MAX_GRID_AXIS;
        // Derive the grid's operators from the corpus itself (same fix as
        // `--serve`'s eval endpoint) rather than trusting `--eval-ops`'s
        // default ("+,-") — an addition-only corpus's char tokenizer has no
        // `-` token at all, so grid cells sampled with `-` would silently
        // mis-tokenize and either warn-and-corrupt or just be wrong, not
        // reflect "the model hasn't learned subtraction" (it was never asked
        // to).
        let grid_ops = crate::dataset::operators_present(&corpus).unwrap_or_else(|| eval_ops.clone());
        if !checkpoint_grids_enabled {
            println!(
                "[train] Checkpoint-grid snapshots disabled: operand range \
                 [{grid_min},{grid_max}] ({grid_axis}x{grid_axis}) exceeds \
                 MAX_GRID_AXIS={}; only the final Grid-tab cache (via --serve) \
                 will be available for this model.",
                crate::eval::MAX_GRID_AXIS
            );
        }

        // on_best_loss closure: whenever `train_with_dropout` reports a new
        // best-so-far smoothed loss (throttled — see `model::should_snapshot`),
        // compute the exhaustive eval grid against the model's CURRENT
        // (mid-training) weights and append it to the `checkpoint_grids`
        // history so a follow-up UI can animate through it. Best-effort: any
        // failure here only warns and never aborts training.
        let mut on_best_loss = |epoch: usize, smoothed_loss: f32, m: &LanguageModel| {
            if !checkpoint_grids_enabled {
                return;
            }
            let (Some(reg), Some(id)) = (reg.as_ref().ok(), model_id_for_cb.as_ref()) else {
                return;
            };
            match crate::eval::run_eval_grid(
                m,
                tokenizer.as_ref(),
                &device,
                grid_min,
                grid_max,
                block_size,
                &grid_ops,
            ) {
                Ok(report) => {
                    let correct = report.correct as i64;
                    let total = report.total as i64;
                    match serde_json::to_string(&report) {
                        Ok(json) => {
                            if let Err(e) = reg.record_checkpoint_grid(
                                id,
                                epoch,
                                smoothed_loss as f64,
                                grid_min,
                                grid_max,
                                &json,
                                correct,
                                total,
                            ) {
                                eprintln!(
                                    "[train] WARNING: failed to record checkpoint grid at \
                                     epoch {epoch} for {id} in smolgpt.db: {e}"
                                );
                            }
                        }
                        Err(e) => eprintln!(
                            "[train] WARNING: failed to serialize checkpoint grid at \
                             epoch {epoch} for {id}: {e}"
                        ),
                    }
                }
                Err(e) => eprintln!(
                    "[train] WARNING: failed to compute checkpoint grid at epoch {epoch} \
                     for {id}: {e}"
                ),
            }
        };

        // on_checkpoint closure: upsert the partial loss trajectory into the
        // `trainings` table at each checkpoint save (every 10 epochs + final)
        // so the web UI shows live training progress. Borrows `&reg` (when
        // open) and the captured `model_id_for_cb`; `upsert_training` takes
        // `&self` so the immutable borrow is fine. The closure is dropped when
        // `train_with_dropout` returns, releasing the borrow before the
        // final `register_model` below. When the registry failed to open,
        // the closure is a no-op.
        let mut on_checkpoint = |outcome: &crate::model::TrainOutcome| {
            let (Some(reg), Some(id)) = (reg.as_ref().ok(), model_id_for_cb.as_ref()) else {
                return;
            };
            let loss_json = serde_json::to_string(&outcome.losses).unwrap_or_else(|e| {
                eprintln!(
                    "[train] WARNING: failed to serialize loss trajectory: {e}; storing []"
                );
                "[]".to_string()
            });
            // Intermediate checkpoint: no freshly-computed training accuracy
            // (it's only computed once, after training finishes — see below).
            // `None` here is safe: `upsert_training` COALESCEs against the
            // previously-stored value rather than clobbering it with NULL.
            if let Err(e) = reg.upsert_training(
                id,
                "sft",
                outcome.epochs_run,
                outcome.early_stopped,
                outcome.final_loss,
                &loss_json,
                "null",
                None,
                None,
                None,
                None,
                None,
            ) {
                eprintln!(
                    "[train] WARNING: failed to upsert training metrics for {id} in smolgpt.db: {e}"
                );
            }
        };

        // Dropout regularizes against overfitting a large/diverse corpus, but
        // for a tiny, fully-memorizable corpus (e.g. a few dozen arithmetic
        // facts) it does the opposite of what you want: it caps how low the
        // training loss can go, since a different random subset of hidden
        // units is zeroed on every step, so the network can never settle into
        // the sharp, fully-confident weights needed to get every training
        // example right at (dropout-free) inference time. `--no-dropout`
        // turns this off for exactly that "I want to memorize a small corpus"
        // case; leave it on (the default) for anything where held-out
        // generalization actually matters. Pass CLI patience/min_delta so
        // `--train` gets early stopping by default (patience=200,
        // min_delta=0.001) — users can disable with `--patience 0`.
        let outcome = model.train_with_dropout(
            dataset,
            &model_path,
            epochs,
            num_batches,
            !no_dropout,
            patience,
            min_delta,
            Some(&mut on_checkpoint),
            lr,
            mask_loss,
            Some(&mut on_best_loss),
        )?;
        println!("Training completed in {:.2?}", now.elapsed());

        // Exact greedy-decoding accuracy over the literal training corpus
        // (distinct from --eval's random-sampled-range accuracy) so the UI
        // can show a real "how much of what it was trained on did it
        // actually learn" number instead of just the opaque final loss.
        let (train_correct, train_total) =
            crate::eval::eval_corpus_exact(&model, tokenizer.as_ref(), &device, &corpus, block_size)?;
        let train_pct = if train_total > 0 {
            train_correct as f64 * 100.0 / train_total as f64
        } else {
            0.0
        };
        println!("Training-set accuracy: {train_correct}/{train_total} ({train_pct:.2}%)");

        // Final register_model upserts the model row with the complete
        // outcome (epochs_run, early_stopped, final note). The final
        // on_checkpoint callback already upserted the trainings row inside
        // train_with_dropout, so the loss trajectory is current; this upsert
        // adds the training-set accuracy now that it's been computed.
        if let Ok(reg) = &reg {
            let rec = crate::registry::ModelRecord::from_training(&training_meta, &outcome);
            match reg.register_model(&rec) {
                Ok(()) => println!("Registered model {} in smolgpt.db", rec.id),
                Err(e) => eprintln!(
                    "[train] WARNING: failed to register model in smolgpt.db: {e}"
                ),
            }
            let loss_json = serde_json::to_string(&outcome.losses).unwrap_or_else(|e| {
                eprintln!("[train] WARNING: failed to serialize loss trajectory: {e}; storing []");
                "[]".to_string()
            });
            if let Err(e) = reg.upsert_training(
                &rec.id,
                "sft",
                outcome.epochs_run,
                outcome.early_stopped,
                outcome.final_loss,
                &loss_json,
                "null",
                Some(train_correct as i64),
                Some(train_total as i64),
                None,
                None,
                None,
            ) {
                eprintln!(
                    "[train] WARNING: failed to upsert final training accuracy for {} in smolgpt.db: {e}",
                    rec.id
                );
            }

            // "Compiled" precompute: run the Jacobian-lens interpretability
            // analysis right now (Gpt-only) and cache it, so `--serve`'s
            // Jacobian tab shows an already-computed result with no on-demand
            // click needed. Same registry row/shape the on-demand route
            // (`serve.rs`'s `jacobian_lens_model`) writes, so a precomputed
            // and an on-demand result are indistinguishable once cached.
            if run_jacobian_lens_flag {
                if !matches!(model_type, crate::args::ModelType::Gpt) {
                    println!(
                        "[train] --jacobian-lens ignored: only applicable to -m gpt models \
                         (this run is -m {model_type:?})"
                    );
                } else {
                    match std::env::current_dir() {
                        Ok(project_root) => {
                            println!("[train] Running Jacobian-lens analysis (--jacobian-lens)...");
                            match crate::jacobian_lens::run_jacobian_lens_for_model(&rec, &project_root) {
                                Ok(outcome) => {
                                    if let Err(e) = reg.record_jacobian_lens(
                                        &rec.id,
                                        &outcome.results_json,
                                        &outcome.plot_dir_rel,
                                        &outcome.plot_files,
                                    ) {
                                        eprintln!(
                                            "[train] WARNING: failed to cache jacobian-lens result for {}: {e}",
                                            rec.id
                                        );
                                    } else {
                                        println!(
                                            "[train] Jacobian-lens analysis cached for {} ({} plots)",
                                            rec.id,
                                            outcome.plot_files.len()
                                        );
                                    }
                                }
                                Err(e) => eprintln!(
                                    "[train] WARNING: --jacobian-lens analysis failed for {}: {e}",
                                    rec.id
                                ),
                            }
                        }
                        Err(e) => eprintln!("[train] WARNING: --jacobian-lens skipped, cwd unavailable: {e}"),
                    }
                }
            }
        }
    }

    if generate {
        println!("Generating from {model_type:?} model ({})", model_path.display());
        let output = model.generate(500, &mut rng, &device)?;
        let decoded_output = tokenizer.decode(&output);
        println!("Generated text: {decoded_output}");
    }

    if eval {
        // Resolve the eval operand range. User override (both flags `Some`)
        // wins in both modes; otherwise smart mode derives from the corpus,
        // legacy mode falls back to 0/999. This matches the registration-time
        // rule in `ModelRecord::from_training` so a `--train` + `--eval` run
        // tests the same range it stored.
        let (effective_min, effective_max) = crate::registry::resolve_eval_range(&training_meta);
        println!(
            "Evaluating {model_type:?} model ({}) on {eval_samples} held-out arithmetic problems \
             (operands in [{effective_min}, {effective_max}])",
            model_path.display()
        );
        let report = crate::eval::run_eval(
            &model,
            tokenizer.as_ref(),
            &device,
            eval_samples,
            effective_min,
            effective_max,
            block_size,
            seed,
            &eval_ops,
        )?;

        // Best-effort: record the eval summary in smolgpt.db so `--serve` can
        // show it on page reload. If the model isn't registered yet (e.g. an
        // eval-only run on a model trained before the DB existed), register a
        // minimal record from Args first. If the DB can't be opened or the
        // model can't be registered, skip recording — eval still printed to
        // stdout above.
        match crate::registry::Registry::open() {
            Ok(reg) => {
                let id = crate::registry::derive_id(&model_path);
                match reg.get_model(&id) {
                    Ok(None) => {
                        let minimal_outcome = crate::model::TrainOutcome::placeholder();
                        let rec = crate::registry::ModelRecord::from_training(
                            &training_meta,
                            &minimal_outcome,
                        );
                        if let Err(e) = reg.register_model(&rec) {
                            eprintln!(
                                "[eval] WARNING: failed to register model {id} in smolgpt.db: {e}"
                            );
                        }
                    }
                    Ok(Some(_)) => {}
                    Err(e) => eprintln!("[eval] WARNING: failed to look up model {id}: {e}"),
                }
                if let Err(e) = reg.record_eval(&id, &report, seed) {
                    eprintln!("[eval] WARNING: failed to record eval for {id}: {e}");
                }
            }
            Err(e) => eprintln!("[eval] WARNING: failed to open smolgpt.db: {e}"),
        }
    }

    if rft {
        // Derive a variant path so the base `.bin` is preserved and the RFT
        // variant is a separate file (the old behavior mutated the base in
        // place). The model was loaded from `model_path` (the base) above;
        // `run_rft` mutates it in-place and saves to `variant_path` each
        // round + each SFT sub-checkpoint.
        let variant_path =
            crate::registry::derive_variant_path(&model_path, "rft");
        let base_id = crate::registry::derive_id(&model_path);
        let variant_id = crate::registry::derive_id(&variant_path);
        println!(
            "Running RFT on {model_type:?} model (base {}) → variant {} for {rft_rounds} rounds \
             (operands in [{rft_min}, {rft_max}])",
            model_path.display(),
            variant_path.display()
        );

        // Build a variant-specific TrainingMeta: same arch/tokenizer/dataset as
        // the base, but `model_path` = variant path (so the ModelRecord's path
        // field points at the variant .bin) and `base_model_id` = base's id
        // (so the UI groups this variant under the base card).
        let rft_meta = training_meta.with_variant(&variant_path, &base_id);

        let reg = crate::registry::Registry::open();
        // Register the variant at RFT start with a placeholder outcome so the
        // card appears (nested under its base) the moment RFT begins. The
        // variant row has base_model_id = base's id.
        if let Ok(reg) = &reg {
            let placeholder = crate::model::TrainOutcome::placeholder();
            let variant_rec =
                crate::registry::ModelRecord::from_training(&rft_meta, &placeholder);
            match reg.register_model(&variant_rec) {
                Ok(()) => println!(
                    "Registered RFT variant {} (base {}) in smolgpt.db at RFT start",
                    variant_rec.id, base_id
                ),
                Err(e) => eprintln!(
                    "[rft] WARNING: failed to register RFT variant {base_id} in smolgpt.db: {e}"
                ),
            }
        }

        // on_round closure: upsert the partial RFT summary into the
        // `trainings` table after each round so the web UI shows live
        // per-round progress. The final round's callback upserts the complete
        // summary.
        let variant_id_for_cb = variant_id.clone();
        let mut on_round = |summary: &crate::rft::RftSummary| {
            let Some(reg) = reg.as_ref().ok() else { return };
            let rft_final_loss = summary
                .per_round_sft_final_losses
                .iter()
                .rev()
                .find_map(|&l| l)
                .unwrap_or(0.0);
            let summary_json = serde_json::to_string(summary).unwrap_or_else(|e| {
                eprintln!(
                    "[rft] WARNING: failed to serialize RftSummary: {e}; storing null"
                );
                "null".to_string()
            });
            if let Err(e) = reg.upsert_training(
                &variant_id_for_cb,
                "rft",
                summary.rounds,
                false,
                rft_final_loss,
                "null",
                &summary_json,
                None,
                None,
                Some(rft_min),
                Some(rft_max),
                Some(&rft_ops),
            ) {
                eprintln!(
                    "[rft] WARNING: failed to upsert RFT metrics for {} in smolgpt.db: {e}",
                    variant_id_for_cb
                );
            }
        };

        let summary = crate::rft::run_rft(
            &model,
            tokenizer.as_ref(),
            &device,
            &variant_path,
            block_size,
            rft_rounds,
            rft_prompts,
            rft_samples,
            rft_temperature,
            rft_epochs,
            num_batches,
            rft_min,
            rft_max,
            &rft_ops,
            seed,
            eval_samples,
            Some(&mut on_round),
        )?;
        // The final on_round callback already upserted the complete summary;
        // log a confirmation. The return value is for the caller's info.
        let rft_final_loss = summary
            .per_round_sft_final_losses
            .iter()
            .rev()
            .find_map(|&l| l)
            .unwrap_or(0.0);
        println!(
            "RFT complete: {} rounds, final SFT loss {:.4}",
            summary.rounds, rft_final_loss
        );
    }

    if grpo {
        // Derive a variant path so the base `.bin` is preserved and the GRPO
        // variant is a separate file (the old behavior mutated the base in
        // place). The model was loaded from `model_path` (the base) above;
        // `run_grpo` mutates it in-place and saves to `variant_path` each
        // round.
        //
        // The suffix includes the mode (`-grpo` for lite, `-grpo-full` for
        // full/PPO-style) so a `--grpo-mode full` run never collides with an
        // existing lite run's `.bin` file or registry id — they're sibling
        // variants of the same base, not the same variant re-trained.
        let grpo_suffix = match grpo_mode {
            crate::args::GrpoMode::Lite => "grpo",
            crate::args::GrpoMode::Full => "grpo-full",
        };
        let variant_path =
            crate::registry::derive_variant_path(&model_path, grpo_suffix);
        let base_id = crate::registry::derive_id(&model_path);
        let variant_id = crate::registry::derive_id(&variant_path);
        println!(
            "Running GRPO on {model_type:?} model (base {}) → variant {} for {grpo_rounds} rounds \
             (G={grpo_group}, prompts/round={grpo_prompts}, T={grpo_temperature}, \
             lr={grpo_lr}, operands in [{grpo_min}, {grpo_max}], ops [{grpo_ops}])",
            model_path.display(),
            variant_path.display()
        );

        // Build a variant-specific TrainingMeta: same arch/tokenizer/dataset as
        // the base, but `model_path` = variant path and `base_model_id` =
        // base's id (so the UI groups this variant under the base card).
        let grpo_meta = training_meta.with_variant(&variant_path, &base_id);

        let reg = crate::registry::Registry::open();
        // Register the variant at GRPO start with a placeholder outcome so
        // the card appears (nested under its base) the moment GRPO begins.
        if let Ok(reg) = &reg {
            let placeholder = crate::model::TrainOutcome::placeholder();
            let variant_rec =
                crate::registry::ModelRecord::from_training(&grpo_meta, &placeholder);
            match reg.register_model(&variant_rec) {
                Ok(()) => println!(
                    "Registered GRPO variant {} (base {}) in smolgpt.db at GRPO start",
                    variant_rec.id, base_id
                ),
                Err(e) => eprintln!(
                    "[grpo] WARNING: failed to register GRPO variant {base_id} in smolgpt.db: {e}"
                ),
            }
        }

        // on_round closure: upsert the partial GRPO summary after each round.
        let variant_id_for_cb = variant_id.clone();
        let mut on_round = |summary: &crate::grpo::GrpoSummary| {
            let Some(reg) = reg.as_ref().ok() else { return };
            let grpo_final_loss = summary
                .per_round_losses
                .iter()
                .rev()
                .find_map(|&l| l)
                .unwrap_or(0.0);
            let summary_json = serde_json::to_string(summary).unwrap_or_else(|e| {
                eprintln!(
                    "[grpo] WARNING: failed to serialize GrpoSummary: {e}; storing null"
                );
                "null".to_string()
            });
            if let Err(e) = reg.upsert_training(
                &variant_id_for_cb,
                "grpo",
                summary.rounds,
                false,
                grpo_final_loss,
                "null",
                &summary_json,
                None,
                None,
                Some(grpo_min),
                Some(grpo_max),
                Some(&grpo_ops),
            ) {
                eprintln!(
                    "[grpo] WARNING: failed to upsert GRPO metrics for {} in smolgpt.db: {e}",
                    variant_id_for_cb
                );
            }
        };

        let summary = crate::grpo::run_grpo(
            &model,
            tokenizer.as_ref(),
            &device,
            &variant_path,
            block_size,
            grpo_rounds,
            grpo_prompts,
            grpo_group,
            grpo_temperature,
            grpo_lr,
            grpo_min,
            grpo_max,
            &grpo_ops,
            grpo_mode,
            grpo_clip_eps,
            grpo_kl_beta,
            grpo_epochs,
            seed,
            eval_samples,
            Some(&mut on_round),
        )?;
        // The final on_round callback already upserted the complete summary.
        let grpo_final_loss = summary
            .per_round_losses
            .iter()
            .rev()
            .find_map(|&l| l)
            .unwrap_or(0.0);
        println!(
            "GRPO complete: {} rounds, final PG loss {:.6}",
            summary.rounds, grpo_final_loss
        );
    }

    if quantize {
        // Derive a variant path so the base `.bin` is preserved and the
        // quantized copy is a separate file — same "no override" rule
        // `--rft`/`--grpo` follow. `model` was already loaded (in f32) from
        // `model_path` (the base) above; it is NOT mutated here, only its
        // in-memory weights are quantized and written out to `variant_path`.
        let variant_path = crate::registry::derive_variant_path(&model_path, "quant");
        let base_id = crate::registry::derive_id(&model_path);
        let variant_id = crate::registry::derive_id(&variant_path);
        println!(
            "Quantizing {model_type:?} model (base {}) → variant {}",
            model_path.display(),
            variant_path.display()
        );

        model.save_quantized(&variant_path)?;

        let base_size = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
        let quant_size = std::fs::metadata(&variant_path).map(|m| m.len()).unwrap_or(0);
        let reduction_pct = if base_size > 0 {
            100.0 * (1.0 - (quant_size as f64 / base_size as f64))
        } else {
            0.0
        };
        println!(
            "Quantization complete: {} ({base_size} bytes) -> {} ({quant_size} bytes), \
             {reduction_pct:.1}% smaller",
            model_path.display(),
            variant_path.display()
        );

        // Re-run the exact training-corpus accuracy check against the
        // quantized variant (loaded fresh from disk, so this exercises the
        // real dequantize-on-load path, not just the in-memory `model`) so
        // the registered note carries a real "quantized accuracy" number
        // rather than assuming it's unchanged from the base.
        let quantized_model = LanguageModel::load(
            model_type,
            &variant_path,
            block_size,
            vocab_size,
            hidden_size,
            &num_heads,
            num_blocks,
            tie_embeddings,
            &device,
        )?;
        let (quant_correct, quant_total) = crate::eval::eval_corpus_exact(
            &quantized_model,
            tokenizer.as_ref(),
            &device,
            &corpus,
            block_size,
        )?;
        let quant_pct = if quant_total > 0 {
            quant_correct as f64 * 100.0 / quant_total as f64
        } else {
            0.0
        };
        println!(
            "Quantized model training-set accuracy: {quant_correct}/{quant_total} ({quant_pct:.2}%)"
        );

        let quant_meta = training_meta.with_variant(&variant_path, &base_id);
        let note = format!(
            "INT8-quantized copy of '{base_id}' (per-tensor symmetric scale = \
             max(abs(x))/127, values stored as i8, dequantized to f32 on load; \
             custom binary format, see src/quantize.rs). File size: {base_size} \
             -> {quant_size} bytes ({reduction_pct:.1}% smaller). Training-set \
             exact-match accuracy: {quant_correct}/{quant_total} ({quant_pct:.2}%) \
             vs the unquantized base."
        );

        match crate::registry::Registry::open() {
            Ok(reg) => {
                let placeholder = crate::model::TrainOutcome::placeholder();
                let mut rec = crate::registry::ModelRecord::from_training(&quant_meta, &placeholder);
                rec.note = note;
                match reg.register_model(&rec) {
                    Ok(()) => println!(
                        "Registered quantized variant {} (base {}) in smolgpt.db",
                        variant_id, base_id
                    ),
                    Err(e) => eprintln!(
                        "[quantize] WARNING: failed to register quantized variant {variant_id} \
                         in smolgpt.db: {e}"
                    ),
                }
                if let Err(e) = reg.upsert_training(
                    &variant_id,
                    "sft",
                    0,
                    false,
                    0.0,
                    "[]",
                    "null",
                    Some(quant_correct as i64),
                    Some(quant_total as i64),
                    None,
                    None,
                    None,
                ) {
                    eprintln!(
                        "[quantize] WARNING: failed to upsert training accuracy for \
                         {variant_id} in smolgpt.db: {e}"
                    );
                }
            }
            Err(e) => eprintln!("[quantize] WARNING: failed to open smolgpt.db: {e}"),
        }
    }

    Ok(())
}
