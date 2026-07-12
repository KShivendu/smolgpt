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
        patience,
        min_delta,
        model_type,
        tokenizer: tokenizer_type,
        vocab_size: target_vocab_size,
        seed,
        block_size,
        hidden_size,
        num_heads,
        num_blocks,
        num_batches,
        serve,
        port,
        host,
    } = args;

    if !train && !generate && !eval && !rft && !grpo && !serve {
        return Err(SmolError::invalid_argument(
            "Either --train, --generate, --eval, --rft, --grpo, or --serve must be specified",
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
        }
    });

    let vocab_size = tokenizer.vocab_size();

    // Eval-only / RFT-only need just the tokenizer (built above) + a loaded
    // model. Skip the encoded-corpus tensor and Dataset construction in those
    // cases to keep them fast and avoid requiring the corpus to be encoded.
    // (The tokenizer still needs the corpus *string* for vocab scanning, which
    // is why `corpus` is loaded unconditionally above.) `--serve` is handled
    // earlier and never reaches this point, so it isn't mentioned here.
    let only_eval = eval && !train && !generate && !rft && !grpo;
    let only_rft = rft && !train && !generate && !eval && !grpo;
    let only_grpo = grpo && !train && !generate && !eval && !rft;

    let mut dataset: Option<Dataset> = None;
    if !only_eval && !only_rft && !only_grpo {
        let encoded_corpus = tokenizer.encode(&corpus);
        let encoded_corpus_len = encoded_corpus.len();
        let data = Tensor::from_vec(encoded_corpus, Shape::from(encoded_corpus_len), &device)?;
        println!(
            "Encoded text tensor shape: {:?}; dtype {:?}",
            data.shape(),
            data.dtype()
        );
        dataset = Some(Dataset::with_rng(data, 0.9, rng.clone())?);
    }

    let num_batches = num_batches;

    // --eval / --rft / --grpo never train from scratch, so the model file must
    // already exist on disk. (RFT does SFT *on the winners*, but it must start
    // from a pretrained model — there's no point sampling completions from a
    // freshly initialized model. GRPO likewise needs a pretrained policy to
    // sample completions worth scoring.)
    if (eval || rft || grpo) && !model_path.exists() {
        return Err(SmolError::invalid_argument(&format!(
            "{} requires an existing model file at {}; train first",
            if grpo { "--grpo" } else if rft { "--rft" } else { "--eval" },
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
            num_heads,
            num_blocks,
            &device,
        )?
    } else {
        println!("Creating new {model_type:?} model");
        LanguageModel::new(
            model_type,
            block_size,
            vocab_size,
            hidden_size,
            num_heads,
            num_blocks,
            &device,
        )?
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
        num_heads,
        num_blocks,
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
            if let Err(e) = reg.upsert_training(
                id,
                "sft",
                outcome.epochs_run,
                outcome.early_stopped,
                outcome.final_loss,
                &loss_json,
                "null",
            ) {
                eprintln!(
                    "[train] WARNING: failed to upsert training metrics for {id} in smolgpt.db: {e}"
                );
            }
        };

        // Dropout on for regular SFT; pass CLI patience/min_delta so `--train`
        // gets early stopping by default (patience=200, min_delta=0.001). Users
        // can disable with `--patience 0`.
        let outcome = model.train_with_dropout(
            dataset,
            &model_path,
            epochs,
            num_batches,
            true,
            patience,
            min_delta,
            Some(&mut on_checkpoint),
        )?;
        println!("Training completed in {:.2?}", now.elapsed());

        // Final register_model upserts the model row with the complete
        // outcome (epochs_run, early_stopped, final note). The final
        // on_checkpoint callback already upserted the trainings row inside
        // train_with_dropout, so the loss trajectory is current.
        if let Ok(reg) = &reg {
            let rec = crate::registry::ModelRecord::from_training(&training_meta, &outcome);
            match reg.register_model(&rec) {
                Ok(()) => println!("Registered model {} in smolgpt.db", rec.id),
                Err(e) => eprintln!(
                    "[train] WARNING: failed to register model in smolgpt.db: {e}"
                ),
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
        let variant_path =
            crate::registry::derive_variant_path(&model_path, "grpo");
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

    Ok(())
}
