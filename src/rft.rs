//! Rejection sampling Fine-Tuning (RFT) for arithmetic.
//!
//! RFT = "sample completions with temperature -> keep the correct ones ->
//! SFT the model on the winners -> eval -> repeat." The reward (exact
//! arithmetic correctness) is only a *filter* — it never enters the gradient.
//! The actual learning is plain supervised next-token cross-entropy (the same
//! `LanguageModel::train` path used by `--train`), which is what makes this
//! "rejection sampling" fine-tuning rather than a policy-gradient method like
//! PPO.
//!
//! Per round:
//!   1. Sample P random prompts (`a op b=` for `a, b` in `[min, max]`,
//!      `op in {+, -}`).
//!   2. For each prompt, sample up to K completions at `temperature` from the
//!      model. Keep the first one whose parsed answer equals the true answer.
//!      The SFT target uses the *true* answer (not the model's possibly-noisy
//!      string) so the gradient signal is clean.
//!   3. Concatenate the winner lines (`a op b=true_answer\n`) into a corpus,
//!      encode it, build a `Dataset`, and SFT the model in-place for E epochs.
//!   4. Eval greedy correctness on held-out problems so the user can see the
//!      trajectory round-by-round.
//!   5. Save and repeat for N rounds.
//!
//! Save target: `run_rft` saves to the `model_path` the caller passes. The
//! caller (`train.rs --rft`) passes a derived variant path
//! (`<base-stem>-rft.bin`) so the base `.bin` the model was loaded from is
//! preserved and the variant is a separate file. The old behavior of
//! overwriting the base in place is gone — RFT no longer clobbers the
//! pretrained checkpoint, so a base model and its RFT variant can coexist as
//! separate registry rows linked by `base_model_id`.

use candle_core::{Device, Shape, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{
    dataset::Dataset,
    error::SmolResult,
    eval,
    model::LanguageModel,
    tokenizer::Tokenizer,
};

/// Per-round + final-trajectory summary of an RFT run, returned by `run_rft`
/// so `train.rs` can persist it into the `trainings` table for the web UI.
/// `winner_counts`, `winner_rates`, `eval_correct_pct`, and
/// `per_round_sft_final_losses` are all parallel Vecs indexed by round
/// (0-based in storage, displayed 1-based in the UI). When a round produced
/// no winners (so SFT was skipped), `per_round_sft_final_losses[i]` is `None`
/// — that distinguishes "no SFT ran" from "SFT ran and ended at loss X".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RftSummary {
    pub rounds: usize,
    pub winner_counts: Vec<usize>,
    pub winner_rates: Vec<f64>,
    pub eval_correct_pct: Vec<f64>,
    pub per_round_sft_final_losses: Vec<Option<f32>>,
}

/// Run the RFT loop on `model` for `rounds` rounds. The model is SFT'd
/// in-place (its `VarMap` is mutated and saved to `model_path` every 10 epochs
/// + at the end of each round's SFT phase), so the caller must pass a model
/// loaded from a pretrained `.bin` and (for reproducibility) a seeded `seed`.
///
/// `model_path` is the SAVE target — the caller (`train.rs --rft`) passes a
/// derived variant path (e.g. `<base>-rft.bin`) so the base `.bin` the model
/// was loaded from is preserved and the variant is a separate file. The old
/// behavior of overwriting the base in place is gone — RFT no longer clobbers
/// the pretrained checkpoint.
///
/// `on_round`: when `Some(cb)`, the callback is invoked after each round's
/// eval with a partial `RftSummary` (rounds completed so far). `train.rs`
/// fills this with a closure that upserts the partial summary into the
/// `trainings` table so the web UI shows live per-round progress (reload the
/// page mid-RFT to see the latest round). The final call (after the last
/// round) upserts the complete summary. When `None`, no per-round callback
/// fires — the caller gets the final summary via the return value and records
/// it once at the end.
#[allow(clippy::too_many_arguments)]
pub fn run_rft(
    model: &LanguageModel,
    tokenizer: &dyn Tokenizer<u32>,
    device: &Device,
    model_path: &std::path::Path,
    block_size: usize,
    rounds: usize,
    prompts_per_round: usize,
    samples_per_prompt: usize,
    temperature: f32,
    sft_epochs: usize,
    sft_num_batches: usize,
    min: i64,
    max: i64,
    ops: &str,
    seed: Option<u64>,
    eval_samples: usize,
    mut on_round: Option<&mut dyn FnMut(&RftSummary)>,
) -> SmolResult<RftSummary> {
    if min > max {
        return Err(crate::error::SmolError::invalid_argument(&format!(
            "--rft-min ({min}) must be <= --rft-max ({max})"
        )));
    }
    if rounds == 0 {
        println!("RFT: 0 rounds requested, nothing to do");
        return Ok(RftSummary {
            rounds: 0,
            winner_counts: Vec::new(),
            winner_rates: Vec::new(),
            eval_correct_pct: Vec::new(),
            per_round_sft_final_losses: Vec::new(),
        });
    }

    // Find the newline token id the same way eval does: encode "\n" and take
    // the first token. For the char tokenizer this is the lowest-codepoint
    // char (newline); for byte-level BPE it's byte 10. Both yield one token.
    let newline_token: u32 = tokenizer
        .encode("\n")
        .into_iter()
        .next()
        .ok_or_else(|| {
            crate::error::SmolError::invalid_argument(
                "Tokenizer produced no encoding for '\\n' (RFT needs a newline stop token)",
            )
        })?;

    println!(
        "RFT: {rounds} rounds, {prompts_per_round} prompts/round, \
         {samples_per_prompt} samples/prompt, T={temperature}, \
         {sft_epochs} SFT epochs/round, operands in [{min}, {max}], \
         block_size={block_size}, newline_token={newline_token}"
    );

    let ops_list: Vec<char> = eval::parse_ops(ops)?;

    // Track per-round stats for the final trajectory summary.
    let mut winner_counts: Vec<usize> = Vec::with_capacity(rounds);
    let mut winner_rates: Vec<f64> = Vec::with_capacity(rounds);
    let mut eval_correct: Vec<f64> = Vec::with_capacity(rounds);
    // Final SFT loss of each round (`None` when SFT was skipped because no
    // winners were sampled that round). Surfaced in the UI as a per-round
    // trajectory column so the user can see whether RFT is converging the SFT
    // step on the winners corpus.
    let mut per_round_sft_final_losses: Vec<Option<f32>> = Vec::with_capacity(rounds);

    for round in 0..rounds {
        println!("\n=== RFT round {}/{rounds} ===", round + 1);

        // Seed the prompt RNG deterministically from `seed` + round index so
        // the same `--seed` reproduces the same prompts and the same sampling
        // trajectory across runs. If `seed` is `None`, fall back to OS entropy
        // (non-reproducible but never collides with a saved seed).
        let mut rng: StdRng = match seed {
            Some(s) => StdRng::seed_from_u64(s.wrapping_add(round as u64)),
            None => StdRng::from_os_rng(),
        };

        // Phase 1: sample completions and collect winners.
        let mut winners: Vec<String> = Vec::with_capacity(prompts_per_round);
        for _ in 0..prompts_per_round {
            let a: i64 = rng.random_range(min..=max);
            let b: i64 = rng.random_range(min..=max);
            let op: char = ops_list[rng.random_range(0..ops_list.len())];
            let true_answer: i64 = if op == '+' { a + b } else { a - b };

            let prompt_str = format!("{a}{op}{b}=");
            let prompt_ids = tokenizer.encode(&prompt_str);

            // Try up to K samples; stop on the first correct one. We only
            // need one winner per prompt — multiple winners for the same
            // prompt would just upweight that exact line in the SFT corpus
            // without adding information.
            for _ in 0..samples_per_prompt {
                let sampled = model.sample_from_prompt(
                    &prompt_ids,
                    block_size,
                    newline_token,
                    temperature,
                    &mut rng,
                    device,
                )?;
                let decoded = tokenizer.decode(&sampled);
                // Clean-stop match: the completion must be exactly the true
                // answer followed by the newline stop — no rambling like
                // "2+2=3junk". A leading-int match alone (e.g. via
                // `parse_leading_int`) is NOT enough: it would count "3junk"
                // as a winner and teach the model to emit the right digit
                // then keep generating, which produces exactly the garbled
                // REPL output GRPO's matching reward was written to avoid.
                // Kept in sync with GRPO's identical criterion in grpo.rs.
                let trimmed = decoded.trim_end_matches('\n');
                if !trimmed.is_empty() && trimmed.parse::<i64>().ok() == Some(true_answer) {
                    // Use the true answer (not the model's string) as the SFT
                    // target: the model's decoded answer may have trailing
                    // junk after the newline-stop, and we want clean targets.
                    winners.push(format!("{a}{op}{b}={true_answer}\n"));
                    break;
                }
            }
        }

        let winners_count = winners.len();
        let winner_rate = if prompts_per_round > 0 {
            winners_count as f64 * 100.0 / prompts_per_round as f64
        } else {
            0.0
        };
        let winners_corpus: String = winners.concat();
        let winners_corpus_tokens = tokenizer.encode(&winners_corpus).len();

        println!(
            "Round {}/{rounds}: prompts={prompts_per_round}, \
             winners={winners_count} ({winner_rate:.2}%), \
             winners_corpus_tokens={winners_corpus_tokens}",
            round + 1,
        );

        winner_counts.push(winners_count);
        winner_rates.push(winner_rate);

        // Phase 2: SFT the model on the winners (skip if nothing was won).
        if winners_count == 0 {
            println!(
                "Round {}/{rounds}: no winners this round — skipping SFT \
                 (nothing to train on), running eval only",
                round + 1,
            );
            per_round_sft_final_losses.push(None);
        } else {
            let encoded = tokenizer.encode(&winners_corpus);
            let encoded_len = encoded.len();
            // The Dataset needs at least `block_size + 1` tokens to produce one
            // training batch (`get_random_batches` indexes `0..total-block_size`).
            // If the winners corpus is shorter than that, fall back to a single
            // tiny batch worth by padding the corpus with itself until it's
            // long enough — this is rare (only when winners are very few and
            // each line is short) and just lets the SFT loop make progress.
            let encoded = if encoded_len <= block_size {
                let mut padded = encoded.clone();
                while padded.len() <= block_size {
                    padded.extend_from_slice(&encoded);
                }
                padded
            } else {
                encoded
            };
            let total_len = encoded.len();
            let data =
                Tensor::from_vec(encoded, Shape::from(total_len), device)?;
            // Seed the Dataset's batch-sampling RNG deterministically from the
            // user `seed` + round index so the SFT step is reproducible across
            // runs with the same `--seed`. (This is distinct from the prompt
            // RNG — both just need to be deterministic, not the same stream.)
            // Without this, two `--seed 42` runs would sample different SFT
            // batches and produce different losses / final weights, breaking
            // reproducibility. If `seed` is `None`, fall back to OS entropy.
            let dataset_rng: StdRng = match seed {
                Some(s) => StdRng::seed_from_u64(s.wrapping_add(round as u64).wrapping_add(0x9e37_79b9_7f4a_7c15)),
                None => StdRng::from_os_rng(),
            };
            let mut dataset = Dataset::with_rng(data, 0.9, dataset_rng)?;

            println!(
                "Round {}/{rounds}: SFT for {sft_epochs} epochs on winners corpus \
                 ({total_len} tokens, {} train / {} val)",
                round + 1,
                dataset.train_size,
                dataset.validation_size,
            );

            // `LanguageModel::train` takes `&std::path::PathBuf`; pass a
            // reference to a PathBuf owned by the caller (we have &Path, so
            // convert via to_path_buf to satisfy the signature). Disable
            // dropout for the RFT SFT step so two runs with the same `--seed`
            // produce identical losses / weights / winner trajectories —
            // candle-nn 0.9.1's dropout uses candle's unseedable CPU RNG.
            // Early stopping is also disabled (patience=0): RFT's per-round SFT
            // budget is short and we want the full `sft_epochs` to fit the
            // winners corpus each round.
            let model_path_buf = model_path.to_path_buf();
            let sft_outcome = model.train_with_dropout(
                &mut dataset,
                &model_path_buf,
                sft_epochs,
                sft_num_batches,
                false,
                0,
                0.0,
                // No per-checkpoint callback for the RFT sub-loop's internal
                // SFT — the per-round upsert happens via the `on_round`
                // callback below, at round granularity.
                None,
                0.001,
                false,
                None,
            )?;
            per_round_sft_final_losses.push(Some(sft_outcome.final_loss));
        }

        // Phase 3: eval greedy correctness on held-out problems. Run even when
        // SFT was skipped so the trajectory has a per-round eval sample.
        let report = eval::run_eval(
            model,
            tokenizer,
            device,
            eval_samples,
            min,
            max,
            block_size,
            seed.map(|s| s.wrapping_add(round as u64).wrapping_add(1)),
            ops,
        )?;
        let pct = if report.total > 0 {
            report.correct as f64 * 100.0 / report.total as f64
        } else {
            0.0
        };
        eval_correct.push(pct);

        // Fire the on_round callback with the partial summary (rounds
        // completed so far) so `train.rs` can upsert it into the `trainings`
        // table for live per-round progress in the web UI. The final round's
        // callback upserts the complete summary.
        if let Some(cb) = on_round.as_mut() {
            cb(&RftSummary {
                rounds: round + 1,
                winner_counts: winner_counts.clone(),
                winner_rates: winner_rates.clone(),
                eval_correct_pct: eval_correct.clone(),
                per_round_sft_final_losses: per_round_sft_final_losses.clone(),
            });
        }
    }

    // Final trajectory summary so the user can see whether RFT is improving
    // the model across rounds.
    println!("\n=== RFT summary ===");
    println!(
        "{:<8} {:<10} {:<14} {:<14} {:<14}",
        "round", "winners", "winner_rate%", "eval_correct%", "sft_final_loss"
    );
    for (i, (&w, &r)) in winner_counts.iter().zip(winner_rates.iter()).enumerate() {
        let eval_pct = eval_correct.get(i).copied().unwrap_or(0.0);
        let sft_loss = per_round_sft_final_losses
            .get(i)
            .copied()
            .flatten()
            .map(|l| format!("{l:.4}"))
            .unwrap_or_else(|| "skipped".to_string());
        println!(
            "{:<8} {:<10} {:<14.2} {:<14.2} {:<14}",
            i + 1,
            w,
            r,
            eval_pct,
            sft_loss,
        );
    }

    Ok(RftSummary {
        rounds,
        winner_counts,
        winner_rates,
        eval_correct_pct: eval_correct,
        per_round_sft_final_losses,
    })
}
