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

use candle_core::{Device, Shape, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    dataset::Dataset,
    error::SmolResult,
    eval,
    model::LanguageModel,
    tokenizer::Tokenizer,
};

/// Run the RFT loop on `model` for `rounds` rounds. The model is SFT'd in-place
/// (its `VarMap` is mutated and saved to `model_path` every 10 epochs + at the
/// end of each round's SFT phase), so the caller must pass a model loaded from
/// `model_path` and (for reproducibility) a seeded `seed`.
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
    min: i64,
    max: i64,
    seed: Option<u64>,
) -> SmolResult<()> {
    if min > max {
        return Err(crate::error::SmolError::invalid_argument(&format!(
            "--rft-min ({min}) must be <= --rft-max ({max})"
        )));
    }
    if rounds == 0 {
        println!("RFT: 0 rounds requested, nothing to do");
        return Ok(());
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

    // Track per-round stats for the final trajectory summary.
    let mut winner_counts: Vec<usize> = Vec::with_capacity(rounds);
    let mut winner_rates: Vec<f64> = Vec::with_capacity(rounds);
    let mut eval_correct: Vec<f64> = Vec::with_capacity(rounds);

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
            let op: char = if rng.random_bool(0.5) { '+' } else { '-' };
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
                let parsed = eval::parse_leading_int(&decoded);
                if parsed == Some(true_answer) {
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
            model.train_with_dropout(&mut dataset, &model_path_buf, sft_epochs, 64, false, 0, 0.0)?;
        }

        // Phase 3: eval greedy correctness on held-out problems. Run even when
        // SFT was skipped so the trajectory has a per-round eval sample.
        let report = eval::run_eval(
            model,
            tokenizer,
            device,
            200,
            min,
            max,
            block_size,
            seed.map(|s| s.wrapping_add(round as u64).wrapping_add(1)),
        )?;
        let pct = if report.total > 0 {
            report.correct as f64 * 100.0 / report.total as f64
        } else {
            0.0
        };
        eval_correct.push(pct);
    }

    // Final trajectory summary so the user can see whether RFT is improving
    // the model across rounds.
    println!("\n=== RFT summary ===");
    println!(
        "{:<8} {:<10} {:<14} {:<14}",
        "round", "winners", "winner_rate%", "eval_correct%"
    );
    for (i, (&w, &r)) in winner_counts.iter().zip(winner_rates.iter()).enumerate() {
        let eval_pct = eval_correct.get(i).copied().unwrap_or(0.0);
        println!(
            "{:<8} {:<10} {:<14.2} {:<14.2}",
            i + 1,
            w,
            r,
            eval_pct,
        );
    }

    Ok(())
}
