//! GRPO-lite (Group Relative Policy Optimization, lite) for arithmetic.
//!
//! GRPO = sample a *group* of G completions per prompt, score each with a
//! reward, compute group-relative advantages `a_i = (r_i - mean_r) / std_r`,
//! and take a policy-gradient step: `loss = -mean_i(a_i * logp(c_i | prompt))`.
//! The gradient pushes UP the logprob of above-average (correct) completions
//! and DOWN the logprob of below-average (wrong) ones. The crucial difference
//! from RFT: wrong completions carry corrective gradient signal — RFT discards
//! them, GRPO uses them.
//!
//! "lite" = group-relative advantages (the part that removes the need for a
//! learned value function) but NO PPO-style ratio clipping and NO KL penalty to
//! a reference model. Those matter most for large models; for a 7K char model
//! they're engineering overhead without payoff. The core "use the negatives"
//! signal is what we're after.
//!
//! Per round:
//!   1. For each of P prompts (`a op b=`), sample G completions at
//!      `temperature`, reward each 1.0/0.0 (exact arithmetic correctness).
//!   2. One `grpo_step` per prompt: group-relative advantage + policy-gradient
//!      backward + AdamW step. (Groups with uniform reward → no step.)
//!   3. Save the model.
//!   4. Eval greedy correctness on held-out problems so the user can see the
//!      trajectory round-by-round.
//!   5. Repeat for N rounds.
//!
//! Save target: `run_grpo` saves to the `model_path` the caller passes. The
//! caller (`train.rs --grpo`) passes a derived variant path
//! (`<base-stem>-grpo.bin`) so the base `.bin` the model was loaded from is
//! preserved and the variant is a separate file. The old behavior of
//! overwriting the base in place is gone — GRPO no longer clobbers the
//! pretrained checkpoint, so a base model and its GRPO variant can coexist as
//! separate registry rows linked by `base_model_id`.
//!
//! The model is mutated in-place (its `VarMap` is updated by the optimizer and
//! saved to `model_path` each round), so the caller must pass a model loaded
//! from a pretrained `.bin` and a seeded `seed` for reproducibility.

use candle_core::Device;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{
    args::GrpoMode,
    error::SmolResult,
    eval, model::LanguageModel, tokenizer::Tokenizer,
};

/// Per-round + final-trajectory summary of a GRPO run, returned by `run_grpo`
/// so `train.rs` can persist it into the `trainings` table for the web UI.
/// All Vecs are parallel, indexed by round (0-based in storage, displayed
/// 1-based in the UI).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GrpoSummary {
    pub rounds: usize,
    pub group_size: usize,
    /// Which GRPO variant produced this run: `"lite"` (REINFORCE + group
    /// baseline, single on-policy step) or `"full"` (PPO-style: importance
    /// ratio + clipping + KL-to-reference + K mini-epochs). `#[serde(default)]
    /// on the UI's view struct fills in `"lite"` for pre-mode DB rows, so
    /// old GRPO rows deserialize cleanly.
    #[serde(default = "default_grpo_mode")]
    pub mode: String,
    /// Per-round mean reward (fraction of the G*P completions that were
    /// correct) as a percentage. Analogous to RFT's winner_rate but over ALL
    /// sampled completions, not just first-correct.
    pub correct_rates: Vec<f64>,
    /// Per-round greedy-decoding correctness (%) from `run_eval`.
    pub eval_correct_pct: Vec<f64>,
    /// Per-round mean policy-gradient loss across the prompts that stepped.
    /// `None` when every group that round had uniform reward (all-correct or
    /// all-wrong), so no prompt actually stepped the optimizer that round —
    /// distinct from `Some(0.0)`, which means steps were taken and the mean
    /// loss genuinely came out to zero.
    pub per_round_losses: Vec<Option<f32>>,
}

/// Serde default for `GrpoSummary::mode` — `"lite"`, since all GRPO runs
/// before the `mode` field existed were lite. Used via
/// `#[serde(default = "default_grpo_mode")]` so pre-mode DB rows (and the
/// `GrpoSummaryView` parse path) deserialize to `"lite"` instead of `""`.
fn default_grpo_mode() -> String {
    "lite".to_string()
}

/// RAII guard that deletes a temp file when dropped. Used by GRPO-full to
/// clean up the frozen-reference snapshot on BOTH the Ok and Err return
/// paths (and on panic) without threading explicit cleanup through every
/// return point. Best-effort: a failed `remove_file` is silently ignored
/// (the file is in `std::env::temp_dir()`, so the OS will reap it anyway).
struct TempFileGuard {
    path: std::path::PathBuf,
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Run the GRPO-lite loop on `model` for `rounds` rounds. The model is updated
/// in-place and saved to `model_path` after each round. See the module doc for
/// the save-target semantics: the caller passes a derived variant path so the
/// base `.bin` is preserved.
///
/// `on_round`: when `Some(cb)`, the callback is invoked after each round's
/// eval with a partial `GrpoSummary` (rounds completed so far). `train.rs`
/// fills this with a closure that upserts the partial summary into the
/// `trainings` table so the web UI shows live per-round progress. The final
/// round's callback upserts the complete summary.
#[allow(clippy::too_many_arguments)]
pub fn run_grpo(
    model: &LanguageModel,
    tokenizer: &dyn Tokenizer<u32>,
    device: &Device,
    model_path: &std::path::PathBuf,
    block_size: usize,
    rounds: usize,
    prompts_per_round: usize,
    group_size: usize,
    temperature: f32,
    lr: f64,
    min: i64,
    max: i64,
    ops: &str,
    mode: GrpoMode,
    clip_eps: f64,
    kl_beta: f64,
    k_epochs: usize,
    seed: Option<u64>,
    eval_samples: usize,
    mut on_round: Option<&mut dyn FnMut(&GrpoSummary)>,
) -> SmolResult<GrpoSummary> {
    let mode_str: String = if mode == GrpoMode::Full {
        "full".to_string()
    } else {
        "lite".to_string()
    };
    if min > max {
        return Err(crate::error::SmolError::invalid_argument(&format!(
            "--grpo-min ({min}) must be <= --grpo-max ({max})"
        )));
    }
    if group_size < 2 {
        return Err(crate::error::SmolError::invalid_argument(
            "--grpo-group must be >= 2 (need a group to compute relative advantage)",
        ));
    }
    if rounds == 0 {
        println!("GRPO: 0 rounds requested, nothing to do");
        return Ok(GrpoSummary {
            rounds: 0,
            group_size,
            mode: mode_str,
            correct_rates: Vec::new(),
            eval_correct_pct: Vec::new(),
            per_round_losses: Vec::new(),
        });
    }

    let ops_list = eval::parse_ops(ops)?;

    // Newline stop token — same convention as eval/RFT.
    let newline_token: u32 = tokenizer
        .encode("\n")
        .into_iter()
        .next()
        .ok_or_else(|| {
            crate::error::SmolError::invalid_argument(
                "Tokenizer produced no encoding for '\\n' (GRPO needs a newline stop token)",
            )
        })?;

    // One optimizer persists across all rounds (so momentum/variance state
    // carries forward). The model's VarMap is mutated in-place by each step.
    let mut optimizer = model.make_optimizer(lr)?;

    // GRPO-full: snapshot a FROZEN reference copy of the policy before any
    // rounds run. The reference shares NO VarMap state with `model`, so the
    // optimizer never touches it and `ref_logp` values stay constant across
    // the K mini-epochs. The snapshot lives in a temp file (cleaned up by
    // `TempFileGuard` on scope exit, including the Err path). Lite mode
    // skips this entirely (no ratio/clip/KL, so no reference needed).
    //
    // The guard is created BEFORE `snapshot` runs, so if `snapshot` errors
    // out (via `?`), the guard's Drop still fires on the early return and
    // reaps the temp file `save` wrote. The guard lives till the end of
    // `run_grpo`, so cleanup happens on both the Ok and Err return paths.
    let ref_tmp: std::path::PathBuf = if mode == GrpoMode::Full {
        std::env::temp_dir().join(format!("smolgpt-grpo-ref-{}.bin", std::process::id()))
    } else {
        std::path::PathBuf::new()
    };
    let _ref_guard: Option<TempFileGuard> = if mode == GrpoMode::Full {
        println!("GRPO-full: reference snapshot at {}", ref_tmp.display());
        Some(TempFileGuard {
            path: ref_tmp.clone(),
        })
    } else {
        None
    };
    let reference: Option<LanguageModel> = if mode == GrpoMode::Full {
        Some(model.snapshot(&ref_tmp, device)?)
    } else {
        None
    };

    let mut correct_rates: Vec<f64> = Vec::with_capacity(rounds);
    let mut eval_correct: Vec<f64> = Vec::with_capacity(rounds);
    let mut per_round_losses: Vec<Option<f32>> = Vec::with_capacity(rounds);

    println!(
        "GRPO-{mode_str}: {rounds} rounds, {prompts_per_round} prompts/round, G={group_size}, \
         T={temperature}, lr={lr}, operands in [{min}, {max}], ops [{}], block_size={block_size}{}",
        ops_list.iter().collect::<String>(),
        if mode == GrpoMode::Full {
            format!(
                ", clip_eps={clip_eps}, kl_beta={kl_beta}, k_epochs={k_epochs}"
            )
        } else {
            String::new()
        },
    );

    for round in 0..rounds {
        println!("\n=== GRPO round {}/{rounds} ===", round + 1);

        // Seed per-round RNG from `seed` + round for reproducibility.
        let mut rng: StdRng = match seed {
            Some(s) => StdRng::seed_from_u64(s.wrapping_add(round as u64)),
            None => StdRng::from_os_rng(),
        };

        let mut total_rewards: usize = 0;
        let mut total_samples: usize = 0;
        let mut loss_sum: f32 = 0.0;
        let mut steps_taken: usize = 0;

        for _ in 0..prompts_per_round {
            let a: i64 = rng.random_range(min..=max);
            let b: i64 = rng.random_range(min..=max);
            let op: char = ops_list[rng.random_range(0..ops_list.len())];
            let true_answer: i64 = if op == '+' { a + b } else { a - b };

            let prompt_str = format!("{a}{op}{b}=");
            let prompt_ids = tokenizer.encode(&prompt_str);

            // Sample G completions for this prompt and score them.
            let mut completions: Vec<Vec<u32>> = Vec::with_capacity(group_size);
            let mut rewards: Vec<f32> = Vec::with_capacity(group_size);
            for _ in 0..group_size {
                let sampled = model.sample_from_prompt(
                    &prompt_ids,
                    block_size,
                    newline_token,
                    temperature,
                    &mut rng,
                    device,
                )?;
                let decoded = tokenizer.decode(&sampled);
                // Clean-stop reward: the completion must be exactly "{answer}"
                // followed by the newline stop — no rambling like "2+2=3". We
                // strip the trailing newline(s) the sampler appends, then the
                // remainder must parse to exactly the true answer. A leading-int
                // match alone (e.g. "2+2=3" for true 2) is NOT enough: that
                // teaches the model to emit the right digit then keep generating,
                // which is what produced the garbled REPL output.
                let trimmed = decoded.trim_end_matches('\n');
                let r: f32 = if !trimmed.is_empty()
                    && trimmed.parse::<i64>().ok() == Some(true_answer)
                {
                    1.0
                } else {
                    0.0
                };
                completions.push(sampled);
                rewards.push(r);
            }

            let rewards_sum: f32 = rewards.iter().sum();
            total_rewards += rewards_sum as usize;
            total_samples += group_size;

            // Group-relative advantage, computed once per group from the
            // sampled rewards. Both modes use the same advantage formula
            // (the "GRPO" part); they differ in what they do with it. A
            // uniform-reward group (std≈0) carries no relative-advantage
            // signal, so we skip the step entirely — for lite, `grpo_step`
            // re-derives advantages internally and applies the same guard;
            // for full, we skip here to avoid the wasted `old_logp`/
            // `ref_logp` scalar forward passes (and `grpo_step_full` would
            // return 0.0 anyway via its own all-zero-advantage guard).
            let eps = 1e-6f32;
            let mean_r = rewards_sum / group_size as f32;
            let var = rewards.iter().map(|r| (r - mean_r).powi(2)).sum::<f32>()
                / group_size as f32;
            let std_r = var.sqrt();
            let uniform = std_r < eps;

            let loss: f32 = if uniform {
                0.0
            } else if mode == GrpoMode::Full {
                // Full (PPO-style): cache `old_logp` (under the sampling
                // policy) and `ref_logp` (under the frozen reference) ONCE
                // per group, then run K mini-epochs of ratio/clip/KL via
                // `grpo_step_full`. Only `logp_theta` is recomputed each
                // mini-epoch; the cached scalars are reused.
                let advantages: Vec<f64> = rewards
                    .iter()
                    .map(|r| ((r - mean_r) / (std_r + eps)) as f64)
                    .collect();
                let mut old_logps: Vec<f64> = vec![0.0; group_size];
                let mut ref_logps: Vec<f64> = vec![0.0; group_size];
                let reference = reference.as_ref().expect(
                    "GRPO-full requires a reference snapshot (mode == Full \
                     but `reference` is None — this is a bug in run_grpo)",
                );
                for i in 0..group_size {
                    old_logps[i] = model.completion_logp_scalar(
                        &prompt_ids,
                        &completions[i],
                        device,
                    )?;
                    ref_logps[i] = reference.completion_logp_scalar(
                        &prompt_ids,
                        &completions[i],
                        device,
                    )?;
                }
                model.grpo_step_full(
                    &prompt_ids,
                    &completions,
                    &advantages,
                    &old_logps,
                    &ref_logps,
                    clip_eps,
                    kl_beta,
                    k_epochs,
                    &mut optimizer,
                    device,
                )?
            } else {
                // Lite: REINFORCE with group-baseline advantage, single
                // on-policy step. `grpo_step` recomputes advantages from
                // `rewards` internally (same formula as above) and applies
                // its own std≈0 guard, so passing `rewards` here is
                // equivalent to passing the precomputed advantages.
                model.grpo_step(&prompt_ids, &completions, &rewards, &mut optimizer, device)?
            };
            // 0.0 unconditionally means "no step taken" (uniform-reward
            // group or all-zero advantages), per both steps' contracts, so
            // this comparison is exact.
            if loss != 0.0 {
                loss_sum += loss;
                steps_taken += 1;
            }
        }

        // Save after each round so progress survives a crash.
        model.save(model_path)?;

        let correct_rate = if total_samples > 0 {
            total_rewards as f64 * 100.0 / total_samples as f64
        } else {
            0.0
        };
        let mean_loss: Option<f32> = if steps_taken > 0 {
            Some(loss_sum / steps_taken as f32)
        } else {
            None
        };

        // Greedy eval on held-out problems (same as RFT) for the trajectory.
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
        let eval_pct = report.correct as f64 * 100.0 / report.total as f64;

        let mean_loss_str = mean_loss
            .map(|l| format!("{l:.6}"))
            .unwrap_or_else(|| "skipped".to_string());
        println!(
            "Round {}/{rounds}: prompts={prompts_per_round}, steps={steps_taken}, \
             correct={total_rewards}/{total_samples} ({correct_rate:.2}%), \
             mean_pg_loss={mean_loss_str}, greedy_eval={eval_pct:.2}% ({}/{})",
            round + 1,
            report.correct,
            report.total,
        );

        correct_rates.push(correct_rate);
        eval_correct.push(eval_pct);
        per_round_losses.push(mean_loss);

        // Fire the on_round callback with the partial summary (rounds
        // completed so far) so `train.rs` can upsert it into the `trainings`
        // table for live per-round progress in the web UI. The final round's
        // callback upserts the complete summary.
        if let Some(cb) = on_round.as_mut() {
            cb(&GrpoSummary {
                rounds: round + 1,
                group_size,
                mode: mode_str.clone(),
                correct_rates: correct_rates.clone(),
                eval_correct_pct: eval_correct.clone(),
                per_round_losses: per_round_losses.clone(),
            });
        }
    }

    println!("\n=== GRPO summary ===");
    println!(
        "{:<8} {:<14} {:<14} {:<14}",
        "round", "correct%", "eval%", "pg_loss"
    );
    for i in 0..rounds {
        let loss_str = per_round_losses[i]
            .map(|l| format!("{l:.6}"))
            .unwrap_or_else(|| "skipped".to_string());
        println!(
            "{:<8} {:<14.2} {:<14.2} {:<14}",
            i + 1,
            correct_rates[i],
            eval_correct[i],
            loss_str,
        );
    }

    Ok(GrpoSummary {
        rounds,
        group_size,
        mode: mode_str,
        correct_rates,
        eval_correct_pct: eval_correct,
        per_round_losses,
    })
}
