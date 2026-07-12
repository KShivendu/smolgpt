//! Greedy-decoding evaluation harness for arithmetic correctness.
//!
//! Prompts a trained `LanguageModel` with `a op b=` (op in {+, -}) and checks
//! whether its greedy (argmax) completion matches the true arithmetic answer.
//! Reports overall accuracy, breakdown by operator, and breakdown by operand
//! digit-length so failure modes are easy to spot.

use candle_core::Device;
use rand::{
    rngs::StdRng,
    Rng, SeedableRng,
};

use crate::{
    error::{SmolError, SmolResult},
    model::LanguageModel,
    tokenizer::Tokenizer,
};

/// Aggregate accuracy report for an eval run.
///
/// `by_digits` buckets samples by `max(a.abs().to_string().len(),
/// b.abs().to_string().len())`:
///   - index 0 -> 1-digit operands
///   - index 1 -> 2-digit operands
///   - index 2 -> 3-digit operands
///   - index 3 -> 4-or-more-digit operands
///
/// Each bucket holds `(correct, total)`.
///
/// `examples` holds up to 10 worked samples so HTTP/JSON consumers (the
/// `--serve` web UI) can render failure modes without scraping stdout.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EvalReport {
    pub total: usize,
    pub correct: usize,
    pub correct_plus: usize,
    pub total_plus: usize,
    pub correct_minus: usize,
    pub total_minus: usize,
    pub by_digits: [(usize, usize); 4],
    #[serde(default)]
    pub examples: Vec<EvalExample>,
}

/// One (prompt, generated_answer, true_answer, correct) sample, kept for the
/// end-of-run examples block so the user can eyeball failure modes. Public so
/// the `--serve` web UI can serialize it to JSON.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvalExample {
    pub prompt: String,
    pub generated: String,
    pub true_answer: i64,
    pub correct: bool,
}

/// Run the arithmetic eval harness.
///
/// For each of `n_samples` problems: sample `a, b` uniformly from `[min, max]`,
/// sample `op` uniformly from `{+, -}`, prompt the model with `a op b=`, greedy
/// decode up to `block_size` new tokens (stopping at the newline token), parse
/// the leading integer from the decoded completion, and compare it to the true
/// answer (`a + b` or `a - b`).
///
/// If `seed` is `Some(s)`, the problem set is generated with a freshly-seeded
/// `StdRng` so the eval is fully reproducible (greedy decoding is deterministic,
/// so two runs with the same seed produce identical output). If `seed` is
/// `None`, OS entropy is used.
pub fn run_eval(
    model: &LanguageModel,
    tokenizer: &dyn Tokenizer<u32>,
    device: &Device,
    n_samples: usize,
    min: i64,
    max: i64,
    block_size: usize,
    seed: Option<u64>,
) -> SmolResult<EvalReport> {
    if min > max {
        return Err(SmolError::invalid_argument(&format!(
            "--eval-min ({min}) must be <= --eval-max ({max})"
        )));
    }

    // Find the newline token id generically: encode "\n" and take the first
    // token. For the char tokenizer this is token 0 (newline has the lowest
    // codepoint); for byte-level BPE it's byte 10. Both yield a single token.
    let newline_token: u32 = tokenizer
        .encode("\n")
        .into_iter()
        .next()
        .ok_or_else(|| SmolError::invalid_argument("Tokenizer produced no encoding for '\\n'"))?;

    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    };

    let mut report = EvalReport::default();

    for _ in 0..n_samples {
        let a: i64 = rng.random_range(min..=max);
        let b: i64 = rng.random_range(min..=max);
        let op: char = if rng.random_bool(0.5) { '+' } else { '-' };
        let true_answer: i64 = if op == '+' { a + b } else { a - b };

        let prompt = format!("{a}{op}{b}=");
        let prompt_ids = tokenizer.encode(&prompt);

        let generated_ids = model.generate_greedy_from_prompt(
            &prompt_ids,
            block_size,
            newline_token,
            device,
        )?;
        let generated_str = tokenizer.decode(&generated_ids);
        let parsed = parse_leading_int(&generated_str);
        let correct = parsed == Some(true_answer);

        // Update aggregates.
        report.total += 1;
        if correct {
            report.correct += 1;
        }
        if op == '+' {
            report.total_plus += 1;
            if correct {
                report.correct_plus += 1;
            }
        } else {
            report.total_minus += 1;
            if correct {
                report.correct_minus += 1;
            }
        }

        let digit_len = std::cmp::max(a.abs().to_string().len(), b.abs().to_string().len());
        let bucket = match digit_len {
            1 => 0,
            2 => 1,
            3 => 2,
            _ => 3,
        };
        report.by_digits[bucket].1 += 1;
        if correct {
            report.by_digits[bucket].0 += 1;
        }

        // Keep the first 10 samples as worked examples for the user.
        if report.examples.len() < 10 {
            report.examples.push(EvalExample {
                prompt,
                generated: generated_str,
                true_answer,
                correct,
            });
        }
    }

    print_report(&report, &report.examples);
    Ok(report)
}

/// Print the eval report and up to 10 worked examples.
fn print_report(report: &EvalReport, examples: &[EvalExample]) {
    let pct = if report.total > 0 {
        report.correct as f64 * 100.0 / report.total as f64
    } else {
        0.0
    };
    println!(
        "Eval: {}/{} = {pct:.2}%",
        report.correct, report.total
    );
    println!("  + : {}/{}", report.correct_plus, report.total_plus);
    println!("  - : {}/{}", report.correct_minus, report.total_minus);
    println!(
        "  1-digit : {}/{}",
        report.by_digits[0].0, report.by_digits[0].1
    );
    println!(
        "  2-digit : {}/{}",
        report.by_digits[1].0, report.by_digits[1].1
    );
    println!(
        "  3-digit : {}/{}",
        report.by_digits[2].0, report.by_digits[2].1
    );
    println!(
        "  4+ digit: {}/{}",
        report.by_digits[3].0, report.by_digits[3].1
    );

    if examples.is_empty() {
        return;
    }
    println!("\nExamples:");
    for ex in examples {
        let mark = if ex.correct { "ok" } else { "FAIL" };
        // Trim the generated string at the first newline (if any) for display
        // compactness — the newline is the stop token, so anything after it is
        // post-stop noise the harness ignored.
        let gen_display: String = ex
            .generated
            .split('\n')
            .next()
            .unwrap_or("")
            .to_string();
        println!("  [{mark}] {}{} (true: {})", ex.prompt, gen_display, ex.true_answer);
    }
}

/// Parse a leading optional `-` followed by digits from `s`. Stops at the first
/// non-digit (after an optional leading `-`). Returns `None` if no digits are
/// present (e.g. the model emitted garbage instead of a number).
///
/// Public so the RFT loop can reuse the exact same parser the eval harness
/// uses to judge correctness — keeping "what counts as a correct answer"
/// consistent between sampling, filtering, and eval.
pub fn parse_leading_int(s: &str) -> Option<i64> {
    let mut chars = s.chars();
    let mut c = chars.next()?;
    let neg = if c == '-' {
        c = chars.next()?;
        true
    } else {
        false
    };
    let mut val: i64 = 0;
    let mut got_digit = false;
    loop {
        if let Some(d) = c.to_digit(10) {
            val = val.checked_mul(10)?.checked_add(d as i64)?;
            got_digit = true;
            match chars.next() {
                Some(nc) => c = nc,
                None => break,
            }
        } else {
            break;
        }
    }
    if got_digit {
        Some(if neg { -val } else { val })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_leading_int_positive() {
        assert_eq!(parse_leading_int("529\n"), Some(529));
        assert_eq!(parse_leading_int("1998"), Some(1998));
        assert_eq!(parse_leading_int("0+1"), Some(0));
    }

    #[test]
    fn test_parse_leading_int_negative() {
        assert_eq!(parse_leading_int("-4\n"), Some(-4));
        assert_eq!(parse_leading_int("-1250\n next"), Some(-1250));
        assert_eq!(parse_leading_int("-0"), Some(0));
    }

    #[test]
    fn test_parse_leading_int_garbage() {
        assert_eq!(parse_leading_int("abc"), None);
        assert_eq!(parse_leading_int(""), None);
        assert_eq!(parse_leading_int("-"), None);
        assert_eq!(parse_leading_int("\n123"), None);
    }
}
