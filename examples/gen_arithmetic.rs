//! Standalone arithmetic-corpus generator for smolgpt.
//!
//! Emits `a+b=c` / `a-b=c` lines to a text file, suitable for dropping into
//! `data/` and training on with `smolgpt -d <path> --train --generate`.
//!
//! Operands `a` and `b` are always non-negative integers (`>= 0`). By default
//! both `+` and `-` are emitted and the result of `a-b` may be negative; use
//! `--ops`, `--min-result`, and `--max-result` to restrict the corpus.
//!
//! Examples:
//!     # addition-only, sums up to 99, no negatives (e.g. 5+6=11, 1+2=3)
//!     cargo run --release --example gen_arithmetic -- \
//!         --output data/arithmetic-add.txt --samples 200000 \
//!         --min 0 --max 99 --ops + --max-result 99
//!
//!     # + and -, no negative results (a-b only when a>=b)
//!     cargo run --release --example gen_arithmetic -- \
//!         --output data/arithmetic-nonneg.txt --ops +,- --min-result 0
//!
//! Reproducible by default (seed = 42). Override with `--seed <N>`.

use std::path::PathBuf;

use clap::Parser;
use rand::Rng;

/// Per-sample rejection-sampling attempt cap. Prevents an infinite loop when
/// the result filters are too restrictive for the operand range.
const MAX_ATTEMPTS_PER_SAMPLE: usize = 4096;

/// When `--dedup` is on, this many consecutive duplicate draws in a row means
/// the unique problem space is effectively exhausted (for small operand ranges
/// the space is finite, e.g. 100 pairs for 0-9 `+`), so we stop early instead
/// of spinning forever trying to reach `--samples`.
const MAX_DUP_STREAK: usize = 4096;

#[derive(Parser, Debug)]
#[clap(
    name = "gen-arithmetic",
    about = "Generate a synthetic arithmetic corpus",
    allow_negative_numbers = true,
)]
struct GenArgs {
    /// Output file path.
    #[clap(short, long, default_value = "data/arithmetic.txt")]
    output: PathBuf,

    /// Number of arithmetic samples to generate.
    #[clap(short, long, default_value_t = 200_000)]
    samples: usize,

    /// Inclusive lower bound for operands (must be >= 0).
    #[clap(long, default_value_t = 0)]
    min: i64,

    /// Inclusive upper bound for operands.
    #[clap(long, default_value_t = 999)]
    max: i64,

    /// Comma-separated operators to emit, e.g. `+` or `+,-`. Default `+,-`.
    #[clap(long, default_value = "+,-")]
    ops: String,

    /// Inclusive lower bound for the result `c`. Skip problems with `c < N`.
    /// Pass `0` to forbid negative results.
    #[clap(long)]
    min_result: Option<i64>,

    /// Inclusive upper bound for the result `c`. Skip problems with `c > N`.
    /// Pass `99` so that `a+b <= 99`.
    #[clap(long)]
    max_result: Option<i64>,

    /// Random seed for reproducibility (default 42).
    #[clap(long, default_value_t = 42)]
    seed: u64,

    /// Deduplicate the output: emit each unique `a op b = c` line at most once.
    /// On by default. With small operand ranges the unique space is finite
    /// (e.g. 100 pairs for operands 0-9 with `+`), so the actual output may be
    /// smaller than `--samples`; the generator stops early once the space is
    /// exhausted and prints how many unique lines it wrote. Output is sorted by
    /// `(op, a, b)` for reproducibility. Pass `--no-dedup` to allow duplicates.
    #[clap(long, default_value_t = true)]
    dedup: bool,
}

/// Parse the `--ops` string into a validated list of operator chars.
fn parse_ops(ops: &str) -> Result<Vec<char>, String> {
    let mut out = Vec::new();
    for raw in ops.split(',') {
        let op = raw.trim();
        match op {
            "+" | "-" => out.push(op.chars().next().unwrap()),
            _ => return Err(format!("--ops entry `{op}` invalid; only `+` and `-` supported")),
        }
    }
    if out.is_empty() {
        return Err("--ops must contain at least one operator".into());
    }
    Ok(out)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = GenArgs::parse();

    if args.min < 0 {
        return Err(format!("--min ({}) must be >= 0", args.min).into());
    }
    if args.min > args.max {
        return Err(format!("--min ({}) must be <= --max ({})", args.min, args.max).into());
    }
    let ops = parse_ops(&args.ops)?;

    let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(args.seed);

    // Collect sampled problems as (op, a, b, c) tuples. When `--dedup` is on we
    // also track the (op, a, b) keys we've already emitted so we can skip
    // repeats; `c` is deterministic given (op, a, b) so it isn't part of the
    // key. Sorting by (op, a, b) at the end gives stable, readable output
    // independent of the sampling order.
    let mut problems: Vec<(char, i64, i64, i64)> = Vec::with_capacity(args.samples);
    let mut seen: std::collections::HashSet<(char, i64, i64)> = std::collections::HashSet::new();
    let mut dup_streak: usize = 0;

    while problems.len() < args.samples {
        let (op, ((a, b), c)) = match rejection_sample(
            &mut rng,
            args.min,
            args.max,
            &ops,
            args.min_result,
            args.max_result,
        ) {
            Ok(v) => v,
            Err(attempts) => {
                return Err(format!(
                    "could not satisfy result filters after {attempts} attempts; \
                     loosen --min-result/--max-result or widen --min/--max"
                )
                .into());
            }
        };
        if args.dedup {
            if !seen.insert((op, a, b)) {
                dup_streak += 1;
                if dup_streak >= MAX_DUP_STREAK {
                    eprintln!(
                        "dedup: exhausted unique problem space after {} unique samples \
                         (requested {}); stopping early",
                        problems.len(),
                        args.samples
                    );
                    break;
                }
                continue;
            }
            dup_streak = 0;
        }
        problems.push((op, a, b, c));
    }

    if args.dedup {
        problems.sort_by_key(|&(op, a, b, _)| (op, a, b));
    }

    let out: String = problems
        .iter()
        .map(|(op, a, b, c)| format!("{a}{op}{b}={c}\n"))
        .collect();

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(&args.output, &out)?;
    let dedup_note = if args.dedup { "unique " } else { "" };
    println!(
        "Wrote {}{}arithmetic samples (operands [{}, {}], ops [{}], result [{}, {}]) to {}",
        problems.len(),
        dedup_note,
        args.min,
        args.max,
        ops.iter().collect::<String>(),
        args.min_result.map(|n| n.to_string()).unwrap_or_else(|| "-inf".into()),
        args.max_result.map(|n| n.to_string()).unwrap_or_else(|| "+inf".into()),
        args.output.display()
    );

    Ok(())
}

/// One sampled problem: `(a, b)` and the computed result `c`.
type Problem = ((i64, i64), i64);

/// Rejection-sample a single problem that satisfies the result filters.
/// Returns `Ok((op_char, ((a,b), c)))` or `Err(attempts)` if the filters are
/// infeasible for the given operand range.
fn rejection_sample(
    rng: &mut rand::rngs::StdRng,
    min: i64,
    max: i64,
    ops: &[char],
    min_result: Option<i64>,
    max_result: Option<i64>,
) -> Result<(char, Problem), usize> {
    for _ in 1..=MAX_ATTEMPTS_PER_SAMPLE {
        let a: i64 = rng.random_range(min..=max);
        let b: i64 = rng.random_range(min..=max);
        let op_idx = rng.random_range(0..ops.len());
        let op = ops[op_idx];
        let c = match op {
            '+' => a + b,
            '-' => a - b,
            _ => unreachable!(),
        };
        if let Some(lo) = min_result {
            if c < lo {
                continue;
            }
        }
        if let Some(hi) = max_result {
            if c > hi {
                continue;
            }
        }
        return Ok((op, ((a, b), c)));
    }
    Err(MAX_ATTEMPTS_PER_SAMPLE)
}
