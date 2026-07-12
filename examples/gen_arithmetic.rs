//! Standalone arithmetic-corpus generator for smolgpt.
//!
//! Emits `a+b=c` / `a-b=c` lines to a text file, suitable for dropping into
//! `data/` and training on with `smolgpt -d <path> --train --generate`.
//!
//! Operands `a` and `b` are always non-negative integers (`>= 0`). The result
//! `c` of `a-b` may still be negative (e.g. `3-7=-4`); only the LHS is kept
//! clean.
//!
//! Run with:
//!     cargo run --release --example gen_arithmetic -- \
//!         --output data/arithmetic.txt --samples 200000 --min 0 --max 999
//!
//! Reproducible by default (seed = 42). Override with `--seed <N>`.

use std::path::PathBuf;

use clap::Parser;
use rand::Rng;

#[derive(Parser, Debug)]
#[clap(
    name = "gen-arithmetic",
    about = "Generate a synthetic arithmetic corpus"
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

    /// Random seed for reproducibility (default 42).
    #[clap(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = GenArgs::parse();

    if args.min < 0 {
        return Err(format!("--min ({}) must be >= 0", args.min).into());
    }
    if args.min > args.max {
        return Err(format!("--min ({}) must be <= --max ({})", args.min, args.max).into());
    }

    let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(args.seed);

    // Rough upper bound: ~24 bytes per line ("999+999=1998\n").
    let mut out = String::with_capacity(args.samples * 24);

    for _ in 0..args.samples {
        let a: i64 = rng.random_range(args.min..=args.max);
        let b: i64 = rng.random_range(args.min..=args.max);
        let (op, r) = if rng.random_bool(0.5) {
            ("+", a + b)
        } else {
            ("-", a - b)
        };
        out.push_str(&format!("{a}{op}{b}={r}\n"));
    }

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(&args.output, out)?;
    println!(
        "Wrote {} arithmetic samples (range [{}, {}]) to {}",
        args.samples,
        args.min,
        args.max,
        args.output.display()
    );

    Ok(())
}
