//! Jacobian-lens interpretability analysis, generalized across any `Gpt`-type
//! model in the registry (see `analysis/jacobian_lens.py`'s module doc for the
//! math). This module is the single place that shells out to the Python
//! script and turns its output into something the registry can cache and
//! `serve.rs` can render — used from two call sites:
//!
//!   - `serve.rs`'s on-demand route (`GET/POST /api/models/{id}/jacobian-lens`)
//!   - `train.rs`'s `--jacobian-lens` "compiled" precompute path, which calls
//!     this directly right after training finishes (Gpt-only, same as here)
//!
//! Both store the result via `Registry::record_jacobian_lens`, so a
//! precomputed and an on-demand result are indistinguishable once cached.
//!
//! This is genuinely slow compared to the rest of this codebase (Rust-only
//! eval/grid computations run in milliseconds to low seconds; this spins up a
//! whole Python/PyTorch process and fits per-layer Jacobians via backprop) —
//! expect several seconds to tens of seconds even for these tiny models,
//! dominated by Python/torch import startup, not the actual math.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{SmolError, SmolResult};
use crate::registry::ModelRecord;

/// Outcome of a successful Jacobian-lens run: the script's raw `results.json`
/// text, the directory (relative to the project root) it wrote its PNG plots
/// into, and the plot filenames within that directory.
pub struct JacobianLensOutcome {
    pub results_json: String,
    pub plot_dir_rel: String,
    pub plot_files: Vec<String>,
}

/// Runs `analysis/jacobian_lens.py` for `record`, which must be a `Gpt`-type
/// model — callers are responsible for checking `record.model_type == "gpt"`
/// first (this returns a plain error for anything else rather than silently
/// no-op'ing, since a caller only invokes this when it already believes the
/// model is eligible; the "not applicable" UX for Bigram/Ngram lives in the
/// caller, one level up, where it can render a clean message instead of
/// surfacing this as a failure).
///
/// Resolves the model's registered architecture (including `heads_schedule`,
/// which may be a non-uniform per-block schedule like `"1,1,4,4"`) and
/// dataset path, invokes the script as a subprocess, and reads back its
/// `results.json` plus whatever PNG plots it wrote. Missing `python3` or a
/// script failure (e.g. missing torch) surface as a clear `SmolError` rather
/// than a panic or an opaque exit code — this is the one place in the app
/// with a real external-process dependency, so the error message says so
/// explicitly.
pub fn run_jacobian_lens_for_model(
    record: &ModelRecord,
    project_root: &Path,
) -> SmolResult<JacobianLensOutcome> {
    if record.model_type != "gpt" {
        return Err(SmolError::invalid_argument(&format!(
            "Jacobian lens is only applicable to Gpt-type models (got model_type={}); \
             Bigram/Ngram models have no transformer layers to lens through.",
            record.model_type
        )));
    }

    let script_path = project_root.join("analysis").join("jacobian_lens.py");
    if !script_path.exists() {
        return Err(SmolError::custom_error(&format!(
            "jacobian_lens.py not found at {}",
            script_path.display()
        )));
    }

    let model_path = crate::serve::resolve_within_root(project_root, &record.path).ok_or_else(|| {
        SmolError::custom_error(&format!(
            "model path not found or escapes project root: {}",
            record.path
        ))
    })?;
    let dataset_path = crate::serve::resolve_within_root(project_root, &record.dataset).ok_or_else(|| {
        SmolError::custom_error(&format!(
            "dataset path not found or escapes project root: {}",
            record.dataset
        ))
    })?;

    // Lossless per-block schedule, falling back to a uniform broadcast of
    // `num_heads` for rows written before the `heads_schedule` column
    // existed — same fallback `run_eval_for_model`/`run_generate` use via
    // `parse_heads_schedule_column`, expressed here as the CLI's own
    // "single number OR comma list" string syntax since that's what the
    // Python script's `--num-heads` flag expects.
    let heads_arg = if record.heads_schedule.is_empty() {
        record.num_heads.to_string()
    } else {
        record.heads_schedule.clone()
    };

    let plot_dir_rel = format!("jacobian_lens_output/{}", record.id);
    let output_dir = project_root.join(&plot_dir_rel);
    std::fs::create_dir_all(&output_dir).map_err(|e| {
        SmolError::custom_error(&format!(
            "failed to create jacobian-lens output dir {}: {e}",
            output_dir.display()
        ))
    })?;

    println!("[jacobian-lens] Running analysis for '{}' (this may take a while — Python/torch startup + Jacobian fitting)", record.id);

    let output = Command::new("python3")
        .arg(&script_path)
        .arg("--model-path")
        .arg(&model_path)
        .arg("--dataset-path")
        .arg(&dataset_path)
        .arg("--block-size")
        .arg(record.block_size.to_string())
        .arg("--hidden-size")
        .arg(record.hidden_size.to_string())
        .arg("--num-heads")
        .arg(&heads_arg)
        .arg("--num-blocks")
        .arg(record.num_blocks.to_string())
        .arg("--vocab-size")
        .arg(record.vocab_size.to_string())
        .arg("--output-dir")
        .arg(&output_dir)
        .output();

    let output = match output {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(SmolError::custom_error(
                "python3 not found on PATH — the Jacobian lens analysis requires a Python 3 \
                 interpreter with torch, numpy, matplotlib, and safetensors installed \
                 (pip install torch numpy matplotlib safetensors). This is a real \
                 environmental dependency, unlike the rest of this Rust web app.",
            ));
        }
        Err(e) => {
            return Err(SmolError::custom_error(&format!(
                "failed to launch python3 for jacobian-lens analysis: {e}"
            )));
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr
            .lines()
            .rev()
            .take(40)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return Err(SmolError::custom_error(&format!(
            "jacobian_lens.py exited with status {}:\n{}",
            output.status,
            if tail.is_empty() {
                "(no stderr output)".to_string()
            } else {
                tail
            }
        )));
    }

    let results_path = output_dir.join("results.json");
    let results_json = std::fs::read_to_string(&results_path).map_err(|e| {
        SmolError::custom_error(&format!(
            "jacobian_lens.py reported success but results.json is missing at {}: {e}",
            results_path.display()
        ))
    })?;

    let mut plot_files: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&output_dir) {
        for entry in entries.flatten() {
            let path: PathBuf = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("png") {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    plot_files.push(name.to_string());
                }
            }
        }
    }
    plot_files.sort();

    Ok(JacobianLensOutcome {
        results_json,
        plot_dir_rel,
        plot_files,
    })
}
