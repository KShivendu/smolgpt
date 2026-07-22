//! SQLite-backed model registry — replaces the manual `models.toml` registry.
//!
//! Every `--train` run auto-records the model's metadata into `smolgpt.db`
//! (via `register_model`), and `--serve` reads the registry from the DB (via
//! `list_models`). The DB is created on first open with `CREATE TABLE IF NOT
//! EXISTS`, so no manual migration step is needed.
//!
//! On the first `--serve` start, if the `models` table is empty and
//! `models.toml` exists, `import_from_toml` seeds the DB from the legacy TOML
//! registry so existing trained models stay visible. After that the DB is the
//! source of truth and `models.toml` is just a seed file.
//!
//! Timestamps are stored as `i64` Unix seconds (no `chrono`/`time` dep) and
//! serialized to ISO 8601 UTC strings for the JSON API via a custom
//! `serialize_with` on the timestamp fields.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};

use crate::args::{EvalMode, ModelType, TokenizerType};
use crate::error::{SmolError, SmolResult};
use crate::eval::EvalReport;
use crate::model::TrainOutcome;
use crate::tokenizer::{BpeTokenizer, SimpleTokenizer, Tokenizer};
// Only used in registry tests (round-tripping an `RftSummary` through the
// `trainings` table). The wire-format view served to the UI lives in
// `serve.rs` as `RftSummaryView`, decoupling the JSON shape from this
// internal struct.
#[cfg(test)]
use crate::rft::RftSummary;

/// Default DB filename, created in the project root (current dir).
const DB_FILENAME: &str = "smolgpt.db";

/// One `[[model]]` entry in the legacy `models.toml`. Kept here (not in
/// `serve.rs`) because `import_from_toml` is the only consumer in the new
/// DB-backed design — the HTTP layer reads `ModelRecord`s from the DB, not
/// `ModelEntry`s from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub path: String,
    pub model_type: String,
    pub tokenizer: String,
    #[serde(default)]
    pub vocab_size: usize,
    #[serde(default)]
    pub block_size: usize,
    #[serde(default)]
    pub hidden_size: usize,
    #[serde(default)]
    pub num_heads: usize,
    #[serde(default)]
    pub num_blocks: usize,
    pub dataset: String,
    pub dataset_name: String,
    #[serde(default)]
    pub eval_min: i64,
    #[serde(default)]
    pub eval_max: i64,
    #[serde(default)]
    pub eval_samples: usize,
    #[serde(default)]
    pub note: String,
}

/// `models.toml` top-level shape: a `[[model]]` array.
#[derive(Debug, Deserialize)]
struct ModelsFile {
    model: Vec<ModelEntry>,
}

/// One row in the `models` table. Mirrors the schema 1:1. Derives `Serialize`
/// for the `/api/models` JSON API and `Deserialize` so it can be round-tripped
/// in tests. Timestamps are stored as `i64` Unix seconds but serialized to
/// ISO 8601 UTC strings for the UI.
///
/// `base_model_id` links an RL variant (RFT/GRPO) back to its base model:
/// `None` (SQL NULL) means "this is a base model"; `Some(base_id)` means "this
/// is a variant of the model with id `base_id`". The web UI groups variants
/// under their base card and renders a `<select>` to switch between them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord {
    pub id: String,
    pub path: String,
    pub model_type: String,
    pub tokenizer: String,
    pub vocab_size: i64,
    pub block_size: i64,
    pub hidden_size: i64,
    pub num_heads: i64,
    pub num_blocks: i64,
    /// Full per-block head-count schedule, as the same comma-separated
    /// string syntax `--num-heads` accepts (e.g. `"1,1,4,4"`, or `"4"` for a
    /// uniform architecture) — round-trips directly into a `--num-heads`
    /// value. This is the LOSSLESS source of truth for reconstructing a
    /// model's exact per-block shapes; `num_heads` above is only a
    /// quick-glance summary (`min` of this schedule) and must not be used
    /// for reconstruction. Empty string for rows written before this column
    /// existed (see `migrate_add_heads_schedule_column`'s doc) — callers
    /// that need a schedule for such a row should fall back to treating
    /// `num_heads` as uniform (`vec![num_heads; num_blocks]`).
    #[serde(default)]
    pub heads_schedule: String,
    /// Whether this model's SFT stage sampled training windows only from true
    /// `"a op b="` fact boundaries (`--aligned-windows`, see
    /// `dataset::compute_fact_boundaries`/`Dataset::sample_start_indices`).
    /// `Some(true)`/`Some(false)` when known (every row written after this
    /// column existed always records the actual CLI flag value used);
    /// `None` (SQL NULL) for rows written before this column existed — their
    /// true historical setting is genuinely NOT recoverable from the DB, so
    /// callers (the Samples tab) must show "unknown" rather than silently
    /// assuming either value. For an RL variant (RFT/GRPO), this describes
    /// the variant's BASE model's SFT stage (copied via `TrainingMeta::with_variant`),
    /// not the RL stage itself.
    #[serde(default)]
    pub aligned_windows: Option<bool>,
    pub dataset: String,
    pub dataset_name: String,
    pub eval_min: i64,
    pub eval_max: i64,
    pub eval_samples: i64,
    pub note: String,
    pub params_estimate: i64,
    /// `None` (SQL NULL) for a base model; `Some(base_id)` for an RL variant
    /// (RFT/GRPO) linked to its base. See the struct doc for the UI grouping.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub base_model_id: Option<String>,
    #[serde(serialize_with = "serialize_iso_i64")]
    pub created_at: i64,
    #[serde(serialize_with = "serialize_iso_i64")]
    pub updated_at: i64,
}

/// One row in the `evals` table — the summary of an eval run. The full
/// `EvalReport` (with `by_digits` + `examples`) is returned live by the eval
/// endpoint; the DB stores only the summary so `--serve` can show the last
/// result on page reload without re-running.
///
/// `eval_min`/`eval_max` are the model's stored operand range at eval time
/// (stamped on the row in `record_eval`). Rows written before this column
/// existed are `NULL` and treated as "unknown range" — `latest_eval` (smart
/// mode) still considers them a match for any current range, so they remain
/// visible until a ranged row supersedes them.
#[derive(Debug, Clone, Serialize)]
pub struct EvalRecord {
    pub id: i64,
    pub model_id: String,
    pub correct: i64,
    pub total: i64,
    pub correct_plus: i64,
    pub total_plus: i64,
    pub correct_minus: i64,
    pub total_minus: i64,
    pub seed: Option<i64>,
    /// Operand range stamped on the row at eval time. `None` for rows
    /// written before the column existed (or when the model wasn't
    /// registered). Smart-mode `latest_eval` matches these against the
    /// model's current range; `None` matches anything.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_min: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_max: Option<i64>,
    #[serde(serialize_with = "serialize_iso_i64")]
    pub run_at: i64,
}

/// The single cached `eval_grids` row for a model — the most recent
/// exhaustive Grid-tab run. `report_json` is the serialized
/// `crate::eval::EvalGridReport` (every cell, not just a summary — see the
/// `eval_grids` table's doc for why this table keeps only one row per model
/// instead of a history like `evals`). `eval_min`/`eval_max` are the model's
/// operand range at the time this grid was computed; `latest_eval_grid`
/// compares them against the model's *current* range and returns `None`
/// (treats the cache as absent) on a mismatch, mirroring `latest_eval`'s
/// smart-mode staleness handling for the sampled eval.
#[derive(Debug, Clone, Serialize)]
pub struct EvalGridRecord {
    pub model_id: String,
    pub eval_min: i64,
    pub eval_max: i64,
    pub report_json: String,
    pub correct: i64,
    pub total: i64,
    #[serde(serialize_with = "serialize_iso_i64")]
    pub run_at: i64,
}

/// One row in the `checkpoint_grids` table — a SINGLE point-in-training
/// snapshot of the exhaustive eval grid, tagged with the epoch/loss at which
/// it was taken. Unlike `eval_grids` (one row per model, overwritten in
/// place on every recompute — see that table's doc), `checkpoint_grids`
/// deliberately accumulates MULTIPLE rows per model over the course of a
/// single `--train` run, so a follow-up UI can build a slider that animates
/// through how the grid changed as training progressed. Written by
/// `train.rs`'s `on_best_loss` callback (see `LanguageModel::train_with_dropout`)
/// whenever the smoothed training loss hits a new best-so-far value, subject
/// to the throttle documented on `model::should_snapshot`.
///
/// `report_json` is the same `crate::eval::EvalGridReport`-shaped JSON as
/// `eval_grids.report_json`, for consistency between the two tables' payload
/// shape. `loss` is the smoothed (rolling-mean) training loss at the epoch
/// this snapshot was taken, matching the value `on_best_loss` fired on — NOT
/// the raw per-epoch loss, so it's directly comparable to the early-stopping
/// smoothing already used elsewhere in this codebase.
#[derive(Debug, Clone, Serialize)]
pub struct CheckpointGridRecord {
    pub id: i64,
    pub model_id: String,
    pub epoch: i64,
    pub loss: f64,
    pub eval_min: i64,
    pub eval_max: i64,
    pub report_json: String,
    pub correct: i64,
    pub total: i64,
    #[serde(serialize_with = "serialize_iso_i64")]
    pub run_at: i64,
}

/// Lightweight metadata about a cached eval-grid row, surfaced on every
/// `ModelView` via `/api/models` (mirroring how `cached_eval`'s summary is
/// already embedded there) so the browser can tell "a cached grid exists"
/// without a round trip, decide whether to show a "Recompute" vs "Run
/// exhaustive grid" button label, and lazily fetch the full cached grid (via
/// `GET /api/models/{id}/eval-grid`) only when one is known to exist. Does
/// NOT carry `report_json` — that stays a deliberate extra (but cheap, DB-only)
/// round trip, the same way `cached_eval` omits `by_digits`/`examples`.
#[derive(Debug, Clone, Serialize)]
pub struct EvalGridSummary {
    pub eval_min: i64,
    pub eval_max: i64,
    pub correct: i64,
    pub total: i64,
    #[serde(serialize_with = "serialize_iso_i64")]
    pub run_at: i64,
}

impl From<&EvalGridRecord> for EvalGridSummary {
    fn from(r: &EvalGridRecord) -> EvalGridSummary {
        EvalGridSummary {
            eval_min: r.eval_min,
            eval_max: r.eval_max,
            correct: r.correct,
            total: r.total,
            run_at: r.run_at,
        }
    }
}

/// The single cached `jacobian_lens_results` row for a model — the most
/// recent Jacobian-lens interpretability run (see `crate::jacobian_lens`).
/// `results_json` is `analysis/jacobian_lens.py`'s `results.json` output
/// verbatim; `plot_dir` is the directory (relative to the project root) the
/// script wrote its PNG plots into; `plot_files` are the PNG filenames within
/// that directory, served individually via
/// `GET /api/models/{id}/jacobian-lens/plot/{filename}`.
#[derive(Debug, Clone, Serialize)]
pub struct JacobianLensRecord {
    pub model_id: String,
    pub results_json: String,
    pub plot_dir: String,
    pub plot_files: Vec<String>,
    #[serde(serialize_with = "serialize_iso_i64")]
    pub computed_at: i64,
}

/// Lightweight metadata about a cached Jacobian-lens run, surfaced on every
/// `ModelView` via `/api/models` (mirroring `EvalGridSummary`) so the browser
/// can tell "a cached analysis exists" without a round trip and skip straight
/// to fetching the full cached result when the Jacobian tab is opened.
#[derive(Debug, Clone, Serialize)]
pub struct JacobianLensSummary {
    #[serde(serialize_with = "serialize_iso_i64")]
    pub computed_at: i64,
}

impl From<&JacobianLensRecord> for JacobianLensSummary {
    fn from(r: &JacobianLensRecord) -> JacobianLensSummary {
        JacobianLensSummary {
            computed_at: r.computed_at,
        }
    }
}

/// One row in the `trainings` table — the metrics of a single training run
/// (SFT or RFT). The structured payloads (`loss_trajectory`, `rft_summary`)
/// are stored as JSON strings and parsed lazily by `serve.rs` when building
/// the `ModelView` so this struct stays cheap to load even for runs with a
/// 2000-point loss series. `loss_trajectory_json` is a JSON array of per-epoch
/// `f32` losses for SFT runs (e.g. `[7.49, 1.65, 1.34, ...]`), and the string
/// `"null"` for RFT runs. `rft_summary_json` is a serialized `RftSummary` for
/// RFT runs and `"null"` for SFT runs. `trained_at` is serialized as ISO 8601
/// UTC for the JSON API (mirrors the timestamp convention on `EvalRecord`).
///
/// `trainings.model_id` is a soft reference (no FK) to `models.id`: training
/// history should survive a re-train UPSERT (which would otherwise cascade if
/// we used `ON DELETE CASCADE`) and should also be insertable before a model
/// row exists (e.g. the rare case of training metrics recorded for a model
/// that's registered later). The application enforces the relationship; the
/// schema does not.
#[derive(Debug, Clone, Serialize)]
pub struct TrainingRecord {
    pub id: i64,
    pub model_id: String,
    /// "sft" or "rft".
    pub kind: String,
    pub epochs_run: i64,
    pub early_stopped: bool,
    pub final_loss: f64,
    /// Raw JSON string of the per-epoch loss series (SFT) or `"null"` (RFT).
    /// Parsed into `Vec<f32>` by `serve.rs` for the UI sparkline.
    pub loss_trajectory_json: String,
    /// Raw JSON string of the `RftSummary` (RFT) or `"null"` (SFT). Parsed
    /// into `RftSummaryView` by `serve.rs` for the UI per-round table.
    pub rft_summary_json: String,
    /// Exact greedy-decoding accuracy over every parseable line of the
    /// training corpus itself (not sampled — the literal training set),
    /// computed once after SFT finishes. `None` for rows written before this
    /// column existed, or for RFT/GRPO rows (their "training set" is the
    /// per-round winners corpus, which changes every round, so a single
    /// post-hoc number wouldn't mean the same thing). Distinct from `evals`,
    /// which samples random operands from a configured range and may include
    /// problems the corpus never contained.
    pub train_correct: Option<i64>,
    pub train_total: Option<i64>,
    /// Inclusive operand range + comma-separated ops (e.g. `"+,-"`) this
    /// row's RL stage actually sampled prompts from (`--rft-min`/`--rft-max`/
    /// `--rft-ops` for `kind == "rft"`, `--grpo-min`/`--grpo-max`/`--grpo-ops`
    /// for `kind == "grpo"`). `None` for SFT rows (no prompt-sampling
    /// concept) and for RFT/GRPO rows written before these columns existed.
    /// Used by `--serve`'s Samples tab to faithfully reconstruct the exact
    /// prompt shape a variant's RL stage trained on.
    pub prompt_min: Option<i64>,
    pub prompt_max: Option<i64>,
    pub prompt_ops: Option<String>,
    #[serde(serialize_with = "serialize_iso_i64")]
    pub trained_at: i64,
}

/// Owned bundle of the training-time metadata needed to build a `ModelRecord`.
/// Passed to `ModelRecord::from_training` so the constructor stays decoupled
/// from the `Args` struct (which `train.rs` destructures inline).
///
/// `eval_min`/`eval_max` are `Option<i64>` so the constructor can tell
/// "user passed `--eval-min 0`" apart from "user omitted the flag" and apply
/// the smart-mode corpus-derivation rule only when they're `None`. `eval_mode`
/// picks between smart (corpus-derived) and legacy (0/999) fallbacks.
///
/// `base_model_id` is `None` for a regular `--train` run (the model is a base
/// model) and `Some(base_id)` for `--rft`/`--grpo` runs (the model is an RL
/// variant linked to its base). The constructor copies it into the
/// `ModelRecord.base_model_id` field so the UI can group variants under their
/// base card.
#[derive(Debug, Clone, Copy)]
pub struct TrainingMeta<'a> {
    pub model_type: ModelType,
    pub tokenizer: TokenizerType,
    pub block_size: usize,
    pub hidden_size: usize,
    /// Full per-block head-count schedule (length == `num_blocks` for a GPT
    /// model; empty for BigramLM, which has no heads concept). Uniform
    /// architectures (the common case, e.g. `--num-heads 4`) are
    /// `vec![4; num_blocks]` — there's no separate scalar field; the
    /// `models.num_heads` DB column (which predates per-block schedules) is
    /// derived from this as `min(heads_schedule)` in `from_training`, and the
    /// exact schedule + the `--num-heads` value needed to reload it are
    /// surfaced in the generated `note` for non-uniform architectures.
    pub heads_schedule: &'a [usize],
    pub num_blocks: usize,
    /// Whether `--aligned-windows` was passed for this SFT run. Always a
    /// concrete `bool` here (never unknown) — a live `--train` run always
    /// knows the actual CLI flag value; unknown only applies to rows written
    /// before this field/column existed (see `ModelRecord.aligned_windows`'s
    /// doc), which this `TrainingMeta` doesn't represent.
    pub aligned_windows: bool,
    pub dataset_path: &'a Path,
    pub model_path: &'a Path,
    pub actual_vocab_size: usize,
    pub eval_min: Option<i64>,
    pub eval_max: Option<i64>,
    pub eval_samples: usize,
    pub eval_mode: EvalMode,
    pub seed: Option<u64>,
    /// `None` for a base model (`--train`); `Some(base_id)` for an RL variant
    /// (`--rft`/`--grpo`) linked to its base. See the struct doc.
    pub base_model_id: Option<&'a str>,
}

impl<'a> TrainingMeta<'a> {
    /// Derive the meta for an RL variant (`--rft`/`--grpo`): same
    /// arch/tokenizer/dataset as `self`, but pointing at the variant's own
    /// `.bin` path and linked to `base_id` so the UI groups it under the base
    /// card. `self` is `Copy`, so this doesn't consume the base meta.
    pub fn with_variant(&self, variant_path: &'a Path, base_id: &'a str) -> TrainingMeta<'a> {
        TrainingMeta {
            model_path: variant_path,
            base_model_id: Some(base_id),
            ..*self
        }
    }
}

/// Handle to the SQLite registry. Wraps a single `rusqlite::Connection`; for
/// the multi-threaded `--serve` runtime, wrap this in `Arc<Mutex<Registry>>`
/// (`Connection` is `Send` but not `Sync`).
pub struct Registry {
    conn: Connection,
}

impl Registry {
    /// Open `smolgpt.db` in the current dir, create the schema if missing, and
    /// enable foreign-key enforcement. Returns a ready-to-use `Registry`.
    pub fn open() -> SmolResult<Registry> {
        let path = std::env::current_dir()
            .map_err(|e| SmolError::custom_error(&format!("cwd: {e}")))?
            .join(DB_FILENAME);
        Self::open_at(&path)
    }

    /// Same as `open` but at an explicit path (used by tests / custom roots).
    pub fn open_at(path: &Path) -> SmolResult<Registry> {
        let conn = Connection::open(path).map_err(|e| {
            SmolError::custom_error(&format!(
                "Failed to open DB at {}: {e}",
                path.display()
            ))
        })?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(|e| SmolError::custom_error(&format!("pragma foreign_keys: {e}")))?;
        // `--train` and `--serve` can open this same file from separate
        // processes concurrently. Without a busy timeout, any write that
        // lands while the other process holds the lock fails immediately
        // with SQLITE_BUSY instead of retrying; WAL lets concurrent readers
        // proceed without blocking on an in-progress writer.
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| SmolError::custom_error(&format!("pragma journal_mode: {e}")))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .map_err(|e| SmolError::custom_error(&format!("busy_timeout: {e}")))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                model_type TEXT NOT NULL,
                tokenizer TEXT NOT NULL,
                vocab_size INTEGER NOT NULL,
                block_size INTEGER NOT NULL,
                hidden_size INTEGER NOT NULL,
                num_heads INTEGER NOT NULL,
                num_blocks INTEGER NOT NULL,
                heads_schedule TEXT NOT NULL DEFAULT '',
                dataset TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                eval_min INTEGER NOT NULL,
                eval_max INTEGER NOT NULL,
                eval_samples INTEGER NOT NULL,
                note TEXT NOT NULL,
                params_estimate INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                correct INTEGER NOT NULL,
                total INTEGER NOT NULL,
                correct_plus INTEGER NOT NULL,
                total_plus INTEGER NOT NULL,
                correct_minus INTEGER NOT NULL,
                total_minus INTEGER NOT NULL,
                seed INTEGER,
                run_at INTEGER NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_evals_model_id ON evals(model_id);
            -- Per-training-run metrics (SFT loss trajectory, RFT per-round
            -- summary). `model_id` is a SOFT reference (no FK): we want
            -- training history to survive a re-train UPSERT on `models` (a
            -- FK with ON DELETE CASCADE would wipe history when the parent
            -- row is replaced) and to be insertable before the `models` row
            -- exists. The application enforces the relationship; the schema
            -- does not. Added in-place via CREATE TABLE IF NOT EXISTS so the
            -- existing smolgpt.db (created before this table existed) is
            -- migrated on the next Registry::open.
            CREATE TABLE IF NOT EXISTS trainings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                epochs_run INTEGER NOT NULL,
                early_stopped INTEGER NOT NULL,
                final_loss REAL NOT NULL,
                loss_trajectory TEXT NOT NULL,
                rft_summary TEXT NOT NULL,
                trained_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_trainings_model_id ON trainings(model_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_trainings_model_kind ON trainings(model_id, kind);
            -- Cache of the most recent exhaustive eval-grid run per model
            -- (`--serve`'s Grid tab). Unlike `evals` (a small per-run summary
            -- row we keep a history of), a grid's `report_json` blob holds
            -- every cell (prompt/generated/true_answer/diff for every (a,b)
            -- combination) — potentially a few hundred KB for a 50x50 grid —
            -- so we keep ONE row per model (model_id is the PRIMARY KEY) and
            -- overwrite it on every recompute, the same 'single row that
            -- updates in place' pattern `trainings` uses for its own large
            -- JSON blob columns, rather than accumulating an unbounded
            -- history of large blobs the way `evals` does for its small
            -- summary rows. `eval_min`/`eval_max` are stamped at cache-write
            -- time (mirroring `evals.eval_min`/`eval_max`) so `latest_eval_grid`
            -- can detect a stale cache the same way `latest_eval` does: if the
            -- model's current range no longer matches the cached row's range,
            -- the cache is treated as absent rather than served stale. Uses a
            -- real FK (unlike `trainings`' soft reference) because `models`
            -- rows are only ever replaced via `ON CONFLICT DO UPDATE` (never
            -- deleted-then-reinserted) for a same-id re-train, so there's no
            -- risk of an UPSERT cascading away a cache that should survive —
            -- the same reasoning `evals` already relies on for its FK.
            CREATE TABLE IF NOT EXISTS eval_grids (
                model_id TEXT PRIMARY KEY,
                eval_min INTEGER NOT NULL,
                eval_max INTEGER NOT NULL,
                report_json TEXT NOT NULL,
                correct INTEGER NOT NULL,
                total INTEGER NOT NULL,
                run_at INTEGER NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
            );
            -- History of exhaustive-eval-grid snapshots taken DURING training,
            -- one row per loss-improvement-triggered checkpoint (see
            -- `model::should_snapshot` for the throttle policy). Unlike
            -- `eval_grids` (one row per model, overwritten in place), this
            -- table intentionally accumulates many rows per model_id so a
            -- follow-up UI can animate through them with a slider ordered by
            -- `epoch`. Uses a real FK (like `eval_grids`, unlike `trainings`'
            -- soft reference) for the same reason `eval_grids` does: `models`
            -- rows are only ever replaced via `ON CONFLICT DO UPDATE`, never
            -- deleted-then-reinserted, so there's no risk of an UPSERT
            -- cascading away history that should survive. A re-train under
            -- the same model id WILL cascade-delete this model's prior
            -- checkpoint-grid history (same as it would for `eval_grids`) —
            -- that's intentional: a fresh training run's snapshots describe a
            -- different loss curve than the old run's.
            CREATE TABLE IF NOT EXISTS checkpoint_grids (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                loss REAL NOT NULL,
                eval_min INTEGER NOT NULL,
                eval_max INTEGER NOT NULL,
                report_json TEXT NOT NULL,
                correct INTEGER NOT NULL,
                total INTEGER NOT NULL,
                run_at INTEGER NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_checkpoint_grids_model_epoch
                ON checkpoint_grids(model_id, epoch);
            -- Cache of the most recent Jacobian-lens interpretability run per
            -- model (`--serve`'s Jacobian tab, or `--train --jacobian-lens`'s
            -- compiled precompute). One row per model (model_id is the
            -- PRIMARY KEY), overwritten in place on every recompute -- same
            -- 'single row, large blob, only the latest is useful' pattern as
            -- `eval_grids`. `results_json` is the JSON `analysis/jacobian_lens.py`
            -- writes to `results.json`. `plot_dir` is the directory (relative
            -- to the project root) the script wrote its PNG plots into --
            -- kept as files on disk rather than DB BLOBs because this
            -- analysis is Gpt-only, low-volume (one row per model, not per
            -- eval run), and file-based storage means the plots can be
            -- inspected directly from the filesystem too; `plot_files` is a
            -- JSON array of the PNG filenames within that directory, read by
            -- `GET /api/models/{id}/jacobian-lens/plot/{filename}`. A re-train
            -- under the same model id cascade-deletes this row (real FK, same
            -- reasoning as `eval_grids`/`checkpoint_grids`) since a retrained
            -- model's internals are a different analysis subject entirely.
            CREATE TABLE IF NOT EXISTS jacobian_lens_results (
                model_id TEXT PRIMARY KEY,
                results_json TEXT NOT NULL,
                plot_dir TEXT NOT NULL,
                plot_files TEXT NOT NULL,
                computed_at INTEGER NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
            );",
        )
        // The UNIQUE index enforces "one row per (model_id, kind)" at the
        // schema level, so `upsert_training` can rely on `ON CONFLICT` for an
        // atomic single-statement upsert instead of a DELETE-then-INSERT pair.
        .map_err(|e| SmolError::custom_error(&format!("schema init: {e}")))?;

        // In-place migration: stamp eval rows with the operand range they
        // were run against. Existing rows get NULL (treated as "unknown
        // range" by latest_eval's smart filter — they match anything). ALTER
        // TABLE ADD COLUMN is idempotent at the SQL level via "duplicate
        // column name" error suppression below.
        migrate_add_eval_range_columns(&conn)?;

        // In-place migration: add `base_model_id` to `models` so RL variants
        // (RFT/GRPO) can be linked to their base model. Existing rows get NULL
        // (treated as "is a base model"). Idempotent: checks PRAGMA
        // table_info(models) first and only ALTERs the missing column.
        migrate_add_base_model_id_column(&conn)?;

        // One-time best-effort backfill: for existing rows whose id ends with
        // `-rft` or `-grpo` (the conventional variant naming) and whose
        // stripped stem is an existing model id, set `base_model_id` so the
        // UI groups them under their base. This links variants created by the
        // old `--rft`/`--grpo` path (which registered variants as top-level
        // models) so they show up as nested variants after the migration. A
        // no-op once every matching row has `base_model_id` set.
        migrate_backfill_base_model_id(&conn)?;

        // In-place migration: add `train_correct`/`train_total` to `trainings`
        // so SFT runs can report exact greedy-decoding accuracy over the
        // actual training corpus (distinct from `evals`, which samples random
        // operands from a range and may include out-of-corpus problems).
        // Existing rows get NULL (treated as "not computed" — pre-migration
        // runs, or non-SFT rows).
        migrate_add_train_accuracy_columns(&conn)?;

        // In-place migration: add `heads_schedule` to `models` — the
        // lossless per-block head-count schedule (CSV, same syntax
        // `--num-heads` accepts), needed to correctly reload a non-uniform
        // architecture. The scalar `num_heads` column alone is lossy (it's
        // `min(heads_schedule)`, e.g. `[1,1,4,4]` and `[1,2,3,4]` both
        // collapse to `num_heads=1`), so `--serve` needs this column to
        // reconstruct the true per-block shapes. Existing rows get ''
        // (empty string) — see the `ModelRecord.heads_schedule` field doc
        // for how callers should fall back for those rows.
        migrate_add_heads_schedule_column(&conn)?;

        // In-place migration: add `aligned_windows` to `models` (see
        // `ModelRecord.aligned_windows`'s doc). Existing rows get NULL
        // (unknown historical setting).
        migrate_add_aligned_windows_column(&conn)?;

        // In-place migration: add `prompt_min`/`prompt_max`/`prompt_ops` to
        // `trainings` (see `TrainingRecord`'s doc) so an RFT/GRPO row records
        // the actual operand range + ops its RL stage sampled prompts from.
        // Existing rows get NULL (unknown — pre-migration RFT/GRPO runs, or
        // SFT rows, which have no prompt-sampling concept).
        migrate_add_prompt_range_columns(&conn)?;

        Ok(Registry { conn })
    }

    /// Insert (or update) a model record. Uses `ON CONFLICT(id) DO UPDATE` so
    /// re-training a model with the same `id` updates the existing row in place
    /// — this preserves the model's eval history (the `evals` FK is
    /// `ON DELETE CASCADE`, so a naive `INSERT OR REPLACE` would delete the
    /// row and cascade-wipe its evals) and preserves `created_at` (only
    /// `updated_at` is refreshed). A pre-delete handles the rare case where a
    /// different `id` already holds the same `path` (UNIQUE constraint): that
    /// other row is removed (its evals cascade-delete, which is correct since
    /// we're replacing that model). `base_model_id` is upserted too, so a
    /// re-train of a variant keeps its base link.
    pub fn register_model(&self, rec: &ModelRecord) -> SmolResult<()> {
        // The pre-delete and the insert must land together: if the process
        // dies between them, the pre-delete's cascade-deleted eval history
        // would be lost with no replacement row ever inserted, contradicting
        // the "preserves eval history" guarantee above. `Connection::execute`
        // takes `&self` (rusqlite manages the underlying handle internally),
        // so a manual BEGIN/COMMIT/ROLLBACK gets atomicity here without
        // widening this method to `&mut self`.
        self.conn
            .execute_batch("BEGIN")
            .map_err(|e| SmolError::custom_error(&format!("register_model begin: {e}")))?;

        let result = self.register_model_inner(rec);

        match result {
            Ok(()) => self
                .conn
                .execute_batch("COMMIT")
                .map_err(|e| SmolError::custom_error(&format!("register_model commit: {e}"))),
            Err(e) => {
                let _ = self.conn.execute_batch("ROLLBACK");
                Err(e)
            }
        }
    }

    fn register_model_inner(&self, rec: &ModelRecord) -> SmolResult<()> {
        self.conn
            .execute(
                "DELETE FROM models WHERE path = ?1 AND id <> ?2",
                params![rec.path, rec.id],
            )
            .map_err(|e| SmolError::custom_error(&format!("register_model pre-delete: {e}")))?;

        self.conn
            .execute(
                "INSERT INTO models (
                    id, path, model_type, tokenizer, vocab_size, block_size,
                    hidden_size, num_heads, num_blocks, heads_schedule,
                    aligned_windows, dataset,
                    dataset_name, eval_min, eval_max, eval_samples, note,
                    params_estimate, base_model_id, created_at, updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                          ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)
                ON CONFLICT(id) DO UPDATE SET
                    path = excluded.path,
                    model_type = excluded.model_type,
                    tokenizer = excluded.tokenizer,
                    vocab_size = excluded.vocab_size,
                    block_size = excluded.block_size,
                    hidden_size = excluded.hidden_size,
                    num_heads = excluded.num_heads,
                    num_blocks = excluded.num_blocks,
                    heads_schedule = excluded.heads_schedule,
                    aligned_windows = excluded.aligned_windows,
                    dataset = excluded.dataset,
                    dataset_name = excluded.dataset_name,
                    eval_min = excluded.eval_min,
                    eval_max = excluded.eval_max,
                    eval_samples = excluded.eval_samples,
                    note = excluded.note,
                    params_estimate = excluded.params_estimate,
                    base_model_id = excluded.base_model_id,
                    updated_at = excluded.updated_at",
                params![
                    rec.id,
                    rec.path,
                    rec.model_type,
                    rec.tokenizer,
                    rec.vocab_size,
                    rec.block_size,
                    rec.hidden_size,
                    rec.num_heads,
                    rec.num_blocks,
                    rec.heads_schedule,
                    rec.aligned_windows,
                    rec.dataset,
                    rec.dataset_name,
                    rec.eval_min,
                    rec.eval_max,
                    rec.eval_samples,
                    rec.note,
                    rec.params_estimate,
                    rec.base_model_id,
                    rec.created_at,
                    rec.updated_at,
                ],
            )
            .map_err(|e| SmolError::custom_error(&format!("register_model: {e}")))?;
        Ok(())
    }

    /// `SELECT * FROM models ORDER BY id` — the source of truth for
    /// `--serve`'s `/api/models`.
    pub fn list_models(&self) -> SmolResult<Vec<ModelRecord>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, path, model_type, tokenizer, vocab_size, block_size,
                        hidden_size, num_heads, num_blocks, heads_schedule,
                        aligned_windows, dataset,
                        dataset_name, eval_min, eval_max, eval_samples, note,
                        params_estimate, base_model_id, created_at, updated_at
                 FROM models ORDER BY id",
            )
            .map_err(|e| SmolError::custom_error(&format!("list_models prepare: {e}")))?;
        let rows = stmt
            .query_map([], map_model_row)
            .map_err(|e| SmolError::custom_error(&format!("list_models query: {e}")))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(|e| {
                SmolError::custom_error(&format!("list_models row: {e}"))
            })?);
        }
        Ok(out)
    }

    /// Fetch a single model by id. Returns `None` if not found.
    pub fn get_model(&self, id: &str) -> SmolResult<Option<ModelRecord>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, path, model_type, tokenizer, vocab_size, block_size,
                        hidden_size, num_heads, num_blocks, heads_schedule,
                        aligned_windows, dataset,
                        dataset_name, eval_min, eval_max, eval_samples, note,
                        params_estimate, base_model_id, created_at, updated_at
                 FROM models WHERE id = ?1",
            )
            .map_err(|e| SmolError::custom_error(&format!("get_model prepare: {e}")))?;
        let row = stmt
            .query_row(params![id], map_model_row)
            .optional()
            .map_err(|e| SmolError::custom_error(&format!("get_model query: {e}")))?;
        Ok(row)
    }

    /// Insert an eval summary row. The full `EvalReport` (with `by_digits` +
    /// `examples`) is not persisted — only the summary the UI shows on reload.
    /// The model's current `eval_min`/`eval_max` (looked up from the `models`
    /// table) are stamped on the row so `latest_eval` can later filter by the
    /// model's current range and hide stale rows from an old range. If the
    /// model isn't registered, NULL is stored — `latest_eval` treats NULL as
    /// "matches any range".
    pub fn record_eval(
        &self,
        model_id: &str,
        report: &EvalReport,
        seed: Option<u64>,
    ) -> SmolResult<()> {
        // Look up the model's current range to stamp on this row. Best-effort:
        // a lookup failure logs and falls back to NULL (which latest_eval
        // treats as "matches anything"), so a transient DB issue doesn't
        // abort the eval-persist write.
        let (eval_min, eval_max) = match self.get_model(model_id) {
            Ok(Some(m)) => (Some(m.eval_min), Some(m.eval_max)),
            Ok(None) => (None, None),
            Err(e) => {
                eprintln!(
                    "[registry] WARNING: record_eval couldn't look up model \
                     '{model_id}' to stamp eval range: {e}; storing NULL"
                );
                (None, None)
            }
        };
        self.conn
            .execute(
                "INSERT INTO evals (model_id, correct, total, correct_plus,
                                    total_plus, correct_minus, total_minus,
                                    seed, eval_min, eval_max, run_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    model_id,
                    report.correct as i64,
                    report.total as i64,
                    report.correct_plus as i64,
                    report.total_plus as i64,
                    report.correct_minus as i64,
                    report.total_minus as i64,
                    seed.map(|s| s as i64),
                    eval_min,
                    eval_max,
                    now_unix(),
                ],
            )
            .map_err(|e| SmolError::custom_error(&format!("record_eval: {e}")))?;
        Ok(())
    }

    /// Most recent eval row for a model that matches the model's *current*
    /// `eval_min`/`eval_max`. Rows with a non-NULL range match only if their
    /// range equals the model's current range; rows with NULL min/max match
    /// any current range (so old pre-migration rows stay visible until a
    /// ranged row supersedes them). Newest match wins, with `id DESC` as the
    /// tie-breaker for rows written within the same wall-clock second.
    /// Returns `None` if the model has no eval rows at all.
    ///
    /// This is the smart-mode lookup. Use `latest_eval_legacy` for the old
    /// unfiltered "newest by run_at" behavior.
    pub fn latest_eval(&self, model_id: &str) -> SmolResult<Option<EvalRecord>> {
        // Look up the model's current range so we can filter rows by it. If
        // the model isn't registered, treat the target range as NULL so the
        // query matches only NULL-range rows (which is correct — a model
        // with no metadata shouldn't pull in another model's ranged rows).
        let (target_min, target_max) = match self.get_model(model_id)? {
            Some(m) => (Some(m.eval_min), Some(m.eval_max)),
            None => (None, None),
        };
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, model_id, correct, total, correct_plus, total_plus,
                        correct_minus, total_minus, seed, eval_min, eval_max,
                        run_at
                 FROM evals
                 WHERE model_id = ?1
                   AND (
                       (eval_min IS NULL AND eval_max IS NULL)
                       OR (eval_min = ?2 AND eval_max = ?3)
                   )
                 ORDER BY run_at DESC, id DESC LIMIT 1",
            )
            .map_err(|e| SmolError::custom_error(&format!("latest_eval prepare: {e}")))?;
        let row = stmt
            .query_row(params![model_id, target_min, target_max], map_eval_row)
            .optional()
            .map_err(|e| SmolError::custom_error(&format!("latest_eval query: {e}")))?;
        Ok(row)
    }

    /// Legacy `latest_eval`: the newest eval row for a model by `run_at DESC,
    /// id DESC`, with NO range filter. Used by `--eval-mode legacy` to revert
    /// to the pre-smart caching behavior. Returns `None` if the model has no
    /// eval rows.
    pub fn latest_eval_legacy(&self, model_id: &str) -> SmolResult<Option<EvalRecord>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, model_id, correct, total, correct_plus, total_plus,
                        correct_minus, total_minus, seed, eval_min, eval_max,
                        run_at
                 FROM evals WHERE model_id = ?1
                 ORDER BY run_at DESC, id DESC LIMIT 1",
            )
            .map_err(|e| SmolError::custom_error(&format!("latest_eval_legacy prepare: {e}")))?;
        let row = stmt
            .query_row(params![model_id], map_eval_row)
            .optional()
            .map_err(|e| SmolError::custom_error(&format!("latest_eval_legacy query: {e}")))?;
        Ok(row)
    }

    /// Insert (or overwrite) the cached exhaustive-eval-grid row for a model.
    /// One row per model (`model_id` is the table's PRIMARY KEY): unlike
    /// `record_eval`, which appends a history row per run, this OVERWRITES the
    /// previous cache on every recompute — see the `eval_grids` table's doc
    /// for why (the JSON blob is large; only the latest grid is ever useful).
    /// `eval_min`/`eval_max` are stamped from the model's *current* range so
    /// `latest_eval_grid` can later detect a stale cache the same way
    /// `latest_eval` does for the sampled eval.
    pub fn record_eval_grid(
        &self,
        model_id: &str,
        eval_min: i64,
        eval_max: i64,
        report_json: &str,
        correct: i64,
        total: i64,
    ) -> SmolResult<()> {
        self.conn
            .execute(
                "INSERT INTO eval_grids (model_id, eval_min, eval_max, report_json,
                                          correct, total, run_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(model_id) DO UPDATE SET
                    eval_min = excluded.eval_min,
                    eval_max = excluded.eval_max,
                    report_json = excluded.report_json,
                    correct = excluded.correct,
                    total = excluded.total,
                    run_at = excluded.run_at",
                params![model_id, eval_min, eval_max, report_json, correct, total, now_unix()],
            )
            .map_err(|e| SmolError::custom_error(&format!("record_eval_grid: {e}")))?;
        Ok(())
    }

    /// The cached eval-grid row for a model, but only if its stamped
    /// `eval_min`/`eval_max` still match the model's *current* range —
    /// otherwise `None`, exactly like `latest_eval`'s smart-mode range
    /// filter (a changed `--eval-min`/`--eval-max` must not silently serve a
    /// grid computed for the old range). If the model isn't registered at
    /// all, also returns `None` (there's no "current range" to compare
    /// against). Returns `None` if there's no cached row yet.
    pub fn latest_eval_grid(&self, model_id: &str) -> SmolResult<Option<EvalGridRecord>> {
        let Some(current) = self.get_model(model_id)? else {
            return Ok(None);
        };
        let mut stmt = self
            .conn
            .prepare(
                "SELECT model_id, eval_min, eval_max, report_json, correct, total, run_at
                 FROM eval_grids WHERE model_id = ?1",
            )
            .map_err(|e| SmolError::custom_error(&format!("latest_eval_grid prepare: {e}")))?;
        let row = stmt
            .query_row(params![model_id], map_eval_grid_row)
            .optional()
            .map_err(|e| SmolError::custom_error(&format!("latest_eval_grid query: {e}")))?;
        Ok(row.filter(|r| r.eval_min == current.eval_min && r.eval_max == current.eval_max))
    }

    /// Insert (or overwrite) the cached Jacobian-lens result row for a model.
    /// One row per model (`model_id` is the table's PRIMARY KEY) — unlike
    /// `record_eval`, this overwrites the previous cache on every recompute,
    /// mirroring `record_eval_grid` (there's no notion of a stale "range" for
    /// this analysis — it's keyed purely on the model's current weights, so a
    /// fresh run always simply replaces the old one).
    pub fn record_jacobian_lens(
        &self,
        model_id: &str,
        results_json: &str,
        plot_dir: &str,
        plot_files: &[String],
    ) -> SmolResult<()> {
        let plot_files_json = serde_json::to_string(plot_files)
            .map_err(|e| SmolError::custom_error(&format!("record_jacobian_lens: serialize plot_files: {e}")))?;
        self.conn
            .execute(
                "INSERT INTO jacobian_lens_results (model_id, results_json, plot_dir, plot_files, computed_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(model_id) DO UPDATE SET
                    results_json = excluded.results_json,
                    plot_dir = excluded.plot_dir,
                    plot_files = excluded.plot_files,
                    computed_at = excluded.computed_at",
                params![model_id, results_json, plot_dir, plot_files_json, now_unix()],
            )
            .map_err(|e| SmolError::custom_error(&format!("record_jacobian_lens: {e}")))?;
        Ok(())
    }

    /// The cached Jacobian-lens row for a model, if one exists. Unlike
    /// `latest_eval_grid`, there's no range-staleness check — this analysis
    /// only goes stale when the model's weights change (a re-train), which
    /// already cascade-deletes this row via the table's FK, so any row
    /// present is always for the model's CURRENT weights.
    pub fn latest_jacobian_lens(&self, model_id: &str) -> SmolResult<Option<JacobianLensRecord>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT model_id, results_json, plot_dir, plot_files, computed_at
                 FROM jacobian_lens_results WHERE model_id = ?1",
            )
            .map_err(|e| SmolError::custom_error(&format!("latest_jacobian_lens prepare: {e}")))?;
        let row = stmt
            .query_row(params![model_id], map_jacobian_lens_row)
            .optional()
            .map_err(|e| SmolError::custom_error(&format!("latest_jacobian_lens query: {e}")))?;
        Ok(row)
    }

    /// Hard cap on stored `checkpoint_grids` rows per model. Even with the
    /// epoch-gap throttle in `model::should_snapshot` (which already bounds a
    /// 10000-epoch run to ~400 snapshots), this is a second, independent
    /// safety net against unbounded growth — e.g. a very long or
    /// very-slow-converging run. 300 is comfortably above what the gap
    /// throttle alone produces for this project's typical run lengths (a few
    /// thousand epochs), so in practice thinning rarely triggers; it exists
    /// as a backstop, not the primary control.
    const MAX_CHECKPOINT_GRIDS_PER_MODEL: i64 = 300;

    /// Insert one exhaustive-eval-grid snapshot into the `checkpoint_grids`
    /// history, then thin old rows if the model has exceeded
    /// `MAX_CHECKPOINT_GRIDS_PER_MODEL`. Called from `train.rs`'s
    /// `on_best_loss` callback (see `LanguageModel::train_with_dropout`)
    /// once per throttled "new best smoothed loss" event.
    pub fn record_checkpoint_grid(
        &self,
        model_id: &str,
        epoch: usize,
        loss: f64,
        eval_min: i64,
        eval_max: i64,
        report_json: &str,
        correct: i64,
        total: i64,
    ) -> SmolResult<()> {
        self.conn
            .execute(
                "INSERT INTO checkpoint_grids (model_id, epoch, loss, eval_min, eval_max,
                                                report_json, correct, total, run_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    model_id,
                    epoch as i64,
                    loss,
                    eval_min,
                    eval_max,
                    report_json,
                    correct,
                    total,
                    now_unix(),
                ],
            )
            .map_err(|e| SmolError::custom_error(&format!("record_checkpoint_grid: {e}")))?;
        self.thin_checkpoint_grids(model_id)?;
        Ok(())
    }

    /// If `model_id` has more than `MAX_CHECKPOINT_GRIDS_PER_MODEL` rows,
    /// delete roughly half of them via UNIFORM thinning (drop every other row
    /// in epoch order, always keeping the very first and very last) rather
    /// than a strict "drop oldest" policy. Uniform thinning preserves the
    /// full epoch range (so the animation still has a start and end frame)
    /// while still reducing storage/row count, whereas dropping only the
    /// oldest rows would permanently destroy early-training resolution every
    /// time the cap is hit.
    fn thin_checkpoint_grids(&self, model_id: &str) -> SmolResult<()> {
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM checkpoint_grids WHERE model_id = ?1",
                params![model_id],
                |row| row.get(0),
            )
            .map_err(|e| SmolError::custom_error(&format!("thin_checkpoint_grids count: {e}")))?;
        if count <= Self::MAX_CHECKPOINT_GRIDS_PER_MODEL {
            return Ok(());
        }
        let ids: Vec<i64> = self
            .conn
            .prepare(
                "SELECT id FROM checkpoint_grids WHERE model_id = ?1 ORDER BY epoch ASC, id ASC",
            )
            .map_err(|e| SmolError::custom_error(&format!("thin_checkpoint_grids prepare: {e}")))?
            .query_map(params![model_id], |row| row.get(0))
            .map_err(|e| SmolError::custom_error(&format!("thin_checkpoint_grids query: {e}")))?
            .filter_map(|r| r.ok())
            .collect();
        let n = ids.len();
        for (i, id) in ids.iter().enumerate() {
            // Always keep the first and last row (preserve the full epoch
            // range); among the rest, drop every other one (even index).
            if i == 0 || i == n - 1 {
                continue;
            }
            if i % 2 == 0 {
                self.conn
                    .execute("DELETE FROM checkpoint_grids WHERE id = ?1", params![id])
                    .map_err(|e| {
                        SmolError::custom_error(&format!("thin_checkpoint_grids delete: {e}"))
                    })?;
            }
        }
        Ok(())
    }

    /// All `checkpoint_grids` rows for a model, ordered by `epoch ASC` — the
    /// full snapshot history a UI slider would animate through (oldest/lowest
    /// epoch first). Returns an empty `Vec` (not an error) if the model has
    /// no snapshots yet.
    pub fn list_checkpoint_grids(&self, model_id: &str) -> SmolResult<Vec<CheckpointGridRecord>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, model_id, epoch, loss, eval_min, eval_max, report_json,
                        correct, total, run_at
                 FROM checkpoint_grids WHERE model_id = ?1 ORDER BY epoch ASC, id ASC",
            )
            .map_err(|e| SmolError::custom_error(&format!("list_checkpoint_grids prepare: {e}")))?;
        let rows = stmt
            .query_map(params![model_id], map_checkpoint_grid_row)
            .map_err(|e| SmolError::custom_error(&format!("list_checkpoint_grids query: {e}")))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(|e| {
                SmolError::custom_error(&format!("list_checkpoint_grids row: {e}"))
            })?);
        }
        Ok(out)
    }

    /// Insert (or replace) a `trainings` row keyed on `(model_id, kind)`. Used
    /// for live progress: SFT calls this at each checkpoint (every 10 epochs +
    /// final), and RFT/GRPO call it after each round, so the web UI shows a
    /// single per-(model, kind) row that updates in place as training
    /// progresses — instead of 400 rows for 400 checkpoints. The
    /// `(model_id, kind)` pair is unique per model (a model has at most one
    /// SFT and one RFT/GRPO training), so the "one row per pair" invariant is
    /// the right grain. `latest_training` then returns the upserted row.
    ///
    /// Implemented as a single atomic `INSERT ... ON CONFLICT(model_id, kind)
    /// DO UPDATE`, relying on the UNIQUE index on `(model_id, kind)` to
    /// enforce the one-row-per-pair invariant at the schema level.
    ///
    /// `kind` is `"sft"`, `"rft"`, or `"grpo"`. One of `loss_trajectory_json` /
    /// `rft_summary_json` is the meaningful payload while the other is
    /// `"null"` (per the schema's NOT NULL constraints we always store a
    /// string, never SQL NULL). `early_stopped` is stored as 0/1; `final_loss`
    /// is REAL.
    /// `train_correct`/`train_total`: exact greedy-decoding accuracy over the
    /// literal training corpus (see `TrainingRecord` doc). `None` for every
    /// call except the final post-training upsert for an SFT row — passing
    /// `None` from an intermediate checkpoint or from RFT/GRPO leaves the
    /// column at its previous value only on `ON CONFLICT`'s first write; on a
    /// fresh INSERT it's simply NULL. Callers that don't have a freshly
    /// computed number (checkpoints, RFT/GRPO) MUST NOT overwrite a
    /// previously-computed final number with `None`, so this method treats
    /// `None` as "leave existing value alone" via `COALESCE` rather than a
    /// literal overwrite.
    ///
    /// `prompt_min`/`prompt_max`/`prompt_ops`: the operand range + ops this
    /// row's RL stage samples prompts from (`--rft-min`/`--rft-max`/
    /// `--rft-ops` or `--grpo-min`/`--grpo-max`/`--grpo-ops`); `None` for SFT
    /// calls (no prompt-sampling concept). Same `COALESCE`-on-`None`
    /// treatment as `train_correct`/`train_total` above, though in practice
    /// every RFT/GRPO round upserts the same constant value for the run.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_training(
        &self,
        model_id: &str,
        kind: &str,
        epochs_run: usize,
        early_stopped: bool,
        final_loss: f32,
        loss_trajectory_json: &str,
        rft_summary_json: &str,
        train_correct: Option<i64>,
        train_total: Option<i64>,
        prompt_min: Option<i64>,
        prompt_max: Option<i64>,
        prompt_ops: Option<&str>,
    ) -> SmolResult<()> {
        self.conn
            .execute(
                "INSERT INTO trainings (model_id, kind, epochs_run, early_stopped,
                                        final_loss, loss_trajectory, rft_summary,
                                        train_correct, train_total,
                                        prompt_min, prompt_max, prompt_ops, trained_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
                 ON CONFLICT(model_id, kind) DO UPDATE SET
                    epochs_run = excluded.epochs_run,
                    early_stopped = excluded.early_stopped,
                    final_loss = excluded.final_loss,
                    loss_trajectory = excluded.loss_trajectory,
                    rft_summary = excluded.rft_summary,
                    train_correct = COALESCE(excluded.train_correct, trainings.train_correct),
                    train_total = COALESCE(excluded.train_total, trainings.train_total),
                    prompt_min = COALESCE(excluded.prompt_min, trainings.prompt_min),
                    prompt_max = COALESCE(excluded.prompt_max, trainings.prompt_max),
                    prompt_ops = COALESCE(excluded.prompt_ops, trainings.prompt_ops),
                    trained_at = excluded.trained_at",
                params![
                    model_id,
                    kind,
                    epochs_run as i64,
                    if early_stopped { 1 } else { 0 },
                    final_loss as f64,
                    loss_trajectory_json,
                    rft_summary_json,
                    train_correct,
                    train_total,
                    prompt_min,
                    prompt_max,
                    prompt_ops,
                    now_unix(),
                ],
            )
            .map_err(|e| SmolError::custom_error(&format!("upsert_training: {e}")))?;
        Ok(())
    }

    /// Most recent `trainings` row for a model (highest `trained_at`, ties
    /// broken by highest `id`). Returns `None` if the model has no recorded
    /// training runs. Used by `--serve`'s `/api/models` to surface the latest
    /// loss trajectory / RFT summary on the model card. With `upsert_training`
    /// there's at most one row per `(model_id, kind)`, so this returns the
    /// newest kind's row (SFT vs RFT/GRPO) by `trained_at`.
    pub fn latest_training(&self, model_id: &str) -> SmolResult<Option<TrainingRecord>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, model_id, kind, epochs_run, early_stopped,
                        final_loss, loss_trajectory, rft_summary,
                        train_correct, train_total,
                        prompt_min, prompt_max, prompt_ops, trained_at
                 FROM trainings WHERE model_id = ?1
                 ORDER BY trained_at DESC, id DESC LIMIT 1",
            )
            .map_err(|e| SmolError::custom_error(&format!("latest_training prepare: {e}")))?;
        let row = stmt
            .query_row(params![model_id], map_training_row)
            .optional()
            .map_err(|e| SmolError::custom_error(&format!("latest_training query: {e}")))?;
        Ok(row)
    }

    /// One-time seeding: parse `models.toml` and insert each entry. Computes
    /// `params_estimate` from the *actual* tokenizer vocab (built from the
    /// corpus) so the estimate matches what `--serve` used to show. If the
    /// corpus can't be loaded for an entry, falls back to the TOML's
    /// `vocab_size` and prints a warning. Returns the number of entries
    /// imported.
    pub fn import_from_toml(&self, path: &Path) -> SmolResult<usize> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            SmolError::custom_error(&format!(
                "Failed to read {}: {e}",
                path.display()
            ))
        })?;
        let models_file: ModelsFile = toml::from_str(&content).map_err(|e| {
            SmolError::custom_error(&format!("Failed to parse {}: {e}", path.display()))
        })?;

        let project_root = std::env::current_dir()
            .map_err(|e| SmolError::custom_error(&format!("cwd: {e}")))?;

        let mut count = 0;
        for entry in &models_file.model {
            let actual_vocab =
                actual_vocab_for_entry(entry, &project_root).unwrap_or_else(|| {
                    eprintln!(
                        "[registry] WARNING: could not load corpus for '{}' \
                         (dataset='{}'); falling back to TOML vocab_size={} \
                         for params_estimate.",
                        entry.id, entry.dataset, entry.vocab_size
                    );
                    entry.vocab_size
                });
            let params_estimate = estimate_params(
                &entry.model_type,
                actual_vocab,
                entry.block_size,
                entry.hidden_size,
                entry.num_blocks,
            )
            .unwrap_or(0) as i64;
            let now = now_unix();
            let rec = ModelRecord {
                id: entry.id.clone(),
                path: entry.path.clone(),
                model_type: entry.model_type.clone(),
                tokenizer: entry.tokenizer.clone(),
                vocab_size: actual_vocab as i64,
                block_size: entry.block_size as i64,
                hidden_size: entry.hidden_size as i64,
                num_heads: entry.num_heads as i64,
                num_blocks: entry.num_blocks as i64,
                // The legacy TOML format has no per-block concept — every
                // imported entry is a uniform architecture, so a bare
                // (comma-free) number is the correct schedule string; it's
                // also what `parse_heads_schedule_column` would fall back to
                // anyway if this were left empty.
                heads_schedule: entry.num_heads.to_string(),
                // The legacy TOML format predates `--aligned-windows`
                // entirely -- there's no historical setting to recover, so
                // this is honestly "unknown" rather than a guessed default.
                aligned_windows: None,
                dataset: entry.dataset.clone(),
                dataset_name: entry.dataset_name.clone(),
                eval_min: entry.eval_min,
                eval_max: entry.eval_max,
                eval_samples: entry.eval_samples as i64,
                note: entry.note.clone(),
                params_estimate,
                base_model_id: None,
                created_at: now,
                updated_at: now,
            };
            self.register_model(&rec)?;
            count += 1;
        }
        Ok(count)
    }

    /// `SELECT COUNT(*) FROM models == 0` — used to decide whether to seed
    /// from `models.toml` on `--serve` start.
    pub fn is_empty(&self) -> SmolResult<bool> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM models", [], |row| row.get(0))
            .map_err(|e| SmolError::custom_error(&format!("is_empty: {e}")))?;
        Ok(count == 0)
    }
}

impl ModelRecord {
    /// Build a `ModelRecord` from training-time metadata + the `TrainOutcome`
    /// returned by `train_with_dropout`. Derives `id` from the model filename,
    /// `dataset_name` from the dataset filename stem, computes
    /// `params_estimate`, auto-generates `note`, and stamps both timestamps
    /// to "now".
    ///
    /// `eval_min`/`eval_max` resolution (per `meta.eval_mode`):
    /// - If the user passed both `--eval-min` and `--eval-max` (`Some`/`Some`),
    ///   those values are used verbatim (user override wins in both modes).
    /// - Otherwise, in `Smart` mode: scan the corpus at `meta.dataset_path`
    ///   via `dataset::operand_range` and use the corpus-derived min/max.
    ///   If the corpus can't be read or has no parseable lines, fall back to
    ///   0/999 (the legacy default) so registration never breaks.
    /// - In `Legacy` mode: skip the corpus scan and use 0/999 directly.
    pub fn from_training(meta: &TrainingMeta, outcome: &TrainOutcome) -> ModelRecord {
        let id = derive_id(meta.model_path);
        let path = meta.model_path.to_string_lossy().to_string();
        let model_type_str = match meta.model_type {
            ModelType::Gpt => "gpt",
            ModelType::Bigram => "bigram",
            ModelType::Ngram => "ngram",
        };
        let tokenizer_str = match meta.tokenizer {
            TokenizerType::Char => "char",
            TokenizerType::Bpe => "bpe",
        };
        let dataset = meta.dataset_path.to_string_lossy().to_string();
        let dataset_name = meta
            .dataset_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let dataset_filename = meta
            .dataset_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let params_estimate = estimate_params(
            model_type_str,
            meta.actual_vocab_size,
            meta.block_size,
            meta.hidden_size,
            meta.num_blocks,
        )
        .unwrap_or(0) as i64;
        // The `models.num_heads` column predates per-block schedules and is
        // a single INTEGER, so a non-uniform schedule can't be stored
        // directly there. Store the minimum entry (equal to the common
        // value when uniform) as the least-misleading single-number
        // summary, and put the actual per-block schedule + reload flag in
        // `note` (see `generate_note`'s doc).
        let representative_num_heads = meta.heads_schedule.iter().copied().min().unwrap_or(0) as i64;
        let note = generate_note(
            model_type_str,
            params_estimate,
            meta.block_size as i64,
            meta.hidden_size as i64,
            representative_num_heads,
            meta.num_blocks as i64,
            tokenizer_str,
            outcome.epochs_run,
            &dataset_filename,
            meta.seed,
            outcome.early_stopped,
            meta.heads_schedule,
        );
        let (eval_min, eval_max) = resolve_eval_range(meta);
        let now = now_unix();
        ModelRecord {
            id,
            path,
            model_type: model_type_str.to_string(),
            tokenizer: tokenizer_str.to_string(),
            vocab_size: meta.actual_vocab_size as i64,
            block_size: meta.block_size as i64,
            hidden_size: meta.hidden_size as i64,
            num_heads: representative_num_heads,
            num_blocks: meta.num_blocks as i64,
            // Lossless per-block schedule (CSV, same syntax `--num-heads`
            // accepts) — the source of truth for reloading this model's
            // exact architecture, unlike `num_heads` above.
            heads_schedule: heads_schedule_to_csv(meta.heads_schedule),
            // A live `--train`/`--rft`/`--grpo` run always knows the actual
            // CLI flag value used for this model's SFT stage (`with_variant`
            // copies it from the base's meta for RL variants).
            aligned_windows: Some(meta.aligned_windows),
            dataset,
            dataset_name,
            eval_min,
            eval_max,
            eval_samples: meta.eval_samples as i64,
            note,
            params_estimate,
            base_model_id: meta.base_model_id.map(|s| s.to_string()),
            created_at: now,
            updated_at: now,
        }
    }
}

/// Resolve the `eval_min`/`eval_max` to store on a `ModelRecord` from the
/// `TrainingMeta`'s `Option<i64>` CLI flags + `eval_mode`. User override
/// (`Some`/`Some`) wins in both modes; otherwise smart mode scans the corpus,
/// legacy mode uses 0/999. Extracted from `from_training` so the same rule
/// can be unit-tested without spinning up a `TrainOutcome`, and reused by
/// `train.rs`'s `--eval` branch (which needs concrete `i64` values for
/// `run_eval`).
pub fn resolve_eval_range(meta: &TrainingMeta) -> (i64, i64) {
    if let (Some(lo), Some(hi)) = (meta.eval_min, meta.eval_max) {
        return (lo, hi);
    }
    match meta.eval_mode {
        EvalMode::Smart => match std::fs::read_to_string(meta.dataset_path)
            .ok()
            .and_then(|corpus| crate::dataset::operand_range(&corpus))
        {
            Some((lo, hi)) => (lo, hi),
            None => (0, 999),
        },
        EvalMode::Legacy => (0, 999),
    }
}

// --- Free helpers ---

/// `rusqlite` row → `ModelRecord` mapper, shared by `list_models` and
/// `get_model` to avoid repeating the 20-column `get` list. `base_model_id`
/// is nullable — rows written before the migration (or base models) come back
/// as `None`. `heads_schedule` defaults to `''` (never NULL — see the
/// migration's doc) for rows written before that column existed.
fn map_model_row(row: &Row) -> rusqlite::Result<ModelRecord> {
    Ok(ModelRecord {
        id: row.get(0)?,
        path: row.get(1)?,
        model_type: row.get(2)?,
        tokenizer: row.get(3)?,
        vocab_size: row.get(4)?,
        block_size: row.get(5)?,
        hidden_size: row.get(6)?,
        num_heads: row.get(7)?,
        num_blocks: row.get(8)?,
        heads_schedule: row.get(9)?,
        aligned_windows: row.get(10)?,
        dataset: row.get(11)?,
        dataset_name: row.get(12)?,
        eval_min: row.get(13)?,
        eval_max: row.get(14)?,
        eval_samples: row.get(15)?,
        note: row.get(16)?,
        params_estimate: row.get(17)?,
        base_model_id: row.get(18)?,
        created_at: row.get(19)?,
        updated_at: row.get(20)?,
    })
}

/// `rusqlite` row → `EvalRecord` mapper, shared by `latest_eval` and
/// `latest_eval_legacy`. `eval_min`/`eval_max` are nullable — rows written
/// before the migration (or with an unregistered model) come back as `None`.
fn map_eval_row(row: &Row) -> rusqlite::Result<EvalRecord> {
    Ok(EvalRecord {
        id: row.get(0)?,
        model_id: row.get(1)?,
        correct: row.get(2)?,
        total: row.get(3)?,
        correct_plus: row.get(4)?,
        total_plus: row.get(5)?,
        correct_minus: row.get(6)?,
        total_minus: row.get(7)?,
        seed: row.get(8)?,
        eval_min: row.get(9)?,
        eval_max: row.get(10)?,
        run_at: row.get(11)?,
    })
}

/// `rusqlite` row → `EvalGridRecord` mapper for `latest_eval_grid`.
fn map_eval_grid_row(row: &Row) -> rusqlite::Result<EvalGridRecord> {
    Ok(EvalGridRecord {
        model_id: row.get(0)?,
        eval_min: row.get(1)?,
        eval_max: row.get(2)?,
        report_json: row.get(3)?,
        correct: row.get(4)?,
        total: row.get(5)?,
        run_at: row.get(6)?,
    })
}

/// `rusqlite` row → `JacobianLensRecord` mapper for `latest_jacobian_lens`.
/// `plot_files` is stored as a JSON array string; a parse failure degrades to
/// an empty Vec (logged by the caller if it wants) rather than failing the
/// whole row read.
fn map_jacobian_lens_row(row: &Row) -> rusqlite::Result<JacobianLensRecord> {
    let plot_files_json: String = row.get(3)?;
    let plot_files: Vec<String> = serde_json::from_str(&plot_files_json).unwrap_or_default();
    Ok(JacobianLensRecord {
        model_id: row.get(0)?,
        results_json: row.get(1)?,
        plot_dir: row.get(2)?,
        plot_files,
        computed_at: row.get(4)?,
    })
}

/// `rusqlite` row → `CheckpointGridRecord` mapper for `list_checkpoint_grids`.
fn map_checkpoint_grid_row(row: &Row) -> rusqlite::Result<CheckpointGridRecord> {
    Ok(CheckpointGridRecord {
        id: row.get(0)?,
        model_id: row.get(1)?,
        epoch: row.get(2)?,
        loss: row.get(3)?,
        eval_min: row.get(4)?,
        eval_max: row.get(5)?,
        report_json: row.get(6)?,
        correct: row.get(7)?,
        total: row.get(8)?,
        run_at: row.get(9)?,
    })
}

/// Idempotent in-place migration: add `eval_min`/`eval_max` columns to the
/// `evals` table. SQLite's `ALTER TABLE ADD COLUMN` leaves existing rows
/// `NULL`, which `latest_eval` treats as "matches any range" so pre-migration
/// rows stay visible until a ranged row supersedes them. Re-running this on
/// a DB that already has the columns is a no-op: we check `PRAGMA
/// table_info(evals)` first and only ALTER the missing columns.
fn migrate_add_eval_range_columns(conn: &Connection) -> SmolResult<()> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(evals)")
        .map_err(|e| SmolError::custom_error(&format!("migrate: table_info: {e}")))?;
    let rows = stmt
        .query_map([], |row| {
            let name: String = row.get(1)?;
            Ok(name)
        })
        .map_err(|e| SmolError::custom_error(&format!("migrate: table_info query: {e}")))?;
    let mut existing: std::collections::HashSet<String> = std::collections::HashSet::new();
    for r in rows {
        let name = r.map_err(|e| SmolError::custom_error(&format!("migrate: row: {e}")))?;
        existing.insert(name);
    }
    if !existing.contains("eval_min") {
        conn.execute_batch("ALTER TABLE evals ADD COLUMN eval_min INTEGER")
            .map_err(|e| SmolError::custom_error(&format!("migrate: add eval_min: {e}")))?;
    }
    if !existing.contains("eval_max") {
        conn.execute_batch("ALTER TABLE evals ADD COLUMN eval_max INTEGER")
            .map_err(|e| SmolError::custom_error(&format!("migrate: add eval_max: {e}")))?;
    }
    Ok(())
}

/// Idempotent in-place migration: add `train_correct`/`train_total` to the
/// `trainings` table. Checks `PRAGMA table_info(trainings)` first and only
/// ALTERs whichever column is missing, so re-running on an already-migrated
/// DB is a no-op.
fn migrate_add_train_accuracy_columns(conn: &Connection) -> SmolResult<()> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(trainings)")
        .map_err(|e| SmolError::custom_error(&format!("migrate train_accuracy: table_info: {e}")))?;
    let rows = stmt
        .query_map([], |row| {
            let name: String = row.get(1)?;
            Ok(name)
        })
        .map_err(|e| SmolError::custom_error(&format!("migrate train_accuracy: query: {e}")))?;
    let mut existing = std::collections::HashSet::new();
    for r in rows {
        let name = r.map_err(|e| SmolError::custom_error(&format!("migrate train_accuracy: row: {e}")))?;
        existing.insert(name);
    }
    if !existing.contains("train_correct") {
        conn.execute_batch("ALTER TABLE trainings ADD COLUMN train_correct INTEGER")
            .map_err(|e| SmolError::custom_error(&format!("migrate train_accuracy: add train_correct: {e}")))?;
    }
    if !existing.contains("train_total") {
        conn.execute_batch("ALTER TABLE trainings ADD COLUMN train_total INTEGER")
            .map_err(|e| SmolError::custom_error(&format!("migrate train_accuracy: add train_total: {e}")))?;
    }
    Ok(())
}

/// Idempotent in-place migration: add `heads_schedule` to the `models`
/// table — the lossless per-block head-count schedule (CSV string, same
/// syntax `--num-heads` accepts). `ALTER TABLE ADD COLUMN ... DEFAULT ''`
/// backfills existing rows with an empty string rather than NULL, so
/// `ModelRecord.heads_schedule` can stay a plain `String` (no `Option`
/// needed) — callers that need a schedule for an empty-string row should
/// fall back to treating `num_heads` as uniform (`vec![num_heads;
/// num_blocks]`), which is correct for every pre-migration row except the
/// one non-uniform experiment (`mask-test-4blocks-heads-1-1-4-4`) registered
/// before this column existed; that row is re-registered by the training
/// experiment that introduced it (see `train.rs`), not backfilled here,
/// since its true schedule was never stored anywhere structured to backfill
/// FROM. Checks `PRAGMA table_info(models)` first so re-running on an
/// already-migrated DB is a no-op.
fn migrate_add_heads_schedule_column(conn: &Connection) -> SmolResult<()> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(models)")
        .map_err(|e| SmolError::custom_error(&format!("migrate heads_schedule: table_info: {e}")))?;
    let rows = stmt
        .query_map([], |row| {
            let name: String = row.get(1)?;
            Ok(name)
        })
        .map_err(|e| SmolError::custom_error(&format!("migrate heads_schedule: query: {e}")))?;
    let mut existing = std::collections::HashSet::new();
    for r in rows {
        let name = r.map_err(|e| SmolError::custom_error(&format!("migrate heads_schedule: row: {e}")))?;
        existing.insert(name);
    }
    if !existing.contains("heads_schedule") {
        conn.execute_batch("ALTER TABLE models ADD COLUMN heads_schedule TEXT NOT NULL DEFAULT ''")
            .map_err(|e| SmolError::custom_error(&format!("migrate heads_schedule: add column: {e}")))?;
    }
    Ok(())
}

/// Idempotent in-place migration: add `aligned_windows` to the `models`
/// table -- whether this model's SFT stage sampled training windows only
/// from true fact boundaries (`--aligned-windows`; see
/// `ModelRecord.aligned_windows`'s doc). SQLite's `ALTER TABLE ADD COLUMN`
/// leaves existing rows NULL (unknown historical setting, since this flag was
/// never tracked before this migration) rather than a guessed default.
/// Checks `PRAGMA table_info(models)` first so re-running on an
/// already-migrated DB is a no-op.
fn migrate_add_aligned_windows_column(conn: &Connection) -> SmolResult<()> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(models)")
        .map_err(|e| SmolError::custom_error(&format!("migrate aligned_windows: table_info: {e}")))?;
    let rows = stmt
        .query_map([], |row| {
            let name: String = row.get(1)?;
            Ok(name)
        })
        .map_err(|e| SmolError::custom_error(&format!("migrate aligned_windows: query: {e}")))?;
    let mut existing = std::collections::HashSet::new();
    for r in rows {
        let name = r.map_err(|e| SmolError::custom_error(&format!("migrate aligned_windows: row: {e}")))?;
        existing.insert(name);
    }
    if !existing.contains("aligned_windows") {
        conn.execute_batch("ALTER TABLE models ADD COLUMN aligned_windows INTEGER")
            .map_err(|e| SmolError::custom_error(&format!("migrate aligned_windows: add column: {e}")))?;
    }
    Ok(())
}

/// Idempotent in-place migration: add `prompt_min`/`prompt_max`/`prompt_ops`
/// to the `trainings` table -- the operand range + ops an RFT/GRPO row's RL
/// stage actually sampled prompts from (see `TrainingRecord`'s doc). Existing
/// rows get NULL (unknown -- pre-migration RFT/GRPO runs, and every SFT row,
/// which has no prompt-sampling concept). Checks `PRAGMA table_info(trainings)`
/// first so re-running on an already-migrated DB is a no-op.
fn migrate_add_prompt_range_columns(conn: &Connection) -> SmolResult<()> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(trainings)")
        .map_err(|e| SmolError::custom_error(&format!("migrate prompt_range: table_info: {e}")))?;
    let rows = stmt
        .query_map([], |row| {
            let name: String = row.get(1)?;
            Ok(name)
        })
        .map_err(|e| SmolError::custom_error(&format!("migrate prompt_range: query: {e}")))?;
    let mut existing = std::collections::HashSet::new();
    for r in rows {
        let name = r.map_err(|e| SmolError::custom_error(&format!("migrate prompt_range: row: {e}")))?;
        existing.insert(name);
    }
    if !existing.contains("prompt_min") {
        conn.execute_batch("ALTER TABLE trainings ADD COLUMN prompt_min INTEGER")
            .map_err(|e| SmolError::custom_error(&format!("migrate prompt_range: add prompt_min: {e}")))?;
    }
    if !existing.contains("prompt_max") {
        conn.execute_batch("ALTER TABLE trainings ADD COLUMN prompt_max INTEGER")
            .map_err(|e| SmolError::custom_error(&format!("migrate prompt_range: add prompt_max: {e}")))?;
    }
    if !existing.contains("prompt_ops") {
        conn.execute_batch("ALTER TABLE trainings ADD COLUMN prompt_ops TEXT")
            .map_err(|e| SmolError::custom_error(&format!("migrate prompt_range: add prompt_ops: {e}")))?;
    }
    Ok(())
}

/// Idempotent in-place migration: add `base_model_id` to the `models` table.
/// SQLite's `ALTER TABLE ADD COLUMN` leaves existing rows NULL, which the UI
/// treats as "is a base model" (a top-level card). Re-running on a DB that
/// already has the column is a no-op: we check `PRAGMA table_info(models)`
/// first and only ALTER the missing column.
fn migrate_add_base_model_id_column(conn: &Connection) -> SmolResult<()> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(models)")
        .map_err(|e| SmolError::custom_error(&format!("migrate base_model_id: table_info: {e}")))?;
    let rows = stmt
        .query_map([], |row| {
            let name: String = row.get(1)?;
            Ok(name)
        })
        .map_err(|e| SmolError::custom_error(&format!("migrate base_model_id: query: {e}")))?;
    let mut existing = std::collections::HashSet::new();
    for r in rows {
        let name = r.map_err(|e| SmolError::custom_error(&format!("migrate base_model_id: row: {e}")))?;
        existing.insert(name);
    }
    if !existing.contains("base_model_id") {
        conn.execute_batch("ALTER TABLE models ADD COLUMN base_model_id TEXT")
            .map_err(|e| SmolError::custom_error(&format!("migrate base_model_id: add column: {e}")))?;
    }
    Ok(())
}

/// One-time best-effort backfill: for existing rows whose id ends with `-rft`
/// or `-grpo` (the conventional variant naming from the old `--rft`/`--grpo`
/// path that registered variants as top-level models) and whose stripped stem
/// is an existing model id, set `base_model_id` to that stripped id. This
/// groups legacy variant rows under their base so the UI's variant `<select>`
/// works for models trained before the `base_model_id` column existed.
///
/// Idempotent: only updates rows where `base_model_id IS NULL`, so re-running
/// on a DB where the backfill already happened is a no-op. Skips rows whose
/// stripped stem doesn't match an existing model id (e.g. `foo-grpo-smoke`,
/// which doesn't end in exactly `-rft`/`-grpo`, or `bar-rft` when `bar` isn't
/// registered) — those stay as top-level base models.
fn migrate_backfill_base_model_id(conn: &Connection) -> SmolResult<()> {
    // Collect all registered ids once so we can check existence without a
    // per-row subquery.
    let id_rows: Vec<String> = conn
        .prepare("SELECT id FROM models")
        .map_err(|e| SmolError::custom_error(&format!("backfill: prepare ids: {e}")))?
        .query_map([], |row| row.get(0))
        .map_err(|e| SmolError::custom_error(&format!("backfill: query ids: {e}")))?
        .filter_map(|r| r.ok())
        .collect();
    let known: std::collections::HashSet<&str> =
        id_rows.iter().map(|s| s.as_str()).collect();

    // Rows that might be variants: id ends in `-rft` or `-grpo` exactly.
    // Only select `id` (not `base_model_id`) since the WHERE clause already
    // filters to NULL base_model_id and reading a NULL column as String fails.
    let candidates: Vec<String> = conn
        .prepare("SELECT id FROM models WHERE base_model_id IS NULL")
        .map_err(|e| SmolError::custom_error(&format!("backfill: prepare candidates: {e}")))?
        .query_map([], |row| row.get(0))
        .map_err(|e| SmolError::custom_error(&format!("backfill: query candidates: {e}")))?
        .filter_map(|r| r.ok())
        .collect();

    for id in candidates {
        // `strip_variant_suffix` only returns `Some` when it actually strips a
        // non-empty `-rft`/`-grpo` suffix, so `base_id` can never equal `id`.
        let Some(base_id) = strip_variant_suffix(&id) else {
            continue;
        };
        if !known.contains(base_id.as_str()) {
            continue;
        }
        conn.execute(
            "UPDATE models SET base_model_id = ?1 WHERE id = ?2 AND base_model_id IS NULL",
            params![base_id, id],
        )
        .map_err(|e| SmolError::custom_error(&format!("backfill: update {id}: {e}")))?;
    }
    Ok(())
}

/// If `id` ends with `-rft` or `-grpo` exactly, return the stem with that
/// suffix stripped; otherwise `None`. E.g. `gpt-arithmetic-add-rft` →
/// `gpt-arithmetic-add`, `gpt-arithmetic-add-1digit-dedup-grpo` →
/// `gpt-arithmetic-add-1digit-dedup`, `gpt-arithmetic-add-1digit-grpo-smoke`
/// → `None` (doesn't end in exactly `-rft`/`-grpo`).
fn strip_variant_suffix(id: &str) -> Option<String> {
    for suffix in ["-rft", "-grpo"] {
        if let Some(stem) = id.strip_suffix(suffix) {
            if !stem.is_empty() {
                return Some(stem.to_string());
            }
        }
    }
    None
}

/// Derive a variant `.bin` path from a base model path by inserting `-rft` or
/// `-grpo` before the extension. E.g. `gpt-arithmetic-add-1digit.bin` →
/// `gpt-arithmetic-add-1digit-rft.bin`. Used by `train.rs`'s `--rft`/`--grpo`
/// branches so the base `.bin` is preserved and the variant is a separate
/// file (the old behavior mutated the base in place). The directory is
/// preserved; only the filename changes.
pub fn derive_variant_path(base_path: &Path, suffix: &str) -> std::path::PathBuf {
    let stem = base_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let ext = base_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("bin");
    base_path.with_file_name(format!("{stem}-{suffix}.{ext}"))
}


/// any future `list_trainings` helper. `early_stopped` is stored as 0/1 in
/// SQLite and decoded to `bool` here.
fn map_training_row(row: &Row) -> rusqlite::Result<TrainingRecord> {
    let early: i64 = row.get(4)?;
    Ok(TrainingRecord {
        id: row.get(0)?,
        model_id: row.get(1)?,
        kind: row.get(2)?,
        epochs_run: row.get(3)?,
        early_stopped: early != 0,
        final_loss: row.get(5)?,
        loss_trajectory_json: row.get(6)?,
        rft_summary_json: row.get(7)?,
        train_correct: row.get(8)?,
        train_total: row.get(9)?,
        prompt_min: row.get(10)?,
        prompt_max: row.get(11)?,
        prompt_ops: row.get(12)?,
        trained_at: row.get(13)?,
    })
}

/// Derive a registry id from a model filename: take the file stem, replace any
/// run of non-alphanumeric chars with a single `-`, lowercase, and trim
/// leading/trailing dashes. E.g. `gpt-arithmetic-1digit-nonneg.bin` →
/// `gpt-arithmetic-1digit-nonneg`.
pub fn derive_id(model_path: &Path) -> String {
    let stem = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let mut out = String::with_capacity(stem.len());
    let mut last_was_sep = false;
    for c in stem.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('-');
            last_was_sep = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Rough param count from the architecture. For GPT this sums token + position
/// embeddings, per-block attention/FFN/norm params, and the LM head; for
/// BigramLM it's `vocab^2`. Approximate (ignores small bias terms where noted)
/// but close enough to distinguish ~7K from ~78K models. Shared between
/// `import_from_toml` (seed) and `ModelRecord::from_training` (train-time) so
/// both paths produce identical estimates for the same arch.
/// `num_heads` isn't a parameter here: multi-head attention splits
/// `hidden_size` across heads without adding any weights, so the head count
/// doesn't affect the total parameter count.
pub fn estimate_params(
    model_type: &str,
    vocab_size: usize,
    block_size: usize,
    hidden_size: usize,
    num_blocks: usize,
) -> Option<usize> {
    Some(match model_type {
        "bigram" => vocab_size * vocab_size,
        // `block_size` stores NgramLM's context length (N - 1) — see
        // `train.rs`'s `--ngram-order` override. Params = the embedding
        // table's row*col count: vocab_size^(N-1) rows * vocab_size cols =
        // vocab_size^N. `checked_pow` guards against overflow for a
        // pathologically large N/vocab combination (falls back to
        // `usize::MAX` rather than panicking).
        "ngram" => {
            let n = (block_size + 1) as u32;
            vocab_size.checked_pow(n).unwrap_or(usize::MAX)
        }
        "gpt" => {
            let h = hidden_size;
            let b = block_size;
            let nb = num_blocks;
            // token + position embeddings
            let mut p = vocab_size * h + b * h;
            // per block: KQV (linear_no_bias) + proj (with bias) + FFN (with
            // bias) + 2 layer norms (weight + bias each).
            p += nb
                * (3 * h * h
                    + h * h + h
                    + h * 4 * h + 4 * h
                    + 4 * h * h + h
                    + 4 * h);
            // lm_head (linear_b with bias=true)
            p += h * vocab_size + vocab_size;
            p
        }
        _ => return None,
    })
}

/// Build the actual tokenizer for a TOML entry (so `import_from_toml` can
/// compute the real `vocab_size` + `params_estimate` instead of trusting the
/// TOML's `vocab_size` field, which is just a BPE target). Returns `None` if
/// the corpus can't be read.
fn actual_vocab_for_entry(entry: &ModelEntry, project_root: &Path) -> Option<usize> {
    let corpus_path = project_root.join(&entry.dataset);
    let corpus = std::fs::read_to_string(&corpus_path).ok()?;
    match entry.tokenizer.as_str() {
        "char" => {
            let t = SimpleTokenizer::new(&corpus);
            Some(t.vocab_size())
        }
        "bpe" => {
            let t = BpeTokenizer::train(&corpus, entry.vocab_size);
            Some(t.vocab_size())
        }
        _ => None,
    }
}

/// Format a per-block head-count schedule as the comma-separated string
/// `--num-heads` already accepts (e.g. `[1, 1, 4, 4]` -> `"1,1,4,4"`), so the
/// value stored in `models.heads_schedule` round-trips directly into a
/// `--num-heads` CLI value / `resolve_heads_schedule` input. Shared by
/// `generate_note` (the human-readable reload hint) and
/// `ModelRecord::from_training` (the lossless DB column).
fn heads_schedule_to_csv(heads_schedule: &[usize]) -> String {
    heads_schedule
        .iter()
        .map(|h| h.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Parse a `ModelRecord`'s stored `heads_schedule` CSV column back into a
/// per-block `Vec<usize>`, with a graceful fallback for rows written before
/// the column existed (or a malformed value): treat `num_heads` as a
/// uniform schedule (`vec![num_heads; num_blocks]`) — correct for every
/// pre-migration row (they were all uniform architectures; per-block
/// schedules didn't exist yet), and the same "best guess" the CLI itself
/// falls back to for a bare `--num-heads N`. Used by `--serve`'s model-load
/// endpoints so they reconstruct the ACTUAL per-block shapes for a
/// non-uniform model instead of a lossy uniform approximation derived from
/// `num_heads` alone.
pub fn parse_heads_schedule_column(heads_schedule: &str, num_heads: i64, num_blocks: i64) -> Vec<usize> {
    let uniform_fallback = || vec![num_heads.max(0) as usize; num_blocks.max(0) as usize];
    if heads_schedule.trim().is_empty() {
        return uniform_fallback();
    }
    let parsed: Result<Vec<usize>, _> = heads_schedule
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect();
    match parsed {
        Ok(v) if !v.is_empty() => v,
        _ => uniform_fallback(),
    }
}

/// Auto-generate the model `note` from training metadata + outcome. Format:
///
/// ```text
/// {model_type} {params}K params (block={b} hidden={h} heads={nh} blocks={nb}),
/// {tokenizer} tokenizer, trained {epochs_run} epochs on {dataset_filename},
/// seed {seed}{early_stop_clause}{heads_schedule_clause}
/// ```
///
/// where `early_stop_clause` = `, early-stopped at epoch {epochs_run}` if early
/// stopping fired, else empty. `{params}` rounds to whole K for >=10K models
/// and one decimal place for smaller ones (so 77582 → "78K", 7300 → "7.3K").
///
/// `heads={nh}` in the summary line is always the single representative value
/// (`min(heads_schedule)`, passed in as `num_heads` — matches the `models`
/// table's scalar column, which predates per-block schedules). For a
/// NON-uniform `heads_schedule` (i.e. not every block has the same head
/// count), `heads_schedule_clause` appends the exact per-block schedule and
/// the `--num-heads` value needed to reload this exact model, e.g.
/// `, heads-schedule=[1,2,4,8] (reload with --num-heads "1,2,4,8")` — a
/// uniform architecture already round-trips via the bare `heads={nh}` in the
/// summary, so the clause is omitted (empty) in that case.
fn generate_note(
    model_type: &str,
    params_estimate: i64,
    block_size: i64,
    hidden_size: i64,
    num_heads: i64,
    num_blocks: i64,
    tokenizer: &str,
    epochs_run: usize,
    dataset_filename: &str,
    seed: Option<u64>,
    early_stopped: bool,
    heads_schedule: &[usize],
) -> String {
    let params_k = if params_estimate >= 10_000 {
        format!("{:.0}K", params_estimate as f64 / 1000.0)
    } else {
        format!("{:.1}K", params_estimate as f64 / 1000.0)
    };
    let seed_str = match seed {
        Some(s) => s.to_string(),
        None => "random".to_string(),
    };
    let early_clause = if early_stopped {
        format!(", early-stopped at epoch {epochs_run}")
    } else {
        String::new()
    };
    let is_uniform = heads_schedule.windows(2).all(|w| w[0] == w[1]);
    let heads_schedule_clause = if heads_schedule.is_empty() || is_uniform {
        String::new()
    } else {
        let schedule_str = heads_schedule_to_csv(heads_schedule);
        format!(
            ", heads-schedule=[{schedule_str}] (reload with --num-heads \"{schedule_str}\")"
        )
    };
    format!(
        "{model_type} {params_k} params (block={block_size} hidden={hidden_size} heads={num_heads} blocks={num_blocks}), {tokenizer} tokenizer, trained {epochs_run} epochs on {dataset}, seed {seed}{early_clause}{heads_schedule_clause}",
        model_type = model_type,
        params_k = params_k,
        block_size = block_size,
        hidden_size = hidden_size,
        num_heads = num_heads,
        num_blocks = num_blocks,
        tokenizer = tokenizer,
        epochs_run = epochs_run,
        dataset = dataset_filename,
        seed = seed_str,
        early_clause = early_clause,
        heads_schedule_clause = heads_schedule_clause,
    )
}

/// Current time as Unix seconds (i64). Falls back to 0 if the system clock is
/// before `UNIX_EPOCH` (shouldn't happen in practice).
pub fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Format an `i64` Unix timestamp as an ISO 8601 UTC string:
/// `YYYY-MM-DDTHH:MM:SSZ`. Implemented from scratch (no `chrono`/`time` dep)
/// via Howard Hinnant's `civil_from_days` algorithm.
pub fn format_iso(unix_seconds: i64) -> String {
    let days = unix_seconds.div_euclid(86400);
    let rem = unix_seconds.rem_euclid(86400);
    let hour = rem / 3600;
    let min = (rem % 3600) / 60;
    let sec = rem % 60;
    let (y, m, d) = days_to_ymd(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, hour, min, sec
    )
}

/// Convert days-since-1970-01-01 to a `(year, month, day)` triple. Hinnant's
/// civil-from-days algorithm; handles negative days (pre-epoch) correctly.
fn days_to_ymd(days: i64) -> (i64, u32, u32) {
    let z = days + 719468; // shift epoch from 1970-01-01 to 0000-03-01
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// `serde` serializer for `i64` Unix seconds → ISO 8601 UTC string. Used via
/// `#[serde(serialize_with = "serialize_iso_i64")]` on the timestamp fields so
/// the DB stores cheap integers but the JSON API serves human-readable times.
fn serialize_iso_i64<S: serde::Serializer>(val: &i64, ser: S) -> Result<S::Ok, S::Error> {
    ser.serialize_str(&format_iso(*val))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_derive_id_basic() {
        assert_eq!(
            derive_id(&PathBuf::from("gpt-arithmetic-1digit-nonneg.bin")),
            "gpt-arithmetic-1digit-nonneg"
        );
        assert_eq!(
            derive_id(&PathBuf::from("gpt-arithmetic-1digit.bin")),
            "gpt-arithmetic-1digit"
        );
    }

    #[test]
    fn test_derive_id_replaces_non_alnum_runs() {
        // Multiple non-alphanumeric chars collapse to a single dash.
        assert_eq!(derive_id(&PathBuf::from("foo  bar__baz.bin")), "foo-bar-baz");
        // Leading/trailing dashes are trimmed.
        assert_eq!(derive_id(&PathBuf::from("__foo__.bin")), "foo");
        // Uppercase is lowercased.
        assert_eq!(derive_id(&PathBuf::from("GPT_Char.Bin")), "gpt-char");
    }

    #[test]
    fn test_estimate_params_bigram() {
        let p = estimate_params("bigram", 100, 16, 32, 2);
        assert_eq!(p, Some(100 * 100));
    }

    #[test]
    fn test_estimate_params_gpt_known() {
        // The 1-digit arithmetic model: char vocab ~14, h=32, b=32, nb=6.
        // Matches the "78K params" note in models.toml.
        let p = estimate_params("gpt", 14, 32, 32, 6).unwrap();
        assert!((p as i64 - 77582).abs() < 100, "got {p}, expected ~77582");
    }

    #[test]
    fn test_estimate_params_unknown_type() {
        assert_eq!(estimate_params("mystery", 100, 16, 32, 2), None);
    }

    #[test]
    fn test_format_iso_known_epoch() {
        // 1970-01-01T00:00:00Z
        assert_eq!(format_iso(0), "1970-01-01T00:00:00Z");
        // 2024-01-01T00:00:00Z = 1704067200
        assert_eq!(format_iso(1704067200), "2024-01-01T00:00:00Z");
        // 2024-07-12T04:20:00Z — spot check the date math.
        let s = format_iso(1720758000);
        assert!(s.starts_with("2024-07-12T04:20:00Z"), "got {s}");
    }

    #[test]
    fn test_format_iso_negative() {
        // Pre-epoch: -1 second → 1969-12-31T23:59:59Z
        assert_eq!(format_iso(-1), "1969-12-31T23:59:59Z");
    }

    #[test]
    fn test_generate_note_no_early_stop() {
        let note = generate_note(
            "gpt",
            77582,
            32,
            32,
            8,
            6,
            "char",
            2000,
            "arithmetic-1digit.txt",
            Some(42),
            false,
            &[8, 8, 8, 8, 8, 8],
        );
        assert!(note.contains("gpt 78K params"));
        assert!(note.contains("block=32 hidden=32 heads=8 blocks=6"));
        assert!(note.contains("char tokenizer"));
        assert!(note.contains("trained 2000 epochs on arithmetic-1digit.txt"));
        assert!(note.contains("seed 42"));
        assert!(!note.contains("early-stopped"));
        // Uniform schedule -> no heads-schedule clause (the bare `heads=8`
        // above already fully describes the architecture).
        assert!(!note.contains("heads-schedule"));
    }

    #[test]
    fn test_generate_note_with_early_stop() {
        let note = generate_note(
            "gpt",
            7300,
            16,
            16,
            4,
            2,
            "char",
            430,
            "arithmetic.txt",
            None,
            true,
            &[4, 4],
        );
        assert!(note.contains("gpt 7.3K params"));
        assert!(note.contains("seed random"));
        assert!(note.contains("early-stopped at epoch 430"));
    }

    /// A non-uniform `heads_schedule` must append the exact per-block
    /// schedule and the `--num-heads` value needed to reload this model —
    /// this is the ONLY place (short of re-running with the same flags) a
    /// user can recover the schedule, since the `models.num_heads` DB column
    /// only stores a single representative (min) value.
    #[test]
    fn test_generate_note_non_uniform_schedule_appends_reload_hint() {
        let note = generate_note(
            "gpt",
            7300,
            16,
            16,
            1, // representative (min) value stored in the DB column
            4,
            "char",
            4000,
            "arithmetic-add-1digit.txt",
            Some(42),
            false,
            &[1, 1, 4, 4],
        );
        assert!(note.contains("heads-schedule=[1,1,4,4]"));
        assert!(note.contains("reload with --num-heads \"1,1,4,4\""));
    }

    #[test]
    fn test_round_trip_register_and_list() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();
        assert!(reg.is_empty().unwrap());

        let now = now_unix();
        let rec = ModelRecord {
            id: "test-model".to_string(),
            path: "test.bin".to_string(),
            model_type: "gpt".to_string(),
            tokenizer: "char".to_string(),
            vocab_size: 14,
            block_size: 16,
            hidden_size: 16,
            num_heads: 4,
            num_blocks: 2,
            heads_schedule: "4,4".to_string(),
            aligned_windows: None,
            dataset: "data/arithmetic.txt".to_string(),
            dataset_name: "arithmetic".to_string(),
            eval_min: 0,
            eval_max: 9,
            eval_samples: 200,
            note: "test note".to_string(),
            params_estimate: 7300,
            base_model_id: None,
            created_at: now,
            updated_at: now,
        };
        reg.register_model(&rec).unwrap();
        assert!(!reg.is_empty().unwrap());

        let list = reg.list_models().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, "test-model");
        assert_eq!(list[0].params_estimate, 7300);

        // Re-register with same id updates (not duplicates).
        let mut rec2 = rec.clone();
        rec2.note = "updated note".to_string();
        reg.register_model(&rec2).unwrap();
        let list = reg.list_models().unwrap();
        assert_eq!(list.len(), 1, "re-register should not duplicate");
        assert_eq!(list[0].note, "updated note");

        // get_model.
        let got = reg.get_model("test-model").unwrap().unwrap();
        assert_eq!(got.id, "test-model");
        assert!(reg.get_model("nope").unwrap().is_none());
    }

    #[test]
    fn test_reregister_preserves_evals_and_created_at() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();

        let now = now_unix();
        let rec = ModelRecord {
            id: "m".to_string(),
            path: "m.bin".to_string(),
            model_type: "gpt".to_string(),
            tokenizer: "char".to_string(),
            vocab_size: 14,
            block_size: 16,
            hidden_size: 16,
            num_heads: 4,
            num_blocks: 2,
            heads_schedule: "4,4".to_string(),
            aligned_windows: None,
            dataset: "data/arithmetic.txt".to_string(),
            dataset_name: "arithmetic".to_string(),
            eval_min: 0,
            eval_max: 9,
            eval_samples: 200,
            note: "v1".to_string(),
            params_estimate: 7300,
            base_model_id: None,
            created_at: now,
            updated_at: now,
        };
        reg.register_model(&rec).unwrap();

        // Record an eval.
        let report = EvalReport {
            total: 10,
            correct: 3,
            correct_plus: 2,
            total_plus: 5,
            correct_minus: 1,
            total_minus: 5,
            by_digits: [(3, 10), (0, 0), (0, 0), (0, 0)],
            examples: Vec::new(),
        };
        reg.record_eval("m", &report, Some(1)).unwrap();
        assert_eq!(reg.latest_eval("m").unwrap().unwrap().correct, 3);

        // Re-register the same id with a new note + later updated_at. With the
        // UPSERT (ON CONFLICT(id) DO UPDATE), this updates in place — evals
        // survive and created_at is preserved (only updated_at moves forward).
        let later = now + 60;
        let mut rec2 = rec.clone();
        rec2.note = "v2".to_string();
        rec2.updated_at = later;
        reg.register_model(&rec2).unwrap();

        let list = reg.list_models().unwrap();
        assert_eq!(list.len(), 1, "re-register must not duplicate");
        assert_eq!(list[0].note, "v2");
        assert_eq!(list[0].created_at, now, "created_at must be preserved");
        assert_eq!(list[0].updated_at, later, "updated_at must move forward");

        // The eval must survive the re-register (INSERT OR REPLACE would have
        // cascade-deleted it via the FK ON DELETE CASCADE).
        let ev = reg.latest_eval("m").unwrap().unwrap();
        assert_eq!(ev.correct, 3, "eval history must survive re-register");
    }

    #[test]
    fn test_record_and_latest_eval() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();

        // Insert a model first (FK on evals.model_id).
        let now = now_unix();
        let rec = ModelRecord {
            id: "m1".to_string(),
            path: "m1.bin".to_string(),
            model_type: "gpt".to_string(),
            tokenizer: "char".to_string(),
            vocab_size: 14,
            block_size: 16,
            hidden_size: 16,
            num_heads: 4,
            num_blocks: 2,
            heads_schedule: "4,4".to_string(),
            aligned_windows: None,
            dataset: "data/arithmetic.txt".to_string(),
            dataset_name: "arithmetic".to_string(),
            eval_min: 0,
            eval_max: 9,
            eval_samples: 200,
            note: "n".to_string(),
            params_estimate: 7300,
            base_model_id: None,
            created_at: now,
            updated_at: now,
        };
        reg.register_model(&rec).unwrap();

        // No evals yet.
        assert!(reg.latest_eval("m1").unwrap().is_none());

        // Record an eval.
        let report = EvalReport {
            total: 200,
            correct: 15,
            correct_plus: 8,
            total_plus: 100,
            correct_minus: 7,
            total_minus: 100,
            by_digits: [(15, 200), (0, 0), (0, 0), (0, 0)],
            examples: Vec::new(),
        };
        reg.record_eval("m1", &report, Some(42)).unwrap();

        let latest = reg.latest_eval("m1").unwrap().unwrap();
        assert_eq!(latest.model_id, "m1");
        assert_eq!(latest.correct, 15);
        assert_eq!(latest.total, 200);
        assert_eq!(latest.seed, Some(42));
    }

    #[test]
    fn test_record_and_latest_training_sft() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();
        assert!(reg.latest_training("nope").unwrap().is_none());

        // Record an SFT training. loss_trajectory is a JSON array of f32; the
        // rft_summary slot is "null" (NOT NULL column, so we store the string).
        let losses = vec![7.49f32, 1.65, 1.34, 1.30, 1.24];
        let loss_json = serde_json::to_string(&losses).unwrap();
        reg.upsert_training(
            "gpt-arithmetic-add",
            "sft",
            2000,
            false,
            1.238,
            &loss_json,
            "null",
            Some(42),
            Some(55),
            None,
            None,
            None,
        )
        .unwrap();

        let t = reg.latest_training("gpt-arithmetic-add").unwrap().unwrap();
        assert_eq!(t.model_id, "gpt-arithmetic-add");
        assert_eq!(t.kind, "sft");
        assert_eq!(t.epochs_run, 2000);
        assert!(!t.early_stopped);
        assert!((t.final_loss - 1.238).abs() < 1e-6);
        assert_eq!(t.loss_trajectory_json, loss_json);
        assert_eq!(t.rft_summary_json, "null");
        assert_eq!(t.train_correct, Some(42));
        assert_eq!(t.train_total, Some(55));
        assert_eq!(t.prompt_min, None);
        assert_eq!(t.prompt_max, None);
        assert_eq!(t.prompt_ops, None);

        // Round-trip the loss JSON.
        let parsed: Vec<f32> =
            serde_json::from_str(&t.loss_trajectory_json).unwrap();
        assert_eq!(parsed, losses);
    }

    #[test]
    fn test_record_and_latest_training_rft() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();

        // Record an RFT training. loss_trajectory is "null"; rft_summary is a
        // serialized RftSummary.
        let summary = RftSummary {
            rounds: 3,
            winner_counts: vec![706, 754, 710],
            winner_rates: vec![70.6, 75.4, 71.0],
            eval_correct_pct: vec![9.5, 15.5, 11.0],
            per_round_sft_final_losses: vec![Some(0.512), Some(0.341), Some(0.288)],
        };
        let summary_json = serde_json::to_string(&summary).unwrap();
        reg.upsert_training(
            "gpt-arithmetic-add-rft",
            "rft",
            3,
            false,
            0.0,
            "null",
            &summary_json,
            None,
            None,
            Some(0),
            Some(999),
            Some("+,-"),
        )
        .unwrap();

        let t = reg.latest_training("gpt-arithmetic-add-rft").unwrap().unwrap();
        assert_eq!(t.kind, "rft");
        assert_eq!(t.epochs_run, 3);
        assert_eq!(t.loss_trajectory_json, "null");
        // Round-trip the RFT summary JSON.
        let parsed: RftSummary =
            serde_json::from_str(&t.rft_summary_json).unwrap();
        assert_eq!(parsed, summary);
        // Round-trip the RL-stage prompt-sampling range/ops.
        assert_eq!(t.prompt_min, Some(0));
        assert_eq!(t.prompt_max, Some(999));
        assert_eq!(t.prompt_ops.as_deref(), Some("+,-"));
    }

    #[test]
    fn test_latest_training_picks_most_recent() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();

        // Two trainings for the same model. The second insert happens later in
        // wall-clock time (trained_at is the order key), so latest_training
        // must return it.
        reg.upsert_training("m", "sft", 100, false, 5.0, "[5.0]", "null", None, None, None, None, None)
            .unwrap();
        // Tiny sleep so `now_unix()` advances by at least 1 second, making the
        // second row strictly newer. (If both rows share the same second, the
        // tie-breaker on `id DESC` still picks the second row.)
        std::thread::sleep(std::time::Duration::from_millis(1100));
        reg.upsert_training("m", "sft", 200, true, 1.2, "[1.2]", "null", None, None, None, None, None)
            .unwrap();

        let t = reg.latest_training("m").unwrap().unwrap();
        assert_eq!(t.epochs_run, 200, "latest_training must pick the newer row");
        assert!(t.early_stopped);
    }

    // --- evals.eval_min / eval_max: range-matched filtering ---

    /// Helper: build a `ModelRecord` with a specific `eval_min`/`eval_max`
    /// range and register it. Used by the filtering tests below.
    fn register_model_with_range(reg: &Registry, id: &str, lo: i64, hi: i64) {
        let now = now_unix();
        let rec = ModelRecord {
            id: id.to_string(),
            path: format!("{id}.bin"),
            model_type: "gpt".to_string(),
            tokenizer: "char".to_string(),
            vocab_size: 14,
            block_size: 16,
            hidden_size: 16,
            num_heads: 4,
            num_blocks: 2,
            heads_schedule: "4,4".to_string(),
            aligned_windows: None,
            dataset: "data/arithmetic.txt".to_string(),
            dataset_name: "arithmetic".to_string(),
            eval_min: lo,
            eval_max: hi,
            eval_samples: 200,
            note: "n".to_string(),
            params_estimate: 7300,
            base_model_id: None,
            created_at: now,
            updated_at: now,
        };
        reg.register_model(&rec).unwrap();
    }

    /// Build a trivial `EvalReport` with `correct` as the only varying field,
    /// so tests can identify which row `latest_eval` returned.
    fn report_with_correct(correct: i64) -> EvalReport {
        EvalReport {
            total: 200,
            correct: correct as usize,
            correct_plus: correct as usize,
            total_plus: 200,
            correct_minus: 0,
            total_minus: 0,
            by_digits: [(correct as usize, 200), (0, 0), (0, 0), (0, 0)],
            examples: Vec::new(),
        }
    }

    #[test]
    fn test_record_eval_stamps_model_range() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();
        register_model_with_range(&reg, "m", 0, 9);

        reg.record_eval("m", &report_with_correct(7), Some(42))
            .unwrap();
        let ev = reg.latest_eval("m").unwrap().unwrap();
        assert_eq!(ev.eval_min, Some(0), "eval row should be stamped with model's eval_min");
        assert_eq!(ev.eval_max, Some(9), "eval row should be stamped with model's eval_max");
    }

    #[test]
    fn test_latest_eval_filters_out_stale_range_rows() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();

        // Register the model with the OLD range (0..999) and record a stale
        // eval row stamped with that range.
        register_model_with_range(&reg, "m", 0, 999);
        reg.record_eval("m", &report_with_correct(0), Some(1))
            .unwrap();
        // Tiny sleep so the second row is strictly newer (run_at advances).
        std::thread::sleep(std::time::Duration::from_millis(1100));

        // Now fix the model's range to 0..9 and record a fresh eval row.
        register_model_with_range(&reg, "m", 0, 9);
        reg.record_eval("m", &report_with_correct(28), Some(2))
            .unwrap();

        // Smart mode: latest_eval must skip the stale 0..999 row and return
        // the fresh 0..9 row, even though the stale row is older. The 28
        // (not 0) proves we got the new row.
        let smart = reg.latest_eval("m").unwrap().unwrap();
        assert_eq!(smart.correct, 28, "smart latest_eval must skip stale-range row");
        assert_eq!(smart.eval_min, Some(0));
        assert_eq!(smart.eval_max, Some(9));

        // Legacy mode: latest_eval_legacy returns the newest row regardless
        // of range — also the 28 row here (it's newer), but the contract is
        // "no range filter", so this just confirms the legacy path works.
        let legacy = reg.latest_eval_legacy("m").unwrap().unwrap();
        assert_eq!(legacy.correct, 28, "legacy latest_eval returns newest row");
    }

    #[test]
    fn test_latest_eval_null_range_rows_match_anything() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();
        register_model_with_range(&reg, "m", 0, 9);

        // Insert a NULL-range eval row directly via raw SQL (simulating a
        // pre-migration row that predates the eval_min/eval_max columns).
        reg.conn
            .execute(
                "INSERT INTO evals (model_id, correct, total, correct_plus,
                                    total_plus, correct_minus, total_minus,
                                    seed, eval_min, eval_max, run_at)
                 VALUES ('m', 12, 200, 12, 200, 0, 0, NULL, NULL, NULL, ?1)",
                params![now_unix()],
            )
            .unwrap();

        // NULL-range row matches the model's current 0..9 range, so
        // latest_eval (smart) returns it.
        let ev = reg.latest_eval("m").unwrap().unwrap();
        assert_eq!(ev.correct, 12);
        assert_eq!(ev.eval_min, None, "NULL-range row should round-trip as None");
        assert_eq!(ev.eval_max, None);
    }

    #[test]
    fn test_latest_eval_smart_returns_none_when_only_mismatched_rows_exist() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();

        // Register with the OLD range and stamp an eval row with that range.
        register_model_with_range(&reg, "m", 0, 999);
        reg.record_eval("m", &report_with_correct(0), Some(1))
            .unwrap();

        // Now fix the range to 0..9 — but don't record a fresh eval row.
        // The only existing row is stamped 0..999, which doesn't match 0..9.
        register_model_with_range(&reg, "m", 0, 9);

        // Smart mode: no matching row → None. This is the "instant
        // invalidation" property: a range fix hides stale rows without
        // deleting them.
        assert!(
            reg.latest_eval("m").unwrap().is_none(),
            "smart latest_eval must hide stale-range rows when no match exists"
        );

        // Legacy mode: still returns the stale row (no range filter).
        let legacy = reg.latest_eval_legacy("m").unwrap().unwrap();
        assert_eq!(legacy.correct, 0, "legacy latest_eval ignores range filter");
    }

    #[test]
    fn test_record_and_latest_eval_grid_round_trip() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();
        register_model_with_range(&reg, "m", 0, 9);

        assert!(reg.latest_eval_grid("m").unwrap().is_none());

        let report_json = r#"{"min":0,"max":9,"grids":[],"correct":12,"total":100}"#;
        reg.record_eval_grid("m", 0, 9, report_json, 12, 100)
            .unwrap();

        let cached = reg.latest_eval_grid("m").unwrap().unwrap();
        assert_eq!(cached.model_id, "m");
        assert_eq!(cached.eval_min, 0);
        assert_eq!(cached.eval_max, 9);
        assert_eq!(cached.correct, 12);
        assert_eq!(cached.total, 100);
        assert_eq!(cached.report_json, report_json);
    }

    #[test]
    fn test_record_and_latest_jacobian_lens_round_trip() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();
        register_model_with_range(&reg, "m", 0, 9);

        assert!(reg.latest_jacobian_lens("m").unwrap().is_none());

        let results_json = r#"{"greedy_accuracy":0.9,"n_facts":55}"#;
        let plot_files = vec!["layer_accuracy.png".to_string(), "group_rank.png".to_string()];
        reg.record_jacobian_lens("m", results_json, "jacobian_lens_output/m", &plot_files)
            .unwrap();

        let cached = reg.latest_jacobian_lens("m").unwrap().unwrap();
        assert_eq!(cached.model_id, "m");
        assert_eq!(cached.results_json, results_json);
        assert_eq!(cached.plot_dir, "jacobian_lens_output/m");
        assert_eq!(cached.plot_files, plot_files);
    }

    #[test]
    fn test_record_jacobian_lens_overwrites_in_place_not_history() {
        // Same "single row per model, overwritten on recompute" contract as
        // `eval_grids` — a second `record_jacobian_lens` call for the same
        // model must overwrite, not add a second row.
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();
        register_model_with_range(&reg, "m", 0, 9);

        reg.record_jacobian_lens("m", "{\"first\":true}", "dir1", &[])
            .unwrap();
        reg.record_jacobian_lens("m", "{\"second\":true}", "dir2", &["a.png".to_string()])
            .unwrap();

        let count: i64 = reg
            .conn
            .query_row("SELECT COUNT(*) FROM jacobian_lens_results WHERE model_id = 'm'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1, "jacobian_lens_results should hold exactly one row per model");

        let cached = reg.latest_jacobian_lens("m").unwrap().unwrap();
        assert_eq!(cached.results_json, "{\"second\":true}");
        assert_eq!(cached.plot_dir, "dir2");
        assert_eq!(cached.plot_files, vec!["a.png".to_string()]);
    }

    #[test]
    fn test_latest_jacobian_lens_unregistered_model_returns_none() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();
        assert!(reg.latest_jacobian_lens("nope").unwrap().is_none());
    }

    #[test]
    fn test_record_eval_grid_overwrites_in_place_not_history() {
        // Unlike `evals` (append-only history), `eval_grids` keeps only the
        // latest row per model — a second `record_eval_grid` call for the
        // same model must overwrite, not add a second row.
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();
        register_model_with_range(&reg, "m", 0, 9);

        reg.record_eval_grid("m", 0, 9, "{\"first\":true}", 1, 100)
            .unwrap();
        reg.record_eval_grid("m", 0, 9, "{\"second\":true}", 2, 100)
            .unwrap();

        let count: i64 = reg
            .conn
            .query_row("SELECT COUNT(*) FROM eval_grids WHERE model_id = 'm'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1, "eval_grids should hold exactly one row per model");

        let cached = reg.latest_eval_grid("m").unwrap().unwrap();
        assert_eq!(cached.correct, 2);
        assert_eq!(cached.report_json, "{\"second\":true}");
    }

    #[test]
    fn test_latest_eval_grid_hides_stale_range_cache() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();

        // Cache a grid computed for the OLD range (0..9).
        register_model_with_range(&reg, "m", 0, 9);
        reg.record_eval_grid("m", 0, 9, "{\"stale\":true}", 5, 100)
            .unwrap();

        // Now the model's configured range changes to 0..19 — the cached
        // grid (computed for 0..9) must be treated as absent, not served
        // stale, exactly like `latest_eval`'s smart-mode range filter.
        register_model_with_range(&reg, "m", 0, 19);
        assert!(
            reg.latest_eval_grid("m").unwrap().is_none(),
            "latest_eval_grid must hide a cache stamped with a superseded range"
        );

        // A fresh grid recomputed for the new range replaces the cache (still
        // one row) and is now visible.
        reg.record_eval_grid("m", 0, 19, "{\"fresh\":true}", 40, 400)
            .unwrap();
        let cached = reg.latest_eval_grid("m").unwrap().unwrap();
        assert_eq!(cached.eval_min, 0);
        assert_eq!(cached.eval_max, 19);
        assert_eq!(cached.correct, 40);
    }

    #[test]
    fn test_latest_eval_grid_unregistered_model_returns_none() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();
        assert!(reg.latest_eval_grid("nope").unwrap().is_none());
    }

    #[test]
    fn test_resolve_eval_range_user_override_wins() {
        // User explicitly passes both flags → use those, regardless of mode.
        let dir = temp_dir::TempDir::new().unwrap();
        // A corpus file we control — used as `dataset_path` for the meta.
        let corpus_path = dir.path().join("corpus.txt");
        std::fs::write(&corpus_path, "3+4=7\n9+9=18\n").unwrap();
        let model_path = dir.path().join("model.bin");

        let meta = TrainingMeta {
            model_type: ModelType::Gpt,
            tokenizer: TokenizerType::Char,
            block_size: 16,
            hidden_size: 16,
            heads_schedule: &[4, 4],
            num_blocks: 2,
            aligned_windows: false,
            dataset_path: &corpus_path,
            model_path: &model_path,
            actual_vocab_size: 14,
            eval_min: Some(5),
            eval_max: Some(50),
            eval_samples: 200,
            eval_mode: EvalMode::Smart,
            seed: Some(42),
            base_model_id: None,
        };
        assert_eq!(resolve_eval_range(&meta), (5, 50));
    }

    #[test]
    fn test_resolve_eval_range_smart_uses_corpus() {
        // Flags omitted in smart mode → corpus-derived range.
        let dir = temp_dir::TempDir::new().unwrap();
        let corpus_path = dir.path().join("corpus.txt");
        std::fs::write(&corpus_path, "3+4=7\n9+9=18\n0+0=0\n").unwrap();
        let model_path = dir.path().join("model.bin");

        let meta = TrainingMeta {
            model_type: ModelType::Gpt,
            tokenizer: TokenizerType::Char,
            block_size: 16,
            hidden_size: 16,
            heads_schedule: &[4, 4],
            num_blocks: 2,
            aligned_windows: false,
            dataset_path: &corpus_path,
            model_path: &model_path,
            actual_vocab_size: 14,
            eval_min: None,
            eval_max: None,
            eval_samples: 200,
            eval_mode: EvalMode::Smart,
            seed: Some(42),
            base_model_id: None,
        };
        assert_eq!(resolve_eval_range(&meta), (0, 9));
    }

    #[test]
    fn test_resolve_eval_range_legacy_uses_default() {
        // Flags omitted in legacy mode → 0/999 (no corpus scan).
        let dir = temp_dir::TempDir::new().unwrap();
        let corpus_path = dir.path().join("corpus.txt");
        // Even with a 1-digit corpus, legacy mode must use 0/999.
        std::fs::write(&corpus_path, "3+4=7\n9+9=18\n").unwrap();
        let model_path = dir.path().join("model.bin");

        let meta = TrainingMeta {
            model_type: ModelType::Gpt,
            tokenizer: TokenizerType::Char,
            block_size: 16,
            hidden_size: 16,
            heads_schedule: &[4, 4],
            num_blocks: 2,
            aligned_windows: false,
            dataset_path: &corpus_path,
            model_path: &model_path,
            actual_vocab_size: 14,
            eval_min: None,
            eval_max: None,
            eval_samples: 200,
            eval_mode: EvalMode::Legacy,
            seed: Some(42),
            base_model_id: None,
        };
        assert_eq!(resolve_eval_range(&meta), (0, 999));
    }

    #[test]
    fn test_resolve_eval_range_smart_falls_back_when_corpus_unreadable() {
        // Corpus path doesn't exist → smart mode falls back to 0/999.
        let dir = temp_dir::TempDir::new().unwrap();
        let corpus_path = dir.path().join("does-not-exist.txt");
        let model_path = dir.path().join("model.bin");

        let meta = TrainingMeta {
            model_type: ModelType::Gpt,
            tokenizer: TokenizerType::Char,
            block_size: 16,
            hidden_size: 16,
            heads_schedule: &[4, 4],
            num_blocks: 2,
            aligned_windows: false,
            dataset_path: &corpus_path,
            model_path: &model_path,
            actual_vocab_size: 14,
            eval_min: None,
            eval_max: None,
            eval_samples: 200,
            eval_mode: EvalMode::Smart,
            seed: Some(42),
            base_model_id: None,
        };
        assert_eq!(resolve_eval_range(&meta), (0, 999));
    }

    #[test]
    fn test_resolve_eval_range_partial_override_uses_corpus() {
        // Only one flag passed → not a full override; smart mode still
        // derives from corpus (per the spec: "If both are Some → use those").
        let dir = temp_dir::TempDir::new().unwrap();
        let corpus_path = dir.path().join("corpus.txt");
        std::fs::write(&corpus_path, "3+4=7\n9+9=18\n0+0=0\n").unwrap();
        let model_path = dir.path().join("model.bin");

        let meta = TrainingMeta {
            model_type: ModelType::Gpt,
            tokenizer: TokenizerType::Char,
            block_size: 16,
            hidden_size: 16,
            heads_schedule: &[4, 4],
            num_blocks: 2,
            aligned_windows: false,
            dataset_path: &corpus_path,
            model_path: &model_path,
            actual_vocab_size: 14,
            eval_min: Some(2),
            eval_max: None,
            eval_samples: 200,
            eval_mode: EvalMode::Smart,
            seed: Some(42),
            base_model_id: None,
        };
        assert_eq!(resolve_eval_range(&meta), (0, 9));
    }

    #[test]
    fn test_migrate_eval_range_columns_is_idempotent() {
        // Opening the same DB twice must not error on the second migration
        // attempt (the ALTER TABLE would fail if we didn't check PRAGMA
        // table_info first).
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let _ = Registry::open_at(&db).unwrap();
        // Second open re-runs migrate_add_eval_range_columns — must succeed.
        let reg = Registry::open_at(&db).unwrap();
        register_model_with_range(&reg, "m", 0, 9);
        reg.record_eval("m", &report_with_correct(1), Some(1))
            .unwrap();
        let ev = reg.latest_eval("m").unwrap().unwrap();
        assert_eq!(ev.eval_min, Some(0));
        assert_eq!(ev.eval_max, Some(9));
    }

    // --- upsert_training: in-place update keyed on (model_id, kind) ---

    #[test]
    fn test_upsert_training_replaces_in_place() {
        // Two upsert_training calls for the same (model_id, kind) must leave
        // exactly one row, holding the second call's data. This is the
        // "live progress" property: 400 checkpoints → 1 row that updates in
        // place, not 400 rows.
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();

        reg.upsert_training("m", "sft", 10, false, 5.0, "[5.0, 4.8]", "null", None, None, None, None, None)
            .unwrap();
        reg.upsert_training("m", "sft", 20, false, 1.2, "[5.0, 4.8, 1.2]", "null", None, None, None, None, None)
            .unwrap();

        let t = reg.latest_training("m").unwrap().unwrap();
        assert_eq!(t.epochs_run, 20, "upsert must hold the SECOND call's data");
        assert!((t.final_loss - 1.2).abs() < 1e-6);
        assert_eq!(t.loss_trajectory_json, "[5.0, 4.8, 1.2]");

        // Confirm only one row exists for (m, sft).
        let count: i64 = reg
            .conn
            .query_row(
                "SELECT COUNT(*) FROM trainings WHERE model_id = 'm' AND kind = 'sft'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "upsert_training must not duplicate rows");
    }

    #[test]
    fn test_upsert_training_distinct_kinds_coexist() {
        // A model can have both an SFT and an RFT/GRPO training row; upsert
        // on one kind must not touch the other.
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();

        reg.upsert_training("m", "sft", 2000, false, 1.2, "[1.2]", "null", None, None, None, None, None)
            .unwrap();
        reg.upsert_training("m", "rft", 3, false, 0.5, "null", "{\"rounds\":3}", None, None, None, None, None)
            .unwrap();

        // Two rows: one per kind.
        let count: i64 = reg
            .conn
            .query_row(
                "SELECT COUNT(*) FROM trainings WHERE model_id = 'm'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 2, "distinct kinds must coexist as separate rows");

        // Upserting SFT again must not touch the RFT row.
        reg.upsert_training("m", "sft", 4000, true, 0.9, "[0.9]", "null", None, None, None, None, None)
            .unwrap();
        let count: i64 = reg
            .conn
            .query_row(
                "SELECT COUNT(*) FROM trainings WHERE model_id = 'm'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 2, "upserting one kind must not duplicate the other");
    }

    // --- base_model_id: migration + grouping ---

    #[test]
    fn test_strip_variant_suffix() {
        assert_eq!(strip_variant_suffix("gpt-arithmetic-add-rft"), Some("gpt-arithmetic-add".to_string()));
        assert_eq!(strip_variant_suffix("gpt-arithmetic-add-1digit-dedup-grpo"), Some("gpt-arithmetic-add-1digit-dedup".to_string()));
        // Doesn't end in exactly `-rft`/`-grpo` → None.
        assert_eq!(strip_variant_suffix("gpt-arithmetic-add-1digit-grpo-smoke"), None);
        assert_eq!(strip_variant_suffix("gpt-arithmetic-add"), None);
        assert_eq!(strip_variant_suffix("foo-bar-baz"), None);
    }

    #[test]
    fn test_derive_variant_path_basic() {
        use std::path::PathBuf;
        assert_eq!(
            derive_variant_path(&PathBuf::from("gpt-arithmetic-add-1digit.bin"), "rft"),
            PathBuf::from("gpt-arithmetic-add-1digit-rft.bin")
        );
        assert_eq!(
            derive_variant_path(&PathBuf::from("foo.bin"), "grpo"),
            PathBuf::from("foo-grpo.bin")
        );
        // Directory is preserved; only the filename changes.
        assert_eq!(
            derive_variant_path(&PathBuf::from("models/gpt.bin"), "rft"),
            PathBuf::from("models/gpt-rft.bin")
        );
        // Missing extension falls back to .bin.
        assert_eq!(
            derive_variant_path(&PathBuf::from("models/gpt"), "rft"),
            PathBuf::from("models/gpt-rft.bin")
        );
    }

    #[test]
    fn test_base_model_id_column_migration_is_idempotent() {
        // Opening the same DB twice must not error on the second
        // base_model_id migration attempt.
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let _ = Registry::open_at(&db).unwrap();
        let reg = Registry::open_at(&db).unwrap();
        register_model_with_range(&reg, "m", 0, 9);
        let m = reg.get_model("m").unwrap().unwrap();
        assert_eq!(m.base_model_id, None, "base model has base_model_id NULL");
    }

    #[test]
    fn test_heads_schedule_column_migration_is_idempotent() {
        // Opening the same DB twice must not error on the second
        // heads_schedule migration attempt, and a freshly-registered row's
        // schedule must round-trip through the DB unchanged.
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let _ = Registry::open_at(&db).unwrap();
        // Second open re-runs migrate_add_heads_schedule_column — must succeed.
        let reg = Registry::open_at(&db).unwrap();
        register_model_with_range(&reg, "m", 0, 9);
        let m = reg.get_model("m").unwrap().unwrap();
        assert_eq!(m.heads_schedule, "4,4");
    }

    #[test]
    fn test_aligned_windows_column_migration_is_idempotent() {
        // Opening the same DB twice must not error on the second
        // aligned_windows migration attempt. A row registered with an
        // explicit Some(bool) round-trips; a row registered via the plain
        // helper (which leaves aligned_windows unset) comes back None
        // (unknown) -- mirroring a pre-existing row from before this column
        // existed.
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let _ = Registry::open_at(&db).unwrap();
        // Second open re-runs migrate_add_aligned_windows_column — must succeed.
        let reg = Registry::open_at(&db).unwrap();

        register_model_with_range(&reg, "unknown-model", 0, 9);
        let m = reg.get_model("unknown-model").unwrap().unwrap();
        assert_eq!(m.aligned_windows, None);

        let now = now_unix();
        let mut aligned_rec = ModelRecord {
            id: "aligned-model".to_string(),
            path: "aligned-model.bin".to_string(),
            model_type: "gpt".to_string(),
            tokenizer: "char".to_string(),
            vocab_size: 14,
            block_size: 16,
            hidden_size: 16,
            num_heads: 4,
            num_blocks: 2,
            heads_schedule: "4,4".to_string(),
            aligned_windows: Some(true),
            dataset: "data/arithmetic.txt".to_string(),
            dataset_name: "arithmetic".to_string(),
            eval_min: 0,
            eval_max: 9,
            eval_samples: 200,
            note: "n".to_string(),
            params_estimate: 7300,
            base_model_id: None,
            created_at: now,
            updated_at: now,
        };
        reg.register_model(&aligned_rec).unwrap();
        let m = reg.get_model("aligned-model").unwrap().unwrap();
        assert_eq!(m.aligned_windows, Some(true));

        aligned_rec.aligned_windows = Some(false);
        reg.register_model(&aligned_rec).unwrap();
        let m = reg.get_model("aligned-model").unwrap().unwrap();
        assert_eq!(m.aligned_windows, Some(false));
    }

    #[test]
    fn test_prompt_range_columns_migration_is_idempotent() {
        // Opening the same DB twice must not error on the second prompt-range
        // migration attempt, and a freshly-upserted row's prompt fields round-
        // trip through the DB unchanged; an SFT row (which never sets these)
        // stays NULL.
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let _ = Registry::open_at(&db).unwrap();
        // Second open re-runs migrate_add_prompt_range_columns — must succeed.
        let reg = Registry::open_at(&db).unwrap();

        reg.upsert_training("m", "sft", 10, false, 1.0, "[1.0]", "null", None, None, None, None, None)
            .unwrap();
        let t = reg.latest_training("m").unwrap().unwrap();
        assert_eq!(t.prompt_min, None);
        assert_eq!(t.prompt_max, None);
        assert_eq!(t.prompt_ops, None);

        reg.upsert_training(
            "m2",
            "grpo",
            3,
            false,
            0.1,
            "null",
            "{\"rounds\":3}",
            None,
            None,
            Some(10),
            Some(500),
            Some("+"),
        )
        .unwrap();
        let t2 = reg.latest_training("m2").unwrap().unwrap();
        assert_eq!(t2.prompt_min, Some(10));
        assert_eq!(t2.prompt_max, Some(500));
        assert_eq!(t2.prompt_ops.as_deref(), Some("+"));
    }

    #[test]
    fn test_parse_heads_schedule_column_uses_stored_schedule() {
        // A populated column wins outright, regardless of num_heads/num_blocks.
        assert_eq!(
            parse_heads_schedule_column("1,1,4,4", 1, 4),
            vec![1, 1, 4, 4]
        );
        assert_eq!(parse_heads_schedule_column("4", 4, 1), vec![4]);
    }

    #[test]
    fn test_parse_heads_schedule_column_falls_back_when_empty() {
        // Empty column (pre-migration row) -> uniform fallback from
        // num_heads/num_blocks.
        assert_eq!(parse_heads_schedule_column("", 4, 2), vec![4, 4]);
        assert_eq!(parse_heads_schedule_column("   ", 2, 3), vec![2, 2, 2]);
    }

    #[test]
    fn test_parse_heads_schedule_column_falls_back_when_malformed() {
        // Malformed/garbage column -> uniform fallback rather than a panic
        // or an empty vec (a corrupted/hand-edited DB shouldn't crash the load).
        assert_eq!(parse_heads_schedule_column("oops", 4, 2), vec![4, 4]);
        assert_eq!(parse_heads_schedule_column("1,oops,4", 4, 2), vec![4, 4]);
    }

    #[test]
    fn test_register_and_read_back_base_model_id() {
        // Registering a variant with base_model_id=Some(base) round-trips
        // through the DB.
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();

        // Register the base first.
        register_model_with_range(&reg, "base", 0, 9);

        // Register a variant with base_model_id pointing at the base.
        let now = now_unix();
        let variant = ModelRecord {
            id: "base-rft".to_string(),
            path: "base-rft.bin".to_string(),
            model_type: "gpt".to_string(),
            tokenizer: "char".to_string(),
            vocab_size: 14,
            block_size: 16,
            hidden_size: 16,
            num_heads: 4,
            num_blocks: 2,
            heads_schedule: "4,4".to_string(),
            aligned_windows: None,
            dataset: "data/arithmetic.txt".to_string(),
            dataset_name: "arithmetic".to_string(),
            eval_min: 0,
            eval_max: 9,
            eval_samples: 200,
            note: "variant".to_string(),
            params_estimate: 7300,
            base_model_id: Some("base".to_string()),
            created_at: now,
            updated_at: now,
        };
        reg.register_model(&variant).unwrap();

        let got = reg.get_model("base-rft").unwrap().unwrap();
        assert_eq!(got.base_model_id, Some("base".to_string()));

        // The base is unaffected.
        let base = reg.get_model("base").unwrap().unwrap();
        assert_eq!(base.base_model_id, None);
    }

    #[test]
    fn test_backfill_links_legacy_variant_rows_to_their_base() {
        // Simulate the legacy DB shape: a base + a variant registered as a
        // top-level row whose id ends with `-rft`. The backfill migration
        // should link the variant's base_model_id to the base.
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();

        register_model_with_range(&reg, "gpt-arithmetic-add", 0, 9);
        register_model_with_range(&reg, "gpt-arithmetic-add-rft", 0, 9);

        // Before re-opening, the variant has base_model_id NULL (the
        // register_model_with_range helper sets it to None).
        let v = reg.get_model("gpt-arithmetic-add-rft").unwrap().unwrap();
        assert_eq!(v.base_model_id, None);

        // Re-open: the backfill migration runs again and links the variant
        // (its id ends with `-rft` and the stripped stem `gpt-arithmetic-add`
        // is a registered model id).
        drop(reg);
        let reg = Registry::open_at(&db).unwrap();
        let v = reg.get_model("gpt-arithmetic-add-rft").unwrap().unwrap();
        assert_eq!(
            v.base_model_id,
            Some("gpt-arithmetic-add".to_string()),
            "backfill must link the variant to its base by id-suffix heuristic"
        );

        // Idempotent: a third open doesn't change anything.
        drop(reg);
        let reg = Registry::open_at(&db).unwrap();
        let v = reg.get_model("gpt-arithmetic-add-rft").unwrap().unwrap();
        assert_eq!(v.base_model_id, Some("gpt-arithmetic-add".to_string()));
    }

    #[test]
    fn test_backfill_skips_when_base_id_not_registered() {
        // A variant whose stripped stem isn't a registered model id stays
        // top-level (base_model_id NULL) — the heuristic only links when
        // the base actually exists.
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();
        // Register only the variant; no base row.
        register_model_with_range(&reg, "lonely-rft", 0, 9);
        drop(reg);
        let reg = Registry::open_at(&db).unwrap();
        let v = reg.get_model("lonely-rft").unwrap().unwrap();
        assert_eq!(
            v.base_model_id, None,
            "backfill must skip variants whose stripped stem isn't registered"
        );
    }

    #[test]
    fn test_checkpoint_grids_round_trip_ordered_by_epoch() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();
        register_model_with_range(&reg, "cg-model", 0, 9);

        // Empty history before any snapshot.
        assert!(reg.list_checkpoint_grids("cg-model").unwrap().is_empty());

        // Insert out of epoch order to confirm the query re-sorts by epoch.
        reg.record_checkpoint_grid("cg-model", 100, 1.5, 0, 9, "{\"a\":1}", 10, 100)
            .unwrap();
        reg.record_checkpoint_grid("cg-model", 1, 5.0, 0, 9, "{\"a\":0}", 2, 100)
            .unwrap();
        reg.record_checkpoint_grid("cg-model", 50, 2.0, 0, 9, "{\"a\":2}", 6, 100)
            .unwrap();

        let rows = reg.list_checkpoint_grids("cg-model").unwrap();
        assert_eq!(rows.len(), 3);
        let epochs: Vec<i64> = rows.iter().map(|r| r.epoch).collect();
        assert_eq!(epochs, vec![1, 50, 100], "rows must be ordered by epoch ASC");
        assert_eq!(rows[0].loss, 5.0);
        assert_eq!(rows[0].report_json, "{\"a\":0}");
        assert_eq!(rows[2].correct, 10);
        assert_eq!(rows[2].total, 100);

        // A different model's history is independent.
        register_model_with_range(&reg, "cg-model-2", 0, 9);
        assert!(reg.list_checkpoint_grids("cg-model-2").unwrap().is_empty());
    }

    #[test]
    fn test_checkpoint_grids_thin_when_over_cap() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();
        register_model_with_range(&reg, "cg-cap-model", 0, 9);

        // Insert well past MAX_CHECKPOINT_GRIDS_PER_MODEL (300) to force
        // thinning to trigger at least once.
        let total_inserted = 320;
        for epoch in 1..=total_inserted {
            reg.record_checkpoint_grid(
                "cg-cap-model",
                epoch,
                1000.0 / epoch as f64,
                0,
                9,
                "{}",
                0,
                100,
            )
            .unwrap();
        }

        let rows = reg.list_checkpoint_grids("cg-cap-model").unwrap();
        assert!(
            rows.len() <= 300,
            "thinning must keep the row count at or under the cap, got {}",
            rows.len()
        );
        assert!(
            rows.len() > 150,
            "uniform thinning must not over-delete (roughly halves, not more), got {}",
            rows.len()
        );
        // Uniform thinning must always keep the first and last epoch, so the
        // full training range stays representable for a UI slider.
        assert_eq!(rows.first().unwrap().epoch, 1, "earliest epoch must survive thinning");
        assert_eq!(
            rows.last().unwrap().epoch,
            total_inserted as i64,
            "latest epoch must survive thinning"
        );
        // Still ordered by epoch.
        let mut prev = 0;
        for r in &rows {
            assert!(r.epoch > prev, "rows must remain strictly increasing by epoch");
            prev = r.epoch;
        }
    }
}
