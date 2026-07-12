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
    pub num_heads: usize,
    pub num_blocks: usize,
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
            CREATE UNIQUE INDEX IF NOT EXISTS idx_trainings_model_kind ON trainings(model_id, kind);",
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
                    hidden_size, num_heads, num_blocks, dataset, dataset_name,
                    eval_min, eval_max, eval_samples, note, params_estimate,
                    base_model_id, created_at, updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                          ?13, ?14, ?15, ?16, ?17, ?18, ?19)
                ON CONFLICT(id) DO UPDATE SET
                    path = excluded.path,
                    model_type = excluded.model_type,
                    tokenizer = excluded.tokenizer,
                    vocab_size = excluded.vocab_size,
                    block_size = excluded.block_size,
                    hidden_size = excluded.hidden_size,
                    num_heads = excluded.num_heads,
                    num_blocks = excluded.num_blocks,
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
                        hidden_size, num_heads, num_blocks, dataset, dataset_name,
                        eval_min, eval_max, eval_samples, note, params_estimate,
                        base_model_id, created_at, updated_at
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
                        hidden_size, num_heads, num_blocks, dataset, dataset_name,
                        eval_min, eval_max, eval_samples, note, params_estimate,
                        base_model_id, created_at, updated_at
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
    pub fn upsert_training(
        &self,
        model_id: &str,
        kind: &str,
        epochs_run: usize,
        early_stopped: bool,
        final_loss: f32,
        loss_trajectory_json: &str,
        rft_summary_json: &str,
    ) -> SmolResult<()> {
        self.conn
            .execute(
                "INSERT INTO trainings (model_id, kind, epochs_run, early_stopped,
                                        final_loss, loss_trajectory, rft_summary,
                                        trained_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(model_id, kind) DO UPDATE SET
                    epochs_run = excluded.epochs_run,
                    early_stopped = excluded.early_stopped,
                    final_loss = excluded.final_loss,
                    loss_trajectory = excluded.loss_trajectory,
                    rft_summary = excluded.rft_summary,
                    trained_at = excluded.trained_at",
                params![
                    model_id,
                    kind,
                    epochs_run as i64,
                    if early_stopped { 1 } else { 0 },
                    final_loss as f64,
                    loss_trajectory_json,
                    rft_summary_json,
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
                        final_loss, loss_trajectory, rft_summary, trained_at
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
        let note = generate_note(
            model_type_str,
            params_estimate,
            meta.block_size as i64,
            meta.hidden_size as i64,
            meta.num_heads as i64,
            meta.num_blocks as i64,
            tokenizer_str,
            outcome.epochs_run,
            &dataset_filename,
            meta.seed,
            outcome.early_stopped,
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
            num_heads: meta.num_heads as i64,
            num_blocks: meta.num_blocks as i64,
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
/// `get_model` to avoid repeating the 19-column `get` list. `base_model_id`
/// is nullable — rows written before the migration (or base models) come back
/// as `None`.
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
        dataset: row.get(9)?,
        dataset_name: row.get(10)?,
        eval_min: row.get(11)?,
        eval_max: row.get(12)?,
        eval_samples: row.get(13)?,
        note: row.get(14)?,
        params_estimate: row.get(15)?,
        base_model_id: row.get(16)?,
        created_at: row.get(17)?,
        updated_at: row.get(18)?,
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
        trained_at: row.get(8)?,
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

/// Auto-generate the model `note` from training metadata + outcome. Format:
///
/// ```text
/// {model_type} {params}K params (block={b} hidden={h} heads={nh} blocks={nb}),
/// {tokenizer} tokenizer, trained {epochs_run} epochs on {dataset_filename},
/// seed {seed}{early_stop_clause}
/// ```
///
/// where `early_stop_clause` = `, early-stopped at epoch {epochs_run}` if early
/// stopping fired, else empty. `{params}` rounds to whole K for >=10K models
/// and one decimal place for smaller ones (so 77582 → "78K", 7300 → "7.3K").
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
    format!(
        "{model_type} {params_k} params (block={block_size} hidden={hidden_size} heads={num_heads} blocks={num_blocks}), {tokenizer} tokenizer, trained {epochs_run} epochs on {dataset}, seed {seed}{early_clause}",
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
        );
        assert!(note.contains("gpt 78K params"));
        assert!(note.contains("block=32 hidden=32 heads=8 blocks=6"));
        assert!(note.contains("char tokenizer"));
        assert!(note.contains("trained 2000 epochs on arithmetic-1digit.txt"));
        assert!(note.contains("seed 42"));
        assert!(!note.contains("early-stopped"));
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
        );
        assert!(note.contains("gpt 7.3K params"));
        assert!(note.contains("seed random"));
        assert!(note.contains("early-stopped at epoch 430"));
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
    }

    #[test]
    fn test_latest_training_picks_most_recent() {
        let dir = temp_dir::TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let reg = Registry::open_at(&db).unwrap();

        // Two trainings for the same model. The second insert happens later in
        // wall-clock time (trained_at is the order key), so latest_training
        // must return it.
        reg.upsert_training("m", "sft", 100, false, 5.0, "[5.0]", "null")
            .unwrap();
        // Tiny sleep so `now_unix()` advances by at least 1 second, making the
        // second row strictly newer. (If both rows share the same second, the
        // tie-breaker on `id DESC` still picks the second row.)
        std::thread::sleep(std::time::Duration::from_millis(1100));
        reg.upsert_training("m", "sft", 200, true, 1.2, "[1.2]", "null")
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
            num_heads: 4,
            num_blocks: 2,
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
            num_heads: 4,
            num_blocks: 2,
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
            num_heads: 4,
            num_blocks: 2,
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
            num_heads: 4,
            num_blocks: 2,
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
            num_heads: 4,
            num_blocks: 2,
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

        reg.upsert_training("m", "sft", 10, false, 5.0, "[5.0, 4.8]", "null")
            .unwrap();
        reg.upsert_training("m", "sft", 20, false, 1.2, "[5.0, 4.8, 1.2]", "null")
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

        reg.upsert_training("m", "sft", 2000, false, 1.2, "[1.2]", "null")
            .unwrap();
        reg.upsert_training("m", "rft", 3, false, 0.5, "null", "{\"rounds\":3}")
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
        reg.upsert_training("m", "sft", 4000, true, 0.9, "[0.9]", "null")
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
}
