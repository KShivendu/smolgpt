//! Local web UI for browsing trained models, their datasets, and running
//! greedy-decoding evals from the browser.
//!
//! Activated by `--serve`. Reads the model registry from `smolgpt.db` (the
//! SQLite DB maintained by `registry.rs`); on the first start, if the DB is
//! empty and `models.toml` exists, it seeds the DB from the legacy TOML file.
//! After that the DB is the source of truth and `models.toml` is just a seed.
//!
//! Routes:
//!   GET  /                       → embedded HTML page
//!   GET  /api/models             → JSON array of model cards (with latest eval)
//!   GET  /api/models/{id}/eval   → runs eval, returns EvalReport JSON
//!   GET  /api/models/{id}/eval-grid → runs exhaustive eval, returns EvalGridReport JSON
//!   GET  /api/models/{id}/checkpoint-grids → history of exhaustive-grid snapshots
//!                                  taken during training (for the Grid tab's
//!                                  training-progress slider), oldest-epoch-first
//!   GET  /api/models/{id}/jacobian-lens?force=<bool> → cache-first Jacobian-lens
//!                                  interpretability result (Gpt-only; see
//!                                  `crate::jacobian_lens`), recomputed via a
//!                                  Python subprocess when forced or uncached
//!   GET  /api/models/{id}/jacobian-lens/plot/{filename} → one PNG plot from a
//!                                  cached Jacobian-lens run
//!   POST /api/repl/tokenize      → encode text with a model's tokenizer
//!   POST /api/repl/generate      → run greedy/sampled decoding from a prompt
//!
//! Eval is CPU-heavy so it runs in `spawn_blocking`; a `Mutex<HashSet>` of
//! in-flight ids returns HTTP 409 for a concurrent second request. Successful
//! evals are recorded into the `evals` table so a page reload shows the last
//! result without re-running. REPL endpoints likewise run their heavy work
//! (corpus read, tokenizer build, model load, inference) in `spawn_blocking`
//! and serialize against each other via a separate `Mutex<()>` (`repl_lock`)
//! so two concurrent REPL requests don't both load models and thrash.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::extract::{Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use candle_core::Device;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::args::{EvalMode, ModelType};
use crate::dataset;
use crate::error::{SmolError, SmolResult};
use crate::eval::{run_eval, run_eval_grid, sample_example_prompts, EvalGridReport, EvalReport};
use crate::model::LanguageModel;
use crate::jacobian_lens::run_jacobian_lens_for_model;
use crate::registry::{
    EvalGridSummary, EvalRecord, JacobianLensSummary, ModelRecord, Registry, TrainingRecord,
};
use crate::tokenizer::{BpeTokenizer, SimpleTokenizer, Tokenizer};

/// Dataset metadata shown on a model card and via the API.
#[derive(Debug, Clone, Serialize)]
struct DatasetInfo {
    path: String,
    name: String,
    line_count: usize,
    byte_size: usize,
    head: Vec<String>,
}

/// Wire-format view of an RFT run, parsed from the `rft_summary` JSON column
/// of the `trainings` table. Mirrors `crate::rft::RftSummary` but is declared
/// separately so the JSON shape served to the UI is decoupled from the
/// internal struct (and so `serve.rs` doesn't pull `rft.rs` into the binary's
/// type graph just for serialization). `per_round_sft_final_losses` is a
/// `Vec<Option<f32>>` because RFT rounds with zero winners skip SFT (the
/// `None` distinguishes "no SFT ran" from "SFT ran and ended at loss X").
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RftSummaryView {
    rounds: usize,
    winner_rates: Vec<f64>,
    eval_correct_pct: Vec<f64>,
    per_round_sft_final_losses: Vec<Option<f32>>,
}

/// Wire-format view of a GRPO-lite run, parsed from the same `rft_summary`
/// JSON column (a GRPO row is distinguished by `kind == "grpo"`). Mirrors
/// `crate::grpo::GrpoSummary`. `correct_rates` is the per-round fraction of
/// ALL G*P sampled completions that were correct (analogous to RFT's
/// winner_rate but over every sample, not just first-correct); `eval_correct_pct`
/// is the per-round greedy-decoding correctness; `per_round_losses` is the
/// per-round mean policy-gradient loss. `mode` is `"lite"` or `"full"`
/// (defaults to `"lite"` for pre-mode DB rows via `#[serde(default)]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GrpoSummaryView {
    rounds: usize,
    group_size: usize,
    #[serde(default = "default_grpo_mode_view")]
    mode: String,
    correct_rates: Vec<f64>,
    eval_correct_pct: Vec<f64>,
    per_round_losses: Vec<Option<f32>>,
}

/// Serde default for `GrpoSummaryView::mode` — `"lite"`, matching
/// `crate::grpo::default_grpo_mode`. Pre-mode DB rows (serialized before
/// the `mode` field existed) deserialize to `"lite"` so the UI renders them
/// with the lite badge rather than an empty string.
fn default_grpo_mode_view() -> String {
    "lite".to_string()
}

/// Wire-format view of a training run, parsed from a `TrainingRecord`. The
/// `loss_trajectory` Vec is parsed from the `loss_trajectory_json` string for
/// SFT runs (empty for RFT); `rft_summary` is parsed from
/// `rft_summary_json` for RFT runs (`None` for SFT). Parse failures are logged
/// and surfaced as empty/None rather than 500ing the whole `/api/models`
/// request — a corrupt JSON blob shouldn't hide the rest of the card.
#[derive(Debug, Clone, Serialize)]
struct TrainingView {
    kind: String,
    epochs_run: i64,
    early_stopped: bool,
    final_loss: f64,
    loss_trajectory: Vec<f32>,
    rft_summary: Option<RftSummaryView>,
    grpo_summary: Option<GrpoSummaryView>,
    /// Exact greedy-decoding accuracy over the literal training corpus
    /// (distinct from `cached_eval`'s random-sampled-range accuracy). `None`
    /// for RFT/GRPO rows and for SFT rows written before this was tracked.
    #[serde(skip_serializing_if = "Option::is_none")]
    train_correct: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    train_total: Option<i64>,
    #[serde(serialize_with = "serialize_iso_i64")]
    trained_at: i64,
}

/// `serde` serializer for `i64` Unix seconds → ISO 8601 UTC string. Same
/// convention as `registry::EvalRecord` / `ModelRecord` timestamps.
fn serialize_iso_i64<S: serde::Serializer>(val: &i64, ser: S) -> Result<S::Ok, S::Error> {
    ser.serialize_str(&crate::registry::format_iso(*val))
}

impl TrainingView {
    /// Build a `TrainingView` from a `TrainingRecord`, parsing the two JSON
    /// blob columns into typed fields. Parse failures are logged and degraded
    /// gracefully (empty Vec / None) so a single corrupt row doesn't break
    /// `/api/models`.
    fn from_record(rec: &TrainingRecord) -> TrainingView {
        let loss_trajectory = if rec.loss_trajectory_json == "null"
            || rec.loss_trajectory_json.is_empty()
        {
            Vec::new()
        } else {
            serde_json::from_str(&rec.loss_trajectory_json).unwrap_or_else(|e| {
                eprintln!(
                    "[serve] WARNING: failed to parse loss_trajectory JSON for \
                     '{}' (kind={}): {e}; showing empty trajectory",
                    rec.model_id, rec.kind
                );
                Vec::new()
            })
        };
        // The `rft_summary_json` column holds the per-round summary JSON for
        // both RFT and GRPO rows (it's a generic JSON blob); `kind` decides
        // which view struct to parse it into. SFT rows store "null".
        let rft_summary = if rec.kind == "rft"
            && rec.rft_summary_json != "null"
            && !rec.rft_summary_json.is_empty()
        {
            match serde_json::from_str::<RftSummaryView>(&rec.rft_summary_json) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!(
                        "[serve] WARNING: failed to parse rft_summary JSON for \
                         '{}' (kind={}): {e}; hiding RFT table",
                        rec.model_id, rec.kind
                    );
                    None
                }
            }
        } else {
            None
        };
        let grpo_summary = if rec.kind == "grpo"
            && rec.rft_summary_json != "null"
            && !rec.rft_summary_json.is_empty()
        {
            match serde_json::from_str::<GrpoSummaryView>(&rec.rft_summary_json) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!(
                        "[serve] WARNING: failed to parse grpo_summary JSON for \
                         '{}' (kind={}): {e}; hiding GRPO table",
                        rec.model_id, rec.kind
                    );
                    None
                }
            }
        } else {
            None
        };
        TrainingView {
            kind: rec.kind.clone(),
            epochs_run: rec.epochs_run,
            early_stopped: rec.early_stopped,
            final_loss: rec.final_loss,
            loss_trajectory,
            rft_summary,
            grpo_summary,
            train_correct: rec.train_correct,
            train_total: rec.train_total,
            trained_at: rec.trained_at,
        }
    }
}

/// Wire-format view of a base/SFT model's example training windows (the
/// Samples tab), built from the model's registered corpus/block_size/
/// `aligned_windows` setting via `dataset::sample_example_windows` -- the
/// SAME sampling logic `--train` uses, so what's shown is genuinely faithful
/// to what training does. `aligned_windows` mirrors `ModelRecord`'s field:
/// `None` means the model predates the column (unknown historical setting),
/// serialized so the UI can render an honest "unknown" state instead of
/// guessing.
#[derive(Debug, Clone, Serialize)]
struct SftSamplesView {
    aligned_windows: Option<bool>,
    windows: Vec<String>,
}

/// Wire-format view of an RFT/GRPO variant's example RL-stage prompts (the
/// Samples tab), built from the variant's registered `prompt_min`/
/// `prompt_max`/`prompt_ops` (see `TrainingRecord`'s doc) via
/// `eval::sample_example_prompts`. `kind` is `"rft"` or `"grpo"`.
#[derive(Debug, Clone, Serialize)]
struct RlSamplesView {
    kind: String,
    prompt_min: i64,
    prompt_max: i64,
    prompt_ops: String,
    prompts: Vec<String>,
}

/// JSON view of a model, combining the registry record with computed dataset
/// info, a load status, an approximate param count, an optional latest
/// eval (from the DB `evals` table), and an optional latest training record
/// (from the DB `trainings` table — SFT loss trajectory or RFT per-round
/// summary). Fields are `Option` so unregistered `.bin` files (which have no
/// metadata) can be represented with nulls.
///
/// `base_model_id`: `None` for a base model, `Some(base_id)` for an RL variant
/// (RFT/GRPO) linked to its base. The UI groups variants under their base
/// card's `variants` array; top-level `/api/models` entries are base models
/// only (variants appear nested, not as separate top-level cards).
///
/// `variants`: only populated for base models; an array of full `ModelView`s
/// for each RL variant linked to this base (by `base_model_id == this base's
/// id`). Empty for variants themselves and for base models with no RL runs.
#[derive(Debug, Serialize)]
struct ModelView {
    id: String,
    path: String,
    status: String,
    model_type: Option<String>,
    tokenizer: Option<String>,
    vocab_size: Option<usize>,
    block_size: Option<usize>,
    hidden_size: Option<usize>,
    num_heads: Option<usize>,
    num_blocks: Option<usize>,
    dataset: Option<String>,
    dataset_name: Option<String>,
    dataset_info: Option<DatasetInfo>,
    eval_min: Option<i64>,
    eval_max: Option<i64>,
    eval_samples: Option<usize>,
    note: Option<String>,
    params_estimate: Option<usize>,
    /// Size in bytes of the model's `.bin` file on disk. `None` if the file
    /// can't be stat'd (e.g. `status` is already `"missing"`).
    file_size_bytes: Option<u64>,
    /// `None` for a base model; `Some(base_id)` for an RL variant. See the
    /// struct doc.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    base_model_id: Option<String>,
    cached_eval: Option<EvalRecord>,
    /// Metadata about the most recent cached exhaustive eval-grid run (Grid
    /// tab), if the model's current operand range still matches the range
    /// the cache was computed for (`Registry::latest_eval_grid`'s smart
    /// staleness filter — analogous to `cached_eval` above). Deliberately
    /// does NOT carry the full grid (every cell) — the browser fetches that
    /// separately, but only when this field says a cache is worth fetching,
    /// so `/api/models` doesn't balloon with every model's full cell data.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    cached_grid: Option<EvalGridSummary>,
    /// Metadata about the most recent cached Jacobian-lens interpretability
    /// run (Jacobian tab), if one exists. `None` both for models that have
    /// never been analyzed and for non-`Gpt` model types (Bigram/Ngram),
    /// which the UI renders as "not applicable" rather than offering the
    /// tab's "run" button. Deliberately omits the full `results_json`/plot
    /// list — those are fetched separately via
    /// `GET /api/models/{id}/jacobian-lens`, mirroring `cached_grid`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    jacobian_lens: Option<JacobianLensSummary>,
    /// Most recent training run (SFT loss trajectory or RFT per-round
    /// summary). `None` when the model has no row in the `trainings` table
    /// (e.g. a model trained before this field existed and not yet
    /// backfilled).
    training: Option<TrainingView>,
    /// RL variants linked to this base model. Empty for variants and for
    /// base models with no RL runs. Each variant is a full `ModelView` (with
    /// its own `cached_eval` + `training`), so the UI can swap to a variant
    /// without an extra round-trip.
    #[serde(default)]
    variants: Vec<ModelView>,
    /// Example SFT training windows (Samples tab). `Some` only for base
    /// models (`base_model_id.is_none()`) whose dataset corpus is readable
    /// and long enough for `block_size`; `None` for variants (they get
    /// `rl_samples` instead) and for base models whose corpus can't be
    /// reconstructed.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    sft_samples: Option<SftSamplesView>,
    /// Example RL-stage prompts (Samples tab). `Some` only for RFT/GRPO
    /// variants (`base_model_id.is_some()`) that have a `trainings` row with
    /// recorded `prompt_min`/`prompt_max`/`prompt_ops`; `None` for base
    /// models and for variants trained before these columns existed.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    rl_samples: Option<RlSamplesView>,
}

/// Query params for the eval endpoint.
#[derive(Debug, Deserialize)]
struct EvalQuery {
    seed: Option<u64>,
}

/// Query params for the eval-grid endpoint. `force` (default `false`) selects
/// between "serve the cached grid if one is fresh" (the default — cheap,
/// no model load) and "recompute unconditionally and refresh the cache" (the
/// Grid tab's "Recompute grid" button).
#[derive(Debug, Deserialize, Default)]
struct EvalGridQuery {
    #[serde(default)]
    force: bool,
}

/// Query params for the jacobian-lens endpoint. `force` (default `false`)
/// selects between "serve the cached result if one exists" and "recompute
/// unconditionally and refresh the cache", mirroring `EvalGridQuery`.
#[derive(Debug, Deserialize, Default)]
struct JacobianLensQuery {
    #[serde(default)]
    force: bool,
}

/// One token in a REPL tokenization view: the numeric id and the substring
/// it decodes to. The UI renders these as chips.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenChip {
    id: u32,
    str: String,
}

/// `POST /api/repl/tokenize` request body.
#[derive(Debug, Deserialize)]
struct TokenizeRequest {
    model_id: String,
    text: String,
}

/// `POST /api/repl/tokenize` response body.
#[derive(Debug, Serialize)]
struct TokenizeResponse {
    tokens: Vec<TokenChip>,
    vocab_size: usize,
    tokenizer_type: String,
}

/// `POST /api/repl/generate` request body. `greedy == true` OR `temperature <= 0`
/// selects greedy decoding; otherwise temperature sampling with a seeded RNG.
#[derive(Debug, Deserialize)]
struct GenerateRequest {
    model_id: String,
    prompt: String,
    max_new_tokens: usize,
    temperature: f32,
    greedy: bool,
}

/// `POST /api/repl/generate` response body. `generated_text` is the decoded
/// completion; `generated_tokens` is the per-token decode of each new id.
#[derive(Debug, Serialize)]
struct GenerateResponse {
    prompt_tokens: Vec<TokenChip>,
    generated_text: String,
    generated_tokens: Vec<TokenChip>,
}

/// Shared server state, cloned (via Arc) into every handler.
#[derive(Clone)]
struct AppState {
    project_root: Arc<PathBuf>,
    in_flight: Arc<Mutex<HashSet<String>>>,
    registry: Arc<Mutex<Registry>>,
    /// Serializes CPU-heavy REPL work (tokenize + generate) so two concurrent
    /// REPL requests don't both load models simultaneously and thrash. Held
    /// only inside `spawn_blocking` closures, never across awaits.
    repl_lock: Arc<Mutex<()>>,
    /// Eval-range behavior mode. `Smart` filters `latest_eval` by the model's
    /// current range (hides stale rows from an old range); `Legacy` returns
    /// the newest row regardless of range. `Copy` so this threads through
    /// `AppState` without an `Arc`.
    eval_mode: EvalMode,
}

/// RAII guard for an entry in the `in_flight` set. `try_acquire` inserts
/// `key` (returning `None` if it's already present, so the caller can 409
/// instead), and the `Drop` impl removes it.
///
/// This matters specifically because the CPU-heavy handlers hold the guard
/// across a `spawn_blocking(...).await`: if the client disconnects mid-await,
/// axum drops the handler's future right there, and any code written *after*
/// that `.await` (e.g. a manual `guard.remove(&key)`) never runs — permanently
/// leaking the entry and wedging that model's endpoint until the server
/// restarts. Tying removal to `Drop` instead means cleanup runs on every exit
/// path (normal return, panic, or the future simply being dropped), not just
/// the one where the `.await` resolves normally.
struct InFlightGuard {
    set: Arc<Mutex<HashSet<String>>>,
    key: String,
}

impl InFlightGuard {
    fn try_acquire(set: Arc<Mutex<HashSet<String>>>, key: String) -> Option<Self> {
        let mut guard = set.lock().unwrap_or_else(|e| e.into_inner());
        if guard.contains(&key) {
            return None;
        }
        guard.insert(key.clone());
        drop(guard);
        Some(InFlightGuard { set, key })
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        let mut guard = self.set.lock().unwrap_or_else(|e| e.into_inner());
        guard.remove(&self.key);
    }
}

/// Entry point — called from `train::do_training` when `--serve` is set.
/// Opens the SQLite registry, seeds it from `models.toml` on first start (if
/// the DB is empty), builds the router, and blocks on the tokio runtime.
/// The model list is read live from the DB on every `/api/models` request, so
/// models trained while the server runs appear without a restart.
pub fn run_serve(host: &str, port: u16, eval_mode: EvalMode) -> SmolResult<()> {
    let project_root = std::env::current_dir()
        .map_err(|e| SmolError::custom_error(&format!("cwd: {e}")))?;

    let registry = Registry::open()?;

    // First-start seeding: if the DB has no models and models.toml exists,
    // import the legacy TOML registry so already-trained models stay visible.
    if registry.is_empty()? {
        let toml_path = project_root.join("models.toml");
        if toml_path.exists() {
            match registry.import_from_toml(&toml_path) {
                Ok(n) => println!(
                    "Seeded {n} models from {} into smolgpt.db",
                    toml_path.display()
                ),
                Err(e) => eprintln!(
                    "[serve] WARNING: failed to seed from {}: {e}",
                    toml_path.display()
                ),
            }
        }
    }

    let state = AppState {
        project_root: Arc::new(project_root),
        in_flight: Arc::new(Mutex::new(HashSet::new())),
        registry: Arc::new(Mutex::new(registry)),
        repl_lock: Arc::new(Mutex::new(())),
        eval_mode,
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| SmolError::custom_error(&format!("tokio runtime: {e}")))?;

    rt.block_on(serve_inner(host, port, state))
}

async fn serve_inner(host: &str, port: u16, state: AppState) -> SmolResult<()> {
    let app = Router::new()
        .route("/", get(index))
        .route("/api/models", get(list_models))
        .route("/api/models/{id}/eval", get(eval_model))
        .route("/api/models/{id}/eval-grid", get(eval_grid_model))
        .route("/api/models/{id}/checkpoint-grids", get(checkpoint_grids_model))
        .route("/api/models/{id}/jacobian-lens", get(jacobian_lens_model))
        .route(
            "/api/models/{id}/jacobian-lens/plot/{filename}",
            get(jacobian_lens_plot),
        )
        .route("/api/repl/tokenize", post(repl_tokenize))
        .route("/api/repl/generate", post(repl_generate))
        .with_state(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| SmolError::custom_error(&format!("bind {addr} failed: {e}")))?;
    println!("Serving smolgpt web UI at http://{host}:{port}");
    axum::serve(listener, app)
        .await
        .map_err(|e| SmolError::custom_error(&format!("server error: {e}")))?;
    Ok(())
}

// --- Route handlers ---

async fn index() -> Html<&'static str> {
    Html(HTML)
}

async fn list_models(State(state): State<AppState>) -> Response {
    // Read the model list live from the DB on every request so models trained
    // while the server is running appear without a restart.
    let project_root = state.project_root.clone();
    let records = {
        let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
        match reg.list_models() {
            Ok(r) => r,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                )
                    .into_response();
            }
        }
    };

    // Fetch all latest_evals in one DB lock scope so a page reload shows the
    // last result without re-running. In smart mode this is the range-matched
    // lookup (hides stale rows from an old range); in legacy mode it's the
    // plain newest-by-run_at lookup.
    let eval_mode = state.eval_mode;
    let latest_evals: std::collections::HashMap<String, EvalRecord> = {
        let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
        records
            .iter()
            .filter_map(|r| {
                let res = match eval_mode {
                    EvalMode::Smart => reg.latest_eval(&r.id),
                    EvalMode::Legacy => reg.latest_eval_legacy(&r.id),
                };
                match res {
                    Ok(Some(e)) => Some((r.id.clone(), e)),
                    Ok(None) => None,
                    Err(e) => {
                        eprintln!("[serve] latest_eval('{}') failed: {e}", r.id);
                        None
                    }
                }
            })
            .collect()
    };

    // Fetch all cached-grid metadata in one DB lock scope, same rationale as
    // latest_evals above. `latest_eval_grid` already applies the smart
    // range-staleness filter (like `latest_eval`), so a model whose eval
    // range changed since its grid was cached correctly comes back `None`
    // here instead of advertising a stale cache.
    let latest_grids: std::collections::HashMap<String, EvalGridSummary> = {
        let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
        records
            .iter()
            .filter_map(|r| match reg.latest_eval_grid(&r.id) {
                Ok(Some(g)) => Some((r.id.clone(), EvalGridSummary::from(&g))),
                Ok(None) => None,
                Err(e) => {
                    eprintln!("[serve] latest_eval_grid('{}') failed: {e}", r.id);
                    None
                }
            })
            .collect()
    };

    // Fetch all cached Jacobian-lens metadata in one DB lock scope, same
    // rationale as latest_grids above. No range-staleness filter here (see
    // `latest_jacobian_lens`'s doc) — any row present is for the model's
    // current weights.
    let latest_jacobian_lenses: std::collections::HashMap<String, JacobianLensSummary> = {
        let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
        records
            .iter()
            .filter_map(|r| match reg.latest_jacobian_lens(&r.id) {
                Ok(Some(j)) => Some((r.id.clone(), JacobianLensSummary::from(&j))),
                Ok(None) => None,
                Err(e) => {
                    eprintln!("[serve] latest_jacobian_lens('{}') failed: {e}", r.id);
                    None
                }
            })
            .collect()
    };

    // Fetch all latest_trainings in one DB lock scope so each card can render
    // its loss trajectory / RFT summary without a per-card DB round-trip.
    let latest_trainings: std::collections::HashMap<String, TrainingRecord> = {
        let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
        records
            .iter()
            .filter_map(|r| match reg.latest_training(&r.id) {
                Ok(Some(t)) => Some((r.id.clone(), t)),
                Ok(None) => None,
                Err(e) => {
                    eprintln!("[serve] latest_training('{}') failed: {e}", r.id);
                    None
                }
            })
            .collect()
    };

    // Build a `ModelView` from a `ModelRecord`, pulling its cached eval and
    // training from the pre-fetched maps. Shared between base cards and
    // variant cards so each variant carries its own eval/training.
    let build_view = |record: &ModelRecord| -> ModelView {
        let cached_eval = latest_evals.get(&record.id).cloned();
        let cached_grid = latest_grids.get(&record.id).cloned();
        let jacobian_lens = latest_jacobian_lenses.get(&record.id).cloned();
        let raw_training = latest_trainings.get(&record.id);
        let training = raw_training.map(TrainingView::from_record);
        // Samples tab: base models get reconstructed SFT training windows;
        // RL variants get reconstructed RL-stage prompts instead (never
        // both -- a variant's OWN samples are its RL prompts; its base's SFT
        // windows are looked up separately by the frontend via
        // `base_model_id`, since both are already present in this same
        // `/api/models` payload).
        let (sft_samples, rl_samples) = if record.base_model_id.is_none() {
            (compute_sft_samples(&project_root, record), None)
        } else {
            (None, compute_rl_samples(raw_training))
        };
        ModelView {
            id: record.id.clone(),
            path: record.path.clone(),
            status: cheap_status(record, &project_root),
            model_type: Some(record.model_type.clone()),
            tokenizer: Some(record.tokenizer.clone()),
            vocab_size: Some(record.vocab_size as usize),
            block_size: Some(record.block_size as usize),
            hidden_size: Some(record.hidden_size as usize),
            num_heads: Some(record.num_heads as usize),
            num_blocks: Some(record.num_blocks as usize),
            dataset: Some(record.dataset.clone()),
            dataset_name: Some(record.dataset_name.clone()),
            dataset_info: compute_dataset_info(&project_root, record),
            eval_min: Some(record.eval_min),
            eval_max: Some(record.eval_max),
            eval_samples: Some(record.eval_samples as usize),
            note: Some(record.note.clone()),
            params_estimate: Some(record.params_estimate as usize),
            file_size_bytes: resolve_within_root(&project_root, &record.path)
                .and_then(|p| std::fs::metadata(p).ok())
                .map(|m| m.len()),
            base_model_id: record.base_model_id.clone(),
            cached_eval,
            cached_grid,
            jacobian_lens,
            training,
            variants: Vec::new(),
            sft_samples,
            rl_samples,
        }
    };

    // Partition records into bases (base_model_id NULL) and variants
    // (base_model_id Some). Variants are grouped under their base; top-level
    // cards are bases only, so variants don't appear as loose cards.
    let mut variants_by_base: std::collections::HashMap<String, Vec<ModelRecord>> =
        std::collections::HashMap::new();
    let mut base_records: Vec<&ModelRecord> = Vec::new();
    for record in &records {
        match &record.base_model_id {
            None => base_records.push(record),
            Some(base_id) => variants_by_base
                .entry(base_id.clone())
                .or_default()
                .push(record.clone()),
        }
    }

    // Build the top-level views: one per base, with its variants nested.
    let mut views: Vec<ModelView> = Vec::with_capacity(base_records.len());
    for base in base_records {
        let mut view = build_view(base);
        if let Some(variant_records) = variants_by_base.get(&base.id) {
            for vr in variant_records {
                view.variants.push(build_view(vr));
            }
        }
        views.push(view);
    }

    Json(views).into_response()
}

async fn eval_model(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Query(query): Query<EvalQuery>,
) -> Response {
    let record = {
        let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
        match get_model_or_404(&reg, &id) {
            Ok(r) => r,
            Err(resp) => return resp,
        }
    };

    // In-flight check: only one eval per model id at a time. Held across the
    // `.await` below as an RAII guard so it's released even if the client
    // disconnects mid-request (see `InFlightGuard`'s doc comment).
    let Some(_flight_guard) = InFlightGuard::try_acquire(state.in_flight.clone(), id.clone())
    else {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": "eval already running for this model"})),
        )
            .into_response();
    };

    let project_root = state.project_root.clone();
    let seed = query.seed.unwrap_or(42);

    // CPU-heavy: load model + run greedy eval inside spawn_blocking so the
    // async runtime stays responsive for other requests.
    let result = tokio::task::spawn_blocking(move || {
        run_eval_for_model(&record, &project_root, seed)
    })
    .await;

    match result {
        Ok(Ok(report)) => {
            // Record the eval summary in the DB so a page reload shows the
            // last result without re-running. Best-effort: a write failure
            // just logs — the JSON is still returned to the caller.
            {
                let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
                if let Err(e) = reg.record_eval(&id, &report, Some(seed)) {
                    eprintln!("[serve] WARNING: failed to record eval for '{id}': {e}");
                }
            }
            Json(report).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("eval task panicked: {e}")})),
        )
            .into_response(),
    }
}

/// `GET /api/models/{id}/eval-grid?force=<bool>` — serve the exhaustive Grid
/// tab's data.
///
/// Cache-first, mirroring the `cached_eval` + "Run eval" UX: with
/// `force=false` (the default — a plain tab-open / tab-switch-back), if
/// `Registry::latest_eval_grid` has a row whose stamped range still matches
/// the model's current range, its `report_json` is parsed and returned
/// directly — a fast DB read, no model load, no `spawn_blocking`. This is
/// what makes "reopen the Grid tab" instant instead of re-running greedy
/// decoding over every cell. With `force=true` (the "Recompute grid" button),
/// or whenever there's no fresh cache, the grid is recomputed from scratch in
/// `spawn_blocking` and the result replaces the cached row via
/// `record_eval_grid` (so the next plain open is instant again).
///
/// The recompute path is guarded the same way as `/eval`: a `Mutex<HashSet>`
/// in-flight check so a duplicate concurrent *recompute* request for the same
/// model 409s instead of double-loading the model. Uses a distinct in-flight
/// key (`"{id}::grid"`) so a grid recompute and a plain sampled-eval request
/// for the same model don't spuriously 409 each other. The cache-hit fast
/// path deliberately skips the in-flight guard entirely — it never loads a
/// model, so there's nothing to serialize against.
///
/// Returns 400 (not 500) when the range exceeds `eval::MAX_GRID_AXIS` — that
/// error is an expected, actionable outcome ("range too large, use the
/// sampled Eval tab instead"), not a server fault. A too-large range is never
/// cached (the compute path errors out before producing a report), so this
/// path is unaffected by caching either way.
async fn eval_grid_model(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Query(query): Query<EvalGridQuery>,
) -> Response {
    let record = {
        let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
        match get_model_or_404(&reg, &id) {
            Ok(r) => r,
            Err(resp) => return resp,
        }
    };

    // Fast path: a fresh cached grid exists and the caller didn't ask to
    // force a recompute. `latest_eval_grid` already applies the smart
    // range-staleness filter, so a `Some` here is guaranteed to match the
    // model's *current* eval_min/eval_max.
    if !query.force {
        let cached = {
            let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
            reg.latest_eval_grid(&id)
        };
        match cached {
            Ok(Some(cached)) => match serde_json::from_str::<EvalGridReport>(&cached.report_json) {
                Ok(report) => return Json(report).into_response(),
                Err(e) => {
                    // Corrupt cache row — log and fall through to recompute
                    // rather than 500ing; a bad blob shouldn't permanently
                    // wedge the Grid tab.
                    eprintln!(
                        "[serve] WARNING: failed to parse cached eval-grid JSON for '{id}': {e}; recomputing"
                    );
                }
            },
            Ok(None) => {}
            Err(e) => {
                eprintln!("[serve] latest_eval_grid('{id}') failed: {e}; recomputing");
            }
        }
    }

    let flight_key = format!("{id}::grid");
    let Some(_flight_guard) = InFlightGuard::try_acquire(state.in_flight.clone(), flight_key)
    else {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": "grid eval already running for this model"})),
        )
            .into_response();
    };

    let project_root = state.project_root.clone();

    let result = tokio::task::spawn_blocking(move || {
        run_eval_grid_for_model(&record, &project_root)
    })
    .await;

    match result {
        Ok(Ok(report)) => {
            // Refresh the cache so the next plain (non-force) open is instant.
            // Best-effort: a write failure just logs — the JSON is still
            // returned to the caller.
            match serde_json::to_string(&report) {
                Ok(report_json) => {
                    let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
                    if let Err(e) = reg.record_eval_grid(
                        &id,
                        report.min,
                        report.max,
                        &report_json,
                        report.correct as i64,
                        report.total as i64,
                    ) {
                        eprintln!("[serve] WARNING: failed to cache eval-grid for '{id}': {e}");
                    }
                }
                Err(e) => {
                    eprintln!("[serve] WARNING: failed to serialize eval-grid for '{id}': {e}");
                }
            }
            Json(report).into_response()
        }
        Ok(Err(e @ SmolError::InvalidArgument(_))) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("grid eval task panicked: {e}")})),
        )
            .into_response(),
    }
}

/// Wire-format view of one `checkpoint_grids` snapshot, served by
/// `GET /api/models/{id}/checkpoint-grids`. `report_json` is parsed
/// server-side into a proper nested `EvalGridReport` value (the exact same
/// shape `renderGrid` already consumes for the plain Grid tab) rather than
/// shipped as a JSON-encoded string-within-JSON, so the browser can hand
/// `report` straight to the existing renderer with no client-side
/// `JSON.parse`.
#[derive(Debug, Serialize)]
struct CheckpointGridView {
    epoch: i64,
    loss: f64,
    correct: i64,
    total: i64,
    #[serde(serialize_with = "serialize_iso_i64")]
    run_at: i64,
    report: EvalGridReport,
}

/// `GET /api/models/{id}/checkpoint-grids` — the full history of
/// exhaustive-eval-grid snapshots captured during training (see
/// `Registry::list_checkpoint_grids` / the `checkpoint_grids` table), for the
/// Grid tab's training-progress slider + play/pause animation. Ordered
/// oldest-epoch-first (`epoch ASC`), matching the animation's natural
/// playback direction.
///
/// Returns a 404 if the model itself isn't registered (via
/// `get_model_or_404`, mirroring `/eval` and `/eval-grid`), but an empty JSON
/// array `[]` (200 OK) — not an error — for a model that IS registered but
/// has no snapshots. That's the common case today: this capture path is new,
/// so old training runs never wrote to `checkpoint_grids`, and the browser
/// needs to tell "no history recorded" apart from a genuine failure so it can
/// show a plain message instead of a broken slider.
///
/// Sent as a single JSON response rather than paginated/streamed: this is a
/// local, single-user dev tool, and even the densest real model on record
/// (`mask-checkpoints`, 102 snapshots over a small operand grid) serializes
/// to a payload in the "a few hundred KB to a few MB" range per the task
/// brief, which a local fetch handles well under a second. Pagination would
/// only be worth the added complexity if snapshot counts or grid sizes grew
/// much larger than what this project's models actually produce.
async fn checkpoint_grids_model(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    let records = {
        let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(resp) = get_model_or_404(&reg, &id) {
            return resp;
        }
        match reg.list_checkpoint_grids(&id) {
            Ok(r) => r,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                )
                    .into_response();
            }
        }
    };

    let mut views = Vec::with_capacity(records.len());
    for rec in &records {
        match serde_json::from_str::<EvalGridReport>(&rec.report_json) {
            Ok(report) => views.push(CheckpointGridView {
                epoch: rec.epoch,
                loss: rec.loss,
                correct: rec.correct,
                total: rec.total,
                run_at: rec.run_at,
                report,
            }),
            Err(e) => {
                // Skip a corrupt row rather than 500ing the whole history —
                // one bad snapshot shouldn't wedge the slider for the rest.
                eprintln!(
                    "[serve] WARNING: failed to parse checkpoint-grid report_json for \
                     '{id}' epoch {}: {e}; skipping this snapshot",
                    rec.epoch
                );
            }
        }
    }
    Json(views).into_response()
}

/// `GET /api/models/{id}/jacobian-lens?force=<bool>` — serve the Jacobian tab's
/// data.
///
/// Cache-first, mirroring `eval_grid_model`'s UX: with `force=false` (a plain
/// tab-open), if `Registry::latest_jacobian_lens` has a row it's returned
/// directly — a fast DB read, no Python subprocess. With `force=true` (the
/// "Run Jacobian lens analysis" button) or when there's no cache yet, the
/// analysis is recomputed in `spawn_blocking` (it shells out to Python, which
/// is genuinely slow — several seconds to tens of seconds, dominated by
/// interpreter/torch import startup, not the math itself) and the result
/// replaces the cached row.
///
/// For a non-`Gpt` model (Bigram/Ngram), returns `200 OK` with
/// `{"not_applicable": true, "reason": "..."}` rather than an error — this is
/// an expected, structural outcome (these models have no transformer layers
/// to lens through), not a failure, so the UI can render a clean message
/// instead of an error banner.
///
/// Guarded by the same `Mutex<HashSet>` in-flight pattern as `/eval`/
/// `/eval-grid`, under a distinct key (`"{id}::jacobian"`) so a concurrent
/// recompute request for the same model 409s instead of double-spawning
/// Python, without colliding with an in-flight plain eval or grid recompute
/// for the same model id.
async fn jacobian_lens_model(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Query(query): Query<JacobianLensQuery>,
) -> Response {
    let record = {
        let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
        match get_model_or_404(&reg, &id) {
            Ok(r) => r,
            Err(resp) => return resp,
        }
    };

    if record.model_type != "gpt" {
        return Json(serde_json::json!({
            "not_applicable": true,
            "reason": format!(
                "Jacobian lens is only applicable to Gpt-type models (this model is '{}'); \
                 Bigram/Ngram models have no transformer layers to lens through.",
                record.model_type
            ),
        }))
        .into_response();
    }

    if !query.force {
        let cached = {
            let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
            reg.latest_jacobian_lens(&id)
        };
        match cached {
            Ok(Some(cached)) => match serde_json::from_str::<serde_json::Value>(&cached.results_json) {
                Ok(mut results) => {
                    if let Some(obj) = results.as_object_mut() {
                        obj.insert(
                            "plot_files".to_string(),
                            serde_json::json!(cached.plot_files),
                        );
                        obj.insert(
                            "computed_at".to_string(),
                            serde_json::json!(crate::registry::format_iso(cached.computed_at)),
                        );
                    }
                    return Json(results).into_response();
                }
                Err(e) => {
                    eprintln!(
                        "[serve] WARNING: failed to parse cached jacobian-lens JSON for '{id}': {e}; recomputing"
                    );
                }
            },
            Ok(None) => {}
            Err(e) => {
                eprintln!("[serve] latest_jacobian_lens('{id}') failed: {e}; recomputing");
            }
        }
    }

    let flight_key = format!("{id}::jacobian");
    let Some(_flight_guard) = InFlightGuard::try_acquire(state.in_flight.clone(), flight_key)
    else {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": "jacobian-lens analysis already running for this model"})),
        )
            .into_response();
    };

    let project_root = state.project_root.clone();
    let record_for_task = record.clone();

    let result = tokio::task::spawn_blocking(move || {
        run_jacobian_lens_for_model(&record_for_task, &project_root)
    })
    .await;

    match result {
        Ok(Ok(outcome)) => {
            let computed_at = crate::registry::now_unix();
            {
                let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
                if let Err(e) = reg.record_jacobian_lens(
                    &id,
                    &outcome.results_json,
                    &outcome.plot_dir_rel,
                    &outcome.plot_files,
                ) {
                    eprintln!("[serve] WARNING: failed to cache jacobian-lens result for '{id}': {e}");
                }
            }
            match serde_json::from_str::<serde_json::Value>(&outcome.results_json) {
                Ok(mut results) => {
                    if let Some(obj) = results.as_object_mut() {
                        obj.insert(
                            "plot_files".to_string(),
                            serde_json::json!(outcome.plot_files),
                        );
                        obj.insert(
                            "computed_at".to_string(),
                            serde_json::json!(crate::registry::format_iso(computed_at)),
                        );
                    }
                    Json(results).into_response()
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": format!("failed to parse jacobian-lens results JSON: {e}")})),
                )
                    .into_response(),
            }
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("jacobian-lens task panicked: {e}")})),
        )
            .into_response(),
    }
}

/// `GET /api/models/{id}/jacobian-lens/plot/{filename}` — serve one PNG plot
/// from a cached Jacobian-lens run. `filename` is matched against the cached
/// row's `plot_files` list (never trusted directly as a path component) so a
/// crafted filename can't read arbitrary files from the plot directory or
/// escape it.
async fn jacobian_lens_plot(
    State(state): State<AppState>,
    AxumPath((id, filename)): AxumPath<(String, String)>,
) -> Response {
    let (plot_dir, plot_files) = {
        let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
        match reg.latest_jacobian_lens(&id) {
            Ok(Some(rec)) => (rec.plot_dir, rec.plot_files),
            Ok(None) => {
                return (StatusCode::NOT_FOUND, "no cached jacobian-lens result for this model").into_response();
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to read jacobian-lens cache: {e}"),
                )
                    .into_response();
            }
        }
    };
    if !plot_files.iter().any(|f| f == &filename) {
        return (StatusCode::NOT_FOUND, "no such plot for this model's cached run").into_response();
    }
    let path = state.project_root.join(&plot_dir).join(&filename);
    match std::fs::read(&path) {
        Ok(bytes) => ([(axum::http::header::CONTENT_TYPE, "image/png")], bytes).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to read plot file {}: {e}", path.display()),
        )
            .into_response(),
    }
}

// --- REPL handlers ---

/// `POST /api/repl/tokenize` — build the model's tokenizer from its dataset,
/// encode the supplied text, and return per-token chips (id + decoded string),
/// the vocab size, and the tokenizer type. Does not load the model.
///
/// Heavy work (reading the corpus + building the tokenizer, which for BPE
/// trains a merge table) runs in `spawn_blocking` and is serialized by
/// `repl_lock` so concurrent REPL requests don't thrash.
async fn repl_tokenize(
    State(state): State<AppState>,
    Json(req): Json<TokenizeRequest>,
) -> Response {
    let record = {
        let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
        match get_model_or_404(&reg, &req.model_id) {
            Ok(r) => r,
            Err(resp) => return resp,
        }
    };

    // Fast path: empty text → empty tokenization, no need to touch the corpus.
    if req.text.is_empty() {
        return Json(TokenizeResponse {
            tokens: Vec::new(),
            vocab_size: 0,
            tokenizer_type: record.tokenizer.clone(),
        })
        .into_response();
    }

    let project_root = state.project_root.clone();
    let repl_lock = state.repl_lock.clone();

    let result = tokio::task::spawn_blocking(move || {
        // Hold the repl_lock only for the duration of the synchronous heavy
        // work — building a BPE tokenizer on a large corpus is the slow part.
        let _guard = repl_lock.lock().unwrap_or_else(|e| e.into_inner());
        let tokenizer = build_tokenizer(&record, &project_root)?;
        let vocab_size = tokenizer.vocab_size();
        let ids = tokenizer.encode(&req.text);
        let chips: Vec<TokenChip> = ids
            .iter()
            .map(|&id| TokenChip {
                id,
                str: tokenizer.decode(&[id]),
            })
            .collect();
        SmolResult::Ok(TokenizeResponse {
            tokens: chips,
            vocab_size,
            tokenizer_type: record.tokenizer.clone(),
        })
    })
    .await;

    match result {
        Ok(Ok(resp)) => Json(resp).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("tokenize task panicked: {e}")})),
        )
            .into_response(),
    }
}

/// `POST /api/repl/generate` — load the model, encode the prompt, run greedy or
/// temperature-sampled decoding, and return the prompt tokenization, the
/// decoded completion, and the per-token decode of each generated id.
///
/// `max_new_tokens` is clamped to the model's `block_size` to avoid runaway
/// generation. `greedy == true` OR `temperature <= 0.0` selects greedy
/// decoding; otherwise temperature sampling runs with a `StdRng` seeded from
/// 42 for reproducibility within a single request. Generation stops early at
/// the newline token (same convention as the eval harness) so line-delimited
/// datasets don't run to the full `max_new_tokens` cap on every request.
async fn repl_generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Response {
    let record = {
        let reg = state.registry.lock().unwrap_or_else(|e| e.into_inner());
        match get_model_or_404(&reg, &req.model_id) {
            Ok(r) => r,
            Err(resp) => return resp,
        }
    };

    let project_root = state.project_root.clone();
    let repl_lock = state.repl_lock.clone();
    // Clamp max_new_tokens to the model's block_size (the GPT position-embedding
    // limit and a safe upper bound on answer length); at least 1 so a 0 (or
    // absent) value from the client can't reach the generation loop, which
    // assumes it always writes at least one token.
    let block_size = record.block_size as usize;
    let max_new_tokens = req.max_new_tokens.min(block_size).max(1);

    let result = tokio::task::spawn_blocking(move || {
        // Serialize heavy work across REPL requests.
        let _guard = repl_lock.lock().unwrap_or_else(|e| e.into_inner());
        let tokenizer = build_tokenizer(&record, &project_root)?;
        let vocab_size = tokenizer.vocab_size();
        let model_type = parse_model_type(&record.model_type)?;

        let model_path = resolve_within_root(&project_root, &record.path).ok_or_else(|| {
            SmolError::custom_error(&format!("model path not found or escapes project root: {}", record.path))
        })?;
        let device = Device::Cpu;
        // Use the lossless per-block schedule (falls back to uniform
        // `num_heads` for pre-migration rows) so non-uniform architectures
        // reload with their EXACT per-block shapes, not a uniform
        // approximation derived from the summary `num_heads` column.
        let heads_schedule = crate::registry::parse_heads_schedule_column(
            &record.heads_schedule,
            record.num_heads,
            record.num_blocks,
        );
        let model = LanguageModel::load(
            model_type,
            &model_path,
            block_size,
            vocab_size,
            record.hidden_size as usize,
            &heads_schedule,
            record.num_blocks as usize,
            // EXPERIMENTAL: `--tie-embeddings` isn't tracked as a registry
            // column (out of scope for this ablation), so `--serve` can only
            // correctly load untied models. A model trained with
            // `--tie-embeddings` won't load correctly here; use the CLI
            // (`--eval`/`--generate` with matching `--tie-embeddings`)
            // for those instead.
            false,
            &device,
        )?;

        let prompt_ids = tokenizer.encode(&req.prompt);
        if prompt_ids.is_empty() {
            return Err(SmolError::invalid_argument(
                "prompt is empty (or contains no characters in the model's vocabulary); \
                 type something the model was trained on, e.g. `1+1=`",
            ));
        }
        let prompt_chips: Vec<TokenChip> = prompt_ids
            .iter()
            .map(|&id| TokenChip {
                id,
                str: tokenizer.decode(&[id]),
            })
            .collect();

        // Stop at the newline token — same convention as `run_eval`. For the
        // char tokenizer this is token 0 (newline sorts first); for byte-level
        // BPE it's byte 10. Both yield a single token. Falling back to 0 here
        // would be silently wrong if 0 happens to be some other real,
        // frequently-generated token (decoding would truncate at the first
        // occurrence of an unrelated token with no error surfaced) — so error
        // out instead, matching `run_eval`'s handling of the same case.
        let stop_token: u32 = tokenizer.encode("\n").into_iter().next().ok_or_else(|| {
            SmolError::invalid_argument("Tokenizer produced no encoding for '\\n'")
        })?;

        let generated_ids = if req.greedy || req.temperature <= 0.0 {
            model.generate_greedy_from_prompt(&prompt_ids, max_new_tokens, stop_token, &device)?
        } else {
            // Seed = 42 for reproducibility within a single request: two
            // identical requests produce identical completions, which makes
            // the playground easy to demo and debug.
            let mut rng: StdRng = StdRng::seed_from_u64(42);
            model.sample_from_prompt(
                &prompt_ids,
                max_new_tokens,
                stop_token,
                req.temperature,
                &mut rng,
                &device,
            )?
        };

        let generated_text = tokenizer.decode(&generated_ids);
        let generated_chips: Vec<TokenChip> = generated_ids
            .iter()
            .map(|&id| TokenChip {
                id,
                str: tokenizer.decode(&[id]),
            })
            .collect();

        SmolResult::Ok(GenerateResponse {
            prompt_tokens: prompt_chips,
            generated_text,
            generated_tokens: generated_chips,
        })
    })
    .await;

    match result {
        Ok(Ok(resp)) => Json(resp).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("generate task panicked: {e}")})),
        )
            .into_response(),
    }
}

// --- Helpers ---

/// Look up `id` in the registry, returning a uniform 404 JSON body on a
/// missing model or a registry error alike (both cases mean "there's no
/// model to operate on"). Shared by every handler that needs a `ModelRecord`
/// before doing its real work.
fn get_model_or_404(reg: &Registry, id: &str) -> Result<ModelRecord, Response> {
    match reg.get_model(id) {
        Ok(Some(r)) => Ok(r),
        _ => Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("model '{id}' not in registry")})),
        )
            .into_response()),
    }
}

/// Join `rel` onto `project_root` and verify the resolved, canonicalized path
/// is still contained within `project_root`. `ModelRecord.dataset`/`.path`
/// values are normally written by the CLI's own registration path, but this
/// server reads them for every `/api/models` and eval/REPL request, so a
/// corrupted or hand-edited `models.toml`/`smolgpt.db` row containing `../..`
/// shouldn't let a request escape the project directory and read/execute
/// arbitrary files reachable from the process's cwd. Returns `None` both when
/// the path doesn't exist (can't canonicalize) and when it resolves outside
/// `project_root` — callers already treat "file not found" and "rejected" the
/// same way (skip / error), so no distinct error variant is needed.
pub(crate) fn resolve_within_root(project_root: &Path, rel: &str) -> Option<PathBuf> {
    let canon_root = project_root.canonicalize().ok()?;
    let canon_candidate = project_root.join(rel).canonicalize().ok()?;
    canon_candidate.starts_with(&canon_root).then_some(canon_candidate)
}

/// Read a dataset file and compute line count, byte size, and the first 5
/// non-empty lines. Returns `None` if the file can't be read.
fn compute_dataset_info(project_root: &Path, record: &ModelRecord) -> Option<DatasetInfo> {
    let path = resolve_within_root(project_root, &record.dataset)?;
    let Ok(content) = std::fs::read_to_string(&path) else {
        return None;
    };
    let byte_size = content.len();
    let non_empty: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    let line_count = non_empty.len();
    let head = non_empty.iter().take(5).map(|s| s.to_string()).collect();
    Some(DatasetInfo {
        path: record.dataset.clone(),
        name: record.dataset_name.clone(),
        line_count,
        byte_size,
        head,
    })
}

/// How many example windows/prompts the Samples tab shows -- "a handful",
/// per the feature's spec. Cheap regardless of size (pure corpus sampling,
/// no model load), but this many is plenty to make the aligned-vs-unaligned
/// / SFT-vs-RL contrast visible without cluttering the tab.
const SAMPLE_COUNT: usize = 6;

/// Build the Samples tab's SFT view for a BASE model: read its corpus and
/// reconstruct `SAMPLE_COUNT` representative training windows via
/// `dataset::sample_example_windows` (the same sampling logic `--train`
/// uses). Returns `None` if the corpus can't be read (mirrors
/// `compute_dataset_info`'s fallback) -- a missing/unreadable dataset simply
/// hides the Samples tab's content rather than erroring the whole
/// `/api/models` response.
fn compute_sft_samples(project_root: &Path, record: &ModelRecord) -> Option<SftSamplesView> {
    let path = resolve_within_root(project_root, &record.dataset)?;
    let corpus = std::fs::read_to_string(&path).ok()?;
    let aligned = record.aligned_windows.unwrap_or(false);
    let windows = dataset::sample_example_windows(&corpus, record.block_size as usize, aligned, SAMPLE_COUNT);
    Some(SftSamplesView {
        aligned_windows: record.aligned_windows,
        windows,
    })
}

/// Build the Samples tab's RL view for an RFT/GRPO variant: reconstruct
/// `SAMPLE_COUNT` representative prompts via `eval::sample_example_prompts`,
/// using the `prompt_min`/`prompt_max`/`prompt_ops` recorded on the variant's
/// `trainings` row (see `TrainingRecord`'s doc). Returns `None` if there's no
/// training row for this variant, or it's missing the prompt fields (e.g. an
/// RFT/GRPO run recorded before these columns existed) -- an honest "no data"
/// rather than guessing a range.
fn compute_rl_samples(training: Option<&TrainingRecord>) -> Option<RlSamplesView> {
    let t = training?;
    if t.kind != "rft" && t.kind != "grpo" {
        return None;
    }
    let (min, max, ops) = (t.prompt_min?, t.prompt_max?, t.prompt_ops.clone()?);
    let prompts = sample_example_prompts(min, max, &ops, SAMPLE_COUNT).unwrap_or_else(|e| {
        eprintln!(
            "[serve] WARNING: failed to sample example prompts for '{}' (kind={}): {e}; showing none",
            t.model_id, t.kind
        );
        Vec::new()
    });
    Some(RlSamplesView {
        kind: t.kind.clone(),
        prompt_min: min,
        prompt_max: max,
        prompt_ops: ops,
        prompts,
    })
}

/// Cheap per-model status for the model list: "ok" if the `.bin` checkpoint
/// exists on disk, "missing" otherwise. The full arch/tokenizer verification
/// (which loads the model) is deliberately NOT done here — it would make
/// `/api/models` slow with many models. Mismatches surface at eval / REPL
/// generate time, when the model is loaded anyway.
fn cheap_status(record: &ModelRecord, project_root: &Path) -> String {
    if project_root.join(&record.path).exists() {
        "ok".to_string()
    } else {
        "missing".to_string()
    }
}

/// Build the tokenizer named by `record.tokenizer` from the model's dataset
/// corpus. Returns a boxed `Tokenizer<u32>` trait object so callers can encode
/// the prompt and decode generated tokens without knowing whether it's a
/// `SimpleTokenizer` or `BpeTokenizer`. Used by both REPL endpoints.
///
/// Mirrors the tokenizer-construction half of `run_eval_for_model` but stops
/// before loading the model (the tokenize endpoint doesn't need it).
fn build_tokenizer(
    record: &ModelRecord,
    project_root: &Path,
) -> SmolResult<Box<dyn Tokenizer<u32>>> {
    let corpus_path = resolve_within_root(project_root, &record.dataset).ok_or_else(|| {
        SmolError::custom_error(&format!("dataset path not found or escapes project root: {}", record.dataset))
    })?;
    let corpus = std::fs::read_to_string(&corpus_path).map_err(|e| {
        SmolError::custom_error(&format!(
            "Failed to read dataset {}: {e}",
            corpus_path.display()
        ))
    })?;

    match record.tokenizer.as_str() {
        "char" => Ok(Box::new(SimpleTokenizer::new(&corpus))),
        "bpe" => Ok(Box::new(BpeTokenizer::train(
            &corpus,
            record.vocab_size as usize,
        ))),
        other => Err(SmolError::invalid_argument(&format!(
            "Unknown tokenizer type: {other}"
        ))),
    }
}

/// Parse the registry's stored `model_type` string into the CLI enum. Shared
/// by every call site that loads a `LanguageModel` from a `ModelRecord`.
fn parse_model_type(model_type: &str) -> SmolResult<ModelType> {
    match model_type {
        "gpt" => Ok(ModelType::Gpt),
        "bigram" => Ok(ModelType::Bigram),
        "ngram" => Ok(ModelType::Ngram),
        other => Err(SmolError::invalid_argument(&format!("Unknown model type: {other}"))),
    }
}

/// Build tokenizer, load model, run eval. Called inside `spawn_blocking`.
fn run_eval_for_model(record: &ModelRecord, project_root: &Path, seed: u64) -> SmolResult<EvalReport> {
    let tokenizer = build_tokenizer(record, project_root)?;
    let vocab_size = tokenizer.vocab_size();
    let model_type = parse_model_type(&record.model_type)?;

    let model_path = resolve_within_root(project_root, &record.path).ok_or_else(|| {
        SmolError::custom_error(&format!("model path not found or escapes project root: {}", record.path))
    })?;
    let device = Device::Cpu;
    // Lossless per-block schedule (falls back to uniform `num_heads` for
    // pre-migration rows) — see the identical comment in `run_generate`'s
    // load call for why this replaces `&[record.num_heads as usize]`.
    let heads_schedule = crate::registry::parse_heads_schedule_column(
        &record.heads_schedule,
        record.num_heads,
        record.num_blocks,
    );
    let model = LanguageModel::load(
        model_type,
        &model_path,
        record.block_size as usize,
        vocab_size,
        record.hidden_size as usize,
        &heads_schedule,
        record.num_blocks as usize,
        // EXPERIMENTAL: `--tie-embeddings` isn't tracked as a registry
        // column, so `--serve` can only correctly load untied models (see
        // the other `LanguageModel::load` call site's comment above).
        false,
        &device,
    )?;

    // Derive which operators to eval from the training corpus itself, the
    // same way `eval_min`/`eval_max` are already derived (smart mode) — a
    // hardcoded "+,-" here would sample subtraction problems for a model
    // trained (and tokenizer-charset-restricted) to addition only, silently
    // mis-tokenizing and failing them rather than skipping them.
    let corpus_path = resolve_within_root(project_root, &record.dataset).ok_or_else(|| {
        SmolError::custom_error(&format!("dataset path not found or escapes project root: {}", record.dataset))
    })?;
    let corpus = std::fs::read_to_string(&corpus_path)
        .map_err(|e| SmolError::custom_error(&format!("Failed to read dataset {}: {e}", corpus_path.display())))?;
    let ops = dataset::operators_present(&corpus).unwrap_or_else(|| "+,-".to_string());

    println!(
        "[serve] Running eval for '{}' ({} samples, range [{},{}], ops={ops}, seed={})",
        record.id, record.eval_samples, record.eval_min, record.eval_max, seed
    );

    run_eval(
        &model,
        tokenizer.as_ref(),
        &device,
        record.eval_samples as usize,
        record.eval_min,
        record.eval_max,
        record.block_size as usize,
        Some(seed),
        &ops,
    )
}

/// Build tokenizer, load model, run the exhaustive grid eval. Called inside
/// `spawn_blocking`. Mirrors `run_eval_for_model` (same tokenizer/model/ops
/// derivation) but calls `run_eval_grid` instead of `run_eval` — no sampling,
/// no seed. The range cap is enforced inside `run_eval_grid` itself, so a
/// too-large range surfaces as an `Err(SmolError::InvalidArgument(_))` here,
/// which `eval_grid_model` maps to HTTP 400.
fn run_eval_grid_for_model(record: &ModelRecord, project_root: &Path) -> SmolResult<EvalGridReport> {
    let tokenizer = build_tokenizer(record, project_root)?;
    let vocab_size = tokenizer.vocab_size();
    let model_type = parse_model_type(&record.model_type)?;

    let model_path = resolve_within_root(project_root, &record.path).ok_or_else(|| {
        SmolError::custom_error(&format!("model path not found or escapes project root: {}", record.path))
    })?;
    let device = Device::Cpu;
    // Lossless per-block schedule (falls back to uniform `num_heads` for
    // pre-migration rows) — see the identical comment in `run_generate`'s
    // load call for why this replaces `&[record.num_heads as usize]`.
    let heads_schedule = crate::registry::parse_heads_schedule_column(
        &record.heads_schedule,
        record.num_heads,
        record.num_blocks,
    );
    let model = LanguageModel::load(
        model_type,
        &model_path,
        record.block_size as usize,
        vocab_size,
        record.hidden_size as usize,
        &heads_schedule,
        record.num_blocks as usize,
        // EXPERIMENTAL: `--tie-embeddings` isn't tracked as a registry
        // column, so `--serve` can only correctly load untied models (see
        // the other `LanguageModel::load` call site's comment above).
        false,
        &device,
    )?;

    let corpus_path = resolve_within_root(project_root, &record.dataset).ok_or_else(|| {
        SmolError::custom_error(&format!("dataset path not found or escapes project root: {}", record.dataset))
    })?;
    let corpus = std::fs::read_to_string(&corpus_path)
        .map_err(|e| SmolError::custom_error(&format!("Failed to read dataset {}: {e}", corpus_path.display())))?;
    let ops = dataset::operators_present(&corpus).unwrap_or_else(|| "+,-".to_string());

    println!(
        "[serve] Running eval-grid for '{}' (range [{},{}], ops={ops})",
        record.id, record.eval_min, record.eval_max
    );

    run_eval_grid(
        &model,
        tokenizer.as_ref(),
        &device,
        record.eval_min,
        record.eval_max,
        record.block_size as usize,
        &ops,
    )
}

const HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>smolgpt — model registry</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0b0f0d;
    --panel: #121713;
    --panel-2: #182019;
    --panel-3: #1d2620;
    --border: #263029;
    --border-soft: #1c2620;
    --text: #e9ede9;
    --text-dim: #8a9690;
    --text-faint: #5c6a63;
    --accent: #f0ad4e;
    --accent-dim: #8a6a2e;
    --accent-ink: #1a1305;
    --good: #5fd97a;
    --good-ink: #0c2412;
    --bad: #ff7a68;
    --bad-ink: #2b0f0b;
    --warn: #e8c547;
    --warn-ink: #241f08;
    --radius: 8px;
    --mono: ui-monospace, "SF Mono", "Cascadia Code", "JetBrains Mono", Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: light) {
    :root {
      color-scheme: light;
      --bg: #f5f4ee;
      --panel: #ffffff;
      --panel-2: #f1f0e8;
      --panel-3: #eae8dd;
      --border: #ddd8c8;
      --border-soft: #e6e2d3;
      --text: #1a1f1a;
      --text-dim: #63695f;
      --text-faint: #93917f;
      --accent: #a8660b;
      --accent-dim: #d6ac72;
      --accent-ink: #fff8ec;
      --good: #1e8a45;
      --good-ink: #eafcef;
      --bad: #c73f2d;
      --bad-ink: #fdf0ee;
      --warn: #93700a;
      --warn-ink: #fbf3de;
    }
  }
  * { box-sizing: border-box; }
  html, body { max-width: 100%; overflow-x: hidden; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    line-height: 1.5;
    padding: 28px clamp(16px, 4vw, 40px) 64px;
  }
  ::selection { background: var(--accent); color: var(--accent-ink); }
  a { color: var(--accent); }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

  /* --- Header --- */
  header.page-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px 20px;
    margin-bottom: 28px;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--border);
  }
  header.page-head .title-block h1 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.01em;
  }
  header.page-head .eyebrow {
    display: block;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: 5px;
  }
  header.page-head p.sub {
    margin: 6px 0 0;
    color: var(--text-dim);
    font-size: 0.82rem;
    max-width: 46ch;
  }
  #summary-bar {
    font-size: 0.78rem;
    color: var(--text-dim);
    white-space: nowrap;
    align-self: flex-end;
  }
  #summary-bar b { color: var(--text); font-weight: 700; }

  /* --- Model picker (base + variant selects) --- */
  .picker-bar {
    display: flex;
    gap: 22px;
    flex-wrap: wrap;
    margin-bottom: 22px;
  }
  .picker-field { display: flex; flex-direction: column; gap: 5px; }
  .picker-field label {
    font-size: 0.68rem;
    color: var(--text-faint);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .picker-select {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.86rem;
    padding: 8px 10px;
    min-width: 280px;
  }
  .picker-select:focus { outline: none; border-color: var(--accent); }

  /* --- Detail panel (single model+variant view) --- */
  .detail-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--text-faint);
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 36px;
  }
  .detail-panel[data-status="ok"] { border-left-color: var(--good); }
  .detail-panel[data-status="missing"] { border-left-color: var(--bad); }
  .detail-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px 18px;
    padding: 13px 16px;
    border-bottom: 1px solid var(--border);
  }
  .row-id { font-size: 0.98rem; font-weight: 700; word-break: break-all; }
  .tag {
    background: var(--panel-3);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 0.02em;
    white-space: nowrap;
  }
  .row-readout { text-align: right; line-height: 1.15; white-space: nowrap; }
  .row-readout .rd-num { font-size: 1.15rem; font-weight: 700; }
  .row-readout .rd-frac { font-size: 0.72rem; color: var(--text-dim); display: block; }
  .row-readout .rd-empty { font-size: 0.78rem; color: var(--text-faint); font-style: italic; }
  .good-text { color: var(--good); }
  .warn-text { color: var(--warn); }
  .bad-text { color: var(--bad); }

  .tabs {
    display: flex;
    gap: 2px;
    padding: 8px 16px 0;
    background: var(--panel-2);
  }
  .tab-btn {
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-dim);
    font-family: var(--mono);
    font-size: 0.76rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 7px 12px;
    cursor: pointer;
  }
  .tab-btn:hover { color: var(--text); }
  .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-panel { display: none; padding: 16px; }
  .tab-panel.active { display: block; }

  /* --- Overview tab --- */
  .kv-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
    gap: 10px 16px;
    margin-bottom: 14px;
  }
  .kv-grid .kv dt { font-size: 0.68rem; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 2px; }
  .kv-grid .kv dd { margin: 0; font-size: 0.88rem; font-weight: 600; }
  .params-line { font-size: 0.84rem; color: var(--text-dim); margin-bottom: 14px; }
  .params-line b { color: var(--text); }
  .dataset-block { margin-bottom: 14px; }
  .dataset-toggle {
    cursor: pointer;
    color: var(--accent);
    font-size: 0.82rem;
    user-select: none;
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .dataset-head {
    margin-top: 7px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 9px 11px;
    font-size: 0.78rem;
    color: var(--text-dim);
    white-space: pre-wrap;
    display: none;
  }
  .dataset-head.open { display: block; }
  .note-line { font-size: 0.82rem; color: var(--text-dim); border-top: 1px solid var(--border-soft); padding-top: 10px; }
  .banner {
    border-radius: 6px;
    padding: 9px 11px;
    font-size: 0.8rem;
    margin-bottom: 14px;
  }
  .banner.red { background: var(--bad-ink); border: 1px solid var(--bad); color: var(--bad); }
  .banner.yellow { background: var(--warn-ink); border: 1px solid var(--warn); color: var(--warn); }

  /* --- Training tab --- */
  .train-head {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    font-size: 0.84rem;
    color: var(--text-dim);
    margin-bottom: 10px;
  }
  .train-kind {
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--bg);
    background: var(--accent);
    border-radius: 4px;
    padding: 2px 7px;
  }
  .train-stop { color: var(--warn); }
  .train-spark { display: block; width: 100%; max-width: 320px; height: 46px; }
  .train-spark .axis { stroke: var(--border); stroke-width: 1; }
  .train-spark .line { fill: none; stroke: var(--accent); stroke-width: 1.6; }
  .train-empty { color: var(--text-faint); font-size: 0.84rem; font-style: italic; }

  /* --- Samples tab --- */
  .samples-section { margin-bottom: 18px; }
  .samples-section + .samples-section { border-top: 1px solid var(--border-soft); padding-top: 14px; }
  .samples-section-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-faint); margin-bottom: 8px; }
  .samples-heading { font-size: 0.84rem; color: var(--text-dim); margin-bottom: 6px; }
  .samples-note { font-size: 0.78rem; color: var(--text-faint); font-style: italic; margin-bottom: 8px; }
  .sample-windows { display: flex; flex-direction: column; gap: 4px; margin-top: 6px; }
  .sample-window {
    font-family: var(--mono);
    font-size: 0.84rem;
    background: var(--code-bg, rgba(127,127,127,0.08));
    border: 1px solid var(--border-soft);
    border-radius: 5px;
    padding: 6px 10px;
    white-space: pre-wrap;
    word-break: break-all;
  }
  .sample-prompt { color: var(--accent); }
  .sample-pos0 {
    background: var(--warn-ink, rgba(255,180,0,0.25));
    color: var(--warn, inherit);
    border-radius: 3px;
    padding: 0 2px;
    font-weight: 700;
  }
  .sample-nl { color: var(--text-faint); font-weight: 400; }
  .data-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  .data-table th, .data-table td {
    text-align: right;
    padding: 5px 10px;
    border-bottom: 1px solid var(--border-soft);
  }
  .data-table th { color: var(--text-faint); font-weight: 600; text-transform: uppercase; font-size: 0.68rem; letter-spacing: 0.04em; }
  .data-table th:first-child, .data-table td:first-child { text-align: left; }
  .data-table td.muted { color: var(--text-faint); }
  .train-foot { font-size: 0.78rem; color: var(--text-dim); margin-top: 8px; }

  /* --- Eval tab --- */
  .eval-toolbar { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
  .btn {
    font-family: var(--mono);
    font-size: 0.82rem;
    font-weight: 600;
    border-radius: 6px;
    padding: 7px 14px;
    cursor: pointer;
    border: 1px solid transparent;
  }
  .btn.primary { background: var(--accent); color: var(--accent-ink); }
  .btn.primary:hover { filter: brightness(1.08); }
  .btn.ghost { background: var(--panel-3); border-color: var(--border); color: var(--text); }
  .btn.ghost:hover { border-color: var(--accent); color: var(--accent); }
  .btn:disabled { opacity: 0.55; cursor: default; }
  .spinner {
    display: inline-block;
    width: 12px; height: 12px;
    border: 2px solid var(--text-faint);
    border-top-color: var(--text);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
    margin-left: 7px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .eval-result { display: none; }
  .eval-result.show { display: block; }
  .readout-row { display: flex; align-items: baseline; gap: 10px; margin-bottom: 14px; }
  .readout-big { font-size: 2.1rem; font-weight: 700; }
  .badge {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 20px;
    font-size: 0.86rem;
    font-weight: 700;
  }
  .badge.good { background: var(--good-ink); color: var(--good); }
  .badge.warn { background: var(--warn-ink); color: var(--warn); }
  .badge.bad { background: var(--bad-ink); color: var(--bad); }
  .eval-result > .data-table { margin-bottom: 12px; }
  .examples-toggle {
    cursor: pointer;
    color: var(--text-dim);
    font-size: 0.8rem;
    user-select: none;
  }
  .examples-toggle:hover { color: var(--accent); }
  .examples { font-size: 0.8rem; margin-top: 8px; display: none; }
  .examples.open { display: block; }
  .examples div { padding: 2px 0; }
  .ex-ok { color: var(--good); }
  .ex-fail { color: var(--bad); }
  .error-msg { color: var(--bad); font-size: 0.82rem; margin-top: 10px; }

  /* --- Grid tab (exhaustive eval) --- */
  .grid-toolbar { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
  .grid-op-tabs { display: flex; gap: 6px; margin-bottom: 12px; }
  .grid-op-btn {
    background: var(--panel-3);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text-dim);
    font-family: var(--mono);
    font-size: 0.78rem;
    font-weight: 700;
    padding: 4px 12px;
    cursor: pointer;
  }
  .grid-op-btn.active { border-color: var(--accent); color: var(--accent); background: var(--accent-ink); }
  .grid-mode-toggle { display: flex; gap: 6px; margin-bottom: 12px; }
  .grid-mode-btn {
    background: var(--panel-3);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text-dim);
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 3px 10px;
    cursor: pointer;
  }
  .grid-mode-btn.active { border-color: var(--accent); color: var(--accent); background: var(--accent-ink); }
  .grid-legend { display: flex; align-items: center; gap: 14px; font-size: 0.76rem; color: var(--text-dim); margin-bottom: 12px; flex-wrap: wrap; }
  .grid-legend .sw { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
  .grid-legend .grad-bar { display: inline-block; width: 70px; height: 11px; border-radius: 2px; margin-right: 2px; vertical-align: -1px; border: 1px solid var(--border-soft); }
  .sw-ok { background: var(--good); }
  .sw-fail { background: var(--bad); }
  .sw-unparsed {
    background-image: repeating-linear-gradient(45deg, var(--bad), var(--bad) 3px, var(--panel-3) 3px, var(--panel-3) 6px);
  }
  .grid-cache-note { font-size: 0.76rem; color: var(--text-faint); font-style: italic; }
  .grid-wrap { overflow: auto; max-width: 100%; border: 1px solid var(--border); border-radius: 6px; }
  table.op-grid { border-collapse: collapse; font-size: 0.68rem; }
  table.op-grid th {
    position: sticky;
    top: 0;
    background: var(--panel-2);
    color: var(--text-faint);
    font-weight: 600;
    padding: 3px 5px;
    text-align: center;
    border: 1px solid var(--border-soft);
    z-index: 1;
  }
  table.op-grid th.corner { left: 0; z-index: 2; background: var(--panel-2); }
  table.op-grid td.row-label {
    position: sticky;
    left: 0;
    background: var(--panel-2);
    color: var(--text-faint);
    font-weight: 600;
    padding: 3px 5px;
    text-align: center;
    border: 1px solid var(--border-soft);
  }
  table.op-grid td.cell {
    width: 20px;
    height: 20px;
    min-width: 20px;
    text-align: center;
    border: 1px solid var(--border-soft);
    cursor: pointer;
    color: transparent;
  }
  table.op-grid td.cell.ok { background: var(--good); }
  table.op-grid td.cell.fail { background: var(--bad); }
  table.op-grid td.cell.unparsed {
    background-image: repeating-linear-gradient(45deg, var(--bad), var(--bad) 4px, var(--panel-3) 4px, var(--panel-3) 8px);
  }
  table.op-grid td.cell:hover { outline: 2px solid var(--accent); outline-offset: -2px; }
  .grid-detail {
    margin-top: 12px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 0.84rem;
    min-height: 1.4em;
  }
  .grid-detail .gd-empty { color: var(--text-faint); font-style: italic; }
  .grid-detail .gd-prompt { color: var(--text); font-weight: 700; }
  .grid-detail .gd-line { margin-top: 3px; color: var(--text-dim); }
  .grid-detail .gd-ok { color: var(--good); font-weight: 700; }
  .grid-detail .gd-fail { color: var(--bad); font-weight: 700; }
  .grid-too-large { color: var(--text-dim); font-size: 0.84rem; }
  .grid-result { display: none; }
  .grid-result.show { display: block; }

  /* --- Jacobian tab --- */
  .jac-result { display: none; }
  .jac-result.show { display: block; }
  .jac-summary { margin-bottom: 14px; }
  .jac-summary-line { color: var(--text-dim); font-size: 0.84rem; margin-bottom: 4px; }
  .jac-cache-note { font-size: 0.76rem; color: var(--text-faint); font-style: italic; }
  .jac-not-applicable { color: var(--text-faint); font-size: 0.86rem; font-style: italic; padding: 8px 0; }
  .jac-loading { color: var(--text-dim); font-size: 0.84rem; display: flex; align-items: center; gap: 8px; }
  .jac-plots { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-top: 14px; }
  .jac-plot { border: 1px solid var(--border); border-radius: 8px; padding: 8px; background: var(--panel-2); }
  .jac-plot img { width: 100%; height: auto; border-radius: 4px; display: block; }
  .jac-plot .jac-plot-name { font-size: 0.72rem; color: var(--text-faint); margin-top: 6px; text-align: center; }

  /* --- Embeddings tab (layer-by-layer PCA/UMAP visualization) --- */
  .emb-result { display: none; }
  .emb-result.show { display: block; }
  .emb-section { margin-bottom: 26px; }
  .emb-section-title { font-weight: 700; margin-bottom: 10px; font-size: 0.78rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em; }
  .emb-we-plot, .emb-layer-plot {
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel-2);
    padding: 8px;
    max-width: 520px;
  }
  .emb-scatter { width: 100%; height: auto; display: block; }
  .emb-legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; font-size: 0.74rem; color: var(--text-dim); }
  .emb-legend-item { display: flex; align-items: center; gap: 4px; }
  .emb-legend .sw { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .emb-umap-note { color: var(--text-faint); font-size: 0.78rem; font-style: italic; margin-top: 10px; }
  .emb-not-applicable { color: var(--text-faint); font-size: 0.86rem; font-style: italic; padding: 8px 0; }
  .emb-neighbors-panel { margin-top: 14px; }
  .emb-neighbors-controls { font-size: 0.76rem; color: var(--text-dim); margin-bottom: 6px; }
  .emb-neighbors-controls input { width: 44px; margin-left: 4px; }
  .emb-neighbors-list { margin: 0 0 6px 20px; padding: 0; font-size: 0.82rem; }
  .emb-neighbors-list li { margin-bottom: 2px; }
  .emb-neighbors-note { font-size: 0.76rem; color: var(--text-faint); font-style: italic; }
  .emb-pca-ref-row { display: flex; align-items: center; gap: 10px; font-size: 0.78rem; color: var(--text-dim); margin-bottom: 10px; flex-wrap: wrap; }
  .emb-pca-ref-row select { margin-left: 4px; }
  .emb-pca-ref-row select:disabled { opacity: 0.5; cursor: not-allowed; }
  .emb-pca-ref-note { font-size: 0.74rem; color: var(--text-faint); font-style: italic; }
  .emb-radar-plot {
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel-2);
    padding: 8px;
    max-width: 340px;
  }
  .emb-radar-svg { width: 100%; height: auto; display: block; }
  .emb-radar-note { font-size: 0.72rem; color: var(--text-faint); margin-top: 8px; max-width: 400px; }
  .emb-shape-note { font-size: 0.78rem; color: var(--text-dim); margin-top: 6px; max-width: 400px; }
  /* Selected token's radar and its neighbors' radars, side by side at equal
     size with a vertical separator -- deliberately NOT smaller thumbnails,
     so the eye can compare shapes directly rather than reading one as the
     "main" plot and the rest as an afterthought. */
  .emb-radar-row { display: flex; align-items: flex-start; gap: 16px; flex-wrap: wrap; margin-top: 8px; }
  .emb-radar-item, .emb-neighbor-radar-item { width: 180px; text-align: center; }
  .emb-radar-item .emb-radar-plot, .emb-neighbor-radar-item .emb-radar-plot { max-width: 180px; padding: 8px; margin: 0 auto; }
  .emb-radar-sep { width: 1px; align-self: stretch; min-height: 160px; background: var(--border); }
  .emb-radar-self-label { font-size: 0.78rem; color: var(--text); font-weight: 600; margin-top: 6px; }
  .emb-neighbor-radars { display: flex; flex-wrap: wrap; gap: 14px; }
  .emb-neighbor-radar-label { font-size: 0.72rem; color: var(--text-dim); margin-top: 4px; word-break: break-word; }
  .emb-neighbor-radar-dist { font-size: 0.68rem; color: var(--text-faint); font-style: italic; }
  /* Causal-context radars: the selected token's own preceding positions in
     its own sentence, left-to-right in reading order -- same fixed-scale
     radar chart style as the neighbor row above, just a different set. */
  .emb-context-radars { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 8px; }
  .emb-context-radar-item { width: 180px; text-align: center; }
  .emb-context-radar-item .emb-radar-plot { max-width: 180px; padding: 8px; margin: 0 auto; }
  .emb-context-radar-label { font-size: 0.72rem; color: var(--text-dim); margin-top: 4px; word-break: break-word; }
  .emb-context-radar-pos { font-size: 0.68rem; color: var(--text-faint); font-style: italic; }
  .emb-context-note { font-size: 0.78rem; color: var(--text-dim); margin-top: 6px; font-style: italic; }

  /* --- Grid tab: training-progress checkpoint animation --- */
  .ckpt-panel:empty { display: none; }
  .ckpt-panel {
    margin-top: 22px;
    padding-top: 18px;
    border-top: 1px solid var(--border);
  }
  .ckpt-loading, .ckpt-empty {
    color: var(--text-faint);
    font-size: 0.82rem;
    font-style: italic;
  }
  .ckpt-header {
    font-size: 0.72rem;
    color: var(--text-faint);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
  }
  .ckpt-header .ckpt-count { text-transform: none; letter-spacing: normal; font-weight: 500; }
  .ckpt-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
    flex-wrap: wrap;
  }
  .ckpt-slider {
    flex: 1 1 220px;
    min-width: 160px;
    accent-color: var(--accent);
    cursor: pointer;
  }
  .ckpt-label {
    font-size: 0.78rem;
    color: var(--text-dim);
    white-space: nowrap;
  }

  /* --- REPL / Playground panel --- */
  .repl-panel { border-top: 1px solid var(--border); padding-top: 26px; }
  .repl-head { margin-bottom: 18px; }
  .repl-head h2 { margin: 0; font-size: 1rem; font-weight: 700; }
  .repl-head .eyebrow { display: block; font-size: 0.68rem; letter-spacing: 0.18em; color: var(--accent); font-weight: 700; margin-bottom: 5px; }
  .repl-head p { margin: 6px 0 0; color: var(--text-dim); font-size: 0.82rem; max-width: 60ch; }
  .repl-grid {
    display: grid;
    grid-template-columns: minmax(280px, 1fr) minmax(300px, 1.1fr);
    gap: 20px;
    align-items: start;
  }
  .repl-pane {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
  }
  .repl-pane h3 {
    margin: 0 0 12px;
    font-size: 0.72rem;
    color: var(--text-faint);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .repl-field { display: flex; flex-direction: column; gap: 5px; margin-bottom: 14px; }
  .repl-field label { font-size: 0.76rem; color: var(--text-dim); }
  .repl select, .repl textarea, .repl input[type="number"] {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.88rem;
    padding: 8px 10px;
  }
  .repl select:focus, .repl textarea:focus, .repl input[type="number"]:focus {
    outline: none;
    border-color: var(--accent);
  }
  .repl textarea {
    font-size: 0.9rem;
    min-height: 68px;
    resize: vertical;
    line-height: 1.4;
  }
  .repl-examples { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 14px; }
  .repl-examples-label { font-size: 0.74rem; color: var(--text-faint); margin-right: 2px; }
  .ex-chip {
    background: var(--panel-3);
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.78rem;
    padding: 3px 10px;
    cursor: pointer;
  }
  .ex-chip:hover { border-color: var(--accent); color: var(--accent); }
  .ex-chip:active { transform: translateY(1px); }
  .repl-controls { display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-end; margin-bottom: 16px; }
  .repl-controls .repl-field { flex: 0 0 auto; min-width: 108px; margin-bottom: 0; }
  .repl-controls .repl-field input[type="number"] { width: 92px; }
  .repl-controls .checkbox-field {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.84rem; color: var(--text-dim);
    padding-bottom: 8px;
  }
  .repl-buttons { display: flex; gap: 10px; flex-wrap: wrap; }
  .repl-error { color: var(--bad); font-size: 0.82rem; margin-top: 10px; }
  .out-section + .out-section { margin-top: 18px; padding-top: 18px; border-top: 1px solid var(--border-soft); }
  .out-section h4 { margin: 0 0 8px; font-size: 0.7rem; color: var(--text-faint); font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; }
  .tok-chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .tok-chip {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    border-radius: 6px;
    padding: 4px 8px 3px;
    border: 1px solid var(--border);
    background: var(--panel-3);
    min-width: 26px;
    font-family: var(--mono);
  }
  .tok-chip .tok-str {
    font-size: 0.82rem;
    color: var(--text);
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .tok-chip .tok-str.ws { color: var(--text-faint); }
  .tok-chip .tok-id { font-size: 0.62rem; color: var(--text-faint); margin-top: 1px; }
  .gen-out {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    font-family: var(--mono);
    font-size: 0.9rem;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
    min-height: 1.4em;
  }
  .repl-empty { color: var(--text-faint); font-size: 0.82rem; font-style: italic; }
  .repl-placeholder { color: var(--text-faint); font-size: 0.82rem; font-style: italic; }

  @media (max-width: 860px) {
    .repl-grid { grid-template-columns: 1fr; }
  }
  @media (max-width: 600px) {
    body { padding: 18px 14px 48px; }
    .row-head { grid-template-columns: 1fr auto; }
    .row-readout { grid-column: 1 / -1; text-align: left; padding-top: 4px; }
    #summary-bar { align-self: flex-start; }
  }
  @media (prefers-reduced-motion: reduce) {
    .spinner { animation: none; }
    .chevron { transition: none; }
  }
</style>
</head>
<body>
<header class="page-head">
  <div class="title-block">
    <span class="eyebrow">SMOLGPT</span>
    <h1>model registry</h1>
    <p class="sub">Trained checkpoints, datasets, and eval history. Expand a model to see architecture, training curves, and run a fresh eval.</p>
  </div>
  <div id="summary-bar"></div>
</header>
<div class="picker-bar">
  <div class="picker-field">
    <label for="base-select">Model</label>
    <select id="base-select" class="picker-select"><option value="">Loading models…</option></select>
  </div>
  <div class="picker-field">
    <label for="variant-select">Variant</label>
    <select id="variant-select" class="picker-select"><option value="">—</option></select>
  </div>
</div>
<div class="detail-panel" id="detail-panel">
  <div class="repl-placeholder" style="padding:16px">Loading models…</div>
</div>
<section class="repl-panel" id="repl-panel">
  <div class="repl-head">
    <span class="eyebrow">PLAYGROUND</span>
    <h2>REPL</h2>
    <p>Pick a model, type a prompt, and tokenize or generate. Generation runs greedy or temperature-sampled decoding on the server.</p>
  </div>
  <div class="repl-grid repl">
    <div class="repl-pane">
      <h3>Input</h3>
      <div class="repl-field">
        <label for="repl-model">Model</label>
        <select id="repl-model"><option value="">Loading models…</option></select>
      </div>
      <div class="repl-field">
        <label for="repl-prompt">Prompt</label>
        <textarea id="repl-prompt" placeholder="3+4=  or  type a prompt…"></textarea>
      </div>
      <div class="repl-examples">
        <span class="repl-examples-label">examples:</span>
        <button class="ex-chip" data-prompt="0+0=">0+0=</button>
        <button class="ex-chip" data-prompt="1+1=">1+1=</button>
        <button class="ex-chip" data-prompt="3+4=">3+4=</button>
        <button class="ex-chip" data-prompt="5+5=">5+5=</button>
        <button class="ex-chip" data-prompt="8+3=">8+3=</button>
        <button class="ex-chip" data-prompt="9+9=">9+9=</button>
      </div>
      <div class="repl-controls">
        <div class="repl-field">
          <label for="repl-max">max new tokens</label>
          <input id="repl-max" type="number" min="1" value="16" />
        </div>
        <div class="repl-field">
          <label for="repl-temp">temperature</label>
          <input id="repl-temp" type="number" min="0" step="0.1" value="1.0" />
        </div>
        <label class="checkbox-field"><input id="repl-greedy" type="checkbox" checked /> greedy</label>
      </div>
      <div class="repl-buttons">
        <button class="btn ghost" id="repl-tokenize-btn">Tokenize</button>
        <button class="btn primary" id="repl-generate-btn">Generate</button>
      </div>
      <div class="repl-error" id="repl-error"></div>
    </div>
    <div class="repl-pane">
      <h3>Output</h3>
      <div id="repl-output-empty" class="repl-placeholder">Run tokenize or generate to see output here.</div>
      <div class="out-section" id="repl-tokenize-section" style="display:none">
        <h4>Tokenization</h4>
        <div class="tok-chips" id="repl-tok-chips"></div>
        <div id="repl-tok-meta" class="repl-empty" style="margin-top:8px"></div>
      </div>
      <div class="out-section" id="repl-gen-section" style="display:none">
        <h4>Prompt tokens</h4>
        <div class="tok-chips" id="repl-prompt-chips"></div>
        <h4 style="margin-top:14px">Generated text</h4>
        <div class="gen-out" id="repl-gen-text"></div>
        <h4 style="margin-top:14px">Generated tokens</h4>
        <div class="tok-chips" id="repl-gen-chips"></div>
      </div>
    </div>
  </div>
</section>
<script>
// Module-level state for the single-detail-panel picker UI: `allModels` is
// the full `/api/models` response (base models with nested `variants`);
// `activeTab` is the currently-selected tab name ("overview"/"training"/
// "eval"/"grid"), preserved across both the base-select and variant-select
// changing so switching either dropdown doesn't bounce the user back to
// Overview.
let allModels = [];
let activeTab = 'overview';
function fmtParams(n) {
  if (n == null) return '—';
  if (n >= 1000) return (n/1000).toFixed(1) + 'K';
  return String(n);
}
function fmtBytes(n) {
  if (n == null) return '—';
  if (n >= 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + ' MB';
  if (n >= 1024) return (n / 1024).toFixed(1) + ' KB';
  return n + ' B';
}
function pctClass(p) {
  if (p >= 50) return 'good';
  if (p >= 20) return 'warn';
  return 'bad';
}
function pctTextClass(p) {
  if (p >= 50) return 'good-text';
  if (p >= 20) return 'warn-text';
  return 'bad-text';
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
// `renderEval` builds the eval panel for a card's Eval tab. The headline
// (X/total + %) is always visible; the per-operator (+/-) and per-answer-digit
// breakdown tables are shown directly (they already live behind a tab click,
// so no further nesting is needed); the worked examples collapse behind a
// separate toggle since there can be up to 10 of them. `by_digits` buckets are
// answer-digit buckets (1-digit answer, 2-digit answer, 3+-digit answer); old
// cached eval rows from before the bucketing change may have a different
// shape — we just render whatever buckets are present, so the UI tolerates
// both old and new shapes without a migration.
function renderEval(el, report) {
  if (!report) return;
  const pct = report.total > 0 ? (report.correct / report.total * 100) : 0;
  const pctStr = pct.toFixed(1) + '%';
  const bd = report.by_digits || [];
  const examples = report.examples || [];
  const digitLabels = ['1-digit answer', '2-digit answer', '3+-digit answer', '4+-digit answer'];
  const rows = bd.map((b, i) => {
    const label = digitLabels[i] || (`${i+1}-digit`);
    return `<tr><td>${label}</td><td>${b?b[0]:0}</td><td>${b?b[1]:0}</td></tr>`;
  }).join('');
  const exHtml = examples.slice(0,10).map(ex => {
    const gen = (ex.generated||'').split('\n')[0];
    const cls = ex.correct ? 'ex-ok' : 'ex-fail';
    const mark = ex.correct ? 'ok' : 'FAIL';
    return `<div class="${cls}">[${mark}] ${escapeHtml(ex.prompt)}${escapeHtml(gen)} (true: ${ex.true_answer})</div>`;
  }).join('');
  el.innerHTML =
    `<div class="readout-row"><span class="readout-big">${report.correct}/${report.total}</span><span class="badge ${pctClass(pct)}">${pctStr}</span></div>` +
    `<table class="data-table"><tr><th>op</th><th>correct</th><th>total</th></tr>` +
    `<tr><td>+</td><td>${report.correct_plus}</td><td>${report.total_plus}</td></tr>` +
    `<tr><td>-</td><td>${report.correct_minus}</td><td>${report.total_minus}</td></tr></table>` +
    (rows ? `<table class="data-table"><tr><th>answer digits</th><th>correct</th><th>total</th></tr>${rows}</table>` : '') +
    (exHtml ? `<div class="examples-toggle">▸ worked examples (${examples.length})</div><div class="examples">${exHtml}</div>` : '');
  el.classList.add('show');
  const exToggle = el.querySelector('.examples-toggle');
  if (exToggle) {
    exToggle.addEventListener('click', () => {
      const box = el.querySelector('.examples');
      if (box) box.classList.toggle('open');
    });
  }
}
// --- Training-metrics rendering ---
//
// `renderTraining` builds the markup for a card's Training tab from the
// `training` field of a `ModelView`. For SFT it shows the epochs/final
// loss/early-stop line plus an inline-SVG sparkline of the per-epoch loss
// trajectory (downsampled to <=120 points so a 2000-epoch series doesn't
// render 2000 SVG nodes). For RFT/GRPO it shows the round count plus a
// per-round table (winner_rate%/correct%, eval%, loss). When `training` is
// null (no `trainings` row for this model), it shows a muted placeholder.
// The table is rendered directly (no <details> needed — it's already behind
// the Training tab click, so a further collapse would just add friction).
function downsampleLosses(losses, maxPoints) {
  if (!losses || !losses.length) return [];
  if (losses.length <= maxPoints) return losses.slice();
  // Stride-based downsampling: pick every Nth point so the sparkline keeps the
  // full epoch range on the x axis. We always include the first and last
  // points so the start and end of training are visible.
  const stride = Math.ceil(losses.length / maxPoints);
  const out = [];
  for (let i = 0; i < losses.length; i += stride) out.push(losses[i]);
  if (out[out.length - 1] !== losses[losses.length - 1]) {
    out.push(losses[losses.length - 1]);
  }
  return out;
}
function sparklineSvg(losses) {
  const W = 320, H = 46, PAD = 3;
  if (!losses || losses.length === 0 || losses.length < 2) {
    // Need at least 2 points to draw a line; otherwise show a flat baseline
    // so the section doesn't look broken.
    return `<svg class="train-spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <line class="axis" x1="${PAD}" y1="${H-PAD}" x2="${W-PAD}" y2="${H-PAD}" />
    </svg>`;
  }
  const minL = Math.min(...losses);
  const maxL = Math.max(...losses);
  const span = Math.max(maxL - minL, 1e-6);
  const x = (i) => PAD + (i / (losses.length - 1)) * (W - 2*PAD);
  const y = (v) => PAD + (1 - (v - minL) / span) * (H - 2*PAD);
  const d = losses.map((v, i) => (i === 0 ? 'M' : 'L') + x(i).toFixed(2) + ' ' + y(v).toFixed(2)).join(' ');
  return `<svg class="train-spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <line class="axis" x1="${PAD}" y1="${H-PAD}" x2="${W-PAD}" y2="${H-PAD}" />
    <path class="line" d="${d}" />
  </svg>`;
}
function fmtLoss(v) {
  if (v == null) return '—';
  return Number(v).toFixed(4);
}
function renderRftTable(s) {
  const rounds = s.rounds || 0;
  const rates = s.winner_rates || [];
  const evals = s.eval_correct_pct || [];
  const sfts = s.per_round_sft_final_losses || [];
  let rows = '';
  for (let i = 0; i < rounds; i++) {
    const r = rates[i] != null ? rates[i].toFixed(1) + '%' : '—';
    const e = evals[i] != null ? evals[i].toFixed(1) + '%' : '—';
    const l = sfts[i];
    const lStr = (l == null) ? '<span class="muted">skipped</span>' : fmtLoss(l);
    rows += `<tr><td>${i+1}</td><td>${r}</td><td>${e}</td><td>${lStr}</td></tr>`;
  }
  return `<table class="data-table">
    <tr><th>round</th><th>winner%</th><th>eval%</th><th>sft_loss</th></tr>
    ${rows}
  </table>`;
}
function renderGrpoTable(s) {
  const rounds = s.rounds || 0;
  const rates = s.correct_rates || [];
  const evals = s.eval_correct_pct || [];
  const losses = s.per_round_losses || [];
  let rows = '';
  for (let i = 0; i < rounds; i++) {
    const r = rates[i] != null ? rates[i].toFixed(1) + '%' : '—';
    const e = evals[i] != null ? evals[i].toFixed(1) + '%' : '—';
    const l = losses[i];
    const lStr = (l == null) ? '<span class="muted">no step</span>' : fmtLoss(l);
    rows += `<tr><td>${i+1}</td><td>${r}</td><td>${e}</td><td>${lStr}</td></tr>`;
  }
  const g = s.group_size || '—';
  const mode = s.mode || 'lite';
  const modeDesc = mode === 'full'
    ? 'PPO-style: importance ratio + clipping + KL-to-reference'
    : 'group-relative advantage, no clipping/KL';
  return `<table class="data-table">
    <tr><th>round</th><th>correct%</th><th>eval%</th><th>pg_loss</th></tr>
    ${rows}
  </table>` + `<div class="train-foot">G=${g} · ${escapeHtml(mode)} · ${modeDesc}</div>`;
}
function renderTraining(m) {
  const t = m.training;
  if (!t) {
    return '<div class="train-empty">no training metrics recorded</div>';
  }
  if (t.kind === 'rft') {
    const s = t.rft_summary;
    const head = `<div class="train-head"><span class="train-kind">rft</span>` +
      `<span>${t.epochs_run} rounds</span></div>`;
    const table = s ? renderRftTable(s) : '<div class="train-empty">RFT summary unavailable</div>';
    return `${head}${table}`;
  }
  if (t.kind === 'grpo') {
    const s = t.grpo_summary;
    const head = `<div class="train-head"><span class="train-kind">grpo</span>` +
      `<span>${t.epochs_run} rounds</span></div>`;
    const table = s ? renderGrpoTable(s) : '<div class="train-empty">GRPO summary unavailable</div>';
    return `${head}${table}`;
  }
  // SFT — head (epochs/final loss/train accuracy/early-stop) + sparkline.
  const stopClause = t.early_stopped
    ? '<span class="train-stop">early-stopped</span>'
    : '<span>completed</span>';
  // Exact greedy-decoding accuracy over the literal training corpus —
  // distinct from the model card's `cached_eval`, which samples random
  // operands from a range and can include problems the corpus never
  // contained. `null` for rows recorded before this was tracked.
  const trainAccClause = (t.train_correct != null && t.train_total != null && t.train_total > 0)
    ? ` · train acc ${(t.train_correct * 100 / t.train_total).toFixed(1)}% (${t.train_correct}/${t.train_total})`
    : '';
  const head = `<div class="train-head"><span class="train-kind">sft</span>` +
    `<span>${t.epochs_run} epochs · final loss ${fmtLoss(t.final_loss)}${trainAccClause} · ${stopClause}</span></div>`;
  const spark = sparklineSvg(downsampleLosses(t.loss_trajectory || [], 120));
  return `${head}${spark}`;
}
// `findModelById` looks up a model by id in the loaded `models` array,
// searching both base cards and their nested `variants`. Returns the
// matching `ModelView` (a base or a variant) or `null`.
function findModelById(models, id) {
  for (const m of models) {
    if (m.id === id) return m;
    if (m.variants) {
      for (const v of m.variants) {
        if (v.id === id) return v;
      }
    }
  }
  return null;
}
// --- Samples tab rendering ---
//
// Renders one text window (a training window or an RL prompt) with its
// position-0 character visually marked (`.sample-pos0`) so it's obvious at a
// glance whether position 0 lands on a clean fact boundary (aligned) or an
// arbitrary mid-fact character (unaligned). Newlines inside a window are
// shown as a visible "\n" marker (not just a literal line break) so a
// trailing/embedded newline is legible instead of collapsing into invisible
// whitespace.
function renderSampleWindow(text, extraClass) {
  const chars = Array.from(text);
  if (!chars.length) return `<div class="sample-window ${extraClass||''}"><span class="muted">(empty)</span></div>`;
  const renderChar = (c) => c === '\n' ? '<span class="sample-nl">\\n</span>\n' : escapeHtml(c);
  const first = `<span class="sample-pos0">${renderChar(chars[0])}</span>`;
  const rest = chars.slice(1).map(renderChar).join('');
  return `<div class="sample-window ${extraClass||''}">${first}${rest}</div>`;
}
function renderSftSamples(view) {
  if (!view) {
    return '<div class="muted">no example windows available (dataset unreadable, or corpus too short for block_size).</div>';
  }
  const known = view.aligned_windows === true || view.aligned_windows === false;
  const alignedBadge = !known
    ? '<span class="badge warn">unknown — trained before this was tracked, assuming unaligned/default</span>'
    : (view.aligned_windows
        ? '<span class="badge good">--aligned-windows ON — every window starts at a true fact boundary</span>'
        : '<span class="badge bad">--aligned-windows OFF (default) — windows start at a random offset, often mid-fact</span>');
  const windowsHtml = (view.windows || []).length
    ? view.windows.map(w => renderSampleWindow(w, '')).join('')
    : '<div class="muted">no windows could be sampled (corpus too short for block_size).</div>';
  return `<div class="samples-heading">SFT training windows (block_size chars each; position 0 highlighted)</div>` +
    `<div style="margin-bottom:8px">${alignedBadge}</div>` +
    `<div class="sample-windows">${windowsHtml}</div>`;
}
function renderRlSamples(view) {
  if (!view) {
    return '<div class="muted">no RL prompt-sampling metadata recorded for this variant (trained before this was tracked).</div>';
  }
  const promptsHtml = (view.prompts || []).length
    ? view.prompts.map(p => renderSampleWindow(p, 'sample-prompt')).join('')
    : '<div class="muted">no prompts could be sampled.</div>';
  return `<div class="samples-heading">${escapeHtml((view.kind||'').toUpperCase())} prompt sampling — operands in [${view.prompt_min}, ${view.prompt_max}], ops "${escapeHtml(view.prompt_ops)}"</div>` +
    `<div class="samples-note">Clean, complete "a op b=" prompts — contrast with the SFT stage's possibly mid-fact windows above.</div>` +
    `<div class="sample-windows">${promptsHtml}</div>`;
}
// Builds the Samples tab: for a base model, just its own SFT windows; for an
// RL variant, its BASE model's SFT windows (looked up via `base_model_id` in
// the already-loaded `allModels`) shown alongside this variant's own RL-stage
// prompts, so switching from base to variant in the picker visibly contrasts
// the two input formats.
function renderSamples(m) {
  if (m.base_model_id) {
    const base = findModelById(allModels, m.base_model_id);
    let html = '';
    html += `<div class="samples-section"><div class="samples-section-title">Base model's SFT stage (${escapeHtml(m.base_model_id)})</div>` +
      renderSftSamples(base ? base.sft_samples : null) + `</div>`;
    html += `<div class="samples-section"><div class="samples-section-title">This variant's RL stage (${escapeHtml(m.training ? m.training.kind : '?')})</div>` +
      renderRlSamples(m.rl_samples) + `</div>`;
    return html;
  }
  return renderSftSamples(m.sft_samples);
}
async function runEval(row, modelId, btn) {
  const resultEl = row.querySelector('.eval-result');
  const errEl = row.querySelector('.error-msg');
  errEl.textContent = '';
  btn.disabled = true;
  btn.innerHTML = 'Running…<span class="spinner"></span>';
  try {
    const res = await fetch('/api/models/' + encodeURIComponent(modelId) + '/eval');
    const data = await res.json();
    if (!res.ok) {
      errEl.textContent = data.error || ('HTTP ' + res.status);
    } else {
      renderEval(resultEl, data);
    }
  } catch (e) {
    errEl.textContent = String(e);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run eval';
  }
}
// --- Exhaustive-grid rendering ---
//
// `renderGrid` builds the full Grid tab UI from an `EvalGridReport`: an
// overall accuracy readout, a legend, a pass/fail-vs-gradient mode toggle, one
// op-toggle button per grid (if there are >1 operators), and the `a` x `b`
// colored table for the active operator+mode. Cells are hoverable/clickable —
// clicking one shows its prompt, the model's raw generated completion, the
// true answer, and the numeric diff in a detail box below the table, so a
// user can verify any individual cell against a manual run in either mode.
//
// Color modes:
//   - "passfail" (default): green = correct, red = incorrect. Exactly the
//     original binary view.
//   - "gradient": cells are colored along a green -> amber -> red scale by
//     how far `diff = parsed_answer - true_answer` is from zero, normalized
//     per-operator-grid against that grid's own max |diff| (so the scale
//     always uses its full range regardless of how wrong this particular
//     model gets). Cells where the model's output didn't parse as a number at
//     all (`diff == null`) are NOT given an arbitrary numeric color — they get
//     a distinct diagonal-stripe pattern (`.cell.unparsed`) so "way off but a
//     number" and "not even a number" stay visually distinguishable.
function lerpChannel(a, b, t) { return Math.round(a + (b - a) * t); }
function hexToRgb(hex) {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function rgbToCss(rgb) { return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`; }
// Interpolates green -> amber -> red for t in [0, 1], reusing the page's own
// --good/--warn/--bad CSS variables (read via getComputedStyle) so the
// gradient always matches the current theme (including the light-mode
// override) rather than hardcoding a separate palette.
function gradientColor(t) {
  const style = getComputedStyle(document.documentElement);
  const good = hexToRgb(style.getPropertyValue('--good').trim());
  const warn = hexToRgb(style.getPropertyValue('--warn').trim());
  const bad = hexToRgb(style.getPropertyValue('--bad').trim());
  const clamped = Math.max(0, Math.min(1, t));
  const [from, to, localT] = clamped <= 0.5
    ? [good, warn, clamped / 0.5]
    : [warn, bad, (clamped - 0.5) / 0.5];
  return rgbToCss([0, 1, 2].map(i => lerpChannel(from[i], to[i], localT)));
}
function renderGrid(el, report) {
  if (!report) return;
  const pct = report.total > 0 ? (report.correct / report.total * 100) : 0;
  const pctStr = pct.toFixed(1) + '%';
  const grids = report.grids || [];
  const opTabs = grids.length > 1
    ? `<div class="grid-op-tabs">${grids.map((g, i) => `<button class="grid-op-btn${i===0?' active':''}" data-op-idx="${i}">${escapeHtml(g.op)}</button>`).join('')}</div>`
    : '';
  el.innerHTML =
    `<div class="readout-row"><span class="readout-big">${report.correct}/${report.total}</span><span class="badge ${pctClass(pct)}">${pctStr}</span><span style="color:var(--text-dim);font-size:0.8rem;">exhaustive over [${report.min},${report.max}]</span></div>` +
    `<div class="grid-mode-toggle"><button class="grid-mode-btn active" data-mode="passfail">pass/fail</button><button class="grid-mode-btn" data-mode="gradient">gradient</button></div>` +
    `<div class="grid-legend"></div>` +
    opTabs +
    `<div class="grid-body"></div>` +
    `<div class="grid-detail"><span class="gd-empty">hover or click a cell to see the prompt, generated output, true answer, and diff</span></div>`;
  el.classList.add('show');

  const bodyEl = el.querySelector('.grid-body');
  const detailEl = el.querySelector('.grid-detail');
  const legendEl = el.querySelector('.grid-legend');
  let mode = 'passfail';

  function renderLegend() {
    if (mode === 'passfail') {
      legendEl.innerHTML = `<span><span class="sw sw-ok"></span>correct</span><span><span class="sw sw-fail"></span>incorrect</span><span>rows = a, columns = b</span>`;
    } else {
      legendEl.innerHTML = `<span class="grad-bar" style="background:linear-gradient(to right, ${gradientColor(0)}, ${gradientColor(0.5)}, ${gradientColor(1)})"></span>` +
        `<span>exact (diff 0)</span><span>&rarr;</span><span>far off</span>` +
        `<span><span class="sw sw-unparsed"></span>unparseable output</span><span>rows = a, columns = b</span>`;
    }
  }

  function showCellDetail(cell) {
    const cls = cell.correct ? 'gd-ok' : 'gd-fail';
    const mark = cell.correct ? 'correct' : 'INCORRECT';
    const gen = (cell.generated || '').split('\n')[0];
    const diffStr = cell.diff == null
      ? '<span class="gd-fail">could not parse a number from the output</span>'
      : `diff ${cell.diff > 0 ? '+' : ''}${cell.diff}`;
    detailEl.innerHTML =
      `<div class="gd-prompt">${escapeHtml(cell.prompt)}</div>` +
      `<div class="gd-line">model output: <b>${escapeHtml(gen) || '<i>(empty)</i>'}</b> · true answer: <b>${cell.true_answer}</b> · <span class="${cls}">${mark}</span> · ${diffStr}</div>`;
  }

  function renderOpGrid(g) {
    const min = report.min, max = report.max;
    // Index cells by (a,b) for O(1) lookup while building the table.
    const byAB = new Map();
    let maxAbsDiff = 1;
    for (const c of g.cells) {
      byAB.set(c.a + ',' + c.b, c);
      if (c.diff != null) maxAbsDiff = Math.max(maxAbsDiff, Math.abs(c.diff));
    }
    let thead = '<tr><th class="corner">a\\b</th>';
    for (let b = min; b <= max; b++) thead += `<th>${b}</th>`;
    thead += '</tr>';
    let rows = '';
    for (let a = min; a <= max; a++) {
      rows += `<tr><td class="row-label">${a}</td>`;
      for (let b = min; b <= max; b++) {
        const cell = byAB.get(a + ',' + b);
        let cls = '';
        let style = '';
        if (cell) {
          if (mode === 'passfail') {
            cls = cell.correct ? 'ok' : 'fail';
          } else if (cell.diff == null) {
            cls = 'unparsed';
          } else {
            style = `style="background:${gradientColor(Math.abs(cell.diff) / maxAbsDiff)}"`;
          }
        }
        rows += `<td class="cell ${cls}" ${style} data-a="${a}" data-b="${b}" title="${a}${escapeHtml(g.op)}${b}=">·</td>`;
      }
      rows += '</tr>';
    }
    const gPct = g.total > 0 ? (g.correct / g.total * 100) : 0;
    bodyEl.innerHTML =
      `<div style="margin-bottom:8px;font-size:0.8rem;color:var(--text-dim);">op <b style="color:var(--text)">${escapeHtml(g.op)}</b>: ${g.correct}/${g.total} (${gPct.toFixed(1)}%)</div>` +
      `<div class="grid-wrap"><table class="op-grid"><thead>${thead}</thead><tbody>${rows}</tbody></table></div>`;
    const cellEls = bodyEl.querySelectorAll('td.cell');
    cellEls.forEach(td => {
      const a = Number(td.dataset.a), b = Number(td.dataset.b);
      const cell = byAB.get(a + ',' + b);
      if (!cell) return;
      td.addEventListener('mouseenter', () => showCellDetail(cell));
      td.addEventListener('click', () => showCellDetail(cell));
    });
  }

  renderLegend();
  if (grids.length) renderOpGrid(grids[0]);
  const opBtns = el.querySelectorAll('.grid-op-btn');
  opBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      opBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderOpGrid(grids[Number(btn.dataset.opIdx)]);
    });
  });
  const modeBtns = el.querySelectorAll('.grid-mode-btn');
  modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      if (btn.dataset.mode === mode) return;
      modeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      mode = btn.dataset.mode;
      renderLegend();
      const activeOpBtn = el.querySelector('.grid-op-btn.active');
      const idx = activeOpBtn ? Number(activeOpBtn.dataset.opIdx) : 0;
      if (grids.length) renderOpGrid(grids[idx]);
    });
  });
}
// --- Jacobian-lens tab rendering ---
//
// `renderJacobianLens` builds the Jacobian tab's content from the JSON
// `GET /api/models/{id}/jacobian-lens` returns: either
// `{"not_applicable": true, "reason": "..."}` for a non-Gpt model, or the
// script's `results.json` shape (see `analysis/jacobian_lens.py`'s `dump`)
// plus the server-added `plot_files` (filenames) and `computed_at` (ISO
// string) fields. Plots are rendered as <img> tags pointing at
// `GET /api/models/{id}/jacobian-lens/plot/{filename}`.
function renderJacobianLens(el, modelId, data) {
  if (!data) return;
  if (data.not_applicable) {
    el.innerHTML = `<div class="jac-not-applicable">${escapeHtml(data.reason || 'not applicable for this model type')}</div>`;
    el.classList.add('show');
    return;
  }
  const pct = (data.greedy_accuracy != null) ? (data.greedy_accuracy * 100).toFixed(1) + '%' : '—';
  const layerAcc = data.layer_lens_top1_accuracy || {};
  const layerKeys = Object.keys(layerAcc).map(Number).sort((a, b) => a - b);
  // Fine-grained checkpoint labeling: index 0 is embeddings; for block k
  // (0-indexed), index 2k+1 is that block's post-attention/pre-MLP
  // checkpoint and index 2k+2 is its full (post-MLP) output. Mirrors
  // `analysis/jacobian_lens.py`'s `layer_label` exactly, so the label shown
  // here always matches what the Python side printed/plotted.
  function jacLayerLabel(idx) {
    if (idx === 0) return 'embeddings';
    const k = Math.floor((idx - 1) / 2);
    const isAttnCheckpoint = (idx - 1) % 2 === 0;
    return isAttnCheckpoint ? `post-block${k}-attn` : `post-block${k}`;
  }
  const layerRow = layerKeys.map(l => {
    return `<tr><td>${jacLayerLabel(l)}</td><td>${(layerAcc[l] * 100).toFixed(1)}%</td></tr>`;
  }).join('');
  const computedNote = data.computed_at
    ? `<span class="jac-cache-note">computed ${escapeHtml(data.computed_at)}</span>`
    : '';
  const plots = data.plot_files || [];
  const plotsHtml = plots.length
    ? `<div class="jac-plots">${plots.map(f =>
        `<div class="jac-plot"><img loading="lazy" src="/api/models/${encodeURIComponent(modelId)}/jacobian-lens/plot/${encodeURIComponent(f)}" alt="${escapeHtml(f)}">` +
        `<div class="jac-plot-name">${escapeHtml(f)}</div></div>`
      ).join('')}</div>`
    : '<div class="jac-not-applicable">no plots recorded for this run</div>';
  el.innerHTML =
    `<div class="jac-summary">` +
      `<div class="readout-row"><span class="readout-big">${pct}</span><span style="color:var(--text-dim);font-size:0.8rem;">greedy exact-match accuracy over ${data.n_facts != null ? data.n_facts : '?'} facts (this reimplementation, cross-check)</span></div>` +
      `<div class="jac-summary-line">Layer-by-layer: when does the lens's top-1 prediction already equal the true answer digit? ${computedNote}</div>` +
      (layerRow ? `<table class="data-table"><tr><th>layer</th><th>lens top1 == true</th></tr>${layerRow}</table>` : '') +
    `</div>` +
    plotsHtml;
  el.classList.add('show');
}
// Fetches the Jacobian tab's data and renders it. `force` mirrors `loadGrid`:
// cache-first on a plain tab open (served instantly if a cached row exists,
// no Python subprocess), unconditional recompute for the "Run Jacobian lens
// analysis" button. This is genuinely slow (a whole Python/torch process
// plus per-layer Jacobian fitting) compared to everything else in this app,
// so the loading state says so explicitly rather than showing a bare
// spinner.
async function loadJacobianLens(panel, modelId, btn, force, silent) {
  const resultEl = panel.querySelector('.jac-result');
  const errEl = panel.querySelector('.jac-error-msg');
  const loadingEl = panel.querySelector('.jac-loading');
  errEl.textContent = '';
  const prevLabel = btn ? btn.textContent : null;
  let succeeded = false;
  if (btn && !silent) {
    btn.disabled = true;
    btn.innerHTML = (force ? 'Recomputing…' : 'Loading…') + '<span class="spinner"></span>';
  }
  if (loadingEl && !silent) {
    loadingEl.style.display = 'flex';
    loadingEl.textContent = 'Running Jacobian-lens analysis — this shells out to a Python/PyTorch ' +
      'process and fits per-layer Jacobians, so it can take anywhere from several seconds to a ' +
      'couple of minutes. Please wait…';
  }
  try {
    const url = '/api/models/' + encodeURIComponent(modelId) + '/jacobian-lens' + (force ? '?force=true' : '');
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) {
      errEl.textContent = data.error || ('HTTP ' + res.status);
      resultEl.classList.remove('show');
    } else {
      renderJacobianLens(resultEl, modelId, data);
      succeeded = !data.not_applicable;
    }
  } catch (e) {
    errEl.textContent = String(e);
  } finally {
    if (loadingEl) loadingEl.style.display = 'none';
    if (btn) {
      btn.disabled = false;
      btn.textContent = succeeded ? 'Recompute Jacobian lens' : prevLabel;
    }
  }
}
// --- Embeddings tab (layer-by-layer PCA/UMAP visualization) ---
//
// Deliberately reuses the SAME `GET /api/models/{id}/jacobian-lens` endpoint
// and cache row as the Jacobian tab (see `analysis/jacobian_lens.py`'s
// `compute_embedding_viz`, point 5 in its module doc) rather than a separate
// route/table -- the embedding-viz payload is computed in the same Python
// subprocess invocation, off the same per-fact forward passes already run
// for the Jacobian-lens fitting, so there's nothing to gain from a second
// cache/compute path. A model analyzed before this feature existed has a
// cached row with no `embedding_viz` key; `renderEmbeddingViz` below detects
// that and asks for a recompute rather than erroring.
//
// Deterministic per-label color for the Embeddings tab (scatter dots,
// legend swatches, and the radar chart's stroke/fill all route through
// this one function, so fixing it here fixes all three at once).
//
// The previous scheme hashed the label's char codes into a hue
// (`hash = (hash*31 + charCode) % 360`, then `hsl(hash, 62%, 50%)`). For
// the single-character labels this tab actually deals with (digits,
// '+', '=', '\n', ...) that hash is just the char code itself on the
// first iteration, and ASCII digits '0'-'9' are codes 48-57 -- so every
// digit landed within a ~10-degree hue band and was nearly
// indistinguishable from every other digit at fixed 62% saturation /
// 50% lightness. That was the actual complaint ("hard to see any
// difference"), not a dark/light-mode contrast problem.
//
// Fix: assign colors in FIRST-SEEN order (not by hashing the label
// text) using golden-angle hue stepping (137.508deg -- the angle that
// avoids any rational-fraction alignment, so hues never bunch up no
// matter how many distinct tokens there are), with saturation/lightness
// cycled across 3 bands so that even the occasional close-hue pair
// (unavoidable once there are many categories) still reads as visually
// distinct. Assignments are cached in `embColorRegistry` for the
// lifetime of the page, so a given token keeps the exact same color
// across the scatter, radar, and legend, and across every frame of the
// layer-slider animation. Bands stay mid-range (45-68% lightness,
// 65-88% saturation) so colors stay legible against both this page's
// dark (#0b0f0d) and light (#f5f4ee) backgrounds.
const embColorRegistry = new Map();
function embLabelColor(label) {
  const s = String(label);
  if (!embColorRegistry.has(s)) {
    const idx = embColorRegistry.size;
    const hue = (idx * 137.508) % 360;
    const band = idx % 3;
    const sat = [78, 65, 88][band];
    const light = [55, 68, 45][band];
    embColorRegistry.set(s, `hsl(${hue.toFixed(1)}, ${sat}%, ${light}%)`);
  }
  return embColorRegistry.get(s);
}
function embScaleFns(points) {
  const W = 460, H = 340, PAD = 26;
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = Math.max(maxX - minX, 1e-9), spanY = Math.max(maxY - minY, 1e-9);
  return {
    W, H,
    x: v => PAD + (v - minX) / spanX * (W - 2 * PAD),
    y: v => H - PAD - (v - minY) / spanY * (H - 2 * PAD),
  };
}
// WE-layer scatter: few enough distinct points (== vocab size) that every
// dot gets an inline text label right next to it, per the spec ("label each
// point with the actual character it represents").
function weScatterSvg(points, labels) {
  if (!points || !points.length) return '<svg class="emb-scatter" viewBox="0 0 460 340"></svg>';
  const { W, H, x, y } = embScaleFns(points);
  let dots = '';
  for (let i = 0; i < points.length; i++) {
    const [px, py] = points[i];
    const label = labels[i];
    const color = embLabelColor(label);
    const cx = x(px).toFixed(1), cy = y(py).toFixed(1);
    dots += `<circle cx="${cx}" cy="${cy}" r="6" fill="${color}" fill-opacity="0.88" stroke="${color}"><title>${escapeHtml(label)}</title></circle>` +
      `<text x="${(x(px) + 9).toFixed(1)}" y="${(y(py) - 8).toFixed(1)}" font-size="11" fill="currentColor">${escapeHtml(label)}</text>`;
  }
  return `<svg class="emb-scatter" viewBox="0 0 ${W} ${H}">${dots}</svg>`;
}
// Fixed duration (ms) for the layer-to-layer glide animation, both on a
// manual slider drag and on each Play auto-advance tick. 300-500ms reads as
// deliberate motion without feeling sluggish against `CKPT_PLAY_INTERVAL_MS`
// (300ms) -- picked 350ms so a Play tick's glide finishes just before (or
// right as) the next tick fires, rather than visibly overlapping/interrupting
// itself.
const EMB_GLIDE_MS = 350;
// Ease-in-out (quadratic) easing for the glide -- linear interpolation reads
// as slightly mechanical for a "points drifting through representation
// space" animation; this accelerates into and decelerates out of each frame.
function easeInOutQuad(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}
// Radar/spider chart of one token's RAW (un-projected, `hidden_size`-dim)
// vector at one layer -- a complement to the lossy 2D PCA/UMAP scatter,
// which only shows 2 of `hidden_size` dimensions' worth of variance.
//
// `maxAbs` MUST be the same fixed value across every call for a given model
// (computed once, server-side, over every layer and every point -- see
// `analysis/jacobian_lens.py`'s `raw_vector_max_abs`) -- never recomputed
// per layer/point here, or the axis would silently rescale between frames
// and make the shape appear to change for a reason that has nothing to do
// with the underlying values actually moving.
//
// Convention for negative values (residual-stream values aren't
// non-negative like a typical radar chart assumes): the full
// `[-maxAbs, +maxAbs]` range is mapped LINEARLY onto `[0, R]`, so a value of
// exactly 0 sits at radius `R/2` (the dashed reference circle), strongly
// negative values sit near the center, and strongly positive values sit
// near the outer edge. This is spelled out in the caption text returned
// alongside the chart, not just implied by the dashed circle.
function radarChartSvg(values, maxAbs, color) {
  const strokeColor = color || 'var(--accent)';
  const W = 300, H = 300, cx = W / 2, cy = H / 2, R = 108;
  const n = values.length;
  const denom = maxAbs > 0 ? maxAbs : 1;
  const pts = values.map((v, i) => {
    const angle = -Math.PI / 2 + i * (2 * Math.PI / n);
    const t = Math.max(-1, Math.min(1, v / denom));
    const r = R / 2 + t * (R / 2);
    return [cx + r * Math.cos(angle), cy + r * Math.sin(angle)];
  });
  const pathD = pts.map((p, i) => (i === 0 ? 'M' : 'L') + p[0].toFixed(2) + ' ' + p[1].toFixed(2)).join(' ') + ' Z';
  let axes = '', labels = '';
  for (let i = 0; i < n; i++) {
    const angle = -Math.PI / 2 + i * (2 * Math.PI / n);
    const x2 = cx + R * Math.cos(angle), y2 = cy + R * Math.sin(angle);
    axes += `<line x1="${cx}" y1="${cy}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" stroke="var(--border)" stroke-width="1"/>`;
    const lx = cx + (R + 16) * Math.cos(angle), ly = cy + (R + 16) * Math.sin(angle);
    labels += `<text x="${lx.toFixed(2)}" y="${ly.toFixed(2)}" font-size="8.5" text-anchor="middle" dominant-baseline="middle" fill="var(--text-faint)">d${i}</text>`;
  }
  const dots = pts.map(p => `<circle cx="${p[0].toFixed(2)}" cy="${p[1].toFixed(2)}" r="2.5" fill="${strokeColor}"/>`).join('');
  return `<svg class="emb-radar-svg" viewBox="0 0 ${W} ${H}">` +
    `<circle cx="${cx}" cy="${cy}" r="${R}" fill="none" stroke="var(--border)" stroke-width="1"/>` +
    `<circle cx="${cx}" cy="${cy}" r="${(R / 2).toFixed(2)}" fill="none" stroke="var(--text-faint)" stroke-width="1" stroke-dasharray="3,3"/>` +
    axes +
    `<path d="${pathD}" fill="${strokeColor}" fill-opacity="0.22" stroke="${strokeColor}" stroke-width="2"/>` +
    dots +
    labels +
    `</svg>`;
}
function embScatterLegend(labels) {
  const uniq = [...new Set(labels)].sort();
  return `<div class="emb-legend">${uniq.map(l =>
    `<span class="emb-legend-item"><span class="sw" style="background:${embLabelColor(l)}"></span>${escapeHtml(l)}</span>`
  ).join('')}</div>`;
}
// Builds the Embeddings tab's content: the WE (token-embedding) layer's
// scatter plot (PCA, and UMAP if the analysis environment had `umap-learn`),
// plus a layer-by-layer slider (mirroring `loadCheckpointGrids`'s
// slider+play-button convention) through every layer boundary's activation
// scatter across the fact corpus.
//
// The per-layer scatter animates smoothly between consecutive layers
// (`EMB_GLIDE_MS` glide, via `requestAnimationFrame`) rather than snapping,
// using the JSON's Procrustes-ALIGNED coordinates (`layer.pca`/`layer.umap`
// -- see `analysis/jacobian_lens.py`'s `align_layer_sequence`), never the
// raw independently-fit ones (`layer.pca_raw`/`layer.umap_raw`), since only
// the aligned sequence has a consistent frame to glide through. The scatter
// is built ONCE as a fixed set of `<circle>` DOM nodes (rather than
// rebuilt via `innerHTML` on every frame) so per-frame updates are just
// `cx`/`cy` attribute writes -- cheap, and it lets a clicked point keep its
// identity (and highlight) across every subsequent layer/mode change.
function renderEmbeddingViz(el, modelId, data) {
  if (!data) return;
  if (data.not_applicable) {
    el.innerHTML = `<div class="emb-not-applicable">${escapeHtml(data.reason || 'not applicable for this model type')}</div>`;
    el.classList.add('show');
    return;
  }
  const ev = data.embedding_viz;
  if (!ev) {
    el.innerHTML = `<div class="emb-not-applicable">This model's cached Jacobian-lens result predates the embedding-visualization feature — click "Recompute" on the Jacobian tab (or the button above) to add it.</div>`;
    el.classList.add('show');
    return;
  }
  const haveUmap = !!ev.have_umap;
  const haveTsne = !!ev.have_tsne;
  const missingNotes = [];
  if (!haveUmap) missingNotes.push('UMAP is not installed in the analysis environment (pip install umap-learn)');
  if (!haveTsne) missingNotes.push('t-SNE is not available (pip install scikit-learn)');
  const umapNote = missingNotes.length
    ? `<div class="emb-umap-note">${missingNotes.join('; ')} — showing ${missingNotes.length === 2 ? 'PCA only' : 'the remaining modes'}.</div>`
    : '';
  const umapBtn = (target) => haveUmap ? `<button class="grid-mode-btn" data-emb-target="${target}" data-mode="umap">UMAP</button>` : '';
  const tsneBtn = (target) => haveTsne ? `<button class="grid-mode-btn" data-emb-target="${target}" data-mode="tsne">t-SNE</button>` : '';
  // Force-directed ("spring") layout is computed entirely client-side (see
  // `forceDirectedLayout` below) -- no server-side dependency, so unlike
  // UMAP/t-SNE this button is never conditionally hidden; a missing
  // `raw_vectors` array (only possible for a cache computed before this
  // field existed) is handled at render time instead (empty plot + note),
  // the same graceful-degradation convention used elsewhere in this tab.
  const springBtn = (target) => `<button class="grid-mode-btn" data-emb-target="${target}" data-mode="spring">Spring</button>`;
  el.innerHTML =
    `<div class="emb-section">` +
      `<div class="emb-section-title">Token embeddings (WE layer)</div>` +
      `<div class="grid-mode-toggle"><button class="grid-mode-btn active" data-emb-target="we" data-mode="pca">PCA</button>${umapBtn('we')}${tsneBtn('we')}${springBtn('we')}</div>` +
      `<div class="emb-we-plot"></div>` +
    `</div>` +
    `<div class="emb-section">` +
      `<div class="emb-section-title">Per-layer activations across the fact corpus</div>` +
      `<div class="grid-mode-toggle"><button class="grid-mode-btn active" data-emb-target="layer" data-mode="pca">PCA</button>${umapBtn('layer')}${tsneBtn('layer')}${springBtn('layer')}` +
        `<button type="button" class="btn ghost emb-clear-select-btn" style="display:none;margin-left:8px;">Clear selection</button>` +
        `<span class="emb-selected-label"></span>` +
      `</div>` +
      `<div class="emb-pca-ref-row">` +
        `<label>PCA axes from: <select class="emb-pca-ref-select">` +
          `<option value="per-layer">per-layer (default, Procrustes-aligned for the glide)</option>` +
          ev.layers.map((L, i) => `<option value="${i}">fixed lens: ${escapeHtml(L.label)}</option>`).join('') +
          `</select></label>` +
        `<span class="emb-pca-ref-note"></span>` +
      `</div>` +
      `<div class="ckpt-controls">` +
        `<button type="button" class="btn ghost emb-play-btn">&#9654; Play</button>` +
        `<input type="range" class="ckpt-slider emb-slider" min="0" max="${ev.layers.length - 1}" value="0" step="1" />` +
        `<span class="ckpt-label emb-layer-label"></span>` +
      `</div>` +
      `<div class="emb-layer-plot"></div>` +
      `<div class="emb-layer-legend"></div>` +
      `<div class="emb-neighbors-panel" style="display:none;">` +
        `<div class="emb-neighbors-controls">nearest neighbors, raw hidden-dim space (Euclidean distance) — K = ` +
          `<input type="number" class="emb-k-input" min="1" max="12" value="3" /></div>` +
        `<ol class="emb-neighbors-list"></ol>` +
        `<div class="emb-neighbors-note"></div>` +
      `</div>` +
    `</div>` +
    `<div class="emb-radar-section" style="display:none;">` +
      `<div class="emb-section-title">Selected token vs. nearest neighbors (raw embedding radar)</div>` +
      `<div class="emb-radar-row">` +
        `<div class="emb-radar-item">` +
          `<div class="emb-radar-plot"></div>` +
          `<div class="emb-radar-self-label"></div>` +
        `</div>` +
        `<div class="emb-radar-sep"></div>` +
        `<div class="emb-neighbor-radars"></div>` +
      `</div>` +
      `<div class="emb-radar-note"></div>` +
      `<div class="emb-shape-note"></div>` +
    `</div>` +
    `<div class="emb-context-section" style="display:none;">` +
      `<div class="emb-section-title">Same sentence — tokens before it (causal context)</div>` +
      `<div class="emb-context-radars"></div>` +
      `<div class="emb-context-note"></div>` +
    `</div>` +
    umapNote;
  el.classList.add('show');

  const wePlotEl = el.querySelector('.emb-we-plot');
  const layerPlotEl = el.querySelector('.emb-layer-plot');
  const layerLegendEl = el.querySelector('.emb-layer-legend');
  const slider = el.querySelector('.emb-slider');
  const layerLabelEl = el.querySelector('.emb-layer-label');
  const playBtn = el.querySelector('.emb-play-btn');
  const clearSelectBtn = el.querySelector('.emb-clear-select-btn');
  const selectedLabelEl = el.querySelector('.emb-selected-label');
  const radarSectionEl = el.querySelector('.emb-radar-section');
  const radarPlotEl = el.querySelector('.emb-radar-plot');
  const radarSelfLabelEl = el.querySelector('.emb-radar-self-label');
  const radarSepEl = el.querySelector('.emb-radar-sep');
  const radarNoteEl = el.querySelector('.emb-radar-note');
  const shapeNoteEl = el.querySelector('.emb-shape-note');
  const neighborRadarsEl = el.querySelector('.emb-neighbor-radars');
  const neighborsPanelEl = el.querySelector('.emb-neighbors-panel');
  const neighborsListEl = el.querySelector('.emb-neighbors-list');
  const neighborsNoteEl = el.querySelector('.emb-neighbors-note');
  const contextSectionEl = el.querySelector('.emb-context-section');
  const contextRadarsEl = el.querySelector('.emb-context-radars');
  const contextNoteEl = el.querySelector('.emb-context-note');
  const kInput = el.querySelector('.emb-k-input');
  let weMode = 'pca', layerMode = 'pca', playTimer = null;
  let selectedIdx = null;
  let currentLayerIdx = 0;
  let currentScreenPoints = null; // last-rendered/settled [x,y] screen coords, per point index
  let currentNeighborIdxs = []; // point indices currently highlighted as raw-space nearest neighbors
  let currentNeighborDists = new Map(); // idx -> distance from selected token, reused by updateNeighborRadars (never recomputed)
  let neighborLineEls = []; // persistent <line> elements, selected -> each neighbor
  let animHandle = null;
  const scaleCache = {};
  const pcaRefSelect = el.querySelector('.emb-pca-ref-select');
  const pcaRefNoteEl = el.querySelector('.emb-pca-ref-note');
  // `null` = current default (each layer's own independently-fit,
  // Procrustes-aligned PCA). A number = "fixed lens" mode: every layer is
  // instead projected through THAT layer's PCA basis, so layers become
  // directly comparable (no alignment ambiguity to resolve -- there's only
  // ever one basis) at the cost of showing each OTHER layer's structure
  // through a possibly ill-fitting lens (that's the whole point: it answers
  // "does layer X's structure look like a rotated version of layer Y's
  // principal directions, or fundamentally different"). PCA-only -- neither
  // UMAP nor t-SNE nor the force-directed layout has a reusable fixed linear
  // transform (t-SNE and force-directed have no out-of-sample projection at
  // all), so this is ignored whenever `layerMode !== 'pca'` (see
  // `projectedPoints`) and the selector is disabled in the UI while any of
  // them is active (see the mode-toggle handler and
  // `updatePcaRefAvailability`).
  let pcaRefLayerIdx = null;
  const fixedBasisCache = {}; // refLayerIdx -> { pointsByLayer, scale }
  // Display name for a mode key ('pca'/'umap'/'tsne'/'spring') --
  // `tsne.toUpperCase()` would read as "TSNE", not the conventional "t-SNE"
  // hyphenated form, and 'spring' reads better spelled out.
  function modeDisplayName(mode) {
    if (mode === 'tsne') return 't-SNE';
    if (mode === 'spring') return 'Spring (force-directed)';
    return mode.toUpperCase();
  }

  // --- Force-directed ("spring") layout: the most literal version of
  // "points fight based on similarity" among the four modes -- builds an
  // explicit k-nearest-neighbor graph from TRUE raw-space (Euclidean, over
  // `raw_vectors`) distance, same distance function `updateNeighbors` uses
  // for true-neighbor highlighting, then simulates a classic
  // Fruchterman-Reingold layout: spring attraction along graph edges pulls
  // true neighbors together, universal inverse-distance repulsion between
  // EVERY pair keeps unrelated points apart (without it, attraction alone
  // would collapse everything to one point). Entirely client-side, no
  // server/Python dependency -- same spirit as the from-scratch power-
  // iteration PCA above, just for a nonlinear, similarity-driven layout
  // instead of a linear variance-maximizing one.
  //
  // K = 4 (edges per node): a common default for small force-directed graphs
  // -- enough to pull each point toward a handful of true neighbors without
  // nearly fully connecting these tiny (13-220 point) graphs, which would
  // wash out the attraction/repulsion contrast. Capped at n-1 for very small
  // layers (e.g. the WE layer can have as few as ~5-13 points).
  const SPRING_K = 4;
  const SPRING_ITERATIONS = 250;
  function knnEdges(vecs, k) {
    const n = vecs.length;
    const edges = new Set();
    const edgeList = [];
    for (let i = 0; i < n; i++) {
      const dists = [];
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        dists.push({ j, d: euclideanDist(vecs[i], vecs[j]) });
      }
      dists.sort((a, b) => a.d - b.d);
      for (let m = 0; m < Math.min(k, dists.length); m++) {
        const j = dists[m].j;
        const key = i < j ? `${i}:${j}` : `${j}:${i}`;
        if (!edges.has(key)) { edges.add(key); edgeList.push([Math.min(i, j), Math.max(i, j)]); }
      }
    }
    return edgeList;
  }
  // Classic Fruchterman-Reingold: attractive force along graph edges scales
  // as dist^2/ideal (spring pulling connected points toward `ideal`
  // separation), repulsive force between EVERY pair scales as ideal^2/dist
  // (Coulomb-style, falls off with distance but never vanishes) -- summed
  // per point, then each point moves a bounded step in the net force
  // direction, with the step bound ("temperature") cooling linearly to 0
  // over the run so the layout settles instead of oscillating forever.
  // Seeded from `seedXY` (that layer's own independent PCA fit, already
  // computed server-side) rather than random points purely for a calmer,
  // more legible start -- the simulation is free to move arbitrarily far
  // from that seed, so this is a starting guess, not a constraint.
  function forceDirectedLayout(vecs, seedXY, k, iterations) {
    const n = vecs.length;
    if (n < 2) return vecs.map(() => [0, 0]);
    const pos = seedXY.map(p => [p[0], p[1]]);
    const mean = meanVector(pos);
    for (let i = 0; i < n; i++) { pos[i][0] -= mean[0]; pos[i][1] -= mean[1]; }
    let sumSq = 0;
    for (let i = 0; i < n; i++) sumSq += pos[i][0] * pos[i][0] + pos[i][1] * pos[i][1];
    const std = Math.sqrt(sumSq / (2 * n)) || 1;
    for (let i = 0; i < n; i++) { pos[i][0] /= std; pos[i][1] /= std; }
    const edges = knnEdges(vecs, Math.min(k, n - 1));
    const ideal = Math.sqrt(1 / n) * 2;
    const t0 = 0.1;
    const disp = Array.from({ length: n }, () => [0, 0]);
    for (let it = 0; it < iterations; it++) {
      const temp = t0 * (1 - it / iterations);
      for (let i = 0; i < n; i++) { disp[i][0] = 0; disp[i][1] = 0; }
      // Repulsive: every pair, O(n^2) -- fine at this scale (<=~220 points).
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1];
          const dist = Math.max(Math.hypot(dx, dy), 1e-6);
          const force = (ideal * ideal) / dist;
          const ux = dx / dist, uy = dy / dist;
          disp[i][0] += ux * force; disp[i][1] += uy * force;
          disp[j][0] -= ux * force; disp[j][1] -= uy * force;
        }
      }
      // Attractive: only along the k-NN graph's edges.
      for (const [i, j] of edges) {
        const dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1];
        const dist = Math.max(Math.hypot(dx, dy), 1e-6);
        const force = (dist * dist) / ideal;
        const ux = dx / dist, uy = dy / dist;
        disp[i][0] -= ux * force; disp[i][1] -= uy * force;
        disp[j][0] += ux * force; disp[j][1] += uy * force;
      }
      for (let i = 0; i < n; i++) {
        const dlen = Math.hypot(disp[i][0], disp[i][1]);
        if (dlen < 1e-12) continue;
        const step = Math.min(dlen, temp) / dlen;
        pos[i][0] += disp[i][0] * step;
        pos[i][1] += disp[i][1] * step;
      }
    }
    return pos;
  }
  // Closed-form 2x2 orthogonal Procrustes (rotation OR reflection, whichever
  // maximizes fit) -- equivalent to `numpy.linalg.svd(M); u @ vt` used
  // server-side (`analysis/jacobian_lens.py`'s `procrustes_align`), but
  // derived directly without a generic SVD: for a 2x2 matrix M, the
  // trace-maximizing orthogonal R is either a pure rotation
  // `[[cosT,-sinT],[sinT,cosT]]` (T = atan2(M10-M01, M00+M11)) or a
  // reflection-rotation `[[c,s],[s,-c]]` (from M00-M11, M01+M10) --
  // whichever of the two achieves the larger `trace(R^T M)`; this is exactly
  // what the SVD-based `u @ vt` also picks (verified numerically against
  // `numpy.linalg.svd` while prototyping this in Python before porting).
  // Needed here (rather than reusing a library) because this alignment must
  // run client-side: the force-directed layout itself is computed in the
  // browser, so there is no server-side pass that could align it instead.
  function procrustesAlign2d(source, target) {
    const n = source.length;
    const srcMean = meanVector(source), tgtMean = meanVector(target);
    const srcC = source.map(p => [p[0] - srcMean[0], p[1] - srcMean[1]]);
    const tgtC = target.map(p => [p[0] - tgtMean[0], p[1] - tgtMean[1]]);
    let m00 = 0, m01 = 0, m10 = 0, m11 = 0;
    for (let i = 0; i < n; i++) {
      m00 += srcC[i][0] * tgtC[i][0]; m01 += srcC[i][0] * tgtC[i][1];
      m10 += srcC[i][1] * tgtC[i][0]; m11 += srcC[i][1] * tgtC[i][1];
    }
    const a = m00 + m11, b = m10 - m01;
    const rotScore = Math.hypot(a, b);
    const theta = Math.atan2(b, a);
    const p = m00 - m11, q = m01 + m10;
    const refScore = Math.hypot(p, q);
    let R;
    if (rotScore >= refScore) {
      const ct = Math.cos(theta), st = Math.sin(theta);
      R = [[ct, -st], [st, ct]];
    } else {
      const denom = refScore > 1e-15 ? refScore : 1;
      const c = p / denom, s = q / denom;
      R = [[c, s], [s, -c]];
    }
    return srcC.map(v => [
      v[0] * R[0][0] + v[1] * R[0][1] + tgtMean[0],
      v[0] * R[1][0] + v[1] * R[1][1] + tgtMean[1],
    ]);
  }
  // Chains `procrustesAlign2d` across the layer sequence -- identical
  // running-frame convention to `align_layer_sequence` server-side (layer 1
  // aligned to layer 0 as-is, layer 2 aligned to layer 1's ALREADY-aligned
  // frame, and so on) so the glide animation moves through one consistent
  // frame instead of partly showing arbitrary per-layer orientation
  // differences.
  function alignLayerSequenceJs(coordsByLayer) {
    const aligned = [];
    let prev = null;
    for (const coords of coordsByLayer) {
      if (!coords) { aligned.push(null); continue; }
      if (prev === null) { aligned.push(coords); } else { aligned.push(procrustesAlign2d(coords, prev)); }
      prev = aligned[aligned.length - 1];
    }
    return aligned;
  }
  let springCache = null; // { pointsByLayer, scale } -- computed once, lazily (see getSpringData)
  let weSpringCache = null; // [x,y][] for the WE-layer scatter -- single static layout, no alignment needed
  // Computes (and caches) the force-directed layout for EVERY per-layer
  // checkpoint, then Procrustes-aligns the sequence exactly like PCA/UMAP/
  // t-SNE (see `alignLayerSequenceJs`'s doc). Lazy + cached because this is
  // the heaviest client-side computation in this tab (O(n^2) repulsion x
  // `SPRING_ITERATIONS` x every layer) -- only paid once, the first time a
  // user actually switches to Spring mode, not on every tab load.
  function getSpringData() {
    if (springCache) return springCache;
    const rawByLayer = ev.layers.map(L => {
      if (!L.raw_vectors) return null;
      return forceDirectedLayout(L.raw_vectors, L.pca_raw, SPRING_K, SPRING_ITERATIONS);
    });
    const aligned = alignLayerSequenceJs(rawByLayer);
    const all = [];
    aligned.forEach(p => { if (p) all.push(...p); });
    const scale = embScaleFns(all.length ? all : [[0, 0], [1, 1]]);
    springCache = { pointsByLayer: aligned, scale };
    return springCache;
  }
  // WE-layer spring layout: a single static scatter (no layer-to-layer
  // glide to align across), so just one force-directed fit, no Procrustes
  // chain needed.
  function getWeSpringData() {
    if (weSpringCache) return weSpringCache;
    if (!ev.we_layer.raw_vectors) return null;
    weSpringCache = forceDirectedLayout(ev.we_layer.raw_vectors, ev.we_layer.pca, SPRING_K, SPRING_ITERATIONS);
    return weSpringCache;
  }

  // --- Minimal from-scratch PCA (top-2 components via power iteration +
  // deflation) for the "fixed lens" mode. `raw_vectors` is only ~13-220
  // points x 16 dims per layer, so this is trivially cheap client-side --
  // no library needed. Power iteration converges to the dominant
  // eigenvector of a symmetric matrix regardless of the random starting
  // vector (given enough iterations and no exact eigenvalue tie); after
  // finding it, the matrix is deflated (the found component's contribution
  // subtracted out) and the process repeats for the 2nd component.
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function vnorm(v) { return Math.sqrt(dot(v, v)) || 1; }
  function normalize(v) { const n = vnorm(v); return v.map(x => x / n); }
  function matVecMul(m, v) { return m.map(row => dot(row, v)); }
  function meanVector(vecs) {
    const d = vecs[0].length;
    const mean = new Array(d).fill(0);
    for (const v of vecs) for (let i = 0; i < d; i++) mean[i] += v[i];
    for (let i = 0; i < d; i++) mean[i] /= vecs.length;
    return mean;
  }
  function covarianceMatrix(centered) {
    const n = centered.length, d = centered[0].length;
    const m = Array.from({ length: d }, () => new Array(d).fill(0));
    for (const row of centered) {
      for (let i = 0; i < d; i++) {
        const ri = row[i];
        if (ri === 0) continue;
        for (let j = 0; j < d; j++) m[i][j] += ri * row[j];
      }
    }
    const denom = Math.max(1, n - 1);
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) m[i][j] /= denom;
    return m;
  }
  function powerIteration(m, iterations) {
    const d = m.length;
    let v = normalize(new Array(d).fill(0).map((_, i) => Math.sin(i + 1.3))); // deterministic seed, not Math.random(), so re-renders are reproducible
    for (let it = 0; it < iterations; it++) v = normalize(matVecMul(m, v));
    const value = dot(v, matVecMul(m, v));
    return { vector: v, value };
  }
  function top2Eigenvectors(m) {
    const d = m.length;
    const { vector: v1, value: lambda1 } = powerIteration(m, 400);
    const deflated = m.map((row, i) => row.map((val, j) => val - lambda1 * v1[i] * v1[j]));
    const { vector: v2 } = powerIteration(deflated, 400);
    return [v1, v2];
  }
  // Builds (and caches) the fixed 2D basis fit on `refLayerIdx`'s raw
  // vectors, then projects EVERY layer's raw vectors through that same
  // basis + that same reference mean (so every layer's coordinates are
  // directly comparable -- no independent re-centering per layer).
  function getFixedBasisData(refLayerIdx) {
    if (fixedBasisCache[refLayerIdx]) return fixedBasisCache[refLayerIdx];
    const refVecs = ev.layers[refLayerIdx].raw_vectors;
    if (!refVecs) return null;
    const mean = meanVector(refVecs);
    const centered = refVecs.map(v => v.map((x, i) => x - mean[i]));
    const cov = covarianceMatrix(centered);
    const [v1, v2] = top2Eigenvectors(cov);
    const pointsByLayer = ev.layers.map(L => {
      if (!L.raw_vectors) return null;
      return L.raw_vectors.map(v => {
        const c = v.map((x, i) => x - mean[i]);
        return [dot(c, v1), dot(c, v2)];
      });
    });
    const all = [];
    pointsByLayer.forEach(p => { if (p) all.push(...p); });
    const scale = embScaleFns(all.length ? all : [[0, 0], [1, 1]]);
    const data = { pointsByLayer, scale };
    fixedBasisCache[refLayerIdx] = data;
    return data;
  }
  function updatePcaRefAvailability() {
    pcaRefSelect.disabled = (layerMode !== 'pca');
    pcaRefNoteEl.textContent = (layerMode !== 'pca')
      ? `(fixed-lens PCA axes don't apply to ${modeDisplayName(layerMode)} -- showing ${modeDisplayName(layerMode)}'s own per-layer/aligned fit)`
      : (pcaRefLayerIdx != null ? '(every layer projected through this one fixed basis)' : '');
  }
  pcaRefSelect.addEventListener('change', () => {
    pcaRefLayerIdx = pcaRefSelect.value === 'per-layer' ? null : Number(pcaRefSelect.value);
    updatePcaRefAvailability();
    // A basis change is a change of REFERENCE FRAME, not a layer transition
    // -- snap instantly rather than gliding (there's nothing to glide FROM
    // in the new frame's terms).
    renderLayer(currentLayerIdx, false);
  });

  // Euclidean distance between two raw hidden-dim vectors. Chosen over
  // cosine because these are pre-LayerNorm residual-stream/embedding
  // vectors where magnitude is part of the signal (LayerNorm inside each
  // block already normalizes scale internally; the raw checkpoints we
  // capture are BEFORE that), so treating two vectors that point the same
  // direction but differ hugely in magnitude as "close" (as cosine would)
  // would hide a real difference. Explicitly labeled as Euclidean in the
  // panel's caption so this choice is never ambiguous to the viewer.
  function euclideanDist(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
    return Math.sqrt(s);
  }
  // Nearest `k` OTHER points to `idx` by raw-vector Euclidean distance, at
  // `layerIdx`. Every individual (fact, position) point is its own
  // candidate (not deduplicated by token identity) -- this is comparing
  // actual instances in this layer, the same population the scatter plots.
  function computeNeighbors(idx, layerIdx, k) {
    const vecs = ev.layers[layerIdx].raw_vectors;
    if (!vecs) return [];
    const target = vecs[idx];
    const dists = [];
    for (let i = 0; i < vecs.length; i++) {
      if (i === idx) continue;
      dists.push({ i, d: euclideanDist(target, vecs[i]) });
    }
    dists.sort((a, b) => a.d - b.d);
    return dists.slice(0, Math.max(1, k));
  }
  // Rebuilds the connecting-line elements for the CURRENT neighbor set
  // (called once per selection/layer/K change, not per animation frame) and
  // immediately positions them from whatever's on screen right now;
  // `setPositions` keeps them tracking the (possibly still-animating)
  // circles every frame afterward.
  function drawNeighborLines(neighborIdxs) {
    neighborLineEls.forEach(l => l.remove());
    neighborLineEls = neighborIdxs.map(ni => {
      const line = document.createElementNS(NS, 'line');
      line.setAttribute('stroke', '#3b82f6');
      line.setAttribute('stroke-width', '1.5');
      line.setAttribute('stroke-dasharray', '4,3');
      line.setAttribute('stroke-opacity', '0.85');
      line.dataset.neighborIdx = String(ni);
      svgEl.insertBefore(line, svgEl.firstChild); // behind every circle
      return line;
    });
    updateNeighborLinePositions(currentScreenPoints);
  }
  function updateNeighborLinePositions(points) {
    if (!points || !points.length || selectedIdx == null) return;
    const p0 = points[selectedIdx];
    if (!p0) return;
    neighborLineEls.forEach(line => {
      const ni = Number(line.dataset.neighborIdx);
      const p1 = points[ni];
      if (!p1) return;
      line.setAttribute('x1', p0[0]);
      line.setAttribute('y1', p0[1]);
      line.setAttribute('x2', p1[0]);
      line.setAttribute('y2', p1[1]);
    });
  }
  // Recomputes the nearest-neighbor set for the current selection at
  // `layerIdx` (called on selection change, K change, AND every layer
  // change -- true-nearest-neighbor structure can shift across layers just
  // like the scatter/radar do) and, as a cheap complement, checks whether
  // the CURRENTLY DISPLAYED 2D projection's apparent-closest point actually
  // matches the true raw-space nearest neighbor -- the direct way to show
  // that PCA/UMAP proximity can be misleading.
  function updateNeighbors(layerIdx) {
    if (selectedIdx == null) {
      neighborsPanelEl.style.display = 'none';
      currentNeighborIdxs = [];
      currentNeighborDists = new Map();
      drawNeighborLines([]);
      return;
    }
    neighborsPanelEl.style.display = '';
    const L = ev.layers[layerIdx];
    if (!L.raw_vectors) {
      neighborsListEl.innerHTML = '';
      neighborsNoteEl.textContent = 'This cached result predates the nearest-neighbor feature — recompute to add raw vectors.';
      currentNeighborIdxs = [];
      currentNeighborDists = new Map();
      drawNeighborLines([]);
      return;
    }
    const k = Math.max(1, Math.min(L.raw_vectors.length - 1, Number(kInput.value) || 3));
    const neighbors = computeNeighbors(selectedIdx, layerIdx, k);
    currentNeighborIdxs = neighbors.map(n => n.i);
    currentNeighborDists = new Map(neighbors.map(n => [n.i, n.d]));
    const ordinal = (n) => (n === 1 ? '1st' : n === 2 ? '2nd' : n === 3 ? '3rd' : `${n}th`);
    neighborsListEl.innerHTML = neighbors.map((n, rank) =>
      `<li>${ordinal(rank + 1)} nearest: "${escapeHtml(pointLabels[n.i])}" — dist ${n.d.toFixed(3)}</li>`
    ).join('');

    // Cheap complement: does the currently-DISPLAYED 2D projection's
    // nearest point agree with the true raw-space nearest neighbor? Uses
    // whichever 2D coordinates are actually on screen right now -- the
    // fixed-lens basis when that mode is active, otherwise the normal
    // per-layer/aligned PCA or UMAP.
    let proj2d;
    if (layerMode === 'pca' && pcaRefLayerIdx != null) {
      const fixedData = getFixedBasisData(pcaRefLayerIdx);
      proj2d = fixedData ? fixedData.pointsByLayer[layerIdx] : null;
    } else if (layerMode === 'spring') {
      const springData = getSpringData();
      proj2d = springData.pointsByLayer[layerIdx];
    } else {
      proj2d = L[layerMode];
    }
    let note = '';
    if (proj2d && neighbors.length) {
      const t2 = proj2d[selectedIdx];
      let best2dIdx = -1, best2dDist = Infinity;
      for (let i = 0; i < proj2d.length; i++) {
        if (i === selectedIdx) continue;
        const dx = proj2d[i][0] - t2[0], dy = proj2d[i][1] - t2[1];
        const d = Math.hypot(dx, dy);
        if (d < best2dDist) { best2dDist = d; best2dIdx = i; }
      }
      const modeLabel = (layerMode === 'pca' && pcaRefLayerIdx != null)
        ? `PCA (fixed lens from ${escapeHtml(ev.layers[pcaRefLayerIdx].label)})`
        : modeDisplayName(layerMode);
      note = (best2dIdx === neighbors[0].i)
        ? `${modeLabel}'s closest-looking point here also IS the true raw-space nearest neighbor.`
        : `${modeLabel} makes "${escapeHtml(pointLabels[best2dIdx])}" look closest, but the true raw-space ` +
          `nearest neighbor is "${escapeHtml(pointLabels[neighbors[0].i])}" — the 2D projection is misleading here.`;
    }
    neighborsNoteEl.textContent = note;
    drawNeighborLines(currentNeighborIdxs);
  }
  kInput.addEventListener('input', () => {
    updateNeighbors(currentLayerIdx);
    updateNeighborRadars(currentLayerIdx);
    applySelectionStyles();
  });
  // Fixed, model-wide radar scale (see `radarChartSvg`'s doc for why this
  // must never be recomputed per layer/point) -- falls back to 1 for a
  // cached result computed before this field existed, so an old cache
  // doesn't crash the radar chart, just shows an under-scaled one until
  // recomputed.
  const radarMaxAbs = ev.raw_vector_max_abs != null ? ev.raw_vector_max_abs : 1;

  // Renders (or hides) the radar chart for whatever's currently selected, at
  // `layerIdx`. Called on selection change AND on every layer change, so the
  // shape always reflects "the selected token, at the layer currently shown
  // on the scatter" -- called with the FINAL target layer index immediately
  // (not per intermediate glide frame): the scatter's position tween is a
  // visual nicety, but the radar chart jumping straight to the target
  // layer's true values is more important than animating it too.
  function updateRadar(layerIdx) {
    if (selectedIdx == null) {
      radarSectionEl.style.display = 'none';
      return;
    }
    radarSectionEl.style.display = '';
    const L = ev.layers[layerIdx];
    const vec = L.raw_vectors ? L.raw_vectors[selectedIdx] : null;
    if (!vec) {
      radarPlotEl.innerHTML = '<div class="emb-not-applicable">This cached result predates the radar-chart feature — recompute to add raw vectors.</div>';
      radarNoteEl.textContent = '';
      shapeNoteEl.textContent = '';
      return;
    }
    radarPlotEl.innerHTML = radarChartSvg(vec, radarMaxAbs, embLabelColor(pointLabels[selectedIdx]));
    radarSelfLabelEl.textContent = `"${pointLabels[selectedIdx]}" (selected)`;
    radarNoteEl.textContent =
      `token "${pointLabels[selectedIdx]}" at ${L.label} · ${vec.length} raw dims · ` +
      `dashed circle = value 0 · outer edge = +${radarMaxAbs.toFixed(2)} · center = -${radarMaxAbs.toFixed(2)} ` +
      `(fixed scale, same across every layer)`;
    // Compact shape-summary readout: PCA-reconstruction-error curve for this
    // token's cloud at this layer (0% = perfectly point/line/plane-like at
    // that rank). Absent for cached results computed before this feature, or
    // for tokens with <2 occurrences at this layer -- shown as an empty note
    // in both cases rather than an error.
    const errs = L.shape_fit_errors ? L.shape_fit_errors[pointLabels[selectedIdx]] : null;
    shapeNoteEl.textContent = (errs && errs.length > 2)
      ? `"${pointLabels[selectedIdx]}" shape fit: line (k=1) error ${(errs[1] * 100).toFixed(1)}%, plane (k=2) error ${(errs[2] * 100).toFixed(1)}%`
      : '';
  }

  // Small comparison-radar row for the selected token's CURRENT nearest
  // neighbors (set + distances come from `updateNeighbors`, which must run
  // first each call site -- never recomputed here). Reuses the same fixed
  // `radarMaxAbs` scale as the main radar so a neighbor's shape is directly
  // comparable, not rescaled to its own range.
  function updateNeighborRadars(layerIdx) {
    if (selectedIdx == null || !currentNeighborIdxs.length) {
      neighborRadarsEl.innerHTML = '';
      radarSepEl.style.display = 'none';
      return;
    }
    const L = ev.layers[layerIdx];
    if (!L.raw_vectors) {
      neighborRadarsEl.innerHTML = '';
      radarSepEl.style.display = 'none';
      return;
    }
    radarSepEl.style.display = '';
    neighborRadarsEl.innerHTML = currentNeighborIdxs.map(ni => {
      const vec = L.raw_vectors[ni];
      if (!vec) return '';
      const label = pointLabels[ni];
      const dist = currentNeighborDists.get(ni);
      const distText = dist != null ? `dist ${dist.toFixed(3)}` : '';
      return `<div class="emb-neighbor-radar-item">` +
        `<div class="emb-radar-plot">${radarChartSvg(vec, radarMaxAbs, embLabelColor(label))}</div>` +
        `<div class="emb-neighbor-radar-label">"${escapeHtml(label)}"</div>` +
        `<div class="emb-neighbor-radar-dist">${distText}</div>` +
      `</div>`;
    }).join('');
  }

  // Radar row for the tokens that come BEFORE the selected one in its own
  // sentence -- its full causal/attention-visible context under this
  // decoder-only model's mask. Unlike `updateNeighborRadars` (nearest by
  // distance, whole corpus), this is a fixed set determined purely by
  // sentence structure, so it doesn't depend on K or on `updateNeighbors`
  // having run. Gracefully no-ops on older cached results that predate
  // `point_fact_idx`/`point_pos_in_fact`.
  function updateContextRadars(layerIdx) {
    if (selectedIdx == null || !ev.point_fact_idx || !ev.point_pos_in_fact) {
      contextSectionEl.style.display = 'none';
      contextRadarsEl.innerHTML = '';
      contextNoteEl.textContent = '';
      return;
    }
    const L = ev.layers[layerIdx];
    if (!L.raw_vectors) {
      contextSectionEl.style.display = 'none';
      contextRadarsEl.innerHTML = '';
      contextNoteEl.textContent = '';
      return;
    }
    contextSectionEl.style.display = '';
    const factIdx = ev.point_fact_idx[selectedIdx];
    const posInFact = ev.point_pos_in_fact[selectedIdx];
    const priorIdxs = [];
    for (let i = 0; i < ev.point_fact_idx.length; i++) {
      if (ev.point_fact_idx[i] === factIdx && ev.point_pos_in_fact[i] < posInFact) {
        priorIdxs.push(i);
      }
    }
    priorIdxs.sort((a, b) => ev.point_pos_in_fact[a] - ev.point_pos_in_fact[b]);
    if (posInFact === 0 || priorIdxs.length === 0) {
      contextRadarsEl.innerHTML = '';
      contextNoteEl.textContent = '(first token in its sentence — no preceding context)';
      return;
    }
    contextNoteEl.textContent = ev.point_prompts ? `sentence: "${escapeHtml(ev.point_prompts[selectedIdx])}"` : '';
    contextRadarsEl.innerHTML = priorIdxs.map(pi => {
      const vec = L.raw_vectors[pi];
      if (!vec) return '';
      const label = pointLabels[pi];
      return `<div class="emb-context-radar-item">` +
        `<div class="emb-radar-plot">${radarChartSvg(vec, radarMaxAbs, embLabelColor(label))}</div>` +
        `<div class="emb-context-radar-label">"${escapeHtml(label)}"</div>` +
        `<div class="emb-context-radar-pos">pos ${ev.point_pos_in_fact[pi]}</div>` +
      `</div>`;
    }).join('');
  }

  function renderWe() {
    if (weMode === 'spring') {
      const pts = getWeSpringData();
      wePlotEl.innerHTML = pts
        ? weScatterSvg(pts, ev.we_layer.labels)
        : '<div class="emb-not-applicable">This cached result predates the force-directed-layout feature (no WE-layer raw vectors) — recompute to add it.</div>';
      return;
    }
    const pts = ev.we_layer[weMode];
    wePlotEl.innerHTML = weScatterSvg(pts, ev.we_layer.labels);
  }
  renderWe();

  // A single consistent screen-space scale per mode, computed once from
  // EVERY layer's points combined -- so gliding between layers only moves
  // points, it never also silently rescales/reframes the plot (which would
  // look like extra motion that isn't really there). Spring mode gets its
  // scale from `getSpringData` instead (computed alongside its own
  // Procrustes-aligned coordinates), since those aren't part of the server
  // JSON's per-layer objects the way pca/umap/tsne are.
  function getScale(mode) {
    if (scaleCache[mode]) return scaleCache[mode];
    let s;
    if (mode === 'spring') {
      s = getSpringData().scale;
    } else {
      const all = [];
      for (const L of ev.layers) {
        const pts = L[mode];
        if (pts) all.push(...pts);
      }
      s = embScaleFns(all.length ? all : [[0, 0], [1, 1]]);
    }
    scaleCache[mode] = s;
    return s;
  }
  // Linear scale factors implied by `getScale(mode)`'s affine x/y functions
  // (derivative of `v => PAD + (v-min)/span*(size-2*PAD)`), used to carry a
  // data-space ellipse radius into screen space. `sy` is `abs()`'d since the
  // y-function's scale is negative (SVG y grows downward). This ignores the
  // (usually small) distortion from x/y having different scale factors
  // interacting with a rotated ellipse -- an accepted approximation, not
  // worth a full affine-transform-of-the-covariance-matrix treatment for a
  // supplementary visual aid.
  function getEllipseScale(mode) {
    const { x, y } = getScale(mode);
    return { x, y, sx: x(1) - x(0), sy: Math.abs(y(1) - y(0)) };
  }
  // Per-token covariance ellipse (`{cx,cy,rx,ry,angle}` in screen space) for
  // the given layer/mode, or `null` when there's nothing to draw: "fixed
  // lens" and "spring" modes have no matching ellipse data (they're not the
  // aligned per-layer projection the server fit ellipses against), and older
  // cached results simply lack the `*_ellipses` fields.
  function projectedEllipses(idx, mode) {
    if (mode === 'pca' && pcaRefLayerIdx != null) return null;
    if (mode === 'spring') return null;
    const L = ev.layers[idx];
    const ellipses = L[mode + '_ellipses'];
    if (!ellipses) return null;
    const { x, y, sx, sy } = getEllipseScale(mode);
    const out = {};
    for (const label in ellipses) {
      const e = ellipses[label];
      out[label] = { cx: x(e.cx), cy: y(e.cy), rx: e.rx * sx, ry: e.ry * sy, angle: e.angle_deg };
    }
    return out;
  }
  function projectedPoints(idx, mode) {
    // "Fixed lens" mode: PCA-only (see `pcaRefLayerIdx`'s doc) -- every
    // layer projected through ONE reference layer's basis, so there's no
    // per-layer independent fit and therefore nothing to Procrustes-align;
    // the fixed-basis data already has its own model-wide scale.
    if (mode === 'pca' && pcaRefLayerIdx != null) {
      const data = getFixedBasisData(pcaRefLayerIdx);
      if (!data) return null;
      const raw2d = data.pointsByLayer[idx];
      if (!raw2d) return null;
      const { x, y } = data.scale;
      return raw2d.map(p => [x(p[0]), y(p[1])]);
    }
    if (mode === 'spring') {
      const data = getSpringData();
      const raw2d = data.pointsByLayer[idx];
      if (!raw2d) return null;
      const { x, y } = data.scale;
      return raw2d.map(p => [x(p[0]), y(p[1])]);
    }
    const L = ev.layers[idx];
    const raw = L[mode];
    if (!raw) return null;
    const { x, y } = getScale(mode);
    return raw.map(p => [x(p[0]), y(p[1])]);
  }

  // Build the persistent SVG + one <circle> per point, ONCE (point count is
  // identical at every layer -- same ordered list of (fact, position)
  // pairs throughout, per `compute_embedding_viz`'s doc).
  const pointLabels = ev.layers[0].point_labels;
  const { W, H } = getScale(layerMode);
  layerPlotEl.innerHTML = `<svg class="emb-scatter" viewBox="0 0 ${W} ${H}"></svg>`;
  const svgEl = layerPlotEl.querySelector('svg');
  const NS = 'http://www.w3.org/2000/svg';

  // One persistent <ellipse> per distinct token label, appended BEFORE the
  // scatter dots below so they render behind them (z-order = DOM order).
  // Colored to match that token's dots/legend/radar via `embLabelColor`;
  // low-opacity fill + a visible stroke so overlapping tokens' ellipses
  // stay distinguishable. Hidden (via `display:none`) rather than removed
  // when a given layer/mode has no matching ellipse data, so toggling modes
  // doesn't need to rebuild these nodes.
  const ellipseEls = new Map();
  [...new Set(pointLabels)].forEach(label => {
    const e = document.createElementNS(NS, 'ellipse');
    e.style.pointerEvents = 'none';
    const color = embLabelColor(label);
    e.setAttribute('fill', color);
    e.setAttribute('fill-opacity', '0.15');
    e.setAttribute('stroke', color);
    e.setAttribute('stroke-width', '1.5');
    e.setAttribute('stroke-opacity', '0.55');
    e.setAttribute('display', 'none');
    svgEl.appendChild(e);
    ellipseEls.set(label, e);
  });
  // Ellipses snap straight to each target frame's values rather than
  // gliding, unlike the scatter dots below -- simpler and lower-risk, and
  // the dots' own tween already conveys the layer-to-layer motion.
  function setEllipsePositions(ellipses) {
    ellipseEls.forEach((e, label) => {
      const t = ellipses ? ellipses[label] : null;
      if (!t) { e.setAttribute('display', 'none'); return; }
      e.removeAttribute('display');
      e.setAttribute('cx', t.cx.toFixed(2));
      e.setAttribute('cy', t.cy.toFixed(2));
      e.setAttribute('rx', Math.max(0, t.rx).toFixed(2));
      e.setAttribute('ry', Math.max(0, t.ry).toFixed(2));
      e.setAttribute('transform', `rotate(${t.angle.toFixed(2)} ${t.cx.toFixed(2)} ${t.cy.toFixed(2)})`);
    });
  }

  const circles = pointLabels.map((label, i) => {
    const c = document.createElementNS(NS, 'circle');
    c.dataset.idx = String(i);
    c.dataset.label = label;
    c.style.cursor = 'pointer';
    const title = document.createElementNS(NS, 'title');
    title.textContent = label; // native hover tooltip, shown regardless of selection
    c.appendChild(title);
    c.addEventListener('click', () => {
      selectedIdx = (selectedIdx === i) ? null : i;
      updateSelectedLabel();
      updateRadar(currentLayerIdx);
      // Must run BEFORE applySelectionStyles: it sets `currentNeighborIdxs`,
      // which the styling pass below reads to ring the new neighbor set.
      updateNeighbors(currentLayerIdx);
      updateNeighborRadars(currentLayerIdx);
      updateContextRadars(currentLayerIdx);
      applySelectionStyles();
    });
    svgEl.appendChild(c);
    return c;
  });

  function updateSelectedLabel() {
    if (selectedIdx == null) {
      selectedLabelEl.textContent = '';
      clearSelectBtn.style.display = 'none';
    } else {
      selectedLabelEl.textContent = `tracking: "${pointLabels[selectedIdx]}"`;
      clearSelectBtn.style.display = '';
    }
  }
  clearSelectBtn.addEventListener('click', () => {
    selectedIdx = null;
    updateSelectedLabel();
    updateRadar(currentLayerIdx);
    updateNeighbors(currentLayerIdx);
    updateNeighborRadars(currentLayerIdx); // clears the panel since currentNeighborIdxs is now empty
    updateContextRadars(currentLayerIdx);
    applySelectionStyles();
  });

  // Distinct visual treatment for the selected point (larger radius +
  // highlight ring) AND for its current raw-space nearest neighbors (a
  // different ring color/style from both the selection and the dimmed
  // background points), applied/reapplied on every position update so both
  // persist across layer changes, mode toggles, and mid-glide frames.
  function applySelectionStyles() {
    const neighborSet = new Set(currentNeighborIdxs);
    circles.forEach((c, i) => {
      const color = embLabelColor(c.dataset.label);
      if (i === selectedIdx) {
        c.setAttribute('r', '8.5');
        c.setAttribute('fill', color);
        c.setAttribute('fill-opacity', '0.95');
        c.setAttribute('stroke', 'var(--text)');
        c.setAttribute('stroke-width', '2.5');
      } else if (neighborSet.has(i)) {
        c.setAttribute('r', '6.5');
        c.setAttribute('fill', color);
        c.setAttribute('fill-opacity', '0.9');
        c.setAttribute('stroke', '#3b82f6'); // distinct blue ring, same color as the connecting lines
        c.setAttribute('stroke-width', '2');
      } else {
        c.setAttribute('r', '4.5');
        c.setAttribute('fill', color);
        c.setAttribute('fill-opacity', selectedIdx == null ? '0.7' : '0.3');
        c.setAttribute('stroke', color);
        c.setAttribute('stroke-width', '0.5');
      }
    });
  }
  function setPositions(points) {
    circles.forEach((c, i) => {
      c.setAttribute('cx', points[i][0].toFixed(2));
      c.setAttribute('cy', points[i][1].toFixed(2));
    });
    currentScreenPoints = points;
    updateNeighborLinePositions(points);
  }
  applySelectionStyles();

  // `animate=false` (initial load, mode toggle) snaps straight to the
  // target layer; `animate=true` (slider drag, Play tick) glides from
  // whatever is currently on screen to the target over `EMB_GLIDE_MS`.
  function renderLayer(idx, animate) {
    const L = ev.layers[idx];
    currentLayerIdx = idx;
    layerLabelEl.textContent = `${L.label} (layer ${idx + 1}/${ev.layers.length})`;
    layerLegendEl.innerHTML = embScatterLegend(L.point_labels);
    updateRadar(idx);
    updateNeighbors(idx);
    updateNeighborRadars(idx); // depends on currentNeighborIdxs, which updateNeighbors just refreshed for this layer
    updateContextRadars(idx);
    // Re-apply selection/neighbor ring styling (r/fill/stroke -- NOT
    // cx/cy) right away, even when the position tween below is about to
    // run: `updateNeighbors` may have just changed the neighbor set for
    // the new target layer, and unlike position, that styling doesn't
    // animate -- it must be correct for the whole glide, not just once it
    // finishes.
    applySelectionStyles();
    setEllipsePositions(projectedEllipses(idx, layerMode));
    const target = projectedPoints(idx, layerMode);
    if (!target) {
      // e.g. UMAP unavailable for this layer -- nothing sane to draw or
      // glide to; leave the previous frame's points in place.
      return;
    }
    if (animHandle) { cancelAnimationFrame(animHandle); animHandle = null; }
    if (!animate || !currentScreenPoints) {
      setPositions(target);
      applySelectionStyles();
      return;
    }
    const start = currentScreenPoints;
    const startTime = performance.now();
    function step(now) {
      const t = Math.min(1, (now - startTime) / EMB_GLIDE_MS);
      const eased = easeInOutQuad(t);
      const interp = start.map((p, i) => [
        p[0] + (target[i][0] - p[0]) * eased,
        p[1] + (target[i][1] - p[1]) * eased,
      ]);
      setPositions(interp);
      if (t < 1) {
        animHandle = requestAnimationFrame(step);
      } else {
        animHandle = null;
        setPositions(target);
      }
    }
    animHandle = requestAnimationFrame(step);
  }
  updatePcaRefAvailability();
  renderLayer(Number(slider.value), false);

  el.querySelectorAll('[data-emb-target]').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.embTarget;
      el.querySelectorAll(`[data-emb-target="${target}"]`).forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (target === 'we') { weMode = btn.dataset.mode; renderWe(); }
      else {
        layerMode = btn.dataset.mode;
        updatePcaRefAvailability();
        renderLayer(Number(slider.value), false);
      }
    });
  });

  function stopPlaying() {
    if (playTimer) {
      clearInterval(playTimer);
      playTimer = null;
      playBtn.innerHTML = '&#9654; Play';
    }
  }
  slider.addEventListener('input', () => {
    stopPlaying();
    renderLayer(Number(slider.value), true);
  });
  playBtn.addEventListener('click', () => {
    if (playTimer) { stopPlaying(); return; }
    playBtn.innerHTML = '&#10074;&#10074; Pause';
    playTimer = setInterval(() => {
      let next = Number(slider.value) + 1;
      if (next > Number(slider.max)) next = 0;
      slider.value = String(next);
      renderLayer(next, true);
    }, CKPT_PLAY_INTERVAL_MS);
  });
}
// Fetches the Embeddings tab's data. Hits the exact same
// `/api/models/{id}/jacobian-lens` endpoint as `loadJacobianLens` (same
// cache row) -- `force` and the loading-state UX mirror that function
// exactly, just rendering into the Embeddings tab's own elements.
async function loadEmbeddingViz(panel, modelId, btn, force, silent) {
  const resultEl = panel.querySelector('.emb-result');
  const errEl = panel.querySelector('.emb-error-msg');
  const loadingEl = panel.querySelector('.emb-loading');
  errEl.textContent = '';
  const prevLabel = btn ? btn.textContent : null;
  let succeeded = false;
  if (btn && !silent) {
    btn.disabled = true;
    btn.innerHTML = (force ? 'Recomputing…' : 'Loading…') + '<span class="spinner"></span>';
  }
  if (loadingEl && !silent) {
    loadingEl.style.display = 'flex';
    loadingEl.textContent = 'Running the Jacobian-lens analysis process (this also extracts the ' +
      'per-layer activations used here) — this shells out to Python/PyTorch and can take from ' +
      'several seconds to a couple of minutes. Please wait…';
  }
  try {
    const url = '/api/models/' + encodeURIComponent(modelId) + '/jacobian-lens' + (force ? '?force=true' : '');
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) {
      errEl.textContent = data.error || ('HTTP ' + res.status);
      resultEl.classList.remove('show');
    } else {
      renderEmbeddingViz(resultEl, modelId, data);
      succeeded = !data.not_applicable;
    }
  } catch (e) {
    errEl.textContent = String(e);
  } finally {
    if (loadingEl) loadingEl.style.display = 'none';
    if (btn) {
      btn.disabled = false;
      btn.textContent = succeeded ? 'Recompute embedding visualization' : prevLabel;
    }
  }
}
// Fetches the Grid tab's data and renders it. `force` selects between the
// cache-first default (a plain tab open — served instantly from the
// `eval_grids` DB cache if fresh, no model load) and an unconditional
// recompute (the "Recompute grid" button). `silent` suppresses the
// button-spinner treatment for the automatic cache-hit load triggered by
// `renderDetailFor`, since that path is expected to resolve near-instantly
// and flashing a spinner for a sub-frame would just be visual noise.
async function loadGrid(panel, modelId, btn, force, silent) {
  const resultEl = panel.querySelector('.grid-result');
  const errEl = panel.querySelector('.grid-error-msg');
  errEl.textContent = '';
  // Remember the label to fall back to on failure — a failed request (e.g.
  // the range-too-large rejection) never writes a cache row, so the button
  // must NOT unconditionally flip to "Recompute grid" afterward (that would
  // falsely imply a cached grid now exists).
  const prevLabel = btn ? btn.textContent : null;
  let succeeded = false;
  if (btn && !silent) {
    btn.disabled = true;
    btn.innerHTML = (force ? 'Recomputing…' : 'Loading…') + '<span class="spinner"></span>';
  }
  try {
    const url = '/api/models/' + encodeURIComponent(modelId) + '/eval-grid' + (force ? '?force=true' : '');
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) {
      errEl.innerHTML = `<div class="grid-too-large">${escapeHtml(data.error || ('HTTP ' + res.status))}</div>`;
      resultEl.classList.remove('show');
    } else {
      renderGrid(resultEl, data);
      succeeded = true;
    }
  } catch (e) {
    errEl.textContent = String(e);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = succeeded ? 'Recompute grid' : prevLabel;
    }
  }
}
// Fixed interval (ms) between animation frames for the checkpoint-grid
// slider's Play button. 300ms is fast enough to read as an animation (over
// 100 snapshots plays through in ~30s) while still slow enough to actually
// register each frame's cells changing, rather than a blur — tunable by
// editing this one constant.
const CKPT_PLAY_INTERVAL_MS = 300;
// Fetches a model's `checkpoint_grids` history (exhaustive-eval-grid
// snapshots captured over the course of training) and, if any exist, builds
// the slider + play/pause animation UI inside the Grid tab's `.ckpt-panel`.
// The slider is indexed by SNAPSHOT POSITION (0..n-1), not by epoch: with up
// to 100+ snapshots clustered densely early in training and sparsely near a
// late-training plateau (per the task brief), an epoch-indexed slider would
// waste most of its range on the flat tail. Position-indexing keeps every
// frame equally reachable; the epoch/loss/accuracy label under the slider
// always names the actual epoch and loss for the current frame, so the
// non-uniform time steps between frames are never ambiguous.
//
// Reuses `renderGrid` — the exact same pass/fail/gradient renderer the plain
// Grid tab already uses — to draw whichever snapshot the slider lands on;
// this function only decides WHICH report object to hand it.
async function loadCheckpointGrids(panel, modelId) {
  const ckptPanel = panel.querySelector('.ckpt-panel');
  if (!ckptPanel) return;
  ckptPanel.innerHTML = '<div class="ckpt-loading">loading training-progress history…</div>';
  let snapshots;
  try {
    const res = await fetch('/api/models/' + encodeURIComponent(modelId) + '/checkpoint-grids');
    if (!res.ok) { ckptPanel.innerHTML = ''; return; }
    snapshots = await res.json();
  } catch (e) {
    ckptPanel.innerHTML = '';
    return;
  }
  if (!Array.isArray(snapshots) || snapshots.length === 0) {
    ckptPanel.innerHTML = '<div class="ckpt-empty">no training-progress history recorded for this model</div>';
    return;
  }

  ckptPanel.innerHTML =
    `<div class="ckpt-header">Training-progress animation <span class="ckpt-count">(${snapshots.length} snapshot${snapshots.length === 1 ? '' : 's'})</span></div>` +
    `<div class="ckpt-controls">` +
      `<button type="button" class="btn ghost ckpt-play-btn">&#9654; Play</button>` +
      `<input type="range" class="ckpt-slider" min="0" max="${snapshots.length - 1}" value="${snapshots.length - 1}" step="1" />` +
      `<span class="ckpt-label"></span>` +
    `</div>` +
    `<div class="ckpt-grid-result"></div>`;

  const slider = ckptPanel.querySelector('.ckpt-slider');
  const label = ckptPanel.querySelector('.ckpt-label');
  const playBtn = ckptPanel.querySelector('.ckpt-play-btn');
  const gridResultEl = ckptPanel.querySelector('.ckpt-grid-result');
  let playTimer = null;

  function showFrame(idx) {
    const snap = snapshots[idx];
    const pct = snap.total > 0 ? (snap.correct / snap.total * 100) : 0;
    label.textContent =
      `epoch ${snap.epoch} · loss ${snap.loss.toFixed(4)} · ${pct.toFixed(1)}% (${snap.correct}/${snap.total}) ` +
      `· snapshot ${idx + 1}/${snapshots.length}`;
    // Preserve the passfail/gradient mode toggle across frames: renderGrid
    // always rebuilds its own DOM (including the mode toggle) starting from
    // "passfail", so remember what was active before this call and, if it
    // was "gradient", re-click that toggle after re-rendering.
    const prevModeBtn = gridResultEl.querySelector('.grid-mode-btn.active');
    const prevMode = prevModeBtn ? prevModeBtn.dataset.mode : 'passfail';
    renderGrid(gridResultEl, snap.report);
    if (prevMode === 'gradient') {
      const gradBtn = gridResultEl.querySelector('.grid-mode-btn[data-mode="gradient"]');
      if (gradBtn) gradBtn.click();
    }
  }

  function stopPlaying() {
    if (playTimer) {
      clearInterval(playTimer);
      playTimer = null;
      playBtn.innerHTML = '&#9654; Play';
    }
  }

  slider.addEventListener('input', () => {
    stopPlaying();
    showFrame(Number(slider.value));
  });

  playBtn.addEventListener('click', () => {
    if (playTimer) {
      stopPlaying();
      return;
    }
    playBtn.innerHTML = '&#10074;&#10074; Pause';
    playTimer = setInterval(() => {
      let next = Number(slider.value) + 1;
      if (next > Number(slider.max)) next = 0; // loop back to the start
      slider.value = String(next);
      showFrame(next);
    }, CKPT_PLAY_INTERVAL_MS);
  });

  showFrame(snapshots.length - 1);
}
// Builds the readout markup (top-right of a row header): the headline eval
// percentage if a cached eval exists, otherwise a muted "not evaluated" note.
function renderReadout(m) {
  if (!m.cached_eval) {
    return '<div class="row-readout"><span class="rd-empty">not evaluated</span></div>';
  }
  const c = m.cached_eval;
  const pct = c.total > 0 ? (c.correct / c.total * 100) : 0;
  return `<div class="row-readout"><span class="rd-num ${pctTextClass(pct)}">${pct.toFixed(1)}%</span><span class="rd-frac">${c.correct}/${c.total} correct</span></div>`;
}
// Builds the three tab panels (overview / training / eval) for a model's
// expanded body. Kept as one function so `swapRowToVariant` and
// `renderModelRow` build identical markup for base and variant views.
function renderTabPanels(m) {
  let overview = '';
  if (m.status === 'missing') {
    overview += '<div class="banner red">checkpoint file missing on disk — the registry entry points at a .bin file that no longer exists.</div>';
  }
  if (m.block_size != null) {
    overview += `<dl class="kv-grid">
      <div class="kv"><dt>block</dt><dd>${m.block_size}</dd></div>
      <div class="kv"><dt>hidden</dt><dd>${m.hidden_size}</dd></div>
      <div class="kv"><dt>heads</dt><dd>${m.num_heads}</dd></div>
      <div class="kv"><dt>layers</dt><dd>${m.num_blocks}</dd></div>
    </dl>`;
  }
  // "Expected" size is the raw-weights estimate (params × 4 bytes for f32) —
  // the actual file is always somewhat larger due to the storage format's
  // per-tensor metadata (name/shape/dtype header), an overhead that's a
  // bigger fraction of the file the smaller the model is. Showing both side
  // by side makes that gap visible instead of just showing one number.
  const expectedBytes = m.params_estimate != null ? m.params_estimate * 4 : null;
  overview += `<div class="params-line"><b>${fmtBytes(m.file_size_bytes)}</b> on disk (expected ${fmtBytes(expectedBytes)} from ${fmtParams(m.params_estimate)} params × 4 bytes) · <span class="tag">${escapeHtml(m.model_type||'?')}</span> <span class="tag">${escapeHtml(m.tokenizer||'?')} tokenizer</span></div>`;
  if (m.dataset_name) {
    overview += `<div class="dataset-block"><span class="dataset-toggle">▸ dataset: ${escapeHtml(m.dataset_name)}</span>`;
    if (m.dataset_info) {
      overview += `<div class="dataset-head">${m.dataset_info.line_count} lines · ${m.dataset_info.byte_size} bytes\n${m.dataset_info.head.map(escapeHtml).join('\n')}</div>`;
    }
    overview += '</div>';
  }
  if (m.note) {
    overview += `<div class="note-line">${escapeHtml(m.note)}</div>`;
  }

  const training = renderTraining(m);
  const samplesTab = renderSamples(m);

  let evalTab = '';
  let gridTab = '';
  let jacobianTab = '';
  let embeddingsTab = '';
  const isGpt = (m.model_type || '') === 'gpt';
  if (m.status === 'ok' || m.status === 'mismatch') {
    evalTab = `<div class="eval-toolbar"><button class="btn primary eval-btn">Run eval</button><button class="btn ghost repl-jump-btn">Open in REPL</button></div>` +
      `<div class="eval-result"></div><div class="error-msg"></div>`;
    const gridBtnLabel = m.cached_grid ? 'Recompute grid' : 'Run exhaustive grid';
    const gridCacheNote = m.cached_grid
      ? `<span class="grid-cache-note">showing cached result from a previous run over [${m.cached_grid.eval_min},${m.cached_grid.eval_max}]</span>`
      : '';
    gridTab = `<div class="grid-toolbar"><button class="btn primary grid-btn">${gridBtnLabel}</button>${gridCacheNote}</div>` +
      `<div class="grid-result"></div><div class="grid-error-msg"></div>` +
      `<div class="ckpt-panel"></div>`;
    if (isGpt) {
      const jacBtnLabel = m.jacobian_lens ? 'Recompute Jacobian lens' : 'Run Jacobian lens analysis';
      const jacCacheNote = m.jacobian_lens ? '<span class="jac-cache-note">showing cached result from a previous run</span>' : '';
      jacobianTab = `<div class="grid-toolbar"><button class="btn primary jac-btn">${jacBtnLabel}</button>${jacCacheNote}</div>` +
        `<div class="jac-loading" style="display:none;"></div>` +
        `<div class="jac-result"></div><div class="grid-error-msg jac-error-msg"></div>`;
      // Embeddings tab shares the Jacobian-lens cache/endpoint (see
      // `loadEmbeddingViz`'s doc) -- same cache-note/button-label logic as
      // the Jacobian tab above, since a cache existing means both tabs have
      // data to show.
      const embBtnLabel = m.jacobian_lens ? 'Recompute embedding visualization' : 'Run embedding visualization';
      const embCacheNote = m.jacobian_lens ? '<span class="jac-cache-note">showing cached result from a previous run</span>' : '';
      embeddingsTab = `<div class="grid-toolbar"><button class="btn primary emb-btn">${embBtnLabel}</button>${embCacheNote}</div>` +
        `<div class="jac-loading emb-loading" style="display:none;"></div>` +
        `<div class="emb-result"></div><div class="grid-error-msg emb-error-msg"></div>`;
    } else {
      jacobianTab = `<div class="jac-not-applicable">Jacobian lens is only applicable to Gpt-type models (this is '${escapeHtml(m.model_type || '?')}') — it needs real transformer layers to lens through.</div>`;
      embeddingsTab = `<div class="emb-not-applicable">Layer-by-layer embedding visualization is only applicable to Gpt-type models (this is '${escapeHtml(m.model_type || '?')}') — it needs real transformer layers to visualize.</div>`;
    }
  } else {
    evalTab = '<div class="train-empty">model checkpoint unavailable — cannot run eval.</div>';
    gridTab = '<div class="train-empty">model checkpoint unavailable — cannot run eval.</div>';
    jacobianTab = '<div class="train-empty">model checkpoint unavailable — cannot run jacobian-lens analysis.</div>';
    embeddingsTab = '<div class="train-empty">model checkpoint unavailable — cannot run embedding visualization.</div>';
  }

  return `
    <div class="tabs">
      <button class="tab-btn active" data-tab="overview">Overview</button>
      <button class="tab-btn" data-tab="training">Training</button>
      <button class="tab-btn" data-tab="samples">Samples</button>
      <button class="tab-btn" data-tab="eval">Eval</button>
      <button class="tab-btn" data-tab="grid">Grid</button>
      <button class="tab-btn" data-tab="jacobian">Jacobian</button>
      <button class="tab-btn" data-tab="embeddings">Embeddings</button>
    </div>
    <div class="tab-panel active" data-panel="overview">${overview}</div>
    <div class="tab-panel" data-panel="training">${training}</div>
    <div class="tab-panel" data-panel="samples">${samplesTab}</div>
    <div class="tab-panel" data-panel="eval">${evalTab}</div>
    <div class="tab-panel" data-panel="grid">${gridTab}</div>
    <div class="tab-panel" data-panel="jacobian">${jacobianTab}</div>
    <div class="tab-panel" data-panel="embeddings">${embeddingsTab}</div>
  `;
}
// Wires dataset-toggle, tab buttons, and the eval/grid/REPL-jump buttons
// inside a freshly rendered detail-panel body. `panel` is the element that
// contains both the tab bar and the tab panels — it also doubles as the
// container `runEval`/`runGrid` search within for `.eval-result` /
// `.grid-result` / the error nodes. `activeTab` (module-level state) is
// updated on every tab click so a subsequent select change can restore it.
function wireDetailBody(panel, modelId) {
  const dsToggle = panel.querySelector('.dataset-toggle');
  if (dsToggle) {
    dsToggle.addEventListener('click', () => {
      const head = dsToggle.nextElementSibling;
      if (head) head.classList.toggle('open');
    });
  }
  const tabBtns = panel.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      panel.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const tp = panel.querySelector(`.tab-panel[data-panel="${btn.dataset.tab}"]`);
      if (tp) tp.classList.add('active');
      activeTab = btn.dataset.tab;
    });
  });
  const evalBtn = panel.querySelector('.eval-btn');
  if (evalBtn) {
    evalBtn.addEventListener('click', () => runEval(panel, modelId, evalBtn));
  }
  const gridBtn = panel.querySelector('.grid-btn');
  if (gridBtn) {
    // The visible button is always the "force a fresh recompute" action —
    // the cache-first "just show me what's there" load happens automatically
    // (see `renderDetailFor`) when a cached grid is known to exist.
    gridBtn.addEventListener('click', () => loadGrid(panel, modelId, gridBtn, true, false));
  }
  const jacBtn = panel.querySelector('.jac-btn');
  if (jacBtn) {
    // Same convention as gridBtn: the visible button always forces a fresh
    // recompute; the cache-first "just show me what's there" load happens
    // automatically in `renderDetailFor` when a cache is known to exist.
    jacBtn.addEventListener('click', () => loadJacobianLens(panel, modelId, jacBtn, true, false));
  }
  const embBtn = panel.querySelector('.emb-btn');
  if (embBtn) {
    embBtn.addEventListener('click', () => loadEmbeddingViz(panel, modelId, embBtn, true, false));
  }
  // "Open in REPL": select this model in the REPL <select>, fire the
  // model-change handler so any model-dependent REPL UI updates, then
  // smooth-scroll the REPL panel into view and focus the prompt box.
  const replJumpBtn = panel.querySelector('.repl-jump-btn');
  if (replJumpBtn) {
    replJumpBtn.addEventListener('click', () => {
      const sel = $('repl-model');
      let found = false;
      for (const opt of sel.options) {
        if (opt.value === modelId) { found = true; break; }
      }
      if (found) {
        sel.value = modelId;
        if (typeof onReplModelChange === 'function') onReplModelChange();
      }
      const replPanel = document.getElementById('repl-panel');
      if (replPanel) replPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      const prompt = $('repl-prompt');
      if (prompt) prompt.focus();
    });
  }
}
// `renderDetailFor` is the single place that (re-)paints the detail panel for
// whichever model+variant id is passed in — called on initial load and again
// on every base-select / variant-select change. It rebuilds the tab panels
// from scratch (so stale data from a previously selected model/variant can
// never leak through) and then restores `activeTab` (module-level state) so
// switching either select keeps the user on the tab they were looking at,
// falling back to Overview only if that tab doesn't exist on this model
// (e.g. variants without a Training row still have all four tabs, so this
// fallback is mostly theoretical, but it mirrors the eval/grid-unavailable
// case for a missing-checkpoint model).
function renderDetailFor(modelId) {
  const m = findModelById(allModels, modelId);
  const panelEl = $('detail-panel');
  if (!m) {
    panelEl.innerHTML = '<div class="repl-placeholder" style="padding:16px">Model not found.</div>';
    return;
  }
  panelEl.dataset.status = m.status;
  panelEl.innerHTML = `
    <div class="detail-head">
      <span class="row-id">${escapeHtml(m.id)}</span>
      ${renderReadout(m)}
    </div>
    <div class="detail-body"></div>
  `;
  const body = panelEl.querySelector('.detail-body');
  body.innerHTML = renderTabPanels(m);
  wireDetailBody(body, m.id);
  if (m.cached_eval) {
    renderEval(body.querySelector('.eval-result'), m.cached_eval);
  }
  // Auto-load the cached grid (if the `/api/models` metadata says one
  // exists) so switching to the Grid tab shows it immediately with no click
  // needed — this is what makes "reopen the Grid tab" instant. Only fires
  // when a cache is known to exist (per `m.cached_grid`), so a model that's
  // never been grid-evaluated doesn't spend a wasted request finding that
  // out on every tab/model switch. `force=false` hits the cheap DB-cache
  // read path server-side (see `eval_grid_model`), not a recompute.
  if (m.cached_grid) {
    loadGrid(body, m.id, body.querySelector('.grid-btn'), false, true);
  }
  // Independently of the plain grid cache above, fetch this model's
  // training-progress checkpoint-grid history (if any) so the Grid tab's
  // slider/animation is ready without an extra click. Only attempted when
  // the Grid tab actually exists (status ok/mismatch put a `.ckpt-panel`
  // placeholder in the markup above; the missing-checkpoint case doesn't).
  if (body.querySelector('.ckpt-panel')) {
    loadCheckpointGrids(body, m.id);
  }
  // Auto-load the cached Jacobian-lens result (if `/api/models` metadata says
  // one exists) so switching to the Jacobian tab shows it immediately, same
  // "reopen is instant" behavior as the Grid tab's cache. Only fires when a
  // cache is known to exist (per `m.jacobian_lens`); a model that's never
  // been analyzed doesn't spend a wasted request finding that out on every
  // tab/model switch.
  if (m.jacobian_lens && body.querySelector('.jac-btn')) {
    loadJacobianLens(body, m.id, body.querySelector('.jac-btn'), false, true);
  }
  // Same auto-load for the Embeddings tab -- it shares the Jacobian-lens
  // cache/endpoint, so `m.jacobian_lens` being present means this tab has
  // data to show too (assuming the cached run is recent enough to include
  // `embedding_viz`; `renderEmbeddingViz` handles the older-cache case).
  if (m.jacobian_lens && body.querySelector('.emb-btn')) {
    loadEmbeddingViz(body, m.id, body.querySelector('.emb-btn'), false, true);
  }
  const targetBtn = body.querySelector(`.tab-btn[data-tab="${activeTab}"]`);
  const targetPanel = body.querySelector(`.tab-panel[data-panel="${activeTab}"]`);
  if (targetBtn && targetPanel) {
    body.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    body.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    targetBtn.classList.add('active');
    targetPanel.classList.add('active');
  } else {
    activeTab = 'overview';
  }
}
// Populates the base-model <select> with one option per top-level model
// (id + a short param-count descriptor). Does not touch the variant select —
// callers repopulate that separately via `populateVariantSelect`.
function populateBaseSelect() {
  const sel = $('base-select');
  sel.innerHTML = '';
  for (const m of allModels) {
    const o = document.createElement('option');
    o.value = m.id;
    o.textContent = `${m.id} · ${fmtBytes(m.file_size_bytes)}`;
    sel.appendChild(o);
  }
}
// Populates the variant <select> for the currently chosen base model: "base"
// first (always the default), then one option per RL variant labeled with
// its kind and round count. Called whenever the base select changes.
function populateVariantSelect(base) {
  const sel = $('variant-select');
  sel.innerHTML = '';
  const baseOpt = document.createElement('option');
  baseOpt.value = base.id;
  baseOpt.textContent = 'base';
  sel.appendChild(baseOpt);
  for (const v of (base.variants || [])) {
    const kind = (v.training && v.training.kind) || (v.id.includes('-rft') ? 'rft' : (v.id.includes('-grpo') ? 'grpo' : 'variant'));
    const rounds = (v.training && v.training.epochs_run) ? ` (${v.training.epochs_run} rounds)` : '';
    const o = document.createElement('option');
    o.value = v.id;
    o.textContent = `${kind}${rounds}`;
    sel.appendChild(o);
  }
  sel.value = base.id;
}
// Base-select change: repopulate the variant select for the newly chosen
// base (always resetting to "base" as the default per the spec) and repaint
// the detail panel — `activeTab` is preserved by `renderDetailFor`.
function onBaseSelectChange() {
  const base = allModels.find(m => m.id === $('base-select').value);
  if (!base) return;
  populateVariantSelect(base);
  renderDetailFor(base.id);
}
// Variant-select change: just repaint the detail panel for the chosen
// variant (or the base itself) — `activeTab` is preserved by
// `renderDetailFor`.
function onVariantSelectChange() {
  const id = $('variant-select').value;
  if (id) renderDetailFor(id);
}
async function load() {
  try {
    const res = await fetch('/api/models');
    const models = await res.json();
    allModels = models;
    const summary = document.getElementById('summary-bar');
    const variantCount = models.reduce((n, m) => n + (m.variants ? m.variants.length : 0), 0);
    if (summary) {
      summary.innerHTML = models.length
        ? `<b>${models.length}</b> model${models.length===1?'':'s'}` +
          (variantCount ? ` · <b>${variantCount}</b> variant${variantCount===1?'':'s'}` : '')
        : '';
    }
    if (!models.length) {
      $('base-select').innerHTML = '<option value="">(no models)</option>';
      $('variant-select').innerHTML = '<option value="">—</option>';
      $('detail-panel').innerHTML = '<div class="repl-placeholder" style="padding:16px">No models found. Train a model with --train to auto-register it in smolgpt.db.</div>';
    } else {
      populateBaseSelect();
      const first = models[0];
      populateVariantSelect(first);
      renderDetailFor(first.id);
      $('base-select').addEventListener('change', onBaseSelectChange);
      $('variant-select').addEventListener('change', onVariantSelectChange);
    }
    // Populate the REPL model <select> with both base models and their RL
    // variants, so the user can REPL against any variant directly.
    const flat = [];
    for (const m of models) {
      flat.push(m);
      if (m.variants) for (const v of m.variants) flat.push(v);
    }
    populateReplModels(flat);
  } catch (e) {
    $('detail-panel').innerHTML = '<div class="banner red">Failed to load: ' + escapeHtml(String(e)) + '</div>';
  }
}

// --- REPL / Playground ---
let replModels = [];
function $(id) { return document.getElementById(id); }
function tokenChipColor(id) {
  // Deterministic hue from token id, rendered as a soft border tint so the
  // dark theme is preserved while still visually separating distinct tokens.
  const hue = (id * 47) % 360;
  return { border: `hsl(${hue}, 55%, 45%)`, bg: `hsla(${hue}, 55%, 45%, 0.14)` };
}
function renderTokChips(el, chips) {
  el.innerHTML = '';
  if (!chips || !chips.length) {
    el.innerHTML = '<span class="repl-empty">(no tokens)</span>';
    return;
  }
  for (const t of chips) {
    const c = document.createElement('span');
    c.className = 'tok-chip';
    const { border, bg } = tokenChipColor(t.id);
    c.style.borderColor = border;
    c.style.background = bg;
    const strEl = document.createElement('span');
    strEl.className = 'tok-str';
    const s = t.str == null ? '' : String(t.str);
    if (s === '' || /\s/.test(s)) strEl.classList.add('ws');
    // Use a visible glyph for whitespace-only tokens so they aren't invisible.
    let display = s;
    if (s === '\n') display = '↵';
    else if (s === ' ') display = '·';
    else if (s === '\t') display = '⇥';
    else if (/^\s+$/.test(s)) display = s.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '⇥');
    strEl.textContent = display === '' ? '∅' : display;
    strEl.title = JSON.stringify(s);
    const idEl = document.createElement('span');
    idEl.className = 'tok-id';
    idEl.textContent = String(t.id);
    c.appendChild(strEl);
    c.appendChild(idEl);
    el.appendChild(c);
  }
}
function setReplBusy(busy) {
  const tok = $('repl-tokenize-btn');
  const gen = $('repl-generate-btn');
  tok.disabled = busy;
  gen.disabled = busy;
  gen.innerHTML = busy ? 'Generating…<span class="spinner"></span>' : 'Generate';
  tok.innerHTML = busy ? 'Tokenizing…<span class="spinner"></span>' : 'Tokenize';
}
function setReplError(msg) { $('repl-error').textContent = msg || ''; }
function hideReplEmptyState() {
  const empty = $('repl-output-empty');
  if (empty) empty.style.display = 'none';
}
function populateReplModels(models) {
  replModels = models || [];
  const sel = $('repl-model');
  sel.innerHTML = '';
  const loadable = replModels.filter(m => m.status === 'ok' || m.status === 'mismatch');
  if (!loadable.length) {
    sel.innerHTML = '<option value="">(no loadable models)</option>';
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  for (const m of loadable) {
    const o = document.createElement('option');
    o.value = m.id;
    o.textContent = m.id + ' · ' + (m.model_type || '?') + '/' + (m.tokenizer || '?');
    sel.appendChild(o);
  }
  onReplModelChange();
}
function onReplModelChange() {
  const id = $('repl-model').value;
  const m = replModels.find(x => x.id === id);
  const max = $('repl-max');
  if (m && m.block_size != null) {
    max.value = String(m.block_size);
    max.max = String(m.block_size);
  } else {
    max.value = '16';
    max.max = '';
  }
}
async function replTokenize() {
  const modelId = $('repl-model').value;
  const text = $('repl-prompt').value;
  if (!modelId) { setReplError('Pick a model first.'); return; }
  setReplError('');
  setReplBusy(true);
  try {
    const res = await fetch('/api/repl/tokenize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId, text })
    });
    const data = await res.json();
    if (!res.ok) { setReplError(data.error || ('HTTP ' + res.status)); return; }
    hideReplEmptyState();
    renderTokChips($('repl-tok-chips'), data.tokens);
    $('repl-tok-meta').textContent =
      (data.tokens ? data.tokens.length : 0) + ' tokens · vocab ' + (data.vocab_size ?? '?') +
      ' · ' + (data.tokenizer_type || '?') + ' tokenizer';
    $('repl-tokenize-section').style.display = 'block';
  } catch (e) {
    setReplError(String(e));
  } finally {
    setReplBusy(false);
  }
}
async function replGenerate() {
  const modelId = $('repl-model').value;
  const prompt = $('repl-prompt').value;
  if (!modelId) { setReplError('Pick a model first.'); return; }
  const maxNew = parseInt($('repl-max').value, 10) || 16;
  const temp = parseFloat($('repl-temp').value);
  const greedy = $('repl-greedy').checked;
  setReplError('');
  setReplBusy(true);
  try {
    const res = await fetch('/api/repl/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_id: modelId,
        prompt,
        max_new_tokens: maxNew,
        temperature: isNaN(temp) ? 0.0 : temp,
        greedy
      })
    });
    const data = await res.json();
    if (!res.ok) { setReplError(data.error || ('HTTP ' + res.status)); return; }
    hideReplEmptyState();
    renderTokChips($('repl-prompt-chips'), data.prompt_tokens);
    $('repl-gen-text').textContent = data.generated_text || '';
    renderTokChips($('repl-gen-chips'), data.generated_tokens);
    $('repl-gen-section').style.display = 'block';
  } catch (e) {
    setReplError(String(e));
  } finally {
    setReplBusy(false);
  }
}
// Wire up REPL controls.
(function initRepl() {
  $('repl-model').addEventListener('change', onReplModelChange);
  $('repl-greedy').addEventListener('change', e => {
    const tempInput = $('repl-temp');
    tempInput.disabled = e.target.checked;
    if (e.target.checked) tempInput.style.opacity = '0.5';
    else tempInput.style.opacity = '1';
  });
  // Initial state: greedy is checked by default → temperature disabled.
  const tempInput = $('repl-temp');
  tempInput.disabled = true;
  tempInput.style.opacity = '0.5';
  $('repl-tokenize-btn').addEventListener('click', replTokenize);
  $('repl-generate-btn').addEventListener('click', replGenerate);
  // Example chips: fill the prompt and fire Generate immediately so the user
  // sees the model's answer without an extra click.
  document.querySelectorAll('.ex-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      $('repl-prompt').value = chip.dataset.prompt;
      replGenerate();
    });
  });
})();
load();
</script>
</body>
</html>
"#;
