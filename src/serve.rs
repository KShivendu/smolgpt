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
use crate::eval::{run_eval, EvalReport};
use crate::model::LanguageModel;
use crate::registry::{EvalRecord, ModelRecord, Registry, TrainingRecord};
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
            trained_at: rec.trained_at,
        }
    }
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
    /// `None` for a base model; `Some(base_id)` for an RL variant. See the
    /// struct doc.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    base_model_id: Option<String>,
    cached_eval: Option<EvalRecord>,
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
}

/// Query params for the eval endpoint.
#[derive(Debug, Deserialize)]
struct EvalQuery {
    seed: Option<u64>,
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
        let training = latest_trainings
            .get(&record.id)
            .map(TrainingView::from_record);
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
            base_model_id: record.base_model_id.clone(),
            cached_eval,
            training,
            variants: Vec::new(),
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

    // In-flight check: only one eval per model id at a time. The critical
    // section is tiny (check + insert) and never awaits, so std::sync::Mutex
    // is safe here.
    {
        let mut guard = state.in_flight.lock().unwrap_or_else(|e| e.into_inner());
        if guard.contains(&id) {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({"error": "eval already running for this model"})),
            )
                .into_response();
        }
        guard.insert(id.clone());
    }

    let project_root = state.project_root.clone();
    let seed = query.seed.unwrap_or(42);

    // CPU-heavy: load model + run greedy eval inside spawn_blocking so the
    // async runtime stays responsive for other requests.
    let result = tokio::task::spawn_blocking(move || {
        run_eval_for_model(&record, &project_root, seed)
    })
    .await;

    // Remove from the in-flight set regardless of outcome.
    {
        let mut guard = state.in_flight.lock().unwrap_or_else(|e| e.into_inner());
        guard.remove(&id);
    }

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
        let model = LanguageModel::load(
            model_type,
            &model_path,
            block_size,
            vocab_size,
            record.hidden_size as usize,
            record.num_heads as usize,
            record.num_blocks as usize,
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
fn resolve_within_root(project_root: &Path, rel: &str) -> Option<PathBuf> {
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
    let model = LanguageModel::load(
        model_type,
        &model_path,
        record.block_size as usize,
        vocab_size,
        record.hidden_size as usize,
        record.num_heads as usize,
        record.num_blocks as usize,
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

const HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>smolgpt — model registry</title>
<style>
  :root {
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --yellow: #d29922;
    --red: #f85149;
    --radius: 10px;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, Segoe UI, Roboto, Inter, sans-serif;
    line-height: 1.5;
    padding: 24px;
  }
  header { margin-bottom: 24px; }
  header h1 { margin: 0 0 4px; font-size: 1.6rem; font-weight: 600; }
  header p { margin: 0; color: var(--muted); font-size: 0.95rem; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 20px;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .card-id { font-size: 1.25rem; font-weight: 600; word-break: break-all; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    background: #21262d;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.8rem;
    color: var(--muted);
  }
  .arch { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.85rem; color: var(--muted); }
  .note { font-size: 0.9rem; color: var(--muted); }
  .params { font-size: 0.9rem; }
  .params b { color: var(--text); }
  .dataset-toggle {
    cursor: pointer;
    color: var(--accent);
    font-size: 0.9rem;
    user-select: none;
  }
  .dataset-head {
    margin-top: 6px;
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    font-family: ui-monospace, monospace;
    font-size: 0.8rem;
    color: var(--muted);
    white-space: pre-wrap;
    display: none;
  }
  .dataset-head.open { display: block; }
  .banner {
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.85rem;
  }
  .banner.red { background: rgba(248,81,73,0.12); border: 1px solid var(--red); color: var(--red); }
  .banner.yellow { background: rgba(210,153,34,0.12); border: 1px solid var(--yellow); color: var(--yellow); }
  .eval-section { border-top: 1px solid var(--border); padding-top: 12px; }
  .eval-btn {
    background: #238636;
    border: 1px solid #2ea043;
    color: #fff;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 0.9rem;
    cursor: pointer;
  }
  .eval-btn:hover { background: #2ea043; }
  .eval-btn:disabled { opacity: 0.6; cursor: default; }
  .repl-btn {
    background: #1f6feb;
    border: 1px solid #388bfd;
    color: #fff;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 0.9rem;
    cursor: pointer;
    margin-left: 8px;
  }
  .repl-btn:hover { background: #388bfd; }

  /* --- Training-metrics section --- */
  .train-section { border-top: 1px solid var(--border); padding-top: 12px; }
  .train-head {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    font-size: 0.9rem;
    color: var(--muted);
  }
  .train-head .train-kind {
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.75rem;
    color: var(--text);
    background: #21262d;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 6px;
  }
  .train-head .train-stop {
    color: var(--yellow);
    font-size: 0.8rem;
  }
  .train-spark {
    display: block;
    margin-top: 8px;
    width: 100%;
    max-width: 240px;
    height: 30px;
  }
  .train-spark line, .train-spark path { stroke: var(--accent); }
  .train-spark .axis { stroke: var(--border); stroke-width: 1; }
  .train-spark .line { fill: none; stroke-width: 1.4; }
  .train-empty { color: var(--muted); font-size: 0.85rem; font-style: italic; }
  .rft-table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.8rem; }
  .rft-table th, .rft-table td {
    text-align: right;
    padding: 3px 8px;
    border-bottom: 1px solid var(--border);
    font-family: ui-monospace, SFMono-Regular, monospace;
  }
  .rft-table th { color: var(--muted); font-weight: 500; text-align: right; }
  .rft-table th:first-child, .rft-table td:first-child { text-align: left; }
  .rft-table td.muted { color: var(--muted); }
  .spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid var(--muted);
    border-top-color: var(--text);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
    margin-left: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .eval-result { margin-top: 12px; display: none; }
  .eval-result.show { display: block; }
  .pct { font-size: 1.8rem; font-weight: 700; }
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-left: 8px;
  }
  .badge.green { background: rgba(63,185,80,0.2); color: var(--green); }
  .badge.yellow { background: rgba(210,153,34,0.2); color: var(--yellow); }
  .badge.red { background: rgba(248,81,73,0.2); color: var(--red); }
  table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 0.85rem; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 500; }
  .examples { font-family: ui-monospace, monospace; font-size: 0.8rem; margin-top: 4px; }
  .examples div { padding: 2px 0; }
  .ex-ok { color: var(--green); }
  .ex-fail { color: var(--red); }
  .error-msg { color: var(--red); font-size: 0.85rem; margin-top: 8px; }
  /* Variant <select> row on base cards + collapsible <details> sections. */
  .variant-row { margin: 6px 0 10px; }
  .variant-row label { color: var(--muted); font-size: 0.85rem; }
  .variant-select {
    background: #0d1117; border: 1px solid var(--border); border-radius: 6px;
    color: var(--text); font-family: inherit; font-size: 0.85rem; padding: 3px 6px;
  }
  .eval-details, .train-details { margin-top: 8px; }
  .eval-details > summary, .train-details > summary {
    cursor: pointer; color: var(--muted); font-size: 0.85rem; user-select: none;
  }
  .eval-details[open] > summary, .train-details[open] > summary { margin-bottom: 6px; }

  /* --- REPL / Playground panel --- */
  .repl {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    margin-top: 24px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .repl h2 { margin: 0; font-size: 1.25rem; font-weight: 600; }
  .repl p.sub { margin: 0; color: var(--muted); font-size: 0.9rem; margin-top: -8px; }
  .repl-field { display: flex; flex-direction: column; gap: 6px; }
  .repl-field label { font-size: 0.85rem; color: var(--muted); }
  .repl select, .repl textarea, .repl input[type="number"] {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-family: inherit;
    font-size: 0.95rem;
    padding: 8px 10px;
  }
  .repl select:focus, .repl textarea:focus, .repl input[type="number"]:focus {
    outline: none;
    border-color: var(--accent);
  }
  .repl textarea {
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 0.9rem;
    min-height: 80px;
    resize: vertical;
    line-height: 1.4;
  }
  .repl-examples {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-top: -4px;
  }
  .repl-examples-label {
    font-size: 0.8rem;
    color: var(--muted);
    margin-right: 2px;
  }
  .ex-chip {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--text);
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 0.82rem;
    padding: 3px 10px;
    cursor: pointer;
    transition: border-color 0.12s, background 0.12s;
  }
  .ex-chip:hover { border-color: var(--accent); background: #161b22; }
  .ex-chip:active { transform: translateY(1px); }
  .repl-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: flex-end;
  }
  .repl-controls .repl-field { flex: 0 0 auto; min-width: 120px; }
  .repl-controls .repl-field input[type="number"] { width: 100px; }
  .repl-controls .checkbox-field {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.9rem; color: var(--muted);
    padding-bottom: 8px;
  }
  .repl-buttons { display: flex; gap: 10px; flex-wrap: wrap; }
  .repl-btn {
    background: #1f6feb;
    border: 1px solid #388bfd;
    color: #fff;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 0.9rem;
    cursor: pointer;
  }
  .repl-btn.secondary { background: #21262d; border: 1px solid var(--border); color: var(--text); }
  .repl-btn:hover { background: #388bfd; }
  .repl-btn.secondary:hover { background: #30363d; }
  .repl-btn:disabled { opacity: 0.6; cursor: default; }
  .repl-section { border-top: 1px solid var(--border); padding-top: 12px; }
  .repl-section h3 { margin: 0 0 8px; font-size: 0.9rem; color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }
  .tok-chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .tok-chip {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    border-radius: 8px;
    padding: 4px 8px 3px;
    border: 1px solid var(--border);
    background: #21262d;
    min-width: 28px;
    font-family: ui-monospace, SFMono-Regular, monospace;
  }
  .tok-chip .tok-str {
    font-size: 0.85rem;
    color: var(--text);
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .tok-chip .tok-str.ws { color: var(--muted); }
  .tok-chip .tok-id { font-size: 0.65rem; color: var(--muted); margin-top: 1px; }
  .gen-out {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 0.9rem;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
    min-height: 1.4em;
  }
  .repl-error { color: var(--red); font-size: 0.85rem; }
  .repl-empty { color: var(--muted); font-size: 0.85rem; font-style: italic; }
  @media (max-width: 600px) {
    body { padding: 12px; }
    .grid { grid-template-columns: 1fr; }
    .repl-controls { flex-direction: column; align-items: stretch; }
    .repl-controls .repl-field input[type="number"] { width: 100%; }
  }
</style>
</head>
<body>
<header>
  <h1>smolgpt — model registry</h1>
  <p>Browse trained models, their datasets, run greedy-decoding evals, and try prompts in the REPL below.</p>
</header>
<main id="cards" class="grid">
  <div style="color:var(--muted)">Loading models…</div>
</main>
<section class="repl" id="repl-panel">
  <h2>REPL / Playground</h2>
  <p class="sub">Pick a model, type a prompt, and tokenize or generate. Generation runs greedy or temperature-sampled decoding on the server.</p>
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
    <button class="repl-btn secondary" id="repl-tokenize-btn">Tokenize</button>
    <button class="repl-btn" id="repl-generate-btn">Generate</button>
  </div>
  <div class="repl-error" id="repl-error"></div>
  <div class="repl-section" id="repl-tokenize-section" style="display:none">
    <h3>Tokenization</h3>
    <div class="tok-chips" id="repl-tok-chips"></div>
    <div id="repl-tok-meta" class="repl-empty" style="margin-top:6px"></div>
  </div>
  <div class="repl-section" id="repl-gen-section" style="display:none">
    <h3>Prompt tokens</h3>
    <div class="tok-chips" id="repl-prompt-chips"></div>
    <h3 style="margin-top:12px">Generated text</h3>
    <div class="gen-out" id="repl-gen-text"></div>
    <h3 style="margin-top:12px">Generated tokens</h3>
    <div class="tok-chips" id="repl-gen-chips"></div>
  </div>
</section>
<script>
function fmtParams(n) {
  if (n == null) return '—';
  if (n >= 1000) return (n/1000).toFixed(1) + 'K';
  return String(n);
}
function pctClass(p) {
  if (p >= 50) return 'green';
  if (p >= 20) return 'yellow';
  return 'red';
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
// `renderEval` builds the eval panel for a card. The headline (X/total + %)
// is always visible; the per-operator (+/-) and per-answer-digit breakdown
// rows live inside a `<details>` so they're collapsed by default and the
// headline stays scannable. `by_digits` buckets are answer-digit buckets
// (1-digit answer, 2-digit answer, 3+-digit answer); old cached eval rows
// from before the bucketing change may have a different shape — we just
// render whatever buckets are present, so the UI tolerates both old and new
// shapes without a migration.
function renderEval(el, report) {
  if (!report) return;
  const pct = report.total > 0 ? (report.correct / report.total * 100) : 0;
  const pctStr = pct.toFixed(1) + '%';
  const bd = report.by_digits || [];
  const examples = report.examples || [];
  // Answer-digit buckets: index 0 = 1-digit answer, 1 = 2-digit answer,
  // 2 = 3+-digit answer. Old operand-digit rows had up to 4 buckets; render
  // whatever's present so legacy cached evals still display.
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
  // Headline stays outside the <details>; breakdown + examples collapse.
  el.innerHTML =
    `<div><span class="pct">${report.correct}/${report.total}</span><span class="badge ${pctClass(pct)}">${pctStr}</span></div>` +
    `<details class="eval-details"><summary>breakdown</summary>` +
    `<table><tr><th>op</th><th>correct</th><th>total</th></tr>` +
    `<tr><td>+</td><td>${report.correct_plus}</td><td>${report.total_plus}</td></tr>` +
    `<tr><td>-</td><td>${report.correct_minus}</td><td>${report.total_minus}</td></tr></table>` +
    (rows ? `<table><tr><th>answer digits</th><th>correct</th><th>total</th></tr>${rows}</table>` : '') +
    `<div class="examples">${exHtml}</div>` +
    `</details>`;
  el.classList.add('show');
}
// --- Training-metrics rendering ---
//
// `renderTraining` builds the markup for the "Training" section of a card from
// the `training` field of a `ModelView`. For SFT it shows the epochs/final
// loss/early-stop line plus an inline-SVG sparkline of the per-epoch loss
// trajectory (downsampled to <=120 points so a 2000-epoch series doesn't
// render 2000 SVG nodes). For RFT it shows the round count plus a tiny
// per-round table (winner_rate%, eval%, sft_final_loss). When `training` is
// null (no `trainings` row for this model), it shows a muted placeholder.
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
  const W = 120, H = 30, PAD = 2;
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
  return `<table class="rft-table">
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
  const modeBadge = mode === 'full'
    ? '<span class="train-kind" style="margin-left:6px">full</span>'
    : '<span class="muted" style="margin-left:6px">lite</span>';
  const modeDesc = mode === 'full'
    ? 'PPO-style: importance ratio + clipping + KL-to-reference'
    : 'group-relative advantage, no clipping/KL';
  return `<table class="rft-table">
    <tr><th>round</th><th>correct%</th><th>eval%</th><th>pg_loss</th></tr>
    ${rows}
  </table>` + `<div class="muted" style="font-size:0.8rem;margin-top:4px">G=${g} · ${modeDesc}${modeBadge}</div>`;
}
function renderTraining(m) {
  const t = m.training;
  if (!t) {
    return '<div class="train-section"><div class="train-empty">no training metrics recorded</div></div>';
  }
  if (t.kind === 'rft') {
    const s = t.rft_summary;
    const head = `<div class="train-head"><span class="train-kind">rft</span>` +
      `<span>RFT ${t.epochs_run} rounds</span></div>`;
    // Per-round table collapses; the round count in the head stays visible.
    const table = s ? renderRftTable(s) : '<div class="train-empty">RFT summary unavailable</div>';
    return `<div class="train-section">${head}<details class="train-details"><summary>per-round</summary>${table}</details></div>`;
  }
  if (t.kind === 'grpo') {
    const s = t.grpo_summary;
    const head = `<div class="train-head"><span class="train-kind">grpo</span>` +
      `<span>GRPO ${t.epochs_run} rounds</span></div>`;
    const table = s ? renderGrpoTable(s) : '<div class="train-empty">GRPO summary unavailable</div>';
    return `<div class="train-section">${head}<details class="train-details"><summary>per-round</summary>${table}</details></div>`;
  }
  // SFT — head (epochs/final loss/early-stop) + sparkline stay visible; no
  // collapsible section needed for SFT since the sparkline is already compact.
  const stopClause = t.early_stopped
    ? '<span class="train-stop">early-stopped</span>'
    : '<span>completed</span>';
  const head = `<div class="train-head"><span class="train-kind">sft</span>` +
    `<span>trained ${t.epochs_run} epochs · final loss ${fmtLoss(t.final_loss)} · ${stopClause}</span></div>`;
  const spark = sparklineSvg(downsampleLosses(t.loss_trajectory || [], 120));
  return `<div class="train-section">${head}${spark}</div>`;
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
async function runEval(card, modelId, btn) {
  const resultEl = card.querySelector('.eval-result');
  const errEl = card.querySelector('.error-msg');
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
// `renderVariantSelect` builds the `<select>` for a base card. Options: the
// base itself ("base") plus one per RL variant, labeled with the kind and
// round count. The `data-base-id` attribute lets the change handler find the
// base card's models array; the selected option's value is the id to swap to.
function renderVariantSelect(m) {
  if (!m.variants || !m.variants.length) return '';
  let opts = `<option value="${escapeHtml(m.id)}">base</option>`;
  for (const v of m.variants) {
    const kind = (v.training && v.training.kind) || (v.id.includes('-rft') ? 'rft' : (v.id.includes('-grpo') ? 'grpo' : 'variant'));
    const rounds = (v.training && v.training.epochs_run) ? ` (${v.training.epochs_run} rounds)` : '';
    opts += `<option value="${escapeHtml(v.id)}">${escapeHtml(kind)}${rounds}</option>`;
  }
  return `<div class="variant-row"><label>variant: <select class="variant-select" data-base-id="${escapeHtml(m.id)}">${opts}</select></label></div>`;
}
// `swapCardToVariant` re-renders a card's training + eval sections for the
// selected variant id. The card's static header (id/arch/params/dataset) is
// left in place; only the variant-dependent sections (training, eval, and
// the eval button's click handler) are refreshed.
function swapCardToVariant(card, models, selectedId) {
  const m = findModelById(models, selectedId);
  if (!m) return;
  // Replace the training section.
  const oldTrain = card.querySelector('.train-section');
  if (oldTrain) {
    const tmp = document.createElement('div');
    tmp.innerHTML = renderTraining(m);
    const newTrain = tmp.firstElementChild;
    if (newTrain) oldTrain.replaceWith(newTrain);
  }
  // Replace the eval section (button + result + error). The eval button's
  // handler captures the new id via the closure below.
  const oldEval = card.querySelector('.eval-section');
  if (oldEval) {
    const tmp = document.createElement('div');
    tmp.innerHTML = `<div class="eval-section"><button class="eval-btn">Run eval</button><div class="eval-result"></div><div class="error-msg"></div></div>`;
    const newEval = tmp.firstElementChild;
    if (newEval) {
      // Render cached eval for the selected variant, if any.
      if (m.cached_eval) {
        renderEval(newEval.querySelector('.eval-result'), m.cached_eval);
      }
      // Wire the eval button to the selected variant id.
      const btn = newEval.querySelector('.eval-btn');
      if (btn) {
        btn.addEventListener('click', () => runEval(card, selectedId, btn));
      }
      oldEval.replaceWith(newEval);
    }
  }
  // Keep the top-level REPL `<select>` in sync with the card's active
  // variant, so the REPL uses the same model the user is looking at on
  // this card. Only updates if the id is loadable (ok/mismatch).
  const replSel = $('repl-model');
  if (replSel && (m.status === 'ok' || m.status === 'mismatch')) {
    replSel.value = selectedId;
    onReplModelChange();
  }
}
function renderCard(m, allModels) {
  const card = document.createElement('div');
  card.className = 'card';
  // Stash the base id + allModels on the card so the variant-select change
  // handler can find them without re-closing over per-card state.
  card.dataset.baseId = m.id;
  // The "active" id is what eval/training/REPL currently show. Starts as the
  // base id; updated by the variant-select change handler.
  card.dataset.activeId = m.id;
  let html = '';
  if (m.status === 'mismatch') {
    html += '<div class="banner red">metadata mismatch — the saved checkpoint doesn\'t match the registered arch/tokenizer. Re-train or fix the smolgpt.db record.</div>';
  }
  html += `<div class="card-id">${escapeHtml(m.id)}</div>`;
  // Variant <select> — only on base cards with at least one variant.
  html += renderVariantSelect(m);
  if (m.model_type) {
    html += `<div class="chips"><span class="chip">${escapeHtml(m.model_type)}</span><span class="chip">${escapeHtml(m.tokenizer)}</span></div>`;
  }
  if (m.block_size != null) {
    html += `<div class="arch">block=${m.block_size} hidden=${m.hidden_size} heads=${m.num_heads} blocks=${m.num_blocks}</div>`;
  }
  html += `<div class="params"><b>${fmtParams(m.params_estimate)}</b> params (approx)</div>`;
  if (m.dataset_name) {
    html += `<div><span class="dataset-toggle">▸ dataset: ${escapeHtml(m.dataset_name)}</span>`;
    if (m.dataset_info) {
      html += `<div class="dataset-head">${m.dataset_info.line_count} lines · ${m.dataset_info.byte_size} bytes\n${m.dataset_info.head.map(escapeHtml).join('\n')}</div>`;
    }
    html += '</div>';
  }
  if (m.note) {
    html += `<div class="note">${escapeHtml(m.note)}</div>`;
  }
  html += renderTraining(m);
  if (m.status === 'ok' || m.status === 'mismatch') {
    html += `<div class="eval-section"><button class="eval-btn">Run eval</button><button class="repl-btn">REPL</button><div class="eval-result"></div><div class="error-msg"></div></div>`;
  }
  card.innerHTML = html;
  const toggle = card.querySelector('.dataset-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const head = toggle.nextElementSibling;
      if (head) head.classList.toggle('open');
    });
  }
  if (m.cached_eval) {
    renderEval(card.querySelector('.eval-result'), m.cached_eval);
  }
  const btn = card.querySelector('.eval-btn');
  if (btn) {
    btn.addEventListener('click', () => runEval(card, m.id, btn));
  }
  // "REPL" button: select this card's active model in the REPL <select>,
  // fire the model-change handler so any model-dependent REPL UI updates,
  // then smooth-scroll the REPL panel into view and focus the prompt box.
  // Uses card.dataset.activeId (not m.id) so if a variant is currently
  // selected via the variant <select>, the REPL opens that variant.
  const replBtn = card.querySelector('.repl-btn');
  if (replBtn) {
    replBtn.addEventListener('click', () => {
      const id = card.dataset.activeId;
      const sel = $('repl-model');
      let found = false;
      for (const opt of sel.options) {
        if (opt.value === id) { found = true; break; }
      }
      if (found) {
        sel.value = id;
        if (typeof onReplModelChange === 'function') onReplModelChange();
      }
      const panel = document.getElementById('repl-panel');
      if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      const prompt = $('repl-prompt');
      if (prompt) prompt.focus();
    });
  }
  // Wire the variant <select>: on change, swap the card's training + eval
  // sections to the selected variant and sync the REPL model <select>.
  const varSel = card.querySelector('.variant-select');
  if (varSel) {
    varSel.addEventListener('change', () => {
      const selectedId = varSel.value;
      card.dataset.activeId = selectedId;
      swapCardToVariant(card, allModels, selectedId);
    });
  }
  return card;
}
async function load() {
  try {
    const res = await fetch('/api/models');
    const models = await res.json();
    const container = document.getElementById('cards');
    container.innerHTML = '';
    if (!models.length) {
      container.innerHTML = '<div style="color:var(--muted)">No models found. Train a model with --train to auto-register it in smolgpt.db.</div>';
    } else {
      for (const m of models) {
        container.appendChild(renderCard(m, models));
      }
    }
    // Populate the REPL model <select> with both base models and their RL
    // variants, so the user can REPL against any variant directly. The
    // variant <select> on a card keeps the REPL in sync when the user picks
    // a variant from a card.
    const flat = [];
    for (const m of models) {
      flat.push(m);
      if (m.variants) for (const v of m.variants) flat.push(v);
    }
    populateReplModels(flat);
  } catch (e) {
    document.getElementById('cards').innerHTML = '<div class="banner red">Failed to load: ' + escapeHtml(String(e)) + '</div>';
  }
}

// --- REPL / Playground ---
let replModels = [];
function $(id) { return document.getElementById(id); }
function tokenChipColor(id) {
  // Deterministic hue from token id, rendered as a soft border tint so the
  // dark theme is preserved while still visually separating distinct tokens.
  const hue = (id * 47) % 360;
  return { border: `hsl(${hue}, 55%, 45%)`, bg: `hsla(${hue}, 55%, 45%, 0.12)` };
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
  $('repl-tokenize-btn').disabled = false;
  $('repl-tokenize-btn').innerHTML = 'Tokenizing…<span class="spinner"></span>';
  try {
    const res = await fetch('/api/repl/tokenize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId, text })
    });
    const data = await res.json();
    if (!res.ok) { setReplError(data.error || ('HTTP ' + res.status)); return; }
    renderTokChips($('repl-tok-chips'), data.tokens);
    $('repl-tok-meta').textContent =
      (data.tokens ? data.tokens.length : 0) + ' tokens · vocab ' + (data.vocab_size ?? '?') +
      ' · ' + (data.tokenizer_type || '?') + ' tokenizer';
    $('repl-tok-meta').classList.remove('repl-empty');
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
