//! Local web UI for browsing trained models, their datasets, and running
//! greedy-decoding evals from the browser.
//!
//! Activated by `--serve`. Reads `models.toml` from the project root to learn
//! which `.bin` files exist and how to load/eval them. Binds to 127.0.0.1 by
//! default (local only — do not expose to the network).
//!
//! Routes:
//!   GET /                       → embedded HTML page
//!   GET /api/models             → JSON array of model cards (no eval)
//!   GET /api/models/{id}/eval   → runs eval, returns EvalReport JSON
//!
//! Eval is CPU-heavy so it runs in `spawn_blocking`; a `Mutex<HashSet>` of
//! in-flight ids returns HTTP 409 for a concurrent second request. Successful
//! evals are cached to `<id>.eval.json` next to models.toml so reloading the
//! page shows the last result without re-running.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::extract::{Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use candle_core::Device;
use serde::{Deserialize, Serialize};

use crate::args::ModelType;
use crate::dataset;
use crate::error::{SmolError, SmolResult};
use crate::eval::{run_eval, EvalReport};
use crate::model::LanguageModel;
use crate::tokenizer::{BpeTokenizer, SimpleTokenizer, Tokenizer};

/// One `[[model]]` entry in `models.toml`.
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

/// Dataset metadata shown on a model card and via the API.
#[derive(Debug, Clone, Serialize)]
struct DatasetInfo {
    path: String,
    name: String,
    line_count: usize,
    byte_size: usize,
    head: Vec<String>,
}

/// Precomputed per-model data: registry entry + load status + param count +
/// dataset info. Computed once at startup so `/api/models` stays fast.
#[derive(Debug, Clone)]
struct ComputedModel {
    entry: ModelEntry,
    status: String,
    params_estimate: Option<usize>,
    dataset_info: Option<DatasetInfo>,
}

/// JSON view of a model, combining the registry entry with computed dataset
/// info, a load status, an approximate param count, and an optional cached
/// eval sidecar. Fields are `Option` so unregistered `.bin` files (which have
/// no metadata) can be represented with nulls.
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
    cached_eval: Option<EvalReport>,
}

/// Query params for the eval endpoint.
#[derive(Debug, Deserialize)]
struct EvalQuery {
    seed: Option<u64>,
}

/// Shared server state, cloned (via Arc) into every handler.
#[derive(Clone)]
struct AppState {
    computed: Arc<Vec<ComputedModel>>,
    project_root: Arc<PathBuf>,
    in_flight: Arc<Mutex<HashSet<String>>>,
}

/// Entry point — called from `train::do_training` when `--serve` is set.
/// Parses `models.toml`, precomputes per-model status/dataset info, builds the
/// router, and blocks on the tokio runtime.
pub fn run_serve(host: &str, port: u16) -> SmolResult<()> {
    let project_root = std::env::current_dir()
        .map_err(|e| SmolError::custom_error(&format!("cwd: {e}")))?;

    let toml_path = project_root.join("models.toml");
    let toml_content = std::fs::read_to_string(&toml_path).map_err(|e| {
        SmolError::custom_error(&format!(
            "Failed to read {}: {e}. Create a [[model]] registry; see the header comment in models.toml.",
            toml_path.display()
        ))
    })?;
    let models_file: ModelsFile = toml::from_str(&toml_content)
        .map_err(|e| SmolError::custom_error(&format!("Failed to parse {}: {e}", toml_path.display())))?;

    // Precompute status, param count, and dataset info for each registered
    // model. This is the slow part (loads each .bin once) and runs at startup
    // so `/api/models` stays fast.
    let mut computed: Vec<ComputedModel> = Vec::with_capacity(models_file.model.len());
    for entry in &models_file.model {
        let (status, params_estimate) = verify_and_estimate(entry, &project_root);
        if status == "mismatch" {
            eprintln!(
                "[serve] WARNING: model '{}' status=mismatch — edit models.toml so the registered \
                 arch/tokenizer matches the saved checkpoint.",
                entry.id
            );
        }
        let dataset_info = compute_dataset_info(&project_root, entry);
        computed.push(ComputedModel {
            entry: entry.clone(),
            status,
            params_estimate,
            dataset_info,
        });
    }

    let state = AppState {
        computed: Arc::new(computed),
        project_root: Arc::new(project_root),
        in_flight: Arc::new(Mutex::new(HashSet::new())),
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
    let mut views: Vec<ModelView> = Vec::new();

    // Registered models — use precomputed status/dataset info.
    for cm in state.computed.iter() {
        let entry = &cm.entry;
        let cached_eval = read_cached_eval(&state.project_root, &entry.id);
        views.push(ModelView {
            id: entry.id.clone(),
            path: entry.path.clone(),
            status: cm.status.clone(),
            model_type: Some(entry.model_type.clone()),
            tokenizer: Some(entry.tokenizer.clone()),
            vocab_size: Some(entry.vocab_size),
            block_size: Some(entry.block_size),
            hidden_size: Some(entry.hidden_size),
            num_heads: Some(entry.num_heads),
            num_blocks: Some(entry.num_blocks),
            dataset: Some(entry.dataset.clone()),
            dataset_name: Some(entry.dataset_name.clone()),
            dataset_info: cm.dataset_info.clone(),
            eval_min: Some(entry.eval_min),
            eval_max: Some(entry.eval_max),
            eval_samples: Some(entry.eval_samples),
            note: Some(entry.note.clone()),
            params_estimate: cm.params_estimate,
            cached_eval,
        });
    }

    // Unregistered .bin files in the project root.
    let registered: HashSet<String> = state.computed.iter().map(|cm| cm.entry.path.clone()).collect();
    if let Ok(entries) = std::fs::read_dir(&*state.project_root) {
        for ent in entries.flatten() {
            let path = ent.path();
            if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                continue;
            }
            let rel = match path.file_name().and_then(|n| n.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            if registered.contains(&rel) {
                continue;
            }
            let id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            views.push(ModelView {
                id,
                path: rel,
                status: "unregistered".to_string(),
                model_type: None,
                tokenizer: None,
                vocab_size: None,
                block_size: None,
                hidden_size: None,
                num_heads: None,
                num_blocks: None,
                dataset: None,
                dataset_name: None,
                dataset_info: None,
                eval_min: None,
                eval_max: None,
                eval_samples: None,
                note: None,
                params_estimate: None,
                cached_eval: None,
            });
        }
    }

    Json(views).into_response()
}

async fn eval_model(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Query(query): Query<EvalQuery>,
) -> Response {
    let Some(entry) = state
        .computed
        .iter()
        .find(|cm| cm.entry.id == id)
        .map(|cm| cm.entry.clone())
    else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("model '{id}' not in registry")})),
        )
            .into_response();
    };

    // In-flight check: only one eval per model id at a time. The critical
    // section is tiny (check + insert) and never awaits, so std::sync::Mutex
    // is safe here.
    {
        let mut guard = state.in_flight.lock().unwrap();
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
        run_eval_for_model(&entry, &project_root, seed)
    })
    .await;

    // Remove from the in-flight set regardless of outcome.
    {
        let mut guard = state.in_flight.lock().unwrap();
        guard.remove(&id);
    }

    match result {
        Ok(Ok(report)) => {
            // Cache to <id>.eval.json next to models.toml so a page reload
            // shows the last result without re-running.
            let sidecar = state.project_root.join(format!("{id}.eval.json"));
            if let Ok(json) = serde_json::to_string_pretty(&report) {
                let _ = std::fs::write(&sidecar, json);
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

// --- Helpers ---

/// Read a dataset file and compute line count, byte size, and the first 5
/// non-empty lines. Returns `None` if the file can't be read.
fn compute_dataset_info(project_root: &Path, entry: &ModelEntry) -> Option<DatasetInfo> {
    let path = project_root.join(&entry.dataset);
    let Ok(content) = std::fs::read_to_string(&path) else {
        return None;
    };
    let byte_size = content.len();
    let non_empty: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    let line_count = non_empty.len();
    let head = non_empty.iter().take(5).map(|s| s.to_string()).collect();
    Some(DatasetInfo {
        path: entry.dataset.clone(),
        name: entry.dataset_name.clone(),
        line_count,
        byte_size,
        head,
    })
}

/// Attempt to load the model with its registered tokenizer. If that fails, try
/// the other tokenizer as a fallback (char <-> bpe). Returns the status
/// ("ok" | "mismatch") and an approximate param count.
fn verify_and_estimate(entry: &ModelEntry, project_root: &Path) -> (String, Option<usize>) {
    match try_load(entry, project_root, &entry.tokenizer) {
        Ok(vocab_size) => {
            let params = estimate_params(entry, vocab_size);
            ("ok".to_string(), params)
        }
        Err(primary_err) => {
            let fallback = if entry.tokenizer == "char" { "bpe" } else { "char" };
            match try_load(entry, project_root, fallback) {
                Ok(vocab_size) => {
                    eprintln!(
                        "[serve] NOTE: model '{}' registered tokenizer '{}' failed ({}); \
                         fallback '{}' loads. Consider updating models.toml.",
                        entry.id, entry.tokenizer, primary_err, fallback
                    );
                    let params = estimate_params(entry, vocab_size);
                    ("ok".to_string(), params)
                }
                Err(fallback_err) => {
                    eprintln!(
                        "[serve] WARNING: model '{}' mismatch — registered tokenizer '{}' \
                         failed ({}); fallback '{}' also failed ({})",
                        entry.id, entry.tokenizer, primary_err, fallback, fallback_err
                    );
                    ("mismatch".to_string(), None)
                }
            }
        }
    }
}

/// Build the tokenizer of the given type, load the model, and return the
/// actual vocab size on success. Does not run eval — the load itself verifies
/// the architecture (block/hidden/heads/blocks) and vocab shape match the
/// saved checkpoint.
fn try_load(entry: &ModelEntry, project_root: &Path, tokenizer_type: &str) -> SmolResult<usize> {
    let corpus_path = project_root.join(&entry.dataset);
    let corpus = std::fs::read_to_string(&corpus_path).map_err(|e| {
        SmolError::custom_error(&format!(
            "Failed to read dataset {}: {e}",
            corpus_path.display()
        ))
    })?;

    let (vocab_size, _tokenizer): (usize, Box<dyn Tokenizer<u32>>) = match tokenizer_type {
        "char" => {
            let t = SimpleTokenizer::new(&corpus);
            (t.vocab_size(), Box::new(t))
        }
        "bpe" => {
            let t = BpeTokenizer::train(&corpus, entry.vocab_size);
            (t.vocab_size(), Box::new(t))
        }
        other => return Err(SmolError::invalid_argument(&format!("Unknown tokenizer type: {other}"))),
    };

    let model_type = match entry.model_type.as_str() {
        "gpt" => ModelType::Gpt,
        "bigram" => ModelType::Bigram,
        other => return Err(SmolError::invalid_argument(&format!("Unknown model type: {other}"))),
    };

    let model_path = project_root.join(&entry.path);
    let device = Device::Cpu;
    let _model = LanguageModel::load(
        model_type,
        &model_path,
        entry.block_size,
        vocab_size,
        entry.hidden_size,
        entry.num_heads,
        entry.num_blocks,
        &device,
    )?;
    Ok(vocab_size)
}

/// Rough param count from the architecture. For GPT this sums embeddings,
/// per-block attention/FFN/norm params, and the LM head; for BigramLM it's
/// `vocab^2`. Approximate (ignores small bias terms where noted) but close
/// enough to distinguish ~7K from ~78K models.
fn estimate_params(entry: &ModelEntry, vocab_size: usize) -> Option<usize> {
    Some(match entry.model_type.as_str() {
        "bigram" => vocab_size * vocab_size,
        "gpt" => {
            let h = entry.hidden_size;
            let b = entry.block_size;
            let nb = entry.num_blocks;
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

/// Read a cached `<id>.eval.json` sidecar if one exists next to models.toml.
fn read_cached_eval(project_root: &Path, id: &str) -> Option<EvalReport> {
    let sidecar = project_root.join(format!("{id}.eval.json"));
    let content = std::fs::read_to_string(&sidecar).ok()?;
    serde_json::from_str(&content).ok()
}

/// Build tokenizer, load model, run eval. Called inside `spawn_blocking`.
fn run_eval_for_model(entry: &ModelEntry, project_root: &Path, seed: u64) -> SmolResult<EvalReport> {
    let corpus_path = project_root.join(&entry.dataset);
    let corpus = dataset::load_corpus(&corpus_path, false);

    let tokenizer: Box<dyn Tokenizer<u32>> = match entry.tokenizer.as_str() {
        "char" => Box::new(SimpleTokenizer::new(&corpus)),
        "bpe" => Box::new(BpeTokenizer::train(&corpus, entry.vocab_size)),
        other => return Err(SmolError::invalid_argument(&format!("Unknown tokenizer type: {other}"))),
    };
    let vocab_size = tokenizer.vocab_size();

    let model_type = match entry.model_type.as_str() {
        "gpt" => ModelType::Gpt,
        "bigram" => ModelType::Bigram,
        other => return Err(SmolError::invalid_argument(&format!("Unknown model type: {other}"))),
    };

    let model_path = project_root.join(&entry.path);
    let device = Device::Cpu;
    let model = LanguageModel::load(
        model_type,
        &model_path,
        entry.block_size,
        vocab_size,
        entry.hidden_size,
        entry.num_heads,
        entry.num_blocks,
        &device,
    )?;

    println!(
        "[serve] Running eval for '{}' ({} samples, range [{},{}], seed={})",
        entry.id, entry.eval_samples, entry.eval_min, entry.eval_max, seed
    );

    run_eval(
        &model,
        tokenizer.as_ref(),
        &device,
        entry.eval_samples,
        entry.eval_min,
        entry.eval_max,
        entry.block_size,
        Some(seed),
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
  @media (max-width: 600px) {
    body { padding: 12px; }
    .grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<header>
  <h1>smolgpt — model registry</h1>
  <p>Browse trained models, their datasets, and run greedy-decoding evals.</p>
</header>
<main id="cards" class="grid">
  <div style="color:var(--muted)">Loading models…</div>
</main>
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
function renderEval(el, report) {
  if (!report) return;
  const pct = report.total > 0 ? (report.correct / report.total * 100) : 0;
  const pctStr = pct.toFixed(1) + '%';
  const bd = report.by_digits || [];
  const examples = report.examples || [];
  const rows = [
    ['1-digit', bd[0]],
    ['2-digit', bd[1]],
    ['3-digit', bd[2]],
    ['4+-digit', bd[3]]
  ].map(([label, b]) => `<tr><td>${label}</td><td>${b?b[0]:0}</td><td>${b?b[1]:0}</td></tr>`).join('');
  const exHtml = examples.slice(0,10).map(ex => {
    const gen = (ex.generated||'').split('\n')[0];
    const cls = ex.correct ? 'ex-ok' : 'ex-fail';
    const mark = ex.correct ? 'ok' : 'FAIL';
    return `<div class="${cls}">[${mark}] ${escapeHtml(ex.prompt)}${escapeHtml(gen)} (true: ${ex.true_answer})</div>`;
  }).join('');
  el.innerHTML =
    `<div><span class="pct">${report.correct}/${report.total}</span><span class="badge ${pctClass(pct)}">${pctStr}</span></div>` +
    `<table><tr><th>op</th><th>correct</th><th>total</th></tr>` +
    `<tr><td>+</td><td>${report.correct_plus}</td><td>${report.total_plus}</td></tr>` +
    `<tr><td>-</td><td>${report.correct_minus}</td><td>${report.total_minus}</td></tr></table>` +
    `<table><tr><th>digits</th><th>correct</th><th>total</th></tr>${rows}</table>` +
    `<div class="examples">${exHtml}</div>`;
  el.classList.add('show');
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
function renderCard(m) {
  const card = document.createElement('div');
  card.className = 'card';
  let html = '';
  if (m.status === 'mismatch') {
    html += '<div class="banner red">metadata mismatch — edit models.toml to load this model</div>';
  }
  if (m.status === 'unregistered') {
    html += '<div class="banner yellow">add a [[model]] entry to models.toml</div>';
  }
  html += `<div class="card-id">${escapeHtml(m.id)}</div>`;
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
  if (m.status === 'ok' || m.status === 'mismatch') {
    html += `<div class="eval-section"><button class="eval-btn">Run eval</button><div class="eval-result"></div><div class="error-msg"></div></div>`;
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
  return card;
}
async function load() {
  try {
    const res = await fetch('/api/models');
    const models = await res.json();
    const container = document.getElementById('cards');
    container.innerHTML = '';
    if (!models.length) {
      container.innerHTML = '<div style="color:var(--muted)">No models found. Add entries to models.toml.</div>';
      return;
    }
    for (const m of models) {
      container.appendChild(renderCard(m));
    }
  } catch (e) {
    document.getElementById('cards').innerHTML = '<div class="banner red">Failed to load: ' + escapeHtml(String(e)) + '</div>';
  }
}
load();
</script>
</body>
</html>
"#;
