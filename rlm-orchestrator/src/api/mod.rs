//! REST API for RLM orchestrator

use crate::orchestrator::{IterationRecord, ProgressEvent, RlmOrchestrator, RlmResult};
use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        Html,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// API state
pub struct ApiState {
    pub orchestrator: Arc<RlmOrchestrator>,
    pub wasm_enabled: bool,
    pub rust_wasm_enabled: bool,
    pub root_provider_name: String,
}

/// Request to process a query
#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    /// The query to process
    pub query: String,
    /// The context to analyze
    pub context: String,
    /// Optional: Override root model (e.g., "glm-4.7", "deepseek-chat")
    #[serde(default)]
    pub root_model: Option<String>,
    /// Optional: Override sub model (e.g., "local-sub", "manager-gemma9b")
    #[serde(default)]
    pub sub_model: Option<String>,
}

/// Response from a query
#[derive(Debug, Serialize)]
pub struct QueryResponse {
    /// Whether the query succeeded
    pub success: bool,
    /// The answer (if successful)
    pub answer: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Number of iterations taken
    pub iterations: usize,
    /// Total sub-LM calls made
    pub sub_calls: usize,
    /// Whether RLM was bypassed (direct call for small contexts)
    pub bypassed: bool,
}

impl From<RlmResult> for QueryResponse {
    fn from(result: RlmResult) -> Self {
        Self {
            success: result.success,
            answer: if result.success {
                Some(result.answer)
            } else {
                None
            },
            error: result.error,
            iterations: result.iterations,
            sub_calls: result.total_sub_calls,
            bypassed: result.bypassed,
        }
    }
}

/// Debug response with full iteration history
#[derive(Debug, Serialize)]
pub struct DebugResponse {
    /// Whether the query succeeded
    pub success: bool,
    /// The final answer
    pub answer: String,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Number of iterations taken
    pub iterations: usize,
    /// Total sub-LM calls made
    pub total_sub_calls: usize,
    /// Full iteration history
    pub history: Vec<IterationRecordJson>,
    /// Original query
    pub query: String,
    /// Context length in characters
    pub context_length: usize,
    /// Total prompt tokens used (RLM approach)
    pub total_prompt_tokens: u32,
    /// Total completion tokens used (RLM approach)
    pub total_completion_tokens: u32,
    /// Estimated baseline tokens (full context approach)
    pub baseline_tokens: u32,
    /// Token savings percentage
    pub token_savings_pct: f64,
    /// Whether RLM was bypassed (direct call for small contexts)
    pub bypassed: bool,
}

/// JSON-serializable iteration record
#[derive(Debug, Serialize)]
pub struct IterationRecordJson {
    pub step: usize,
    pub llm_response: String,
    pub commands: String,
    pub output: String,
    pub error: Option<String>,
    pub sub_calls: usize,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

impl From<&IterationRecord> for IterationRecordJson {
    fn from(r: &IterationRecord) -> Self {
        Self {
            step: r.step,
            llm_response: r.llm_response.clone(),
            commands: r.commands.clone(),
            output: r.output.clone(),
            error: r.error.clone(),
            sub_calls: r.sub_calls,
            prompt_tokens: r.tokens.prompt_tokens,
            completion_tokens: r.tokens.completion_tokens,
        }
    }
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub wasm_enabled: bool,
    pub rust_wasm_enabled: bool,
}

/// SSE event for streaming progress
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "query_start")]
    QueryStart {
        context_chars: usize,
        query_len: usize,
    },
    #[serde(rename = "iteration_start")]
    IterationStart { step: usize },
    #[serde(rename = "llm_start")]
    LlmStart { step: usize },
    #[serde(rename = "llm_complete")]
    LlmComplete {
        step: usize,
        duration_ms: u64,
        prompt_tokens: u32,
        completion_tokens: u32,
        response_preview: String,
    },
    #[serde(rename = "commands")]
    Commands { step: usize, commands: String },
    #[serde(rename = "wasm_compile_start")]
    WasmCompileStart { step: usize },
    #[serde(rename = "wasm_compile_complete")]
    WasmCompileComplete { step: usize, duration_ms: u64 },
    #[serde(rename = "wasm_run_complete")]
    WasmRunComplete { step: usize, duration_ms: u64 },
    #[serde(rename = "cli_compile_start")]
    CliCompileStart { step: usize },
    #[serde(rename = "cli_compile_complete")]
    CliCompileComplete { step: usize, duration_ms: u64 },
    #[serde(rename = "cli_run_complete")]
    CliRunComplete { step: usize, duration_ms: u64 },
    #[serde(rename = "command_complete")]
    CommandComplete {
        step: usize,
        output_preview: String,
        exec_ms: u64,
    },
    #[serde(rename = "iteration_complete")]
    IterationComplete {
        step: usize,
        llm_response: String,
        commands: String,
        output: String,
        error: Option<String>,
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    #[serde(rename = "final_answer")]
    FinalAnswer { answer: String },
    #[serde(rename = "complete")]
    Complete {
        success: bool,
        answer: String,
        error: Option<String>,
        iterations: usize,
        total_sub_calls: usize,
        context_length: usize,
        total_prompt_tokens: u32,
        total_completion_tokens: u32,
        total_duration_ms: u64,
    },
    #[serde(rename = "error")]
    Error { message: String },
}

/// Create the API router
pub fn create_router(state: Arc<ApiState>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/query", post(process_query))
        .route("/debug", post(debug_query))
        .route("/stream", post(stream_query))
        .route("/visualize", get(visualize_page))
        .route("/samples/war-and-peace", get(serve_war_and_peace))
        .route("/samples/large-logs", get(serve_large_logs))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024)) // 10MB for large contexts
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Health check endpoint
async fn health_check(State(state): State<Arc<ApiState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        wasm_enabled: state.wasm_enabled,
        rust_wasm_enabled: state.rust_wasm_enabled,
    })
}

/// Serve War and Peace text file for large context demos
async fn serve_war_and_peace() -> Result<String, (StatusCode, String)> {
    // Try multiple possible locations
    let paths = [
        "/Users/mike/Downloads/war-and-peace-tolstoy-clean.txt",
        "../demo/war-and-peace-needle.txt",
        "demo/war-and-peace-needle.txt",
    ];

    for path in &paths {
        if let Ok(content) = tokio::fs::read_to_string(path).await {
            return Ok(content);
        }
    }

    Err((StatusCode::NOT_FOUND, "War and Peace file not found".to_string()))
}

/// Generate large log data for demos
async fn serve_large_logs() -> String {
    let mut logs = String::with_capacity(500_000);
    let error_types = ["AuthenticationFailed", "ConnectionTimeout", "RequestFailed", "ValidationError", "DatabaseError", "PermissionDenied", "RateLimited", "ServiceUnavailable"];
    let ips = ["192.168.1.100", "10.0.0.50", "172.16.0.25", "10.0.0.75", "192.168.1.200", "172.16.0.30", "10.0.0.60", "192.168.1.105", "10.0.0.80", "172.16.0.40"];
    let endpoints = ["/api/users", "/api/data", "/api/health", "/api/products", "/api/orders", "/api/auth", "/api/settings", "/api/batch"];
    let methods = ["GET", "POST", "PUT", "DELETE"];

    for i in 0..5000 {
        let hour = 10 + (i / 360) % 14;
        let minute = (i / 6) % 60;
        let second = (i * 7) % 60;

        let is_error = i % 7 == 0;
        let ip = ips[i % ips.len()];
        let endpoint = endpoints[i % endpoints.len()];
        let method = methods[i % methods.len()];

        if is_error {
            let error_type = error_types[i % error_types.len()];
            logs.push_str(&format!(
                "2024-01-15 {:02}:{:02}:{:02} [ERROR] {} from {} - {} {} failed\n",
                hour, minute, second, error_type, ip, method, endpoint
            ));
        } else {
            let status = if i % 5 == 0 { "201 Created" } else { "200 OK" };
            let ms = 10 + (i * 13) % 500;
            logs.push_str(&format!(
                "2024-01-15 {:02}:{:02}:{:02} [INFO] Request from {} - {} {} - {} - {}ms\n",
                hour, minute, second, ip, method, endpoint, status, ms
            ));
        }
    }

    logs
}

/// Process a query
async fn process_query(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, String)> {
    match state
        .orchestrator
        .process(&request.query, &request.context)
        .await
    {
        Ok(result) => Ok(Json(result.into())),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

/// Debug query - returns full iteration history
async fn debug_query(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<DebugResponse>, (StatusCode, String)> {
    let context_length = request.context.len();
    let query = request.query.clone();

    match state
        .orchestrator
        .process(&request.query, &request.context)
        .await
    {
        Ok(result) => {
            // Estimate baseline tokens: ~1 token per 4 chars for context + query overhead
            let baseline_tokens = (context_length as u32 / 4) + 200; // 200 for query/system prompt
            let rlm_total = result.total_prompt_tokens + result.total_completion_tokens;
            let token_savings_pct = if baseline_tokens > 0 {
                ((baseline_tokens as f64 - rlm_total as f64) / baseline_tokens as f64) * 100.0
            } else {
                0.0
            };

            Ok(Json(DebugResponse {
                success: result.success,
                answer: result.answer,
                error: result.error,
                iterations: result.iterations,
                total_sub_calls: result.total_sub_calls,
                history: result.history.iter().map(|r| r.into()).collect(),
                query,
                context_length,
                total_prompt_tokens: result.total_prompt_tokens,
                total_completion_tokens: result.total_completion_tokens,
                baseline_tokens,
                token_savings_pct,
                bypassed: result.bypassed,
            }))
        }
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

/// Streaming query - returns SSE events for real-time progress
async fn stream_query(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<QueryRequest>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = mpsc::channel::<StreamEvent>(100);
    let context_length = request.context.len();

    // Spawn the processing task
    let orchestrator = state.orchestrator.clone();
    let query = request.query.clone();
    let context = request.context.clone();

    tokio::spawn(async move {
        let query_start = std::time::Instant::now();
        // Create a progress callback that sends events to the channel
        let tx_clone = tx.clone();
        let callback = Box::new(move |event: ProgressEvent| {
            tracing::debug!("Progress event: {:?}", event);
            let stream_event = match event {
                ProgressEvent::QueryStart { context_chars, query_len } => {
                    StreamEvent::QueryStart { context_chars, query_len }
                }
                ProgressEvent::IterationStart { step } => StreamEvent::IterationStart { step },
                ProgressEvent::LlmCallStart { step } => StreamEvent::LlmStart { step },
                ProgressEvent::LlmCallComplete {
                    step,
                    duration_ms,
                    prompt_tokens,
                    completion_tokens,
                    response_preview,
                } => StreamEvent::LlmComplete {
                    step,
                    duration_ms,
                    prompt_tokens,
                    completion_tokens,
                    response_preview,
                },
                ProgressEvent::CommandsExtracted { step, commands } => {
                    StreamEvent::Commands { step, commands }
                }
                ProgressEvent::WasmCompileStart { step } => StreamEvent::WasmCompileStart { step },
                ProgressEvent::WasmCompileComplete { step, duration_ms } => {
                    StreamEvent::WasmCompileComplete { step, duration_ms }
                }
                ProgressEvent::WasmRunComplete { step, duration_ms } => {
                    StreamEvent::WasmRunComplete { step, duration_ms }
                }
                ProgressEvent::CliCompileStart { step } => StreamEvent::CliCompileStart { step },
                ProgressEvent::CliCompileComplete { step, duration_ms } => {
                    StreamEvent::CliCompileComplete { step, duration_ms }
                }
                ProgressEvent::CliRunComplete { step, duration_ms } => {
                    StreamEvent::CliRunComplete { step, duration_ms }
                }
                ProgressEvent::CommandComplete {
                    step,
                    output_preview,
                    exec_ms,
                } => StreamEvent::CommandComplete {
                    step,
                    output_preview,
                    exec_ms,
                },
                ProgressEvent::IterationComplete { step, record } => {
                    StreamEvent::IterationComplete {
                        step,
                        llm_response: record.llm_response.clone(),
                        commands: record.commands.clone(),
                        output: record.output.clone(),
                        error: record.error.clone(),
                        prompt_tokens: record.tokens.prompt_tokens,
                        completion_tokens: record.tokens.completion_tokens,
                    }
                }
                ProgressEvent::FinalAnswer { answer } => StreamEvent::FinalAnswer { answer },
                ProgressEvent::Complete { .. } => {
                    // We'll send the complete event with full data after process returns
                    return;
                }
            };
            // Use try_send to avoid blocking - drop events if channel is full
            match tx_clone.try_send(stream_event) {
                Ok(_) => tracing::debug!("Sent event to channel"),
                Err(e) => tracing::warn!("Failed to send event: {:?}", e),
            }
        });

        // Run the query with progress callback
        match orchestrator
            .process_with_progress(&query, &context, Some(callback))
            .await
        {
            Ok(result) => {
                let _ = tx
                    .send(StreamEvent::Complete {
                        success: result.success,
                        answer: result.answer,
                        error: result.error,
                        iterations: result.iterations,
                        total_sub_calls: result.total_sub_calls,
                        context_length,
                        total_prompt_tokens: result.total_prompt_tokens,
                        total_completion_tokens: result.total_completion_tokens,
                        total_duration_ms: query_start.elapsed().as_millis() as u64,
                    })
                    .await;
            }
            Err(e) => {
                let _ = tx
                    .send(StreamEvent::Error {
                        message: e.to_string(),
                    })
                    .await;
            }
        }
    });

    // Convert the receiver to a stream of SSE events
    let stream = ReceiverStream::new(rx).map(|event| {
        let json = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
        Ok(Event::default().data(json))
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Visualization page (no caching to ensure fresh JS)
async fn visualize_page(State(state): State<Arc<ApiState>>) -> (axum::http::HeaderMap, Html<String>) {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::header::CACHE_CONTROL,
        "no-cache, no-store, must-revalidate".parse().unwrap(),
    );
    headers.insert(
        axum::http::header::PRAGMA,
        "no-cache".parse().unwrap(),
    );
    // Fill in the root provider name
    let html = VISUALIZE_HTML.replace("{root_provider}", &state.root_provider_name);
    (headers, Html(html))
}

const VISUALIZE_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RLM Visualizer</title>
    <style>
        :root {
            --bg: #1a1a2e;
            --card: #16213e;
            --accent: #0f3460;
            --highlight: #3b82f6;
            --text: #eee;
            --muted: #888;
            --success: #2ca02c;
            --error: #d62728;
            --progress: #3b82f6;
            /* D3.js-style categorical palette for distinct colors */
            --color-llm: #1f77b4;      /* Blue - root LLM */
            --color-dsl: #17becf;      /* Cyan - L1 DSL */
            --color-wasm: #ff7f0e;     /* Orange - L2 WASM */
            --color-cli: #9467bd;      /* Purple - L3 CLI */
            --color-llm4: #e377c2;     /* Pink - L4 LLM delegation */
            --color-done: #2ca02c;     /* Green - success */
            --color-error: #d62728;    /* Red - error */
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body {
            font-family: 'SF Mono', 'Consolas', monospace;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            overflow: hidden;
        }
        .container {
            max-width: 100%;
            height: 100vh;
            display: flex;
            flex-direction: column;
            padding: 15px 30px;
        }

        /* Header with title and button */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            flex-shrink: 0;
        }
        h1 {
            font-size: 1.4rem;
            color: var(--highlight);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        h1 span { font-size: 1.8rem; }
        button {
            background: var(--highlight);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            font-weight: 600;
            transition: transform 0.1s, opacity 0.2s;
        }
        button:hover { opacity: 0.9; }
        button:active { transform: scale(0.98); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Tab bar */
        .tab-bar {
            display: flex;
            gap: 0;
            margin-bottom: 15px;
            flex-shrink: 0;
        }
        .tab {
            padding: 10px 25px;
            background: var(--accent);
            color: var(--muted);
            border: none;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: background 0.2s, color 0.2s;
        }
        .tab:first-child { border-radius: 8px 0 0 8px; }
        .tab:last-child { border-radius: 0 8px 8px 0; }
        .tab.active {
            background: var(--highlight);
            color: white;
        }
        .tab:hover:not(.active) { background: #1a4a7a; }

        /* Tab content container */
        .tab-content {
            flex: 1;
            min-height: 0;
            display: none;
            overflow: hidden;
        }
        .tab-content.active {
            display: flex;
            flex-direction: column;
        }

        /* Input Data tab */
        .input-section {
            background: var(--card);
            padding: 15px 20px;
            border-radius: 12px;
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .example-selector {
            margin-bottom: 12px;
            flex-shrink: 0;
        }
        .example-selector select {
            max-width: 400px;
            width: 100%;
            background: var(--bg);
            border: 1px solid var(--accent);
            border-radius: 8px;
            padding: 10px;
            color: var(--text);
            font-family: inherit;
            font-size: 1rem;
        }
        .example-selector select:focus {
            outline: none;
            border-color: var(--highlight);
        }
        .example-tags {
            display: flex;
            gap: 8px;
            margin-top: 8px;
            flex-wrap: wrap;
        }
        .tag {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 1rem;
            text-transform: uppercase;
            font-weight: 600;
        }
        .tag.dsl { background: #34d399; color: #000; }
        .tag.wasm { background: var(--color-wasm); color: #000; }
        .tag.combined { background: #8b5cf6; color: #fff; }
        .tag.basic { background: var(--accent); }
        .tag.aggregation { background: #7c3aed; }
        .tag.large-context { background: #dc2626; color: #fff; }

        /* Example info box */
        .example-info {
            background: var(--card);
            border-radius: 10px;
            padding: 15px;
            margin-top: 12px;
            display: none;
            border-left: 4px solid var(--highlight);
        }
        .example-info.visible {
            display: block;
        }
        .example-info-row {
            display: flex;
            gap: 20px;
            margin-bottom: 10px;
        }
        .example-info-item {
            flex: 1;
        }
        .example-info-label {
            font-size: 1rem;
            color: var(--muted);
            text-transform: uppercase;
            margin-bottom: 4px;
            cursor: help;
            border-bottom: 1px dotted var(--muted);
            display: inline-block;
        }
        .example-info-value {
            font-size: 1.1rem;
            font-weight: 600;
        }
        .example-info-value.benchmark {
            color: #818cf8;
        }
        .example-info-value.level {
            color: #34d399;
        }
        .example-info-desc {
            color: var(--text);
            line-height: 1.5;
            margin-top: 8px;
            padding-top: 10px;
            border-top: 1px solid var(--accent);
        }

        .input-row {
            display: flex;
            gap: 15px;
            margin-bottom: 12px;
            flex-shrink: 0;
        }
        .query-col {
            width: 100%;
            flex-shrink: 0;
        }
        label {
            font-size: 1rem;
            color: var(--muted);
            margin-bottom: 5px;
            display: block;
        }
        textarea {
            width: 100%;
            background: var(--bg);
            border: 1px solid var(--accent);
            border-radius: 8px;
            padding: 12px;
            color: var(--text);
            font-family: inherit;
            font-size: 1rem;
            resize: none;
        }
        textarea:focus {
            outline: none;
            border-color: var(--highlight);
        }
        #query { height: 80px; }

        .data-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        .data-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        .context-stats {
            font-size: 1rem;
            color: var(--muted);
        }
        #context {
            flex: 1;
            min-height: 100px;
        }

        /* Output tab */
        .output-section {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
            overflow: hidden;
        }

        /* Progress section */
        .progress-section {
            background: var(--card);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 15px;
            flex-shrink: 0;
            position: relative;
        }
        .progress-section.expanded {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .progress-header h2 {
            font-size: 1rem;
            color: var(--progress);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .progress-query {
            flex: 1;
            font-size: 0.9rem;
            color: var(--muted);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            margin: 0 15px;
            font-style: italic;
        }
        .progress-status {
            font-size: 1rem;
            color: var(--muted);
            flex-shrink: 0;
        }
        .progress-status.active { color: var(--progress); }
        .progress-log {
            background: var(--bg);
            border-radius: 8px;
            padding: 12px;
            flex: 1;
            overflow-y: auto;
            font-size: 1rem;
            line-height: 1.6;
            min-height: 100px;
        }
        .progress-log .event {
            margin-bottom: 4px;
            display: flex;
            gap: 8px;
        }
        .progress-log .event-time {
            color: var(--muted);
            min-width: 60px;
        }
        .progress-log .event-icon { min-width: 20px; }
        .progress-log .event-llm { color: var(--color-llm); }
        .progress-log .event-wasm { color: var(--color-wasm); }
        .progress-log .event-cli { color: var(--color-cli); }
        .progress-log .event-cmd { color: var(--color-dsl); }
        .progress-log .event-done { color: var(--color-done); }
        .progress-log .event-error { color: var(--color-error); }

        /* Progress legend - absolute position at bottom-right of progress section */
        .progress-legend {
            position: absolute;
            bottom: 25px;
            right: 25px;
            background: rgba(30, 41, 59, 0.98);
            border-radius: 8px;
            padding: 10px 14px;
            z-index: 100;
            font-size: 1rem;
            display: flex;
            flex-direction: column;
            gap: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: help;
        }
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .legend-dot.llm { background: var(--color-llm); }
        .legend-dot.cmd { background: var(--color-dsl); }
        .legend-dot.wasm { background: var(--color-wasm); }
        .legend-dot.cli { background: var(--color-cli); }
        .legend-dot.llm4 { background: var(--color-llm4); }
        .legend-dot.done { background: var(--color-done); }
        .legend-dot.error { background: var(--color-error); }
        /* Legend text matches bullet color */
        .legend-item.llm .legend-label { color: var(--color-llm); }
        .legend-item.dsl .legend-label { color: var(--color-dsl); }
        .legend-item.wasm .legend-label { color: var(--color-wasm); }
        .legend-item.cli .legend-label { color: var(--color-cli); }
        .legend-item.llm4 .legend-label { color: var(--color-llm4); }
        .legend-item.done .legend-label { color: var(--color-done); }
        .legend-item.error .legend-label { color: var(--color-error); }
        .legend-label {
            white-space: nowrap;
        }
        /* Custom tooltips */
        .has-tooltip {
            position: relative;
        }
        .has-tooltip .tooltip {
            display: none;
            position: absolute;
            bottom: 100%;
            left: 0;
            background: #0f172a;
            color: #e2e8f0;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 1rem;
            white-space: nowrap;
            margin-bottom: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            border: 1px solid #334155;
            z-index: 1000;
        }
        .has-tooltip:hover .tooltip {
            display: block;
        }
        .legend-item {
            position: relative;
        }
        .legend-item .tooltip {
            left: auto;
            right: 0;
            white-space: normal;
            width: 280px;
        }

        /* Results section */
        .results-section {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
            overflow: hidden;
        }
        .summary-bar {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
            flex-shrink: 0;
        }
        .stat {
            background: var(--accent);
            padding: 10px 15px;
            border-radius: 8px;
        }
        .stat.wasm-stat { border: 2px solid var(--color-wasm); }
        .stat-value {
            font-size: 1.3rem;
            font-weight: bold;
            color: var(--highlight);
        }
        .stat-value.wasm { color: var(--color-wasm); }
        .stat-label {
            font-size: 1rem;
            color: var(--muted);
            text-transform: uppercase;
        }

        .answer-box {
            background: linear-gradient(135deg, #065f46, #064e3b);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 15px;
            flex-shrink: 0;
        }
        .answer-box h3 {
            color: var(--success);
            margin-bottom: 8px;
            font-size: 1rem;
        }
        .answer-box.error {
            background: linear-gradient(135deg, #7f1d1d, #450a0a);
        }
        .answer-box.error h3 { color: var(--error); }

        .results {
            display: grid;
            grid-template-columns: 250px 1fr;
            gap: 15px;
            flex: 1;
            min-height: 0;
            overflow: hidden;
        }
        .timeline {
            background: var(--card);
            border-radius: 12px;
            padding: 12px;
            overflow-y: auto;
        }
        .timeline h2 {
            font-size: 1rem;
            color: var(--muted);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .step {
            position: relative;
            padding: 10px 12px;
            margin-bottom: 6px;
            background: var(--accent);
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.2s;
            border-left: 3px solid transparent;
        }
        .step:hover { background: #1a4a7a; }
        .step.active {
            background: #1a4a7a;
            border-left-color: var(--highlight);
        }
        .step.has-error { border-left-color: var(--error); }
        .step.is-final { border-left-color: var(--success); }
        .step.has-wasm { border-left-color: var(--color-wasm); }
        .step-number {
            font-weight: bold;
            color: var(--highlight);
            font-size: 1rem;
        }
        .step-meta {
            font-size: 1rem;
            color: var(--muted);
            margin-top: 3px;
        }
        .step-badges {
            display: flex;
            gap: 4px;
            margin-top: 4px;
        }
        .badge {
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 1rem;
            font-weight: 600;
        }
        .badge.wasm { background: var(--color-wasm); color: #000; }

        .detail-panel {
            background: var(--card);
            border-radius: 12px;
            padding: 15px;
            overflow-y: auto;
        }
        .detail-section {
            margin-bottom: 15px;
        }
        .detail-section h3 {
            font-size: 1rem;
            color: var(--highlight);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .detail-section h3.wasm-title { color: var(--color-wasm); }
        .code-block {
            background: var(--bg);
            border-radius: 8px;
            padding: 12px;
            overflow-x: auto;
            font-size: 1rem;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 200px;
            overflow-y: auto;
        }
        .code-block.json { color: #7dd3fc; }
        .code-block.output { color: #a5f3fc; }
        .code-block.error { color: var(--error); }
        .code-block.rust { color: var(--color-wasm); }

        .hidden { display: none !important; }

        .wasm-info {
            background: linear-gradient(135deg, #78350f, #451a03);
            border-radius: 8px;
            padding: 10px 15px;
            margin-bottom: 15px;
            border: 1px solid var(--color-wasm);
            flex-shrink: 0;
        }
        .wasm-info h3 {
            color: var(--color-wasm);
            margin-bottom: 5px;
            font-size: 1rem;
        }
        .wasm-info p {
            color: var(--text);
            font-size: 1rem;
            line-height: 1.4;
        }

        .flow-diagram {
            display: flex;
            align-items: center;
            gap: 5px;
            margin-bottom: 15px;
            flex-wrap: wrap;
            flex-shrink: 0;
        }
        .flow-node {
            padding: 6px 10px;
            background: var(--accent);
            border-radius: 6px;
            font-size: 1rem;
        }
        .flow-node.query { background: var(--highlight); }
        .flow-node.final { background: #065f46; }
        .flow-node.wasm { background: var(--color-wasm); color: #000; }
        .flow-arrow {
            color: var(--muted);
            font-size: 1rem;
        }

        /* Waiting message */
        .waiting-message {
            text-align: center;
            color: var(--muted);
            padding: 40px;
        }
        .waiting-message h2 {
            font-size: 1.2rem;
            margin-bottom: 10px;
        }

        /* Result modal */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.2s, visibility 0.2s;
        }
        .modal-overlay.visible {
            opacity: 1;
            visibility: visible;
        }
        .modal {
            background: var(--card);
            border-radius: 12px;
            width: 90%;
            max-width: 700px;
            max-height: 80vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            transform: scale(0.95);
            transition: transform 0.2s;
        }
        .modal-overlay.visible .modal {
            transform: scale(1);
        }
        .modal-large {
            max-width: 1000px;
            max-height: 90vh;
        }
        .modal-large .results {
            grid-template-columns: 1fr 1fr;
        }
        .modal-large .answer-box {
            margin-bottom: 15px;
        }
        .modal-large .summary-bar {
            margin-bottom: 15px;
        }
        .modal-large .flow-diagram {
            margin-bottom: 15px;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            border-bottom: 1px solid var(--accent);
        }
        .modal-header h2 {
            color: var(--success);
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .modal-header h2.error { color: var(--error); }
        .modal-close {
            background: none;
            border: none;
            color: var(--muted);
            font-size: 1.5rem;
            cursor: pointer;
            padding: 0 5px;
            line-height: 1;
        }
        .modal-close:hover { color: var(--text); }
        .modal-body {
            padding: 20px;
            overflow-y: auto;
            flex: 1;
        }
        .modal-result {
            background: var(--bg);
            border-radius: 8px;
            padding: 15px;
            font-size: 1rem;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 50vh;
            overflow-y: auto;
        }
        .modal-stats {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        .modal-stat {
            background: var(--accent);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 1rem;
        }
        .modal-stat-value {
            color: var(--highlight);
            font-weight: bold;
        }
        .modal-stat-label {
            color: var(--muted);
            font-size: 1rem;
        }
        .modal-footer {
            padding: 15px 20px;
            border-top: 1px solid var(--accent);
            display: flex;
            justify-content: flex-end;
        }
        .modal-footer button {
            padding: 8px 20px;
            font-size: 1rem;
        }

        /* Show Result button */
        .btn-secondary {
            background: var(--accent);
            margin-right: 10px;
        }
        .btn-secondary:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }
        .btn-secondary:not(:disabled):hover {
            background: #1a4a7a;
        }

        /* Settings panel */
        .settings-btn {
            background: var(--accent);
            padding: 10px 15px;
            margin-right: 10px;
            font-size: 1.2rem;
        }
        .settings-btn:hover {
            background: #1a4a7a;
        }
        .settings-panel {
            position: absolute;
            top: 60px;
            right: 30px;
            background: var(--card);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
            z-index: 100;
            display: none;
            min-width: 280px;
        }
        .settings-panel.visible {
            display: block;
        }
        .settings-panel h3 {
            margin-bottom: 15px;
            color: var(--highlight);
            font-size: 1.1rem;
        }
        .setting-row {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 10px;
        }
        .setting-row label {
            flex: 1;
            color: var(--text);
        }
        .setting-row input[type="range"] {
            flex: 2;
            accent-color: var(--highlight);
        }
        .setting-value {
            min-width: 50px;
            text-align: right;
            color: var(--highlight);
            font-weight: bold;
        }

        @media (max-width: 900px) {
            .results { grid-template-columns: 1fr; }
            .input-row { flex-direction: column; }
            .query-col { width: 100%; }
            .container { padding: 10px; }
            .modal { width: 95%; max-height: 90vh; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header with title and buttons -->
        <div class="header">
            <h1><span>🔄</span> RLM Visualizer</h1>
            <div>
                <button class="settings-btn" onclick="toggleSettings()" title="Settings">⚙️</button>
                <button id="runBtn" onclick="runQuery()">Run RLM Query</button>
                <button id="showResultBtn" class="btn-secondary" onclick="showResultModal()" disabled>Show Result</button>
            </div>
        </div>

        <!-- Settings Panel -->
        <div id="settingsPanel" class="settings-panel">
            <h3>⚙️ Display Settings</h3>
            <div class="setting-row">
                <label>Font Size</label>
                <input type="range" id="fontSizeSlider" min="16" max="40" value="20" oninput="updateFontSize(this.value)">
                <span id="fontSizeValue" class="setting-value">20px</span>
            </div>
        </div>

        <!-- Tab bar -->
        <div class="tab-bar">
            <button class="tab active" onclick="switchTab('input')">Input Data</button>
            <button class="tab" onclick="switchTab('output')">Output</button>
        </div>

        <!-- Input Data Tab -->
        <div id="inputTab" class="tab-content active">
            <div class="input-section">
                <div class="example-selector">
                    <label for="exampleSelect">Load Example</label>
                    <select id="exampleSelect" onchange="loadExample()">
                        <option value="">-- Select an example --</option>
                        <optgroup label="Level 1: DSL (Easy - Text Operations)">
                            <option value="count_errors" selected>Count ERROR lines</option>
                            <option value="find_pattern">Find text pattern</option>
                            <option value="dsl_slice_lines">Extract line range</option>
                        </optgroup>
                        <optgroup label="Level 2: WASM (Medium - Computation)">
                            <option value="wasm_unique_ips">Count unique IP addresses</option>
                            <option value="wasm_error_ranking">Rank errors by frequency</option>
                            <option value="wasm_word_freq">Word frequency analysis</option>
                            <option value="wasm_response_times">Response time percentiles</option>
                        </optgroup>
                        <optgroup label="Level 2: WASM (Large Context)">
                            <option value="large_logs_errors">Large Logs: Error Ranking (5000 lines)</option>
                            <option value="large_logs_ips">Large Logs: Unique IPs (5000 lines)</option>
                        </optgroup>
                        <optgroup label="Combined (Advanced - Not Yet Implemented)">
                            <option value="war_peace_family">War and Peace: Family Tree (3.2MB)</option>
                        </optgroup>
                    </select>
                    <div class="example-tags" id="exampleTags"></div>

                    <!-- Example info panel -->
                    <div id="exampleInfo" class="example-info visible">
                        <div class="example-info-row">
                            <div class="example-info-item">
                                <div class="example-info-label has-tooltip">
                                    Problem Type
                                    <span class="tooltip">RLM Paper Problem Types</span>
                                </div>
                                <div id="infoBenchmark" class="example-info-value benchmark has-tooltip">
                                    Simple NIAH
                                    <span id="infoBenchmarkTooltip" class="tooltip">NIAH = Needle-in-a-Haystack: Find specific information in large input data</span>
                                </div>
                            </div>
                            <div class="example-info-item">
                                <div class="example-info-label has-tooltip">
                                    Capability Level
                                    <span class="tooltip">L1=DSL text ops, L2=WASM sandboxed, L3=CLI native binary, L4=LLM delegation</span>
                                </div>
                                <div id="infoLevel" class="example-info-value level has-tooltip">
                                    Level 1 (DSL)
                                    <span id="infoLevelTooltip" class="tooltip">DSL: slice, lines, regex, find, count, split, len - fast text operations</span>
                                </div>
                            </div>
                            <div class="example-info-item">
                                <div class="example-info-label has-tooltip">
                                    Max Iterations
                                    <span class="tooltip">Maximum LLM rounds before timeout. DSL tasks: 3-5, WASM: 10, Large input: 15-20</span>
                                </div>
                                <div id="infoMaxIter" class="example-info-value" style="color: #fbbf24;">
                                    5
                                </div>
                            </div>
                            <div class="example-info-item">
                                <div class="example-info-label has-tooltip">
                                    Root LLM
                                    <span class="tooltip">The primary LLM used for orchestration and decision-making</span>
                                </div>
                                <div id="infoRootLlm" class="example-info-value" style="color: #1f77b4;">
                                    {root_provider}
                                </div>
                            </div>
                        </div>
                        <div id="infoDesc" class="example-info-desc">Simple NIAH: Pattern counting. Uses regex/find commands for deterministic O(n) search.</div>
                    </div>
                </div>

                <div class="input-row">
                    <div class="query-col">
                        <label for="query">Query</label>
                        <textarea id="query" placeholder="What do you want to know about the data?">How many ERROR lines are there? Use only DSL commands (regex, find, count, lines).</textarea>
                    </div>
                </div>

                <div class="data-container">
                    <div class="data-label">
                        <label for="context">Input Data</label>
                        <span id="contextStats" class="context-stats"></span>
                    </div>
                    <textarea id="context" placeholder="Paste your text/code/logs here..." oninput="updateContextStats()">Line 1: INFO - System started
Line 2: ERROR - Connection failed
Line 3: INFO - Retrying connection
Line 4: ERROR - Timeout occurred
Line 5: WARNING - High memory usage
Line 6: INFO - Connection established
Line 7: ERROR - Invalid input received</textarea>
                </div>
            </div>
        </div>

        <!-- Output Tab -->
        <div id="outputTab" class="tab-content">
            <div class="output-section">
                <!-- Progress section (always visible during run) -->
                <div id="progressSection" class="progress-section expanded">
                    <div class="progress-header">
                        <h2>Live Progress</h2>
                        <span id="progressQuery" class="progress-query"></span>
                        <span id="progressStatus" class="progress-status">Waiting...</span>
                    </div>
                    <div id="progressLog" class="progress-log">
                        <div class="waiting-message">
                            <h2>Ready to run</h2>
                            <p>Click "Run RLM Query" to start processing</p>
                        </div>
                    </div>
                    <!-- Color legend (positioned at bottom of visible area) -->
                    <div class="progress-legend">
                        <div class="legend-item llm has-tooltip">
                            <span class="legend-dot llm"></span>
                            <span class="legend-label">LLM</span>
                            <span class="tooltip">Root LLM: Orchestration calls to the main language model</span>
                        </div>
                        <div class="legend-item dsl has-tooltip">
                            <span class="legend-dot cmd"></span>
                            <span class="legend-label">L1 DSL</span>
                            <span class="tooltip">Level 1 DSL: Text commands like slice, lines, regex, find, count</span>
                        </div>
                        <div class="legend-item wasm has-tooltip">
                            <span class="legend-dot wasm"></span>
                            <span class="legend-label">L2 WASM</span>
                            <span class="tooltip">Level 2 WASM: Sandboxed Rust code execution via WebAssembly</span>
                        </div>
                        <div class="legend-item cli has-tooltip">
                            <span class="legend-dot cli"></span>
                            <span class="legend-label">L3 CLI</span>
                            <span class="tooltip">Level 3 CLI: Native Rust binary execution (no sandbox)</span>
                        </div>
                        <div class="legend-item llm4 has-tooltip">
                            <span class="legend-dot llm4"></span>
                            <span class="legend-label">L4 LLM</span>
                            <span class="tooltip">Level 4 LLM: Delegated calls to sub-LLMs for chunk analysis</span>
                        </div>
                        <div class="legend-item done has-tooltip">
                            <span class="legend-dot done"></span>
                            <span class="legend-label">Done</span>
                            <span class="tooltip">Query iteration or final answer completed successfully</span>
                        </div>
                        <div class="legend-item error has-tooltip">
                            <span class="legend-dot error"></span>
                            <span class="legend-label">Error</span>
                            <span class="tooltip">An error occurred during command execution</span>
                        </div>
                    </div>
                </div>

                <!-- Results now shown in modal popup -->
            </div>
        </div>
    </div>

    <!-- Result Modal -->
    <div id="resultModal" class="modal-overlay" onclick="hideResultModal(event)">
        <div class="modal modal-large" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h2 id="modalTitle">✅ Result</h2>
                <button class="modal-close" onclick="hideResultModal()">&times;</button>
            </div>
            <div class="modal-body">
                <!-- Answer -->
                <div id="modalAnswerBox" class="answer-box">
                    <h3>Final Answer</h3>
                    <div id="modalAnswerText"></div>
                </div>

                <!-- Stats bar -->
                <div id="modalSummaryBar" class="summary-bar">
                    <div class="stat">
                        <div class="stat-value" id="modalStatIterations">-</div>
                        <div class="stat-label">Iterations</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="modalStatSubCalls">-</div>
                        <div class="stat-label">Sub-LM Calls</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="modalStatContext">-</div>
                        <div class="stat-label">Input Chars</div>
                    </div>
                    <div class="stat wasm-stat" id="modalStatWasm" style="display: none;">
                        <div class="stat-value wasm">-</div>
                        <div class="stat-label">WASM Steps</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="modalStatTokens">-</div>
                        <div class="stat-label">Tokens</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="modalStatDuration">-</div>
                        <div class="stat-label">Duration</div>
                    </div>
                </div>

                <!-- WASM info -->
                <div id="modalWasmInfo" class="wasm-info hidden">
                    <h3>🔧 WASM Execution Used</h3>
                    <p></p>
                </div>

                <!-- Flow diagram -->
                <div id="modalFlowDiagram" class="flow-diagram"></div>

                <!-- Timeline and details -->
                <div class="results">
                    <div class="timeline">
                        <h2>Timeline</h2>
                        <div id="modalTimeline"></div>
                    </div>
                    <div class="detail-panel">
                        <div id="modalDetailContent">
                            <p style="color: var(--muted);">Click on a step to see details</p>
                        </div>
                    </div>
                </div>
            </div>
            <div class="modal-footer">
                <button onclick="hideResultModal()">Close</button>
            </div>
        </div>
    </div>

    <script>
        let currentData = null;
        let eventSource = null;
        let startTime = null;

        // Initialize on load
        document.addEventListener('DOMContentLoaded', () => {
            // Load saved font size
            const savedFontSize = localStorage.getItem('rlmFontSize') || '20';
            updateFontSize(savedFontSize);
            document.getElementById('fontSizeSlider').value = savedFontSize;

            // Load default example (count_errors is selected by default)
            document.getElementById('exampleSelect').value = 'count_errors';
            loadExample();

            // Add Esc key listener to close modal and settings
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    hideResultModal();
                    document.getElementById('settingsPanel').classList.remove('visible');
                }
            });
            // Close settings when clicking outside
            document.addEventListener('click', (e) => {
                const panel = document.getElementById('settingsPanel');
                const btn = e.target.closest('.settings-btn');
                if (!btn && !panel.contains(e.target)) {
                    panel.classList.remove('visible');
                }
            });
        });

        // Settings functions
        function toggleSettings() {
            document.getElementById('settingsPanel').classList.toggle('visible');
        }

        function updateFontSize(size) {
            document.documentElement.style.fontSize = size + 'px';
            document.getElementById('fontSizeValue').textContent = size + 'px';
            localStorage.setItem('rlmFontSize', size);
        }

        // Modal functions
        function showResultModal() {
            if (!currentData) return;

            const data = currentData;
            const modal = document.getElementById('resultModal');
            const title = document.getElementById('modalTitle');

            // Set title based on success/failure
            if (data.success) {
                title.textContent = '✅ Result';
                title.className = '';
            } else {
                title.textContent = '❌ Error';
                title.className = 'error';
            }

            // Answer box
            const answerBox = document.getElementById('modalAnswerBox');
            answerBox.className = 'answer-box' + (data.success ? '' : ' error');
            document.getElementById('modalAnswerText').textContent = data.success ? data.answer : (data.error || 'Failed');

            // Check for WASM usage
            const wasmSteps = data.history.filter(s => hasWasm(s.commands));
            const usedWasm = wasmSteps.length > 0;

            // Stats
            document.getElementById('modalStatIterations').textContent = data.iterations;
            document.getElementById('modalStatSubCalls').textContent = data.total_sub_calls;
            document.getElementById('modalStatContext').textContent = data.context_length.toLocaleString();

            const wasmStatEl = document.getElementById('modalStatWasm');
            if (wasmStatEl) {
                wasmStatEl.querySelector('.stat-value').textContent = wasmSteps.length;
                wasmStatEl.style.display = usedWasm ? 'block' : 'none';
            }

            const totalTokens = (data.total_prompt_tokens || 0) + (data.total_completion_tokens || 0);
            document.getElementById('modalStatTokens').textContent = totalTokens.toLocaleString();
            document.getElementById('modalStatTokens').title = `${(data.total_prompt_tokens || 0).toLocaleString()} prompt + ${(data.total_completion_tokens || 0).toLocaleString()} completion`;

            if (data.total_duration_ms) {
                const durationSec = (data.total_duration_ms / 1000).toFixed(1);
                document.getElementById('modalStatDuration').textContent = durationSec + 's';
            }

            // WASM info
            const wasmInfoEl = document.getElementById('modalWasmInfo');
            if (wasmInfoEl) {
                if (usedWasm) {
                    wasmInfoEl.classList.remove('hidden');
                    wasmInfoEl.querySelector('p').textContent =
                        `This query used rust_wasm in ${wasmSteps.length} step(s) to perform custom analysis.`;
                } else {
                    wasmInfoEl.classList.add('hidden');
                }
            }

            // Flow diagram
            const flow = document.getElementById('modalFlowDiagram');
            flow.innerHTML = '<div class="flow-node query">Query</div>';
            data.history.forEach((step, i) => {
                flow.innerHTML += '<span class="flow-arrow">→</span>';
                const isFinal = step.output.startsWith('FINAL:');
                const isWasm = hasWasm(step.commands);
                let nodeClass = isWasm ? ' wasm' : (isFinal ? ' final' : '');
                let icon = step.error ? '⚠️' : (isWasm ? '🔧' : '');
                flow.innerHTML += `<div class="flow-node${nodeClass}">${icon}Step ${step.step}</div>`;
            });
            if (data.success) {
                flow.innerHTML += '<span class="flow-arrow">→</span><div class="flow-node final">Answer</div>';
            }

            // Timeline
            const timeline = document.getElementById('modalTimeline');
            timeline.innerHTML = data.history.map((step, i) => {
                const hasError = !!step.error;
                const isFinal = step.output.startsWith('FINAL:');
                const isWasm = hasWasm(step.commands);
                let stepClass = hasError ? ' has-error' : (isWasm ? ' has-wasm' : (isFinal ? ' is-final' : ''));

                let badges = '';
                if (isWasm) {
                    badges = '<div class="step-badges"><span class="badge wasm">WASM</span></div>';
                }

                return `<div class="step${stepClass}" onclick="showModalStep(${i})">
                    <div class="step-number">Step ${step.step}</div>
                    <div class="step-meta">
                        ${step.commands ? formatCommandTypes(step.commands) : 'No commands'}
                        ${step.sub_calls > 0 ? ` • ${step.sub_calls} sub-calls` : ''}
                        ${hasError ? ' • Error' : ''}
                    </div>
                    ${badges}
                </div>`;
            }).join('');

            // Show first step
            if (data.history.length > 0) {
                showModalStep(0);
            }

            modal.classList.add('visible');
        }

        function showModalStep(index) {
            const step = currentData.history[index];

            // Update active state in modal timeline
            document.querySelectorAll('#modalTimeline .step').forEach((el, i) => {
                el.classList.toggle('active', i === index);
            });

            // Render detail
            let html = '';
            const isWasm = hasWasm(step.commands);
            const rustCode = extractRustCode(step.commands);

            if (isWasm && rustCode) {
                html += `<div class="detail-section">
                    <h3 class="wasm-title">🔧 Rust Code (rust_wasm)</h3>
                    <pre class="code-block rust">${escapeHtml(rustCode)}</pre>
                </div>`;
            }

            html += `<div class="detail-section">
                <h3>LLM Response</h3>
                <pre>${escapeHtml(step.llm_response)}</pre>
            </div>`;

            if (step.commands) {
                html += `<div class="detail-section">
                    <h3>Commands</h3>
                    <pre class="code-block">${escapeHtml(step.commands)}</pre>
                </div>`;
            }

            html += `<div class="detail-section">
                <h3>Output</h3>
                <pre>${escapeHtml(step.output)}</pre>
            </div>`;

            if (step.error) {
                html += `<div class="detail-section">
                    <h3 style="color: var(--error)">Error</h3>
                    <pre style="color: var(--error)">${escapeHtml(step.error)}</pre>
                </div>`;
            }

            document.getElementById('modalDetailContent').innerHTML = html;
        }

        function hideResultModal(event) {
            // If called from background click, only close if clicking the overlay itself
            if (event && event.target !== event.currentTarget) return;
            document.getElementById('resultModal').classList.remove('visible');
        }

        // Tab switching
        function switchTab(tabName) {
            // Update tab buttons
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`.tab:nth-child(${tabName === 'input' ? 1 : 2})`).classList.add('active');

            // Update tab content
            document.getElementById('inputTab').classList.toggle('active', tabName === 'input');
            document.getElementById('outputTab').classList.toggle('active', tabName === 'output');
        }

        function updateContextStats() {
            const context = document.getElementById('context').value;
            const lines = context.split('\n');
            const lineCount = lines.length;
            const charCount = context.length;
            const sizeKb = (charCount / 1024).toFixed(1);

            // Update stats
            document.getElementById('contextStats').textContent = `${lineCount.toLocaleString()} lines, ${charCount.toLocaleString()} chars (${sizeKb}KB)`;
        }

        function logProgress(icon, message, className = '') {
            const log = document.getElementById('progressLog');
            const elapsed = startTime ? ((Date.now() - startTime) / 1000).toFixed(1) + 's' : '0.0s';
            const event = document.createElement('div');
            event.className = 'event';
            event.innerHTML = `<span class="event-time">${elapsed}</span><span class="event-icon">${icon}</span><span class="${className}">${escapeHtml(message)}</span>`;
            log.appendChild(event);
            log.scrollTop = log.scrollHeight;
        }

        function clearProgress() {
            document.getElementById('progressLog').innerHTML = '';
            document.getElementById('progressStatus').textContent = 'Connecting...';
            document.getElementById('progressStatus').className = 'progress-status active';
            // Clear any existing wait timer
            if (window.llmWaitTimer) {
                clearInterval(window.llmWaitTimer);
                window.llmWaitTimer = null;
            }
        }

        // Example data for the dropdown with paper benchmark categories
        const examples = {
            // ========================================
            // LEVEL 1: DSL (Easy - Text Operations)
            // ========================================
            count_errors: {
                query: "How many ERROR lines are there? Use only DSL commands (regex, find, count, lines).",
                context: generateLogContext(100),
                tags: ['dsl', 'count'],
                benchmark: 'Simple NIAH',
                level: 'Level 1 (DSL)',
                maxIterations: 5,
                description: 'Simple NIAH: Pattern counting. Uses regex/find commands for deterministic O(n) search.'
            },
            find_pattern: {
                query: "Find all lines containing 'AuthenticationFailed'. Use only DSL commands (find, regex, lines).",
                context: generateLogContext(100),
                tags: ['dsl', 'search'],
                benchmark: 'Simple NIAH',
                level: 'Level 1 (DSL)',
                maxIterations: 5,
                description: 'Simple NIAH: Pattern search. Uses find command for exact string matching.'
            },
            dsl_slice_lines: {
                query: "Extract lines 10 through 20 from this data. Use only DSL commands (slice, lines).",
                context: generateLogContext(50),
                tags: ['dsl', 'slice'],
                benchmark: 'Simple NIAH',
                level: 'Level 1 (DSL)',
                maxIterations: 3,
                description: 'Simple NIAH: Line extraction. Uses slice/lines commands for O(1) range access.'
            },

            // ========================================
            // LEVEL 2: WASM (Medium - Computation)
            // ========================================
            wasm_unique_ips: {
                query: "How many unique IP addresses are in these logs?",
                context: generateLogContext(200),
                tags: ['wasm', 'aggregation'],
                benchmark: 'OOLONG',
                level: 'Level 2 (WASM)',
                maxIterations: 10,
                description: 'OOLONG: Count unique items. Uses rust_wasm_intent with HashSet for O(1) uniqueness.'
            },
            wasm_error_ranking: {
                query: "Rank the error types from most to least frequent",
                context: generateErrorLogs(150),
                tags: ['wasm', 'aggregation', 'ranking'],
                benchmark: 'OOLONG',
                level: 'Level 2 (WASM)',
                maxIterations: 10,
                description: 'OOLONG: Frequency ranking. Uses rust_wasm_mapreduce for map (extract) then reduce (count+sort).'
            },
            wasm_word_freq: {
                query: "What are the top 10 most common words in this text?",
                context: generateTextSample(),
                tags: ['wasm', 'aggregation', 'text-analysis'],
                benchmark: 'OOLONG',
                level: 'Level 2 (WASM)',
                maxIterations: 10,
                description: 'OOLONG: Word frequency analysis. Uses rust_wasm with HashMap for counting, then sorting.'
            },
            wasm_response_times: {
                query: "Calculate the p50, p95, and p99 response time percentiles",
                context: generateResponseTimes(300),
                tags: ['wasm', 'statistics', 'percentiles'],
                benchmark: 'OOLONG',
                level: 'Level 2 (WASM)',
                maxIterations: 10,
                description: 'OOLONG: Statistical computation. Uses rust_wasm to parse, sort, and compute percentiles.'
            },

            // ========================================
            // LEVEL 2: WASM (Large Context)
            // ========================================
            large_logs_errors: {
                query: "Rank the error types from most to least frequent. Show the count for each error type.",
                context: null,
                loadUrl: '/samples/large-logs',
                tags: ['wasm', 'large-context', 'aggregation'],
                benchmark: 'BrowseComp-Plus',
                level: 'Level 2 (WASM)',
                maxIterations: 15,
                description: 'BrowseComp-Plus: Aggregate error types from 5000 log lines. Uses rust_wasm_mapreduce.'
            },
            large_logs_ips: {
                query: "How many unique IP addresses appear in these logs? List the top 10 most active IPs.",
                context: null,
                loadUrl: '/samples/large-logs',
                tags: ['wasm', 'large-context', 'aggregation'],
                benchmark: 'BrowseComp-Plus',
                level: 'Level 2 (WASM)',
                maxIterations: 15,
                description: 'BrowseComp-Plus: Extract and rank unique IPs. Uses rust_wasm with HashSet/HashMap.'
            },

            // ========================================
            // COMBINED (Advanced - Not Yet Implemented)
            // ========================================
            war_peace_family: {
                query: "Build a family tree for the main characters. Identify characters who appear multiple times and are related to each other (by blood or marriage). Show the relationships in a structured format.",
                context: null,
                loadUrl: '/samples/war-and-peace',
                tags: ['combined', 'large-context', 'synthesis'],
                benchmark: 'S-NIAH',
                level: 'Level 1 + Level 2',
                maxIterations: 20,
                description: 'S-NIAH: Find scattered character mentions in 3.2MB text. NOT YET IMPLEMENTED.'
            }
        };

        // Generate log context with N lines
        function generateLogContext(n) {
            const lines = [];
            const ips = ['192.168.1.100', '10.0.0.50', '172.16.0.25', '10.0.0.75', '192.168.1.200', '172.16.0.30', '10.0.0.60', '192.168.1.105'];
            const endpoints = ['/api/users', '/api/data', '/api/health', '/api/products', '/api/orders'];
            const errors = ['AuthenticationFailed', 'ConnectionTimeout', 'RequestFailed', 'ValidationError'];
            for (let i = 0; i < n; i++) {
                const h = 10 + Math.floor(i / 60) % 14;
                const m = i % 60;
                const s = (i * 7) % 60;
                const ip = ips[i % ips.length];
                const ep = endpoints[i % endpoints.length];
                if (i % 5 === 0) {
                    const err = errors[i % errors.length];
                    lines.push(`2024-01-15 ${h}:${m.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')} [ERROR] ${err} from ${ip} - Request to ${ep} failed`);
                } else {
                    const ms = 10 + (i * 13) % 500;
                    lines.push(`2024-01-15 ${h}:${m.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')} [INFO] Request from ${ip} - GET ${ep} - 200 OK - ${ms}ms`);
                }
            }
            return lines.join('\n');
        }

        // Generate error-only logs
        function generateErrorLogs(n) {
            const lines = [];
            const ips = ['192.168.1.100', '10.0.0.50', '172.16.0.25', '10.0.0.75', '192.168.1.200'];
            const errors = ['AuthenticationFailed', 'ConnectionTimeout', 'RequestFailed', 'ValidationError', 'DatabaseError', 'RateLimited'];
            const weights = [4, 3, 2, 2, 1, 1];  // AuthenticationFailed most common
            const weighted = errors.flatMap((e, i) => Array(weights[i]).fill(e));
            for (let i = 0; i < n; i++) {
                const h = 10 + Math.floor(i / 60) % 14;
                const m = i % 60;
                const s = (i * 7) % 60;
                const ip = ips[i % ips.length];
                const err = weighted[(i * 7) % weighted.length];
                lines.push(`2024-01-15 ${h}:${m.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')} [${err}] from ${ip} - Operation failed`);
            }
            return lines.join('\n');
        }

        // Generate text sample for word frequency
        function generateTextSample() {
            const sentences = [
                'The quick brown fox jumps over the lazy dog.',
                'A journey of a thousand miles begins with a single step.',
                'To be or not to be, that is the question.',
                'All that glitters is not gold.',
                'The only thing we have to fear is fear itself.',
                'In the beginning was the word, and the word was with God.',
                'It was the best of times, it was the worst of times.',
                'Call me Ishmael.',
                'The quick fox ran across the field.',
                'Happy families are all alike; every unhappy family is unhappy in its own way.'
            ];
            const result = [];
            for (let i = 0; i < 50; i++) {
                result.push(sentences[i % sentences.length]);
            }
            return result.join(' ');
        }

        // Generate response time logs
        function generateResponseTimes(n) {
            const lines = [];
            const endpoints = ['/api/users', '/api/data', '/api/health', '/api/products', '/api/orders', '/api/batch', '/api/upload'];
            const methods = ['GET', 'POST', 'PUT', 'DELETE'];
            for (let i = 0; i < n; i++) {
                const ep = endpoints[i % endpoints.length];
                const method = methods[i % methods.length];
                // Realistic distribution: mostly fast, some slow outliers
                const base = 20 + (i * 7) % 100;
                const ms = i % 20 === 0 ? base * 10 : (i % 7 === 0 ? base * 3 : base);
                lines.push(`${method} ${ep} - ${ms}ms`);
            }
            return lines.join('\n');
        }

        async function loadExample() {
            const select = document.getElementById('exampleSelect');
            const example = examples[select.value];
            if (example) {
                document.getElementById('query').value = example.query;

                // Show tags with benchmark and level info
                const tagsHtml = example.tags.map(tag => {
                    let tagClass = 'basic';
                    if (tag === 'dsl') tagClass = 'dsl';
                    else if (tag === 'wasm') tagClass = 'wasm';
                    else if (tag === 'combined') tagClass = 'combined';
                    else if (tag === 'large-context') tagClass = 'large-context';
                    else if (tag === 'aggregation' || tag === 'ranking') tagClass = 'aggregation';
                    return `<span class="tag ${tagClass}">${tag}</span>`;
                }).join('');

                document.getElementById('exampleTags').innerHTML = tagsHtml;

                // Show example info panel
                const infoPanel = document.getElementById('exampleInfo');
                infoPanel.classList.add('visible');

                // Set benchmark with appropriate tooltip
                const benchmark = example.benchmark || 'General';
                const benchmarkEl = document.getElementById('infoBenchmark');
                const tooltipEl = document.getElementById('infoBenchmarkTooltip');

                // Update text (keep first text node, tooltip is separate)
                benchmarkEl.childNodes[0].textContent = benchmark + ' ';

                // Set tooltip based on benchmark type
                const benchmarkTooltips = {
                    'Simple NIAH': 'NIAH = Needle-in-a-Haystack: Find specific information in large input data',
                    'S-NIAH': 'S-NIAH = Semantic Needle-in-a-Haystack: Find meaning-based patterns',
                    'OOLONG': 'OOLONG = Long-context reasoning benchmark requiring multi-step analysis',
                    'BrowseComp-Plus': 'BrowseComp-Plus = Web data extraction and structured analysis',
                    'General': 'General purpose query without specific benchmark category'
                };
                tooltipEl.textContent = benchmarkTooltips[benchmark] || benchmark;

                // Set level with appropriate tooltip
                const level = example.level || 'Level 1 (DSL)';
                const levelEl = document.getElementById('infoLevel');
                const levelTooltipEl = document.getElementById('infoLevelTooltip');

                // Update text (keep first text node, tooltip is separate)
                levelEl.childNodes[0].textContent = level + ' ';

                // Set tooltip based on level
                const levelTooltips = {
                    'Level 1 (DSL)': 'DSL: slice, lines, regex, find, count, split, len, set, get, print - fast text operations on input data',
                    'Level 2 (WASM)': 'WASM: Sandboxed Rust code with fuel+memory limits. rust_wasm_mapreduce for statistics, sorting, aggregation',
                    'Level 3 (CLI)': 'CLI: Full Rust stdlib access. Binary data, file I/O, network access, external libraries. Process isolation only',
                    'Level 4 (LLM)': 'LLM Delegation: Chunk data to specialized sub-LLMs for summarization, entity extraction, semantic analysis'
                };
                levelTooltipEl.textContent = levelTooltips[level] || level;

                // Set max iterations
                document.getElementById('infoMaxIter').textContent = example.maxIterations || 10;

                document.getElementById('infoDesc').textContent = example.description || '';

                // Load context - either from URL or use static value
                if (example.loadUrl) {
                    document.getElementById('context').value = 'Loading large context...';
                    updateContextStats();
                    try {
                        const response = await fetch(example.loadUrl);
                        if (!response.ok) throw new Error(`HTTP ${response.status}`);
                        const text = await response.text();
                        document.getElementById('context').value = text;
                        updateContextStats();
                    } catch (err) {
                        document.getElementById('context').value = `Error loading: ${err.message}`;
                        updateContextStats();
                    }
                } else {
                    document.getElementById('context').value = example.context;
                    updateContextStats();
                }
            } else {
                document.getElementById('exampleTags').innerHTML = '';
                document.getElementById('exampleInfo').classList.remove('visible');
            }
        }

        async function runQuery() {
            const query = document.getElementById('query').value;
            const context = document.getElementById('context').value;
            const btn = document.getElementById('runBtn');

            // Cancel any existing connection
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }

            btn.disabled = true;
            btn.textContent = 'Running...';
            startTime = Date.now();

            // Disable Show Result button and clear old data
            document.getElementById('showResultBtn').disabled = true;
            currentData = null;

            // Switch to Output tab
            switchTab('output');

            // Show progress section expanded
            document.getElementById('progressSection').classList.remove('hidden');
            document.getElementById('progressSection').classList.add('expanded');
            clearProgress();

            // Show query preview in progress header
            const queryPreview = query.length > 80 ? query.substring(0, 80) + '...' : query;
            document.getElementById('progressQuery').textContent = '"' + queryPreview + '"';

            // Build the history as we receive events
            const history = [];
            let currentStep = null;

            try {
                // Use fetch to POST and get SSE stream
                const response = await fetch('/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, context })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const jsonStr = line.slice(6);
                            if (jsonStr.trim()) {
                                try {
                                    const event = JSON.parse(jsonStr);
                                    handleStreamEvent(event, history);
                                } catch (e) {
                                    console.error('Parse error:', e, jsonStr);
                                }
                            }
                        }
                    }
                }

                document.getElementById('progressStatus').textContent = 'Complete';
                document.getElementById('progressStatus').className = 'progress-status';

            } catch (err) {
                logProgress('❌', `Error: ${err.message}`, 'event-error');
                document.getElementById('progressStatus').textContent = 'Failed';
                alert('Error: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Run RLM Query';
            }
        }

        function handleStreamEvent(event, history) {
            const type = event.type;

            switch (type) {
                case 'iteration_start':
                    logProgress('🔄', `Starting iteration ${event.step}`, 'event-llm');
                    break;
                case 'llm_start':
                    logProgress('⏳', `Calling LLM... (large input may take minutes)`, 'event-llm');
                    // Start a timer to show we're still waiting
                    if (!window.llmWaitTimer) {
                        window.llmWaitTimer = setInterval(() => {
                            const elapsed = startTime ? Math.floor((Date.now() - startTime) / 1000) : 0;
                            document.getElementById('progressStatus').textContent = `Waiting for LLM... ${elapsed}s`;
                        }, 1000);
                    }
                    break;
                case 'query_start':
                    const sizeKb = (event.context_chars / 1024).toFixed(1);
                    logProgress('📊', `Input: ${event.context_chars.toLocaleString()} chars (${sizeKb}KB), Query: ${event.query_len} chars`, 'event-info');
                    break;
                case 'llm_complete':
                    // Clear the wait timer
                    if (window.llmWaitTimer) {
                        clearInterval(window.llmWaitTimer);
                        window.llmWaitTimer = null;
                    }
                    document.getElementById('progressStatus').textContent = 'Processing...';
                    logProgress('✓', `LLM responded (${event.duration_ms}ms, ${event.prompt_tokens}p + ${event.completion_tokens}c tokens)`, 'event-llm');
                    break;
                case 'commands':
                    const cmdTypes = formatCommandTypes(event.commands);
                    const isWasmCmd = event.commands.includes('rust_wasm');
                    const isCliCmd = event.commands.includes('rust_cli_intent');
                    const cssClass = isCliCmd ? 'event-cli' : (isWasmCmd ? 'event-wasm' : 'event-cmd');
                    logProgress('▶', `Executing: ${cmdTypes}`, cssClass);
                    break;
                case 'wasm_compile_start':
                    logProgress('🔧', `Compiling WASM...`, 'event-wasm');
                    break;
                case 'wasm_compile_complete':
                    logProgress('✓', `WASM compiled (${event.duration_ms}ms)`, 'event-wasm');
                    break;
                case 'wasm_run_complete':
                    logProgress('⚡', `WASM executed (${event.duration_ms}ms)`, 'event-wasm');
                    break;
                case 'cli_compile_start':
                    logProgress('🔧', `Compiling CLI binary...`, 'event-cli');
                    break;
                case 'cli_compile_complete':
                    logProgress('✓', `CLI compiled (${event.duration_ms}ms)`, 'event-cli');
                    break;
                case 'cli_run_complete':
                    logProgress('⚡', `CLI executed (${event.duration_ms}ms)`, 'event-cli');
                    break;
                case 'command_complete':
                    logProgress('◀', `Output (${event.exec_ms}ms): ${event.output_preview.substring(0, 50)}...`, 'event-cmd');
                    break;
                case 'iteration_complete':
                    history.push({
                        step: event.step,
                        llm_response: event.llm_response,
                        commands: event.commands,
                        output: event.output,
                        error: event.error,
                        sub_calls: 0,
                        prompt_tokens: event.prompt_tokens || 0,
                        completion_tokens: event.completion_tokens || 0
                    });
                    break;
                case 'final_answer':
                    logProgress('✅', `Final answer received`, 'event-done');
                    break;
                case 'complete':
                    const durationSec = (event.total_duration_ms / 1000).toFixed(1);
                    logProgress('🏁', `Complete: ${event.iterations} iterations, ${durationSec}s, ${event.success ? 'success' : 'failed'}`, 'event-done');

                    // Build the final data object and render results
                    currentData = {
                        success: event.success,
                        answer: event.answer,
                        error: event.error,
                        iterations: event.iterations,
                        total_sub_calls: event.total_sub_calls,
                        context_length: event.context_length,
                        total_prompt_tokens: event.total_prompt_tokens,
                        total_completion_tokens: event.total_completion_tokens,
                        total_duration_ms: event.total_duration_ms,
                        history: history
                    };
                    renderResults(currentData);
                    break;
                case 'error':
                    logProgress('❌', `Error: ${event.message}`, 'event-error');
                    break;
            }
        }

        function hasWasm(commands) {
            if (!commands) return false;
            return commands.includes('rust_wasm') || commands.includes('wasm_wat') || commands.includes('"op": "wasm"');
        }

        function getCommandTypes(commands) {
            if (!commands) return [];
            const types = [];
            // DSL commands
            if (commands.includes('"slice"') || commands.includes('"op": "slice"')) types.push('slice');
            if (commands.includes('"lines"') || commands.includes('"op": "lines"')) types.push('lines');
            if (commands.includes('"regex"') || commands.includes('"op": "regex"')) types.push('regex');
            if (commands.includes('"find"') || commands.includes('"op": "find"')) types.push('find');
            if (commands.includes('"count"') || commands.includes('"op": "count"')) types.push('count');
            if (commands.includes('"split"') || commands.includes('"op": "split"')) types.push('split');
            if (commands.includes('"len"') || commands.includes('"op": "len"')) types.push('len');
            if (commands.includes('"set"') || commands.includes('"op": "set"')) types.push('set');
            if (commands.includes('"get"') || commands.includes('"op": "get"')) types.push('get');
            if (commands.includes('"print"') || commands.includes('"op": "print"')) types.push('print');
            if (commands.includes('"final"') || commands.includes('"op": "final"')) types.push('final');
            // WASM/CLI commands
            if (commands.includes('rust_wasm') || commands.includes('"op": "wasm"')) types.push('rust_wasm');
            if (commands.includes('rust_cli_intent')) types.push('rust_cli');
            return types;
        }

        function formatCommandTypes(commands) {
            const types = getCommandTypes(commands);
            if (types.length === 0) return 'commands';
            if (types.includes('rust_wasm')) return 'rust_wasm';
            if (types.includes('rust_cli')) return 'rust_cli';
            return types.join(', ');
        }

        function extractRustCode(commands) {
            if (!commands) return null;
            const match = commands.match(/"code"\s*:\s*"([^"]+)"/);
            if (match) {
                return match[1].replace(/\\n/g, '\n').replace(/\\"/g, '"').replace(/\\\\/g, '\\');
            }
            return null;
        }

        function renderResults(data) {
            // Keep progress section expanded (don't collapse it)
            // The user should still be able to see the progress after dismissing the modal

            // Enable Show Result button and auto-show modal
            document.getElementById('showResultBtn').disabled = false;
            showResultModal();
        }

        // Keep showStep for backward compatibility (not used anymore)
        function showStep(index) {
            showModalStep(index);
        }

        function formatJson(str) {
            try {
                // Try to parse and format each line as JSON
                const lines = str.split('\n').filter(l => l.trim());
                return lines.map(line => {
                    try {
                        return JSON.stringify(JSON.parse(line), null, 2);
                    } catch {
                        return line;
                    }
                }).join('\n\n');
            } catch {
                return str;
            }
        }

        function escapeHtml(str) {
            return str.replace(/&/g, '&amp;')
                      .replace(/</g, '&lt;')
                      .replace(/>/g, '&gt;')
                      .replace(/"/g, '&quot;');
        }
    </script>
</body>
</html>
"##;
