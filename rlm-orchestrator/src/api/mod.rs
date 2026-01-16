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
    #[serde(rename = "iteration_start")]
    IterationStart { step: usize },
    #[serde(rename = "llm_start")]
    LlmStart { step: usize },
    #[serde(rename = "llm_complete")]
    LlmComplete {
        step: usize,
        duration_ms: u64,
        response_preview: String,
    },
    #[serde(rename = "commands")]
    Commands { step: usize, commands: String },
    #[serde(rename = "wasm_compile_start")]
    WasmCompileStart { step: usize },
    #[serde(rename = "wasm_compile_complete")]
    WasmCompileComplete { step: usize, duration_ms: u64 },
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
        // Create a progress callback that sends events to the channel
        let tx_clone = tx.clone();
        let callback = Box::new(move |event: ProgressEvent| {
            let stream_event = match event {
                ProgressEvent::IterationStart { step } => StreamEvent::IterationStart { step },
                ProgressEvent::LlmCallStart { step } => StreamEvent::LlmStart { step },
                ProgressEvent::LlmCallComplete {
                    step,
                    duration_ms,
                    response_preview,
                } => StreamEvent::LlmComplete {
                    step,
                    duration_ms,
                    response_preview,
                },
                ProgressEvent::CommandsExtracted { step, commands } => {
                    StreamEvent::Commands { step, commands }
                }
                ProgressEvent::WasmCompileStart { step } => StreamEvent::WasmCompileStart { step },
                ProgressEvent::WasmCompileComplete { step, duration_ms } => {
                    StreamEvent::WasmCompileComplete { step, duration_ms }
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
                    }
                }
                ProgressEvent::FinalAnswer { answer } => StreamEvent::FinalAnswer { answer },
                ProgressEvent::Complete {
                    iterations: _,
                    success: _,
                } => {
                    // We'll send the complete event with full data after process returns
                    return;
                }
            };
            // Use try_send to avoid blocking - drop events if channel is full
            let _ = tx_clone.try_send(stream_event);
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

/// Visualization page
async fn visualize_page() -> Html<&'static str> {
    Html(VISUALIZE_HTML)
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
            --highlight: #e94560;
            --text: #eee;
            --muted: #888;
            --success: #4ade80;
            --error: #f87171;
            --wasm: #f59e0b;
            --progress: #3b82f6;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'SF Mono', 'Consolas', monospace;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 20px 40px;
        }
        .container { max-width: 100%; margin: 0 auto; }
        h1 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: var(--highlight);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        h1 span { font-size: 2rem; }
        .input-section {
            background: var(--card);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .input-row {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 15px;
            margin-bottom: 15px;
        }
        label {
            font-size: 0.85rem;
            color: var(--muted);
            margin-bottom: 5px;
            display: block;
        }
        textarea, input, select {
            width: 100%;
            background: var(--bg);
            border: 1px solid var(--accent);
            border-radius: 8px;
            padding: 12px;
            color: var(--text);
            font-family: inherit;
            font-size: 0.9rem;
            resize: vertical;
        }
        textarea:focus, input:focus, select:focus {
            outline: none;
            border-color: var(--highlight);
        }
        #query { height: 60px; }
        #context { height: 150px; }
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

        .example-selector {
            margin-bottom: 15px;
        }
        .example-selector select {
            max-width: 400px;
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
            font-size: 0.7rem;
            text-transform: uppercase;
            font-weight: 600;
        }
        .tag.wasm { background: var(--wasm); color: #000; }
        .tag.basic { background: var(--accent); }
        .tag.aggregation { background: #7c3aed; }
        .tag.large-context { background: #dc2626; color: #fff; }

        .results {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
        }
        .timeline {
            background: var(--card);
            border-radius: 12px;
            padding: 15px;
        }
        .timeline h2 {
            font-size: 0.9rem;
            color: var(--muted);
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .step {
            position: relative;
            padding: 12px 15px;
            margin-bottom: 8px;
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
        .step.has-wasm { border-left-color: var(--wasm); }
        .step-number {
            font-weight: bold;
            color: var(--highlight);
        }
        .step-meta {
            font-size: 0.75rem;
            color: var(--muted);
            margin-top: 4px;
        }
        .step-badges {
            display: flex;
            gap: 4px;
            margin-top: 4px;
        }
        .badge {
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.65rem;
            font-weight: 600;
        }
        .badge.wasm { background: var(--wasm); color: #000; }

        .detail-panel {
            background: var(--card);
            border-radius: 12px;
            padding: 20px;
        }
        .detail-section {
            margin-bottom: 20px;
        }
        .detail-section h3 {
            font-size: 0.85rem;
            color: var(--highlight);
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .detail-section h3.wasm-title { color: var(--wasm); }
        .code-block {
            background: var(--bg);
            border-radius: 8px;
            padding: 15px;
            overflow-x: auto;
            font-size: 0.85rem;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .code-block.json { color: #7dd3fc; }
        .code-block.output { color: #a5f3fc; }
        .code-block.error { color: var(--error); }
        .code-block.rust { color: var(--wasm); }

        .summary-bar {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .stat {
            background: var(--accent);
            padding: 15px 20px;
            border-radius: 8px;
        }
        .stat.wasm-stat { border: 2px solid var(--wasm); }
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--highlight);
        }
        .stat-value.wasm { color: var(--wasm); }
        .stat-label {
            font-size: 0.75rem;
            color: var(--muted);
            text-transform: uppercase;
        }

        .answer-box {
            background: linear-gradient(135deg, #065f46, #064e3b);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .answer-box h3 {
            color: var(--success);
            margin-bottom: 10px;
        }
        .answer-box.error {
            background: linear-gradient(135deg, #7f1d1d, #450a0a);
        }
        .answer-box.error h3 { color: var(--error); }

        .hidden { display: none; }

        .flow-diagram {
            display: flex;
            align-items: center;
            gap: 5px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .flow-node {
            padding: 8px 12px;
            background: var(--accent);
            border-radius: 6px;
            font-size: 0.75rem;
        }
        .flow-node.query { background: var(--highlight); }
        .flow-node.final { background: #065f46; }
        .flow-node.wasm { background: var(--wasm); color: #000; }
        .flow-arrow {
            color: var(--muted);
            font-size: 1.2rem;
        }

        .wasm-info {
            background: linear-gradient(135deg, #78350f, #451a03);
            border-radius: 12px;
            padding: 15px 20px;
            margin-bottom: 20px;
            border: 1px solid var(--wasm);
        }
        .wasm-info h3 {
            color: var(--wasm);
            margin-bottom: 8px;
            font-size: 0.9rem;
        }
        .wasm-info p {
            color: var(--text);
            font-size: 0.85rem;
            line-height: 1.5;
        }

        /* Progress section */
        .progress-section {
            background: var(--card);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .progress-header h2 {
            font-size: 0.9rem;
            color: var(--progress);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .progress-status {
            font-size: 0.8rem;
            color: var(--muted);
        }
        .progress-status.active { color: var(--progress); }
        .progress-log {
            background: var(--bg);
            border-radius: 8px;
            padding: 15px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 0.8rem;
            line-height: 1.6;
        }
        .progress-log .event {
            margin-bottom: 4px;
            display: flex;
            gap: 8px;
        }
        .progress-log .event-time {
            color: var(--muted);
            min-width: 70px;
        }
        .progress-log .event-icon { min-width: 20px; }
        .progress-log .event-llm { color: var(--highlight); }
        .progress-log .event-wasm { color: var(--wasm); }
        .progress-log .event-cmd { color: #7dd3fc; }
        .progress-log .event-done { color: var(--success); }

        /* Context preview */
        .context-preview {
            background: var(--bg);
            border-radius: 8px;
            padding: 12px;
            font-size: 0.75rem;
            margin-top: 8px;
            max-height: 100px;
            overflow: hidden;
        }
        .context-preview .line {
            color: var(--muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .context-preview .ellipsis {
            color: var(--highlight);
            text-align: center;
            padding: 4px 0;
        }
        .context-stats {
            font-size: 0.75rem;
            color: var(--muted);
            margin-top: 8px;
        }

        @media (max-width: 900px) {
            .results { grid-template-columns: 1fr; }
            .input-row { grid-template-columns: 1fr; }
            body { padding: 15px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1><span>🔄</span> RLM Visualizer</h1>

        <div class="input-section">
            <div class="example-selector">
                <label for="exampleSelect">Load Example</label>
                <select id="exampleSelect" onchange="loadExample()">
                    <option value="">-- Select an example --</option>
                    <optgroup label="Large Context Demos">
                        <option value="war_peace_family">War and Peace: Family Tree (3.2MB)</option>
                        <option value="large_logs_errors">Large Logs: Error Ranking (5000 lines)</option>
                        <option value="large_logs_ips">Large Logs: Unique IPs (5000 lines)</option>
                    </optgroup>
                    <optgroup label="WASM Use Cases">
                        <option value="wasm_unique_ips">Unique IP addresses (rust_wasm)</option>
                        <option value="wasm_error_ranking">Rank errors by frequency (rust_wasm)</option>
                        <option value="wasm_word_freq">Word frequency analysis (rust_wasm)</option>
                        <option value="wasm_response_times">Response time percentiles (rust_wasm)</option>
                    </optgroup>
                    <optgroup label="Basic Commands">
                        <option value="count_errors">Count ERROR lines</option>
                        <option value="find_pattern">Find text pattern</option>
                    </optgroup>
                </select>
                <div class="example-tags" id="exampleTags"></div>
            </div>

            <div class="input-row">
                <div>
                    <label for="query">Query</label>
                    <textarea id="query" placeholder="What do you want to know about the context?">How many ERROR lines are there?</textarea>
                </div>
                <div>
                    <label for="context">Context <span id="contextStats" class="context-stats"></span></label>
                    <textarea id="context" placeholder="Paste your text/code here..." oninput="updateContextPreview()">Line 1: INFO - System started
Line 2: ERROR - Connection failed
Line 3: INFO - Retrying connection
Line 4: ERROR - Timeout occurred
Line 5: WARNING - High memory usage
Line 6: INFO - Connection established
Line 7: ERROR - Invalid input received</textarea>
                    <div id="contextPreview" class="context-preview"></div>
                </div>
            </div>
            <button id="runBtn" onclick="runQuery()">Run RLM Query</button>
        </div>

        <div id="progressSection" class="progress-section hidden">
            <div class="progress-header">
                <h2>Live Progress</h2>
                <span id="progressStatus" class="progress-status">Waiting...</span>
            </div>
            <div id="progressLog" class="progress-log"></div>
        </div>

        <div id="resultsSection" class="hidden">
            <div class="summary-bar">
                <div class="stat">
                    <div class="stat-value" id="statIterations">-</div>
                    <div class="stat-label">Iterations</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="statSubCalls">-</div>
                    <div class="stat-label">Sub-LM Calls</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="statContext">-</div>
                    <div class="stat-label">Context Chars</div>
                </div>
                <div class="stat wasm-stat" id="statWasm" style="display: none;">
                    <div class="stat-value wasm">-</div>
                    <div class="stat-label">WASM Steps</div>
                </div>
            </div>

            <div id="wasmInfo" class="wasm-info hidden">
                <h3>🔧 WASM Execution Used</h3>
                <p></p>
            </div>

            <div id="flowDiagram" class="flow-diagram"></div>

            <div id="answerBox" class="answer-box">
                <h3>Final Answer</h3>
                <div id="answerText"></div>
            </div>

            <div class="results">
                <div class="timeline">
                    <h2>Execution Timeline</h2>
                    <div id="timeline"></div>
                </div>

                <div class="detail-panel">
                    <div id="detailContent">
                        <p style="color: var(--muted);">Click on a step to see details</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentData = null;
        let eventSource = null;
        let startTime = null;

        // Initialize context preview on load
        document.addEventListener('DOMContentLoaded', updateContextPreview);

        function updateContextPreview() {
            const context = document.getElementById('context').value;
            const lines = context.split('\n');
            const lineCount = lines.length;
            const charCount = context.length;

            // Update stats
            document.getElementById('contextStats').textContent = `(${lineCount} lines, ${charCount.toLocaleString()} chars)`;

            // Show head/tail preview
            const preview = document.getElementById('contextPreview');
            if (lineCount <= 6) {
                preview.innerHTML = lines.map(l => `<div class="line">${escapeHtml(l)}</div>`).join('');
            } else {
                const head = lines.slice(0, 3).map(l => `<div class="line">${escapeHtml(l)}</div>`).join('');
                const tail = lines.slice(-3).map(l => `<div class="line">${escapeHtml(l)}</div>`).join('');
                preview.innerHTML = head + `<div class="ellipsis">... ${lineCount - 6} more lines ...</div>` + tail;
            }
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
        }

        // Example data for the dropdown
        const examples = {
            // Large context demos that load from server
            war_peace_family: {
                query: "Build a family tree for the main characters. Identify characters who appear multiple times and are related to each other (by blood or marriage). Show the relationships in a structured format.",
                context: null,  // Will be loaded from /samples/war-and-peace
                loadUrl: '/samples/war-and-peace',
                tags: ['large-context', 'synthesis', 'literature'],
                description: 'Analyzes 3.2MB of text to extract character relationships'
            },
            large_logs_errors: {
                query: "Rank the error types from most to least frequent. Show the count for each error type.",
                context: null,  // Will be loaded from /samples/large-logs
                loadUrl: '/samples/large-logs',
                tags: ['large-context', 'wasm', 'aggregation'],
                description: 'Analyzes 5000 log lines using rust_wasm HashMap'
            },
            large_logs_ips: {
                query: "How many unique IP addresses appear in these logs? List the top 10 most active IPs.",
                context: null,  // Will be loaded from /samples/large-logs
                loadUrl: '/samples/large-logs',
                tags: ['large-context', 'wasm', 'aggregation'],
                description: 'Extracts and counts unique IPs from 5000 log lines'
            },
            // WASM demos with moderate context (triggers RLM, not bypass)
            wasm_unique_ips: {
                query: "How many unique IP addresses are in these logs?",
                context: generateLogContext(200),
                tags: ['wasm', 'aggregation'],
                description: 'Uses rust_wasm with HashSet to count unique IPs'
            },
            wasm_error_ranking: {
                query: "Rank the error types from most to least frequent",
                context: generateErrorLogs(150),
                tags: ['wasm', 'aggregation', 'ranking'],
                description: 'Uses rust_wasm with HashMap to count and sort error types'
            },
            wasm_word_freq: {
                query: "What are the top 10 most common words in this text?",
                context: generateTextSample(),
                tags: ['wasm', 'aggregation', 'text-analysis'],
                description: 'Uses rust_wasm with HashMap for word frequency analysis'
            },
            wasm_response_times: {
                query: "Calculate the p50, p95, and p99 response time percentiles",
                context: generateResponseTimes(300),
                tags: ['wasm', 'statistics', 'percentiles'],
                description: 'Uses rust_wasm to parse times and calculate percentiles'
            },
            // Basic demos (still larger than before)
            count_errors: {
                query: "How many ERROR lines are there?",
                context: generateLogContext(100),
                tags: ['basic', 'count'],
                description: 'Error counting using basic commands'
            },
            find_pattern: {
                query: "Find all lines containing 'AuthenticationFailed'",
                context: generateLogContext(100),
                tags: ['basic', 'search'],
                description: 'Pattern search using find/regex commands'
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

                // Show tags immediately
                const tagsHtml = example.tags.map(tag => {
                    let tagClass = 'basic';
                    if (tag === 'wasm') tagClass = 'wasm';
                    else if (tag === 'large-context') tagClass = 'large-context';
                    else if (tag === 'aggregation' || tag === 'ranking') tagClass = 'aggregation';
                    return `<span class="tag ${tagClass}">${tag}</span>`;
                }).join('');
                document.getElementById('exampleTags').innerHTML = tagsHtml + `<span style="color: var(--muted); font-size: 0.75rem; margin-left: 8px;">${example.description}</span>`;

                // Load context - either from URL or use static value
                if (example.loadUrl) {
                    document.getElementById('context').value = 'Loading large context...';
                    updateContextPreview();
                    try {
                        const response = await fetch(example.loadUrl);
                        if (!response.ok) throw new Error(`HTTP ${response.status}`);
                        const text = await response.text();
                        document.getElementById('context').value = text;
                        updateContextPreview();
                    } catch (err) {
                        document.getElementById('context').value = `Error loading: ${err.message}`;
                        updateContextPreview();
                    }
                } else {
                    document.getElementById('context').value = example.context;
                    updateContextPreview();
                }
            } else {
                document.getElementById('exampleTags').innerHTML = '';
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

            // Show progress section, hide results
            document.getElementById('progressSection').classList.remove('hidden');
            document.getElementById('resultsSection').classList.add('hidden');
            clearProgress();

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
                    logProgress('⏳', `Calling LLM...`, 'event-llm');
                    break;
                case 'llm_complete':
                    logProgress('✓', `LLM responded (${event.duration_ms}ms)`, 'event-llm');
                    break;
                case 'commands':
                    const isWasm = event.commands.includes('rust_wasm');
                    logProgress('▶', `Commands: ${isWasm ? 'rust_wasm' : 'executing...'}`, isWasm ? 'event-wasm' : 'event-cmd');
                    break;
                case 'wasm_compile_start':
                    logProgress('🔧', `Compiling WASM...`, 'event-wasm');
                    break;
                case 'wasm_compile_complete':
                    logProgress('✓', `WASM compiled (${event.duration_ms}ms)`, 'event-wasm');
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
                        prompt_tokens: 0,
                        completion_tokens: 0
                    });
                    break;
                case 'final_answer':
                    logProgress('✅', `Final answer received`, 'event-done');
                    break;
                case 'complete':
                    logProgress('🏁', `Complete: ${event.iterations} iterations, ${event.success ? 'success' : 'failed'}`, 'event-done');

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

        function extractRustCode(commands) {
            if (!commands) return null;
            const match = commands.match(/"code"\s*:\s*"([^"]+)"/);
            if (match) {
                return match[1].replace(/\\n/g, '\n').replace(/\\"/g, '"').replace(/\\\\/g, '\\');
            }
            return null;
        }

        function renderResults(data) {
            document.getElementById('resultsSection').classList.remove('hidden');

            // Check for WASM usage across all steps
            const wasmSteps = data.history.filter(s => hasWasm(s.commands));
            const usedWasm = wasmSteps.length > 0;

            // Stats
            document.getElementById('statIterations').textContent = data.iterations;
            document.getElementById('statSubCalls').textContent = data.total_sub_calls;
            document.getElementById('statContext').textContent = data.context_length.toLocaleString();

            // Update WASM stat
            const wasmStatEl = document.getElementById('statWasm');
            if (wasmStatEl) {
                wasmStatEl.querySelector('.stat-value').textContent = wasmSteps.length;
                wasmStatEl.style.display = usedWasm ? 'block' : 'none';
            }

            // Show WASM info box if WASM was used
            const wasmInfoEl = document.getElementById('wasmInfo');
            if (wasmInfoEl) {
                if (usedWasm) {
                    wasmInfoEl.classList.remove('hidden');
                    wasmInfoEl.querySelector('p').textContent =
                        `This query used rust_wasm in ${wasmSteps.length} step(s) to perform custom analysis. ` +
                        `WASM enables complex operations like aggregation, sorting, and statistical calculations that aren't possible with basic text commands.`;
                } else {
                    wasmInfoEl.classList.add('hidden');
                }
            }

            // Answer
            const answerBox = document.getElementById('answerBox');
            answerBox.className = 'answer-box' + (data.success ? '' : ' error');
            document.getElementById('answerText').textContent = data.success ? data.answer : (data.error || 'Failed');

            // Flow diagram
            const flow = document.getElementById('flowDiagram');
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
            const timeline = document.getElementById('timeline');
            timeline.innerHTML = data.history.map((step, i) => {
                const hasError = !!step.error;
                const isFinal = step.output.startsWith('FINAL:');
                const isWasm = hasWasm(step.commands);
                let stepClass = hasError ? ' has-error' : (isWasm ? ' has-wasm' : (isFinal ? ' is-final' : ''));

                let badges = '';
                if (isWasm) {
                    badges = '<div class="step-badges"><span class="badge wasm">WASM</span></div>';
                }

                return `<div class="step${stepClass}" onclick="showStep(${i})">
                    <div class="step-number">Step ${step.step}</div>
                    <div class="step-meta">
                        ${step.commands ? (isWasm ? 'rust_wasm executed' : 'Commands executed') : 'No commands'}
                        ${step.sub_calls > 0 ? ` • ${step.sub_calls} sub-calls` : ''}
                        ${hasError ? ' • Error' : ''}
                    </div>
                    ${badges}
                </div>`;
            }).join('');

            // Show first step
            if (data.history.length > 0) {
                showStep(0);
            }
        }

        function showStep(index) {
            const step = currentData.history[index];

            // Update active state
            document.querySelectorAll('.step').forEach((el, i) => {
                el.classList.toggle('active', i === index);
            });

            // Render detail
            let html = '';
            const isWasm = hasWasm(step.commands);
            const rustCode = extractRustCode(step.commands);

            // Show Rust code separately if this is a WASM step
            if (isWasm && rustCode) {
                html += `<div class="detail-section">
                    <h3 class="wasm-title">🔧 Rust Code (rust_wasm)</h3>
                    <pre class="code-block rust">${escapeHtml(rustCode)}</pre>
                </div>`;
            }

            if (step.commands) {
                html += `<div class="detail-section">
                    <h3>${isWasm ? 'Command JSON' : 'Commands (JSON)'}</h3>
                    <pre class="code-block json">${escapeHtml(formatJson(step.commands))}</pre>
                </div>`;
            }

            if (step.output) {
                html += `<div class="detail-section">
                    <h3>Output</h3>
                    <pre class="code-block output">${escapeHtml(step.output)}</pre>
                </div>`;
            }

            if (step.error) {
                html += `<div class="detail-section">
                    <h3>Error</h3>
                    <pre class="code-block error">${escapeHtml(step.error)}</pre>
                </div>`;
            }

            if (step.sub_calls > 0) {
                html += `<div class="detail-section">
                    <h3>Sub-LM Calls</h3>
                    <p>${step.sub_calls} call(s) made to sub-LM</p>
                </div>`;
            }

            document.getElementById('detailContent').innerHTML = html || '<p style="color: var(--muted);">No data for this step</p>';
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
