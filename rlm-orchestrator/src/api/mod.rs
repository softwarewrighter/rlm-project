//! REST API for RLM orchestrator

use crate::orchestrator::{IterationRecord, RlmOrchestrator, RlmResult};
use axum::{
    extract::State,
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
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

/// Create the API router
pub fn create_router(state: Arc<ApiState>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/query", post(process_query))
        .route("/debug", post(debug_query))
        .route("/visualize", get(visualize_page))
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
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'SF Mono', 'Consolas', monospace;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
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

        @media (max-width: 900px) {
            .results { grid-template-columns: 1fr; }
            .input-row { grid-template-columns: 1fr; }
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
                    <optgroup label="Basic Commands">
                        <option value="count_errors">Count ERROR lines</option>
                        <option value="find_pattern">Find text pattern</option>
                    </optgroup>
                    <optgroup label="WASM Use Cases">
                        <option value="wasm_unique_ips">Unique IP addresses (rust_wasm)</option>
                        <option value="wasm_error_ranking">Rank errors by frequency (rust_wasm)</option>
                        <option value="wasm_word_freq">Word frequency analysis (rust_wasm)</option>
                        <option value="wasm_response_times">Response time percentiles (rust_wasm)</option>
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
                    <label for="context">Context</label>
                    <textarea id="context" placeholder="Paste your text/code here...">Line 1: INFO - System started
Line 2: ERROR - Connection failed
Line 3: INFO - Retrying connection
Line 4: ERROR - Timeout occurred
Line 5: WARNING - High memory usage
Line 6: INFO - Connection established
Line 7: ERROR - Invalid input received</textarea>
                </div>
            </div>
            <button id="runBtn" onclick="runQuery()">Run RLM Query</button>
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

        // Example data for the dropdown
        const examples = {
            count_errors: {
                query: "How many ERROR lines are there?",
                context: `Line 1: INFO - System started
Line 2: ERROR - Connection failed
Line 3: INFO - Retrying connection
Line 4: ERROR - Timeout occurred
Line 5: WARNING - High memory usage
Line 6: INFO - Connection established
Line 7: ERROR - Invalid input received`,
                tags: ['basic', 'count'],
                description: 'Simple error counting using basic commands'
            },
            find_pattern: {
                query: "Find all lines containing 'Connection'",
                context: `Line 1: INFO - System started
Line 2: ERROR - Connection failed
Line 3: INFO - Retrying connection
Line 4: ERROR - Timeout occurred
Line 5: WARNING - High memory usage
Line 6: INFO - Connection established
Line 7: ERROR - Invalid input received`,
                tags: ['basic', 'search'],
                description: 'Pattern search using find/regex commands'
            },
            wasm_unique_ips: {
                query: "How many unique IP addresses are in these logs?",
                context: `2024-01-15 10:23:45 [INFO] Request from 192.168.1.100 - GET /api/users - 200 OK - 45ms
2024-01-15 10:23:46 [ERROR] AuthenticationFailed from 10.0.0.50 - Invalid token
2024-01-15 10:23:47 [INFO] Request from 192.168.1.100 - POST /api/data - 201 Created - 123ms
2024-01-15 10:23:48 [WARN] ConnectionTimeout from 172.16.0.25 - Database slow
2024-01-15 10:23:49 [INFO] Request from 10.0.0.50 - GET /api/health - 200 OK - 12ms
2024-01-15 10:23:50 [ERROR] RequestFailed from 192.168.1.100 - Service unavailable
2024-01-15 10:23:51 [INFO] Request from 172.16.0.25 - GET /api/users - 200 OK - 67ms
2024-01-15 10:23:52 [ERROR] ValidationError from 10.0.0.75 - Missing required field
2024-01-15 10:23:53 [INFO] Request from 10.0.0.75 - PUT /api/users/1 - 200 OK - 89ms
2024-01-15 10:23:54 [ERROR] AuthenticationFailed from 192.168.1.200 - Expired session`,
                tags: ['wasm', 'aggregation'],
                description: 'Uses rust_wasm with HashSet to count unique IPs'
            },
            wasm_error_ranking: {
                query: "Rank the error types from most to least frequent",
                context: `2024-01-15 10:23:46 [AuthenticationFailed] from 10.0.0.50 - Invalid token
2024-01-15 10:23:48 [ConnectionTimeout] from 172.16.0.25 - Database slow
2024-01-15 10:23:50 [RequestFailed] from 192.168.1.100 - Service unavailable
2024-01-15 10:23:52 [ValidationError] from 10.0.0.75 - Missing field
2024-01-15 10:23:54 [AuthenticationFailed] from 192.168.1.200 - Expired session
2024-01-15 10:23:56 [ConnectionTimeout] from 172.16.0.30 - Timeout
2024-01-15 10:23:58 [AuthenticationFailed] from 10.0.0.60 - Bad credentials
2024-01-15 10:24:00 [RequestFailed] from 192.168.1.105 - 503 error
2024-01-15 10:24:02 [ConnectionTimeout] from 172.16.0.25 - Pool exhausted
2024-01-15 10:24:04 [AuthenticationFailed] from 10.0.0.50 - Token revoked
2024-01-15 10:24:06 [ValidationError] from 192.168.1.100 - Invalid format
2024-01-15 10:24:08 [RequestFailed] from 10.0.0.75 - Gateway timeout`,
                tags: ['wasm', 'aggregation', 'ranking'],
                description: 'Uses rust_wasm with HashMap to count and sort error types'
            },
            wasm_word_freq: {
                query: "What are the top 5 most common words in this text?",
                context: `The quick brown fox jumps over the lazy dog. The dog was not amused by the fox.
The fox tried again to jump over the dog but the dog moved away.
A lazy afternoon with the quick fox and the lazy dog made for an interesting scene.
The brown fox was quick but the dog was quicker this time.`,
                tags: ['wasm', 'aggregation', 'text-analysis'],
                description: 'Uses rust_wasm with HashMap for word frequency analysis'
            },
            wasm_response_times: {
                query: "Calculate the p50, p95, and p99 response time percentiles",
                context: `GET /api/users - 45ms
POST /api/data - 123ms
GET /api/health - 12ms
GET /api/users - 67ms
PUT /api/users/1 - 89ms
GET /api/products - 234ms
POST /api/orders - 456ms
GET /api/users - 34ms
DELETE /api/cache - 23ms
GET /api/stats - 567ms
POST /api/upload - 1234ms
GET /api/download - 89ms
PUT /api/settings - 45ms
GET /api/config - 28ms
POST /api/batch - 789ms`,
                tags: ['wasm', 'statistics', 'percentiles'],
                description: 'Uses rust_wasm to parse times and calculate percentiles'
            }
        };

        function loadExample() {
            const select = document.getElementById('exampleSelect');
            const example = examples[select.value];
            if (example) {
                document.getElementById('query').value = example.query;
                document.getElementById('context').value = example.context;

                // Show tags
                const tagsHtml = example.tags.map(tag => {
                    const tagClass = tag === 'wasm' ? 'wasm' : (tag === 'aggregation' || tag === 'ranking' ? 'aggregation' : 'basic');
                    return `<span class="tag ${tagClass}">${tag}</span>`;
                }).join('');
                document.getElementById('exampleTags').innerHTML = tagsHtml + `<span style="color: var(--muted); font-size: 0.75rem; margin-left: 8px;">${example.description}</span>`;
            } else {
                document.getElementById('exampleTags').innerHTML = '';
            }
        }

        async function runQuery() {
            const query = document.getElementById('query').value;
            const context = document.getElementById('context').value;
            const btn = document.getElementById('runBtn');

            btn.disabled = true;
            btn.textContent = 'Running...';

            try {
                const response = await fetch('/debug', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, context })
                });

                const data = await response.json();
                currentData = data;
                renderResults(data);
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Run RLM Query';
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
