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
}

/// Request to process a query
#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    /// The query to process
    pub query: String,
    /// The context to analyze
    pub context: String,
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
async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Process a query
async fn process_query(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, String)> {
    match state.orchestrator.process(&request.query, &request.context).await {
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

    match state.orchestrator.process(&request.query, &request.context).await {
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
        textarea, input {
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
        textarea:focus, input:focus {
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
        .step-number {
            font-weight: bold;
            color: var(--highlight);
        }
        .step-meta {
            font-size: 0.75rem;
            color: var(--muted);
            margin-top: 4px;
        }

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
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--highlight);
        }
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
        .flow-arrow {
            color: var(--muted);
            font-size: 1.2rem;
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

        function renderResults(data) {
            document.getElementById('resultsSection').classList.remove('hidden');

            // Stats
            document.getElementById('statIterations').textContent = data.iterations;
            document.getElementById('statSubCalls').textContent = data.total_sub_calls;
            document.getElementById('statContext').textContent = data.context_length.toLocaleString();

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
                flow.innerHTML += `<div class="flow-node${isFinal ? ' final' : ''}">${step.error ? '⚠️' : ''}Step ${step.step}</div>`;
            });
            if (data.success) {
                flow.innerHTML += '<span class="flow-arrow">→</span><div class="flow-node final">Answer</div>';
            }

            // Timeline
            const timeline = document.getElementById('timeline');
            timeline.innerHTML = data.history.map((step, i) => {
                const hasError = !!step.error;
                const isFinal = step.output.startsWith('FINAL:');
                return `<div class="step${hasError ? ' has-error' : ''}${isFinal ? ' is-final' : ''}"
                             onclick="showStep(${i})">
                    <div class="step-number">Step ${step.step}</div>
                    <div class="step-meta">
                        ${step.commands ? 'Commands executed' : 'No commands'}
                        ${step.sub_calls > 0 ? ` • ${step.sub_calls} sub-calls` : ''}
                        ${hasError ? ' • Error' : ''}
                    </div>
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

            if (step.commands) {
                html += `<div class="detail-section">
                    <h3>Commands (JSON)</h3>
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
