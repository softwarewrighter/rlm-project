//! RLM orchestration logic

use crate::commands::{
    extract_commands, extract_final, CommandExecutor, ExecutionResult, LlmQueryCallback,
};
use crate::pool::LlmPool;
use crate::provider::{LlmRequest, ProviderError};
use crate::RlmConfig;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Progress events emitted during RLM processing for real-time feedback
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// Query starting (emitted once at the beginning)
    QueryStart {
        context_chars: usize,
        query_len: usize,
    },
    /// Starting a new iteration
    IterationStart { step: usize },
    /// LLM call starting
    LlmCallStart { step: usize },
    /// LLM call completed
    LlmCallComplete {
        step: usize,
        duration_ms: u64,
        prompt_tokens: u32,
        completion_tokens: u32,
        response_preview: String,
    },
    /// Commands extracted and about to execute
    CommandsExtracted { step: usize, commands: String },
    /// WASM compilation starting
    WasmCompileStart { step: usize },
    /// WASM compilation complete
    WasmCompileComplete { step: usize, duration_ms: u64 },
    /// WASM execution complete (separate from compilation)
    WasmRunComplete { step: usize, duration_ms: u64 },
    /// CLI code generation starting (LLM call)
    CliCodegenStart { step: usize },
    /// CLI code generation complete (LLM call)
    CliCodegenComplete { step: usize, duration_ms: u64 },
    /// CLI compilation starting (rustc)
    CliCompileStart { step: usize },
    /// CLI compilation complete (rustc)
    CliCompileComplete { step: usize, duration_ms: u64 },
    /// CLI execution complete
    CliRunComplete { step: usize, duration_ms: u64 },
    /// Command execution complete
    CommandComplete {
        step: usize,
        output_preview: String,
        exec_ms: u64,
    },
    /// Iteration finished
    IterationComplete {
        step: usize,
        record: IterationRecord,
    },
    /// Final answer found
    FinalAnswer { answer: String },
    /// Processing complete
    Complete {
        iterations: usize,
        success: bool,
        total_duration_ms: u64,
    },
}

/// Callback type for progress events
pub type ProgressCallback = Box<dyn Fn(ProgressEvent) + Send + Sync>;

/// Errors from RLM orchestration
#[derive(Error, Debug)]
pub enum OrchestratorError {
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),

    #[error("Command error: {0}")]
    Command(#[from] crate::commands::CommandError),

    #[error("Max iterations ({0}) exceeded")]
    MaxIterations(usize),

    #[error("Max sub-calls ({0}) exceeded")]
    MaxSubCalls(usize),

    #[error("No commands found in response")]
    NoCommands,
}

/// Token usage for an iteration
#[derive(Debug, Clone, Default)]
pub struct IterationTokens {
    /// Tokens in the prompt sent to root LLM
    pub prompt_tokens: u32,
    /// Tokens in the completion from root LLM
    pub completion_tokens: u32,
}

/// Timing information for an iteration
#[derive(Debug, Clone, Default)]
pub struct IterationTiming {
    /// Time spent waiting for LLM response (milliseconds)
    pub llm_ms: u64,
    /// Time spent executing commands (milliseconds)
    pub exec_ms: u64,
    /// Time spent compiling rust_wasm (milliseconds, if any)
    pub compile_ms: u64,
    /// Time spent running WASM (milliseconds, if any)
    pub wasm_run_ms: u64,
}

/// Record of a single RLM iteration
#[derive(Debug, Clone)]
pub struct IterationRecord {
    /// Step number (1-indexed)
    pub step: usize,
    /// Raw LLM response (full text before parsing)
    pub llm_response: String,
    /// Commands that were executed (JSON extracted from response)
    pub commands: String,
    /// Output from execution
    pub output: String,
    /// Error if any
    pub error: Option<String>,
    /// Number of sub-LM calls made in this iteration
    pub sub_calls: usize,
    /// Token usage for this iteration
    pub tokens: IterationTokens,
    /// Timing information for this iteration
    pub timing: IterationTiming,
}

/// Result of an RLM query
#[derive(Debug)]
pub struct RlmResult {
    /// Final answer
    pub answer: String,
    /// Number of iterations taken
    pub iterations: usize,
    /// History of iterations
    pub history: Vec<IterationRecord>,
    /// Total sub-LM calls made
    pub total_sub_calls: usize,
    /// Whether the query succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Total prompt tokens used (root LLM only)
    pub total_prompt_tokens: u32,
    /// Total completion tokens used (root LLM only)
    pub total_completion_tokens: u32,
    /// Context size in characters (for comparison)
    pub context_chars: usize,
    /// Whether RLM was bypassed (direct LLM call for small contexts)
    pub bypassed: bool,
}

/// RLM Orchestrator
pub struct RlmOrchestrator {
    config: RlmConfig,
    pool: Arc<LlmPool>,
}

impl RlmOrchestrator {
    /// Create a new orchestrator
    pub fn new(config: RlmConfig, pool: Arc<LlmPool>) -> Self {
        Self { config, pool }
    }

    /// Create the llm_query callback for sub-LM calls
    fn create_llm_query_callback(&self, total_sub_calls: Arc<AtomicUsize>) -> LlmQueryCallback {
        let pool = Arc::clone(&self.pool);
        let max_sub_calls = self.config.max_sub_calls;

        Arc::new(move |prompt: &str| {
            // Check if we've exceeded max sub-calls
            let current = total_sub_calls.load(Ordering::Relaxed);
            if current >= max_sub_calls {
                return Err(format!("Max sub-calls ({}) exceeded", max_sub_calls));
            }

            // Get the current tokio runtime handle
            let handle = tokio::runtime::Handle::try_current()
                .map_err(|e| format!("No tokio runtime available: {}", e))?;

            let pool = Arc::clone(&pool);
            let prompt = prompt.to_string();

            // Use block_in_place to allow blocking within the async runtime
            let result = tokio::task::block_in_place(|| {
                handle.block_on(async {
                    let request =
                        LlmRequest::new("You are a helpful assistant. Answer concisely.", &prompt);
                    pool.complete(&request, true).await
                })
            });

            match result {
                Ok(response) => {
                    total_sub_calls.fetch_add(1, Ordering::Relaxed);
                    Ok(response.content)
                }
                Err(e) => Err(e.to_string()),
            }
        })
    }

    /// Check if bypass should be used for this context size
    pub fn should_bypass(&self, context_len: usize) -> bool {
        self.config.bypass_enabled && context_len < self.config.bypass_threshold
    }

    /// Process directly without RLM iteration (for small contexts)
    async fn process_direct(
        &self,
        query: &str,
        context: &str,
    ) -> Result<RlmResult, OrchestratorError> {
        info!(
            context_len = context.len(),
            threshold = self.config.bypass_threshold,
            "Bypassing RLM - context below threshold"
        );

        let system_prompt = "You are a helpful assistant. Answer the question based on the provided context. Be extremely concise - respond with just the answer (number, name, or short phrase). Do not explain or elaborate.";
        let prompt = format!("Context:\n{}\n\nQuestion: {}", context, query);

        let llm_start = Instant::now();
        let request = LlmRequest::new(system_prompt, &prompt);
        let response = self.pool.complete(&request, false).await?;
        let llm_ms = llm_start.elapsed().as_millis() as u64;

        let (prompt_tokens, completion_tokens) = response
            .usage
            .map(|u| (u.prompt_tokens, u.completion_tokens))
            .unwrap_or((0, 0));

        Ok(RlmResult {
            answer: response.content.clone(),
            iterations: 1,
            history: vec![IterationRecord {
                step: 1,
                llm_response: response.content,
                commands: "(direct)".to_string(),
                output: "Bypassed RLM - small context".to_string(),
                error: None,
                sub_calls: 0,
                tokens: IterationTokens {
                    prompt_tokens,
                    completion_tokens,
                },
                timing: IterationTiming {
                    llm_ms,
                    exec_ms: 0,
                    compile_ms: 0,
                    wasm_run_ms: 0,
                },
            }],
            total_sub_calls: 0,
            success: true,
            error: None,
            total_prompt_tokens: prompt_tokens,
            total_completion_tokens: completion_tokens,
            context_chars: context.len(),
            bypassed: true,
        })
    }

    /// Process a query over the given context (no progress callbacks)
    pub async fn process(
        &self,
        query: &str,
        context: &str,
    ) -> Result<RlmResult, OrchestratorError> {
        self.process_with_options(query, context, None, false).await
    }

    /// Process a query with optional progress callbacks for real-time feedback
    pub async fn process_with_progress(
        &self,
        query: &str,
        context: &str,
        progress: Option<ProgressCallback>,
    ) -> Result<RlmResult, OrchestratorError> {
        self.process_with_options(query, context, progress, false).await
    }

    /// Process a query with all options
    pub async fn process_with_options(
        &self,
        query: &str,
        context: &str,
        progress: Option<ProgressCallback>,
        force_rlm: bool,
    ) -> Result<RlmResult, OrchestratorError> {
        // Track overall query duration
        let query_start = Instant::now();

        // Helper to emit progress events
        let emit = |event: ProgressEvent| {
            if let Some(ref cb) = progress {
                cb(event);
            }
        };

        // Check if we should bypass RLM for small contexts (unless force_rlm is set)
        if !force_rlm && self.should_bypass(context.len()) {
            return self.process_direct(query, context).await;
        }

        let mut history = Vec::new();
        let total_sub_calls = Arc::new(AtomicUsize::new(0));
        let mut total_prompt_tokens: u32 = 0;
        let mut total_completion_tokens: u32 = 0;
        let context_chars = context.len();

        // Emit query start event
        emit(ProgressEvent::QueryStart {
            context_chars,
            query_len: query.len(),
        });

        let system_prompt = self.build_system_prompt(context.len());

        // Create the llm_query callback
        let llm_query_cb = self.create_llm_query_callback(Arc::clone(&total_sub_calls));

        // Create command executor (persists across iterations)
        let mut executor = CommandExecutor::with_wasm_config(
            context.to_string(),
            self.config.max_sub_calls,
            &self.config.wasm,
        )
        .with_llm_callback(llm_query_cb);

        for iteration in 0..self.config.max_iterations {
            let step = iteration + 1;
            info!(iteration = step, "Starting RLM iteration");
            emit(ProgressEvent::IterationStart { step });

            // Check if we've exceeded max sub-calls
            if total_sub_calls.load(Ordering::Relaxed) >= self.config.max_sub_calls {
                emit(ProgressEvent::Complete {
                    iterations: step,
                    success: false,
                    total_duration_ms: query_start.elapsed().as_millis() as u64,
                });
                return Err(OrchestratorError::MaxSubCalls(self.config.max_sub_calls));
            }

            // Build the prompt with history
            let prompt = self.build_prompt(query, context, &history);

            // Call the root LLM (with timing)
            emit(ProgressEvent::LlmCallStart { step });
            let llm_start = Instant::now();
            let request = LlmRequest::new(&system_prompt, &prompt)
                .with_max_tokens(2048);  // Ensure enough tokens for WASM code
            let response = self.pool.complete(&request, false).await?;
            let llm_ms = llm_start.elapsed().as_millis() as u64;

            // Track token usage (extract before emitting event)
            let iter_tokens = response
                .usage
                .as_ref()
                .map(|u| IterationTokens {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                })
                .unwrap_or_default();
            total_prompt_tokens += iter_tokens.prompt_tokens;
            total_completion_tokens += iter_tokens.completion_tokens;

            // Emit LLM complete with preview and tokens (UTF-8 safe truncation)
            let response_preview = if response.content.chars().count() > 100 {
                format!("{}...", response.content.chars().take(100).collect::<String>())
            } else {
                response.content.clone()
            };
            emit(ProgressEvent::LlmCallComplete {
                step,
                duration_ms: llm_ms,
                prompt_tokens: iter_tokens.prompt_tokens,
                completion_tokens: iter_tokens.completion_tokens,
                response_preview,
            });

            debug!(content_len = response.content.len(), "Got LLM response");

            // Try to extract JSON commands first
            let commands_json = extract_commands(&response.content);

            // Store LLM response for history
            let llm_response = response.content.clone();

            if let Some(json) = commands_json {
                emit(ProgressEvent::CommandsExtracted {
                    step,
                    commands: json.clone(),
                });

                // Check if this might involve code compilation
                let is_cli = json.contains("rust_cli_intent");
                let is_wasm = json.contains("rust_wasm");
                if is_cli {
                    // CLI: emit codegen start (LLM call to generate code)
                    emit(ProgressEvent::CliCodegenStart { step });
                } else if is_wasm {
                    emit(ProgressEvent::WasmCompileStart { step });
                }

                // Execute the commands (with timing)
                let exec_start = Instant::now();
                let exec_result = executor.execute_json(&json);
                let exec_ms = exec_start.elapsed().as_millis() as u64;
                let compile_ms = executor.last_compile_time_ms();
                let codegen_ms = executor.last_codegen_time_ms();

                // Emit CLI codegen complete and compile complete separately
                if is_cli {
                    if codegen_ms > 0 {
                        emit(ProgressEvent::CliCodegenComplete {
                            step,
                            duration_ms: codegen_ms,
                        });
                    }
                    // Emit compile start now (after codegen)
                    if compile_ms > 0 {
                        emit(ProgressEvent::CliCompileStart { step });
                        emit(ProgressEvent::CliCompileComplete {
                            step,
                            duration_ms: compile_ms,
                        });
                    }
                } else if is_wasm && compile_ms > 0 {
                    emit(ProgressEvent::WasmCompileComplete {
                        step,
                        duration_ms: compile_ms,
                    });
                }

                let wasm_run_ms = executor.last_wasm_run_time_ms();
                let cli_run_ms = executor.last_cli_run_time_ms();
                let timing = IterationTiming {
                    llm_ms,
                    exec_ms,
                    compile_ms,
                    wasm_run_ms,
                };

                // Emit run complete events - separate CLI and WASM
                if cli_run_ms > 0 {
                    emit(ProgressEvent::CliRunComplete {
                        step,
                        duration_ms: cli_run_ms,
                    });
                }
                if wasm_run_ms > 0 {
                    emit(ProgressEvent::WasmRunComplete {
                        step,
                        duration_ms: wasm_run_ms,
                    });
                }

                match exec_result {
                    Ok(ExecutionResult::Final { answer, sub_calls }) => {
                        let record = IterationRecord {
                            step,
                            llm_response,
                            commands: json,
                            output: format!("FINAL: {}", &answer),
                            error: None,
                            sub_calls,
                            tokens: iter_tokens,
                            timing,
                        };
                        emit(ProgressEvent::CommandComplete {
                            step,
                            output_preview: format!("FINAL: {}", &answer),
                            exec_ms,
                        });
                        emit(ProgressEvent::IterationComplete {
                            step,
                            record: record.clone(),
                        });
                        emit(ProgressEvent::FinalAnswer {
                            answer: answer.clone(),
                        });
                        emit(ProgressEvent::Complete {
                            iterations: step,
                            success: true,
                            total_duration_ms: query_start.elapsed().as_millis() as u64,
                        });
                        history.push(record);

                        return Ok(RlmResult {
                            answer,
                            iterations: step,
                            history,
                            total_sub_calls: total_sub_calls.load(Ordering::Relaxed),
                            success: true,
                            error: None,
                            total_prompt_tokens,
                            total_completion_tokens,
                            context_chars,
                            bypassed: false,
                        });
                    }
                    Ok(ExecutionResult::Continue { output, sub_calls }) => {
                        let truncated_output = self.truncate_output(&output);
                        let output_preview = if output.chars().count() > 100 {
                            format!("{}...", output.chars().take(100).collect::<String>())
                        } else {
                            output.clone()
                        };
                        emit(ProgressEvent::CommandComplete {
                            step,
                            output_preview,
                            exec_ms,
                        });
                        let record = IterationRecord {
                            step,
                            llm_response,
                            commands: json,
                            output: truncated_output,
                            error: None,
                            sub_calls,
                            tokens: iter_tokens,
                            timing,
                        };
                        emit(ProgressEvent::IterationComplete {
                            step,
                            record: record.clone(),
                        });
                        history.push(record);
                    }
                    Err(e) => {
                        emit(ProgressEvent::CommandComplete {
                            step,
                            output_preview: format!("ERROR: {}", e),
                            exec_ms,
                        });
                        let record = IterationRecord {
                            step,
                            llm_response,
                            commands: json,
                            output: String::new(),
                            error: Some(e.to_string()),
                            sub_calls: 0,
                            tokens: iter_tokens,
                            timing,
                        };
                        emit(ProgressEvent::IterationComplete {
                            step,
                            record: record.clone(),
                        });
                        history.push(record);
                        debug!("Command execution had error: {}", e);
                    }
                }
            } else {
                // No JSON commands - check for FINAL in plain text (fallback)
                let timing = IterationTiming {
                    llm_ms,
                    exec_ms: 0,
                    compile_ms: 0,
                    wasm_run_ms: 0,
                };

                if let Some(final_answer) = extract_final(&response.content) {
                    let record = IterationRecord {
                        step,
                        llm_response,
                        commands: String::new(),
                        output: format!("FINAL: {}", &final_answer),
                        error: None,
                        sub_calls: 0,
                        tokens: iter_tokens,
                        timing,
                    };
                    emit(ProgressEvent::FinalAnswer {
                        answer: final_answer.clone(),
                    });
                    emit(ProgressEvent::IterationComplete {
                        step,
                        record: record.clone(),
                    });
                    emit(ProgressEvent::Complete {
                        iterations: step,
                        success: true,
                        total_duration_ms: query_start.elapsed().as_millis() as u64,
                    });
                    history.push(record);
                    return Ok(RlmResult {
                        answer: final_answer,
                        iterations: step,
                        history,
                        total_sub_calls: total_sub_calls.load(Ordering::Relaxed),
                        success: true,
                        error: None,
                        total_prompt_tokens,
                        total_completion_tokens,
                        context_chars,
                        bypassed: false,
                    });
                }

                // No commands and no FINAL - log and continue
                warn!("No commands found in response, continuing");
                let record = IterationRecord {
                    step,
                    llm_response: response.content.clone(),
                    commands: String::new(),
                    output: String::new(),
                    error: Some("No JSON commands found in response".to_string()),
                    sub_calls: 0,
                    tokens: iter_tokens,
                    timing,
                };
                emit(ProgressEvent::IterationComplete {
                    step,
                    record: record.clone(),
                });
                history.push(record);
            }
        }

        // Max iterations reached - still return token info
        emit(ProgressEvent::Complete {
            iterations: self.config.max_iterations,
            success: false,
            total_duration_ms: query_start.elapsed().as_millis() as u64,
        });
        Ok(RlmResult {
            answer: String::new(),
            iterations: self.config.max_iterations,
            history,
            total_sub_calls: total_sub_calls.load(Ordering::Relaxed),
            success: false,
            error: Some(format!(
                "Max iterations ({}) exceeded",
                self.config.max_iterations
            )),
            total_prompt_tokens,
            total_completion_tokens,
            context_chars,
            bypassed: false,
        })
    }

    fn build_system_prompt(&self, context_len: usize) -> String {
        // Build level status strings
        let dsl_status = if self.config.dsl.enabled { "ENABLED" } else { "DISABLED" };
        let wasm_status = if self.config.wasm.enabled { "ENABLED" } else { "DISABLED" };
        let cli_status = if self.config.cli.enabled { "ENABLED" } else { "DISABLED" };
        let llm_status = if self.config.llm_delegation.enabled { "ENABLED" } else { "DISABLED" };

        // Build available commands section based on enabled levels
        let mut commands_section = String::new();

        // Level 1: DSL commands (always shown if enabled)
        if self.config.dsl.enabled {
            commands_section.push_str(r#"
### Level 1: DSL Operations (text extraction and search) [ENABLED]
- {{"op": "slice", "start": 0, "end": 1000}} - Get characters [start:end]
- {{"op": "lines", "start": 0, "end": 100}} - Get lines [start:end]
- {{"op": "len"}} - Get context length
- {{"op": "count", "what": "lines"}} - Count lines/chars/words
- {{"op": "regex", "pattern": "class \\w+"}} - Find regex matches
- {{"op": "find", "text": "error"}} - Find text occurrences
- {{"op": "set", "name": "x", "value": "..."}} - Set variable
- {{"op": "get", "name": "x"}} - Get variable value

DSL is for: Finding/extracting specific text, counting occurrences of ONE pattern
DSL CANNOT: Count frequencies of MULTIPLE items, compute statistics, sort results
"#);
        }

        // Level 2: WASM commands
        if self.config.wasm.enabled {
            commands_section.push_str(r#"
### Level 2: WASM Computation (sandboxed code execution) [ENABLED]

TWO WASM OPERATIONS - choose the right one:

1. rust_wasm_mapreduce - SIMPLE per-line extraction + ONE aggregation:
   {{"op": "rust_wasm_mapreduce", "intent": "EXTRACT something from each line", "combiner": "count|unique|sum", "store": "result"}}
   Use ONLY for: simple frequency OR simple unique count OR simple sum (ONE thing)

2. rust_wasm_intent - COMPLEX queries, multiple requirements, sorting, ranking:
   {{"op": "rust_wasm_intent", "intent": "FULL ANALYSIS DESCRIPTION", "store": "result"}}
   Use for: ANY query asking for multiple things, top N, ranking, percentiles, sorting

CHOOSING - IMPORTANT:
- Query asks for ONE simple thing (just unique count OR just frequency) -> mapreduce
- Query asks for MULTIPLE things or ranking -> rust_wasm_intent (ALWAYS)
- "unique count AND top N" / "how many AND list" -> rust_wasm_intent
- "top N most active/common" -> rust_wasm_intent
- "percentile" / "sort" / "rank" -> rust_wasm_intent

EXAMPLES:
- "How many unique IPs?" -> mapreduce combiner="unique"
- "Count each error type" -> mapreduce combiner="count"
- "How many unique IPs AND list top 10" -> rust_wasm_intent (TWO requirements!)
- "Top 10 most active IPs" -> rust_wasm_intent (needs count + sort)
"#);
        }

        // Level 3: CLI commands
        if self.config.cli.enabled {
            commands_section.push_str(r#"
### Level 3: CLI Computation (native binary, PREFERRED for analysis) [ENABLED]
- {{"op": "rust_cli_intent", "intent": "what to compute", "store": "result"}}
  Full Rust stdlib (HashMap, HashSet, contains, split, sort). Fast and reliable.

USE CLI FOR:
- Large datasets (1000+ lines) - WASM may timeout
- Complex aggregations (frequency counting, top-N ranking)
- Operations needing HashMap/HashSet
- Sorting, percentiles, statistics
- When WASM fails or produces errors

WASM LIMITATIONS (why CLI is better for complex tasks):
- No HashMap/HashSet (use custom byte-level helpers)
- 64MB memory limit
- Fuel-based instruction limits
- String methods can panic (TwoWaySearcher issue)
"#);
        }

        // Level 4: LLM Delegation
        if self.config.llm_delegation.enabled {
            commands_section.push_str(r#"
### Level 4: LLM Delegation (semantic analysis) [ENABLED]
- {{"op": "llm_query", "prompt": "Analyze: ${{var}}", "store": "result"}}
  IMPORTANT: Sub-LLM has NO access to original context!
  You MUST extract content first, then pass via ${{var}}.
"#);
        }

        // Build workflow section based on available levels
        let workflow = if self.config.cli.enabled {
            r#"
## Workflow
1. **Simple queries**: Use find/regex/lines to extract, then final
2. **Aggregation/analysis**: Use rust_cli_intent (PREFERRED) for counting, ranking
3. **Semantic analysis**: Extract content first, then use llm_query

## Dataset Size Guidance
- Small (<500 lines): DSL commands (find, regex, lines) work well
- Medium (500-2000 lines): WASM works but CLI is faster
- Large (2000+ lines): ALWAYS use rust_cli_intent - WASM may timeout

## Example: Frequency Analysis (use rust_cli_intent)

```json
{{"op": "rust_cli_intent", "intent": "Count each error type and rank by frequency", "store": "counts"}}
{{"op": "final_var", "name": "counts"}}
```

## Example: Top-N with Percentiles

```json
{{"op": "rust_cli_intent", "intent": "Find top 10 most active IPs and calculate p50, p95, p99 response times", "store": "analysis"}}
{{"op": "final_var", "name": "analysis"}}
```"#
        } else if self.config.wasm.enabled {
            r#"
## Workflow
1. **Simple queries**: Use find/regex/lines to extract, then final
2. **ONE simple aggregation**: Use rust_wasm_mapreduce
3. **Complex/multi-part queries**: Use rust_wasm_intent (handles everything in one shot)

## Examples

Simple unique count (ONE thing):
```json
{{"op": "rust_wasm_mapreduce", "intent": "Extract the word after 'from ' from each line", "combiner": "unique", "store": "result"}}
{{"op": "final_var", "name": "result"}}
```

Combined query - unique count AND top 10 (use rust_wasm_intent for MULTIPLE requirements):
```json
{{"op": "rust_wasm_intent", "intent": "Extract IP after 'from ', count unique IPs, also count frequency of each IP and return top 10 most active", "store": "result"}}
{{"op": "final_var", "name": "result"}}
```

Percentiles (needs sorting - use rust_wasm_intent):
```json
{{"op": "rust_wasm_intent", "intent": "Extract the number before 'ms' from each line, sort all numbers, compute p50 p95 p99 percentiles", "store": "result"}}
{{"op": "final_var", "name": "result"}}
```"#
        } else {
            r#"
## Workflow
1. Use find/regex/lines to extract relevant content
2. Use count for simple counting
3. Return result with final

## Example: Finding Content

```json
{{"op": "find", "text": "error", "store": "matches"}}
{{"op": "final_var", "name": "matches"}}
```"#
        };

        format!(
            r#"You are an RLM (Recursive Language Model) agent that answers queries about large contexts.

The context has {context_len} characters. You interact with it using JSON commands.

## Capability Levels
- Level 1 (DSL): Text extraction, filtering [{dsl_status}]
- Level 2 (WASM): Sandboxed computation [{wasm_status}]
- Level 3 (CLI): Native binary execution [{cli_status}]
- Level 4 (LLM): Semantic analysis delegation [{llm_status}]

Use the LOWEST level that can accomplish the task.

## Available Commands
{commands_section}

### Finishing
- {{"op": "final", "answer": "The result is..."}} - Simple text answers
- {{"op": "final_var", "name": "result"}} - Output a computed variable (PREFERRED!)
{workflow}

## CRITICAL: Command Format
- Output ONE command per response (only the first JSON block is executed)
- After each command, wait for its output before issuing the next command
- Variables persist across iterations

## Important Notes
- Always use `store` to save results you need later
- Use final_var for computed results
- If WASM fails, simplify the intent description"#,
            context_len = context_len,
            dsl_status = dsl_status,
            wasm_status = wasm_status,
            cli_status = cli_status,
            llm_status = llm_status,
            commands_section = commands_section,
            workflow = workflow,
        )
    }

    fn build_prompt(&self, query: &str, context: &str, history: &[IterationRecord]) -> String {
        // Only include first/last of context if it's large (UTF-8 safe)
        let context_preview = if context.chars().count() > 2000 {
            let first_500: String = context.chars().take(500).collect();
            let last_500: String = context.chars().rev().take(500).collect::<Vec<_>>().into_iter().rev().collect();
            format!(
                "[First 500 chars]\n{}\n\n[Last 500 chars]\n{}",
                first_500,
                last_500
            )
        } else {
            context.to_string()
        };

        let mut prompt = format!(
            "QUERY: {}\n\nCONTEXT PREVIEW ({} total chars):\n{}\n\n",
            query,
            context.len(),
            context_preview
        );

        if !history.is_empty() {
            prompt.push_str("EXECUTION HISTORY:\n");
            for record in history {
                prompt.push_str(&format!("\n--- Step {} ---\n", record.step));
                if !record.commands.is_empty() {
                    prompt.push_str(&format!("Commands:\n```json\n{}\n```\n", record.commands));
                }
                if let Some(error) = &record.error {
                    prompt.push_str(&format!("Error: {}\n", error));
                } else if !record.output.is_empty() {
                    prompt.push_str(&format!("Output:\n{}\n", record.output));
                }
                if record.sub_calls > 0 {
                    prompt.push_str(&format!("(Made {} sub-LM calls)\n", record.sub_calls));
                }
            }
            prompt.push_str(
                "\nContinue analysis. Use {{\"op\": \"final\", \"answer\": \"...\"}} when done.\n",
            );
        }

        prompt
    }

    fn truncate_output(&self, output: &str) -> String {
        if output.len() > self.config.output_limit {
            let half = self.config.output_limit / 2;
            format!(
                "{}... [truncated {} chars] ...{}",
                &output[..half],
                output.len() - self.config.output_limit,
                &output[output.len() - half..]
            )
        } else {
            output.to_string()
        }
    }
}
