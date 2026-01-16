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
    /// Starting a new iteration
    IterationStart { step: usize },
    /// LLM call starting
    LlmCallStart { step: usize },
    /// LLM call completed
    LlmCallComplete {
        step: usize,
        duration_ms: u64,
        response_preview: String,
    },
    /// Commands extracted and about to execute
    CommandsExtracted { step: usize, commands: String },
    /// WASM compilation starting
    WasmCompileStart { step: usize },
    /// WASM compilation complete
    WasmCompileComplete { step: usize, duration_ms: u64 },
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
    Complete { iterations: usize, success: bool },
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
        self.process_with_progress(query, context, None).await
    }

    /// Process a query with optional progress callbacks for real-time feedback
    pub async fn process_with_progress(
        &self,
        query: &str,
        context: &str,
        progress: Option<ProgressCallback>,
    ) -> Result<RlmResult, OrchestratorError> {
        // Helper to emit progress events
        let emit = |event: ProgressEvent| {
            if let Some(ref cb) = progress {
                cb(event);
            }
        };

        // Check if we should bypass RLM for small contexts
        if self.should_bypass(context.len()) {
            return self.process_direct(query, context).await;
        }

        let mut history = Vec::new();
        let total_sub_calls = Arc::new(AtomicUsize::new(0));
        let mut total_prompt_tokens: u32 = 0;
        let mut total_completion_tokens: u32 = 0;
        let context_chars = context.len();

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
                });
                return Err(OrchestratorError::MaxSubCalls(self.config.max_sub_calls));
            }

            // Build the prompt with history
            let prompt = self.build_prompt(query, context, &history);

            // Call the root LLM (with timing)
            emit(ProgressEvent::LlmCallStart { step });
            let llm_start = Instant::now();
            let request = LlmRequest::new(&system_prompt, &prompt);
            let response = self.pool.complete(&request, false).await?;
            let llm_ms = llm_start.elapsed().as_millis() as u64;

            // Emit LLM complete with preview (UTF-8 safe truncation)
            let response_preview = if response.content.chars().count() > 100 {
                format!("{}...", response.content.chars().take(100).collect::<String>())
            } else {
                response.content.clone()
            };
            emit(ProgressEvent::LlmCallComplete {
                step,
                duration_ms: llm_ms,
                response_preview,
            });

            // Track token usage
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

                // Check if this might involve WASM compilation
                if json.contains("rust_wasm") {
                    emit(ProgressEvent::WasmCompileStart { step });
                }

                // Execute the commands (with timing)
                let exec_start = Instant::now();
                let exec_result = executor.execute_json(&json);
                let exec_ms = exec_start.elapsed().as_millis() as u64;
                let compile_ms = executor.last_compile_time_ms();

                // Emit compile complete if there was compilation
                if compile_ms > 0 {
                    emit(ProgressEvent::WasmCompileComplete {
                        step,
                        duration_ms: compile_ms,
                    });
                }

                let timing = IterationTiming {
                    llm_ms,
                    exec_ms,
                    compile_ms,
                };

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
        format!(
            r#"You are an RLM (Recursive Language Model) agent that answers queries about large contexts.

The context has {context_len} characters. You interact with it using JSON commands.

## Available Commands

Context operations:
- {{"op": "slice", "start": 0, "end": 1000}} - Get characters [start:end]
- {{"op": "lines", "start": 0, "end": 100}} - Get lines [start:end]
- {{"op": "len"}} - Get context length
- {{"op": "count", "what": "lines"}} - Count lines/chars/words

Search operations:
- {{"op": "regex", "pattern": "class \\w+"}} - Find regex matches
- {{"op": "find", "text": "error"}} - Find text occurrences

Variables:
- {{"op": "set", "name": "x", "value": "..."}} - Set variable
- {{"op": "get", "name": "x"}} - Get variable value
- {{"op": "print", "var": "x"}} - Print variable

Sub-LM calls (for semantic analysis of EXTRACTED content):
- {{"op": "llm_query", "prompt": "Analyze this: ${{extracted_text}}", "store": "analysis"}}
  IMPORTANT: The sub-LLM has NO access to the original context!
  You MUST first extract relevant content using find/regex/lines/slice,
  then include that content in the prompt using variable references like ${{var}}.

WASM (dynamic code execution):
- {{"op": "wasm", "module": "line_counter"}} - Run pre-compiled WASM module
- {{"op": "rust_wasm", "code": "pub fn analyze(input: &str) -> String {{...}}", "store": "result"}}
  Compile and execute custom Rust code. The function receives the FULL CONTEXT as input and returns a String.
- {{"op": "rust_wasm", "code": "...", "on": "$myvar", "store": "result"}}
  With "on": operates on a stored variable instead of the full context.

IMPORTANT for rust_wasm:
- WITHOUT "on:" → input is the FULL original context (usually what you want for log analysis!)
- WITH "on": "$var" → input is the content of that variable
- For analyzing logs/files, just omit "on:" and process the full context directly.

Available WASM modules: line_counter (counts lines in context)

## rust_wasm: Custom Analysis Functions

Use rust_wasm when built-in commands cannot express the analysis you need.

Function signature (REQUIRED):
```rust
pub fn analyze(input: &str) -> String {{
    // Your analysis code here
    // Return result as String
}}
```

Available in your code:
- All core Rust: iterators, pattern matching, string operations
- Collections: HashMap, HashSet, BTreeMap, BTreeSet, VecDeque
- NO: file I/O, network, process spawning, environment access

Example - Word frequency count:
```json
{{"op": "rust_wasm", "code": "pub fn analyze(input: &str) -> String {{ let mut counts: HashMap<&str, usize> = HashMap::new(); for word in input.split_whitespace() {{ *counts.entry(word).or_insert(0) += 1; }} let mut pairs: Vec<_> = counts.into_iter().collect(); pairs.sort_by(|a, b| b.1.cmp(&a.1)); pairs.iter().take(10).map(|(w, c)| format!(\"{{}}: {{}}\", w, c)).collect::<Vec<_>>().join(\", \") }}", "store": "top_words"}}
```

Example - Extract numbers and sum:
```json
{{"op": "rust_wasm", "code": "pub fn analyze(input: &str) -> String {{ let sum: i64 = input.split_whitespace().filter_map(|s| s.parse::<i64>().ok()).sum(); sum.to_string() }}", "on": "$data", "store": "total"}}
```

Example - Custom pattern extraction:
```json
{{"op": "rust_wasm", "code": "pub fn analyze(input: &str) -> String {{ input.lines().filter(|line| line.contains(\"ERROR\") && line.contains(\"timeout\")).count().to_string() }}", "store": "timeout_errors"}}
```

Example - Count items by category (HashMap aggregation):
```json
{{"op": "rust_wasm", "code": "pub fn analyze(input: &str) -> String {{ let mut counts: HashMap<&str, usize> = HashMap::new(); for line in input.lines() {{ if line.contains(\"ERROR\") {{ let cat = if line.contains(\"Timeout\") {{ \"Timeout\" }} else if line.contains(\"Auth\") {{ \"Auth\" }} else {{ \"Other\" }}; *counts.entry(cat).or_insert(0) += 1; }} }} let mut v: Vec<_> = counts.into_iter().collect(); v.sort_by(|a,b| b.1.cmp(&a.1)); v.iter().map(|(k,c)| format!(\"{{}}:{{}}\", k, c)).collect::<Vec<_>>().join(\", \") }}", "store": "error_counts"}}
{{"op": "final_var", "name": "error_counts"}}
```
Note: No "on:" parameter - processes the FULL context directly. Uses final_var to output the result.

When to use rust_wasm vs built-in commands:
- Use find/regex for simple searches and pattern matching
- Use count for basic counting (lines, words, chars)
- Use rust_wasm for: aggregations, frequency analysis, custom filtering, numeric computations, multi-step transformations

Finishing:
- {{"op": "final", "answer": "The result is..."}} - Use for simple text answers
- {{"op": "final_var", "name": "result"}} - Use to output a computed variable (PREFERRED for rust_wasm results!)

IMPORTANT: Use final_var when your answer is stored in a variable from rust_wasm or other computation.
Do NOT try to interpolate variables in final's answer string - use final_var instead.

## Variable References
- Variables persist across iterations - no need to re-extract data you already have!
- Use $var in the "on" parameter: {{"op": "rust_wasm", ..., "on": "$mydata"}}
- Variable interpolation in strings (${{var}}) has limited support - prefer final_var for outputs.

## Workflow (MANDATORY: SEARCH FIRST!)

CRITICAL RULE: You MUST search the context BEFORE concluding anything doesn't exist.
Never assume - always verify with find/regex commands first!

1. SEARCH: Use find/regex with SIMPLE keywords (1-2 words). Start broad, then narrow.
   - If searching for "Prince Andrei's secret vault" → try just "secret vault" or "password"
   - If first search returns 0 matches, TRY DIFFERENT KEYWORDS
   - Use the most distinctive word from the query (e.g., "vault", "password", "secret")

2. Extract: Use lines/slice to get content around matches
3. Analyze: If needed, use llm_query to analyze EXTRACTED content (pass it via ${{var}})
4. Finish: Use final with your answer

IMPORTANT:
- Always search/extract content BEFORE using llm_query.
- The sub-LLM in llm_query cannot see the original document - you must pass extracted text to it.
- NEVER give up after one failed search - try at least 2-3 different search terms!

## Example: Finding specific content

```json
{{"op": "find", "text": "secret", "store": "matches"}}
```
Then examine results and extract context:
```json
{{"op": "lines", "start": 230, "end": 240, "store": "context"}}
{{"op": "final", "answer": "Found: ${{context}}"}}
```

## Example: Using llm_query correctly

```json
{{"op": "find", "text": "error", "store": "error_lines"}}
{{"op": "llm_query", "prompt": "Categorize these errors: ${{error_lines}}", "store": "analysis"}}
{{"op": "final_var", "name": "analysis"}}
```
Note: The extracted ${{error_lines}} is passed TO the sub-LLM in the prompt.

## Important Notes
- Variables store strings and PERSIST across iterations - don't re-extract data you already have!
- The `regex` command stores matched text (one per line). Use `count` with `what: "lines"` on that variable.
- Always use `store` to save results you need later.
- Wrap commands in ```json blocks. Execute multiple commands per iteration.
- For aggregation tasks (counting by category, frequency analysis), use rust_wasm with HashMap on the FULL context.
- Prefer final_var over final with variable interpolation for computed results."#,
            context_len = context_len
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
