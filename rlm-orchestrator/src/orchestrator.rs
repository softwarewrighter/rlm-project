//! RLM orchestration logic

use crate::RlmConfig;
use crate::commands::{
    CommandExecutor, ExecutionResult, LlmDelegateCallback, LlmDelegateParams, LlmDelegateResult,
    LlmQueryCallback, NestedIterationSummary, extract_commands, extract_final,
};
use crate::pool::LlmPool;
use crate::provider::{LlmRequest, ProviderError};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Safe UTF-8 string truncation
fn truncate_to_char_boundary(s: &str, max_bytes: usize) -> &str {
    if max_bytes >= s.len() {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

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
    /// LLM delegation starting (nested RLM)
    LlmDelegateStart {
        step: usize,
        task_preview: String,
        context_len: usize,
        depth: usize,
    },
    /// LLM delegation complete (nested RLM)
    LlmDelegateComplete {
        step: usize,
        duration_ms: u64,
        nested_iterations: usize,
        success: bool,
    },
    /// Nested RLM iteration (shows what worker is doing)
    NestedIteration {
        depth: usize,
        step: usize,
        llm_response_preview: String,
        commands_preview: String,
        output_preview: String,
        has_error: bool,
    },
    /// LLM reduce starting (chunked processing)
    LlmReduceStart {
        step: usize,
        num_chunks: usize,
        total_chars: usize,
        directive_preview: String,
    },
    /// LLM reduce chunk starting
    LlmReduceChunkStart {
        step: usize,
        chunk_num: usize,
        total_chunks: usize,
        chunk_chars: usize,
    },
    /// LLM reduce chunk complete
    LlmReduceChunkComplete {
        step: usize,
        chunk_num: usize,
        total_chunks: usize,
        duration_ms: u64,
        result_preview: String,
    },
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
    /// Phased processing starting (for large contexts)
    PhasedStart {
        context_chars: usize,
        phase_count: usize,
    },
    /// Starting a specific phase
    PhaseStart {
        phase: usize,
        name: String,
        description: String,
    },
    /// Phase completed
    PhaseComplete {
        phase: usize,
        name: String,
        duration_ms: u64,
        result_preview: String,
    },
}

/// Callback type for progress events (Arc for cloning into nested callbacks)
pub type ProgressCallback = Arc<dyn Fn(ProgressEvent) + Send + Sync>;

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

    /// Create the llm_delegate callback for nested RLM calls
    fn create_llm_delegate_callback(
        &self,
        total_sub_calls: Arc<AtomicUsize>,
    ) -> LlmDelegateCallback {
        let pool = Arc::clone(&self.pool);
        let config = self.config.clone();
        let max_sub_calls = self.config.max_sub_calls;
        let max_recursion_depth = self.config.llm_delegation.max_recursion_depth;

        Arc::new(move |params: LlmDelegateParams| {
            // Check recursion depth
            if params.current_depth > max_recursion_depth {
                return Err(format!(
                    "Max recursion depth ({}) exceeded",
                    max_recursion_depth
                ));
            }

            // Check max sub-calls
            let current = total_sub_calls.load(Ordering::Relaxed);
            if current >= max_sub_calls {
                return Err(format!("Max sub-calls ({}) exceeded", max_sub_calls));
            }

            // Get the current tokio runtime handle
            let handle = tokio::runtime::Handle::try_current()
                .map_err(|e| format!("No tokio runtime available: {}", e))?;

            let pool = Arc::clone(&pool);
            let config = config.clone();
            let total_sub_calls = Arc::clone(&total_sub_calls);
            let task = params.task.clone();
            let context = params.context.clone();
            let max_iterations = params.max_iterations;
            let current_depth = params.current_depth;
            let levels = params.levels.clone();

            // Use block_in_place to allow blocking within the async runtime
            let result = tokio::task::block_in_place(|| {
                handle.block_on(async {
                    // Create a nested orchestrator with restricted config
                    let mut nested_config = config.clone();
                    nested_config.max_iterations = max_iterations;
                    nested_config.level_priority = levels.clone();

                    // Disable llm_delegate in nested instances to prevent infinite recursion
                    // Keep llm_query enabled for simple semantic checks
                    nested_config.llm_delegation.enabled = true; // Keep enabled for llm_query

                    let nested_orchestrator = RlmOrchestrator::new(nested_config, pool);

                    // Run the nested RLM with depth tracking
                    // Note: We force RLM mode for nested calls (no bypass)
                    nested_orchestrator
                        .process_nested(&task, &context, current_depth, &levels)
                        .await
                })
            });

            total_sub_calls.fetch_add(1, Ordering::Relaxed);

            match result {
                Ok(rlm_result) => {
                    // Convert history to nested iteration summaries
                    let nested_history: Vec<NestedIterationSummary> = rlm_result
                        .history
                        .iter()
                        .map(|record| NestedIterationSummary {
                            step: record.step,
                            llm_response_preview: record.llm_response.chars().take(100).collect(),
                            commands_preview: record.commands.chars().take(100).collect(),
                            output_preview: record.output.chars().take(100).collect(),
                            has_error: record.error.is_some(),
                        })
                        .collect();

                    Ok(LlmDelegateResult {
                        answer: rlm_result.answer,
                        iterations: rlm_result.iterations,
                        success: rlm_result.success,
                        nested_history,
                    })
                }
                Err(e) => Err(e.to_string()),
            }
        })
    }

    /// Process a nested RLM call (called from llm_delegate)
    async fn process_nested(
        &self,
        query: &str,
        context: &str,
        depth: usize,
        _levels: &[String],
    ) -> Result<RlmResult, OrchestratorError> {
        info!(
            depth = depth,
            context_len = context.len(),
            query_len = query.len(),
            "Processing nested RLM call"
        );

        let mut history = Vec::new();
        let total_sub_calls = Arc::new(AtomicUsize::new(0));
        let mut total_prompt_tokens: u32 = 0;
        let mut total_completion_tokens: u32 = 0;
        let context_chars = context.len();

        // Build system prompt for nested context (simpler, focused on the task)
        let system_prompt = self.build_nested_system_prompt(context.len());

        // Create callbacks (no delegation in nested calls to prevent infinite recursion)
        let llm_query_cb = self.create_llm_query_callback(Arc::clone(&total_sub_calls));

        // Create command executor without delegation callback (prevents nested delegation)
        let mut executor = CommandExecutor::with_wasm_config(
            context.to_string(),
            self.config.max_sub_calls,
            &self.config.wasm,
        )
        .with_llm_callback(llm_query_cb)
        .with_recursion_depth(depth)
        .with_max_recursion_depth(self.config.llm_delegation.max_recursion_depth);

        // Set max chunks for llm_reduce based on iteration limit (fail fast if too large)
        executor.set_max_llm_reduce_chunks(self.config.max_iterations);

        for iteration in 0..self.config.max_iterations {
            let step = iteration + 1;
            debug!(iteration = step, depth = depth, "Nested RLM iteration");

            // Build the prompt with history
            let prompt = self.build_prompt(query, context, &history);

            // Call the LLM
            let request = LlmRequest::new(&system_prompt, &prompt).with_max_tokens(4096);
            let response = self.pool.complete(&request, false).await?;

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

            let llm_response = response.content.clone();

            // Log the worker LLM response for debugging
            debug!(
                depth = depth,
                step = step,
                response_len = response.content.len(),
                response_preview = %response.content.chars().take(200).collect::<String>(),
                "Worker LLM response"
            );

            // Try to extract JSON commands
            let commands_json = extract_commands(&response.content);

            debug!(
                depth = depth,
                step = step,
                has_commands = commands_json.is_some(),
                "Worker command extraction"
            );

            if let Some(json) = commands_json {
                let exec_result = executor.execute_json(&json);

                match exec_result {
                    Ok(ExecutionResult::Final {
                        answer,
                        sub_calls: _,
                    }) => {
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
                        let record = IterationRecord {
                            step,
                            llm_response,
                            commands: json,
                            output: self.truncate_output(&output),
                            error: None,
                            sub_calls,
                            tokens: iter_tokens,
                            timing: IterationTiming::default(),
                        };
                        history.push(record);
                    }
                    Err(e) => {
                        let record = IterationRecord {
                            step,
                            llm_response,
                            commands: json,
                            output: String::new(),
                            error: Some(e.to_string()),
                            sub_calls: 0,
                            tokens: iter_tokens,
                            timing: IterationTiming::default(),
                        };
                        history.push(record);
                    }
                }
            } else {
                // Check for FINAL in plain text
                if let Some(final_answer) = extract_final(&response.content) {
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

                let record = IterationRecord {
                    step,
                    llm_response: response.content,
                    commands: String::new(),
                    output: String::new(),
                    error: Some("No JSON commands found".to_string()),
                    sub_calls: 0,
                    tokens: iter_tokens,
                    timing: IterationTiming::default(),
                };
                history.push(record);
            }
        }

        // Max iterations reached
        Ok(RlmResult {
            answer: String::new(),
            iterations: self.config.max_iterations,
            history,
            total_sub_calls: total_sub_calls.load(Ordering::Relaxed),
            success: false,
            error: Some(format!(
                "Nested RLM: Max iterations ({}) exceeded",
                self.config.max_iterations
            )),
            total_prompt_tokens,
            total_completion_tokens,
            context_chars,
            bypassed: false,
        })
    }

    /// Build system prompt for nested RLM calls (simpler than root)
    fn build_nested_system_prompt(&self, context_len: usize) -> String {
        let nested_levels = &self.config.llm_delegation.nested_levels;

        let mut commands = String::new();

        // Add DSL commands if enabled
        if nested_levels.contains(&"dsl".to_string()) || self.config.dsl.enabled {
            commands.push_str(
                r#"
## Level 1: DSL (Text Operations)
- {"op": "regex", "pattern": "...", "store": "matches"} - Find patterns
- {"op": "find", "text": "...", "store": "results"} - Find text
- {"op": "lines", "start": N, "end": M} - Get line range
- {"op": "count", "what": "lines"} - Count lines/chars/words
- {"op": "len"} - Get context length
"#,
            );
        }

        // Add CLI commands if enabled in nested_levels
        if nested_levels.contains(&"cli".to_string()) && self.config.cli.enabled {
            commands.push_str(
                r#"
## Level 3: CLI (Native Computation) - PREFERRED for complex analysis
- {"op": "rust_cli_intent", "intent": "...", "store": "result"} - Run native code

USE rust_cli_intent FOR:
- Counting frequencies, unique values, top-N rankings
- Statistics (percentiles, averages)
- Complex aggregations
- Large data processing
"#,
            );
        }

        // Add llm_query for semantic checks
        commands.push_str(
            r#"
## Semantic Check
- {"op": "llm_query", "prompt": "...", "store": "result"} - Simple yes/no question

## Finishing
- {"op": "final", "answer": "..."} - Return your analysis
- {"op": "final_var", "name": "result"} - Return computed variable
"#,
        );

        format!(
            r#"You are a WORKER RLM agent analyzing data.
The context has {context_len} characters.

CRITICAL: You have LIMITED iterations (max 10). Work efficiently!

{commands}

## WORKFLOW (follow this order)

### Step 1: Explore the Data First (REQUIRED)
Before extracting, ALWAYS peek at the data structure:
- Use {{"op": "lines", "start": 1, "end": 50}} to see the beginning
- Use {{"op": "count", "what": "lines"}} to understand size
- Look for section markers, headers, dividers

### Step 2: Extract with Flexible Patterns
If first pattern fails, TRY ALTERNATIVES:
- Broad pattern first: {{"op": "regex", "pattern": "witness|statement"}}
- Then narrow down: {{"op": "regex", "pattern": "WITNESS.*?:"}}
- Use find for exact text: {{"op": "find", "text": "Evidence"}}

### Step 3: Analyze and Summarize
- Use CLI for computation: {{"op": "rust_cli_intent", "intent": "count unique names"}}
- Use llm_query for semantic checks: {{"op": "llm_query", "prompt": "Is this reliable?"}}

### Step 4: Return a CONCISE Result
- {{"op": "final", "answer": "Your summary here"}} - Return directly
- {{"op": "final_var", "name": "result"}} - Return stored variable

## CRITICAL RULES
1. EXPLORE before EXTRACT - peek at data structure first
2. If pattern not found, try simpler patterns or substrings
3. DO NOT return raw data - SUMMARIZE your findings
4. Keep final answer under 1000 chars
5. Finish in 3-5 iterations if possible

## DEBUGGING
If no matches found:
1. Check first 50 lines to see actual format
2. Try case-insensitive or partial matches
3. Look for different delimiters (===, ---, ***, etc.)"#
        )
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
        self.process_with_options(query, context, progress, false)
            .await
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

        // Check if we should use phased processing for very large contexts
        // Phased processing guides the LLM through: assess -> index -> select -> retrieve -> analyze
        // Skip if force_rlm is set (to prevent infinite recursion from Phase 5)
        debug!(
            force_rlm = force_rlm,
            context_len = context.len(),
            phased_threshold = self.config.phased_threshold,
            cli_enabled = self.config.cli.enabled,
            "Checking phased processing conditions"
        );
        if !force_rlm && context.len() > self.config.phased_threshold && self.config.cli.enabled {
            info!(
                context_len = context.len(),
                threshold = self.config.phased_threshold,
                "Using phased processing for large context"
            );
            return self.process_phased(query, context, progress.clone()).await;
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
        .with_llm_callback(llm_query_cb)
        .with_recursion_depth(0) // Root level
        .with_max_recursion_depth(self.config.llm_delegation.max_recursion_depth)
        .with_nested_levels(self.config.llm_delegation.nested_levels.clone());

        // Set max chunks for llm_reduce based on iteration limit (fail fast if too large)
        executor.set_max_llm_reduce_chunks(self.config.max_iterations);

        // Add llm_delegate callback if LLM delegation is enabled
        if self.config.llm_delegation.enabled {
            let llm_delegate_cb = self.create_llm_delegate_callback(Arc::clone(&total_sub_calls));
            executor = executor.with_llm_delegate_callback(llm_delegate_cb);
        }

        // Set up progress callback for llm_reduce (tracks current step atomically)
        let current_step = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        if progress.is_some() {
            use crate::commands::CommandProgress;
            // Clone the Arc'd callback for use in the closure
            let progress_arc: Arc<dyn Fn(ProgressEvent) + Send + Sync> = match &progress {
                Some(cb) => Arc::clone(cb),
                None => unreachable!(),
            };
            let step_ref = Arc::clone(&current_step);
            let cmd_progress_cb: crate::commands::CommandProgressCallback =
                Arc::new(move |event: CommandProgress| {
                    let step = step_ref.load(std::sync::atomic::Ordering::Relaxed);
                    match event {
                        CommandProgress::LlmReduceStart {
                            num_chunks,
                            total_chars,
                            directive_preview,
                        } => progress_arc(ProgressEvent::LlmReduceStart {
                            step,
                            num_chunks,
                            total_chars,
                            directive_preview,
                        }),
                        CommandProgress::LlmReduceChunkStart {
                            chunk_num,
                            total_chunks,
                            chunk_chars,
                        } => progress_arc(ProgressEvent::LlmReduceChunkStart {
                            step,
                            chunk_num,
                            total_chunks,
                            chunk_chars,
                        }),
                        CommandProgress::LlmReduceChunkComplete {
                            chunk_num,
                            total_chunks,
                            duration_ms,
                            result_preview,
                        } => progress_arc(ProgressEvent::LlmReduceChunkComplete {
                            step,
                            chunk_num,
                            total_chunks,
                            duration_ms,
                            result_preview,
                        }),
                    }
                });
            executor.set_progress_callback(cmd_progress_cb);
        }

        for iteration in 0..self.config.max_iterations {
            let step = iteration + 1;
            current_step.store(step, std::sync::atomic::Ordering::Relaxed);
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
            let request = LlmRequest::new(&system_prompt, &prompt).with_max_tokens(4096); // Ensure enough tokens for SVG/complex output
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
                format!(
                    "{}...",
                    response.content.chars().take(100).collect::<String>()
                )
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

                // Check if this might involve code compilation or LLM delegation
                let is_cli = json.contains("rust_cli_intent");
                let is_wasm = json.contains("rust_wasm");
                let is_llm_delegate = json.contains("llm_delegate");

                if is_cli {
                    // CLI: emit codegen start (LLM call to generate code)
                    emit(ProgressEvent::CliCodegenStart { step });
                } else if is_wasm {
                    emit(ProgressEvent::WasmCompileStart { step });
                } else if is_llm_delegate {
                    // Extract task preview from JSON for progress event
                    let task_preview =
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json) {
                            parsed
                                .get("task")
                                .and_then(|t| t.as_str())
                                .map(|s| s.chars().take(50).collect::<String>())
                                .unwrap_or_else(|| "...".to_string())
                        } else {
                            "...".to_string()
                        };
                    emit(ProgressEvent::LlmDelegateStart {
                        step,
                        task_preview,
                        context_len: context.len(), // Will be overridden if "on" is used
                        depth: 1,                   // Will be adjusted based on actual depth
                    });
                }

                // Clear nested history before execution
                executor.clear_nested_history();

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

                // Emit nested iteration events for llm_delegate visibility
                let nested_history = executor.last_nested_history();
                if !nested_history.is_empty() {
                    let depth = executor.last_delegate_depth();
                    for summary in nested_history {
                        emit(ProgressEvent::NestedIteration {
                            depth,
                            step: summary.step,
                            llm_response_preview: summary.llm_response_preview.clone(),
                            commands_preview: summary.commands_preview.clone(),
                            output_preview: summary.output_preview.clone(),
                            has_error: summary.has_error,
                        });
                    }
                    // Emit delegate complete with success based on last iteration
                    let success = nested_history.last().map(|s| !s.has_error).unwrap_or(false);
                    emit(ProgressEvent::LlmDelegateComplete {
                        step,
                        duration_ms: exec_ms,
                        nested_iterations: nested_history.len(),
                        success,
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

    /// Process a large context using phased approach:
    /// 1. Assessment: Sample the data to understand format and encoding
    /// 2. Strategy: Develop a reduction strategy based on the query
    /// 3. Reduction: Use CLI/DSL to filter relevant data
    /// 4. Analysis: Process reduced data with LLM delegation
    async fn process_phased(
        &self,
        query: &str,
        context: &str,
        progress: Option<ProgressCallback>,
    ) -> Result<RlmResult, OrchestratorError> {
        let query_start = Instant::now();
        let context_chars = context.len();
        let mut total_prompt_tokens: u32 = 0;
        let mut total_completion_tokens: u32 = 0;
        let total_sub_calls = Arc::new(AtomicUsize::new(0));

        // Helper to emit progress events
        let emit = |event: ProgressEvent| {
            if let Some(ref cb) = progress {
                cb(event);
            }
        };

        emit(ProgressEvent::PhasedStart {
            context_chars,
            phase_count: 5,
        });

        info!(
            context_chars = context_chars,
            query_len = query.len(),
            "Starting phased processing for large context"
        );

        // ========================================================================
        // PHASE 1: ASSESSMENT - Sample the data to understand its format
        // ========================================================================
        emit(ProgressEvent::PhaseStart {
            phase: 1,
            name: "Assessment".to_string(),
            description: "Sampling data to understand format and patterns".to_string(),
        });
        let phase1_start = Instant::now();

        // Sample first 100 lines and middle 50 lines
        let lines: Vec<&str> = context.lines().collect();
        let total_lines = lines.len();
        let first_sample: String = lines
            .iter()
            .take(100)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        let mid_start = total_lines / 2;
        let mid_sample: String = lines
            .iter()
            .skip(mid_start)
            .take(50)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");

        let assessment_prompt = format!(
            r#"You are analyzing a large document ({} chars, {} lines) to understand its format before processing.

SAMPLES FROM THE DOCUMENT:

=== FIRST 100 LINES ===
{}

=== MIDDLE 50 LINES (starting at line {}) ===
{}

QUESTIONS TO ANSWER:
1. What type of document is this? (novel, logs, data file, etc.)
2. What character encoding patterns do you notice? (accented characters, special symbols, etc.)
3. What are the key structural patterns? (chapters, sections, timestamps, etc.)
4. If looking for names/entities, what patterns would identify them? (capitalization, titles like "Prince", "Count", etc.)

Provide a brief assessment (2-3 sentences per question). This will inform how to filter the data."#,
            context_chars, total_lines, first_sample, mid_start, mid_sample
        );

        let assessment_request = LlmRequest::new(
            "You are a document analyst. Analyze the provided samples and answer the questions concisely.",
            &assessment_prompt,
        )
        .with_max_tokens(1024);

        eprintln!("   ⏳ Calling LLM for assessment...");
        let assessment_response = self.pool.complete(&assessment_request, false).await?;
        eprintln!("   ✓ Assessment LLM call complete");
        if let Some(usage) = &assessment_response.usage {
            total_prompt_tokens += usage.prompt_tokens;
            total_completion_tokens += usage.completion_tokens;
        }
        total_sub_calls.fetch_add(1, Ordering::Relaxed);

        let assessment = assessment_response.content.clone();
        let phase1_duration = phase1_start.elapsed().as_millis() as u64;

        emit(ProgressEvent::PhaseComplete {
            phase: 1,
            name: "Assessment".to_string(),
            duration_ms: phase1_duration,
            result_preview: if assessment.len() > 200 {
                format!("{}...", truncate_to_char_boundary(&assessment, 200))
            } else {
                assessment.clone()
            },
        });

        info!(
            duration_ms = phase1_duration,
            "Phase 1 (Assessment) complete"
        );

        // ========================================================================
        // PHASE 2: INDEXING - Build term index using index_terms DSL command
        // ========================================================================
        emit(ProgressEvent::PhaseStart {
            phase: 2,
            name: "Indexing".to_string(),
            description: "Building term index (proper nouns, titles)".to_string(),
        });
        let phase2_start = Instant::now();

        eprintln!("   ⏳ Building proper noun index...");

        // Run index_terms command to extract proper nouns
        let index_cmd = crate::commands::Command::IndexTerms {
            pattern: "proper_nouns".to_string(),
            min_count: 3,         // At least 3 occurrences
            max_terms: 300,       // Top 300 terms
            include_lines: false, // Compact format
            on: None,
            store: Some("term_index".to_string()),
        };

        let mut executor = CommandExecutor::new(context.to_string(), self.config.max_sub_calls);
        let index_result = executor.execute_one(&index_cmd);

        let term_index = match index_result {
            Ok(_) => executor
                .get_variable("term_index")
                .cloned()
                .unwrap_or_else(|| "No terms found".to_string()),
            Err(e) => {
                warn!("Index building failed: {}", e);
                "Index building failed".to_string()
            }
        };

        // Also get titles index
        let titles_cmd = crate::commands::Command::IndexTerms {
            pattern: "titles".to_string(),
            min_count: 1,
            max_terms: 100,
            include_lines: false,
            on: None,
            store: Some("titles_index".to_string()),
        };
        let mut executor2 = CommandExecutor::new(context.to_string(), self.config.max_sub_calls);
        let _ = executor2.execute_one(&titles_cmd);
        let titles_index = executor2
            .get_variable("titles_index")
            .cloned()
            .unwrap_or_default();

        let phase2_duration = phase2_start.elapsed().as_millis() as u64;

        let index_preview = if term_index.len() > 500 {
            format!("{}...", &term_index[..500])
        } else {
            term_index.clone()
        };

        eprintln!("   ✓ Index built: {} chars", term_index.len());

        emit(ProgressEvent::PhaseComplete {
            phase: 2,
            name: "Indexing".to_string(),
            duration_ms: phase2_duration,
            result_preview: index_preview.clone(),
        });

        info!(
            duration_ms = phase2_duration,
            index_size = term_index.len(),
            "Phase 2 (Indexing) complete"
        );

        // ========================================================================
        // PHASE 3: SELECTION - LLM selects relevant terms from index
        // ========================================================================
        emit(ProgressEvent::PhaseStart {
            phase: 3,
            name: "Selection".to_string(),
            description: "LLM selecting relevant terms from index".to_string(),
        });
        let phase3_start = Instant::now();

        let selection_prompt = format!(
            r#"You have a term index from a large document ({} chars).

DOCUMENT ASSESSMENT:
{}

USER'S QUERY:
"{}"

TERM INDEX (proper nouns sorted by frequency):
{}

TITLED CHARACTERS:
{}

YOUR TASK:
Select the terms that are MOST RELEVANT to answering the query.

OUTPUT FORMAT - List ONLY the relevant terms, one per line:
SELECTED_TERMS:
term1
term2
term3
...

GUIDELINES:
- For family tree queries: select character names that appear frequently (>10 occurrences)
- Include variations of names (e.g., both "Rostóv" and "Natásha")
- Include relationship terms if present (father, mother, wife, son, daughter)
- Aim for 20-50 key terms that will capture relevant passages
- Be selective - we'll extract only lines containing these terms"#,
            context_chars,
            assessment,
            query,
            if term_index.len() > 8000 {
                &term_index[..8000]
            } else {
                &term_index
            },
            if titles_index.len() > 2000 {
                &titles_index[..2000]
            } else {
                &titles_index
            }
        );

        let selection_request = LlmRequest::new(
            "You are selecting relevant terms from an index. Output only the SELECTED_TERMS list.",
            &selection_prompt,
        )
        .with_max_tokens(1024);

        eprintln!("   ⏳ LLM selecting relevant terms...");
        let selection_response = self.pool.complete(&selection_request, false).await?;
        eprintln!("   ✓ Term selection complete");

        if let Some(usage) = &selection_response.usage {
            total_prompt_tokens += usage.prompt_tokens;
            total_completion_tokens += usage.completion_tokens;
        }
        total_sub_calls.fetch_add(1, Ordering::Relaxed);

        // Parse selected terms from response
        let selection_content = &selection_response.content;
        let selected_terms: Vec<String> = selection_content
            .lines()
            .skip_while(|line| !line.contains("SELECTED_TERMS"))
            .skip(1) // Skip the SELECTED_TERMS: line itself
            .filter(|line| {
                !line.trim().is_empty() && !line.starts_with('#') && !line.starts_with('-')
            })
            .map(|line| line.trim().to_string())
            .filter(|term| term.len() >= 3) // Skip very short terms
            .take(100) // Max 100 terms
            .collect();

        let phase3_duration = phase3_start.elapsed().as_millis() as u64;

        eprintln!("   ✓ Selected {} terms for filtering", selected_terms.len());

        emit(ProgressEvent::PhaseComplete {
            phase: 3,
            name: "Selection".to_string(),
            duration_ms: phase3_duration,
            result_preview: format!(
                "{} terms: {}",
                selected_terms.len(),
                selected_terms
                    .iter()
                    .take(10)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        });

        info!(
            duration_ms = phase3_duration,
            term_count = selected_terms.len(),
            "Phase 3 (Selection) complete"
        );

        // ========================================================================
        // PHASE 4: RETRIEVAL - Extract lines containing selected terms
        // ========================================================================
        emit(ProgressEvent::PhaseStart {
            phase: 4,
            name: "Retrieval".to_string(),
            description: format!(
                "Extracting lines with {} selected terms",
                selected_terms.len()
            ),
        });
        let phase4_start = Instant::now();

        eprintln!("   ⏳ Extracting lines containing selected terms...");

        // Build regex pattern from selected terms
        let filtered_data = if selected_terms.is_empty() {
            eprintln!("   ⚠ No terms selected, using fallback filter");
            self.fallback_filter(context, query)
        } else {
            // Escape regex special chars and join with |
            let pattern_terms: Vec<String> =
                selected_terms.iter().map(|t| regex::escape(t)).collect();
            let combined_pattern = pattern_terms.join("|");

            match regex::Regex::new(&combined_pattern) {
                Ok(re) => {
                    let matching_lines: Vec<&str> =
                        context.lines().filter(|line| re.is_match(line)).collect();

                    eprintln!("   ✓ Found {} matching lines", matching_lines.len());
                    matching_lines.join("\n")
                }
                Err(e) => {
                    eprintln!("   ⚠ Regex error: {}, using fallback", e);
                    self.fallback_filter(context, query)
                }
            }
        };

        let phase4_duration = phase4_start.elapsed().as_millis() as u64;
        let reduction_ratio = if context_chars > 0 {
            ((context_chars - filtered_data.len()) as f64 / context_chars as f64 * 100.0) as u32
        } else {
            0
        };

        eprintln!(
            "   ✓ Reduced {} -> {} chars ({}% reduction)",
            context_chars,
            filtered_data.len(),
            reduction_ratio
        );

        emit(ProgressEvent::PhaseComplete {
            phase: 4,
            name: "Retrieval".to_string(),
            duration_ms: phase4_duration,
            result_preview: format!(
                "{} -> {} chars ({}% reduction)",
                context_chars,
                filtered_data.len(),
                reduction_ratio
            ),
        });

        info!(
            original_chars = context_chars,
            filtered_chars = filtered_data.len(),
            reduction_percent = reduction_ratio,
            duration_ms = phase4_duration,
            "Phase 4 (Retrieval) complete"
        );

        // Secondary reduction using paragraph-based chunking
        // This preserves semantic context better than line-by-line scoring
        let target_size = self.config.target_analysis_size;
        let filtered_data = if filtered_data.len() > target_size * 2 {
            eprintln!(
                "   📉 Paragraph-based reduction: {} -> target ~{}K",
                filtered_data.len(),
                target_size / 1000
            );

            // Group lines into logical chunks
            // Since retrieval gives us single lines, group consecutive lines
            // that share character names into "pseudo-paragraphs"
            let lines: Vec<&str> = filtered_data.lines().collect();
            eprintln!("   Processing {} lines into chunks", lines.len());

            // Just use individual lines as units for scoring
            // This is more appropriate for line-by-line retrieved data
            let paragraphs: Vec<&str> = lines.to_vec();

            // Relationship terms that indicate explicit connections
            let strong_terms = [
                "father of",
                "mother of",
                "son of",
                "daughter of",
                "brother of",
                "sister of",
                "wife of",
                "husband of",
                "married to",
                "married",
                "child of",
                "parent of",
                "his father",
                "her father",
                "his mother",
                "her mother",
                "his son",
                "her son",
                "his daughter",
                "her daughter",
                "his wife",
                "her husband",
                "his brother",
                "her sister",
            ];
            let weak_terms = [
                "father", "mother", "son", "daughter", "brother", "sister", "wife", "husband",
                "family", "child", "parent", "Prince", "Princess", "Count", "Countess",
            ];

            // Score each line by relationship term density
            let mut scored_lines: Vec<(usize, &str)> = paragraphs
                .iter()
                .map(|para| {
                    let lower = para.to_lowercase();
                    // Strong terms worth 3 points, weak terms worth 1 point
                    let strong_score: usize = strong_terms
                        .iter()
                        .filter(|term| lower.contains(*term))
                        .count()
                        * 3;
                    let weak_score: usize = weak_terms
                        .iter()
                        .filter(|term| lower.contains(*term))
                        .count();
                    (strong_score + weak_score, *para)
                })
                .filter(|(score, _)| *score > 0)
                .collect();

            // Sort by score (descending)
            scored_lines.sort_by(|a, b| b.0.cmp(&a.0));

            // Take top lines until we hit target
            let mut result = String::new();
            let mut line_count = 0;
            for (score, line) in scored_lines {
                if result.len() + line.len() > target_size && !result.is_empty() {
                    break;
                }
                if score > 1 {
                    // Only lines with meaningful relationship content
                    result.push_str(line.trim());
                    result.push('\n');
                    line_count += 1;
                }
            }

            eprintln!(
                "   ✓ Kept {} high-value lines: {} -> {} chars",
                line_count,
                filtered_data.len(),
                result.len()
            );
            result
        } else {
            filtered_data
        };

        // ========================================================================
        // PHASE 5: ANALYSIS - Process reduced data with normal RLM
        // ========================================================================
        emit(ProgressEvent::PhaseStart {
            phase: 5,
            name: "Analysis".to_string(),
            description: format!("Processing {} chars with RLM", filtered_data.len()),
        });
        let phase5_start = Instant::now();

        // Now run normal RLM on the filtered data
        // Use process_with_options with force_rlm=true to skip phased check
        eprintln!(
            "   ⏳ Starting RLM analysis on {} chars...",
            filtered_data.len()
        );
        let analysis_result = Box::pin(self.process_with_options(
            query,
            &filtered_data,
            progress.clone(),
            true, // force_rlm - don't re-trigger phased processing
        ))
        .await?;
        eprintln!(
            "   ✓ RLM analysis complete ({} iterations)",
            analysis_result.iterations
        );

        let phase5_duration = phase5_start.elapsed().as_millis() as u64;

        emit(ProgressEvent::PhaseComplete {
            phase: 5,
            name: "Analysis".to_string(),
            duration_ms: phase5_duration,
            result_preview: if analysis_result.answer.len() > 200 {
                format!(
                    "{}...",
                    truncate_to_char_boundary(&analysis_result.answer, 200)
                )
            } else {
                analysis_result.answer.clone()
            },
        });

        // Combine results
        let total_duration = query_start.elapsed().as_millis() as u64;

        emit(ProgressEvent::Complete {
            iterations: analysis_result.iterations,
            success: analysis_result.success,
            total_duration_ms: total_duration,
        });

        Ok(RlmResult {
            answer: analysis_result.answer,
            iterations: analysis_result.iterations,
            history: analysis_result.history,
            total_sub_calls: total_sub_calls.load(Ordering::Relaxed)
                + analysis_result.total_sub_calls,
            success: analysis_result.success,
            error: analysis_result.error,
            total_prompt_tokens: total_prompt_tokens + analysis_result.total_prompt_tokens,
            total_completion_tokens: total_completion_tokens
                + analysis_result.total_completion_tokens,
            context_chars, // Original context size
            bypassed: false,
        })
    }

    /// Fallback filter for when CLI extraction fails
    /// Extracts lines containing common relationship/entity patterns
    fn fallback_filter(&self, context: &str, query: &str) -> String {
        let query_lower = query.to_lowercase();
        let lines: Vec<&str> = context.lines().collect();

        // Common patterns to look for based on query keywords
        let patterns: Vec<&str> = if query_lower.contains("family")
            || query_lower.contains("tree")
            || query_lower.contains("relationship")
        {
            vec![
                "father", "mother", "son", "daughter", "brother", "sister", "wife", "husband",
                "married", "family", "child", "parent", "Prince", "Princess", "Count", "Countess",
                "Duke", "Duchess",
            ]
        } else if query_lower.contains("character") || query_lower.contains("name") {
            vec![
                "Prince", "Princess", "Count", "Countess", "Duke", "Duchess", "General", "Colonel",
                "Captain", "Monsieur", "Madame",
            ]
        } else {
            // Generic: keep lines with capitalized words (potential names/entities)
            vec![]
        };

        let filtered: Vec<&str> = if patterns.is_empty() {
            // Keep lines with multiple capitalized words
            lines
                .iter()
                .filter(|line| {
                    let caps: usize = line
                        .split_whitespace()
                        .filter(|w| {
                            w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                                && w.len() > 2
                        })
                        .count();
                    caps >= 2
                })
                .take(5000) // Limit to 5000 lines
                .cloned()
                .collect()
        } else {
            lines
                .iter()
                .filter(|line| {
                    let lower = line.to_lowercase();
                    patterns.iter().any(|p| lower.contains(&p.to_lowercase()))
                })
                .take(5000)
                .cloned()
                .collect()
        };

        filtered.join("\n")
    }

    fn build_system_prompt(&self, context_len: usize) -> String {
        // Check if we're in coordinator mode (base LLM only delegates)
        let coordinator_mode =
            self.config.llm_delegation.enabled && self.config.llm_delegation.coordinator_mode;

        // Build level status strings
        let dsl_status = if self.config.dsl.enabled && !coordinator_mode {
            "ENABLED"
        } else {
            "DISABLED"
        };
        let wasm_status = if self.config.wasm.enabled && !coordinator_mode {
            "ENABLED"
        } else {
            "DISABLED"
        };
        let cli_status = if self.config.cli.enabled && !coordinator_mode {
            "ENABLED"
        } else {
            "DISABLED"
        };
        let llm_status = if self.config.llm_delegation.enabled {
            "ENABLED"
        } else {
            "DISABLED"
        };

        // Build available commands section based on enabled levels
        let mut commands_section = String::new();

        // In coordinator mode, skip L1/L2/L3 - base LLM only delegates
        // Level 1: DSL commands (always shown if enabled and not in coordinator mode)
        if self.config.dsl.enabled && !coordinator_mode {
            commands_section.push_str(
                r#"
### Level 1: DSL Operations (text extraction and search) [ENABLED]
- {"op": "slice", "start": 0, "end": 1000} - Get characters [start:end]
- {"op": "lines", "start": 0, "end": 100} - Get lines [start:end]
- {"op": "len"} - Get context length
- {"op": "count", "what": "lines"} - Count lines/chars/words
- {"op": "regex", "pattern": "class \\w+"} - Find regex matches
- {"op": "find", "text": "error"} - Find text occurrences
- {"op": "set", "name": "x", "value": "..."} - Set variable
- {"op": "get", "name": "x"} - Get variable value

DSL is for: Finding/extracting specific text, counting occurrences of ONE pattern
DSL CANNOT: Count frequencies of MULTIPLE items, compute statistics, sort results
"#,
            );
        }

        // Level 2: WASM commands (not in coordinator mode)
        if self.config.wasm.enabled && !coordinator_mode {
            commands_section.push_str(r#"
### Level 2: WASM Computation (sandboxed code execution) [ENABLED]

TWO WASM OPERATIONS - choose the right one:

1. rust_wasm_mapreduce - SIMPLE per-line extraction + ONE aggregation:
   {"op": "rust_wasm_mapreduce", "intent": "EXTRACT something from each line", "combiner": "count|unique|sum", "store": "result"}
   Use ONLY for: simple frequency OR simple unique count OR simple sum (ONE thing)

2. rust_wasm_intent - COMPLEX queries, multiple requirements, sorting, ranking:
   {"op": "rust_wasm_intent", "intent": "FULL ANALYSIS DESCRIPTION", "store": "result"}
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

        // Level 3: CLI commands (not in coordinator mode)
        if self.config.cli.enabled && !coordinator_mode {
            commands_section.push_str(
                r#"
### Level 3: CLI Computation (native binary, PREFERRED for analysis) [ENABLED]
- {"op": "rust_cli_intent", "intent": "what to compute", "store": "result"}
  Full Rust stdlib (HashMap, HashSet, contains, split, sort). Fast and reliable.

USE CLI FOR:
- Large datasets (1000+ lines) - WASM may timeout
- Complex aggregations (frequency counting, top-N ranking)
- Operations needing HashMap/HashSet
- Sorting, percentiles, statistics
- When WASM fails or produces errors
- **CRITICAL: Data reduction for very large contexts** (see below)

WASM LIMITATIONS (why CLI is better for complex tasks):
- No HashMap/HashSet (use custom byte-level helpers)
- 64MB memory limit
- Fuel-based instruction limits
- String methods can panic (TwoWaySearcher issue)

## SMALL CONTEXTS (<50,000 chars): USE DSL + LLM COMMANDS

For small contexts, prefer SIMPLE commands:
1. Use regex/find to extract patterns
2. Use llm_reduce for semantic extraction (splits into manageable chunks)
3. Use llm_query for synthesis/reasoning
4. AVOID rust_cli_intent - it's slow and can fail to compile

Example for small context family tree extraction:
{"op": "llm_reduce", "directive": "Extract all family relationships: who is related to whom (father, mother, son, daughter, spouse, sibling)", "store": "relationships"}
{"op": "llm_query", "prompt": "Build family trees from these relationships: ${relationships}", "store": "trees"}
{"op": "final_var", "name": "trees"}

## VERY LARGE CONTEXTS (>500,000 chars): USE CLI TO REDUCE FIRST

If the context is larger than ~500KB, you MUST use rust_cli_intent to FILTER/EXTRACT
relevant data BEFORE using llm_reduce. Otherwise llm_reduce will create too many chunks
and fail.

STRATEGY FOR MASSIVE FILES (millions of characters):
1. Identify what data is relevant to the query
2. Use rust_cli_intent to extract ONLY relevant lines/sentences
   Example: {"op": "rust_cli_intent", "intent": "Extract lines containing family relationship words (father, mother, son, daughter, brother, sister, married, wife, husband) near capitalized names", "store": "filtered"}
3. THEN use llm_reduce on the filtered data (now much smaller)
4. Use llm_query to synthesize final answer

Example for family tree from a novel:
{"op": "rust_cli_intent", "intent": "Extract sentences containing relationship words (father, mother, son, daughter, married, wife, husband, brother, sister) along with any capitalized proper names nearby. Also count name frequencies.", "store": "relationships"}
{"op": "llm_reduce", "directive": "From these relationship sentences, extract family connections: who is related to whom and how", "on": "relationships", "store": "family_data"}
{"op": "llm_query", "prompt": "Build family trees from: ${family_data}", "store": "trees"}
{"op": "final_var", "name": "trees"}
"#,
            );
        }

        // Level 4: LLM Delegation
        if self.config.llm_delegation.enabled {
            if coordinator_mode {
                // Coordinator mode: Base LLM only delegates, doesn't process data directly
                commands_section.push_str(
                    r#"
### COORDINATOR MODE: Your Role

You are a COORDINATOR. You do NOT process data directly. You delegate ALL tasks to sub-LLMs.

YOUR TOOLS:

1. **llm_reduce** - Extract information from chunks (PREFERRED for large contexts):
   {"op": "llm_reduce", "directive": "Extract names, claims, and evidence from each section", "store": "result"}

   This splits the document into ~10K-char chunks and processes each sequentially.
   Each worker extracts information and accumulates findings.
   Use this first to systematically extract information from the entire document.

2. **llm_query** - Analyze extracted data (PREFERRED for reasoning):
   {"op": "llm_query", "prompt": "Based on these findings: ${case_data}\n\nQuestion: Who is the murderer?", "store": "answer"}

   Use AFTER llm_reduce to reason about the extracted findings.
   Include the extracted variable in your prompt with ${variable_name} syntax.
   This is a simple LLM call - no commands, just reasoning.

3. **final** / **final_var** - Return your answer:
   {"op": "final", "answer": "Based on the analysis: Colonel Pemberton is the murderer because..."}
   {"op": "final_var", "name": "answer"}

## WORKFLOW FOR LARGE DOCUMENTS

Step 1 - EXTRACT (use llm_reduce):
{"op": "llm_reduce", "directive": "Extract: witness names and statements, physical evidence, alibis, timelines, motives", "store": "findings"}

Step 2 - ANALYZE (use llm_query with extracted data):
{"op": "llm_query", "prompt": "Analyze these findings from a murder investigation:\n\n${findings}\n\nCross-reference witness statements with physical evidence. Find contradictions. Who had motive, means, and opportunity? Name the murderer with supporting evidence.", "store": "conclusion"}

Step 3 - RETURN:
{"op": "final_var", "name": "conclusion"}

## EXAMPLE: Murder Mystery

```json
{"op": "llm_reduce", "directive": "Extract: (1) witness names and their statements, (2) physical evidence, (3) alibis and timelines, (4) motives mentioned", "store": "case_data"}
```

```json
{"op": "llm_query", "prompt": "Based on this murder case data:\n\n${case_data}\n\nAnalyze the evidence. Cross-reference witness statements. Identify contradictions. Who had motive, means, and opportunity? Name the murderer.", "store": "answer"}
```

```json
{"op": "final_var", "name": "answer"}
```

## CRITICAL RULES

- Use llm_reduce FIRST to extract data from the document
- Use llm_query to ANALYZE extracted data (include ${variable} in prompt)
- NEVER try to read or extract data yourself - ALWAYS use llm_reduce
"#,
                );
            } else {
                // Normal mode: Base LLM has access to all tools including delegation
                commands_section.push_str(
                    r#"
### Level 4: LLM Delegation (semantic reasoning) [ENABLED]

USE llm_delegate WHEN THE QUERY REQUIRES:
- Understanding meaning (who, why, what happened)
- Cross-referencing information
- Identifying contradictions or relationships
- Reasoning about content (not just extracting or counting)

RECOGNIZING SEMANTIC TASKS:
- "Who did X?" -> llm_delegate (requires reasoning)
- "Cross-reference A with B" -> llm_delegate (requires understanding)
- "Find contradictions" -> llm_delegate (requires semantic comparison)
- "Identify the killer/culprit" -> llm_delegate (requires multi-hop reasoning)

COMMANDS:
1. llm_delegate - Nested RLM with reasoning capabilities:
   {"op": "llm_delegate", "task": "Analyze this evidence and identify contradictions", "on": "extracted_data", "store": "analysis"}

2. llm_query - Quick semantic check (no tools):
   {"op": "llm_query", "prompt": "Does this text mention Colonel Pemberton?", "store": "check"}

WORKFLOW FOR SEMANTIC REASONING:
1. Extract the relevant section(s) with DSL: {"op": "regex", "pattern": "CASE SUMMARY.*?END", "store": "summary"}
2. Delegate analysis: {"op": "llm_delegate", "task": "Based on this summary, who is the murderer and why?", "on": "summary", "store": "result"}
3. Return: {"op": "final_var", "name": "result"}

IMPORTANT: If the query asks WHO/WHY/WHAT (semantic), use llm_delegate.
If the query asks HOW MANY/RANK/SORT (computation), use rust_cli_intent.
"#,
                );
            }
        }

        // Build workflow section based on available levels
        let workflow = if coordinator_mode {
            // In coordinator mode, workflow is already explained in the L4 section
            ""
        } else if self.config.llm_delegation.enabled && self.config.cli.enabled {
            r#"
## CRITICAL: Match Tool to Task Type

SEMANTIC REASONING (use llm_delegate):
- Questions starting with WHO, WHY, WHAT HAPPENED
- Tasks requiring cross-referencing, contradiction detection
- Analysis needing understanding of meaning
- Example: "Who murdered X?" -> extract evidence, then llm_delegate

COMPUTATION (use rust_cli_intent):
- Questions starting with HOW MANY, RANK, TOP N
- Tasks requiring counting, sorting, aggregating
- Analysis of patterns/frequencies
- Example: "Count unique IPs" -> rust_cli_intent

## Standard Workflow
1. Extract relevant data with DSL (regex, find, lines)
2. For semantic analysis: llm_delegate on extracted data
3. For computation: rust_cli_intent on extracted data
4. Return result with final or final_var

## Example: Frequency Analysis (use rust_cli_intent)

```json
{"op": "rust_cli_intent", "intent": "Count each error type and rank by frequency", "store": "counts"}
{"op": "final_var", "name": "counts"}
```

## Example: Top-N with Percentiles

```json
{"op": "rust_cli_intent", "intent": "Find top 10 most active IPs and calculate p50, p95, p99 response times", "store": "analysis"}
{"op": "final_var", "name": "analysis"}
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
{"op": "rust_wasm_mapreduce", "intent": "Extract the word after 'from ' from each line", "combiner": "unique", "store": "result"}
{"op": "final_var", "name": "result"}
```

Combined query - unique count AND top 10 (use rust_wasm_intent for MULTIPLE requirements):
```json
{"op": "rust_wasm_intent", "intent": "Extract IP after 'from ', count unique IPs, also count frequency of each IP and return top 10 most active", "store": "result"}
{"op": "final_var", "name": "result"}
```

Percentiles (needs sorting - use rust_wasm_intent):
```json
{"op": "rust_wasm_intent", "intent": "Extract the number before 'ms' from each line, sort all numbers, compute p50 p95 p99 percentiles", "store": "result"}
{"op": "final_var", "name": "result"}
```"#
        } else {
            r#"
## Workflow
1. Use find/regex/lines to extract relevant content
2. Use count for simple counting
3. Return result with final

## Example: Finding Content

```json
{"op": "find", "text": "error", "store": "matches"}
{"op": "final_var", "name": "matches"}
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
            let last_500: String = context
                .chars()
                .rev()
                .take(500)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            format!(
                "[First 500 chars]\n{}\n\n[Last 500 chars]\n{}",
                first_500, last_500
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
            // Find valid UTF-8 character boundaries to avoid panics with multi-byte chars
            let start_end = output
                .char_indices()
                .take_while(|(i, _)| *i < half)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0);
            let end_start = output
                .char_indices()
                .find(|(i, _)| *i >= output.len().saturating_sub(half))
                .map(|(i, _)| i)
                .unwrap_or(output.len());
            format!(
                "{}... [truncated {} chars] ...{}",
                &output[..start_end],
                output.len() - self.config.output_limit,
                &output[end_start..]
            )
        } else {
            output.to_string()
        }
    }
}
