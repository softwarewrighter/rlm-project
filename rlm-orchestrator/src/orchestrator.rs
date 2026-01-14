//! RLM orchestration logic

use crate::commands::{
    extract_commands, extract_final, CommandExecutor, ExecutionResult, LlmQueryCallback,
};
use crate::pool::LlmPool;
use crate::provider::{LlmRequest, ProviderError};
use crate::RlmConfig;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, info, warn};

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

        let request = LlmRequest::new(system_prompt, &prompt);
        let response = self.pool.complete(&request, false).await?;

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

    /// Process a query over the given context
    pub async fn process(
        &self,
        query: &str,
        context: &str,
    ) -> Result<RlmResult, OrchestratorError> {
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
        let mut executor = CommandExecutor::new(context.to_string(), self.config.max_sub_calls)
            .with_llm_callback(llm_query_cb);

        for iteration in 0..self.config.max_iterations {
            info!(iteration = iteration + 1, "Starting RLM iteration");

            // Check if we've exceeded max sub-calls
            if total_sub_calls.load(Ordering::Relaxed) >= self.config.max_sub_calls {
                return Err(OrchestratorError::MaxSubCalls(self.config.max_sub_calls));
            }

            // Build the prompt with history
            let prompt = self.build_prompt(query, context, &history);

            // Call the root LLM
            let request = LlmRequest::new(&system_prompt, &prompt);
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

            debug!(content_len = response.content.len(), "Got LLM response");

            // Try to extract JSON commands first
            let commands_json = extract_commands(&response.content);

            // Store LLM response for history
            let llm_response = response.content.clone();

            if let Some(json) = commands_json {
                // Execute the commands
                match executor.execute_json(&json) {
                    Ok(ExecutionResult::Final { answer, sub_calls }) => {
                        history.push(IterationRecord {
                            step: iteration + 1,
                            llm_response,
                            commands: json,
                            output: format!("FINAL: {}", &answer),
                            error: None,
                            sub_calls,
                            tokens: iter_tokens,
                        });

                        return Ok(RlmResult {
                            answer,
                            iterations: iteration + 1,
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
                        history.push(IterationRecord {
                            step: iteration + 1,
                            llm_response,
                            commands: json,
                            output: truncated_output,
                            error: None,
                            sub_calls,
                            tokens: iter_tokens,
                        });
                    }
                    Err(e) => {
                        history.push(IterationRecord {
                            step: iteration + 1,
                            llm_response,
                            commands: json,
                            output: String::new(),
                            error: Some(e.to_string()),
                            sub_calls: 0,
                            tokens: iter_tokens,
                        });
                        debug!("Command execution had error: {}", e);
                    }
                }
            } else {
                // No JSON commands - check for FINAL in plain text (fallback)
                if let Some(final_answer) = extract_final(&response.content) {
                    history.push(IterationRecord {
                        step: iteration + 1,
                        llm_response,
                        commands: String::new(),
                        output: format!("FINAL: {}", &final_answer),
                        error: None,
                        sub_calls: 0,
                        tokens: iter_tokens,
                    });
                    return Ok(RlmResult {
                        answer: final_answer,
                        iterations: iteration + 1,
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
                history.push(IterationRecord {
                    step: iteration + 1,
                    llm_response: response.content.clone(),
                    commands: String::new(),
                    output: String::new(),
                    error: Some("No JSON commands found in response".to_string()),
                    sub_calls: 0,
                    tokens: iter_tokens,
                });
            }
        }

        // Max iterations reached - still return token info
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
- {{"op": "wasm_wat", "wat": "(module ...)"}} - Compile and run WAT code

Available WASM modules: line_counter (counts lines in context)

Finishing:
- {{"op": "final", "answer": "The result is..."}}
- {{"op": "final_var", "name": "result"}}

## Variable References
Use ${{var}} or $var in strings to reference stored variables.

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
- Variables store strings only. Use `count` with `what: "matches"` after `regex` to count results.
- The `regex` command stores matched text (one per line). Use `count` with `what: "lines"` on that variable.
- Always use `store` to save results you need later.
- Wrap commands in ```json blocks. Execute multiple commands per iteration."#,
            context_len = context_len
        )
    }

    fn build_prompt(&self, query: &str, context: &str, history: &[IterationRecord]) -> String {
        // Only include first/last of context if it's large
        let context_preview = if context.len() > 2000 {
            format!(
                "[First 500 chars]\n{}\n\n[Last 500 chars]\n{}",
                &context[..500.min(context.len())],
                &context[context.len().saturating_sub(500)..]
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
