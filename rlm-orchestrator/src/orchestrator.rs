//! RLM orchestration logic

use crate::commands::{extract_commands, extract_final, CommandExecutor, ExecutionResult, LlmQueryCallback};
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

/// Record of a single RLM iteration
#[derive(Debug, Clone)]
pub struct IterationRecord {
    /// Step number (1-indexed)
    pub step: usize,
    /// Commands that were executed (JSON)
    pub commands: String,
    /// Output from execution
    pub output: String,
    /// Error if any
    pub error: Option<String>,
    /// Number of sub-LM calls made in this iteration
    pub sub_calls: usize,
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
                    let request = LlmRequest::new(
                        "You are a helpful assistant. Answer concisely.",
                        &prompt,
                    );
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

    /// Process a query over the given context
    pub async fn process(
        &self,
        query: &str,
        context: &str,
    ) -> Result<RlmResult, OrchestratorError> {
        let mut history = Vec::new();
        let total_sub_calls = Arc::new(AtomicUsize::new(0));

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

            debug!(content_len = response.content.len(), "Got LLM response");

            // Try to extract JSON commands first
            let commands_json = extract_commands(&response.content);

            if let Some(json) = commands_json {
                // Execute the commands
                match executor.execute_json(&json) {
                    Ok(ExecutionResult::Final { answer, sub_calls }) => {
                        history.push(IterationRecord {
                            step: iteration + 1,
                            commands: json,
                            output: format!("FINAL: {}", &answer),
                            error: None,
                            sub_calls,
                        });

                        return Ok(RlmResult {
                            answer,
                            iterations: iteration + 1,
                            history,
                            total_sub_calls: total_sub_calls.load(Ordering::Relaxed),
                            success: true,
                            error: None,
                        });
                    }
                    Ok(ExecutionResult::Continue { output, sub_calls }) => {
                        let truncated_output = self.truncate_output(&output);
                        history.push(IterationRecord {
                            step: iteration + 1,
                            commands: json,
                            output: truncated_output,
                            error: None,
                            sub_calls,
                        });
                    }
                    Err(e) => {
                        history.push(IterationRecord {
                            step: iteration + 1,
                            commands: json,
                            output: String::new(),
                            error: Some(e.to_string()),
                            sub_calls: 0,
                        });
                        debug!("Command execution had error: {}", e);
                    }
                }
            } else {
                // No JSON commands - check for FINAL in plain text (fallback)
                if let Some(final_answer) = extract_final(&response.content) {
                    return Ok(RlmResult {
                        answer: final_answer,
                        iterations: iteration + 1,
                        history,
                        total_sub_calls: total_sub_calls.load(Ordering::Relaxed),
                        success: true,
                        error: None,
                    });
                }

                // No commands and no FINAL - log and continue
                warn!("No commands found in response, continuing");
                history.push(IterationRecord {
                    step: iteration + 1,
                    commands: String::new(),
                    output: String::new(),
                    error: Some("No JSON commands found in response".to_string()),
                    sub_calls: 0,
                });
            }
        }

        // Max iterations reached
        Err(OrchestratorError::MaxIterations(self.config.max_iterations))
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

Sub-LM calls:
- {{"op": "llm_query", "prompt": "Summarize: ${{chunk}}", "store": "summary"}}

WASM (dynamic code execution):
- {{"op": "wasm", "module": "line_counter"}} - Run pre-compiled WASM module
- {{"op": "wasm_wat", "wat": "(module ...)"}} - Compile and run WAT code

Available WASM modules: line_counter (counts lines in context)

Finishing:
- {{"op": "final", "answer": "The result is..."}}
- {{"op": "final_var", "name": "result"}}

## Variable References
Use ${{var}} or $var in strings to reference stored variables.

## Workflow
1. Explore: Get length, first/last lines to understand structure
2. Search: Use regex/find to locate relevant content
3. Extract: Slice or get specific lines
4. Analyze: Use llm_query for semantic analysis of chunks
5. Finish: Output final answer

## Example

```json
{{"op": "len", "store": "total_len"}}
{{"op": "lines", "start": 0, "end": 10, "store": "header"}}
{{"op": "regex", "pattern": "class \\w+", "store": "classes"}}
{{"op": "count", "what": "matches", "store": "class_count"}}
{{"op": "final", "answer": "Found ${{class_count}} classes"}}
```

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
            prompt.push_str("\nContinue analysis. Use {{\"op\": \"final\", \"answer\": \"...\"}} when done.\n");
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
