//! RLM orchestration logic

use crate::pool::LlmPool;
use crate::provider::{LlmRequest, LlmResponse, ProviderError};
use crate::repl::{extract_code, extract_final, PythonRepl, ReplError};
use crate::RlmConfig;
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors from RLM orchestration
#[derive(Error, Debug)]
pub enum OrchestratorError {
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),

    #[error("REPL error: {0}")]
    Repl(#[from] ReplError),

    #[error("Max iterations ({0}) exceeded")]
    MaxIterations(usize),

    #[error("Max sub-calls ({0}) exceeded")]
    MaxSubCalls(usize),

    #[error("No code block found in response")]
    NoCodeBlock,
}

/// Record of a single RLM iteration
#[derive(Debug, Clone)]
pub struct IterationRecord {
    /// Step number (1-indexed)
    pub step: usize,
    /// Code that was executed
    pub code: String,
    /// Output from execution
    pub output: String,
    /// Error if any
    pub error: Option<String>,
    /// Number of sub-LM calls made
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

    /// Process a query over the given context
    pub async fn process(
        &self,
        query: &str,
        context: &str,
    ) -> Result<RlmResult, OrchestratorError> {
        let mut repl = PythonRepl::new();
        let mut history = Vec::new();
        let mut total_sub_calls = 0;

        let system_prompt = self.build_system_prompt(context.len());

        for iteration in 0..self.config.max_iterations {
            info!(iteration = iteration + 1, "Starting RLM iteration");

            // Build the prompt with history
            let prompt = self.build_prompt(query, context, &history);

            // Call the root LLM
            let request = LlmRequest::new(&system_prompt, &prompt);
            let response = self.pool.complete(&request, false).await?;

            debug!(content_len = response.content.len(), "Got LLM response");

            // Check for FINAL answer first (only if no code block)
            let code = extract_code(&response.content);

            if code.is_none() {
                if let Some(final_answer) = extract_final(&response.content) {
                    // Handle FINAL_VAR - look up the variable
                    let answer = if final_answer.starts_with("__FINAL_VAR__") {
                        let var_name = &final_answer[13..];
                        repl.get_variable(var_name).unwrap_or_else(|| {
                            format!("Variable '{}' not found", var_name)
                        })
                    } else {
                        final_answer
                    };

                    return Ok(RlmResult {
                        answer,
                        iterations: iteration + 1,
                        history,
                        total_sub_calls,
                        success: true,
                        error: None,
                    });
                }
            }

            // Extract and execute code
            let code = match code {
                Some(c) => c,
                None => {
                    // No code and no FINAL - this is an error
                    warn!("No code block found in response");
                    return Err(OrchestratorError::NoCodeBlock);
                }
            };

            // Execute the code
            // TODO: Inject llm_query function for sub-LM calls
            let (output, error) = match repl.execute(&code, context) {
                Ok(output) => (self.truncate_output(&output), None),
                Err(e) => (String::new(), Some(e.to_string())),
            };

            history.push(IterationRecord {
                step: iteration + 1,
                code,
                output: output.clone(),
                error: error.clone(),
                sub_calls: 0, // TODO: Track sub-calls
            });

            if error.is_some() {
                debug!("Code execution had error, continuing");
            }
        }

        // Max iterations reached
        Err(OrchestratorError::MaxIterations(self.config.max_iterations))
    }

    fn build_system_prompt(&self, context_len: usize) -> String {
        format!(
            r#"You are an RLM (Recursive Language Model) agent tasked with answering queries over large contexts.

Your context is a text with {context_len} total characters.

The REPL environment provides:
1. `context` - the full input (may be huge, use programmatic access)
2. `llm_query(prompt)` - recursive sub-LM call for semantic analysis
3. Standard Python: re, json, collections, itertools, etc.

WORKFLOW:
1. Explore: Probe the context structure (first/last chars, line count, patterns)
2. Process: Write code to filter, extract, or transform relevant data
3. Recurse: Use llm_query() for semantic analysis of chunks
4. Conclude: When ready, output FINAL(your answer) or FINAL_VAR(variable_name)

RULES:
- Always wrap code in ```python or ```repl blocks
- Use print() for debugging output
- Variables persist between iterations
- Only call FINAL when you have the complete answer

Example iteration:
```python
# Explore structure
print(f"Context length: {{len(context)}} chars")
print(f"First 200 chars: {{context[:200]}}")
```

When done:
```python
answer = "The result is..."
```
FINAL_VAR(answer)"#,
            context_len = context_len
        )
    }

    fn build_prompt(&self, query: &str, context: &str, history: &[IterationRecord]) -> String {
        let mut prompt = format!("QUERY: {}\n\nCONTEXT:\n{}\n\n", query, context);

        if !history.is_empty() {
            prompt.push_str("EXECUTION HISTORY:\n");
            for record in history {
                prompt.push_str(&format!("\n--- Step {} ---\n", record.step));
                prompt.push_str(&format!("Code:\n```python\n{}\n```\n", record.code));
                if let Some(error) = &record.error {
                    prompt.push_str(&format!("Error: {}\n", error));
                } else {
                    prompt.push_str(&format!("Output:\n{}\n", record.output));
                }
            }
            prompt.push_str("\nContinue from where you left off. Remember to use FINAL() when you have the answer.\n");
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
