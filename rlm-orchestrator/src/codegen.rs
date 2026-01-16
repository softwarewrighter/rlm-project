//! Two-LLM Code Generation
//!
//! This module provides Rust code generation using a specialized coding LLM
//! based on intent descriptions from the base LLM.
//!
//! The workflow:
//! 1. Base LLM (DeepSeek-chat) describes WHAT it wants to compute in plain English
//! 2. This module calls the coding LLM (DeepSeek-coder or qwen2.5-coder)
//! 3. The coding LLM generates safe Rust code using our helper functions

use crate::provider::{LiteLLMProvider, LlmProvider, LlmRequest, OllamaProvider, ProviderError};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

#[derive(Error, Debug)]
pub enum CodeGenError {
    #[error("Code generation LLM error: {0}")]
    LlmError(#[from] ProviderError),

    #[error("Invalid code generated: {0}")]
    InvalidCode(String),

    #[error("Code generation not configured")]
    NotConfigured,
}

/// Provider type for code generation
#[derive(Debug, Clone, PartialEq)]
pub enum CodeGenProviderType {
    Ollama,
    LiteLLM,
}

/// Configuration for the code generation LLM
#[derive(Debug, Clone)]
pub struct CodeGenConfig {
    /// Provider type (Ollama or LiteLLM)
    pub provider_type: CodeGenProviderType,
    /// Server URL (Ollama or LiteLLM gateway)
    pub url: String,
    /// Model to use (e.g., "deepseek/deepseek-coder" or "qwen2.5-coder:14b")
    pub model: String,
    /// API key (for LiteLLM)
    pub api_key: Option<String>,
    /// Temperature for code generation (lower = more deterministic)
    pub temperature: f32,
}

impl Default for CodeGenConfig {
    fn default() -> Self {
        Self {
            provider_type: CodeGenProviderType::Ollama,
            url: "http://localhost:11434".to_string(),
            model: "qwen2.5-coder:14b".to_string(),
            api_key: None,
            temperature: 0.1,
        }
    }
}

/// Wrapper for different provider types
enum ProviderWrapper {
    Ollama(OllamaProvider),
    LiteLLM(LiteLLMProvider),
}

/// Code generator using a specialized coding LLM
pub struct CodeGenerator {
    provider: Arc<RwLock<Option<ProviderWrapper>>>,
    config: CodeGenConfig,
}

impl CodeGenerator {
    /// Create a new code generator with Ollama
    pub fn new(config: CodeGenConfig) -> Self {
        let provider = match config.provider_type {
            CodeGenProviderType::Ollama => {
                ProviderWrapper::Ollama(OllamaProvider::new(&config.url, &config.model))
            }
            CodeGenProviderType::LiteLLM => {
                let api_key = config.api_key.clone().unwrap_or_default();
                ProviderWrapper::LiteLLM(LiteLLMProvider::with_base_url(&config.url, &api_key, &config.model))
            }
        };
        Self {
            provider: Arc::new(RwLock::new(Some(provider))),
            config,
        }
    }

    /// Create an unconfigured code generator (will return errors)
    pub fn unconfigured() -> Self {
        Self {
            provider: Arc::new(RwLock::new(None)),
            config: CodeGenConfig::default(),
        }
    }

    /// Generate Rust code from an intent description
    pub async fn generate(&self, intent: &str) -> Result<String, CodeGenError> {
        let provider_guard = self.provider.read().await;
        let provider = provider_guard
            .as_ref()
            .ok_or(CodeGenError::NotConfigured)?;

        let system_prompt = Self::build_system_prompt();
        let request = LlmRequest::new(system_prompt, intent)
            .with_temperature(self.config.temperature)
            .with_max_tokens(2048);

        let response = match provider {
            ProviderWrapper::Ollama(p) => p.complete(&request).await?,
            ProviderWrapper::LiteLLM(p) => p.complete(&request).await?,
        };

        // Extract just the Rust code from the response
        let code = Self::extract_code(&response.content)?;
        Ok(code)
    }

    /// Generate Rust code for streaming reduce pattern
    ///
    /// This generates:
    /// - A State struct to hold accumulated data
    /// - init_state() -> State to initialize
    /// - process_line(state: &mut State, line: &str) to process each line
    /// - finalize(state: &State) -> String to produce final result
    pub async fn generate_reduce(&self, intent: &str) -> Result<String, CodeGenError> {
        let provider_guard = self.provider.read().await;
        let provider = provider_guard
            .as_ref()
            .ok_or(CodeGenError::NotConfigured)?;

        let system_prompt = Self::build_reduce_system_prompt();
        let request = LlmRequest::new(system_prompt, intent)
            .with_temperature(self.config.temperature)
            .with_max_tokens(2048);

        let response = match provider {
            ProviderWrapper::Ollama(p) => p.complete(&request).await?,
            ProviderWrapper::LiteLLM(p) => p.complete(&request).await?,
        };

        // Extract the reduce code from the response
        let code = Self::extract_reduce_code(&response.content)?;
        Ok(code)
    }

    /// Build the system prompt for the coding LLM
    fn build_system_prompt() -> &'static str {
        r#"You are a Rust code generator. Your code runs in a WASM sandbox with these constraints:

ENVIRONMENT CONSTRAINTS:
- Code compiles to WebAssembly (WASM) with limited memory
- Input is ASCII-only text (no unicode handling needed)
- Rust's pattern-matching string methods (contains, find, split) use an algorithm
  called TwoWaySearcher that PANICS in WASM due to memory constraints
- HashMap/HashSet also panic in WASM - use Vec instead

Because of these constraints, you MUST use the provided byte-level helper functions
instead of Rust's standard string methods. The helpers work by simple byte comparison.

OUTPUT RULES:
1. Output ONLY Rust code - no explanations, no markdown
2. Start directly with: pub fn analyze(input: &str) -> String {

SAFE HELPERS (already defined - use these instead of stdlib):
- has(s, "pat") -> bool        Check if string contains pattern
- count(s, "pat") -> usize     Count occurrences of pattern in string
- after(s, "pat") -> &str      Get text after first occurrence of pattern
- before(s, "pat") -> &str     Get text before first occurrence of pattern
- word(s, n) -> &str           Get nth whitespace-separated word (0-indexed)
- slice(s, start, end) -> &str Get substring by byte position
- parse_int(s) -> i64          Parse integer (returns 0 on failure)
- eq(a, b) -> bool             Compare strings for equality

SAFE STDLIB:
- Vec, Vec::new(), .push(), .len(), indexing
- .lines() iterator
- .trim(), .to_string(), .is_empty(), .len()
- format!(), .push_str(), .join()
- .iter(), .map(), .filter(), .collect()
- .sort_by() for sorting

FORBIDDEN (code will be REJECTED - these panic in WASM):
- .contains() -> use has() instead
- .find() / .rfind() -> use after()/before() instead
- .split() / .split_once() -> use word() or iterate with after()
- .matches() / .match_indices() -> use has() in a loop
- .replace() / .replacen() -> build new string with format!()
- .strip_prefix() / .strip_suffix() -> use after()/before()
- == for string comparison -> use eq() instead
- HashMap / HashSet / BTreeMap -> use Vec<(String, usize)>
- .unwrap() -> use .unwrap_or() or .unwrap_or_default()

COUNTING PATTERN (use this for frequency counting):
```rust
let mut counts: Vec<(String, usize)> = Vec::new();
// For each item to count:
let key = some_string.to_string();
let mut found = false;
for i in 0..counts.len() {
    if eq(&counts[i].0, &key) {
        counts[i].1 += 1;
        found = true;
        break;
    }
}
if !found {
    counts.push((key, 1));
}
```

EXAMPLE - Count error types:
pub fn analyze(input: &str) -> String {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for line in input.lines() {
        if has(line, "[ERROR]") {
            let err_type = word(after(line, "[ERROR] "), 0).to_string();
            if !err_type.is_empty() {
                let mut found = false;
                for i in 0..counts.len() {
                    if eq(&counts[i].0, &err_type) {
                        counts[i].1 += 1;
                        found = true;
                        break;
                    }
                }
                if !found {
                    counts.push((err_type, 1));
                }
            }
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    counts.iter()
        .map(|(k, c)| format!("{}: {}", k, c))
        .collect::<Vec<_>>()
        .join("\n")
}

Intent: "Sum all numbers found after 'value:' in each line"
Output:
pub fn analyze(input: &str) -> String {
    let mut sum: i64 = 0;
    for line in input.lines() {
        if has(line, "value:") {
            let val = parse_int(word(after(line, "value:"), 0).trim());
            sum += val;
        }
    }
    sum.to_string()
}

Intent: "Count lines containing WARNING"
Output:
pub fn analyze(input: &str) -> String {
    let count = input.lines().filter(|l| has(l, "WARNING")).count();
    count.to_string()
}

Intent: "Count occurrences of 'Smith', 'Brown', 'Wilson' in the text"
Output:
pub fn analyze(input: &str) -> String {
    let smith = count(input, "Smith");
    let brown = count(input, "Brown");
    let wilson = count(input, "Wilson");
    format!("Smith: {}, Brown: {}, Wilson: {}", smith, brown, wilson)
}

Now generate code for the following intent. Output ONLY the Rust function, starting with `pub fn analyze`:"#
    }

    /// Build the system prompt for streaming reduce code generation
    fn build_reduce_system_prompt() -> &'static str {
        r#"You are a Rust code generator for STREAMING REDUCE operations. Your code processes data LINE BY LINE to handle arbitrarily large datasets without running out of memory.

STREAMING REDUCE PATTERN:
Instead of processing all data at once, you generate code that:
1. Defines a State struct to hold accumulated values
2. Initializes the state with init_state()
3. Processes each line with process_line() - accumulating into state
4. Produces final result with finalize()

OUTPUT RULES:
1. Output ONLY Rust code - no explanations, no markdown
2. Must define: struct State { ... }
3. Must define: fn init_state() -> State
4. Must define: fn process_line(state: &mut State, line: &str)
5. Must define: fn finalize(state: &State) -> String

ENVIRONMENT CONSTRAINTS (same as non-streaming):
- Code compiles to WebAssembly (WASM) with limited memory
- Input is ASCII-only text (no unicode handling needed)
- TwoWaySearcher PANICS in WASM - use helpers instead
- HashMap/HashSet panic in WASM - use Vec<(String, T)> instead

SAFE HELPERS (already defined - use these):
- has(s, "pat") -> bool        Check if string contains pattern
- count(s, "pat") -> usize     Count occurrences of pattern
- after(s, "pat") -> &str      Get text after first occurrence
- before(s, "pat") -> &str     Get text before first occurrence
- word(s, n) -> &str           Get nth whitespace-separated word (0-indexed)
- slice(s, start, end) -> &str Get substring by byte position
- parse_int(s) -> i64          Parse integer (returns 0 on failure)
- eq(a, b) -> bool             Compare strings for equality

SAFE STDLIB:
- Vec, Vec::new(), .push(), .len(), indexing
- .trim(), .to_string(), .is_empty(), .len()
- format!(), .push_str(), .join()
- .iter(), .map(), .filter(), .collect()
- .sort_by() for sorting

FORBIDDEN (these panic in WASM):
- .contains(), .find(), .rfind() -> use has(), after(), before()
- .split(), .split_once(), .rsplit() -> use word() or iterate with after()
- .matches(), .match_indices() -> use has() in a loop
- .replace(), .replacen() -> build new string with format!()
- .strip_prefix(), .strip_suffix() -> use after(), before()
- .starts_with(), .ends_with() -> use has() helper
- str == str, str != str -> use eq(a, b) for ALL string comparisons!
- HashMap / HashSet / BTreeMap -> use Vec<(String, usize)>
- .unwrap() -> use .unwrap_or() or .unwrap_or_default()

CRITICAL: For string comparison, ALWAYS use eq(a, b), NEVER use == or !=

REDUCIBLE OPERATIONS (can be computed with streaming):
- Count, Sum, Min, Max
- Mean (track sum and count, divide in finalize)
- Frequency counting (track Vec<(key, count)>)
- Finding unique values
- Top-N by frequency

NOT REDUCIBLE (require all data at once):
- Median, Percentiles (need sorted data)
- Mode (need full frequency then find max)

EXAMPLE - Count unique IPs and rank top 10:

struct State {
    ip_counts: Vec<(String, usize)>,
}

fn init_state() -> State {
    State {
        ip_counts: Vec::new(),
    }
}

fn process_line(state: &mut State, line: &str) {
    // Extract IP address (first word)
    let ip = word(line, 0).to_string();
    if ip.is_empty() {
        return;
    }

    // Update count for this IP
    let mut found = false;
    for i in 0..state.ip_counts.len() {
        if eq(&state.ip_counts[i].0, &ip) {
            state.ip_counts[i].1 += 1;
            found = true;
            break;
        }
    }
    if !found {
        state.ip_counts.push((ip, 1));
    }
}

fn finalize(state: &State) -> String {
    // Sort by count descending
    let mut sorted = state.ip_counts.clone();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    // Take top 10 and format
    let mut result = format!("Unique IPs: {}\n\nTop 10:\n", sorted.len());
    for (i, (ip, count)) in sorted.iter().take(10).enumerate() {
        result.push_str(&format!("{}. {} ({} requests)\n", i + 1, ip, count));
    }
    result
}

EXAMPLE - Count error types:

struct State {
    error_counts: Vec<(String, usize)>,
}

fn init_state() -> State {
    State {
        error_counts: Vec::new(),
    }
}

fn process_line(state: &mut State, line: &str) {
    if !has(line, "[ERROR]") {
        return;
    }
    let err_type = word(after(line, "[ERROR] "), 0).to_string();
    if err_type.is_empty() {
        return;
    }

    let mut found = false;
    for i in 0..state.error_counts.len() {
        if eq(&state.error_counts[i].0, &err_type) {
            state.error_counts[i].1 += 1;
            found = true;
            break;
        }
    }
    if !found {
        state.error_counts.push((err_type, 1));
    }
}

fn finalize(state: &State) -> String {
    let mut sorted = state.error_counts.clone();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.iter()
        .map(|(k, c)| format!("{}: {}", k, c))
        .collect::<Vec<_>>()
        .join("\n")
}

Now generate streaming reduce code for the following intent. Output ONLY the struct State, init_state, process_line, and finalize functions:"#
    }

    /// Extract Rust code from LLM response
    fn extract_code(response: &str) -> Result<String, CodeGenError> {
        let response = response.trim();

        // If it starts with pub fn, use as-is
        if response.starts_with("pub fn analyze") {
            return Ok(response.to_string());
        }

        // Try to extract from markdown code block
        if let Some(start) = response.find("```rust") {
            let code_start = start + 7;
            if let Some(end) = response[code_start..].find("```") {
                let code = response[code_start..code_start + end].trim();
                if code.starts_with("pub fn analyze") {
                    return Ok(code.to_string());
                }
            }
        }

        // Try to extract from plain code block
        if let Some(start) = response.find("```") {
            let code_start = start + 3;
            // Skip language identifier if present
            let code_start = if let Some(newline) = response[code_start..].find('\n') {
                code_start + newline + 1
            } else {
                code_start
            };
            if let Some(end) = response[code_start..].find("```") {
                let code = response[code_start..code_start + end].trim();
                if code.starts_with("pub fn analyze") {
                    return Ok(code.to_string());
                }
            }
        }

        // Try to find pub fn analyze anywhere
        if let Some(start) = response.find("pub fn analyze") {
            // Find the end - look for matching braces
            let code = &response[start..];
            if let Some(end) = Self::find_function_end(code) {
                return Ok(code[..end].to_string());
            }
        }

        Err(CodeGenError::InvalidCode(format!(
            "Could not extract valid Rust function from response: {}",
            if response.len() > 200 {
                format!("{}...", &response[..200])
            } else {
                response.to_string()
            }
        )))
    }

    /// Find the end of a function by matching braces
    fn find_function_end(code: &str) -> Option<usize> {
        let mut depth = 0;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, c) in code.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match c {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '{' if !in_string => depth += 1,
                '}' if !in_string => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i + 1);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Extract reduce code from LLM response
    ///
    /// Looks for: struct State, fn init_state, fn process_line, fn finalize
    fn extract_reduce_code(response: &str) -> Result<String, CodeGenError> {
        let response = response.trim();

        // Try to extract from markdown code block first
        let code = if let Some(start) = response.find("```rust") {
            let code_start = start + 7;
            if let Some(end) = response[code_start..].find("```") {
                response[code_start..code_start + end].trim()
            } else {
                response
            }
        } else if let Some(start) = response.find("```") {
            let code_start = start + 3;
            let code_start = if let Some(newline) = response[code_start..].find('\n') {
                code_start + newline + 1
            } else {
                code_start
            };
            if let Some(end) = response[code_start..].find("```") {
                response[code_start..code_start + end].trim()
            } else {
                response
            }
        } else {
            response
        };

        // Validate that all required components are present
        let has_state = code.contains("struct State");
        let has_init = code.contains("fn init_state");
        let has_process = code.contains("fn process_line");
        let has_finalize = code.contains("fn finalize");

        if !has_state {
            return Err(CodeGenError::InvalidCode(
                "Missing 'struct State' definition".to_string(),
            ));
        }
        if !has_init {
            return Err(CodeGenError::InvalidCode(
                "Missing 'fn init_state()' function".to_string(),
            ));
        }
        if !has_process {
            return Err(CodeGenError::InvalidCode(
                "Missing 'fn process_line()' function".to_string(),
            ));
        }
        if !has_finalize {
            return Err(CodeGenError::InvalidCode(
                "Missing 'fn finalize()' function".to_string(),
            ));
        }

        // Find the start (struct State) and extract everything through finalize
        if let Some(struct_start) = code.find("struct State") {
            // Find the last function (finalize) and its end
            if let Some(finalize_start) = code.rfind("fn finalize") {
                let finalize_code = &code[finalize_start..];
                if let Some(end) = Self::find_function_end(finalize_code) {
                    let full_end = finalize_start + end;
                    return Ok(code[struct_start..full_end].to_string());
                }
            }
        }

        Err(CodeGenError::InvalidCode(format!(
            "Could not extract valid reduce code from response: {}",
            if code.len() > 200 {
                format!("{}...", &code[..200])
            } else {
                code.to_string()
            }
        )))
    }

    /// Check if the code generator is configured
    pub async fn is_configured(&self) -> bool {
        self.provider.read().await.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_code_direct() {
        let response = r#"pub fn analyze(input: &str) -> String {
    "test".to_string()
}"#;
        let code = CodeGenerator::extract_code(response).unwrap();
        assert!(code.starts_with("pub fn analyze"));
    }

    #[test]
    fn test_extract_code_markdown() {
        let response = r#"Here's the code:

```rust
pub fn analyze(input: &str) -> String {
    "test".to_string()
}
```

This counts things."#;
        let code = CodeGenerator::extract_code(response).unwrap();
        assert!(code.starts_with("pub fn analyze"));
    }

    #[test]
    fn test_extract_code_plain_block() {
        let response = r#"```
pub fn analyze(input: &str) -> String {
    "test".to_string()
}
```"#;
        let code = CodeGenerator::extract_code(response).unwrap();
        assert!(code.starts_with("pub fn analyze"));
    }

    #[test]
    fn test_find_function_end() {
        let code = r#"pub fn analyze(input: &str) -> String {
    if true {
        "yes"
    } else {
        "no"
    }.to_string()
}
// More stuff"#;
        let end = CodeGenerator::find_function_end(code).unwrap();
        assert!(code[..end].ends_with('}'));
        assert!(!code[..end].contains("More stuff"));
    }
}
