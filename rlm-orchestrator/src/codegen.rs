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
