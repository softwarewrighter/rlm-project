//! Pre-built WASM tools library for common analytical tasks
//!
//! This module provides a collection of pre-compiled WASM functions that the LLM
//! can invoke directly, avoiding the overhead of generating and compiling Rust code
//! for common operations.

use super::compiler::{CompilerConfig, RustCompiler};
use std::collections::HashMap;
use std::sync::Arc;

/// Catalog of pre-built WASM tools
pub struct WasmToolLibrary {
    /// Pre-compiled WASM bytes indexed by tool name
    tools: HashMap<String, Arc<Vec<u8>>>,
    /// Tool descriptions for the LLM prompt
    descriptions: HashMap<String, ToolDescription>,
}

/// Description of a pre-built tool for the LLM prompt
#[derive(Clone, Debug)]
pub struct ToolDescription {
    pub name: String,
    pub description: String,
    pub example: String,
}

impl WasmToolLibrary {
    /// Create a new library and pre-compile all tools
    pub fn new() -> Self {
        let mut lib = Self {
            tools: HashMap::new(),
            descriptions: HashMap::new(),
        };
        lib.compile_all_tools();
        lib
    }

    /// Get pre-compiled WASM bytes for a tool
    pub fn get(&self, name: &str) -> Option<Arc<Vec<u8>>> {
        self.tools.get(name).cloned()
    }

    /// List all available tools with descriptions
    pub fn list(&self) -> Vec<&ToolDescription> {
        self.descriptions.values().collect()
    }

    /// Get tool description for a specific tool
    pub fn describe(&self, name: &str) -> Option<&ToolDescription> {
        self.descriptions.get(name)
    }

    /// Generate a prompt fragment listing available tools
    pub fn prompt_fragment(&self) -> String {
        let mut s = String::from("Available WASM tools (use {\"op\": \"wasm\", \"tool\": \"<name>\"}):\n");
        for desc in self.descriptions.values() {
            s.push_str(&format!("  - {}: {}\n", desc.name, desc.description));
            s.push_str(&format!("    Example: {}\n", desc.example));
        }
        s
    }

    fn compile_all_tools(&mut self) {
        let compiler = match RustCompiler::new(CompilerConfig::default()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to initialize Rust compiler for WASM library: {}", e);
                return;
            }
        };

        // Tool 1: count_pattern - Count occurrences of a substring
        self.compile_and_register(
            &compiler,
            "count_pattern",
            r#"
pub fn analyze(input: &str) -> String {
    // Input format: "pattern|text_to_search"
    let parts: Vec<&str> = input.splitn(2, '|').collect();
    if parts.len() != 2 {
        return "Error: Expected format 'pattern|text'".to_string();
    }
    let pattern = parts[0];
    let text = parts[1];
    text.matches(pattern).count().to_string()
}
"#,
            "Count occurrences of a pattern in text",
            r#"{"op": "wasm", "tool": "count_pattern", "args": "ERROR|<context>"}"#,
        );

        // Tool 2: count_lines_matching - Count lines containing a pattern
        self.compile_and_register(
            &compiler,
            "count_lines_matching",
            r#"
pub fn analyze(input: &str) -> String {
    // Input format: "pattern|text_to_search"
    let parts: Vec<&str> = input.splitn(2, '|').collect();
    if parts.len() != 2 {
        return "Error: Expected format 'pattern|text'".to_string();
    }
    let pattern = parts[0];
    let text = parts[1];
    text.lines().filter(|line| line.contains(pattern)).count().to_string()
}
"#,
            "Count lines containing a pattern",
            r#"{"op": "wasm", "tool": "count_lines_matching", "args": "ERROR|<context>"}"#,
        );

        // Tool 3: extract_between - Extract text between delimiters
        self.compile_and_register(
            &compiler,
            "extract_between",
            r#"
pub fn analyze(input: &str) -> String {
    // Input format: "start_delim|end_delim|text"
    let parts: Vec<&str> = input.splitn(3, '|').collect();
    if parts.len() != 3 {
        return "Error: Expected format 'start|end|text'".to_string();
    }
    let start = parts[0];
    let end = parts[1];
    let text = parts[2];

    let mut results = Vec::new();
    let mut search_from = 0;

    while let Some(start_idx) = text[search_from..].find(start) {
        let actual_start = search_from + start_idx + start.len();
        if actual_start >= text.len() {
            break;
        }
        if let Some(end_idx) = text[actual_start..].find(end) {
            results.push(&text[actual_start..actual_start + end_idx]);
            search_from = actual_start + end_idx + end.len();
        } else {
            break;
        }
    }

    results.join("\n")
}
"#,
            "Extract text between start and end delimiters",
            r#"{"op": "wasm", "tool": "extract_between", "args": "[|]|<context>"}"#,
        );

        // Tool 4: word_frequency - Get word frequency counts
        self.compile_and_register(
            &compiler,
            "word_frequency",
            r#"
pub fn analyze(input: &str) -> String {
    let mut freq: HashMap<&str, usize> = HashMap::new();

    for word in input.split_whitespace() {
        // Clean up punctuation
        let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric());
        if !cleaned.is_empty() {
            *freq.entry(cleaned).or_insert(0) += 1;
        }
    }

    // Sort by frequency descending, take top 20
    let mut pairs: Vec<_> = freq.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));

    pairs.iter()
        .take(20)
        .map(|(word, count)| format!("{}: {}", word, count))
        .collect::<Vec<_>>()
        .join("\n")
}
"#,
            "Get top 20 word frequencies (sorted by count)",
            r#"{"op": "wasm", "tool": "word_frequency"}"#,
        );

        // Tool 5: categorize_lines - Group lines by matching patterns
        self.compile_and_register(
            &compiler,
            "categorize_lines",
            r#"
pub fn analyze(input: &str) -> String {
    // Input format: "pattern1,pattern2,pattern3|text"
    let parts: Vec<&str> = input.splitn(2, '|').collect();
    if parts.len() != 2 {
        return "Error: Expected format 'patterns|text'".to_string();
    }

    let patterns: Vec<&str> = parts[0].split(',').collect();
    let text = parts[1];

    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut other_count = 0usize;

    for line in text.lines() {
        let mut matched = false;
        for pattern in &patterns {
            if line.contains(*pattern) {
                *counts.entry(*pattern).or_insert(0) += 1;
                matched = true;
                break;
            }
        }
        if !matched && !line.trim().is_empty() {
            other_count += 1;
        }
    }

    let mut result: Vec<String> = patterns.iter()
        .map(|p| format!("{}: {}", p, counts.get(p).unwrap_or(&0)))
        .collect();

    result.push(format!("(other): {}", other_count));
    result.join("\n")
}
"#,
            "Categorize lines by matching patterns",
            r#"{"op": "wasm", "tool": "categorize_lines", "args": "ERROR,WARN,INFO|<context>"}"#,
        );

        // Tool 6: extract_field - Extract a specific field from structured lines
        self.compile_and_register(
            &compiler,
            "extract_field",
            r#"
pub fn analyze(input: &str) -> String {
    // Input format: "delimiter|field_index|text"
    let parts: Vec<&str> = input.splitn(3, '|').collect();
    if parts.len() != 3 {
        return "Error: Expected format 'delimiter|field_index|text'".to_string();
    }

    let delimiter = parts[0];
    let field_idx: usize = match parts[1].parse() {
        Ok(n) => n,
        Err(_) => return "Error: field_index must be a number".to_string(),
    };
    let text = parts[2];

    text.lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(delimiter).collect();
            fields.get(field_idx).map(|s| s.trim())
        })
        .collect::<Vec<_>>()
        .join("\n")
}
"#,
            "Extract a specific field from delimiter-separated lines",
            r#"{"op": "wasm", "tool": "extract_field", "args": ",|2|<context>"}"#,
        );

        // Tool 7: unique_values - Get unique values from lines
        self.compile_and_register(
            &compiler,
            "unique_values",
            r#"
pub fn analyze(input: &str) -> String {
    let unique: HashSet<&str> = input.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    let mut sorted: Vec<_> = unique.into_iter().collect();
    sorted.sort();

    format!("{} unique values:\n{}", sorted.len(), sorted.join("\n"))
}
"#,
            "Get sorted unique values from lines",
            r#"{"op": "wasm", "tool": "unique_values"}"#,
        );

        // Tool 8: sum_numbers - Sum all numbers found in text
        self.compile_and_register(
            &compiler,
            "sum_numbers",
            r#"
pub fn analyze(input: &str) -> String {
    let sum: i64 = input
        .split(|c: char| !c.is_ascii_digit() && c != '-')
        .filter_map(|s| s.parse::<i64>().ok())
        .sum();
    sum.to_string()
}
"#,
            "Sum all numbers found in the text",
            r#"{"op": "wasm", "tool": "sum_numbers"}"#,
        );

        // Tool 9: statistics - Calculate min, max, avg of numbers
        self.compile_and_register(
            &compiler,
            "statistics",
            r#"
pub fn analyze(input: &str) -> String {
    let numbers: Vec<f64> = input
        .split(|c: char| !c.is_ascii_digit() && c != '-' && c != '.')
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();

    if numbers.is_empty() {
        return "No numbers found".to_string();
    }

    let count = numbers.len();
    let sum: f64 = numbers.iter().sum();
    let min = numbers.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = numbers.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg = sum / count as f64;

    format!(
        "count: {}\nmin: {:.2}\nmax: {:.2}\navg: {:.2}\nsum: {:.2}",
        count, min, max, avg, sum
    )
}
"#,
            "Calculate statistics (count, min, max, avg, sum) for numbers",
            r#"{"op": "wasm", "tool": "statistics"}"#,
        );

        // Tool 10: group_by_prefix - Group lines by their prefix
        self.compile_and_register(
            &compiler,
            "group_by_prefix",
            r#"
pub fn analyze(input: &str) -> String {
    // Input format: "prefix_length|text"
    let parts: Vec<&str> = input.splitn(2, '|').collect();
    if parts.len() != 2 {
        return "Error: Expected format 'prefix_length|text'".to_string();
    }

    let prefix_len: usize = match parts[0].parse() {
        Ok(n) => n,
        Err(_) => return "Error: prefix_length must be a number".to_string(),
    };
    let text = parts[1];

    let mut groups: HashMap<String, usize> = HashMap::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.len() >= prefix_len {
            let prefix = &trimmed[..prefix_len];
            *groups.entry(prefix.to_string()).or_insert(0) += 1;
        }
    }

    let mut pairs: Vec<_> = groups.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));

    pairs.iter()
        .map(|(prefix, count)| format!("{}: {}", prefix, count))
        .collect::<Vec<_>>()
        .join("\n")
}
"#,
            "Group lines by their prefix of specified length",
            r#"{"op": "wasm", "tool": "group_by_prefix", "args": "5|<context>"}"#,
        );
    }

    fn compile_and_register(
        &mut self,
        compiler: &RustCompiler,
        name: &str,
        code: &str,
        description: &str,
        example: &str,
    ) {
        match compiler.compile(code) {
            Ok(wasm_bytes) => {
                self.tools.insert(name.to_string(), Arc::new(wasm_bytes));
                self.descriptions.insert(
                    name.to_string(),
                    ToolDescription {
                        name: name.to_string(),
                        description: description.to_string(),
                        example: example.to_string(),
                    },
                );
            }
            Err(e) => {
                eprintln!("Failed to compile WASM tool '{}': {}", name, e);
            }
        }
    }
}

impl Default for WasmToolLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::WasmExecutor;
    use crate::wasm::WasmConfig;

    fn setup() -> (WasmToolLibrary, WasmExecutor) {
        let lib = WasmToolLibrary::new();
        let executor = WasmExecutor::new(WasmConfig::default()).unwrap();
        (lib, executor)
    }

    #[test]
    fn test_count_pattern() {
        let (lib, executor) = setup();
        let wasm = lib.get("count_pattern").expect("Tool should exist");

        let result = executor.execute(&wasm, "run_analyze", "ERROR|line1 ERROR here\nline2 ERROR again\nline3 ok");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "2");
    }

    #[test]
    fn test_count_lines_matching() {
        let (lib, executor) = setup();
        let wasm = lib.get("count_lines_matching").expect("Tool should exist");

        let result = executor.execute(&wasm, "run_analyze", "ERROR|line1 ERROR here\nline2 ERROR again\nline3 ok");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "2");
    }

    #[test]
    fn test_word_frequency() {
        let (lib, executor) = setup();
        let wasm = lib.get("word_frequency").expect("Tool should exist");

        let result = executor.execute(&wasm, "run_analyze", "hello world hello world hello");
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("hello: 3"));
        assert!(output.contains("world: 2"));
    }

    #[test]
    fn test_categorize_lines() {
        let (lib, executor) = setup();
        let wasm = lib.get("categorize_lines").expect("Tool should exist");

        let input = "ERROR,WARN,INFO|ERROR: db failed\nWARN: slow query\nINFO: started\nERROR: timeout";
        let result = executor.execute(&wasm, "run_analyze", input);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("ERROR: 2"));
        assert!(output.contains("WARN: 1"));
        assert!(output.contains("INFO: 1"));
    }

    #[test]
    fn test_statistics() {
        let (lib, executor) = setup();
        let wasm = lib.get("statistics").expect("Tool should exist");

        let result = executor.execute(&wasm, "run_analyze", "latency: 10ms, latency: 20ms, latency: 30ms");
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("count: 3"));
        assert!(output.contains("min: 10"));
        assert!(output.contains("max: 30"));
        assert!(output.contains("avg: 20"));
    }

    #[test]
    fn test_library_lists_all_tools() {
        let lib = WasmToolLibrary::new();
        let tools = lib.list();

        // Should have all 10 tools
        assert!(tools.len() >= 9, "Expected at least 9 tools, got {}", tools.len());

        // Check some specific tools exist
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"count_pattern"));
        assert!(names.contains(&"word_frequency"));
        assert!(names.contains(&"statistics"));
    }
}
