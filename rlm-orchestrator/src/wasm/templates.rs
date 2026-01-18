//! WASM Template Framework with SPI Hooks
//!
//! This module provides safe, pre-tested framework templates that the LLM
//! fills in with simple hook implementations. The framework handles:
//! - Iteration (safe line-by-line processing)
//! - Aggregation (HashMap, sorting, formatting)
//! - Error handling (no panics from LLM code)
//!
//! The LLM only provides simple, single-line processing functions.

/// Available template types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateType {
    /// Group lines by classification and count occurrences
    GroupCount,
    /// Filter lines matching a predicate
    FilterLines,
    /// Transform each line
    MapLines,
    /// Extract numbers and compute statistics
    NumericStats,
    /// Count lines matching a predicate
    CountMatching,
}

/// Pre-built hooks that LLMs can select by name
/// These are safe, tested implementations for common patterns
pub struct PrebuiltHooks;

impl PrebuiltHooks {
    /// Get a prebuilt hook by name for a given template type
    /// Returns (hook_code, description)
    pub fn get(template: TemplateType, hook_name: &str) -> Option<(&'static str, &'static str)> {
        match (template, hook_name) {
            // GroupCount hooks
            (TemplateType::GroupCount, "error_type") => Some((
                r#"fn classify(line: &str) -> Option<String> {
    // Extract error type: [ERROR] ErrorType
    if !line.contains("[ERROR]") { return None; }
    line.split_whitespace()
        .skip_while(|w| *w != "[ERROR]")
        .nth(1)
        .map(|s| s.to_string())
}"#,
                "Groups by error type after [ERROR] marker",
            )),
            (TemplateType::GroupCount, "log_level") => Some((
                r#"fn classify(line: &str) -> Option<String> {
    // Extract log level: [INFO], [ERROR], [WARN], etc.
    for level in &["[ERROR]", "[WARN]", "[INFO]", "[DEBUG]", "[TRACE]"] {
        if line.contains(level) { return Some(level.to_string()); }
    }
    None
}"#,
                "Groups by log level [ERROR]/[WARN]/[INFO]/etc.",
            )),
            (TemplateType::GroupCount, "http_status") => Some((
                r#"fn classify(line: &str) -> Option<String> {
    // Extract HTTP status code (3 digit number)
    line.split_whitespace()
        .find(|w| w.len() == 3 && w.chars().all(|c| c.is_ascii_digit()))
        .map(|s| s.to_string())
}"#,
                "Groups by HTTP status code (200, 404, 500, etc.)",
            )),
            (TemplateType::GroupCount, "ip_address") => Some((
                r#"fn classify(line: &str) -> Option<String> {
    // Extract IP address pattern
    line.split_whitespace()
        .find(|w| {
            let parts: Vec<&str> = w.split('.').collect();
            parts.len() == 4 && parts.iter().all(|p| p.parse::<u8>().is_ok())
        })
        .map(|s| s.to_string())
}"#,
                "Groups by IP address",
            )),
            (TemplateType::GroupCount, "first_word") => Some((
                r#"fn classify(line: &str) -> Option<String> {
    line.split_whitespace().next().map(|s| s.to_string())
}"#,
                "Groups by first word of each line",
            )),
            (TemplateType::GroupCount, "endpoint") => Some((
                r#"fn classify(line: &str) -> Option<String> {
    // Extract API endpoint (starts with /)
    line.split_whitespace()
        .find(|w| w.starts_with('/') && !w.contains("//"))
        .map(|s| s.to_string())
}"#,
                "Groups by API endpoint path",
            )),

            // FilterLines hooks
            (TemplateType::FilterLines, "errors") => Some((
                r#"fn filter(line: &str) -> bool {
    line.contains("[ERROR]") || line.contains("ERROR") || line.contains("error")
}"#,
                "Keeps lines containing error indicators",
            )),
            (TemplateType::FilterLines, "warnings") => Some((
                r#"fn filter(line: &str) -> bool {
    line.contains("[WARN]") || line.contains("WARN") || line.contains("warning")
}"#,
                "Keeps lines containing warning indicators",
            )),
            (TemplateType::FilterLines, "non_empty") => Some((
                r#"fn filter(line: &str) -> bool {
    !line.trim().is_empty()
}"#,
                "Keeps non-empty lines",
            )),

            // NumericStats hooks
            (TemplateType::NumericStats, "response_time_ms") => Some((
                r#"fn extract_number(line: &str) -> Option<i64> {
    // Extract response time like "123ms" or "123 ms"
    line.split_whitespace()
        .find(|w| w.ends_with("ms"))
        .and_then(|w| w.trim_end_matches("ms").parse().ok())
}"#,
                "Extracts response time in milliseconds",
            )),
            (TemplateType::NumericStats, "last_number") => Some((
                r#"fn extract_number(line: &str) -> Option<i64> {
    // Extract the last number on each line
    line.split_whitespace()
        .filter_map(|w| w.parse::<i64>().ok())
        .last()
}"#,
                "Extracts the last number on each line",
            )),
            (TemplateType::NumericStats, "first_number") => Some((
                r#"fn extract_number(line: &str) -> Option<i64> {
    // Extract the first number on each line
    line.split_whitespace()
        .filter_map(|w| w.parse::<i64>().ok())
        .next()
}"#,
                "Extracts the first number on each line",
            )),

            // CountMatching hooks
            (TemplateType::CountMatching, "errors") => Some((
                r#"fn matches(line: &str) -> bool {
    line.contains("[ERROR]")
}"#,
                "Counts lines with [ERROR]",
            )),
            (TemplateType::CountMatching, "warnings") => Some((
                r#"fn matches(line: &str) -> bool {
    line.contains("[WARN]")
}"#,
                "Counts lines with [WARN]",
            )),

            _ => None,
        }
    }

    /// List all prebuilt hooks for a template type
    pub fn list(template: TemplateType) -> Vec<(&'static str, &'static str)> {
        let all_hooks: Vec<(&'static str, &'static str)> = match template {
            TemplateType::GroupCount => vec![
                ("error_type", "Groups by error type after [ERROR] marker"),
                (
                    "log_level",
                    "Groups by log level [ERROR]/[WARN]/[INFO]/etc.",
                ),
                ("http_status", "Groups by HTTP status code"),
                ("ip_address", "Groups by IP address"),
                ("first_word", "Groups by first word"),
                ("endpoint", "Groups by API endpoint path"),
            ],
            TemplateType::FilterLines => vec![
                ("errors", "Keeps lines with error indicators"),
                ("warnings", "Keeps lines with warning indicators"),
                ("non_empty", "Keeps non-empty lines"),
            ],
            TemplateType::MapLines => vec![],
            TemplateType::NumericStats => vec![
                ("response_time_ms", "Extracts response time (ms)"),
                ("last_number", "Extracts last number on line"),
                ("first_number", "Extracts first number on line"),
            ],
            TemplateType::CountMatching => vec![
                ("errors", "Counts [ERROR] lines"),
                ("warnings", "Counts [WARN] lines"),
            ],
        };
        all_hooks
    }
}

impl TemplateType {
    /// Parse a template type from a string name
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "group_count" => Some(Self::GroupCount),
            "filter_lines" => Some(Self::FilterLines),
            "map_lines" => Some(Self::MapLines),
            "numeric_stats" => Some(Self::NumericStats),
            "count_matching" => Some(Self::CountMatching),
            _ => None,
        }
    }

    /// Get the SPI function signature the LLM must implement
    pub fn spi_signature(&self) -> &'static str {
        match self {
            Self::GroupCount => "fn classify(line: &str) -> Option<String>",
            Self::FilterLines => "fn filter(line: &str) -> bool",
            Self::MapLines => "fn transform(line: &str) -> String",
            Self::NumericStats => "fn extract_number(line: &str) -> Option<i64>",
            Self::CountMatching => "fn matches(line: &str) -> bool",
        }
    }

    /// Get example SPI implementation for the LLM
    pub fn spi_example(&self) -> &'static str {
        match self {
            Self::GroupCount => {
                r#"fn classify(line: &str) -> Option<String> {
    // Return Some(category) to count this line under that category
    // Return None to skip this line
    if line.contains("[ERROR]") {
        // Extract word after [ERROR]
        line.split_whitespace()
            .skip_while(|w| *w != "[ERROR]")
            .nth(1)
            .map(|s| s.to_string())
    } else {
        None
    }
}"#
            }
            Self::FilterLines => {
                r#"fn filter(line: &str) -> bool {
    // Return true to include this line, false to exclude
    line.contains("ERROR") && line.contains("timeout")
}"#
            }
            Self::MapLines => {
                r#"fn transform(line: &str) -> String {
    // Transform the line and return the result
    line.split_whitespace().take(3).collect::<Vec<_>>().join(" ")
}"#
            }
            Self::NumericStats => {
                r#"fn extract_number(line: &str) -> Option<i64> {
    // Extract a number from this line, or None to skip
    line.split_whitespace()
        .last()
        .and_then(|s| s.trim_end_matches("ms").parse().ok())
}"#
            }
            Self::CountMatching => {
                r#"fn matches(line: &str) -> bool {
    // Return true if this line should be counted
    line.contains("[ERROR]")
}"#
            }
        }
    }
}

/// Template definitions with framework code
pub struct TemplateFramework;

impl TemplateFramework {
    /// Generate Rust source for a template with LLM-provided hook.
    /// The returned code defines `pub fn analyze(input: &str) -> String`
    /// which the RustCompiler will wrap with WASM exports.
    pub fn generate_module(template: TemplateType, llm_hook: &str) -> String {
        let framework = Self::get_framework_code(template);

        // NOTE: We do NOT include WASM exports here - the RustCompiler adds them.
        // We just generate the pure Rust code: LLM hook + framework analyze function.
        format!(
            r#"// Template: {:?}
// LLM-provided hook:
{llm_hook}

// Framework code:
{framework}
"#,
            template,
            llm_hook = llm_hook,
            framework = framework
        )
    }

    /// Get the framework code for a template type
    fn get_framework_code(template: TemplateType) -> &'static str {
        match template {
            TemplateType::GroupCount => {
                r#"
pub fn analyze(input: &str) -> String {
    let mut counts: HashMap<String, usize> = HashMap::new();

    for line in input.lines() {
        // Call LLM-provided classify hook
        if let Some(key) = classify(line) {
            *counts.entry(key).or_insert(0) += 1;
        }
    }

    // Sort by count descending
    let mut pairs: Vec<_> = counts.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));

    // Format output
    pairs.iter()
        .map(|(k, c)| format!("{}: {}", k, c))
        .collect::<Vec<_>>()
        .join("\n")
}
"#
            }
            TemplateType::FilterLines => {
                r#"
pub fn analyze(input: &str) -> String {
    let mut results = Vec::new();

    for line in input.lines() {
        // Call LLM-provided filter hook
        if filter(line) {
            results.push(line);
        }
    }

    format!("{} matching lines:\n{}", results.len(), results.join("\n"))
}
"#
            }
            TemplateType::MapLines => {
                r#"
pub fn analyze(input: &str) -> String {
    let mut results = Vec::new();

    for line in input.lines() {
        // Call LLM-provided transform hook
        results.push(transform(line));
    }

    results.join("\n")
}
"#
            }
            TemplateType::NumericStats => {
                r#"
pub fn analyze(input: &str) -> String {
    let mut numbers = Vec::new();

    for line in input.lines() {
        // Call LLM-provided extract_number hook
        if let Some(n) = extract_number(line) {
            numbers.push(n);
        }
    }

    if numbers.is_empty() {
        return "No numbers extracted".to_string();
    }

    let count = numbers.len();
    let sum: i64 = numbers.iter().sum();
    let min = *numbers.iter().min().unwrap();
    let max = *numbers.iter().max().unwrap();
    let avg = sum as f64 / count as f64;

    // Sort for percentiles
    numbers.sort();
    let p50 = numbers[count / 2];
    let p95 = numbers[count * 95 / 100];
    let p99 = numbers[count * 99 / 100];

    format!(
        "count: {}\nsum: {}\nmin: {}\nmax: {}\navg: {:.2}\np50: {}\np95: {}\np99: {}",
        count, sum, min, max, avg, p50, p95, p99
    )
}
"#
            }
            TemplateType::CountMatching => {
                r#"
pub fn analyze(input: &str) -> String {
    let mut count = 0usize;

    for line in input.lines() {
        // Call LLM-provided matches hook
        if matches(line) {
            count += 1;
        }
    }

    count.to_string()
}
"#
            }
        }
    }

    /// Generate prompt documentation for the LLM
    pub fn prompt_documentation() -> String {
        let mut doc = String::from(
            r#"## WASM Templates (PREFERRED over raw rust_wasm)

Instead of writing full Rust code, use templates with simple hooks:

```json
{"op": "wasm_template", "template": "<template_name>", "hook": "<your_hook_code>", "store": "result"}
```

Available templates:

"#,
        );

        for template in [
            TemplateType::GroupCount,
            TemplateType::FilterLines,
            TemplateType::MapLines,
            TemplateType::NumericStats,
            TemplateType::CountMatching,
        ] {
            doc.push_str(&format!(
                "### {}\n",
                format!("{:?}", template).to_lowercase()
            ));
            doc.push_str(&format!("Hook signature: `{}`\n", template.spi_signature()));
            doc.push_str("Example:\n```rust\n");
            doc.push_str(template.spi_example());
            doc.push_str("\n```\n\n");
        }

        doc.push_str(r#"
IMPORTANT: Hook code must be a complete function definition matching the signature.
The framework handles iteration, aggregation, and error-safe processing.
Your hook only processes ONE line at a time.

Example for counting error types:
```json
{"op": "wasm_template", "template": "group_count", "hook": "fn classify(line: &str) -> Option<String> { if line.contains(\"[ERROR]\") { line.split_whitespace().skip_while(|w| *w != \"[ERROR]\").nth(1).map(|s| s.to_string()) } else { None } }", "store": "error_counts"}
{"op": "final_var", "name": "error_counts"}
```
"#);

        doc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_group_count_module() {
        let hook = r#"fn classify(line: &str) -> Option<String> {
    if line.contains("[ERROR]") {
        Some("error".to_string())
    } else {
        None
    }
}"#;
        let source = TemplateFramework::generate_module(TemplateType::GroupCount, hook);
        assert!(source.contains("fn classify"));
        assert!(source.contains("fn analyze"));
        assert!(source.contains("HashMap"));
    }

    #[test]
    fn test_template_types() {
        assert_eq!(
            TemplateType::parse("group_count"),
            Some(TemplateType::GroupCount)
        );
        assert_eq!(
            TemplateType::parse("filter_lines"),
            Some(TemplateType::FilterLines)
        );
        assert_eq!(TemplateType::parse("invalid"), None);
    }
}
