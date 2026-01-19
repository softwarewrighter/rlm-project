//! Structured command execution for RLM
//!
//! Instead of executing arbitrary Python code, the LLM outputs JSON commands
//! that are executed by this pure-Rust executor.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

use crate::codegen::{CodeGenConfig, CodeGenerator};
use crate::wasm::{
    CompilerConfig, ModuleCache, PrebuiltHooks, RustCompiler, TemplateFramework, TemplateType,
    WasmConfig, WasmExecutor, WasmToolLibrary,
};

/// Safely truncate a string at a valid UTF-8 character boundary.
/// Returns the largest prefix of `s` that is at most `max_bytes` bytes.
fn truncate_to_char_boundary(s: &str, max_bytes: usize) -> &str {
    if max_bytes >= s.len() {
        return s;
    }
    // Find the largest valid char boundary <= max_bytes
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Sanitize string to ASCII-only for safe WASM processing.
/// Non-ASCII bytes are replaced with '?' to preserve structure.
fn to_ascii(s: &str) -> String {
    s.bytes()
        .map(|b| if b.is_ascii() { b as char } else { '?' })
        .collect()
}

/// Split text into chunks with semantic awareness.
///
/// This function tries to break at natural boundaries (paragraphs, sections)
/// to avoid splitting mid-sentence or mid-thought. With semantic breaks,
/// overlap is not needed to preserve context.
///
/// Strategy:
/// 1. Split on double newlines (paragraphs) or section markers (===, ---, etc.)
/// 2. Group paragraphs into chunks up to target_size
/// 3. If a single paragraph exceeds target_size, include it as-is (don't split mid-paragraph)
fn split_into_chunks(text: &str, target_size: usize, _step_size: usize) -> Vec<String> {
    // Find paragraph boundaries: double newlines, or section markers
    let paragraph_patterns = ["\n\n", "\n===", "\n---", "\n***", "\n["];

    // Split into paragraphs/sections
    let mut paragraphs: Vec<&str> = Vec::new();
    let mut last_end = 0;
    let mut pos = 0;

    while pos < text.len() {
        // Find the next paragraph break
        let mut found_break = None;
        for pattern in &paragraph_patterns {
            if let Some(idx) = text[pos..].find(pattern) {
                let abs_idx = pos + idx;
                if found_break.is_none() || abs_idx < found_break.unwrap() {
                    found_break = Some(abs_idx);
                }
            }
        }

        match found_break {
            Some(break_idx) => {
                // Include up to the break point
                if break_idx > last_end {
                    paragraphs.push(&text[last_end..break_idx]);
                }
                // Skip the break pattern (find where content resumes)
                pos = break_idx + 1;
                while pos < text.len() && text[pos..].starts_with('\n') {
                    pos += 1;
                }
                last_end = pos;
            }
            None => {
                // No more breaks, take the rest
                if last_end < text.len() {
                    paragraphs.push(&text[last_end..]);
                }
                break;
            }
        }
    }

    // Group paragraphs into chunks up to target_size
    let mut chunks: Vec<String> = Vec::new();
    let mut current_chunk = String::new();

    for para in paragraphs {
        let para_trimmed = para.trim();
        if para_trimmed.is_empty() {
            continue;
        }

        // If adding this paragraph would exceed target, start a new chunk
        // But always include at least one paragraph per chunk
        if !current_chunk.is_empty() && current_chunk.len() + para_trimmed.len() + 2 > target_size {
            chunks.push(current_chunk);
            current_chunk = String::new();
        }

        if !current_chunk.is_empty() {
            current_chunk.push_str("\n\n");
        }
        current_chunk.push_str(para_trimmed);
    }

    // Don't forget the last chunk
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    // If we ended up with no chunks (unusual), just do simple splitting
    if chunks.is_empty() && !text.is_empty() {
        let chars: Vec<char> = text.chars().collect();
        for chunk in chars.chunks(target_size) {
            chunks.push(chunk.iter().collect());
        }
    }

    chunks
}

/// Errors from command execution
#[derive(Error, Debug)]
pub enum CommandError {
    #[error("Invalid command: {0}")]
    InvalidCommand(String),

    #[error("JSON parse error: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("Regex error: {0}")]
    RegexError(#[from] regex::Error),

    #[error("Variable not found: {0}")]
    VariableNotFound(String),

    #[error("Invalid slice range: {start}..{end} for length {len}")]
    InvalidSlice {
        start: usize,
        end: usize,
        len: usize,
    },

    #[error("LLM query failed: {0}")]
    LlmError(String),

    #[error("Max sub-calls exceeded: {0}")]
    MaxSubCalls(usize),

    #[error("WASM error: {0}")]
    WasmError(#[from] crate::wasm::WasmError),

    #[error("WASM module not found: {0}")]
    WasmModuleNotFound(String),

    #[error("Rust compilation failed:\n{0}")]
    RustCompileError(String),

    #[error("Rust compiler not available: {0}")]
    RustCompilerUnavailable(String),

    #[error("Code generation failed: {0}")]
    CodeGenError(String),

    #[error("Code generation LLM not configured")]
    CodeGenNotConfigured,

    #[error("Max recursion depth ({0}) exceeded")]
    MaxRecursionDepth(usize),

    #[error("LLM delegate callback not configured")]
    LlmDelegateNotConfigured,
}

/// A single command that can be executed
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Command {
    /// Slice the context: {"op": "slice", "start": 0, "end": 1000}
    /// Supports Python-style negative indices (e.g., -20 means 20 from the end)
    Slice {
        start: i64,
        #[serde(default)]
        end: Option<i64>,
        #[serde(default)]
        store: Option<String>,
    },

    /// Get specific lines: {"op": "lines", "start": 0, "end": 100}
    /// Supports Python-style negative indices (e.g., -10 means 10 lines from the end)
    Lines {
        #[serde(default)]
        start: Option<i64>,
        #[serde(default)]
        end: Option<i64>,
        #[serde(default)]
        store: Option<String>,
    },

    /// Regex search: {"op": "regex", "pattern": "class \\w+"}
    Regex {
        pattern: String,
        #[serde(default)]
        on: Option<String>,
        #[serde(default)]
        store: Option<String>,
    },

    /// Find text: {"op": "find", "text": "error"}
    Find {
        text: String,
        #[serde(default)]
        on: Option<String>,
        #[serde(default)]
        store: Option<String>,
    },

    /// Count something: {"op": "count", "what": "lines"}
    Count {
        what: CountTarget,
        #[serde(default)]
        on: Option<String>,
        #[serde(default)]
        store: Option<String>,
    },

    /// Split text: {"op": "split", "delimiter": "\n"}
    Split {
        delimiter: String,
        #[serde(default)]
        on: Option<String>,
        #[serde(default)]
        store: Option<String>,
    },

    /// Get length: {"op": "len"}
    Len {
        #[serde(default)]
        on: Option<String>,
        #[serde(default)]
        store: Option<String>,
    },

    /// Set a variable: {"op": "set", "name": "foo", "value": "bar"}
    Set { name: String, value: String },

    /// Get a variable: {"op": "get", "name": "foo"}
    Get { name: String },

    /// Print/output a value: {"op": "print", "value": "..."}
    Print {
        #[serde(default)]
        value: Option<String>,
        #[serde(default)]
        var: Option<String>,
    },

    /// Call sub-LM: {"op": "llm_query", "prompt": "Summarize: ..."}
    LlmQuery {
        prompt: String,
        #[serde(default)]
        store: Option<String>,
    },

    /// Recursive LLM delegation: {"op": "llm_delegate", "task": "Analyze relationships", "on": "extracted_data"}
    ///
    /// Creates a nested RLM instance with full tool access (L1-L3) on a subset of data.
    /// Unlike llm_query (which is just a simple LLM call), llm_delegate runs a full
    /// RLM loop with iteration, variable storage, and command execution.
    ///
    /// Use cases:
    /// - Semantic analysis of extracted data chunks
    /// - Cross-referencing information across sections
    /// - Complex reasoning that requires tool access
    ///
    /// Example: {"op": "llm_delegate", "task": "Summarize key claims from each witness", "on": "witnesses", "store": "summary"}
    LlmDelegate {
        /// Task description for the nested RLM instance
        task: String,
        /// Source variable to use as context (None = full original context)
        #[serde(default)]
        on: Option<String>,
        /// Maximum iterations for the nested RLM (default: 5)
        #[serde(default = "default_delegate_iterations")]
        max_iterations: Option<usize>,
        /// Capability levels for nested RLM (default: ["dsl", "wasm"])
        /// Note: llm_delegate is NOT available in nested RLM to prevent infinite recursion
        #[serde(default)]
        levels: Option<Vec<String>>,
        /// Store result in variable
        #[serde(default)]
        store: Option<String>,
    },

    /// Chunked reduce over context: {"op": "llm_reduce", "directive": "Extract names and claims", "store": "result"}
    ///
    /// Processes the context in chunks, passing accumulated state through each:
    /// 1. Split context into chunks of `chunk_size` (default: 10000 chars) with optional overlap
    /// 2. For each chunk, call worker LLM with: directive + previous_result + chunk
    /// 3. Worker returns updated accumulated result, passed to next chunk
    /// 4. Final accumulated result stored in variable
    ///
    /// Use cases:
    /// - Processing large documents that don't fit in worker context
    /// - Accumulating information across sections (like map-reduce)
    /// - Summarizing or extracting from long texts
    LlmReduce {
        /// Directive for each worker (what to extract/analyze from each chunk)
        directive: String,
        /// Chunk size in characters (default: 10000)
        #[serde(default = "default_chunk_size")]
        chunk_size: Option<usize>,
        /// Overlap between chunks in characters (default: 500)
        /// Helps preserve context across chunk boundaries
        #[serde(default = "default_overlap")]
        overlap: Option<usize>,
        /// Source variable to use as context (None = full original context)
        #[serde(default)]
        on: Option<String>,
        /// Store result in variable
        #[serde(default)]
        store: Option<String>,
    },

    /// Final answer: {"op": "final", "answer": "The result is..."}
    Final { answer: String },

    /// Final from variable: {"op": "final_var", "name": "result"}
    FinalVar { name: String },

    /// Execute pre-compiled WASM module: {"op": "wasm", "module": "line_counter"}
    /// Or pre-built tool: {"op": "wasm", "tool": "count_pattern", "args": "ERROR|<context>"}
    Wasm {
        /// Name of pre-compiled module from WasmLibrary (legacy)
        #[serde(default)]
        module: Option<String>,
        /// Name of pre-built tool from WasmToolLibrary
        #[serde(default)]
        tool: Option<String>,
        /// Arguments to pass to tool (format: "arg1|arg2|..." prepended to input)
        #[serde(default)]
        args: Option<String>,
        /// Function to call (default: "analyze" or "run_analyze" for tools)
        #[serde(default = "default_wasm_function")]
        function: String,
        /// Variable to use as input (default: context)
        #[serde(default)]
        on: Option<String>,
        /// Store result in variable
        #[serde(default)]
        store: Option<String>,
    },

    /// Compile and execute WAT code: {"op": "wasm_wat", "wat": "(module ...)"}
    WasmWat {
        /// WebAssembly Text format code
        wat: String,
        /// Function to call (default: "analyze")
        #[serde(default = "default_wasm_function")]
        function: String,
        /// Variable to use as input (default: context)
        #[serde(default)]
        on: Option<String>,
        /// Store result in variable
        #[serde(default)]
        store: Option<String>,
    },

    /// Compile and execute Rust code: {"op": "rust_wasm", "code": "pub fn analyze..."}
    ///
    /// The code must define: `pub fn analyze(input: &str) -> String`
    /// Available: HashMap, HashSet, Vec, iterators, string operations
    /// NOT available: std::fs, std::net, std::process (sandboxed)
    RustWasm {
        /// Rust source code (must define `pub fn analyze(input: &str) -> String`)
        code: String,
        /// Variable to use as input (default: context)
        #[serde(default)]
        on: Option<String>,
        /// Store result in variable
        #[serde(default)]
        store: Option<String>,
    },

    /// Use a template with hook: {"op": "wasm_template", "template": "group_count", "prebuilt": "error_type"}
    /// Or with custom hook: {"op": "wasm_template", "template": "group_count", "hook": "fn classify..."}
    ///
    /// PREFER using prebuilt hooks - they are tested and reliable!
    /// Available templates: group_count, filter_lines, map_lines, numeric_stats, count_matching
    WasmTemplate {
        /// Template name: group_count, filter_lines, map_lines, numeric_stats, count_matching
        template: String,
        /// Prebuilt hook name (PREFERRED): error_type, log_level, http_status, ip_address, etc.
        #[serde(default)]
        prebuilt: Option<String>,
        /// Custom hook function code (fallback if no prebuilt matches)
        #[serde(default)]
        hook: Option<String>,
        /// Variable to use as input (default: context)
        #[serde(default)]
        on: Option<String>,
        /// Store result in variable
        #[serde(default)]
        store: Option<String>,
    },

    /// Two-LLM code generation: {"op": "rust_wasm_intent", "intent": "count each error type..."}
    ///
    /// Uses a specialized coding LLM (qwen2.5-coder) to generate Rust code from a
    /// natural language description. The coding LLM is constrained to use safe
    /// helper functions that work reliably in WASM.
    ///
    /// Example: {"op": "rust_wasm_intent", "intent": "Count occurrences of each error type after [ERROR] marker and return sorted by frequency", "store": "counts"}
    RustWasmIntent {
        /// Natural language description of what the code should do
        intent: String,
        /// Variable to use as input (default: context)
        #[serde(default)]
        on: Option<String>,
        /// Store result in variable
        #[serde(default)]
        store: Option<String>,
    },

    /// Two-LLM streaming reduce for large datasets: {"op": "rust_wasm_reduce_intent", "intent": "count unique IPs..."}
    ///
    /// Like rust_wasm_intent but uses a streaming reduce pattern to handle arbitrarily
    /// large datasets. The coding LLM generates a State struct and reduce functions
    /// (init_state, process_line, finalize) that are applied line-by-line.
    ///
    /// Use this for large datasets that would overflow WASM memory if processed all at once.
    /// The orchestrator handles chunking the data and calling the reducer for each line.
    ///
    /// Limitations: Algorithms that require all data at once (median, percentiles) cannot
    /// be computed using reduce. Use rust_wasm_intent for those (with smaller datasets).
    ///
    /// Example: {"op": "rust_wasm_reduce_intent", "intent": "Count unique IP addresses and rank top 10 most active", "store": "ip_counts"}
    RustWasmReduceIntent {
        /// Natural language description of what to compute via reduce
        intent: String,
        /// Variable to use as input (default: context)
        #[serde(default)]
        on: Option<String>,
        /// Store result in variable
        #[serde(default)]
        store: Option<String>,
        /// Chunk size in bytes (default: 64KB for efficient processing)
        #[serde(default)]
        chunk_size: Option<usize>,
    },

    /// Two-LLM stateless map-reduce for large datasets: {"op": "rust_wasm_mapreduce", "intent": "count error types...", "combiner": "count"}
    ///
    /// A more scalable alternative to rust_wasm_reduce_intent. Uses a true map-reduce pattern:
    /// 1. Map (WASM): Each line is transformed to zero or more (key, value) pairs - NO STATE in WASM
    /// 2. Shuffle (Native Rust): Pairs are grouped by key using native HashMap
    /// 3. Reduce (Native Rust): Values for each key are combined (sum, count, max, min, list)
    ///
    /// This is more reliable than rust_wasm_reduce_intent because:
    /// - WASM is completely stateless (no growing state between chunks)
    /// - All aggregation happens in native Rust (not WASM)
    /// - HashMap operations use native Rust (not WASM)
    ///
    /// Example: {"op": "rust_wasm_mapreduce", "intent": "Extract error type from each [ERROR] line", "combiner": "count", "store": "error_counts"}
    #[serde(alias = "rust_wasm_mapreduce")]
    RustWasmMapReduce {
        /// Natural language description of what key-value pairs to emit for each line
        intent: String,
        /// How to combine values for each key: "count", "sum", "max", "min", "list", "first", "last"
        combiner: String,
        /// Variable to use as input (default: context)
        #[serde(default)]
        on: Option<String>,
        /// Store result in variable
        #[serde(default)]
        store: Option<String>,
        /// Sort output by value descending (default: true for count/sum)
        #[serde(default)]
        sort_desc: Option<bool>,
        /// Limit output to top N results (default: all)
        #[serde(default)]
        limit: Option<usize>,
    },

    /// Native CLI binary for large dataset analysis: {"op": "rust_cli_intent", "intent": "count error types..."}
    ///
    /// Compiles Rust to a native binary (not WASM) for maximum compatibility and performance.
    /// The binary reads from stdin and writes to stdout.
    ///
    /// SECURITY NOTE: Native binaries run WITHOUT sandbox protection.
    /// Code validation prevents dangerous operations, but this is less secure than WASM.
    /// Future: Add sandboxing via Docker/LXC/seccomp.
    ///
    /// Advantages over WASM:
    /// - Full Rust standard library (HashMap, regex, all string ops)
    /// - No memory limits
    /// - No TwoWaySearcher/memcmp crashes
    /// - Faster execution for large datasets
    ///
    /// Example: {"op": "rust_cli_intent", "intent": "Count each error type and rank by frequency", "store": "error_counts"}
    RustCliIntent {
        /// Natural language description of what to compute
        intent: String,
        /// Variable to use as input (default: context)
        #[serde(default)]
        on: Option<String>,
        /// Store result in variable
        #[serde(default)]
        store: Option<String>,
        /// Timeout in seconds (default: 30)
        #[serde(default)]
        timeout_secs: Option<u64>,
    },
}

fn default_wasm_function() -> String {
    "analyze".to_string()
}

fn default_delegate_iterations() -> Option<usize> {
    Some(10) // Increased from 5 to give workers more time
}

fn default_chunk_size() -> Option<usize> {
    Some(10000) // 10K characters per chunk
}

fn default_overlap() -> Option<usize> {
    Some(500) // 500 char overlap between chunks for context continuity
}

impl Command {
    /// Get the capability level required for this command
    ///
    /// Returns one of: "dsl", "wasm", "cli", "llm_delegation"
    pub fn required_level(&self) -> &'static str {
        match self {
            // Level 1: DSL - Safe text operations
            Command::Slice { .. }
            | Command::Lines { .. }
            | Command::Regex { .. }
            | Command::Find { .. }
            | Command::Count { .. }
            | Command::Split { .. }
            | Command::Len { .. }
            | Command::Set { .. }
            | Command::Get { .. }
            | Command::Print { .. }
            | Command::Final { .. }
            | Command::FinalVar { .. } => "dsl",

            // Level 2: WASM - Sandboxed computation
            Command::Wasm { .. }
            | Command::WasmWat { .. }
            | Command::RustWasm { .. }
            | Command::WasmTemplate { .. }
            | Command::RustWasmIntent { .. }
            | Command::RustWasmReduceIntent { .. }
            | Command::RustWasmMapReduce { .. } => "wasm",

            // Level 3: CLI - Native binary execution
            Command::RustCliIntent { .. } => "cli",

            // Level 4: LLM Delegation - Chunk-based LLM analysis
            Command::LlmQuery { .. }
            | Command::LlmDelegate { .. }
            | Command::LlmReduce { .. } => "llm_delegation",
        }
    }

    /// Get the operation name for this command (for logging/display)
    pub fn op_name(&self) -> &'static str {
        match self {
            Command::Slice { .. } => "slice",
            Command::Lines { .. } => "lines",
            Command::Regex { .. } => "regex",
            Command::Find { .. } => "find",
            Command::Count { .. } => "count",
            Command::Split { .. } => "split",
            Command::Len { .. } => "len",
            Command::Set { .. } => "set",
            Command::Get { .. } => "get",
            Command::Print { .. } => "print",
            Command::LlmQuery { .. } => "llm_query",
            Command::LlmDelegate { .. } => "llm_delegate",
            Command::LlmReduce { .. } => "llm_reduce",
            Command::Final { .. } => "final",
            Command::FinalVar { .. } => "final_var",
            Command::Wasm { .. } => "wasm",
            Command::WasmWat { .. } => "wasm_wat",
            Command::RustWasm { .. } => "rust_wasm",
            Command::WasmTemplate { .. } => "wasm_template",
            Command::RustWasmIntent { .. } => "rust_wasm_intent",
            Command::RustWasmReduceIntent { .. } => "rust_wasm_reduce_intent",
            Command::RustWasmMapReduce { .. } => "rust_wasm_mapreduce",
            Command::RustCliIntent { .. } => "rust_cli_intent",
        }
    }
}

/// What to count
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CountTarget {
    Lines,
    Chars,
    Words,
    Matches,
}

/// Result of command execution
#[derive(Debug)]
pub enum ExecutionResult {
    /// Continue executing more commands
    Continue { output: String, sub_calls: usize },
    /// Final answer reached
    Final { answer: String, sub_calls: usize },
}

/// Callback type for llm_query
pub type LlmQueryCallback = Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>;

/// Parameters for llm_delegate callback
#[derive(Debug, Clone)]
pub struct LlmDelegateParams {
    /// The task/query for the nested RLM instance
    pub task: String,
    /// The context data for the nested RLM
    pub context: String,
    /// Maximum iterations for the nested RLM
    pub max_iterations: usize,
    /// Capability levels for the nested RLM (e.g., ["dsl", "wasm"])
    pub levels: Vec<String>,
    /// Current recursion depth (for tracking/limiting)
    pub current_depth: usize,
}

/// Summary of a nested iteration (for progress reporting)
#[derive(Debug, Clone)]
pub struct NestedIterationSummary {
    pub step: usize,
    pub llm_response_preview: String,
    pub commands_preview: String,
    pub output_preview: String,
    pub has_error: bool,
}

/// Result from llm_delegate callback
#[derive(Debug)]
pub struct LlmDelegateResult {
    /// The final answer from the nested RLM
    pub answer: String,
    /// Number of iterations the nested RLM took
    pub iterations: usize,
    /// Whether the nested RLM succeeded
    pub success: bool,
    /// Summary of each nested iteration (for progress visibility)
    pub nested_history: Vec<NestedIterationSummary>,
}

/// Callback type for llm_delegate (runs nested RLM instance)
pub type LlmDelegateCallback =
    Arc<dyn Fn(LlmDelegateParams) -> Result<LlmDelegateResult, String> + Send + Sync>;

/// Command executor with variable store
pub struct CommandExecutor {
    /// Variable store
    variables: HashMap<String, String>,
    /// The context being analyzed
    context: String,
    /// Last result (for chaining)
    last_result: String,
    /// Callback for llm_query
    llm_callback: Option<LlmQueryCallback>,
    /// Callback for llm_delegate (nested RLM)
    llm_delegate_callback: Option<LlmDelegateCallback>,
    /// Current recursion depth (0 = root, 1 = first nested, etc.)
    recursion_depth: usize,
    /// Maximum allowed recursion depth
    max_recursion_depth: usize,
    /// Default capability levels for nested RLM
    nested_levels: Vec<String>,
    /// Sub-call counter
    sub_calls: usize,
    /// Max sub-calls allowed
    max_sub_calls: usize,
    /// WASM executor for dynamic code
    wasm_executor: Option<WasmExecutor>,
    /// Library of pre-built WASM tools
    wasm_tool_library: Option<WasmToolLibrary>,
    /// Rust to WASM compiler (None if rustc not available)
    rust_compiler: Option<RustCompiler>,
    /// Cache for compiled WASM modules
    wasm_cache: ModuleCache,
    /// Last rust_wasm compile time in milliseconds (for instrumentation)
    last_compile_time_ms: u64,
    /// Time spent executing WASM (not compiling)
    last_wasm_run_time_ms: u64,
    /// Time spent generating code via LLM (for CLI intent)
    last_codegen_time_ms: u64,
    /// Time spent executing CLI binary (separate from WASM)
    last_cli_run_time_ms: u64,
    /// Code generator for two-LLM architecture (None if not configured)
    code_generator: Option<CodeGenerator>,
    /// Directory for CLI binary cache
    cli_binary_cache_dir: std::path::PathBuf,
    /// Last nested history from llm_delegate (for progress visibility)
    last_nested_history: Vec<NestedIterationSummary>,
    /// Last llm_delegate depth (for progress events)
    last_delegate_depth: usize,
}

impl CommandExecutor {
    /// Create a new executor with default WASM settings
    pub fn new(context: String, max_sub_calls: usize) -> Self {
        Self::with_wasm_config(context, max_sub_calls, &crate::WasmConfig::default())
    }

    /// Create a new executor with custom WASM configuration
    pub fn with_wasm_config(
        context: String,
        max_sub_calls: usize,
        wasm_config: &crate::WasmConfig,
    ) -> Self {
        // Initialize WASM executor if enabled
        let wasm_executor = if wasm_config.enabled {
            let config = WasmConfig {
                fuel_limit: wasm_config.fuel_limit,
                memory_limit: wasm_config.memory_limit,
                timeout_ms: 5000,
            };
            WasmExecutor::new(config).ok()
        } else {
            None
        };

        // Initialize Rust compiler if enabled
        let rust_compiler = if wasm_config.rust_wasm_enabled {
            let config = CompilerConfig {
                rustc_path: wasm_config
                    .rustc_path
                    .as_ref()
                    .map(std::path::PathBuf::from),
                timeout_secs: 30,
                opt_level: "2".to_string(),
            };
            match RustCompiler::new(config) {
                Ok(compiler) => {
                    tracing::info!("Rust WASM compiler available");
                    Some(compiler)
                }
                Err(e) => {
                    tracing::debug!("Rust WASM compiler not available: {}", e);
                    None
                }
            }
        } else {
            tracing::debug!("rust_wasm disabled by configuration");
            None
        };

        // Initialize module cache
        let wasm_cache = if let Some(ref dir) = wasm_config.cache_dir {
            use crate::wasm::CacheConfig;
            ModuleCache::new(CacheConfig {
                memory_size: wasm_config.cache_size,
                disk_dir: Some(std::path::PathBuf::from(dir)),
                max_disk_bytes: 100 * 1024 * 1024,
            })
        } else {
            ModuleCache::memory_only(wasm_config.cache_size)
        };

        // Initialize tool library if WASM is enabled and compiler is available
        let wasm_tool_library = if wasm_config.enabled && wasm_config.rust_wasm_enabled {
            // Tool library compilation is slow, make it optional
            if std::env::var("RLM_PRECOMPILE_TOOLS").is_ok() {
                tracing::info!("Pre-compiling WASM tool library...");
                Some(WasmToolLibrary::new())
            } else {
                None
            }
        } else {
            None
        };

        // Initialize code generator if configured
        let code_generator = if let Some(ref url) = wasm_config.codegen_url {
            use crate::codegen::CodeGenProviderType;
            let provider_type = if wasm_config.codegen_provider == "litellm" {
                CodeGenProviderType::LiteLLM
            } else {
                CodeGenProviderType::Ollama
            };
            let api_key = std::env::var("LITELLM_MASTER_KEY")
                .or_else(|_| std::env::var("LITELLM_API_KEY"))
                .ok();
            tracing::info!(
                "Code generation LLM configured: {} via {:?} (model: {})",
                url,
                provider_type,
                wasm_config.codegen_model
            );
            Some(CodeGenerator::new(CodeGenConfig {
                provider_type,
                url: url.clone(),
                model: wasm_config.codegen_model.clone(),
                api_key,
                temperature: 0.1,
            }))
        } else {
            None
        };

        // Initialize CLI binary cache directory
        let cli_binary_cache_dir = if let Some(ref dir) = wasm_config.cache_dir {
            std::path::PathBuf::from(dir).join("cli_binaries")
        } else {
            std::env::temp_dir().join("rlm_cli_cache")
        };
        // Create the directory if it doesn't exist
        if let Err(e) = std::fs::create_dir_all(&cli_binary_cache_dir) {
            tracing::warn!(
                "Failed to create CLI cache dir {:?}: {}",
                cli_binary_cache_dir,
                e
            );
        }

        Self {
            variables: HashMap::new(),
            context,
            last_result: String::new(),
            llm_callback: None,
            llm_delegate_callback: None,
            recursion_depth: 0,
            max_recursion_depth: 3,
            nested_levels: vec!["dsl".to_string(), "wasm".to_string()],
            sub_calls: 0,
            max_sub_calls,
            wasm_executor,
            wasm_tool_library,
            rust_compiler,
            wasm_cache,
            last_compile_time_ms: 0,
            last_wasm_run_time_ms: 0,
            last_codegen_time_ms: 0,
            last_cli_run_time_ms: 0,
            code_generator,
            cli_binary_cache_dir,
            last_nested_history: Vec::new(),
            last_delegate_depth: 0,
        }
    }

    /// Get the last rust_wasm compile time in milliseconds (for instrumentation)
    pub fn last_compile_time_ms(&self) -> u64 {
        self.last_compile_time_ms
    }

    /// Get the last WASM execution time in milliseconds (for instrumentation)
    pub fn last_wasm_run_time_ms(&self) -> u64 {
        self.last_wasm_run_time_ms
    }

    /// Get the last CLI code generation time in milliseconds (for instrumentation)
    pub fn last_codegen_time_ms(&self) -> u64 {
        self.last_codegen_time_ms
    }

    /// Get the last CLI binary execution time in milliseconds (for instrumentation)
    pub fn last_cli_run_time_ms(&self) -> u64 {
        self.last_cli_run_time_ms
    }

    /// Get the last nested history from llm_delegate (for progress visibility)
    pub fn last_nested_history(&self) -> &[NestedIterationSummary] {
        &self.last_nested_history
    }

    /// Get the last delegate depth (for progress events)
    pub fn last_delegate_depth(&self) -> usize {
        self.last_delegate_depth
    }

    /// Clear the nested history (call before each command execution)
    pub fn clear_nested_history(&mut self) {
        self.last_nested_history.clear();
        self.last_delegate_depth = 0;
    }

    /// Check if Rust WASM compilation is available
    pub fn rust_wasm_available(&self) -> bool {
        self.rust_compiler.is_some() && self.wasm_executor.is_some()
    }

    /// Set the LLM callback
    pub fn with_llm_callback(mut self, callback: LlmQueryCallback) -> Self {
        self.llm_callback = Some(callback);
        self
    }

    /// Set the LLM delegate callback for nested RLM instances
    pub fn with_llm_delegate_callback(mut self, callback: LlmDelegateCallback) -> Self {
        self.llm_delegate_callback = Some(callback);
        self
    }

    /// Set the current recursion depth (for nested RLM instances)
    pub fn with_recursion_depth(mut self, depth: usize) -> Self {
        self.recursion_depth = depth;
        self
    }

    /// Set the maximum recursion depth
    pub fn with_max_recursion_depth(mut self, depth: usize) -> Self {
        self.max_recursion_depth = depth;
        self
    }

    /// Set the default nested levels for llm_delegate
    pub fn with_nested_levels(mut self, levels: Vec<String>) -> Self {
        self.nested_levels = levels;
        self
    }

    /// Get a variable or the context
    fn resolve_source(&self, name: &Option<String>) -> Result<&str, CommandError> {
        match name {
            None => Ok(&self.context),
            Some(n) if n == "_" || n == "last" => Ok(&self.last_result),
            Some(n) if n == "context" => Ok(&self.context),
            Some(n) => {
                // Strip leading $ if present (LLMs often include it)
                let var_name = n.strip_prefix('$').unwrap_or(n);
                self.variables
                    .get(var_name)
                    .map(|s| s.as_str())
                    .ok_or_else(|| CommandError::VariableNotFound(var_name.to_string()))
            }
        }
    }

    /// Store result in variable or last_result
    fn store_result(&mut self, store: &Option<String>, value: String) {
        if let Some(name) = store {
            self.variables.insert(name.clone(), value.clone());
        }
        self.last_result = value;
    }

    /// Resolve a Python-style index (negative = from end) to a positive index
    /// Returns None if the resulting index is out of bounds
    fn resolve_index(idx: i64, len: usize) -> Option<usize> {
        if idx >= 0 {
            let pos = idx as usize;
            if pos <= len {
                Some(pos)
            } else {
                Some(len) // Clamp to length
            }
        } else {
            // Negative index: count from end
            let from_end = (-idx) as usize;
            if from_end <= len {
                Some(len - from_end)
            } else {
                Some(0) // Clamp to 0
            }
        }
    }

    /// Execute a sequence of commands from JSON
    pub fn execute_json(&mut self, json: &str) -> Result<ExecutionResult, CommandError> {
        // Try to parse as array of commands first, then single command
        let commands: Vec<Command> = if json.trim().starts_with('[') {
            serde_json::from_str(json)?
        } else {
            // First, try to parse as a single multi-line object
            if let Ok(cmd) = serde_json::from_str::<Command>(json) {
                vec![cmd]
            } else {
                // Try parsing multiple JSON objects (one per line - single-line format)
                let mut cmds = Vec::new();
                for line in json.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with("//") {
                        continue;
                    }
                    // Only try single-line JSON (starts AND ends with braces)
                    if line.starts_with('{') && line.ends_with('}')
                        && let Ok(cmd) = serde_json::from_str(line) {
                            cmds.push(cmd);
                        }
                }
                if cmds.is_empty() {
                    // Nothing worked, return the error from trying to parse the whole thing
                    vec![serde_json::from_str(json)?]
                } else {
                    cmds
                }
            }
        };

        self.execute_commands(&commands)
    }

    /// Execute a sequence of commands
    pub fn execute_commands(
        &mut self,
        commands: &[Command],
    ) -> Result<ExecutionResult, CommandError> {
        let mut output = String::new();
        let initial_sub_calls = self.sub_calls;

        for cmd in commands {
            match self.execute_one(cmd)? {
                ExecutionResult::Continue { output: out, .. } => {
                    if !out.is_empty() {
                        if !output.is_empty() {
                            output.push('\n');
                        }
                        output.push_str(&out);
                    }
                }
                ExecutionResult::Final { answer, sub_calls } => {
                    return Ok(ExecutionResult::Final {
                        answer,
                        sub_calls: self.sub_calls - initial_sub_calls + sub_calls,
                    });
                }
            }
        }

        Ok(ExecutionResult::Continue {
            output,
            sub_calls: self.sub_calls - initial_sub_calls,
        })
    }

    /// Execute a single command
    fn execute_one(&mut self, cmd: &Command) -> Result<ExecutionResult, CommandError> {
        match cmd {
            Command::Slice { start, end, store } => {
                let len = self.context.len();
                let start_idx = Self::resolve_index(*start, len).unwrap_or(0);
                let end_idx = end
                    .map(|e| Self::resolve_index(e, len).unwrap_or(len))
                    .unwrap_or(len);

                // Ensure start <= end
                let (start_idx, end_idx) = if start_idx > end_idx {
                    (end_idx, start_idx)
                } else {
                    (start_idx, end_idx)
                };

                let result = self.context[start_idx..end_idx].to_string();
                self.store_result(store, result);
                Ok(ExecutionResult::Continue {
                    output: String::new(),
                    sub_calls: 0,
                })
            }

            Command::Lines { start, end, store } => {
                let lines: Vec<&str> = self.context.lines().collect();
                let len = lines.len();
                let start_idx = start
                    .map(|s| Self::resolve_index(s, len).unwrap_or(0))
                    .unwrap_or(0);
                let end_idx = end
                    .map(|e| Self::resolve_index(e, len).unwrap_or(len))
                    .unwrap_or(len);

                // Ensure start <= end
                let (start_idx, end_idx) = if start_idx > end_idx {
                    (end_idx, start_idx)
                } else {
                    (start_idx, end_idx)
                };

                let result = lines[start_idx..end_idx].join("\n");
                self.store_result(store, result);
                Ok(ExecutionResult::Continue {
                    output: String::new(),
                    sub_calls: 0,
                })
            }

            Command::Regex { pattern, on, store } => {
                let source = self.resolve_source(on)?.to_string();
                // Case-insensitive regex by default - find matching LINES
                let re = regex::RegexBuilder::new(pattern)
                    .case_insensitive(true)
                    .build()?;

                let mut raw_lines: Vec<&str> = Vec::new();
                let mut formatted_lines: Vec<String> = Vec::new();
                for (line_num, line) in source.lines().enumerate() {
                    if re.is_match(line) {
                        raw_lines.push(line);
                        formatted_lines.push(format!("L{}: {}", line_num + 1, line));
                    }
                }

                let count = raw_lines.len();
                // Store raw lines without prefixes (for use by reduce/other ops)
                let result = raw_lines.join("\n");

                // Always show preview with line numbers (first 5 matches, truncated if needed)
                let preview = if count > 0 {
                    let preview_lines: Vec<&str> =
                        formatted_lines.iter().take(5).map(|s| s.as_str()).collect();
                    let preview_text = preview_lines.join("\n");
                    let truncated = if count > 5 {
                        format!("\n... and {} more", count - 5)
                    } else {
                        String::new()
                    };
                    format!(":\n{}{}", preview_text, truncated)
                } else {
                    String::new()
                };

                let output = format!("Found {} lines matching '{}'{}", count, pattern, preview);
                self.store_result(store, result);
                Ok(ExecutionResult::Continue {
                    output,
                    sub_calls: 0,
                })
            }

            Command::Find { text, on, store } => {
                let source = self.resolve_source(on)?;
                // Fuzzy search: split into words and find lines containing ANY word
                // This makes searches like "Prince Andrei secret vault" find lines
                // containing "vault" or "secret" even if the exact phrase doesn't match
                let words: Vec<String> = text
                    .split_whitespace()
                    .filter(|w| w.len() >= 3) // Only words with 3+ chars
                    .map(|w| w.to_lowercase())
                    .collect();

                let mut matching_lines: Vec<(usize, String, usize)> = Vec::new(); // (line_num, line, word_count)

                for (line_num, line) in source.lines().enumerate() {
                    let line_lower = line.to_lowercase();
                    let word_matches = words.iter().filter(|w| line_lower.contains(*w)).count();
                    if word_matches > 0 {
                        matching_lines.push((line_num + 1, line.to_string(), word_matches));
                    }
                }

                // Sort by number of matching words (descending) to show most relevant first
                matching_lines.sort_by(|a, b| b.2.cmp(&a.2));

                let count = matching_lines.len();
                // Store raw lines without prefixes (for use by reduce/other ops)
                let raw_lines: Vec<&str> = matching_lines
                    .iter()
                    .map(|(_, line, _)| line.as_str())
                    .collect();
                let result = raw_lines.join("\n");

                // Format with line numbers for preview only
                let formatted: Vec<String> = matching_lines
                    .iter()
                    .map(|(num, line, _)| format!("L{}: {}", num, line))
                    .collect();

                // Always show preview (first 5 matches, truncated if needed)
                let preview = if count > 0 {
                    let preview_lines: Vec<&str> =
                        formatted.iter().take(5).map(|s| s.as_str()).collect();
                    let preview_text = preview_lines.join("\n");
                    let truncated = if count > 5 {
                        format!("\n... and {} more", count - 5)
                    } else {
                        String::new()
                    };
                    format!(":\n{}{}", preview_text, truncated)
                } else {
                    String::new()
                };

                let search_desc = if words.len() > 1 {
                    format!("any of [{}]", words.join(", "))
                } else {
                    format!("'{}'", text)
                };
                let output = format!("Found {} lines matching {}{}", count, search_desc, preview);
                self.store_result(store, result);
                Ok(ExecutionResult::Continue {
                    output,
                    sub_calls: 0,
                })
            }

            Command::Count { what, on, store } => {
                let source = self.resolve_source(on)?;
                let count = match what {
                    CountTarget::Lines => source.lines().count(),
                    CountTarget::Chars => source.chars().count(),
                    CountTarget::Words => source.split_whitespace().count(),
                    CountTarget::Matches => self.last_result.lines().count(),
                };
                self.store_result(store, count.to_string());
                Ok(ExecutionResult::Continue {
                    output: format!("{}", count),
                    sub_calls: 0,
                })
            }

            Command::Split {
                delimiter,
                on,
                store,
            } => {
                let source = self.resolve_source(on)?.to_string();
                let parts: Vec<&str> = source.split(delimiter.as_str()).collect();
                let part_count = parts.len();
                let result = parts.join("\n");
                self.store_result(store, result);
                Ok(ExecutionResult::Continue {
                    output: format!("Split into {} parts", part_count),
                    sub_calls: 0,
                })
            }

            Command::Len { on, store } => {
                let source = self.resolve_source(on)?;
                let len = source.len();
                self.store_result(store, len.to_string());
                Ok(ExecutionResult::Continue {
                    output: format!("{}", len),
                    sub_calls: 0,
                })
            }

            Command::Set { name, value } => {
                // Expand variables in value
                let expanded = self.expand_vars(value);
                self.variables.insert(name.clone(), expanded);
                Ok(ExecutionResult::Continue {
                    output: String::new(),
                    sub_calls: 0,
                })
            }

            Command::Get { name } => {
                let value = self
                    .variables
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| format!("<undefined: {}>", name));
                Ok(ExecutionResult::Continue {
                    output: value,
                    sub_calls: 0,
                })
            }

            Command::Print { value, var } => {
                let output = if let Some(v) = value {
                    self.expand_vars(v)
                } else if let Some(name) = var {
                    self.variables
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| format!("<undefined: {}>", name))
                } else {
                    self.last_result.clone()
                };
                Ok(ExecutionResult::Continue {
                    output,
                    sub_calls: 0,
                })
            }

            Command::LlmQuery { prompt, store } => {
                if self.sub_calls >= self.max_sub_calls {
                    return Err(CommandError::MaxSubCalls(self.max_sub_calls));
                }

                let callback = self.llm_callback.as_ref().ok_or_else(|| {
                    CommandError::LlmError("No LLM callback configured".to_string())
                })?;

                let expanded_prompt = self.expand_vars(prompt);
                let result = callback(&expanded_prompt).map_err(CommandError::LlmError)?;

                self.sub_calls += 1;
                self.store_result(store, result.clone());

                Ok(ExecutionResult::Continue {
                    output: result,
                    sub_calls: 1,
                })
            }

            Command::LlmDelegate {
                task,
                on,
                max_iterations,
                levels,
                store,
            } => {
                // Check recursion depth limit
                if self.recursion_depth >= self.max_recursion_depth {
                    return Err(CommandError::MaxRecursionDepth(self.max_recursion_depth));
                }

                // Get the delegate callback
                let callback = self
                    .llm_delegate_callback
                    .as_ref()
                    .ok_or(CommandError::LlmDelegateNotConfigured)?;

                // Resolve the context for the nested RLM
                let nested_context = match on {
                    Some(var_name) => self
                        .variables
                        .get(var_name)
                        .ok_or_else(|| CommandError::VariableNotFound(var_name.clone()))?
                        .clone(),
                    None => self.context.clone(),
                };

                // Determine capability levels (use provided or defaults)
                let capability_levels =
                    levels.clone().unwrap_or_else(|| self.nested_levels.clone());

                // Build delegate parameters
                let params = LlmDelegateParams {
                    task: self.expand_vars(task),
                    context: nested_context,
                    max_iterations: max_iterations.unwrap_or(5),
                    levels: capability_levels,
                    current_depth: self.recursion_depth + 1,
                };

                // Save depth before params is moved
                let delegate_depth = params.current_depth;

                tracing::info!(
                    task = %params.task,
                    context_len = params.context.len(),
                    depth = params.current_depth,
                    levels = ?params.levels,
                    "Starting nested RLM delegation"
                );

                // Execute the nested RLM
                let result = callback(params)
                    .map_err(|e| CommandError::LlmError(format!("LLM delegation failed: {}", e)))?;

                tracing::info!(
                    answer_len = result.answer.len(),
                    iterations = result.iterations,
                    success = result.success,
                    nested_steps = result.nested_history.len(),
                    "Nested RLM delegation complete"
                );

                // Store nested history for progress visibility
                self.last_nested_history = result.nested_history;
                self.last_delegate_depth = delegate_depth;

                // Store the result
                self.store_result(store, result.answer.clone());
                self.sub_calls += 1;

                let output = if result.success {
                    format!(
                        "[Nested RLM completed in {} iterations]\n{}",
                        result.iterations, result.answer
                    )
                } else {
                    format!(
                        "[Nested RLM failed after {} iterations]\n{}",
                        result.iterations, result.answer
                    )
                };

                Ok(ExecutionResult::Continue {
                    output,
                    sub_calls: 1,
                })
            }

            Command::LlmReduce {
                directive,
                chunk_size,
                overlap,
                on,
                store,
            } => {
                // Use llm_query callback for simple LLM calls (more reliable than full RLM)
                let llm_callback = self
                    .llm_callback
                    .as_ref()
                    .ok_or(CommandError::LlmDelegateNotConfigured)?;

                // Resolve the source context
                let source_context = match on {
                    Some(var_name) => self
                        .variables
                        .get(var_name)
                        .ok_or_else(|| CommandError::VariableNotFound(var_name.clone()))?
                        .clone(),
                    None => self.context.clone(),
                };

                // Chunking parameters
                let chunk_sz = chunk_size.unwrap_or(10000);
                let overlap_sz = overlap.unwrap_or(500);
                let step_sz = chunk_sz.saturating_sub(overlap_sz).max(1000); // Minimum step of 1000

                // Split into overlapping chunks, trying to break at paragraph boundaries
                let chunks = split_into_chunks(&source_context, chunk_sz, step_sz);

                let num_chunks = chunks.len();
                tracing::info!(
                    directive = %directive,
                    chunk_size = chunk_sz,
                    overlap = overlap_sz,
                    step_size = step_sz,
                    num_chunks = num_chunks,
                    total_len = source_context.len(),
                    "Starting LLM reduce over chunks (simple LLM mode)"
                );

                // Process chunks sequentially, accumulating results
                // Use simple LLM calls instead of full RLM for reliability
                let mut accumulated_result = String::new();

                for (i, chunk) in chunks.iter().enumerate() {
                    let chunk_num = i + 1;

                    // Build a prompt that includes the chunk directly
                    // CRITICAL: Ask for COMPLETE accumulated findings, not just delta
                    let prompt = if accumulated_result.is_empty() {
                        format!(
                            r#"You are analyzing chunk {chunk_num} of {num_chunks} from a larger document.

TASK: {directive}

DOCUMENT CHUNK:
{chunk}

---
INSTRUCTIONS:
1. Extract information relevant to the task above
2. Be concise and well-structured (use bullet points or sections)
3. If no relevant information, respond: "No relevant information found."

Provide your findings:"#
                        )
                    } else {
                        format!(
                            r#"You are analyzing chunk {chunk_num} of {num_chunks} from a larger document.

TASK: {directive}

ACCUMULATED FINDINGS SO FAR:
{accumulated_result}

NEW DOCUMENT CHUNK:
{chunk}

---
INSTRUCTIONS:
1. Review the accumulated findings above
2. Check this new chunk for any ADDITIONAL relevant information
3. Output the COMPLETE UPDATED FINDINGS (previous + new combined)
4. If no new information, output the previous findings unchanged
5. Keep the same format and structure

IMPORTANT: Output ALL accumulated findings, not just new additions.

Complete updated findings:"#
                        )
                    };

                    tracing::debug!(
                        chunk = chunk_num,
                        of = num_chunks,
                        chunk_len = chunk.len(),
                        prompt_len = prompt.len(),
                        "Processing chunk with simple LLM call"
                    );

                    // Make simple LLM call
                    let result = llm_callback(&prompt)
                        .map_err(|e| CommandError::LlmError(format!("LLM reduce chunk {} failed: {}", chunk_num, e)))?;

                    accumulated_result = result;
                }

                tracing::info!(
                    num_chunks = num_chunks,
                    result_len = accumulated_result.len(),
                    "LLM reduce complete (simple mode)"
                );

                // Store the result
                self.store_result(store, accumulated_result.clone());
                self.sub_calls += num_chunks;

                let output = format!(
                    "[LLM reduce complete: {} chunks processed]\n{}",
                    num_chunks, accumulated_result
                );

                Ok(ExecutionResult::Continue {
                    output,
                    sub_calls: num_chunks,
                })
            }

            Command::Final { answer } => {
                let expanded = self.expand_vars(answer);
                Ok(ExecutionResult::Final {
                    answer: expanded,
                    sub_calls: 0,
                })
            }

            Command::FinalVar { name } => {
                let answer = self
                    .variables
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| format!("<undefined: {}>", name));
                Ok(ExecutionResult::Final {
                    answer,
                    sub_calls: 0,
                })
            }

            Command::Wasm {
                module: _, // Legacy module syntax deprecated
                tool,
                args,
                function,
                on,
                store,
            } => {
                let executor = self.wasm_executor.as_ref().ok_or_else(|| {
                    CommandError::WasmError(crate::wasm::WasmError::CompileError(
                        "WASM not available".to_string(),
                    ))
                })?;

                // Resolve the source input
                let source = self.resolve_source(on)?.to_string();

                // Try tool library first if a tool name is specified
                if let Some(tool_name) = tool {
                    // Check if tool library is available
                    let tool_lib = self.wasm_tool_library.as_ref().ok_or_else(|| {
                        CommandError::InvalidCommand(
                            "WASM tool library not available. Set RLM_PRECOMPILE_TOOLS=1"
                                .to_string(),
                        )
                    })?;

                    let wasm_bytes = tool_lib
                        .get(tool_name)
                        .ok_or_else(|| CommandError::WasmModuleNotFound(tool_name.clone()))?;

                    // Prepend args to source if provided
                    let input = if let Some(args_str) = args {
                        format!("{}|{}", args_str, source)
                    } else {
                        source
                    };

                    // Tools use run_analyze function
                    let func_name = if function == "analyze" {
                        "run_analyze"
                    } else {
                        function
                    };
                    let result = executor.execute(&wasm_bytes, func_name, &input)?;

                    self.store_result(store, result.clone());
                    return Ok(ExecutionResult::Continue {
                        output: format!("wasm_tool {}: {}", tool_name, result),
                        sub_calls: 0,
                    });
                }

                // Legacy module syntax no longer supported
                Err(CommandError::InvalidCommand(
                    "WASM command requires 'tool' parameter. Legacy module syntax deprecated."
                        .to_string(),
                ))
            }

            Command::WasmWat {
                wat,
                function,
                on,
                store,
            } => {
                let executor = self.wasm_executor.as_ref().ok_or_else(|| {
                    CommandError::WasmError(crate::wasm::WasmError::CompileError(
                        "WASM not available".to_string(),
                    ))
                })?;

                let wasm_bytes = executor.compile_wat(wat)?;
                let source = self.resolve_source(on)?.to_string();
                let result = executor.execute(&wasm_bytes, function, &source)?;

                self.store_result(store, result.clone());
                Ok(ExecutionResult::Continue {
                    output: format!("WASM (inline).{}: {}", function, result),
                    sub_calls: 0,
                })
            }

            Command::RustWasm { code, on, store } => {
                // Reset timing (will be set if we actually compile/run)
                self.last_compile_time_ms = 0;
                self.last_wasm_run_time_ms = 0;

                // Check if Rust compiler is available
                let compiler = self.rust_compiler.as_ref().ok_or_else(|| {
                    CommandError::RustCompilerUnavailable(
                        "rustc not found. Install Rust: https://rustup.rs".to_string(),
                    )
                })?;

                // Check if WASM executor is available
                let executor = self.wasm_executor.as_ref().ok_or_else(|| {
                    CommandError::WasmError(crate::wasm::WasmError::CompileError(
                        "WASM runtime not available".to_string(),
                    ))
                })?;

                // Check cache first
                let wasm_bytes = if let Some(cached) = self.wasm_cache.get(code) {
                    tracing::debug!("Cache hit for rust_wasm");
                    cached
                } else {
                    // Compile Rust to WASM (with timing)
                    tracing::debug!("Compiling Rust code ({} bytes)", code.len());
                    let compile_start = Instant::now();
                    let compiled = compiler
                        .compile(code)
                        .map_err(|e| CommandError::RustCompileError(e.to_string()))?;
                    self.last_compile_time_ms = compile_start.elapsed().as_millis() as u64;
                    tracing::info!("Rust compilation took {}ms", self.last_compile_time_ms);

                    // Cache the result
                    self.wasm_cache.put(code, compiled.clone());
                    compiled
                };

                // Execute the compiled WASM (with timing)
                // Sanitize to ASCII for safe byte-level processing
                let source = to_ascii(self.resolve_source(on)?);
                let exec_start = std::time::Instant::now();
                let result = executor
                    .execute(&wasm_bytes, "run_analyze", &source)
                    .map_err(|e| {
                        CommandError::WasmError(crate::wasm::WasmError::ExecutionError(
                            e.to_string(),
                        ))
                    })?;
                self.last_wasm_run_time_ms = exec_start.elapsed().as_millis() as u64;

                self.store_result(store, result.clone());

                // Return concise output
                let preview = if result.len() > 100 {
                    format!("{}...", truncate_to_char_boundary(&result, 97))
                } else {
                    result.clone()
                };
                Ok(ExecutionResult::Continue {
                    output: format!("rust_wasm result: {}", preview),
                    sub_calls: 0,
                })
            }

            Command::WasmTemplate {
                template,
                prebuilt,
                hook,
                on,
                store,
            } => {
                // Reset timing (will be set if we actually compile/run)
                self.last_compile_time_ms = 0;
                self.last_wasm_run_time_ms = 0;

                // Parse template type
                let template_type = TemplateType::parse(template).ok_or_else(|| {
                    CommandError::InvalidCommand(format!(
                        "Unknown template '{}'. Available: group_count, filter_lines, map_lines, numeric_stats, count_matching",
                        template
                    ))
                })?;

                // Get hook code: prefer prebuilt, fall back to custom hook
                let hook_code = if let Some(prebuilt_name) = prebuilt {
                    // Try to get prebuilt hook
                    let (code, desc) = PrebuiltHooks::get(template_type, prebuilt_name)
                        .ok_or_else(|| {
                            let available: Vec<_> = PrebuiltHooks::list(template_type)
                                .iter()
                                .map(|(name, _)| *name)
                                .collect();
                            CommandError::InvalidCommand(format!(
                                "Unknown prebuilt hook '{}' for template '{}'. Available: {}",
                                prebuilt_name,
                                template,
                                available.join(", ")
                            ))
                        })?;
                    tracing::info!("Using prebuilt hook '{}': {}", prebuilt_name, desc);
                    code.to_string()
                } else if let Some(custom_hook) = hook {
                    // Use custom hook code
                    tracing::debug!("Using custom hook code");
                    custom_hook.clone()
                } else {
                    return Err(CommandError::InvalidCommand(
                        "wasm_template requires either 'prebuilt' or 'hook' parameter".to_string(),
                    ));
                };

                // Check if Rust compiler is available
                let compiler = self.rust_compiler.as_ref().ok_or_else(|| {
                    CommandError::RustCompilerUnavailable(
                        "rustc not found. Install Rust: https://rustup.rs".to_string(),
                    )
                })?;

                // Check if WASM executor is available
                let executor = self.wasm_executor.as_ref().ok_or_else(|| {
                    CommandError::WasmError(crate::wasm::WasmError::CompileError(
                        "WASM runtime not available".to_string(),
                    ))
                })?;

                // Generate complete module source from template + hook
                let full_source = TemplateFramework::generate_module(template_type, &hook_code);
                tracing::debug!(
                    "Generated template source ({} bytes) for {}",
                    full_source.len(),
                    template
                );

                // Check cache using the full generated source as key
                let wasm_bytes = if let Some(cached) = self.wasm_cache.get(&full_source) {
                    tracing::debug!("Cache hit for wasm_template");
                    cached
                } else {
                    // Compile Rust to WASM (with timing)
                    tracing::debug!("Compiling template code ({} bytes)", full_source.len());
                    let compile_start = Instant::now();
                    let compiled = compiler
                        .compile(&full_source)
                        .map_err(|e| CommandError::RustCompileError(e.to_string()))?;
                    self.last_compile_time_ms = compile_start.elapsed().as_millis() as u64;
                    tracing::info!(
                        "Template compilation took {}ms (template: {})",
                        self.last_compile_time_ms,
                        template
                    );

                    // Cache the result
                    self.wasm_cache.put(&full_source, compiled.clone());
                    compiled
                };

                // Execute the compiled WASM (with timing)
                // Sanitize to ASCII for safe byte-level processing
                let source = to_ascii(self.resolve_source(on)?);
                let exec_start = std::time::Instant::now();
                let result = executor
                    .execute(&wasm_bytes, "run_analyze", &source)
                    .map_err(|e| {
                        CommandError::WasmError(crate::wasm::WasmError::ExecutionError(
                            e.to_string(),
                        ))
                    })?;
                self.last_wasm_run_time_ms = exec_start.elapsed().as_millis() as u64;

                self.store_result(store, result.clone());

                // Return concise output
                let preview = if result.len() > 100 {
                    format!("{}...", truncate_to_char_boundary(&result, 97))
                } else {
                    result.clone()
                };
                let hook_desc = prebuilt.as_ref().map(|s| s.as_str()).unwrap_or("custom");
                Ok(ExecutionResult::Continue {
                    output: format!("wasm_template({}/{}): {}", template, hook_desc, preview),
                    sub_calls: 0,
                })
            }

            Command::RustWasmIntent { intent, on, store } => {
                // Reset timing
                self.last_compile_time_ms = 0;
                self.last_wasm_run_time_ms = 0;

                // Check if code generator is configured
                let generator = self
                    .code_generator
                    .as_ref()
                    .ok_or(CommandError::CodeGenNotConfigured)?;

                // Check if Rust compiler is available
                let compiler = self.rust_compiler.as_ref().ok_or_else(|| {
                    CommandError::RustCompilerUnavailable(
                        "rustc not found. Install Rust: https://rustup.rs".to_string(),
                    )
                })?;

                // Check if WASM executor is available
                let executor = self.wasm_executor.as_ref().ok_or_else(|| {
                    CommandError::WasmError(crate::wasm::WasmError::CompileError(
                        "WASM runtime not available".to_string(),
                    ))
                })?;

                // Generate code using the coding LLM
                tracing::info!("Generating code for intent: {}", intent);
                let codegen_start = Instant::now();

                // We need to use block_in_place to call async code from sync context
                let handle = tokio::runtime::Handle::try_current()
                    .map_err(|e| CommandError::CodeGenError(format!("No tokio runtime: {}", e)))?;

                let generator_ref = generator;
                let intent_clone = intent.clone();
                let code = tokio::task::block_in_place(|| {
                    handle.block_on(async { generator_ref.generate(&intent_clone).await })
                })
                .map_err(|e| CommandError::CodeGenError(e.to_string()))?;

                let codegen_ms = codegen_start.elapsed().as_millis() as u64;
                tracing::info!(
                    "Code generation took {}ms, generated {} bytes",
                    codegen_ms,
                    code.len()
                );
                tracing::debug!("Generated code:\n{}", code);

                // Check cache first (using generated code as key)
                let wasm_bytes = if let Some(cached) = self.wasm_cache.get(&code) {
                    tracing::debug!("Cache hit for generated code");
                    cached
                } else {
                    // Compile the generated Rust to WASM
                    tracing::debug!("Compiling generated code ({} bytes)", code.len());
                    let compile_start = Instant::now();
                    let compiled = compiler.compile(&code).map_err(|e| {
                        tracing::error!("Compilation failed for code:\n{}", code);
                        CommandError::RustCompileError(e.to_string())
                    })?;
                    self.last_compile_time_ms = compile_start.elapsed().as_millis() as u64;
                    tracing::info!("Rust compilation took {}ms", self.last_compile_time_ms);

                    // Cache the result
                    self.wasm_cache.put(&code, compiled.clone());
                    compiled
                };

                // Execute the compiled WASM
                // Sanitize to ASCII for safe byte-level processing
                let source = to_ascii(self.resolve_source(on)?);
                let exec_start = Instant::now();
                let result = executor
                    .execute(&wasm_bytes, "run_analyze", &source)
                    .map_err(|e| {
                        tracing::error!("WASM execution failed for code:\n{}", code);
                        CommandError::WasmError(crate::wasm::WasmError::ExecutionError(
                            e.to_string(),
                        ))
                    })?;
                self.last_wasm_run_time_ms = exec_start.elapsed().as_millis() as u64;

                self.store_result(store, result.clone());

                // Return concise output
                let preview = if result.len() > 100 {
                    format!("{}...", truncate_to_char_boundary(&result, 97))
                } else {
                    result.clone()
                };
                Ok(ExecutionResult::Continue {
                    output: format!(
                        "rust_wasm_intent (codegen: {}ms, compile: {}ms, run: {}ms): {}",
                        codegen_ms, self.last_compile_time_ms, self.last_wasm_run_time_ms, preview
                    ),
                    sub_calls: 0,
                })
            }

            Command::RustWasmReduceIntent {
                intent,
                on,
                store,
                chunk_size,
            } => {
                // Reset timing
                self.last_compile_time_ms = 0;
                self.last_wasm_run_time_ms = 0;

                // Check if code generator is configured
                let generator = self
                    .code_generator
                    .as_ref()
                    .ok_or(CommandError::CodeGenNotConfigured)?;

                // Check if Rust compiler is available
                let compiler = self.rust_compiler.as_ref().ok_or_else(|| {
                    CommandError::RustCompilerUnavailable(
                        "rustc not found. Install Rust: https://rustup.rs".to_string(),
                    )
                })?;

                // Check if WASM executor is available
                let executor = self.wasm_executor.as_ref().ok_or_else(|| {
                    CommandError::WasmError(crate::wasm::WasmError::CompileError(
                        "WASM runtime not available".to_string(),
                    ))
                })?;

                // Generate reduce code using the coding LLM
                tracing::info!("Generating reduce code for intent: {}", intent);
                let codegen_start = Instant::now();

                // Use block_in_place to call async code from sync context
                let handle = tokio::runtime::Handle::try_current()
                    .map_err(|e| CommandError::CodeGenError(format!("No tokio runtime: {}", e)))?;

                let generator_ref = generator;
                let intent_clone = intent.clone();
                let code = tokio::task::block_in_place(|| {
                    handle.block_on(async { generator_ref.generate_reduce(&intent_clone).await })
                })
                .map_err(|e| CommandError::CodeGenError(e.to_string()))?;

                let codegen_ms = codegen_start.elapsed().as_millis() as u64;
                tracing::info!(
                    "Reduce code generation took {}ms, generated {} bytes",
                    codegen_ms,
                    code.len()
                );
                tracing::debug!("Generated reduce code:\n{}", code);

                // Compile the reduce code to WASM
                let wasm_bytes = if let Some(cached) = self.wasm_cache.get(&code) {
                    tracing::debug!("Cache hit for generated reduce code");
                    cached
                } else {
                    tracing::debug!("Compiling reduce code ({} bytes)", code.len());
                    let compile_start = Instant::now();
                    let compiled = compiler.compile_reduce(&code).map_err(|e| {
                        tracing::error!("Reduce compilation failed for code:\n{}", code);
                        CommandError::RustCompileError(e.to_string())
                    })?;
                    self.last_compile_time_ms = compile_start.elapsed().as_millis() as u64;
                    tracing::info!("Reduce compilation took {}ms", self.last_compile_time_ms);

                    // Cache the result
                    self.wasm_cache.put(&code, compiled.clone());
                    compiled
                };

                // Create reduce instance
                let mut reduce_instance =
                    executor.create_reduce_instance(&wasm_bytes).map_err(|e| {
                        tracing::error!("Failed to create reduce instance");
                        CommandError::WasmError(e)
                    })?;

                // Get source data (sanitize to ASCII)
                let source = to_ascii(self.resolve_source(on)?);
                let source_len = source.len();
                let chunk_bytes = chunk_size.unwrap_or(4 * 1024); // Default 4KB chunks (smaller for WASM safety)

                // Initialize the reduce state
                let exec_start = Instant::now();
                reduce_instance.init().map_err(|e| {
                    tracing::error!("Reduce init failed");
                    CommandError::WasmError(e)
                })?;

                // Process data in chunks
                let mut bytes_processed = 0usize;
                let mut chunks_processed = 0usize;

                // Split on line boundaries within chunk size
                let mut start = 0;
                while start < source.len() {
                    // Find a good chunk boundary (at a newline)
                    let mut end = std::cmp::min(start + chunk_bytes, source.len());

                    // If not at the end, try to find a newline to split at
                    if end < source.len()
                        && let Some(newline_pos) = source[start..end].rfind('\n')
                    {
                        end = start + newline_pos + 1; // Include the newline
                    }

                    let chunk = &source[start..end];
                    reduce_instance.process_chunk(chunk).map_err(|e| {
                        tracing::error!(
                            "Reduce process_chunk failed at chunk {}",
                            chunks_processed
                        );
                        // Save crash info to file for debugging
                        if let Ok(mut f) = std::fs::File::create("/tmp/rlm-wasm-crash.rs") {
                            use std::io::Write;
                            let _ = writeln!(f, "// WASM crash at chunk {}", chunks_processed);
                            let _ = writeln!(f, "// Error: {}", e);
                            let _ = writeln!(
                                f,
                                "// Chunk preview: {:?}",
                                &chunk[..chunk.len().min(200)]
                            );
                            let _ = writeln!(f, "\n{}", code);
                        }
                        CommandError::WasmError(e)
                    })?;

                    bytes_processed += chunk.len();
                    chunks_processed += 1;
                    start = end;
                }

                // Finalize and get result
                let result = reduce_instance.finalize().map_err(|e| {
                    tracing::error!("Reduce finalize failed");
                    CommandError::WasmError(e)
                })?;

                self.last_wasm_run_time_ms = exec_start.elapsed().as_millis() as u64;
                tracing::info!(
                    "Reduce execution: {} chunks, {} bytes in {}ms",
                    chunks_processed,
                    bytes_processed,
                    self.last_wasm_run_time_ms
                );

                self.store_result(store, result.clone());

                // Return concise output
                let preview = if result.len() > 100 {
                    format!("{}...", truncate_to_char_boundary(&result, 97))
                } else {
                    result.clone()
                };
                Ok(ExecutionResult::Continue {
                    output: format!(
                        "rust_wasm_reduce (codegen: {}ms, compile: {}ms, reduce {} chunks/{}KB in {}ms): {}",
                        codegen_ms,
                        self.last_compile_time_ms,
                        chunks_processed,
                        source_len / 1024,
                        self.last_wasm_run_time_ms,
                        preview
                    ),
                    sub_calls: 0,
                })
            }

            Command::RustWasmMapReduce {
                intent,
                combiner,
                on,
                store,
                sort_desc,
                limit,
            } => {
                // Reset timing
                self.last_compile_time_ms = 0;
                self.last_wasm_run_time_ms = 0;

                // Check if code generator is configured
                let generator = self
                    .code_generator
                    .as_ref()
                    .ok_or(CommandError::CodeGenNotConfigured)?;

                // Check if Rust compiler is available
                let compiler = self.rust_compiler.as_ref().ok_or_else(|| {
                    CommandError::RustCompilerUnavailable(
                        "rustc not found. Install Rust: https://rustup.rs".to_string(),
                    )
                })?;

                // Check if WASM executor is available
                let executor = self.wasm_executor.as_ref().ok_or_else(|| {
                    CommandError::WasmError(crate::wasm::WasmError::CompileError(
                        "WASM runtime not available".to_string(),
                    ))
                })?;

                // Generate map code using the coding LLM
                tracing::info!("Generating stateless map code for intent: {}", intent);
                let codegen_start = Instant::now();

                // Use block_in_place to call async code from sync context
                let handle = tokio::runtime::Handle::try_current()
                    .map_err(|e| CommandError::CodeGenError(format!("No tokio runtime: {}", e)))?;

                let generator_ref = generator;
                let intent_clone = intent.clone();
                let code = tokio::task::block_in_place(|| {
                    handle.block_on(async { generator_ref.generate_map(&intent_clone).await })
                })
                .map_err(|e| CommandError::CodeGenError(e.to_string()))?;

                let codegen_ms = codegen_start.elapsed().as_millis() as u64;
                tracing::info!(
                    "Map code generation took {}ms, generated {} bytes",
                    codegen_ms,
                    code.len()
                );
                tracing::debug!("Generated map code:\n{}", code);

                // Compile the map code to WASM
                let wasm_bytes = if let Some(cached) = self.wasm_cache.get(&code) {
                    tracing::debug!("Cache hit for generated map code");
                    cached
                } else {
                    tracing::debug!("Compiling map code ({} bytes)", code.len());
                    let compile_start = Instant::now();
                    let compiled = compiler.compile_map(&code).map_err(|e| {
                        tracing::error!("Map compilation failed for code:\n{}", code);
                        CommandError::RustCompileError(e.to_string())
                    })?;
                    self.last_compile_time_ms = compile_start.elapsed().as_millis() as u64;
                    tracing::info!("Map compilation took {}ms", self.last_compile_time_ms);

                    // Cache the result
                    self.wasm_cache.put(&code, compiled.clone());
                    compiled
                };

                // Create map instance
                let mut map_instance = executor.create_map_instance(&wasm_bytes).map_err(|e| {
                    tracing::error!("Failed to create map instance");
                    CommandError::WasmError(e)
                })?;

                // Get source data (sanitize to ASCII)
                let source = to_ascii(self.resolve_source(on)?);

                // MAP PHASE: Process each line, collect key-value pairs
                let exec_start = Instant::now();
                let mut all_pairs: Vec<(String, String)> = Vec::new();
                let mut lines_processed = 0usize;

                for line in source.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }

                    let pairs = map_instance.map_line(line).map_err(|e| {
                        tracing::error!("Map failed at line {}", lines_processed);
                        // Save crash info
                        if let Ok(mut f) = std::fs::File::create("/tmp/rlm-wasm-map-crash.rs") {
                            use std::io::Write;
                            let _ = writeln!(f, "// WASM map crash at line {}", lines_processed);
                            let _ = writeln!(f, "// Error: {}", e);
                            let _ = writeln!(f, "// Line: {:?}", &line[..line.len().min(200)]);
                            let _ = writeln!(f, "\n{}", code);
                        }
                        CommandError::WasmError(e)
                    })?;

                    all_pairs.extend(pairs);
                    lines_processed += 1;
                }

                let map_ms = exec_start.elapsed().as_millis() as u64;
                tracing::info!(
                    "Map phase: {} lines -> {} pairs in {}ms",
                    lines_processed,
                    all_pairs.len(),
                    map_ms
                );

                // SHUFFLE PHASE: Group by key using native HashMap
                let shuffle_start = Instant::now();
                let mut grouped: HashMap<String, Vec<String>> = HashMap::new();
                for (key, value) in all_pairs {
                    grouped.entry(key).or_default().push(value);
                }
                let shuffle_ms = shuffle_start.elapsed().as_millis() as u64;
                tracing::info!(
                    "Shuffle phase: {} unique keys in {}ms",
                    grouped.len(),
                    shuffle_ms
                );

                // REDUCE PHASE: Combine values for each key
                let reduce_start = Instant::now();
                let mut results: Vec<(String, String)> = Vec::new();

                for (key, values) in grouped {
                    let combined = match combiner.to_lowercase().as_str() {
                        "count" => values.len().to_string(),
                        "sum" => {
                            let sum: i64 =
                                values.iter().filter_map(|v| v.parse::<i64>().ok()).sum();
                            sum.to_string()
                        }
                        "max" => values
                            .iter()
                            .filter_map(|v| v.parse::<i64>().ok())
                            .max()
                            .map(|v| v.to_string())
                            .unwrap_or_default(),
                        "min" => values
                            .iter()
                            .filter_map(|v| v.parse::<i64>().ok())
                            .min()
                            .map(|v| v.to_string())
                            .unwrap_or_default(),
                        "first" => values.first().cloned().unwrap_or_default(),
                        "last" => values.last().cloned().unwrap_or_default(),
                        "list" => values.join(", "),
                        _ => values.len().to_string(), // Default to count
                    };
                    results.push((key, combined));
                }

                // Sort by value (descending) for count/sum combiners
                let should_sort = sort_desc
                    .unwrap_or(matches!(combiner.to_lowercase().as_str(), "count" | "sum"));
                if should_sort {
                    results.sort_by(|a, b| {
                        let a_num: i64 = a.1.parse().unwrap_or(0);
                        let b_num: i64 = b.1.parse().unwrap_or(0);
                        b_num.cmp(&a_num)
                    });
                }

                // Apply limit if specified
                if let Some(n) = limit {
                    results.truncate(*n);
                }

                let reduce_ms = reduce_start.elapsed().as_millis() as u64;
                self.last_wasm_run_time_ms = map_ms + shuffle_ms + reduce_ms;

                // Format result
                let result: String = results
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v))
                    .collect::<Vec<_>>()
                    .join("\n");

                self.store_result(store, result.clone());

                // Return concise output
                let preview = if result.len() > 100 {
                    format!("{}...", truncate_to_char_boundary(&result, 97))
                } else {
                    result.clone()
                };
                Ok(ExecutionResult::Continue {
                    output: format!(
                        "rust_wasm_mapreduce (codegen: {}ms, compile: {}ms, map: {}ms, shuffle: {}ms, reduce: {}ms, {} lines -> {} keys): {}",
                        codegen_ms,
                        self.last_compile_time_ms,
                        map_ms,
                        shuffle_ms,
                        reduce_ms,
                        lines_processed,
                        results.len(),
                        preview
                    ),
                    sub_calls: 0,
                })
            }

            Command::RustCliIntent {
                intent,
                on,
                store,
                timeout_secs,
            } => {
                // Reset timing
                self.last_compile_time_ms = 0;
                self.last_wasm_run_time_ms = 0;
                self.last_codegen_time_ms = 0;
                self.last_cli_run_time_ms = 0;

                // Check if code generator is configured
                let generator = self
                    .code_generator
                    .as_ref()
                    .ok_or(CommandError::CodeGenNotConfigured)?;

                // Check if Rust compiler is available
                let compiler = self.rust_compiler.as_ref().ok_or_else(|| {
                    CommandError::RustCompilerUnavailable(
                        "rustc not found. Install Rust: https://rustup.rs".to_string(),
                    )
                })?;

                // Generate CLI code using the coding LLM
                tracing::info!("Generating CLI code for intent: {}", intent);
                let codegen_start = Instant::now();

                // Use block_in_place to call async code from sync context
                let handle = tokio::runtime::Handle::try_current()
                    .map_err(|e| CommandError::CodeGenError(format!("No tokio runtime: {}", e)))?;

                let generator_ref = generator;
                let intent_clone = intent.clone();
                let code = tokio::task::block_in_place(|| {
                    handle.block_on(async { generator_ref.generate_cli(&intent_clone).await })
                })
                .map_err(|e| CommandError::CodeGenError(e.to_string()))?;

                self.last_codegen_time_ms = codegen_start.elapsed().as_millis() as u64;
                tracing::info!(
                    "CLI code generation took {}ms, generated {} bytes",
                    self.last_codegen_time_ms,
                    code.len()
                );
                tracing::debug!("Generated CLI code:\n{}", code);

                // Compile the CLI binary
                let compile_start = Instant::now();
                let binary_path = compiler
                    .compile_cli(&code, &self.cli_binary_cache_dir)
                    .map_err(|e| {
                        tracing::error!("CLI compilation failed for code:\n{}", code);
                        CommandError::RustCompileError(e.to_string())
                    })?;
                self.last_compile_time_ms = compile_start.elapsed().as_millis() as u64;
                tracing::info!(
                    "CLI compilation took {}ms (binary: {:?})",
                    self.last_compile_time_ms,
                    binary_path
                );

                // Get source data
                let source = self.resolve_source(on)?.to_string();
                let source_len = source.len();

                // Execute the binary with stdin piped
                let exec_start = Instant::now();
                let timeout = std::time::Duration::from_secs(timeout_secs.unwrap_or(30));

                let mut child = std::process::Command::new(&binary_path)
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .map_err(|e| {
                        CommandError::CodeGenError(format!("Failed to spawn CLI binary: {}", e))
                    })?;

                // Write input to stdin
                if let Some(mut stdin) = child.stdin.take() {
                    use std::io::Write;
                    stdin.write_all(source.as_bytes()).map_err(|e| {
                        CommandError::CodeGenError(format!("Failed to write to stdin: {}", e))
                    })?;
                    // stdin is dropped here, closing it
                }

                // Wait for completion with timeout
                let output = match child.wait_with_output() {
                    Ok(output) => {
                        if exec_start.elapsed() > timeout {
                            return Err(CommandError::CodeGenError(
                                "CLI binary execution timed out".to_string(),
                            ));
                        }
                        output
                    }
                    Err(e) => {
                        return Err(CommandError::CodeGenError(format!(
                            "CLI binary execution failed: {}",
                            e
                        )));
                    }
                };

                let exec_ms = exec_start.elapsed().as_millis() as u64;
                self.last_cli_run_time_ms = exec_ms;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    tracing::error!("CLI binary failed: {}", stderr);
                    return Err(CommandError::CodeGenError(format!(
                        "CLI binary exited with error: {}",
                        stderr
                    )));
                }

                let result = String::from_utf8_lossy(&output.stdout).trim().to_string();
                self.store_result(store, result.clone());

                // Return concise output
                let preview = if result.len() > 100 {
                    format!("{}...", truncate_to_char_boundary(&result, 97))
                } else {
                    result.clone()
                };
                Ok(ExecutionResult::Continue {
                    output: format!(
                        "rust_cli_intent (codegen: {}ms, compile: {}ms, exec: {}ms, {} bytes processed): {}",
                        self.last_codegen_time_ms,
                        self.last_compile_time_ms,
                        exec_ms,
                        source_len,
                        preview
                    ),
                    sub_calls: 0,
                })
            }
        }
    }

    /// Expand ${var} references in a string
    fn expand_vars(&self, s: &str) -> String {
        let mut result = s.to_string();

        // Expand ${var} patterns - allow dots/underscores in var names
        let re = Regex::new(r"\$\{([\w.]+)\}").unwrap();
        for cap in re.captures_iter(s) {
            let var_expr = &cap[1];
            // Handle property access like ${var.count} - extract base name
            let var_name = var_expr.split('.').next().unwrap_or(var_expr);
            let replacement = self
                .variables
                .get(var_name)
                .map(|s| s.as_str())
                .unwrap_or("");
            result = result.replace(&cap[0], replacement);
        }

        // Also support $var pattern (without braces)
        let re2 = Regex::new(r"\$(\w+)").unwrap();
        for cap in re2.captures_iter(&result.clone()) {
            let var_name = &cap[1];
            // Skip if already expanded (had braces)
            if !s.contains(&format!("${{{}}}", var_name)) {
                let replacement = self
                    .variables
                    .get(var_name)
                    .map(|s| s.as_str())
                    .unwrap_or("");
                result = result.replace(&cap[0], replacement);
            }
        }

        result
    }

    /// Get a variable value
    pub fn get_variable(&self, name: &str) -> Option<&String> {
        self.variables.get(name)
    }

    /// Get total sub-calls made
    pub fn sub_calls(&self) -> usize {
        self.sub_calls
    }
}

/// Extract JSON command blocks from LLM response
pub fn extract_commands(response: &str) -> Option<String> {
    // Look for ```json blocks
    let patterns = [
        ("```json\n", "```"),
        ("```json\r\n", "```"),
        ("```\n", "```"),
    ];

    for (start_pat, end_pat) in &patterns {
        if let Some(start_idx) = response.find(start_pat) {
            let code_start = start_idx + start_pat.len();
            if let Some(end_idx) = response[code_start..].find(end_pat) {
                let content = response[code_start..code_start + end_idx].trim();
                // Verify it looks like JSON
                if content.starts_with('{') || content.starts_with('[') {
                    return Some(content.to_string());
                }
            }
        }
    }

    // Try to find inline JSON (single line starting with {)
    for line in response.lines() {
        let line = line.trim();
        if line.starts_with('{') && line.ends_with('}') {
            return Some(line.to_string());
        }
    }

    // Try to find multi-line JSON object (without code fence)
    // Look for a line starting with '{' and find the matching '}'
    let trimmed = response.trim();
    if trimmed.starts_with('{') {
        // Find the matching closing brace
        let mut depth = 0;
        let mut end_idx = None;
        for (i, c) in trimmed.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end_idx = Some(i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        if let Some(end) = end_idx {
            let json = &trimmed[..end];
            // Verify it's valid JSON by trying to check basic structure
            if json.contains("\"op\"") {
                return Some(json.to_string());
            }
        }
    }

    None
}

/// Extract FINAL answer from LLM response (fallback for non-JSON responses)
pub fn extract_final(response: &str) -> Option<String> {
    // FINAL(answer) pattern - handle multiline with nested parens
    if let Some(start) = response.find("FINAL(") {
        let content_start = start + 6;
        let mut depth = 1;
        let mut end_idx = None;
        for (i, c) in response[content_start..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end_idx = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        if let Some(end) = end_idx {
            return Some(response[content_start..content_start + end].to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_command() {
        let mut exec = CommandExecutor::new("Hello, World!".to_string(), 10);
        let cmd = Command::Slice {
            start: 0,
            end: Some(5),
            store: Some("greeting".to_string()),
        };
        exec.execute_one(&cmd).unwrap();
        assert_eq!(exec.get_variable("greeting"), Some(&"Hello".to_string()));
    }

    #[test]
    fn test_slice_negative_indices() {
        // Test Python-style negative indices
        let mut exec = CommandExecutor::new("Hello, World!".to_string(), 10);

        // -6 means 6 from end = position 7 ("World!")
        let cmd = Command::Slice {
            start: -6,
            end: None,
            store: Some("result".to_string()),
        };
        exec.execute_one(&cmd).unwrap();
        assert_eq!(exec.get_variable("result"), Some(&"World!".to_string()));

        // Start at 0, end at -1 means everything except last char
        let cmd2 = Command::Slice {
            start: 0,
            end: Some(-1),
            store: Some("result2".to_string()),
        };
        exec.execute_one(&cmd2).unwrap();
        assert_eq!(
            exec.get_variable("result2"),
            Some(&"Hello, World".to_string())
        );
    }

    #[test]
    fn test_lines_negative_indices() {
        let mut exec = CommandExecutor::new("line1\nline2\nline3\nline4\nline5".to_string(), 10);

        // Get last 2 lines
        let cmd = Command::Lines {
            start: Some(-2),
            end: None,
            store: Some("result".to_string()),
        };
        exec.execute_one(&cmd).unwrap();
        assert_eq!(
            exec.get_variable("result"),
            Some(&"line4\nline5".to_string())
        );
    }

    #[test]
    fn test_count_lines() {
        let mut exec = CommandExecutor::new("line1\nline2\nline3".to_string(), 10);
        let cmd = Command::Count {
            what: CountTarget::Lines,
            on: None,
            store: Some("count".to_string()),
        };
        exec.execute_one(&cmd).unwrap();
        assert_eq!(exec.get_variable("count"), Some(&"3".to_string()));
    }

    #[test]
    fn test_regex_command() {
        let mut exec = CommandExecutor::new("class Foo:\nclass Bar:".to_string(), 10);
        let cmd = Command::Regex {
            pattern: r"class \w+".to_string(),
            on: None,
            store: Some("classes".to_string()),
        };
        let result = exec.execute_one(&cmd).unwrap();
        if let ExecutionResult::Continue { output, .. } = result {
            assert!(output.contains("2 lines matching"));
        }
    }

    #[test]
    fn test_variable_expansion() {
        let mut exec = CommandExecutor::new("test".to_string(), 10);
        exec.variables
            .insert("name".to_string(), "World".to_string());
        let expanded = exec.expand_vars("Hello, ${name}!");
        assert_eq!(expanded, "Hello, World!");
    }

    #[test]
    fn test_extract_commands() {
        let response = r#"I'll analyze this:
```json
{"op": "count", "what": "lines"}
```
"#;
        let cmds = extract_commands(response);
        assert!(cmds.is_some());
        assert!(cmds.unwrap().contains("count"));
    }

    #[test]
    fn test_json_parsing() {
        let json = r#"{"op": "slice", "start": 0, "end": 100}"#;
        let mut exec = CommandExecutor::new("x".repeat(200), 10);
        let result = exec.execute_json(json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rust_wasm_available() {
        let exec = CommandExecutor::new("test".to_string(), 10);
        // Just check that the method exists and returns a bool
        let available = exec.rust_wasm_available();
        println!("rust_wasm_available: {}", available);
    }

    #[test]
    fn test_rust_wasm_line_count() {
        let mut exec = CommandExecutor::new("line1\nline2\nline3".to_string(), 10);

        if !exec.rust_wasm_available() {
            println!("Skipping test: rust_wasm not available");
            return;
        }

        let code = r#"
            pub fn analyze(input: &str) -> String {
                input.lines().count().to_string()
            }
        "#;

        let cmd = Command::RustWasm {
            code: code.to_string(),
            on: None,
            store: Some("count".to_string()),
        };

        let result = exec.execute_one(&cmd);
        match result {
            Ok(ExecutionResult::Continue { output, .. }) => {
                println!("Output: {}", output);
                assert_eq!(exec.get_variable("count"), Some(&"3".to_string()));
            }
            Ok(ExecutionResult::Final { .. }) => {
                panic!("Unexpected Final result");
            }
            Err(e) => {
                println!(
                    "Error (may be expected if rustc not configured for wasm): {}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_rust_wasm_word_count() {
        let mut exec = CommandExecutor::new("hello world foo bar baz".to_string(), 10);

        if !exec.rust_wasm_available() {
            println!("Skipping test: rust_wasm not available");
            return;
        }

        let code = r#"
            pub fn analyze(input: &str) -> String {
                input.split_whitespace().count().to_string()
            }
        "#;

        let cmd = Command::RustWasm {
            code: code.to_string(),
            on: None,
            store: Some("words".to_string()),
        };

        let result = exec.execute_one(&cmd);
        match result {
            Ok(ExecutionResult::Continue { .. }) => {
                assert_eq!(exec.get_variable("words"), Some(&"5".to_string()));
            }
            Err(e) => {
                println!("Error: {}", e);
            }
            _ => {}
        }
    }

    #[test]
    fn test_rust_wasm_with_hashmap() {
        let mut exec = CommandExecutor::new("a b a c b a".to_string(), 10);

        if !exec.rust_wasm_available() {
            println!("Skipping test: rust_wasm not available");
            return;
        }

        let code = r#"
            pub fn analyze(input: &str) -> String {
                let mut counts: HashMap<&str, usize> = HashMap::new();
                for word in input.split_whitespace() {
                    *counts.entry(word).or_insert(0) += 1;
                }
                // Find the most common word
                counts.iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(word, count)| format!("{}: {}", word, count))
                    .unwrap_or_else(|| "none".to_string())
            }
        "#;

        let cmd = Command::RustWasm {
            code: code.to_string(),
            on: None,
            store: Some("most_common".to_string()),
        };

        let result = exec.execute_one(&cmd);
        match result {
            Ok(ExecutionResult::Continue { .. }) => {
                let value = exec.get_variable("most_common");
                println!("Most common: {:?}", value);
                // 'a' appears 3 times
                assert!(value.is_some());
                assert!(value.unwrap().contains("a: 3"));
            }
            Err(e) => {
                println!("Error: {}", e);
            }
            _ => {}
        }
    }

    #[test]
    fn test_rust_wasm_cache_hit() {
        let mut exec = CommandExecutor::new("test input".to_string(), 10);

        if !exec.rust_wasm_available() {
            println!("Skipping test: rust_wasm not available");
            return;
        }

        let code = r#"
            pub fn analyze(input: &str) -> String {
                input.len().to_string()
            }
        "#;

        let cmd = Command::RustWasm {
            code: code.to_string(),
            on: None,
            store: Some("len".to_string()),
        };

        // First execution - compiles
        let _ = exec.execute_one(&cmd);

        // Second execution - should hit cache
        let result = exec.execute_one(&cmd);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rust_wasm_compile_error() {
        let mut exec = CommandExecutor::new("test".to_string(), 10);

        if !exec.rust_wasm_available() {
            println!("Skipping test: rust_wasm not available");
            return;
        }

        let code = r#"
            pub fn analyze(input: &str) -> String {
                this_is_not_valid_rust
            }
        "#;

        let cmd = Command::RustWasm {
            code: code.to_string(),
            on: None,
            store: None,
        };

        let result = exec.execute_one(&cmd);
        assert!(result.is_err());
        if let Err(e) = result {
            println!("Expected error: {}", e);
            assert!(e.to_string().contains("compilation failed") || e.to_string().contains("Rust"));
        }
    }
}
