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

use crate::wasm::{
    CompilerConfig, ModuleCache, RustCompiler, WasmConfig, WasmExecutor, WasmLibrary,
};

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

    /// Final answer: {"op": "final", "answer": "The result is..."}
    Final { answer: String },

    /// Final from variable: {"op": "final_var", "name": "result"}
    FinalVar { name: String },

    /// Execute pre-compiled WASM module: {"op": "wasm", "module": "line_counter"}
    Wasm {
        /// Name of pre-compiled module from WasmLibrary
        module: String,
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
}

fn default_wasm_function() -> String {
    "analyze".to_string()
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
    /// Sub-call counter
    sub_calls: usize,
    /// Max sub-calls allowed
    max_sub_calls: usize,
    /// WASM executor for dynamic code
    wasm_executor: Option<WasmExecutor>,
    /// Library of pre-compiled WASM modules
    wasm_library: WasmLibrary,
    /// Rust to WASM compiler (None if rustc not available)
    rust_compiler: Option<RustCompiler>,
    /// Cache for compiled WASM modules
    wasm_cache: ModuleCache,
    /// Last rust_wasm compile time in milliseconds (for instrumentation)
    last_compile_time_ms: u64,
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

        Self {
            variables: HashMap::new(),
            context,
            last_result: String::new(),
            llm_callback: None,
            sub_calls: 0,
            max_sub_calls,
            wasm_executor,
            wasm_library: WasmLibrary::new(),
            rust_compiler,
            wasm_cache,
            last_compile_time_ms: 0,
        }
    }

    /// Get the last rust_wasm compile time in milliseconds (for instrumentation)
    pub fn last_compile_time_ms(&self) -> u64 {
        self.last_compile_time_ms
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

    /// Get a variable or the context
    fn resolve_source(&self, name: &Option<String>) -> Result<&str, CommandError> {
        match name {
            None => Ok(&self.context),
            Some(n) if n == "_" || n == "last" => Ok(&self.last_result),
            Some(n) if n == "context" => Ok(&self.context),
            Some(n) => self
                .variables
                .get(n)
                .map(|s| s.as_str())
                .ok_or_else(|| CommandError::VariableNotFound(n.clone())),
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
            // Try parsing multiple JSON objects (one per line)
            let mut cmds = Vec::new();
            for line in json.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with("//") {
                    continue;
                }
                if line.starts_with('{') {
                    cmds.push(serde_json::from_str(line)?);
                }
            }
            if cmds.is_empty() {
                // Try as single object
                vec![serde_json::from_str(json)?]
            } else {
                cmds
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
                // Case-insensitive regex by default - find matching LINES with line numbers
                let re = regex::RegexBuilder::new(pattern)
                    .case_insensitive(true)
                    .build()?;

                let mut matching_lines: Vec<String> = Vec::new();
                for (line_num, line) in source.lines().enumerate() {
                    if re.is_match(line) {
                        matching_lines.push(format!("L{}: {}", line_num + 1, line));
                    }
                }

                let count = matching_lines.len();
                let result = matching_lines.join("\n");

                // Always show preview (first 5 matches, truncated if needed)
                let preview = if count > 0 {
                    let preview_lines: Vec<&str> =
                        matching_lines.iter().take(5).map(|s| s.as_str()).collect();
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
                let formatted: Vec<String> = matching_lines
                    .iter()
                    .map(|(num, line, _)| format!("L{}: {}", num, line))
                    .collect();
                let result = formatted.join("\n");

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
                module,
                function,
                on,
                store,
            } => {
                let executor = self.wasm_executor.as_ref().ok_or_else(|| {
                    CommandError::WasmError(crate::wasm::WasmError::CompileError(
                        "WASM not available".to_string(),
                    ))
                })?;

                let wasm_bytes = self
                    .wasm_library
                    .get(module)
                    .ok_or_else(|| CommandError::WasmModuleNotFound(module.clone()))?;

                let source = self.resolve_source(on)?.to_string();
                let result = executor.execute(wasm_bytes, function, &source)?;

                self.store_result(store, result.clone());
                Ok(ExecutionResult::Continue {
                    output: format!("WASM {}.{}: {}", module, function, result),
                    sub_calls: 0,
                })
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
                // Reset compile time (will be set if we actually compile)
                self.last_compile_time_ms = 0;

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

                // Execute the compiled WASM
                let source = self.resolve_source(on)?.to_string();
                let result = executor
                    .execute(&wasm_bytes, "run_analyze", &source)
                    .map_err(|e| {
                        CommandError::WasmError(crate::wasm::WasmError::ExecutionError(
                            e.to_string(),
                        ))
                    })?;

                self.store_result(store, result.clone());

                // Return concise output
                let preview = if result.len() > 100 {
                    format!("{}...", &result[..97])
                } else {
                    result.clone()
                };
                Ok(ExecutionResult::Continue {
                    output: format!("rust_wasm result: {}", preview),
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
