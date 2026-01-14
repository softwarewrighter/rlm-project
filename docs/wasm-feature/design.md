# RLM Dynamic WASM Analysis - Detailed Design

## 1. Command Interface

### 1.1 New Command: `rust_wasm`

```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(input: &str) -> String { input.lines().count().to_string() }",
  "on": "context",
  "store": "line_count"
}
```

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `code` | string | Yes | - | Rust source code for the analysis function |
| `on` | string | No | "context" | Variable to use as input |
| `store` | string | No | null | Variable to store output |

### 1.2 Function Signature Requirements

The LLM-provided code must define a function with this exact signature:

```rust
pub fn analyze(input: &str) -> String
```

- **Input**: `&str` - the document or variable content
- **Output**: `String` - the analysis result (always text)

### 1.3 Available Standard Library

The following `std` modules are available:
- `std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque}`
- `std::string::String`
- `std::vec::Vec`
- `std::iter::Iterator`
- `std::cmp::{Ord, PartialOrd}`
- `std::fmt::Write`

**Not Available** (no_std limitation of WASM):
- `std::fs` - no filesystem
- `std::net` - no network
- `std::process` - no subprocesses
- `std::thread` - no threading
- `std::time::Instant` - no real time (Duration is okay)

## 2. Code Template

### 2.1 WASM Module Template

The LLM's code is wrapped in this template before compilation:

```rust
#![no_std]
#![no_main]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;

// Global allocator for WASM
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// Panic handler
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

// Result storage
static mut RESULT_PTR: *const u8 = core::ptr::null();
static mut RESULT_LEN: usize = 0;
static mut RESULT_STORAGE: Option<String> = None;

// Memory allocation for input
#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {
    let mut buf = Vec::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    core::mem::forget(buf);
    ptr
}

// User-provided analysis function
// --- USER CODE START ---
{user_code}
// --- USER CODE END ---

// Main entry point called by host
#[no_mangle]
pub extern "C" fn run_analyze(ptr: *const u8, len: usize) -> i32 {
    unsafe {
        let input = core::str::from_utf8_unchecked(core::slice::from_raw_parts(ptr, len));
        let result = analyze(input);
        RESULT_STORAGE = Some(result);
        if let Some(ref s) = RESULT_STORAGE {
            RESULT_PTR = s.as_ptr();
            RESULT_LEN = s.len();
        }
        0
    }
}

#[no_mangle]
pub extern "C" fn get_result_ptr() -> *const u8 {
    unsafe { RESULT_PTR }
}

#[no_mangle]
pub extern "C" fn get_result_len() -> usize {
    unsafe { RESULT_LEN }
}
```

### 2.2 Alternative: Full std Template

For simpler compilation (slower, larger modules):

```rust
use std::collections::HashMap;

// Result storage
static mut RESULT: String = String::new();

#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {
    let mut buf = Vec::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

// --- USER CODE ---
{user_code}
// --- END USER CODE ---

#[no_mangle]
pub extern "C" fn run_analyze(ptr: *const u8, len: usize) -> i32 {
    unsafe {
        let input = std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len));
        RESULT = analyze(input);
        0
    }
}

#[no_mangle]
pub extern "C" fn get_result_ptr() -> *const u8 {
    unsafe { RESULT.as_ptr() }
}

#[no_mangle]
pub extern "C" fn get_result_len() -> usize {
    unsafe { RESULT.len() }
}
```

## 3. Rust Compiler Service

### 3.1 Module Structure

```rust
// src/wasm/compiler.rs

use std::path::{Path, PathBuf};
use std::process::Command;
use std::io::Write;
use tempfile::TempDir;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CompileError {
    #[error("Rust compiler not found at {0}")]
    CompilerNotFound(PathBuf),

    #[error("WASM target not installed. Run: rustup target add wasm32-unknown-unknown")]
    TargetNotInstalled,

    #[error("Compilation failed:\n{0}")]
    CompilationFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid source code: {0}")]
    InvalidSource(String),
}

pub struct RustCompiler {
    rustc_path: PathBuf,
    template: String,
}

impl RustCompiler {
    pub fn new() -> Result<Self, CompileError> {
        let rustc_path = Self::find_rustc()?;
        Self::check_wasm_target(&rustc_path)?;

        Ok(Self {
            rustc_path,
            template: include_str!("templates/wasm_template.rs").to_string(),
        })
    }

    pub fn compile(&self, user_code: &str) -> Result<Vec<u8>, CompileError> {
        // Validate user code
        self.validate_source(user_code)?;

        // Create temp directory
        let temp_dir = TempDir::new()?;
        let source_path = temp_dir.path().join("module.rs");
        let output_path = temp_dir.path().join("module.wasm");

        // Generate full source
        let full_source = self.template.replace("{user_code}", user_code);

        // Write source file
        std::fs::write(&source_path, &full_source)?;

        // Compile
        let output = Command::new(&self.rustc_path)
            .args([
                "--target", "wasm32-unknown-unknown",
                "--crate-type", "cdylib",
                "-O",  // Optimize
                "-o", output_path.to_str().unwrap(),
                source_path.to_str().unwrap(),
            ])
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CompileError::CompilationFailed(
                Self::clean_error_message(&stderr, user_code)
            ));
        }

        // Read WASM output
        let wasm = std::fs::read(&output_path)?;
        Ok(wasm)
    }

    fn validate_source(&self, code: &str) -> Result<(), CompileError> {
        // Check for forbidden patterns
        let forbidden = [
            "include!", "include_str!", "include_bytes!",
            "std::fs", "std::net", "std::process",
            "extern crate", "#![",
        ];

        for pattern in forbidden {
            if code.contains(pattern) {
                return Err(CompileError::InvalidSource(
                    format!("Forbidden pattern: {}", pattern)
                ));
            }
        }

        // Must contain analyze function
        if !code.contains("fn analyze") {
            return Err(CompileError::InvalidSource(
                "Must define: pub fn analyze(input: &str) -> String".to_string()
            ));
        }

        Ok(())
    }

    fn find_rustc() -> Result<PathBuf, CompileError> {
        // Try PATH first
        if let Ok(output) = Command::new("rustc").arg("--version").output() {
            if output.status.success() {
                return Ok(PathBuf::from("rustc"));
            }
        }

        // Try common locations
        let locations = [
            "~/.cargo/bin/rustc",
            "/usr/local/bin/rustc",
            "/usr/bin/rustc",
        ];

        for loc in locations {
            let path = PathBuf::from(shellexpand::tilde(loc).to_string());
            if path.exists() {
                return Ok(path);
            }
        }

        Err(CompileError::CompilerNotFound(PathBuf::from("rustc")))
    }

    fn check_wasm_target(rustc: &Path) -> Result<(), CompileError> {
        let output = Command::new(rustc)
            .args(["--print", "target-list"])
            .output()?;

        let targets = String::from_utf8_lossy(&output.stdout);
        if targets.contains("wasm32-unknown-unknown") {
            Ok(())
        } else {
            Err(CompileError::TargetNotInstalled)
        }
    }

    fn clean_error_message(stderr: &str, user_code: &str) -> String {
        // Extract relevant error lines and map line numbers to user code
        // ... implementation details
        stderr.to_string()
    }
}
```

### 3.2 Compilation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Compilation Pipeline                          │
│                                                                  │
│  1. Validate      2. Template      3. Write       4. Compile    │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐    ┌─────────┐  │
│  │ Check   │─────▶│ Insert  │─────▶│ Temp    │───▶│ rustc   │  │
│  │ Source  │      │ Code    │      │ File    │    │ --target│  │
│  └─────────┘      └─────────┘      └─────────┘    │ wasm32  │  │
│       │                                           └────┬────┘  │
│       │ Error                                          │       │
│       ▼                                                ▼       │
│  ┌─────────┐                                     ┌─────────┐  │
│  │ Return  │                                     │ Read    │  │
│  │ Invalid │                                     │ .wasm   │  │
│  │ Source  │                                     └────┬────┘  │
│  └─────────┘                                          │       │
│                                                       ▼       │
│                           5. Cleanup            ┌─────────┐  │
│                           ┌─────────┐           │ Return  │  │
│                           │ Delete  │◀──────────│ Bytes   │  │
│                           │ Temp    │           └─────────┘  │
│                           └─────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Module Cache

### 4.1 Cache Implementation

```rust
// src/wasm/cache.rs

use lru::LruCache;
use md5::{Md5, Digest};
use std::path::PathBuf;
use std::num::NonZeroUsize;

pub struct ModuleCache {
    /// In-memory LRU cache
    memory: LruCache<String, Vec<u8>>,
    /// Disk cache directory
    cache_dir: PathBuf,
    /// Whether disk caching is enabled
    disk_enabled: bool,
}

impl ModuleCache {
    pub fn new(memory_size: usize, cache_dir: PathBuf) -> Self {
        let memory = LruCache::new(NonZeroUsize::new(memory_size).unwrap());
        let disk_enabled = std::fs::create_dir_all(&cache_dir).is_ok();

        Self {
            memory,
            cache_dir,
            disk_enabled,
        }
    }

    pub fn get(&mut self, source: &str) -> Option<Vec<u8>> {
        let key = Self::hash_source(source);

        // Check memory cache first
        if let Some(wasm) = self.memory.get(&key) {
            return Some(wasm.clone());
        }

        // Check disk cache
        if self.disk_enabled {
            let disk_path = self.cache_dir.join(&key);
            if let Ok(wasm) = std::fs::read(&disk_path) {
                // Promote to memory cache
                self.memory.put(key, wasm.clone());
                return Some(wasm);
            }
        }

        None
    }

    pub fn put(&mut self, source: &str, wasm: Vec<u8>) {
        let key = Self::hash_source(source);

        // Write to disk cache
        if self.disk_enabled {
            let disk_path = self.cache_dir.join(&key);
            let _ = std::fs::write(&disk_path, &wasm);
        }

        // Write to memory cache
        self.memory.put(key, wasm);
    }

    fn hash_source(source: &str) -> String {
        let mut hasher = Md5::new();
        hasher.update(source.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn clear(&mut self) {
        self.memory.clear();
        if self.disk_enabled {
            let _ = std::fs::remove_dir_all(&self.cache_dir);
            let _ = std::fs::create_dir_all(&self.cache_dir);
        }
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            memory_entries: self.memory.len(),
            memory_capacity: self.memory.cap().get(),
            disk_entries: self.disk_entry_count(),
        }
    }

    fn disk_entry_count(&self) -> usize {
        if self.disk_enabled {
            std::fs::read_dir(&self.cache_dir)
                .map(|entries| entries.count())
                .unwrap_or(0)
        } else {
            0
        }
    }
}

pub struct CacheStats {
    pub memory_entries: usize,
    pub memory_capacity: usize,
    pub disk_entries: usize,
}
```

## 5. Command Executor Integration

### 5.1 New Command Variant

```rust
// In src/commands/mod.rs

/// Execute Rust code compiled to WASM
RustWasm {
    /// Rust source code (must define `pub fn analyze(input: &str) -> String`)
    code: String,
    /// Variable to use as input (default: context)
    #[serde(default)]
    on: Option<String>,
    /// Store result in variable
    #[serde(default)]
    store: Option<String>,
}
```

### 5.2 Execution Logic

```rust
Command::RustWasm { code, on, store } => {
    let source = self.resolve_source(on)?;

    // Check cache first
    let wasm_bytes = if let Some(cached) = self.wasm_cache.get(&code) {
        cached
    } else {
        // Compile
        let compiled = self.rust_compiler
            .compile(&code)
            .map_err(|e| CommandError::WasmCompileError(e.to_string()))?;

        // Cache the result
        self.wasm_cache.put(&code, compiled.clone());
        compiled
    };

    // Execute
    let executor = self.wasm_executor.as_ref()
        .ok_or_else(|| CommandError::WasmError("WASM not available".into()))?;

    let result = executor.execute(&wasm_bytes, "run_analyze", &source)?;

    self.store_result(store, result.clone());
    Ok(ExecutionResult::Continue {
        output: format!("WASM result ({} chars)", result.len()),
        sub_calls: 0,
    })
}
```

## 6. System Prompt Updates

### 6.1 New Command Documentation

Add to the system prompt:

```
## Custom Analysis with Rust WASM

For complex analysis that can't be done with find/regex/count, write custom Rust code:

{{"op": "rust_wasm", "code": "pub fn analyze(input: &str) -> String {{ ... }}"}}

Requirements:
- Must define: pub fn analyze(input: &str) -> String
- Return value is always a String
- Available: HashMap, HashSet, Vec, iterators, string operations
- NOT available: file I/O, network, threads

Example - Count unique words:
{{"op": "rust_wasm", "code": "pub fn analyze(input: &str) -> String {{ use std::collections::HashSet; let words: HashSet<_> = input.split_whitespace().collect(); words.len().to_string() }}", "store": "unique_count"}}

Example - Extract numbers and sum:
{{"op": "rust_wasm", "code": "pub fn analyze(input: &str) -> String {{ let sum: i64 = input.split_whitespace().filter_map(|w| w.parse::<i64>().ok()).sum(); sum.to_string() }}"}}

When to use rust_wasm:
- Aggregations (count unique, sum, average)
- Complex parsing (CSV, JSON lines)
- Multi-step transformations
- When find/regex would need many iterations
```

### 6.2 Example Prompts for LLM

```
// Word frequency top 10
pub fn analyze(input: &str) -> String {
    use std::collections::HashMap;
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for word in input.split_whitespace() {
        *counts.entry(word).or_insert(0) += 1;
    }
    let mut pairs: Vec<_> = counts.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    pairs.iter().take(10)
        .map(|(w, c)| format!("{}: {}", w, c))
        .collect::<Vec<_>>()
        .join("\n")
}

// Extract email domains
pub fn analyze(input: &str) -> String {
    input.split_whitespace()
        .filter(|w| w.contains('@'))
        .filter_map(|email| email.split('@').nth(1))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join("\n")
}

// Parse CSV and compute column average
pub fn analyze(input: &str) -> String {
    let values: Vec<f64> = input.lines()
        .skip(1) // Skip header
        .filter_map(|line| line.split(',').nth(2)) // Third column
        .filter_map(|val| val.trim().parse().ok())
        .collect();
    if values.is_empty() {
        "No numeric values found".to_string()
    } else {
        let avg = values.iter().sum::<f64>() / values.len() as f64;
        format!("Average: {:.2}", avg)
    }
}
```

## 7. Error Handling

### 7.1 Compilation Errors

When compilation fails, return a helpful message:

```
WASM compilation failed:
error[E0425]: cannot find value `x` in this scope
  --> user code line 3
   |
 3 |     let y = x + 1;
   |             ^ not found in this scope

Fix the code and try again.
```

### 7.2 Runtime Errors

| Error | Message | Recovery |
|-------|---------|----------|
| Out of fuel | "WASM execution exceeded instruction limit (10M)" | Simplify algorithm |
| Out of memory | "WASM exceeded memory limit (64MB)" | Reduce data structures |
| Panic | "WASM module panicked" | Fix logic error |
| Invalid output | "WASM returned invalid UTF-8" | Ensure String output |

## 8. Testing Strategy

### 8.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_simple_function() {
        let compiler = RustCompiler::new().unwrap();
        let code = r#"
            pub fn analyze(input: &str) -> String {
                input.len().to_string()
            }
        "#;
        let result = compiler.compile(code);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_invalid_code() {
        let compiler = RustCompiler::new().unwrap();
        let code = "this is not rust";
        let result = compiler.compile(code);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_compiled_module() {
        let compiler = RustCompiler::new().unwrap();
        let executor = WasmExecutor::new(WasmConfig::default()).unwrap();

        let code = r#"
            pub fn analyze(input: &str) -> String {
                input.lines().count().to_string()
            }
        "#;

        let wasm = compiler.compile(code).unwrap();
        let result = executor.execute(&wasm, "run_analyze", "a\nb\nc");
        assert_eq!(result.unwrap(), "3");
    }

    #[test]
    fn test_cache_hit() {
        let mut cache = ModuleCache::new(10, PathBuf::from("/tmp/test-cache"));
        let code = "test code";
        let wasm = vec![0, 1, 2, 3];

        cache.put(code, wasm.clone());
        let cached = cache.get(code);

        assert_eq!(cached, Some(wasm));
    }
}
```

### 8.2 Integration Tests

```rust
#[tokio::test]
async fn test_rust_wasm_command() {
    let orchestrator = create_test_orchestrator();
    let context = "word1 word2 word1 word3 word1";

    let result = orchestrator.process(
        "Count unique words using WASM",
        context
    ).await.unwrap();

    // Should use rust_wasm command and return "3"
    assert!(result.answer.contains("3"));
}
```

## 9. Configuration

### 9.1 Full Config Structure

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct WasmFeatureConfig {
    /// Enable dynamic Rust compilation
    pub enabled: bool,

    /// Compiler settings
    pub compiler: CompilerConfig,

    /// Runtime limits
    pub runtime: RuntimeConfig,

    /// Cache settings
    pub cache: CacheConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompilerConfig {
    /// Path to rustc (auto-detect if not set)
    pub rustc_path: Option<PathBuf>,

    /// Additional rustc flags
    pub extra_flags: Vec<String>,

    /// Compilation timeout in seconds
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeConfig {
    /// Maximum WASM instructions (fuel)
    pub fuel_limit: u64,

    /// Maximum memory in bytes
    pub memory_limit: usize,

    /// Execution timeout in milliseconds
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CacheConfig {
    /// In-memory cache size
    pub memory_size: usize,

    /// Disk cache directory
    pub disk_dir: PathBuf,

    /// Maximum disk cache size in bytes
    pub max_disk_size: u64,
}
```
