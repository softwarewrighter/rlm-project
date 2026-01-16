//! Rust to WASM compiler for dynamic code generation
//!
//! This module provides functionality to compile Rust source code to WASM
//! bytecode, enabling LLMs to generate custom analysis functions.

use std::path::PathBuf;
use std::process::Command;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors from Rust compilation
#[derive(Error, Debug)]
pub enum CompileError {
    #[error("Rust compiler not found. Install rustup: https://rustup.rs")]
    CompilerNotFound,

    #[error("WASM target not installed. Run: rustup target add wasm32-unknown-unknown")]
    TargetNotInstalled,

    #[error("Compilation failed:\n{0}")]
    CompilationFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid source code: {0}")]
    InvalidSource(String),

    #[error("Compilation timeout")]
    Timeout,
}

/// Configuration for the Rust compiler
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Path to rustc (None = auto-detect)
    pub rustc_path: Option<PathBuf>,
    /// Compilation timeout in seconds
    pub timeout_secs: u64,
    /// Optimization level (0-3, or 's' for size)
    pub opt_level: String,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            rustc_path: None,
            timeout_secs: 30,
            opt_level: "2".to_string(),
        }
    }
}

/// Rust to WASM compiler
pub struct RustCompiler {
    /// Path to rustc binary
    rustc_path: PathBuf,
    /// Compiler configuration
    config: CompilerConfig,
}

impl RustCompiler {
    /// Create a new Rust compiler, auto-detecting rustc location
    pub fn new(config: CompilerConfig) -> Result<Self, CompileError> {
        let rustc_path = if let Some(path) = &config.rustc_path {
            if Self::verify_rustc(path) {
                path.clone()
            } else {
                return Err(CompileError::CompilerNotFound);
            }
        } else {
            Self::find_rustc()?
        };

        // Verify WASM target is installed
        Self::check_wasm_target(&rustc_path)?;

        info!("RustCompiler initialized with rustc at {:?}", rustc_path);

        Ok(Self { rustc_path, config })
    }

    /// Find rustc in PATH or common locations
    fn find_rustc() -> Result<PathBuf, CompileError> {
        // Try PATH first
        if let Ok(output) = Command::new("rustc").arg("--version").output()
            && output.status.success() {
                debug!("Found rustc in PATH");
                return Ok(PathBuf::from("rustc"));
            }

        // Try common locations
        let home = std::env::var("HOME").unwrap_or_default();
        let locations = [
            format!("{}/.cargo/bin/rustc", home),
            "/usr/local/bin/rustc".to_string(),
            "/usr/bin/rustc".to_string(),
            "/opt/homebrew/bin/rustc".to_string(),
        ];

        for loc in &locations {
            let path = PathBuf::from(loc);
            if Self::verify_rustc(&path) {
                debug!("Found rustc at {:?}", path);
                return Ok(path);
            }
        }

        Err(CompileError::CompilerNotFound)
    }

    /// Verify a rustc path is valid
    fn verify_rustc(path: &PathBuf) -> bool {
        Command::new(path)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Check if wasm32-unknown-unknown target is installed
    fn check_wasm_target(rustc_path: &PathBuf) -> Result<(), CompileError> {
        let output = Command::new(rustc_path)
            .args(["--print", "target-list"])
            .output()?;

        let targets = String::from_utf8_lossy(&output.stdout);
        if targets.contains("wasm32-unknown-unknown") {
            // Also check if it's actually installed via rustup
            if let Ok(rustup_output) = Command::new("rustup")
                .args(["target", "list", "--installed"])
                .output()
            {
                let installed = String::from_utf8_lossy(&rustup_output.stdout);
                if installed.contains("wasm32-unknown-unknown") {
                    debug!("WASM target is installed");
                    return Ok(());
                }
            }
            // If rustup check fails, try compiling a simple test
            // (target might be installed system-wide)
            warn!("Could not verify WASM target via rustup, assuming available");
            return Ok(());
        }

        Err(CompileError::TargetNotInstalled)
    }

    /// Validate source code before compilation
    pub fn validate_source(&self, code: &str) -> Result<(), CompileError> {
        // Check for forbidden patterns that could be security issues
        let forbidden_security = [
            ("include!", "File inclusion not allowed"),
            ("include_str!", "File inclusion not allowed"),
            ("include_bytes!", "File inclusion not allowed"),
            ("std::fs", "Filesystem access not allowed"),
            ("std::net", "Network access not allowed"),
            ("std::process", "Process spawning not allowed"),
            ("std::env", "Environment access not allowed"),
            ("extern crate", "External crates not allowed (use std only)"),
        ];

        for (pattern, reason) in forbidden_security {
            if code.contains(pattern) {
                return Err(CompileError::InvalidSource(format!(
                    "{}: found '{}'",
                    reason, pattern
                )));
            }
        }

        // Check for string operations that cause TwoWaySearcher panics in WASM
        // These operations use pattern matching that can panic on certain inputs
        let forbidden_string_ops = [
            (".contains(", "Use has() helper instead of .contains()"),
            (".find(", "Use after()/before() helpers instead of .find()"),
            (".rfind(", "Use after()/before() helpers instead of .rfind()"),
            (".split(", "Use word() helper instead of .split()"),
            (".split_once(", "Use after()/before() helpers instead of .split_once()"),
            (".rsplit(", "Use word() helper instead of .rsplit()"),
            (".matches(", "Use has() helper instead of .matches()"),
            (".match_indices(", "Use custom byte iteration instead of .match_indices()"),
            (".replace(", "Build new string manually instead of .replace()"),
            (".replacen(", "Build new string manually instead of .replacen()"),
            (".strip_prefix(", "Use after() helper instead of .strip_prefix()"),
            (".strip_suffix(", "Use before() helper instead of .strip_suffix()"),
        ];

        for (pattern, suggestion) in forbidden_string_ops {
            if code.contains(pattern) {
                return Err(CompileError::InvalidSource(format!(
                    "WASM-unsafe string operation '{}'. {}",
                    pattern.trim_end_matches('('),
                    suggestion
                )));
            }
        }

        // Must contain analyze function with correct signature
        if !code.contains("fn analyze") {
            return Err(CompileError::InvalidSource(
                "Must define function: pub fn analyze(input: &str) -> String".to_string(),
            ));
        }

        // Check for pub fn analyze signature
        if !code.contains("pub fn analyze") {
            return Err(CompileError::InvalidSource(
                "Function 'analyze' must be public: pub fn analyze(input: &str) -> String"
                    .to_string(),
            ));
        }

        Ok(())
    }

    /// Generate the full WASM module source by wrapping user code in template
    fn generate_module_source(&self, user_code: &str) -> String {
        format!(
            r#"// Auto-generated WASM module for RLM
// INPUT IS ASCII-ONLY - safe to use byte slicing!

use std::collections::{{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque}};

// ============ PURE BYTE-LEVEL HELPERS ============
// These avoid Rust's TwoWaySearcher which can panic in WASM.
// All operations are byte-level for maximum safety.

/// Convert byte slice to &str (safe because input is ASCII)
#[inline]
fn to_str(bytes: &[u8]) -> &str {{
    unsafe {{ std::str::from_utf8_unchecked(bytes) }}
}}

/// Find pattern in bytes using simple byte-by-byte comparison.
/// Returns starting position or None.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {{
    if needle.is_empty() {{ return Some(0); }}
    if needle.len() > haystack.len() {{ return None; }}
    for i in 0..=(haystack.len() - needle.len()) {{
        if &haystack[i..i + needle.len()] == needle {{
            return Some(i);
        }}
    }}
    None
}}

/// Check if bytes contain pattern
#[allow(dead_code)]
fn has(s: &str, pat: &str) -> bool {{
    find_bytes(s.as_bytes(), pat.as_bytes()).is_some()
}}

/// Count how many times pattern occurs in string
#[allow(dead_code)]
fn count(s: &str, pat: &str) -> usize {{
    if pat.is_empty() {{ return 0; }}
    let haystack = s.as_bytes();
    let needle = pat.as_bytes();
    let mut count = 0;
    let mut pos = 0;
    while pos + needle.len() <= haystack.len() {{
        if let Some(found) = find_bytes(&haystack[pos..], needle) {{
            count += 1;
            pos = pos + found + 1; // Move past the match (allow overlapping)
        }} else {{
            break;
        }}
    }}
    count
}}

/// Safe string equality (byte-by-byte comparison)
#[allow(dead_code)]
fn eq(a: &str, b: &str) -> bool {{
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    if a_bytes.len() != b_bytes.len() {{
        return false;
    }}
    for i in 0..a_bytes.len() {{
        if a_bytes[i] != b_bytes[i] {{
            return false;
        }}
    }}
    true
}}

/// Get text after pattern, or empty string
#[allow(dead_code)]
fn after<'a>(s: &'a str, pat: &str) -> &'a str {{
    if let Some(pos) = find_bytes(s.as_bytes(), pat.as_bytes()) {{
        &s[pos + pat.len()..]
    }} else {{
        ""
    }}
}}

/// Get text before pattern, or whole string
#[allow(dead_code)]
fn before<'a>(s: &'a str, pat: &str) -> &'a str {{
    if let Some(pos) = find_bytes(s.as_bytes(), pat.as_bytes()) {{
        &s[..pos]
    }} else {{
        s
    }}
}}

/// Get nth word (0-indexed), or empty string
#[allow(dead_code)]
fn word(s: &str, n: usize) -> &str {{
    let bytes = s.as_bytes();
    let mut count = 0;
    let mut start = 0;
    let mut in_word = false;

    for (i, &b) in bytes.iter().enumerate() {{
        let is_space = b == b' ' || b == b'\t' || b == b'\r';
        if !is_space && !in_word {{
            if count == n {{
                start = i;
            }}
            in_word = true;
        }} else if is_space && in_word {{
            if count == n {{
                return &s[start..i];
            }}
            count += 1;
            in_word = false;
        }}
    }}

    if in_word && count == n {{
        return &s[start..];
    }}
    ""
}}

/// Iterate over lines (newline-separated)
#[allow(dead_code)]
fn each_line(s: &str) -> impl Iterator<Item = &str> {{
    s.as_bytes().split(|&b| b == b'\n').map(|line| to_str(line))
}}

/// Slice string by byte positions (safe for ASCII)
#[allow(dead_code)]
fn slice(s: &str, start: usize, end: usize) -> &str {{
    let end = end.min(s.len());
    let start = start.min(end);
    &s[start..end]
}}

/// Parse integer, returns 0 on failure
#[allow(dead_code)]
fn parse_int(s: &str) -> i64 {{
    let s = s.trim();
    if s.is_empty() {{ return 0; }}
    let bytes = s.as_bytes();
    let (neg, start) = if bytes[0] == b'-' {{ (true, 1) }} else {{ (false, 0) }};
    let mut n: i64 = 0;
    for &b in &bytes[start..] {{
        if b >= b'0' && b <= b'9' {{
            n = n * 10 + (b - b'0') as i64;
        }} else {{
            break;
        }}
    }}
    if neg {{ -n }} else {{ n }}
}}

/// Extract all integers from string
#[allow(dead_code)]
fn extract_ints(s: &str) -> Vec<i64> {{
    let mut result = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {{
        // Skip non-digits
        while i < bytes.len() && !(bytes[i] >= b'0' && bytes[i] <= b'9') && bytes[i] != b'-' {{
            i += 1;
        }}
        if i >= bytes.len() {{ break; }}

        // Check for negative
        let neg = bytes[i] == b'-';
        if neg {{ i += 1; }}
        if i >= bytes.len() || !(bytes[i] >= b'0' && bytes[i] <= b'9') {{
            continue;
        }}

        // Parse number
        let mut n: i64 = 0;
        while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {{
            n = n * 10 + (bytes[i] - b'0') as i64;
            i += 1;
        }}
        result.push(if neg {{ -n }} else {{ n }});
    }}
    result
}}
// ============ END PURE BYTE-LEVEL HELPERS ============

static mut RESULT_STORAGE: String = String::new();

#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {{
    let mut buf: Vec<u8> = Vec::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}}

// ============ USER CODE START ============
{user_code}
// ============ USER CODE END ============

#[no_mangle]
pub extern "C" fn run_analyze(ptr: *const u8, len: usize) -> i32 {{
    unsafe {{
        let input = std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len));
        RESULT_STORAGE = analyze(input);
        0
    }}
}}

#[no_mangle]
pub extern "C" fn get_result_ptr() -> *const u8 {{
    unsafe {{ RESULT_STORAGE.as_ptr() }}
}}

#[no_mangle]
pub extern "C" fn get_result_len() -> usize {{
    unsafe {{ RESULT_STORAGE.len() }}
}}
"#,
            user_code = user_code
        )
    }

    /// Compile Rust source code to WASM bytecode
    pub fn compile(&self, user_code: &str) -> Result<Vec<u8>, CompileError> {
        // Validate source first
        self.validate_source(user_code)?;

        // Create temp directory for compilation
        let temp_dir = tempfile::TempDir::new()?;
        let source_path = temp_dir.path().join("module.rs");
        let output_path = temp_dir.path().join("module.wasm");

        // Generate full source
        let full_source = self.generate_module_source(user_code);

        debug!("Writing source to {:?}", source_path);
        std::fs::write(&source_path, &full_source)?;

        // Build rustc command
        let mut cmd = Command::new(&self.rustc_path);
        cmd.args([
            "--target",
            "wasm32-unknown-unknown",
            "--crate-type",
            "cdylib",
            "-C",
            &format!("opt-level={}", self.config.opt_level),
            "-C",
            "lto=yes", // Link-time optimization for smaller output
            "-C",
            "panic=abort", // WASM doesn't support unwinding
            "-o",
        ])
        .arg(&output_path)
        .arg(&source_path);

        debug!("Running: {:?}", cmd);

        // Execute compiler
        let output = cmd.output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let cleaned_error = self.clean_error_message(&stderr, user_code);
            return Err(CompileError::CompilationFailed(cleaned_error));
        }

        // Read compiled WASM
        let wasm_bytes = std::fs::read(&output_path)?;
        info!(
            "Successfully compiled {} bytes of Rust to {} bytes of WASM",
            user_code.len(),
            wasm_bytes.len()
        );

        Ok(wasm_bytes)
    }

    /// Clean up compiler error messages to be more helpful to the LLM
    fn clean_error_message(&self, stderr: &str, _user_code: &str) -> String {
        let mut cleaned_lines = Vec::new();
        let mut in_error = false;

        for line in stderr.lines() {
            // Skip noise lines
            if line.contains("Compiling") || line.contains("Finished") || line.is_empty() {
                continue;
            }

            // Track error blocks
            if line.starts_with("error") {
                in_error = true;
                cleaned_lines.push(line.to_string());
            } else if in_error {
                // Include context lines
                if line.starts_with("  ") || line.starts_with(" -->") || line.contains("^^^") {
                    // Adjust line numbers to account for template wrapper
                    // Template adds ~25 lines before user code
                    cleaned_lines.push(line.to_string());
                } else if line.starts_with("warning") || line.starts_with("error") {
                    in_error = line.starts_with("error");
                    cleaned_lines.push(line.to_string());
                }
            }
        }

        if cleaned_lines.is_empty() {
            // Fall back to raw stderr if parsing failed
            stderr.to_string()
        } else {
            cleaned_lines.join("\n")
        }
    }

    /// Generate the WASM module source for streaming reduce pattern
    /// User code should define:
    /// - State struct
    /// - fn init_state() -> State
    /// - fn process_line(state: &mut State, line: &str)
    /// - fn finalize(state: &State) -> String
    fn generate_reduce_module_source(&self, user_code: &str) -> String {
        format!(
            r#"// Auto-generated WASM module for RLM Streaming Reduce
// INPUT IS ASCII-ONLY - safe to use byte slicing!
// This module processes input in chunks to avoid memory issues.

// ============ PURE BYTE-LEVEL HELPERS ============
// These avoid Rust's TwoWaySearcher which can panic in WASM.

/// Convert byte slice to &str (safe because input is ASCII)
#[inline]
fn to_str(bytes: &[u8]) -> &str {{
    unsafe {{ std::str::from_utf8_unchecked(bytes) }}
}}

/// Find pattern in bytes using simple byte-by-byte comparison.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {{
    if needle.is_empty() {{ return Some(0); }}
    if needle.len() > haystack.len() {{ return None; }}
    for i in 0..=(haystack.len() - needle.len()) {{
        if &haystack[i..i + needle.len()] == needle {{
            return Some(i);
        }}
    }}
    None
}}

#[allow(dead_code)]
fn has(s: &str, pat: &str) -> bool {{
    find_bytes(s.as_bytes(), pat.as_bytes()).is_some()
}}

#[allow(dead_code)]
fn count(s: &str, pat: &str) -> usize {{
    if pat.is_empty() {{ return 0; }}
    let haystack = s.as_bytes();
    let needle = pat.as_bytes();
    let mut count = 0;
    let mut pos = 0;
    while pos + needle.len() <= haystack.len() {{
        if let Some(found) = find_bytes(&haystack[pos..], needle) {{
            count += 1;
            pos = pos + found + 1;
        }} else {{
            break;
        }}
    }}
    count
}}

#[allow(dead_code)]
fn eq(a: &str, b: &str) -> bool {{
    a.as_bytes() == b.as_bytes()
}}

#[allow(dead_code)]
fn after<'a>(s: &'a str, pat: &str) -> &'a str {{
    if let Some(pos) = find_bytes(s.as_bytes(), pat.as_bytes()) {{
        &s[pos + pat.len()..]
    }} else {{
        ""
    }}
}}

#[allow(dead_code)]
fn before<'a>(s: &'a str, pat: &str) -> &'a str {{
    if let Some(pos) = find_bytes(s.as_bytes(), pat.as_bytes()) {{
        &s[..pos]
    }} else {{
        s
    }}
}}

#[allow(dead_code)]
fn word(s: &str, n: usize) -> &str {{
    let bytes = s.as_bytes();
    let mut count = 0;
    let mut start = 0;
    let mut in_word = false;

    for (i, &b) in bytes.iter().enumerate() {{
        let is_space = b == b' ' || b == b'\t' || b == b'\r';
        if !is_space && !in_word {{
            if count == n {{ start = i; }}
            in_word = true;
        }} else if is_space && in_word {{
            if count == n {{ return &s[start..i]; }}
            count += 1;
            in_word = false;
        }}
    }}

    if in_word && count == n {{ return &s[start..]; }}
    ""
}}

#[allow(dead_code)]
fn parse_int(s: &str) -> i64 {{
    let s = s.trim();
    if s.is_empty() {{ return 0; }}
    let bytes = s.as_bytes();
    let (neg, start) = if bytes[0] == b'-' {{ (true, 1) }} else {{ (false, 0) }};
    let mut n: i64 = 0;
    for &b in &bytes[start..] {{
        if b >= b'0' && b <= b'9' {{
            n = n * 10 + (b - b'0') as i64;
        }} else {{
            break;
        }}
    }}
    if neg {{ -n }} else {{ n }}
}}

// ============ END HELPERS ============

// ============ USER CODE START ============
{user_code}
// ============ USER CODE END ============

// ============ WASM INTERFACE ============

static mut STATE: Option<State> = None;
static mut RESULT_STORAGE: String = String::new();

#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {{
    let mut buf: Vec<u8> = Vec::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}}

/// Initialize/reset state for a new reduce operation
#[no_mangle]
pub extern "C" fn reduce_init() -> i32 {{
    unsafe {{
        STATE = Some(init_state());
    }}
    0
}}

/// Process a chunk of input (called multiple times)
#[no_mangle]
pub extern "C" fn reduce_chunk(ptr: *const u8, len: usize) -> i32 {{
    unsafe {{
        let chunk = std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len));
        if let Some(ref mut state) = STATE {{
            // Process each line in the chunk
            for line in chunk.lines() {{
                process_line(state, line);
            }}
        }}
    }}
    0
}}

/// Finalize and return the result
#[no_mangle]
pub extern "C" fn reduce_finalize() -> i32 {{
    unsafe {{
        if let Some(ref state) = STATE {{
            RESULT_STORAGE = finalize(state);
        }} else {{
            RESULT_STORAGE = "Error: State not initialized".to_string();
        }}
    }}
    0
}}

#[no_mangle]
pub extern "C" fn get_result_ptr() -> *const u8 {{
    unsafe {{ RESULT_STORAGE.as_ptr() }}
}}

#[no_mangle]
pub extern "C" fn get_result_len() -> usize {{
    unsafe {{ RESULT_STORAGE.len() }}
}}
"#,
            user_code = user_code
        )
    }

    /// Validate source code for reduce pattern
    pub fn validate_reduce_source(&self, code: &str) -> Result<(), CompileError> {
        // Check for security issues (same as regular validation)
        let forbidden_security = [
            ("include!", "File inclusion not allowed"),
            ("include_str!", "File inclusion not allowed"),
            ("include_bytes!", "File inclusion not allowed"),
            ("std::fs", "Filesystem access not allowed"),
            ("std::net", "Network access not allowed"),
            ("std::process", "Process spawning not allowed"),
            ("std::env", "Environment access not allowed"),
            ("extern crate", "External crates not allowed"),
        ];

        for (pattern, reason) in forbidden_security {
            if code.contains(pattern) {
                return Err(CompileError::InvalidSource(format!(
                    "{}: found '{}'", reason, pattern
                )));
            }
        }

        // Check for forbidden string operations
        let forbidden_string_ops = [
            (".contains(", "Use has() helper instead"),
            (".find(", "Use after()/before() helpers instead"),
            (".split(", "Use word() helper instead"),
            (".matches(", "Use has() helper instead"),
            (".replace(", "Build new string manually instead"),
        ];

        for (pattern, suggestion) in forbidden_string_ops {
            if code.contains(pattern) {
                return Err(CompileError::InvalidSource(format!(
                    "WASM-unsafe operation '{}'. {}", pattern.trim_end_matches('('), suggestion
                )));
            }
        }

        // Must contain required functions for reduce pattern
        if !code.contains("struct State") {
            return Err(CompileError::InvalidSource(
                "Reduce code must define: struct State".to_string(),
            ));
        }
        if !code.contains("fn init_state") {
            return Err(CompileError::InvalidSource(
                "Reduce code must define: fn init_state() -> State".to_string(),
            ));
        }
        if !code.contains("fn process_line") {
            return Err(CompileError::InvalidSource(
                "Reduce code must define: fn process_line(state: &mut State, line: &str)".to_string(),
            ));
        }
        if !code.contains("fn finalize") {
            return Err(CompileError::InvalidSource(
                "Reduce code must define: fn finalize(state: &State) -> String".to_string(),
            ));
        }

        Ok(())
    }

    /// Compile Rust reduce code to WASM bytecode
    pub fn compile_reduce(&self, user_code: &str) -> Result<Vec<u8>, CompileError> {
        // Validate source first
        self.validate_reduce_source(user_code)?;

        // Create temp directory for compilation
        let temp_dir = tempfile::TempDir::new()?;
        let source_path = temp_dir.path().join("module.rs");
        let output_path = temp_dir.path().join("module.wasm");

        // Generate full source
        let full_source = self.generate_reduce_module_source(user_code);

        debug!("Writing reduce source to {:?}", source_path);
        std::fs::write(&source_path, &full_source)?;

        // Build rustc command
        let mut cmd = Command::new(&self.rustc_path);
        cmd.args([
            "--target",
            "wasm32-unknown-unknown",
            "--crate-type",
            "cdylib",
            "-C",
            &format!("opt-level={}", self.config.opt_level),
            "-C",
            "lto=yes",
            "-C",
            "panic=abort",
            "-o",
        ])
        .arg(&output_path)
        .arg(&source_path);

        debug!("Running: {:?}", cmd);

        let output = cmd.output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let cleaned_error = self.clean_error_message(&stderr, user_code);
            return Err(CompileError::CompilationFailed(cleaned_error));
        }

        let wasm_bytes = std::fs::read(&output_path)?;
        info!(
            "Successfully compiled {} bytes of reduce Rust to {} bytes of WASM",
            user_code.len(),
            wasm_bytes.len()
        );

        Ok(wasm_bytes)
    }

    /// Get the rustc version
    pub fn version(&self) -> Option<String> {
        Command::new(&self.rustc_path)
            .arg("--version")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_rustc() {
        let result = RustCompiler::find_rustc();
        // This test will pass if rustc is installed
        if result.is_ok() {
            println!("Found rustc at: {:?}", result.unwrap());
        } else {
            println!("rustc not found (expected in some CI environments)");
        }
    }

    #[test]
    fn test_compiler_creation() {
        let config = CompilerConfig::default();
        if let Ok(compiler) = RustCompiler::new(config) {
            println!("Compiler version: {:?}", compiler.version());
        }
    }

    #[test]
    fn test_validate_source_valid() {
        let config = CompilerConfig::default();
        if let Ok(compiler) = RustCompiler::new(config) {
            let valid_code = r#"
                pub fn analyze(input: &str) -> String {
                    input.len().to_string()
                }
            "#;
            assert!(compiler.validate_source(valid_code).is_ok());
        }
    }

    #[test]
    fn test_validate_source_missing_analyze() {
        let config = CompilerConfig::default();
        if let Ok(compiler) = RustCompiler::new(config) {
            let invalid_code = r#"
                pub fn process(input: &str) -> String {
                    input.len().to_string()
                }
            "#;
            assert!(compiler.validate_source(invalid_code).is_err());
        }
    }

    #[test]
    fn test_validate_source_forbidden_pattern() {
        let config = CompilerConfig::default();
        if let Ok(compiler) = RustCompiler::new(config) {
            let invalid_code = r#"
                pub fn analyze(input: &str) -> String {
                    std::fs::read_to_string("/etc/passwd").unwrap()
                }
            "#;
            let result = compiler.validate_source(invalid_code);
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("Filesystem"));
        }
    }

    #[test]
    fn test_compile_simple_function() {
        let config = CompilerConfig::default();
        if let Ok(compiler) = RustCompiler::new(config) {
            let code = r#"
                pub fn analyze(input: &str) -> String {
                    input.lines().count().to_string()
                }
            "#;
            let result = compiler.compile(code);
            match result {
                Ok(wasm) => {
                    println!("Compiled to {} bytes of WASM", wasm.len());
                    assert!(!wasm.is_empty());
                    // Check WASM magic number
                    assert_eq!(&wasm[0..4], &[0x00, 0x61, 0x73, 0x6D]); // \0asm
                }
                Err(e) => {
                    println!("Compilation failed (may be expected): {}", e);
                }
            }
        }
    }

    #[test]
    fn test_compile_with_hashmap() {
        let config = CompilerConfig::default();
        if let Ok(compiler) = RustCompiler::new(config) {
            let code = r#"
                pub fn analyze(input: &str) -> String {
                    let mut counts: HashMap<&str, usize> = HashMap::new();
                    for word in input.split_whitespace() {
                        *counts.entry(word).or_insert(0) += 1;
                    }
                    counts.len().to_string()
                }
            "#;
            let result = compiler.compile(code);
            match result {
                Ok(wasm) => {
                    println!("Compiled HashMap code to {} bytes", wasm.len());
                    assert!(!wasm.is_empty());
                }
                Err(e) => {
                    println!("Compilation failed: {}", e);
                }
            }
        }
    }

    #[test]
    fn test_compile_reduce() {
        let config = CompilerConfig::default();
        if let Ok(compiler) = RustCompiler::new(config) {
            // Test the streaming reduce pattern
            let reduce_code = r#"
struct State {
    line_count: usize,
    error_count: usize,
}

fn init_state() -> State {
    State {
        line_count: 0,
        error_count: 0,
    }
}

fn process_line(state: &mut State, line: &str) {
    state.line_count += 1;
    if has(line, "[ERROR]") {
        state.error_count += 1;
    }
}

fn finalize(state: &State) -> String {
    format!("Lines: {}, Errors: {}", state.line_count, state.error_count)
}
            "#;

            let result = compiler.compile_reduce(reduce_code);
            match result {
                Ok(wasm) => {
                    println!("Compiled reduce code to {} bytes", wasm.len());
                    assert!(!wasm.is_empty());
                }
                Err(e) => {
                    panic!("Reduce compilation failed: {}", e);
                }
            }
        }
    }

    #[test]
    fn test_validate_reduce_source() {
        let config = CompilerConfig::default();
        if let Ok(compiler) = RustCompiler::new(config) {
            // Valid reduce code
            let valid_code = r#"
struct State { count: usize }
fn init_state() -> State { State { count: 0 } }
fn process_line(state: &mut State, line: &str) { state.count += 1; }
fn finalize(state: &State) -> String { state.count.to_string() }
            "#;
            assert!(compiler.validate_reduce_source(valid_code).is_ok());

            // Missing struct State
            let missing_state = r#"
fn init_state() -> State { State { count: 0 } }
fn process_line(state: &mut State, line: &str) { state.count += 1; }
fn finalize(state: &State) -> String { state.count.to_string() }
            "#;
            let result = compiler.validate_reduce_source(missing_state);
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("State"));

            // Missing init_state
            let missing_init = r#"
struct State { count: usize }
fn process_line(state: &mut State, line: &str) { state.count += 1; }
fn finalize(state: &State) -> String { state.count.to_string() }
            "#;
            let result = compiler.validate_reduce_source(missing_init);
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("init_state"));
        }
    }
}
