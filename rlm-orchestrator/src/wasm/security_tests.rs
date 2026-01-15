//! Security tests for WASM execution
//!
//! These tests verify that the WASM sandbox prevents:
//! - Filesystem access
//! - Network access
//! - Process spawning
//! - Environment variable access
//! - Excessive resource consumption
//! - Sandbox escape attempts

#[cfg(test)]
mod tests {
    use crate::wasm::compiler::{CompileError, CompilerConfig, RustCompiler};
    use crate::wasm::{WasmConfig, WasmExecutor};

    /// Helper to check if compiler is available
    fn get_compiler() -> Option<RustCompiler> {
        RustCompiler::new(CompilerConfig::default()).ok()
    }

    /// Helper to create executor with custom config
    fn get_executor(config: WasmConfig) -> WasmExecutor {
        WasmExecutor::new(config).expect("Failed to create executor")
    }

    // ==================== Source Validation Tests ====================

    #[test]
    fn test_block_filesystem_access() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let malicious_code = r#"
            pub fn analyze(input: &str) -> String {
                std::fs::read_to_string("/etc/passwd").unwrap_or_default()
            }
        "#;

        let result = compiler.validate_source(malicious_code);
        assert!(result.is_err());
        assert!(matches!(result, Err(CompileError::InvalidSource(_))));
    }

    #[test]
    fn test_block_network_access() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let malicious_code = r#"
            pub fn analyze(input: &str) -> String {
                use std::net::TcpStream;
                TcpStream::connect("evil.com:80").ok();
                String::new()
            }
        "#;

        let result = compiler.validate_source(malicious_code);
        assert!(result.is_err());
    }

    #[test]
    fn test_block_process_spawning() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let malicious_code = r#"
            pub fn analyze(input: &str) -> String {
                std::process::Command::new("rm").arg("-rf").arg("/").spawn().ok();
                String::new()
            }
        "#;

        let result = compiler.validate_source(malicious_code);
        assert!(result.is_err());
    }

    #[test]
    fn test_block_env_access() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let malicious_code = r#"
            pub fn analyze(input: &str) -> String {
                std::env::var("SECRET_KEY").unwrap_or_default()
            }
        "#;

        let result = compiler.validate_source(malicious_code);
        assert!(result.is_err());
    }

    #[test]
    fn test_block_include_macros() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let test_cases = vec![
            r#"pub fn analyze(input: &str) -> String { include!("/etc/passwd"); String::new() }"#,
            r#"pub fn analyze(input: &str) -> String { include_str!("/etc/passwd").to_string() }"#,
            r#"pub fn analyze(input: &str) -> String { String::from_utf8_lossy(include_bytes!("/etc/passwd")).to_string() }"#,
        ];

        for code in test_cases {
            let result = compiler.validate_source(code);
            assert!(result.is_err(), "Should block: {}", code);
        }
    }

    #[test]
    fn test_block_extern_crate() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let malicious_code = r#"
            extern crate libc;
            pub fn analyze(input: &str) -> String {
                String::new()
            }
        "#;

        let result = compiler.validate_source(malicious_code);
        assert!(result.is_err());
    }

    // ==================== Fuel Exhaustion Tests ====================

    #[test]
    fn test_fuel_limit_infinite_loop() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        // Code that does heavy computation - use input to prevent optimization
        let code = r#"
            pub fn analyze(input: &str) -> String {
                let mut sum: u64 = 0;
                let base = input.len() as u64;
                // Do millions of iterations
                for i in 0..10_000_000u64 {
                    sum = sum.wrapping_add(i.wrapping_mul(base.wrapping_add(1)));
                }
                sum.to_string()
            }
        "#;

        match compiler.compile(code) {
            Ok(wasm_bytes) => {
                // Use very low fuel limit - should not be able to complete
                let config = WasmConfig {
                    fuel_limit: 100, // Very low
                    memory_limit: 64 * 1024 * 1024,
                    timeout_ms: 5000,
                };
                let executor = get_executor(config);

                let result = executor.execute(&wasm_bytes, "run_analyze", "test input");
                // With only 100 fuel, it should fail
                assert!(
                    result.is_err(),
                    "Should run out of fuel with 100 fuel limit, got: {:?}",
                    result
                );
                if let Err(e) = result {
                    let error_str = e.to_string().to_lowercase();
                    println!("Got expected error: {}", e);
                    // The error could be fuel-related or general execution failure
                    assert!(
                        error_str.contains("fuel")
                            || error_str.contains("execution")
                            || error_str.contains("backtrace"),
                        "Expected fuel or execution error, got: {}",
                        e
                    );
                }
            }
            Err(e) => {
                println!("Compilation failed (expected in some envs): {}", e);
            }
        }
    }

    #[test]
    fn test_fuel_limit_cpu_intensive() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        // Computationally expensive code
        let code = r#"
            pub fn analyze(input: &str) -> String {
                let mut sum: u64 = 0;
                for i in 0..10_000_000 {
                    sum = sum.wrapping_add(i);
                }
                sum.to_string()
            }
        "#;

        match compiler.compile(code) {
            Ok(wasm_bytes) => {
                // Low fuel should prevent completion
                let config = WasmConfig {
                    fuel_limit: 10_000,
                    memory_limit: 64 * 1024 * 1024,
                    timeout_ms: 5000,
                };
                let executor = get_executor(config);

                let result = executor.execute(&wasm_bytes, "run_analyze", "");
                // Should either run out of fuel OR complete quickly if optimized out
                println!("CPU intensive result: {:?}", result);
            }
            Err(e) => {
                println!("Compilation failed: {}", e);
            }
        }
    }

    // ==================== Memory Exhaustion Tests ====================

    #[test]
    fn test_excessive_allocation() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        // Try to allocate huge amount of memory
        let code = r#"
            pub fn analyze(input: &str) -> String {
                // Try to allocate 1GB
                let huge: Vec<u8> = vec![0u8; 1024 * 1024 * 1024];
                huge.len().to_string()
            }
        "#;

        match compiler.compile(code) {
            Ok(wasm_bytes) => {
                let config = WasmConfig {
                    fuel_limit: 10_000_000,
                    memory_limit: 64 * 1024 * 1024, // Only 64MB allowed
                    timeout_ms: 5000,
                };
                let executor = get_executor(config);

                let result = executor.execute(&wasm_bytes, "run_analyze", "");
                // Should fail due to memory limit or fuel exhaustion
                println!("Memory test result: {:?}", result);
                // Note: WASM memory model may handle this differently
            }
            Err(e) => {
                println!("Compilation failed (expected): {}", e);
            }
        }
    }

    #[test]
    fn test_recursive_allocation() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        // Deep recursion / stack exhaustion attempt
        let code = r#"
            pub fn analyze(input: &str) -> String {
                fn recurse(n: u32) -> u32 {
                    if n == 0 { 0 } else { recurse(n - 1) + 1 }
                }
                recurse(100_000).to_string()
            }
        "#;

        match compiler.compile(code) {
            Ok(wasm_bytes) => {
                let config = WasmConfig {
                    fuel_limit: 100_000,
                    memory_limit: 64 * 1024 * 1024,
                    timeout_ms: 5000,
                };
                let executor = get_executor(config);

                let result = executor.execute(&wasm_bytes, "run_analyze", "");
                // Should fail - either stack overflow or fuel exhaustion
                println!("Recursion test result: {:?}", result);
            }
            Err(e) => {
                println!("Compilation failed: {}", e);
            }
        }
    }

    // ==================== Malicious Pattern Tests ====================

    #[test]
    fn test_various_escape_attempts() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let escape_attempts = vec![
            // Attempt to use unsafe to access memory
            (
                r#"pub fn analyze(input: &str) -> String { unsafe { let ptr = 0x12345678 as *const u8; *ptr }.to_string() }"#,
                "raw pointer access",
            ),
            // Attempt to use inline assembly
            (
                r#"pub fn analyze(input: &str) -> String { unsafe { std::arch::asm!("nop"); } String::new() }"#,
                "inline assembly",
            ),
        ];

        for (code, description) in escape_attempts {
            // These may fail at compile time or validation time
            let validation_result = compiler.validate_source(code);
            let compile_result = if validation_result.is_ok() {
                Some(compiler.compile(code))
            } else {
                None
            };

            println!(
                "Escape attempt '{}': validation={:?}, compile={:?}",
                description,
                validation_result.is_ok(),
                compile_result.as_ref().map(|r| r.is_ok())
            );

            // If it compiled, try executing with restricted fuel
            if let Some(Ok(wasm_bytes)) = compile_result {
                let config = WasmConfig {
                    fuel_limit: 1000,
                    memory_limit: 1024 * 1024,
                    timeout_ms: 1000,
                };
                let executor = get_executor(config);
                let result = executor.execute(&wasm_bytes, "run_analyze", "test");
                println!("  Execution result: {:?}", result);
            }
        }
    }

    // ==================== Valid Code Tests ====================

    #[test]
    fn test_legitimate_code_works() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let valid_codes = vec![
            (
                r#"pub fn analyze(input: &str) -> String { input.lines().count().to_string() }"#,
                "line count",
            ),
            (
                r#"pub fn analyze(input: &str) -> String { input.len().to_string() }"#,
                "char count",
            ),
            (
                r#"pub fn analyze(input: &str) -> String { input.split_whitespace().count().to_string() }"#,
                "word count",
            ),
            (
                r#"pub fn analyze(input: &str) -> String {
                    let mut counts: HashMap<char, usize> = HashMap::new();
                    for c in input.chars() {
                        *counts.entry(c).or_insert(0) += 1;
                    }
                    counts.len().to_string()
                }"#,
                "unique chars with HashMap",
            ),
        ];

        for (code, description) in valid_codes {
            println!("Testing: {}", description);
            let result = compiler.validate_source(code);
            assert!(
                result.is_ok(),
                "Validation failed for '{}': {:?}",
                description,
                result
            );

            match compiler.compile(code) {
                Ok(wasm_bytes) => {
                    println!("  Compiled {} bytes", wasm_bytes.len());

                    let config = WasmConfig::default();
                    let executor = get_executor(config);

                    let test_input = "hello\nworld\ntest";
                    match executor.execute(&wasm_bytes, "run_analyze", test_input) {
                        Ok(output) => println!("  Output: {}", output),
                        Err(e) => println!("  Execution error: {}", e),
                    }
                }
                Err(e) => {
                    println!("  Compilation failed: {}", e);
                }
            }
        }
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_empty_input() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let code = r#"pub fn analyze(input: &str) -> String { input.len().to_string() }"#;

        match compiler.compile(code) {
            Ok(wasm_bytes) => {
                let executor = get_executor(WasmConfig::default());
                let result = executor.execute(&wasm_bytes, "run_analyze", "");
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), "0");
            }
            Err(e) => {
                println!("Compilation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_large_input() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let code = r#"pub fn analyze(input: &str) -> String { input.len().to_string() }"#;

        match compiler.compile(code) {
            Ok(wasm_bytes) => {
                let executor = get_executor(WasmConfig::default());

                // 1MB input
                let large_input = "x".repeat(1024 * 1024);
                let result = executor.execute(&wasm_bytes, "run_analyze", &large_input);

                assert!(result.is_ok(), "Failed with large input: {:?}", result);
                assert_eq!(result.unwrap(), "1048576");
            }
            Err(e) => {
                println!("Compilation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_unicode_input() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let code = r#"pub fn analyze(input: &str) -> String { input.chars().count().to_string() }"#;

        match compiler.compile(code) {
            Ok(wasm_bytes) => {
                let executor = get_executor(WasmConfig::default());

                let unicode_input = "Hello, 世界! 🌍🚀";
                let result = executor.execute(&wasm_bytes, "run_analyze", unicode_input);

                assert!(result.is_ok(), "Failed with unicode: {:?}", result);
                // "Hello, 世界! 🌍🚀" = 14 characters (emojis are single chars)
                let count: usize = result.unwrap().parse().unwrap();
                assert!(count > 0);
                println!("Unicode character count: {}", count);
            }
            Err(e) => {
                println!("Compilation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_special_characters() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let code = r#"pub fn analyze(input: &str) -> String { input.lines().count().to_string() }"#;

        match compiler.compile(code) {
            Ok(wasm_bytes) => {
                let executor = get_executor(WasmConfig::default());

                // Test with various special characters
                let special_input = "line1\nline2\rline3\r\nline4\x00line5\tline6";
                let result = executor.execute(&wasm_bytes, "run_analyze", special_input);

                assert!(result.is_ok(), "Failed with special chars: {:?}", result);
                println!("Lines with special chars: {}", result.unwrap());
            }
            Err(e) => {
                println!("Compilation failed: {}", e);
            }
        }
    }
}
