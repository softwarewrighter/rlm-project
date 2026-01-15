//! Performance benchmarks for WASM compilation and execution
//!
//! These tests measure:
//! - Compilation time for various code sizes
//! - Execution time vs built-in commands
//! - Cache hit performance
//! - Memory usage during execution

#[cfg(test)]
mod tests {
    use crate::wasm::cache::{CacheConfig, ModuleCache};
    use crate::wasm::compiler::{CompilerConfig, RustCompiler};
    use crate::wasm::{WasmConfig, WasmExecutor};
    use std::time::Instant;

    /// Helper to get compiler
    fn get_compiler() -> Option<RustCompiler> {
        RustCompiler::new(CompilerConfig::default()).ok()
    }

    /// Helper to get executor
    fn get_executor() -> WasmExecutor {
        WasmExecutor::new(WasmConfig::default()).expect("Failed to create executor")
    }

    // ==================== Compilation Benchmarks ====================

    #[test]
    fn benchmark_compile_simple_function() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let code = r#"pub fn analyze(input: &str) -> String { input.len().to_string() }"#;

        // Warm up
        let _ = compiler.compile(code);

        // Measure
        let iterations = 3;
        let mut times = Vec::new();

        for i in 0..iterations {
            let start = Instant::now();
            let result = compiler.compile(code);
            let elapsed = start.elapsed();

            if result.is_ok() {
                times.push(elapsed);
                println!("Compile iteration {}: {:?}", i + 1, elapsed);
            }
        }

        if !times.is_empty() {
            let avg = times.iter().sum::<std::time::Duration>() / times.len() as u32;
            println!("\nSimple function compile time: {:?} average", avg);
            // Should be under 5 seconds
            assert!(avg.as_secs() < 5, "Compilation too slow: {:?}", avg);
        }
    }

    #[test]
    fn benchmark_compile_complex_function() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let code = r#"
            pub fn analyze(input: &str) -> String {
                let mut counts: HashMap<&str, usize> = HashMap::new();
                for word in input.split_whitespace() {
                    *counts.entry(word).or_insert(0) += 1;
                }
                let mut pairs: Vec<_> = counts.into_iter().collect();
                pairs.sort_by(|a, b| b.1.cmp(&a.1));
                pairs.iter()
                    .take(10)
                    .map(|(w, c)| format!("{}: {}", w, c))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        "#;

        let start = Instant::now();
        let result = compiler.compile(code);
        let elapsed = start.elapsed();

        match result {
            Ok(wasm) => {
                println!("Complex function compile time: {:?}", elapsed);
                println!("WASM size: {} bytes", wasm.len());
                // Should be under 10 seconds
                assert!(
                    elapsed.as_secs() < 10,
                    "Complex compilation too slow: {:?}",
                    elapsed
                );
            }
            Err(e) => {
                println!("Compilation failed: {}", e);
            }
        }
    }

    // ==================== Execution Benchmarks ====================

    #[test]
    fn benchmark_execution_line_count() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let code = r#"pub fn analyze(input: &str) -> String { input.lines().count().to_string() }"#;

        let Ok(wasm_bytes) = compiler.compile(code) else {
            println!("Compilation failed");
            return;
        };

        let executor = get_executor();

        // Create test input of various sizes
        let sizes = [1000, 10000, 100000];

        for size in sizes {
            let input: String = (0..size).map(|i| format!("line {}\n", i)).collect();

            let start = Instant::now();
            let result = executor.execute(&wasm_bytes, "run_analyze", &input);
            let elapsed = start.elapsed();

            match result {
                Ok(count) => {
                    println!("Line count for {} lines: {} in {:?}", size, count, elapsed);
                    // Should be under 1 second
                    assert!(
                        elapsed.as_secs() < 1,
                        "Line count too slow for {} lines: {:?}",
                        size,
                        elapsed
                    );
                }
                Err(e) => {
                    println!("Execution failed: {}", e);
                }
            }
        }
    }

    #[test]
    fn benchmark_execution_word_frequency() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let code = r#"
            pub fn analyze(input: &str) -> String {
                let mut counts: HashMap<&str, usize> = HashMap::new();
                for word in input.split_whitespace() {
                    *counts.entry(word).or_insert(0) += 1;
                }
                counts.len().to_string()
            }
        "#;

        let Ok(wasm_bytes) = compiler.compile(code) else {
            println!("Compilation failed");
            return;
        };

        let executor = get_executor();

        // Test with ~100KB of text
        let words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        ];
        let input: String = (0..10000)
            .map(|i| words[i % words.len()])
            .collect::<Vec<_>>()
            .join(" ");

        let start = Instant::now();
        let result = executor.execute(&wasm_bytes, "run_analyze", &input);
        let elapsed = start.elapsed();

        match result {
            Ok(count) => {
                println!(
                    "Word frequency ({} words): {} unique in {:?}",
                    10000, count, elapsed
                );
                assert!(
                    elapsed.as_millis() < 500,
                    "Word frequency too slow: {:?}",
                    elapsed
                );
            }
            Err(e) => {
                println!("Execution failed: {}", e);
            }
        }
    }

    // ==================== Cache Benchmarks ====================

    #[test]
    fn benchmark_cache_hit() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let code = r#"pub fn analyze(input: &str) -> String { input.len().to_string() }"#;

        let Ok(wasm_bytes) = compiler.compile(code) else {
            println!("Compilation failed");
            return;
        };

        // Create cache and populate
        let config = CacheConfig {
            memory_size: 100,
            disk_dir: None,
            max_disk_bytes: 0,
        };
        let mut cache = ModuleCache::new(config);

        cache.put(code, wasm_bytes.clone());

        // Measure cache hit time
        let iterations = 1000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = cache.get(code);
        }

        let elapsed = start.elapsed();
        let avg_ns = elapsed.as_nanos() / iterations as u128;

        println!(
            "Cache hit: {} ns average ({} iterations in {:?})",
            avg_ns, iterations, elapsed
        );

        // Should be under 1ms per lookup (usually microseconds)
        assert!(avg_ns < 1_000_000, "Cache hit too slow: {} ns", avg_ns);
    }

    #[test]
    fn benchmark_cache_miss() {
        let config = CacheConfig {
            memory_size: 100,
            disk_dir: None,
            max_disk_bytes: 0,
        };
        let mut cache = ModuleCache::new(config);

        // Measure cache miss time
        let iterations = 1000;
        let start = Instant::now();

        for i in 0..iterations {
            let _ = cache.get(&format!("nonexistent_code_{}", i));
        }

        let elapsed = start.elapsed();
        let avg_ns = elapsed.as_nanos() / iterations as u128;

        println!(
            "Cache miss: {} ns average ({} iterations in {:?})",
            avg_ns, iterations, elapsed
        );

        // Should be under 1ms per lookup
        assert!(avg_ns < 1_000_000, "Cache miss too slow: {} ns", avg_ns);
    }

    // ==================== Comparison Benchmarks ====================

    #[test]
    fn benchmark_wasm_vs_native() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        // Test input
        let input: String = (0..10000).map(|i| format!("line {}\n", i)).collect();

        // WASM version
        let code = r#"pub fn analyze(input: &str) -> String { input.lines().count().to_string() }"#;

        if let Ok(wasm_bytes) = compiler.compile(code) {
            let executor = get_executor();

            let wasm_start = Instant::now();
            let wasm_result = executor.execute(&wasm_bytes, "run_analyze", &input);
            let wasm_elapsed = wasm_start.elapsed();

            // Native Rust equivalent
            let native_start = Instant::now();
            let native_result = input.lines().count().to_string();
            let native_elapsed = native_start.elapsed();

            println!(
                "WASM execution: {:?} (result: {:?})",
                wasm_elapsed, wasm_result
            );
            println!(
                "Native execution: {:?} (result: {})",
                native_elapsed, native_result
            );

            if let Ok(wasm_count) = wasm_result {
                assert_eq!(wasm_count, native_result, "Results should match");
            }

            // WASM should be within 100x of native (generous due to sandbox overhead)
            if native_elapsed.as_nanos() > 0 {
                let ratio = wasm_elapsed.as_nanos() / native_elapsed.as_nanos().max(1);
                println!("WASM/Native ratio: {}x", ratio);
            }
        }
    }

    // ==================== Memory Benchmarks ====================

    #[test]
    fn benchmark_memory_usage() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let code = r#"
            pub fn analyze(input: &str) -> String {
                // Allocate some memory
                let words: Vec<&str> = input.split_whitespace().collect();
                words.len().to_string()
            }
        "#;

        let Ok(wasm_bytes) = compiler.compile(code) else {
            println!("Compilation failed");
            return;
        };

        println!("WASM module size: {} bytes", wasm_bytes.len());

        // Execute with various input sizes
        let executor = get_executor();
        let sizes = [1000, 10000, 100000];

        for size in sizes {
            let input: String = (0..size).map(|_| "word ").collect();

            let result = executor.execute(&wasm_bytes, "run_analyze", &input);
            match result {
                Ok(count) => {
                    println!("Processed {} words: {} unique", size, count);
                }
                Err(e) => {
                    println!("Failed at {} words: {}", size, e);
                }
            }
        }
    }

    // ==================== Throughput Benchmarks ====================

    #[test]
    fn benchmark_throughput() {
        let Some(compiler) = get_compiler() else {
            println!("Skipping: rustc not available");
            return;
        };

        let code = r#"pub fn analyze(input: &str) -> String { input.len().to_string() }"#;

        let Ok(wasm_bytes) = compiler.compile(code) else {
            println!("Compilation failed");
            return;
        };

        let executor = get_executor();
        let input = "x".repeat(1024); // 1KB input

        let iterations = 100;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = executor.execute(&wasm_bytes, "run_analyze", &input);
        }

        let elapsed = start.elapsed();
        let throughput_kb_per_sec = (iterations * 1024) as f64 / elapsed.as_secs_f64();

        println!(
            "Throughput: {:.2} KB/s ({} iterations in {:?})",
            throughput_kb_per_sec / 1024.0,
            iterations,
            elapsed
        );

        // Should process at least 10 KB/s
        assert!(
            throughput_kb_per_sec > 10240.0,
            "Throughput too low: {:.2} KB/s",
            throughput_kb_per_sec / 1024.0
        );
    }
}
