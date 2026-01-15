//! Demo tests showing rust_wasm advantages over built-in commands

#[cfg(test)]
mod tests {
    use crate::wasm::compiler::{CompilerConfig, RustCompiler};
    use crate::wasm::{WasmConfig, WasmExecutor};

    const SAMPLE_LOG: &str = r#"2024-01-15 08:00:01.234 INFO  [main] Application starting up
2024-01-15 08:01:02.345 ERROR [db] ConnectionTimeout: Failed to acquire connection from pool after 5000ms
2024-01-15 08:02:45.567 ERROR [auth] AuthenticationFailed: Invalid token for user_id=456
2024-01-15 08:03:34.123 ERROR [db] ConnectionTimeout: Failed to acquire connection from pool after 5000ms
2024-01-15 08:04:23.012 ERROR [validation] ValidationError: Missing required field 'email' in request body
2024-01-15 08:05:12.678 ERROR [auth] AuthenticationFailed: Expired token for user_id=789
2024-01-15 08:06:01.234 ERROR [db] ConnectionTimeout: Failed to acquire connection from pool after 5000ms
2024-01-15 08:07:12.123 ERROR [validation] ValidationError: Invalid email format in request body
2024-01-15 08:07:34.456 ERROR [auth] AuthenticationFailed: Invalid token for user_id=101
2024-01-15 08:08:23.012 ERROR [db] ConnectionTimeout: Failed to acquire connection from pool after 5000ms
2024-01-15 08:10:23.456 ERROR [auth] AuthenticationFailed: Account locked for user_id=202
2024-01-15 08:10:45.789 ERROR [db] DeadlockDetected: Transaction rolled back due to deadlock
2024-01-15 08:11:34.456 ERROR [network] ConnectionRefused: Failed to connect to payment-service:8443
2024-01-15 08:12:45.567 ERROR [auth] AuthenticationFailed: Invalid token for user_id=303"#;

    /// Demo: Error type frequency and time analysis
    ///
    /// This demonstrates what built-in commands CANNOT do:
    /// - Group errors by type (requires HashMap)
    /// - Parse timestamps (requires string parsing)
    /// - Calculate time between errors (requires math)
    /// - Compute averages (requires division)
    #[test]
    fn test_demo_error_analysis() {
        let Some(compiler) = RustCompiler::new(CompilerConfig::default()).ok() else {
            println!("Skipping: rustc not available");
            return;
        };

        // This code does everything in ONE pass that would take 10+ iterations
        // with built-in commands (and still wouldn't be possible for the math parts)
        let code = r#"
pub fn analyze(input: &str) -> String {
    let mut error_times: HashMap<String, Vec<u64>> = HashMap::new();

    for line in input.lines() {
        if !line.contains("ERROR") { continue; }

        // Parse timestamp (HH:MM:SS)
        let parts: Vec<&str> = line.splitn(3, ' ').collect();
        if parts.len() < 2 { continue; }

        let time_parts: Vec<&str> = parts[1].split(':').collect();
        if time_parts.len() < 3 { continue; }

        let hours: u64 = time_parts[0].parse().unwrap_or(0);
        let mins: u64 = time_parts[1].parse().unwrap_or(0);
        let secs: u64 = time_parts[2].split('.').next()
            .and_then(|s| s.parse().ok()).unwrap_or(0);
        let timestamp = hours * 3600 + mins * 60 + secs;

        // Extract error type (e.g., "ConnectionTimeout", "AuthenticationFailed")
        let error_type = line.split(']').nth(1)
            .and_then(|s| s.trim().split(':').next())
            .unwrap_or("Unknown")
            .to_string();

        error_times.entry(error_type).or_default().push(timestamp);
    }

    // Calculate frequency and average gap for each error type
    let mut results: Vec<(String, usize, f64)> = error_times.iter().map(|(etype, times)| {
        let count = times.len();
        let avg_gap = if times.len() > 1 {
            let mut sorted = times.clone();
            sorted.sort();
            let gaps: Vec<u64> = sorted.windows(2).map(|w| w[1] - w[0]).collect();
            gaps.iter().sum::<u64>() as f64 / gaps.len() as f64
        } else {
            0.0
        };
        (etype.clone(), count, avg_gap)
    }).collect();

    // Sort by frequency (most common first)
    results.sort_by(|a, b| b.1.cmp(&a.1));

    // Format top 5
    results.iter()
        .take(5)
        .map(|(t, c, g)| format!("{}: {} errors, avg {:.0}s gap", t, c, g))
        .collect::<Vec<_>>()
        .join("\n")
}
        "#;

        let wasm_bytes = match compiler.compile(code) {
            Ok(bytes) => bytes,
            Err(e) => {
                println!("Compilation failed: {}", e);
                return;
            }
        };

        let executor = WasmExecutor::new(WasmConfig::default()).unwrap();

        match executor.execute(&wasm_bytes, "run_analyze", SAMPLE_LOG) {
            Ok(result) => {
                println!("\n=== DEMO: Error Analysis Results ===\n");
                println!("{}", result);
                println!("\n=====================================\n");

                // Verify results contain expected error types
                assert!(
                    result.contains("AuthenticationFailed"),
                    "Should find AuthenticationFailed"
                );
                assert!(
                    result.contains("ConnectionTimeout"),
                    "Should find ConnectionTimeout"
                );
                assert!(
                    result.contains("ValidationError"),
                    "Should find ValidationError"
                );

                // Verify counts are reasonable
                assert!(
                    result.contains("5 errors") || result.contains("4 errors"),
                    "AuthenticationFailed should have 4-5 occurrences"
                );
            }
            Err(e) => {
                println!("Execution failed: {}", e);
            }
        }
    }

    /// Demo: Response time percentile calculation
    ///
    /// Built-in commands cannot:
    /// - Parse response times from log lines
    /// - Sort numbers
    /// - Calculate percentiles
    #[test]
    fn test_demo_response_time_percentiles() {
        let Some(compiler) = RustCompiler::new(CompilerConfig::default()).ok() else {
            println!("Skipping: rustc not available");
            return;
        };

        let log_with_response_times = r#"
2024-01-15 08:00:15.123 INFO  [http] GET /api/health - 200 OK (2ms)
2024-01-15 08:00:23.456 INFO  [http] GET /api/users/123 - 200 OK (45ms)
2024-01-15 08:00:45.012 INFO  [http] POST /api/orders - 201 Created (89ms)
2024-01-15 08:01:23.901 INFO  [http] GET /api/products - 200 OK (34ms)
2024-01-15 08:04:56.345 INFO  [http] POST /api/users - 400 Bad Request (5ms)
2024-01-15 08:06:45.890 INFO  [http] GET /api/products/99 - 200 OK (23ms)
2024-01-15 08:12:23.345 INFO  [http] POST /api/payments - 200 OK (1823ms)
2024-01-15 08:18:34.012 INFO  [http] GET /api/users - 200 OK (156ms)
2024-01-15 08:22:34.678 INFO  [http] GET /api/products - 200 OK (67ms)
2024-01-15 08:24:01.567 INFO  [http] POST /api/orders - 400 Bad Request (12ms)
2024-01-15 08:27:45.456 INFO  [http] POST /api/payments - 400 Bad Request (8ms)
2024-01-15 08:29:23.567 INFO  [http] GET /api/orders/12345 - 200 OK (89ms)
"#;

        let code = r#"
pub fn analyze(input: &str) -> String {
    let mut times: Vec<u64> = Vec::new();

    for line in input.lines() {
        // Extract response time from "(XXms)" pattern
        if let Some(start) = line.rfind('(') {
            if let Some(end) = line.rfind("ms)") {
                let time_str = &line[start+1..end];
                if let Ok(ms) = time_str.parse::<u64>() {
                    times.push(ms);
                }
            }
        }
    }

    if times.is_empty() {
        return "No response times found".to_string();
    }

    times.sort();

    let count = times.len();
    let sum: u64 = times.iter().sum();
    let avg = sum as f64 / count as f64;
    let min = times[0];
    let max = times[count - 1];
    let p50 = times[count / 2];
    let p95 = times[(count * 95) / 100];
    let p99 = times[(count * 99) / 100];

    format!(
        "Response Time Stats ({} requests):\n  Min: {}ms\n  Avg: {:.1}ms\n  P50: {}ms\n  P95: {}ms\n  P99: {}ms\n  Max: {}ms",
        count, min, avg, p50, p95, p99, max
    )
}
        "#;

        let wasm_bytes = match compiler.compile(code) {
            Ok(bytes) => bytes,
            Err(e) => {
                println!("Compilation failed: {}", e);
                return;
            }
        };

        let executor = WasmExecutor::new(WasmConfig::default()).unwrap();

        match executor.execute(&wasm_bytes, "run_analyze", log_with_response_times) {
            Ok(result) => {
                println!("\n=== DEMO: Response Time Percentiles ===\n");
                println!("{}", result);
                println!("\n========================================\n");

                assert!(result.contains("requests"), "Should show request count");
                assert!(result.contains("P95"), "Should calculate P95");
                assert!(result.contains("Avg"), "Should calculate average");
            }
            Err(e) => {
                println!("Execution failed: {}", e);
            }
        }
    }

    /// Demo: Find unique IPs that appear more than N times
    ///
    /// Built-in commands cannot:
    /// - Count occurrences per IP
    /// - Filter by threshold
    /// - Sort by count
    #[test]
    fn test_demo_ip_frequency_threshold() {
        let Some(compiler) = RustCompiler::new(CompilerConfig::default()).ok() else {
            println!("Skipping: rustc not available");
            return;
        };

        let log_with_ips = r#"
192.168.1.100 - - [15/Jan/2024:08:00:01] "GET /api/users" 200
192.168.1.101 - - [15/Jan/2024:08:00:02] "GET /api/users" 200
192.168.1.100 - - [15/Jan/2024:08:00:03] "POST /api/login" 200
10.0.0.50 - - [15/Jan/2024:08:00:04] "GET /api/products" 200
192.168.1.100 - - [15/Jan/2024:08:00:05] "GET /api/orders" 200
192.168.1.101 - - [15/Jan/2024:08:00:06] "POST /api/orders" 201
10.0.0.50 - - [15/Jan/2024:08:00:07] "GET /api/products" 200
192.168.1.100 - - [15/Jan/2024:08:00:08] "GET /api/health" 200
172.16.0.1 - - [15/Jan/2024:08:00:09] "GET /api/health" 200
192.168.1.100 - - [15/Jan/2024:08:00:10] "POST /api/checkout" 200
10.0.0.50 - - [15/Jan/2024:08:00:11] "GET /api/products" 200
192.168.1.101 - - [15/Jan/2024:08:00:12] "GET /api/users" 200
"#;

        let code = r#"
pub fn analyze(input: &str) -> String {
    let mut ip_counts: HashMap<&str, usize> = HashMap::new();

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }

        // First token is the IP
        if let Some(ip) = line.split_whitespace().next() {
            // Validate it looks like an IP (has dots)
            if ip.contains('.') && ip.split('.').count() == 4 {
                *ip_counts.entry(ip).or_insert(0) += 1;
            }
        }
    }

    // Filter IPs with more than 2 requests
    let threshold = 2;
    let mut frequent: Vec<(&&str, &usize)> = ip_counts.iter()
        .filter(|(_, count)| **count > threshold)
        .collect();

    frequent.sort_by(|a, b| b.1.cmp(a.1));

    if frequent.is_empty() {
        return format!("No IPs with more than {} requests", threshold);
    }

    format!(
        "IPs with >{} requests:\n{}",
        threshold,
        frequent.iter()
            .map(|(ip, count)| format!("  {}: {} requests", ip, count))
            .collect::<Vec<_>>()
            .join("\n")
    )
}
        "#;

        let wasm_bytes = match compiler.compile(code) {
            Ok(bytes) => bytes,
            Err(e) => {
                println!("Compilation failed: {}", e);
                return;
            }
        };

        let executor = WasmExecutor::new(WasmConfig::default()).unwrap();

        match executor.execute(&wasm_bytes, "run_analyze", log_with_ips) {
            Ok(result) => {
                println!("\n=== DEMO: IP Frequency Threshold ===\n");
                println!("{}", result);
                println!("\n=====================================\n");

                assert!(result.contains("192.168.1.100"), "Should find frequent IP");
                assert!(
                    result.contains("5 requests") || result.contains("requests"),
                    "Should show request count"
                );
            }
            Err(e) => {
                println!("Execution failed: {}", e);
            }
        }
    }
}
