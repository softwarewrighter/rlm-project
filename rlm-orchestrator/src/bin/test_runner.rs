//! RLM Integration Test Runner
//!
//! Runs accuracy tests against the RLM server and measures:
//! - Accuracy (correct answers)
//! - Iterations (steps to answer)
//! - Latency (time to answer)
//!
//! Usage: cargo run --bin rlm-test
//!
//! Note: This is not a cargo bench microbenchmark - it's an integration test suite
//! that requires the RLM server to be running.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};

const RLM_SERVER: &str = "http://localhost:4539";
const TIMEOUT_SECS: u64 = 120;

/// Safely truncate a string at a valid UTF-8 character boundary.
fn truncate_to_char_boundary(s: &str, max_bytes: usize) -> &str {
    if max_bytes >= s.len() {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[derive(Debug, Deserialize)]
struct BenchmarkSpec {
    name: String,
    category: String,
    #[allow(dead_code)]
    description: String,
    #[serde(default)]
    context: Option<String>,
    context_generator: Option<String>,
    context_params: Option<ContextParams>,
    /// URL path to fetch context from server (e.g., "/samples/war-and-peace")
    #[serde(default)]
    context_url: Option<String>,
    queries: Vec<QuerySpec>,
}

#[derive(Debug, Deserialize)]
struct ContextParams {
    filler_line: Option<String>,
    total_lines: Option<usize>,
    needle_line: Option<usize>,
    needle: Option<String>,
    distribution: Option<HashMap<String, usize>>,
}

#[derive(Debug, Deserialize)]
struct QuerySpec {
    query: String,
    expected: serde_json::Value,
    #[serde(default = "default_match_type")]
    match_type: String,
}

fn default_match_type() -> String {
    "contains".to_string()
}

#[derive(Debug, Serialize)]
struct QueryRequest {
    query: String,
    context: String,
}

#[derive(Debug, Deserialize)]
struct DebugResponse {
    answer: Option<String>,
    iterations: Option<usize>,
    error: Option<String>,
    #[serde(default)]
    total_prompt_tokens: u32,
    #[serde(default)]
    total_completion_tokens: u32,
    #[serde(default)]
    baseline_tokens: u32,
    #[serde(default)]
    token_savings_pct: f64,
    #[serde(default)]
    bypassed: bool,
}

#[derive(Debug)]
struct QueryResult {
    #[allow(dead_code)]
    query: String,
    #[allow(dead_code)]
    expected: String,
    #[allow(dead_code)]
    actual: String,
    passed: bool,
    iterations: usize,
    latency: Duration,
    #[allow(dead_code)]
    error: Option<String>,
    rlm_tokens: u32,
    baseline_tokens: u32,
    savings_pct: f64,
}

#[derive(Debug)]
struct TestResult {
    name: String,
    #[allow(dead_code)]
    category: String,
    #[allow(dead_code)]
    queries: Vec<QueryResult>,
    total: usize,
    passed: usize,
    failed: usize,
    avg_iterations: f64,
    avg_latency: Duration,
    total_rlm_tokens: u32,
    total_baseline_tokens: u32,
    avg_savings_pct: f64,
}

async fn generate_context(client: &Client, spec: &BenchmarkSpec) -> String {
    // First check for direct context
    if let Some(ctx) = &spec.context {
        return ctx.clone();
    }

    // Check for URL-based context
    if let Some(url_path) = &spec.context_url {
        let url = format!("{}{}", RLM_SERVER, url_path);
        match client.get(&url).send().await {
            Ok(response) if response.status().is_success() => match response.text().await {
                Ok(text) => return text,
                Err(e) => eprintln!("Error reading context from {}: {}", url, e),
            },
            Ok(response) => eprintln!(
                "Error fetching context from {}: HTTP {}",
                url,
                response.status()
            ),
            Err(e) => eprintln!("Error fetching context from {}: {}", url, e),
        }
    }

    let generator = spec.context_generator.as_deref().unwrap_or("");
    let params = spec.context_params.as_ref();

    match generator {
        "repeat_with_needle" => {
            let params = params.expect("repeat_with_needle requires context_params");
            let filler = params.filler_line.as_deref().unwrap_or("Line {n}");
            let total = params.total_lines.unwrap_or(100);
            let needle_line = params.needle_line.unwrap_or(50);
            let needle = params.needle.as_deref().unwrap_or("NEEDLE");

            let mut lines = Vec::with_capacity(total);
            for i in 0..total {
                if i + 1 == needle_line {
                    lines.push(needle.to_string());
                } else {
                    let line = filler
                        .replace("{n}", &(i + 1).to_string())
                        .replace("{sector}", &format!("S{:03}", i % 100));
                    lines.push(line);
                }
            }
            lines.join("\n")
        }
        "generate_log" => {
            let params = params.expect("generate_log requires context_params");
            let dist = params.distribution.as_ref().expect("distribution required");

            let messages: HashMap<&str, Vec<&str>> = [
                (
                    "INFO",
                    vec![
                        "Request processed",
                        "User logged in",
                        "Cache hit",
                        "Connected",
                    ],
                ),
                (
                    "DEBUG",
                    vec![
                        "Variable x=42",
                        "Entering process()",
                        "Loop iter 5",
                        "Memory 45%",
                    ],
                ),
                (
                    "WARN",
                    vec![
                        "High memory",
                        "Slow query: 2.5s",
                        "Deprecated API",
                        "Rate limit",
                    ],
                ),
                (
                    "ERROR",
                    vec![
                        "Connection refused",
                        "Query timeout",
                        "Invalid input",
                        "Auth failed",
                    ],
                ),
                (
                    "FATAL",
                    vec![
                        "Out of memory",
                        "DB unreachable",
                        "Critical failure",
                        "Shutdown",
                    ],
                ),
            ]
            .into_iter()
            .collect();

            let default_msgs = vec!["Unknown event"];
            let mut entries = Vec::new();

            for (level, count) in dist {
                let msgs = messages.get(level.as_str()).unwrap_or(&default_msgs);
                for i in 0..*count {
                    let day = (i % 28) + 1;
                    let hour = i % 24;
                    let minute = (i * 7) % 60;
                    let second = (i * 13) % 60;
                    let msg = msgs[i % msgs.len()];
                    entries.push(format!(
                        "[2024-01-{:02} {:02}:{:02}:{:02}] {}: {}",
                        day, hour, minute, second, level, msg
                    ));
                }
            }

            // Deterministic shuffle
            entries.sort_by_key(|e| e.bytes().map(|b| b as usize).sum::<usize>());
            entries.join("\n")
        }
        _ => spec.context.clone().unwrap_or_default(),
    }
}

fn check_match(actual: &str, expected: &serde_json::Value, match_type: &str) -> bool {
    let actual_lower = actual.to_lowercase();

    match match_type {
        "exact" => expected
            .as_str()
            .map(|e| actual.trim() == e.trim())
            .unwrap_or(false),
        "contains" => expected
            .as_str()
            .map(|e| actual_lower.contains(&e.to_lowercase()))
            .unwrap_or(false),
        "contains_all" => expected
            .as_array()
            .map(|arr| {
                arr.iter().all(|e| {
                    e.as_str()
                        .map(|s| actual_lower.contains(&s.to_lowercase()))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false),
        "regex" => expected
            .as_str()
            .and_then(|pattern| {
                Regex::new(&format!("(?i){}", pattern))
                    .ok()
                    .map(|re| re.is_match(actual))
            })
            .unwrap_or(false),
        _ => false,
    }
}

struct QueryMetrics {
    answer: String,
    iterations: usize,
    latency: Duration,
    error: Option<String>,
    rlm_tokens: u32,
    baseline_tokens: u32,
    savings_pct: f64,
    bypassed: bool,
}

async fn run_query(client: &Client, query: &str, context: &str) -> QueryMetrics {
    let start = Instant::now();

    let result = client
        .post(format!("{}/debug", RLM_SERVER))
        .json(&QueryRequest {
            query: query.to_string(),
            context: context.to_string(),
        })
        .timeout(Duration::from_secs(TIMEOUT_SECS))
        .send()
        .await;

    let latency = start.elapsed();

    match result {
        Ok(response) => {
            if !response.status().is_success() {
                return QueryMetrics {
                    answer: String::new(),
                    iterations: 0,
                    latency,
                    error: Some(format!("HTTP {}", response.status())),
                    rlm_tokens: 0,
                    baseline_tokens: 0,
                    savings_pct: 0.0,
                    bypassed: false,
                };
            }

            match response.json::<DebugResponse>().await {
                Ok(data) => QueryMetrics {
                    answer: data.answer.unwrap_or_default(),
                    iterations: data.iterations.unwrap_or(0),
                    latency,
                    error: data.error,
                    rlm_tokens: data.total_prompt_tokens + data.total_completion_tokens,
                    baseline_tokens: data.baseline_tokens,
                    savings_pct: data.token_savings_pct,
                    bypassed: data.bypassed,
                },
                Err(e) => QueryMetrics {
                    answer: String::new(),
                    iterations: 0,
                    latency,
                    error: Some(e.to_string()),
                    rlm_tokens: 0,
                    baseline_tokens: 0,
                    savings_pct: 0.0,
                    bypassed: false,
                },
            }
        }
        Err(e) => QueryMetrics {
            answer: String::new(),
            iterations: 0,
            latency,
            error: Some(e.to_string()),
            rlm_tokens: 0,
            baseline_tokens: 0,
            savings_pct: 0.0,
            bypassed: false,
        },
    }
}

async fn run_test_suite(client: &Client, spec: BenchmarkSpec) -> Result<TestResult, String> {
    let context = generate_context(client, &spec).await;

    println!("\n{}", "=".repeat(70));
    println!("Test Suite: {}", spec.name);
    println!("Category: {}", spec.category);
    println!(
        "Context size: {} chars (~{} tokens)",
        context.len(),
        context.len() / 4
    );
    println!("{}", "=".repeat(70));

    let mut results = Vec::new();
    let mut consecutive_errors = 0;

    for q in &spec.queries {
        let query_display = if q.query.chars().count() > 55 {
            format!("{}...", q.query.chars().take(55).collect::<String>())
        } else {
            q.query.clone()
        };
        println!("\nQuery: {}", query_display);

        let metrics = run_query(client, &q.query, &context).await;

        // Fail fast on infrastructure errors
        if let Some(err) = &metrics.error {
            consecutive_errors += 1;
            if consecutive_errors >= 2 {
                println!("  ✗ ERROR: {}", err);
                println!("\n⚠️  Multiple consecutive errors - failing fast");
                println!("    Check that the server is running and healthy");
                return Err(format!("Infrastructure error: {}", err));
            }
        } else {
            consecutive_errors = 0;
        }

        let passed =
            metrics.error.is_none() && check_match(&metrics.answer, &q.expected, &q.match_type);

        let actual_display = if metrics.answer.len() > 200 {
            format!("{}...", truncate_to_char_boundary(&metrics.answer, 200))
        } else {
            metrics.answer.clone()
        };

        let qr = QueryResult {
            query: q.query.clone(),
            expected: format!("{}", q.expected),
            actual: actual_display.clone(),
            passed,
            iterations: metrics.iterations,
            latency: metrics.latency,
            error: metrics.error.clone(),
            rlm_tokens: metrics.rlm_tokens,
            baseline_tokens: metrics.baseline_tokens,
            savings_pct: metrics.savings_pct,
        };

        if passed {
            let bypass_indicator = if metrics.bypassed { " [BYPASS]" } else { "" };
            println!(
                "  ✓ PASSED ({} iters, {:?}, {} tokens, {:.0}% saved){}",
                metrics.iterations,
                metrics.latency,
                metrics.rlm_tokens,
                metrics.savings_pct,
                bypass_indicator
            );
        } else {
            println!("  ✗ FAILED");
            if let Some(ref err) = metrics.error {
                println!("    Error: {}", err);
            } else {
                println!("    Expected: {}", q.expected);
                println!("    Got: {}", actual_display);
            }
        }

        results.push(qr);
    }

    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();

    let avg_iterations = if total > 0 {
        results.iter().map(|r| r.iterations).sum::<usize>() as f64 / total as f64
    } else {
        0.0
    };

    let avg_latency = if total > 0 {
        Duration::from_nanos(
            results.iter().map(|r| r.latency.as_nanos()).sum::<u128>() as u64 / total as u64,
        )
    } else {
        Duration::ZERO
    };

    let total_rlm_tokens: u32 = results.iter().map(|r| r.rlm_tokens).sum();
    let total_baseline_tokens: u32 = results.iter().map(|r| r.baseline_tokens).sum();
    let avg_savings_pct = if total > 0 {
        results.iter().map(|r| r.savings_pct).sum::<f64>() / total as f64
    } else {
        0.0
    };

    Ok(TestResult {
        name: spec.name,
        category: spec.category,
        queries: results,
        total,
        passed,
        failed: total - passed,
        avg_iterations,
        avg_latency,
        total_rlm_tokens,
        total_baseline_tokens,
        avg_savings_pct,
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let bench_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("benchmarks");

    // If a specific test name is provided, run only that test
    let filter = args.get(1).map(|s| s.as_str());

    let mut test_files: Vec<_> = fs::read_dir(&bench_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "json").unwrap_or(false))
        .filter(|p| {
            if let Some(name) = filter {
                p.file_stem()
                    .map(|s| s.to_string_lossy().contains(name))
                    .unwrap_or(false)
            } else {
                true
            }
        })
        .collect();

    test_files.sort();

    if test_files.is_empty() {
        if let Some(name) = filter {
            println!("No test files matching '{}' found in {:?}", name, bench_dir);
            println!("\nAvailable tests:");
            for e in fs::read_dir(&bench_dir)?.flatten() {
                if e.path()
                    .extension()
                    .map(|ext| ext == "json")
                    .unwrap_or(false)
                {
                    println!("  - {}", e.path().file_stem().unwrap().to_string_lossy());
                }
            }
        } else {
            println!("No test files found in {:?}", bench_dir);
        }
        return Ok(());
    }

    if let Some(name) = filter {
        println!("Running tests matching: {}", name);
    }
    println!("Found {} test file(s)", test_files.len());

    // Check server health
    let client = Client::new();
    match client.get(format!("{}/health", RLM_SERVER)).send().await {
        Ok(r) if r.status().is_success() => {
            println!("Connected to RLM server at {}", RLM_SERVER);
        }
        Ok(r) => {
            eprintln!("Server not healthy: {}", r.status());
            return Ok(());
        }
        Err(e) => {
            eprintln!("Cannot connect to RLM server at {}: {}", RLM_SERVER, e);
            eprintln!("Start the server first: cargo run --bin rlm-server");
            return Ok(());
        }
    }

    // Run tests
    let mut results = Vec::new();

    for path in test_files {
        match fs::read_to_string(&path) {
            Ok(content) => match serde_json::from_str::<BenchmarkSpec>(&content) {
                Ok(spec) => match run_test_suite(&client, spec).await {
                    Ok(result) => results.push(result),
                    Err(e) => {
                        eprintln!("\n❌ Test suite aborted: {}", e);
                        std::process::exit(2);
                    }
                },
                Err(e) => eprintln!("Error parsing {:?}: {}", path, e),
            },
            Err(e) => eprintln!("Error reading {:?}: {}", path, e),
        }
    }

    // Print summary
    println!("\n{}", "=".repeat(90));
    println!("TEST SUMMARY");
    println!("{}", "=".repeat(90));

    let total_passed: usize = results.iter().map(|r| r.passed).sum();
    let total_queries: usize = results.iter().map(|r| r.total).sum();

    println!(
        "\n{:<25} {:<5} {:<5} {:<6} {:<10} {:<10} {:<10} {:<8}",
        "Test Suite", "Pass", "Fail", "Iters", "Latency", "RLM Tok", "Base Tok", "Saved"
    );
    println!("{}", "-".repeat(90));

    for r in &results {
        println!(
            "{:<25} {:<5} {:<5} {:<6.1} {:<10.0?} {:<10} {:<10} {:.0}%",
            r.name,
            r.passed,
            r.failed,
            r.avg_iterations,
            r.avg_latency,
            r.total_rlm_tokens,
            r.total_baseline_tokens,
            r.avg_savings_pct
        );
    }

    println!("{}", "-".repeat(90));

    let accuracy = if total_queries > 0 {
        total_passed as f64 / total_queries as f64 * 100.0
    } else {
        0.0
    };

    println!(
        "\nOverall: {}/{} passed ({:.1}%)",
        total_passed, total_queries, accuracy
    );

    if !results.is_empty() {
        let avg_iters: f64 =
            results.iter().map(|r| r.avg_iterations).sum::<f64>() / results.len() as f64;
        let avg_lat = Duration::from_nanos(
            results
                .iter()
                .map(|r| r.avg_latency.as_nanos())
                .sum::<u128>() as u64
                / results.len() as u64,
        );
        let total_rlm: u32 = results.iter().map(|r| r.total_rlm_tokens).sum();
        let total_baseline: u32 = results.iter().map(|r| r.total_baseline_tokens).sum();
        let overall_savings = if total_baseline > 0 {
            ((total_baseline as f64 - total_rlm as f64) / total_baseline as f64) * 100.0
        } else {
            0.0
        };

        println!("Average Iterations: {:.1}", avg_iters);
        println!("Average Latency: {:?}", avg_lat);
        println!("\n--- Token Usage (RLM vs Baseline) ---");
        println!("Total RLM Tokens:      {:>8}", total_rlm);
        println!("Total Baseline Tokens: {:>8} (estimated)", total_baseline);
        println!("Overall Token Savings: {:>7.1}%", overall_savings);
    }

    // Exit with error code if tests failed
    if total_passed < total_queries {
        std::process::exit(1);
    }

    Ok(())
}
