//! Multi-Source RLM Experiment
//!
//! Demonstrates RLM's value for large context analysis by:
//! 1. Fetching documentation from multiple public sources (Rust crates)
//! 2. Combining into a single large context
//! 3. Running queries through RLM to show token savings
//!
//! Use case: Compare Rust HTTP client crates (reqwest, ureq, hyper, surf)
//!
//! All sources are public APIs (crates.io) - robots.txt compliant.

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const RLM_SERVER: &str = "http://localhost:4539";
const CACHE_DIR: &str = "cache";

#[derive(Debug, Deserialize)]
struct CratesResponse {
    #[serde(rename = "crate")]
    crate_info: CrateInfo,
}

#[derive(Debug, Deserialize)]
struct CrateInfo {
    #[serde(rename = "name")]
    _name: String,
    downloads: u64,
    recent_downloads: Option<u64>,
    max_version: String,
    description: Option<String>,
    repository: Option<String>,
    #[serde(rename = "documentation")]
    _documentation: Option<String>,
}

#[derive(Debug, Serialize)]
struct RlmRequest {
    query: String,
    context: String,
}

#[derive(Debug, Deserialize)]
struct RlmResponse {
    answer: String,
    iterations: u32,
    total_prompt_tokens: Option<u32>,
    total_completion_tokens: Option<u32>,
    bypassed: Option<bool>,
    #[serde(rename = "context_chars")]
    _context_chars: Option<usize>,
}

struct Source {
    name: &'static str,
    crates_io_url: &'static str,
    readme_url: &'static str,
    description: &'static str,
}

const SOURCES: &[Source] = &[
    Source {
        name: "reqwest",
        crates_io_url: "https://crates.io/api/v1/crates/reqwest",
        readme_url: "https://raw.githubusercontent.com/seanmonstar/reqwest/master/README.md",
        description: "Popular async HTTP client",
    },
    Source {
        name: "ureq",
        crates_io_url: "https://crates.io/api/v1/crates/ureq",
        readme_url: "https://raw.githubusercontent.com/algesten/ureq/main/README.md",
        description: "Minimal sync HTTP client",
    },
    Source {
        name: "hyper",
        crates_io_url: "https://crates.io/api/v1/crates/hyper",
        readme_url: "https://raw.githubusercontent.com/hyperium/hyper/master/README.md",
        description: "Low-level HTTP library",
    },
    Source {
        name: "surf",
        crates_io_url: "https://crates.io/api/v1/crates/surf",
        readme_url: "https://raw.githubusercontent.com/http-rs/surf/main/README.md",
        description: "Async HTTP client (async-std)",
    },
    Source {
        name: "attohttpc",
        crates_io_url: "https://crates.io/api/v1/crates/attohttpc",
        readme_url: "https://raw.githubusercontent.com/sbstp/attohttpc/master/README.md",
        description: "Tiny blocking HTTP client",
    },
    Source {
        name: "isahc",
        crates_io_url: "https://crates.io/api/v1/crates/isahc",
        readme_url: "https://raw.githubusercontent.com/sagebind/isahc/master/README.md",
        description: "Practical HTTP client (curl-based)",
    },
    Source {
        name: "minreq",
        crates_io_url: "https://crates.io/api/v1/crates/minreq",
        readme_url: "https://raw.githubusercontent.com/neonmoe/minreq/master/README.md",
        description: "Minimal HTTP client",
    },
    Source {
        name: "awc",
        crates_io_url: "https://crates.io/api/v1/crates/awc",
        readme_url: "https://raw.githubusercontent.com/actix/actix-web/master/awc/README.md",
        description: "Actix Web Client",
    },
];

struct Question {
    query: &'static str,
    expected_keywords: &'static [&'static str],
}

const QUESTIONS: &[Question] = &[
    Question {
        query: "Which of these HTTP clients support async/await? List them.",
        expected_keywords: &["reqwest", "hyper", "surf"],
    },
    Question {
        query: "Which crate is best for simple, blocking HTTP requests with minimal dependencies?",
        expected_keywords: &["ureq"],
    },
    Question {
        query: "Compare the download counts of these crates. Which is most popular?",
        expected_keywords: &["reqwest", "download"],
    },
    Question {
        query: "Which crate would you recommend for a CLI tool that needs to make a few HTTP requests and wants minimal compile time?",
        expected_keywords: &["ureq", "minimal"],
    },
];

fn get_cache_path(url: &str) -> PathBuf {
    let hash = format!("{:x}", md5::compute(url.as_bytes()));
    PathBuf::from(CACHE_DIR).join(format!("{}.txt", hash))
}

fn fetch_url(client: &Client, url: &str, use_cache: bool) -> Result<String, String> {
    let cache_path = get_cache_path(url);

    if use_cache && cache_path.exists() {
        println!("  [CACHE] {}...", &url[..url.len().min(60)]);
        return fs::read_to_string(&cache_path).map_err(|e| e.to_string());
    }

    println!("  [FETCH] {}...", &url[..url.len().min(60)]);

    let response = client
        .get(url)
        .header("User-Agent", "RLM-Experiment/1.0 (research)")
        .timeout(Duration::from_secs(30))
        .send()
        .map_err(|e| e.to_string())?;

    let content = response.text().map_err(|e| e.to_string())?;

    // Cache the result
    if let Some(parent) = cache_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(&cache_path, &content);

    Ok(content)
}

fn build_context(client: &Client) -> Result<String, String> {
    let mut sections = Vec::new();

    for source in SOURCES {
        let mut section = format!(
            "\n{}\nCRATE: {}\nDescription: {}\n{}\n\n",
            "=".repeat(60),
            source.name,
            source.description,
            "=".repeat(60)
        );

        // Fetch crates.io metadata
        if let Ok(content) = fetch_url(client, source.crates_io_url, true)
            && let Ok(data) = serde_json::from_str::<CratesResponse>(&content)
        {
            let c = &data.crate_info;
            section.push_str("## Crates.io Metadata\n");
            section.push_str(&format!("- Downloads: {}\n", c.downloads));
            if let Some(recent) = c.recent_downloads {
                section.push_str(&format!("- Recent Downloads: {}\n", recent));
            }
            section.push_str(&format!("- Max Version: {}\n", c.max_version));
            if let Some(desc) = &c.description {
                section.push_str(&format!("- Description: {}\n", desc));
            }
            if let Some(repo) = &c.repository {
                section.push_str(&format!("- Repository: {}\n", repo));
            }
            section.push('\n');
        }

        // Fetch README (include full content for larger context)
        if let Ok(content) = fetch_url(client, source.readme_url, true) {
            section.push_str(&format!("## README\n{}\n\n", content));
        }

        sections.push(section);
    }

    Ok(sections.join("\n"))
}

fn query_rlm(
    client: &Client,
    query: &str,
    context: &str,
) -> Result<(RlmResponse, Duration), String> {
    let request = RlmRequest {
        query: query.to_string(),
        context: context.to_string(),
    };

    let start = Instant::now();

    let response = client
        .post(format!("{}/debug", RLM_SERVER))
        .json(&request)
        .timeout(Duration::from_secs(300))
        .send()
        .map_err(|e| format!("Request failed: {}", e))?;

    let elapsed = start.elapsed();

    if !response.status().is_success() {
        return Err(format!("Server error: {}", response.status()));
    }

    let result: RlmResponse = response.json().map_err(|e| format!("Parse error: {}", e))?;

    Ok((result, elapsed))
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("Multi-Source RLM Experiment (Rust)");
    println!("Comparing Rust HTTP Client Crates");
    println!("{}", "=".repeat(70));
    println!();

    // Create cache directory
    let _ = fs::create_dir_all(CACHE_DIR);

    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client");

    // Step 1: Build context
    println!("Step 1: Fetching source data...");
    let context = match build_context(&client) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to build context: {}", e);
            return;
        }
    };

    let context_chars = context.len();
    let context_tokens = context_chars / 4; // rough estimate

    println!();
    println!(
        "Context built: {} chars (~{} tokens)",
        context_chars, context_tokens
    );
    println!();

    // Save context for inspection
    let context_file = PathBuf::from(CACHE_DIR).join("combined_context.txt");
    let _ = fs::write(&context_file, &context);
    println!("Context saved to: {}", context_file.display());
    println!();

    // Step 2: Check RLM server
    println!("Step 2: Checking RLM server...");
    match client.get(format!("{}/health", RLM_SERVER)).send() {
        Ok(resp) if resp.status().is_success() => {
            println!("  Server status: healthy");
        }
        _ => {
            eprintln!("  ERROR: RLM server not running at {}", RLM_SERVER);
            eprintln!("  Start it with: ./scripts/run-server.sh");
            return;
        }
    }
    println!();

    // Step 3: Run queries
    println!("Step 3: Running queries...");
    println!();

    let mut passed = 0;
    let mut total_savings = 0.0;

    for (i, q) in QUESTIONS.iter().enumerate() {
        print!("Query {}: {}... ", i + 1, &q.query[..q.query.len().min(55)]);

        match query_rlm(&client, q.query, &context) {
            Ok((result, latency)) => {
                let answer_lower = result.answer.to_lowercase();
                let found: Vec<_> = q
                    .expected_keywords
                    .iter()
                    .filter(|kw| answer_lower.contains(&kw.to_lowercase()))
                    .collect();

                let test_passed = !found.is_empty();
                if test_passed {
                    passed += 1;
                }

                let rlm_tokens = result.total_prompt_tokens.unwrap_or(0)
                    + result.total_completion_tokens.unwrap_or(0);
                let baseline_tokens = context_tokens + 100;
                let savings = if baseline_tokens > 0 {
                    ((baseline_tokens as f64 - rlm_tokens as f64) / baseline_tokens as f64) * 100.0
                } else {
                    0.0
                };
                total_savings += savings;

                let bypass_str = if result.bypassed.unwrap_or(false) {
                    " [BYPASS]"
                } else {
                    ""
                };

                println!();
                println!(
                    "  {} ({} iters, {:.1}s, {} tokens, {:.0}% saved){}",
                    if test_passed { "PASS" } else { "FAIL" },
                    result.iterations,
                    latency.as_secs_f64(),
                    rlm_tokens,
                    savings,
                    bypass_str
                );
                println!(
                    "  Answer: {}...",
                    &result.answer[..result.answer.len().min(80)]
                );
                println!();
            }
            Err(e) => {
                println!();
                println!("  ERROR: {}", e);
                println!();
            }
        }
    }

    // Summary
    println!("{}", "=".repeat(70));
    println!("SUMMARY");
    println!("{}", "=".repeat(70));
    println!(
        "Context size: {} chars (~{} tokens)",
        context_chars, context_tokens
    );
    println!("Queries: {}/{} passed", passed, QUESTIONS.len());
    println!(
        "Average token savings: {:.0}%",
        total_savings / QUESTIONS.len() as f64
    );
    println!();
    println!("This experiment demonstrates RLM's ability to efficiently answer");
    println!("questions about large multi-source contexts without sending the");
    println!("entire context to the LLM for each query.");
}
