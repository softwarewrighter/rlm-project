//! RLM CLI - Query large files with small LLMs
//!
//! Usage:
//!   rlm <file> <query> [--model <model>] [--ollama-url <url>] [--verbose]
//!
//! Example:
//!   rlm war-and-peace.txt "What is the secret passphrase?" --verbose
//!   rlm large-log.txt "How many ERROR lines?" --model llama3.2:3b

use anyhow::{Context, Result};
use colored::Colorize;
use rlm::orchestrator::RlmOrchestrator;
use rlm::pool::{LlmPool, LoadBalanceStrategy, ProviderRole};
use rlm::provider::{LiteLLMProvider, OllamaProvider};
use rlm::{ProgressCallback, ProgressEvent, ProviderConfig, RlmConfig};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

const DEFAULT_MODEL: &str = "llama3.2:3b";
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const DEFAULT_LITELLM_URL: &str = "http://localhost:4000";

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

/// Create a progress callback for real-time output based on verbosity level
fn create_progress_callback(verbose: u8) -> ProgressCallback {
    Box::new(move |event: ProgressEvent| {
        let mut stderr = std::io::stderr();
        match event {
            ProgressEvent::IterationStart { step } => {
                // Always show iteration start (minimal output)
                eprintln!("{}", format!("┌─ Iteration {} ", step).cyan());
                let _ = stderr.flush();
            }
            ProgressEvent::LlmCallStart { step: _ } => {
                if verbose >= 1 {
                    eprint!("{}", "│ ⏳ Calling LLM...".dimmed());
                    let _ = stderr.flush();
                }
            }
            ProgressEvent::LlmCallComplete {
                step: _,
                duration_ms,
                response_preview,
            } => {
                if verbose >= 1 {
                    // Clear the "Calling LLM..." line and show result
                    eprint!("\r{}", " ".repeat(30)); // Clear line
                    eprintln!(
                        "\r{} {}",
                        "│".cyan(),
                        format!("⏱  LLM: {}ms", duration_ms).dimmed()
                    );
                    if verbose >= 2 {
                        eprintln!("{} {}", "│".cyan(), "▼ Response preview:".blue());
                        let preview = truncate_to_char_boundary(&response_preview, 200);
                        for line in preview.lines().take(3) {
                            eprintln!("{}   {}", "│".cyan(), line.green());
                        }
                    }
                    let _ = stderr.flush();
                }
            }
            ProgressEvent::CommandsExtracted { step: _, commands } => {
                if verbose >= 2 {
                    eprintln!("{} {}", "│".cyan(), "▶ Commands:".yellow());
                    let preview = truncate_to_char_boundary(&commands, 150);
                    eprintln!("{}   {}", "│".cyan(), preview.yellow());
                    let _ = stderr.flush();
                }
            }
            ProgressEvent::WasmCompileStart { step: _ } => {
                if verbose >= 1 {
                    eprint!("{} {}", "│".cyan(), "🔧 Compiling WASM...".dimmed());
                    let _ = stderr.flush();
                }
            }
            ProgressEvent::WasmCompileComplete {
                step: _,
                duration_ms,
            } => {
                if verbose >= 1 {
                    eprintln!(" {}", format!("done ({}ms)", duration_ms).green());
                    let _ = stderr.flush();
                }
            }
            ProgressEvent::CommandComplete {
                step: _,
                output_preview,
                exec_ms,
            } => {
                if verbose >= 1 {
                    eprintln!(
                        "{} {} {}",
                        "│".cyan(),
                        "◀ Exec:".magenta(),
                        format!("{}ms", exec_ms).dimmed()
                    );
                    if verbose >= 2 && !output_preview.is_empty() {
                        let preview = truncate_to_char_boundary(&output_preview, 150);
                        eprintln!("{}   {}", "│".cyan(), preview.cyan());
                    }
                    let _ = stderr.flush();
                }
            }
            ProgressEvent::IterationComplete { step: _, record: _ } => {
                eprintln!("{}", "└────────────────────────────────────────".cyan());
                let _ = stderr.flush();
            }
            ProgressEvent::FinalAnswer { answer } => {
                if verbose >= 1 {
                    let preview = truncate_to_char_boundary(&answer, 100);
                    eprintln!("{} {}", "✓ Final:".green().bold(), preview);
                    let _ = stderr.flush();
                }
            }
            ProgressEvent::Complete {
                iterations,
                success,
            } => {
                if success {
                    eprintln!(
                        "{}",
                        format!("Completed in {} iteration(s)", iterations)
                            .green()
                            .dimmed()
                    );
                } else {
                    eprintln!(
                        "{}",
                        format!("Stopped after {} iteration(s)", iterations)
                            .yellow()
                            .dimmed()
                    );
                }
                let _ = stderr.flush();
            }
        }
    })
}

fn print_usage() {
    eprintln!(
        r#"
{} - Query large files with small LLMs using Recursive Language Models

{}
    rlm <FILE> <QUERY> [OPTIONS]

{}
    <FILE>     Path to the file to analyze
    <QUERY>    Question to ask about the file

{}
    -m, --model <MODEL>         Model to use (default: llama3.2:3b for Ollama)
    -u, --ollama-url <URL>      Ollama server URL (default: http://localhost:11434)
    --litellm                   Use LiteLLM gateway instead of Ollama
    --litellm-url <URL>         LiteLLM server URL (default: http://localhost:4000)
    --litellm-key <KEY>         LiteLLM API key (or set LITELLM_MASTER_KEY env var)
    -n, --max-iterations <N>    Maximum RLM iterations (default: 20)
    -v, --verbose               Show detailed iteration info
    -vv                         Extra verbose (show full LLM commands)
    --dry-run                   Show what would be done without executing
    --no-wasm                   Disable all WASM features
    --no-rust-wasm              Disable rust_wasm command (keep pre-compiled WASM)
    -h, --help                  Print this help message

{}
    rlm document.txt "What is the main topic?"
    rlm logs.txt "Count the ERROR lines" -m phi3:3.8b
    rlm war-and-peace.txt "Find the hidden passphrase" -vv
    rlm data.log "Analyze errors" --litellm -m deepseek-coder

{}
    RLM enables small LLMs to analyze documents much larger than their
    context window by iteratively exploring the content using commands
    like 'find', 'slice', 'lines', and 'count' instead of reading everything.
"#,
        "RLM CLI".bold(),
        "USAGE:".bold(),
        "ARGS:".bold(),
        "OPTIONS:".bold(),
        "EXAMPLES:".bold(),
        "HOW IT WORKS:".bold(),
    );
}

struct CliArgs {
    file: PathBuf,
    query: String,
    model: String,
    ollama_url: String,
    use_litellm: bool,
    litellm_url: String,
    litellm_key: Option<String>,
    max_iterations: usize,
    verbose: u8, // 0=off, 1=verbose, 2=extra verbose
    dry_run: bool,
    wasm_enabled: bool,
    rust_wasm_enabled: bool,
}

fn parse_args() -> Result<CliArgs> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 || args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        print_usage();
        std::process::exit(if args.iter().any(|a| a == "--help" || a == "-h") {
            0
        } else {
            1
        });
    }

    let file = PathBuf::from(&args[1]);
    let query = args[2].clone();

    let mut model = DEFAULT_MODEL.to_string();
    let mut ollama_url = DEFAULT_OLLAMA_URL.to_string();
    let mut use_litellm = false;
    let mut litellm_url = DEFAULT_LITELLM_URL.to_string();
    let mut litellm_key: Option<String> = std::env::var("LITELLM_MASTER_KEY").ok();
    let mut max_iterations = 20;
    let mut verbose: u8 = 0;
    let mut dry_run = false;
    let mut wasm_enabled = true;
    let mut rust_wasm_enabled = true;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                if i < args.len() {
                    model = args[i].clone();
                }
            }
            "--ollama-url" | "-u" => {
                i += 1;
                if i < args.len() {
                    ollama_url = args[i].clone();
                }
            }
            "--max-iterations" | "-n" => {
                i += 1;
                if i < args.len() {
                    max_iterations = args[i].parse().unwrap_or(20);
                }
            }
            "--verbose" | "-v" => {
                verbose = verbose.max(1);
            }
            "-vv" => {
                verbose = 2;
            }
            "--dry-run" => {
                dry_run = true;
            }
            "--no-wasm" => {
                wasm_enabled = false;
                rust_wasm_enabled = false;
            }
            "--no-rust-wasm" => {
                rust_wasm_enabled = false;
            }
            "--litellm" => {
                use_litellm = true;
            }
            "--litellm-url" => {
                i += 1;
                if i < args.len() {
                    litellm_url = args[i].clone();
                }
            }
            "--litellm-key" => {
                i += 1;
                if i < args.len() {
                    litellm_key = Some(args[i].clone());
                }
            }
            _ => {}
        }
        i += 1;
    }

    Ok(CliArgs {
        file,
        query,
        model,
        ollama_url,
        use_litellm,
        litellm_url,
        litellm_key,
        max_iterations,
        verbose,
        dry_run,
        wasm_enabled,
        rust_wasm_enabled,
    })
}

fn print_header(args: &CliArgs, file_size: usize, line_count: usize) {
    eprintln!();
    eprintln!(
        "{}",
        "╭──────────────────────────────────────────────────────────────╮".blue()
    );
    eprintln!(
        "{}  {}                   {}",
        "│".blue(),
        "RLM CLI - Recursive Language Model Query".bold(),
        "│".blue()
    );
    eprintln!(
        "{}",
        "├──────────────────────────────────────────────────────────────┤".blue()
    );
    eprintln!(
        "{}  {}   {}",
        "│".blue(),
        "File:".dimmed(),
        args.file.display()
    );
    eprintln!(
        "{}  {}   {} chars ({} lines, ~{} tokens)",
        "│".blue(),
        "Size:".dimmed(),
        file_size,
        line_count,
        file_size / 4
    );
    if args.use_litellm {
        eprintln!(
            "{}  {}  {} (via LiteLLM @ {})",
            "│".blue(),
            "Model:".dimmed(),
            args.model,
            args.litellm_url
        );
    } else {
        eprintln!(
            "{}  {}  {} (Ollama @ {})",
            "│".blue(),
            "Model:".dimmed(),
            args.model,
            args.ollama_url
        );
    }
    eprintln!(
        "{}  {}  {}",
        "│".blue(),
        "Query:".dimmed(),
        if args.query.chars().count() > 50 {
            format!("{}...", args.query.chars().take(47).collect::<String>())
        } else {
            args.query.clone()
        }
    );
    eprintln!(
        "{}",
        "╰──────────────────────────────────────────────────────────────╯".blue()
    );
    eprintln!();
}

fn print_results(
    answer: &str,
    iterations: usize,
    sub_calls: usize,
    prompt_tokens: u32,
    completion_tokens: u32,
    context_chars: usize,
) {
    eprintln!();
    eprintln!(
        "{}",
        "╭──────────────────────────────────────────────────────────────╮".green()
    );
    eprintln!(
        "{}  {}                                                     {}",
        "│".green(),
        "Results".bold(),
        "│".green()
    );
    eprintln!(
        "{}",
        "├──────────────────────────────────────────────────────────────┤".green()
    );
    eprintln!(
        "{}  {}     {}",
        "│".green(),
        "Iterations:".dimmed(),
        iterations
    );
    eprintln!(
        "{}  {}   {}",
        "│".green(),
        "Sub-LM calls:".dimmed(),
        sub_calls
    );
    eprintln!(
        "{}  {}    {} prompt + {} completion",
        "│".green(),
        "Tokens used:".dimmed(),
        prompt_tokens,
        completion_tokens
    );

    // Calculate token savings
    let baseline_tokens = context_chars / 4; // rough estimate
    let actual_tokens = (prompt_tokens + completion_tokens) as usize;
    if baseline_tokens > actual_tokens {
        let savings = ((baseline_tokens - actual_tokens) as f64 / baseline_tokens as f64) * 100.0;
        eprintln!(
            "{}  {}  {} vs direct approach",
            "│".green(),
            "Token savings:".dimmed(),
            format!("{:.0}%", savings).bold()
        );
    }
    eprintln!(
        "{}",
        "╰──────────────────────────────────────────────────────────────╯".green()
    );
    eprintln!();
    eprintln!("{}", "Answer:".bold());
    eprintln!(
        "{}",
        "════════════════════════════════════════════════════════════════".green()
    );
    println!("{answer}");
    eprintln!(
        "{}",
        "════════════════════════════════════════════════════════════════".green()
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;

    // Read the file
    let context = std::fs::read_to_string(&args.file)
        .with_context(|| format!("Failed to read file: {}", args.file.display()))?;

    let file_size = context.len();
    let line_count = context.lines().count();

    print_header(&args, file_size, line_count);

    // Dry run - just show what would happen
    if args.dry_run {
        eprintln!("{}", "DRY RUN MODE - No LLM calls will be made".yellow());
        eprintln!();
        eprintln!("{}", "Would perform the following:".dimmed());
        eprintln!("  1. Connect to Ollama at {}", args.ollama_url);
        eprintln!("  2. Load model: {}", args.model);
        eprintln!(
            "  3. Send document ({} chars) to RLM orchestrator",
            file_size
        );
        eprintln!("  4. Execute up to {} iterations", args.max_iterations);
        eprintln!("  5. Return answer");
        eprintln!();
        eprintln!("{}", "Available RLM commands:".dimmed());
        eprintln!("  - {}      Search for text", "find".yellow());
        eprintln!("  - {}     Pattern matching", "regex".yellow());
        eprintln!("  - {}     Extract character range", "slice".green());
        eprintln!("  - {}     Extract line range", "lines".green());
        eprintln!("  - {}     Count lines/chars/matches", "count".magenta());
        eprintln!("  - {} Delegate to sub-LLM", "llm_query".cyan());
        if args.wasm_enabled {
            eprintln!("  - {}      Execute pre-compiled WASM", "wasm".blue());
            if args.rust_wasm_enabled {
                eprintln!("  - {} Compile & execute Rust code", "rust_wasm".blue());
            }
        }
        eprintln!("  - {}     Return answer", "final".bold());
        eprintln!();
        eprintln!("{}", "WASM features:".dimmed());
        eprintln!(
            "  - WASM enabled: {}",
            if args.wasm_enabled { "yes" } else { "no" }
        );
        eprintln!(
            "  - rust_wasm enabled: {}",
            if args.rust_wasm_enabled { "yes" } else { "no" }
        );
        return Ok(());
    }

    // Create config
    let (provider_type, base_url, api_key) = if args.use_litellm {
        (
            "litellm".to_string(),
            args.litellm_url.clone(),
            args.litellm_key.clone(),
        )
    } else {
        ("ollama".to_string(), args.ollama_url.clone(), None)
    };

    let config = RlmConfig {
        max_iterations: args.max_iterations,
        max_sub_calls: 50,
        output_limit: 10000,
        bypass_enabled: false, // Always use RLM for demo
        bypass_threshold: 0,
        providers: vec![ProviderConfig {
            provider_type,
            base_url: base_url.clone(),
            model: args.model.clone(),
            api_key: api_key.clone(),
            weight: 1,
            role: "root".to_string(),
        }],
        wasm: rlm::WasmConfig {
            enabled: args.wasm_enabled,
            rust_wasm_enabled: args.rust_wasm_enabled,
            ..Default::default()
        },
    };

    // Create pool with appropriate provider
    let mut pool = LlmPool::new(LoadBalanceStrategy::RoundRobin);
    if args.use_litellm {
        let api_key = api_key.unwrap_or_else(|| {
            eprintln!(
                "{} No LiteLLM API key provided. Set LITELLM_MASTER_KEY or use --litellm-key",
                "Warning:".yellow()
            );
            "".to_string()
        });
        let provider = LiteLLMProvider::with_base_url(&base_url, &api_key, &args.model);
        pool.add_provider(Arc::new(provider), 1, ProviderRole::Both);
    } else {
        let provider = OllamaProvider::new(&args.ollama_url, &args.model);
        pool.add_provider(Arc::new(provider), 1, ProviderRole::Both);
    }
    let pool = Arc::new(pool);

    // Create orchestrator
    let orchestrator = RlmOrchestrator::new(config, pool);

    eprintln!("{}", "Starting RLM processing...".dimmed());
    eprintln!();
    let _ = std::io::stderr().flush();

    // Create progress callback for real-time output
    let progress_callback = Some(create_progress_callback(args.verbose));

    // Process query with real-time progress
    let result = orchestrator
        .process_with_progress(&args.query, &context, progress_callback)
        .await;

    match result {
        Ok(rlm_result) => {
            eprintln!(); // Blank line before results
            print_results(
                &rlm_result.answer,
                rlm_result.iterations,
                rlm_result.total_sub_calls,
                rlm_result.total_prompt_tokens,
                rlm_result.total_completion_tokens,
                rlm_result.context_chars,
            );
        }
        Err(e) => {
            eprintln!("{} {}", "Error:".red().bold(), e);
            std::process::exit(1);
        }
    }

    Ok(())
}
