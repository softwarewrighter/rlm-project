//! RLM CLI - Query large files with small LLMs
//!
//! Usage:
//!   rlm <file> <query> [--model <model>] [--ollama-url <url>] [--verbose]
//!
//! Example:
//!   rlm war-and-peace.txt "What is the secret passphrase?" --verbose
//!   rlm large-log.txt "How many ERROR lines?" --model llama3.2:3b

use anyhow::{Context, Result};
use rlm::orchestrator::RlmOrchestrator;
use rlm::pool::{LlmPool, LoadBalanceStrategy, ProviderRole};
use rlm::provider::OllamaProvider;
use rlm::{ProviderConfig, RlmConfig};
use std::path::PathBuf;
use std::sync::Arc;

const DEFAULT_MODEL: &str = "llama3.2:3b";
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

// ANSI color codes
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const BLUE: &str = "\x1b[34m";
const MAGENTA: &str = "\x1b[35m";
const CYAN: &str = "\x1b[36m";

fn print_usage() {
    eprintln!(
        r#"
{BOLD}RLM CLI{RESET} - Query large files with small LLMs using Recursive Language Models

{BOLD}USAGE:{RESET}
    rlm <FILE> <QUERY> [OPTIONS]

{BOLD}ARGS:{RESET}
    <FILE>     Path to the file to analyze
    <QUERY>    Question to ask about the file

{BOLD}OPTIONS:{RESET}
    -m, --model <MODEL>         Ollama model to use (default: llama3.2:3b)
    -u, --ollama-url <URL>      Ollama server URL (default: http://localhost:11434)
    -n, --max-iterations <N>    Maximum RLM iterations (default: 20)
    -v, --verbose               Show detailed iteration info
    -vv                         Extra verbose (show full LLM commands)
    --dry-run                   Show what would be done without executing
    -h, --help                  Print this help message

{BOLD}EXAMPLES:{RESET}
    rlm document.txt "What is the main topic?"
    rlm logs.txt "Count the ERROR lines" -m phi3:3.8b
    rlm war-and-peace.txt "Find the hidden passphrase" -vv

{BOLD}HOW IT WORKS:{RESET}
    RLM enables small LLMs to analyze documents much larger than their
    context window by iteratively exploring the content using commands
    like 'find', 'slice', 'lines', and 'count' instead of reading everything.
"#
    );
}

struct CliArgs {
    file: PathBuf,
    query: String,
    model: String,
    ollama_url: String,
    max_iterations: usize,
    verbose: u8, // 0=off, 1=verbose, 2=extra verbose
    dry_run: bool,
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
    let mut max_iterations = 20;
    let mut verbose: u8 = 0;
    let mut dry_run = false;

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
            _ => {}
        }
        i += 1;
    }

    Ok(CliArgs {
        file,
        query,
        model,
        ollama_url,
        max_iterations,
        verbose,
        dry_run,
    })
}

fn print_header(args: &CliArgs, file_size: usize, line_count: usize) {
    eprintln!();
    eprintln!("{BLUE}╭──────────────────────────────────────────────────────────────╮{RESET}");
    eprintln!("{BLUE}│{RESET}  {BOLD}RLM CLI{RESET} - Recursive Language Model Query                   {BLUE}│{RESET}");
    eprintln!("{BLUE}├──────────────────────────────────────────────────────────────┤{RESET}");
    eprintln!(
        "{BLUE}│{RESET}  {DIM}File:{RESET}   {}",
        args.file.display()
    );
    eprintln!(
        "{BLUE}│{RESET}  {DIM}Size:{RESET}   {} chars ({} lines, ~{} tokens)",
        file_size,
        line_count,
        file_size / 4
    );
    eprintln!("{BLUE}│{RESET}  {DIM}Model:{RESET}  {}", args.model);
    eprintln!(
        "{BLUE}│{RESET}  {DIM}Query:{RESET}  {}",
        if args.query.len() > 50 {
            format!("{}...", &args.query[..47])
        } else {
            args.query.clone()
        }
    );
    eprintln!("{BLUE}╰──────────────────────────────────────────────────────────────╯{RESET}");
    eprintln!();
}

fn print_iteration(step: usize, llm_response: &str, commands: &str, output: &str, verbose: u8) {
    eprintln!("{CYAN}┌─ Iteration {step} ─────────────────────────────────────────────────{RESET}");

    // At -vv level, show the full LLM response
    if verbose >= 2 && !llm_response.is_empty() {
        eprintln!("{CYAN}│{RESET}");
        eprintln!("{CYAN}│{RESET} {BLUE}▼ LLM Response:{RESET}");
        // Show LLM response with green color, truncated if very long
        let response_preview = if llm_response.len() > 500 {
            format!(
                "{}...\n{DIM}({} chars total){RESET}",
                &llm_response[..497],
                llm_response.len()
            )
        } else {
            llm_response.to_string()
        };
        for line in response_preview.lines() {
            eprintln!("{CYAN}│{RESET}   {GREEN}{line}{RESET}");
        }
        eprintln!("{CYAN}│{RESET}");
    }

    // Show the JSON command(s)
    if !commands.is_empty() && commands != "(direct)" {
        eprintln!("{CYAN}│{RESET} {YELLOW}▶ Command(s):{RESET}");
        // Pretty print the JSON
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(commands) {
            if let Ok(pretty) = serde_json::to_string_pretty(&json) {
                for line in pretty.lines() {
                    eprintln!("{CYAN}│{RESET}   {YELLOW}{line}{RESET}");
                }
            } else {
                eprintln!("{CYAN}│{RESET}   {YELLOW}{commands}{RESET}");
            }
        } else {
            // Not valid JSON, show raw
            eprintln!("{CYAN}│{RESET}   {YELLOW}{commands}{RESET}");
        }
    } else if !commands.is_empty() {
        eprintln!("{CYAN}│{RESET} {YELLOW}▶ Command:{RESET} {commands}");
    }

    // Show output (truncated) with cyan color
    if !output.is_empty() {
        if verbose >= 2 || output.len() < 100 {
            let output_preview = if output.len() > 300 {
                format!(
                    "{}...\n{DIM}({} chars total){RESET}",
                    &output[..297],
                    output.len()
                )
            } else {
                output.to_string()
            };
            eprintln!("{CYAN}│{RESET} {MAGENTA}◀ Output:{RESET}");
            for line in output_preview.lines() {
                eprintln!("{CYAN}│{RESET}   {CYAN}{line}{RESET}");
            }
        } else {
            eprintln!(
                "{CYAN}│{RESET} {MAGENTA}◀ Output:{RESET} {DIM}{} chars{RESET}",
                output.len()
            );
        }
    }

    eprintln!("{CYAN}└────────────────────────────────────────────────────────────────{RESET}");
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
    eprintln!("{GREEN}╭──────────────────────────────────────────────────────────────╮{RESET}");
    eprintln!("{GREEN}│{RESET}  {BOLD}Results{RESET}                                                     {GREEN}│{RESET}");
    eprintln!("{GREEN}├──────────────────────────────────────────────────────────────┤{RESET}");
    eprintln!("{GREEN}│{RESET}  {DIM}Iterations:{RESET}     {iterations}");
    eprintln!("{GREEN}│{RESET}  {DIM}Sub-LM calls:{RESET}   {sub_calls}");
    eprintln!(
        "{GREEN}│{RESET}  {DIM}Tokens used:{RESET}    {} prompt + {} completion",
        prompt_tokens, completion_tokens
    );

    // Calculate token savings
    let baseline_tokens = context_chars / 4; // rough estimate
    let actual_tokens = (prompt_tokens + completion_tokens) as usize;
    if baseline_tokens > actual_tokens {
        let savings = ((baseline_tokens - actual_tokens) as f64 / baseline_tokens as f64) * 100.0;
        eprintln!(
            "{GREEN}│{RESET}  {DIM}Token savings:{RESET}  {BOLD}{:.0}%{RESET} vs direct approach",
            savings
        );
    }
    eprintln!("{GREEN}╰──────────────────────────────────────────────────────────────╯{RESET}");
    eprintln!();
    eprintln!("{BOLD}Answer:{RESET}");
    eprintln!("{GREEN}════════════════════════════════════════════════════════════════{RESET}");
    println!("{answer}");
    eprintln!("{GREEN}════════════════════════════════════════════════════════════════{RESET}");
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
        eprintln!("{YELLOW}DRY RUN MODE - No LLM calls will be made{RESET}");
        eprintln!();
        eprintln!("{DIM}Would perform the following:{RESET}");
        eprintln!("  1. Connect to Ollama at {}", args.ollama_url);
        eprintln!("  2. Load model: {}", args.model);
        eprintln!(
            "  3. Send document ({} chars) to RLM orchestrator",
            file_size
        );
        eprintln!("  4. Execute up to {} iterations", args.max_iterations);
        eprintln!("  5. Return answer");
        eprintln!();
        eprintln!("{DIM}Available RLM commands:{RESET}");
        eprintln!("  - {YELLOW}find{RESET}      Search for text");
        eprintln!("  - {YELLOW}regex{RESET}     Pattern matching");
        eprintln!("  - {GREEN}slice{RESET}     Extract character range");
        eprintln!("  - {GREEN}lines{RESET}     Extract line range");
        eprintln!("  - {MAGENTA}count{RESET}     Count lines/chars/matches");
        eprintln!("  - {CYAN}llm_query{RESET} Delegate to sub-LLM");
        eprintln!("  - {BOLD}final{RESET}     Return answer");
        return Ok(());
    }

    // Create config
    let config = RlmConfig {
        max_iterations: args.max_iterations,
        max_sub_calls: 50,
        output_limit: 10000,
        bypass_enabled: false, // Always use RLM for demo
        bypass_threshold: 0,
        providers: vec![ProviderConfig {
            provider_type: "ollama".to_string(),
            base_url: args.ollama_url.clone(),
            model: args.model.clone(),
            api_key: None,
            weight: 1,
            role: "root".to_string(),
        }],
    };

    // Create pool with Ollama provider
    let mut pool = LlmPool::new(LoadBalanceStrategy::RoundRobin);
    let provider = OllamaProvider::new(&args.ollama_url, &args.model);
    pool.add_provider(Arc::new(provider), 1, ProviderRole::Both);
    let pool = Arc::new(pool);

    // Create orchestrator
    let orchestrator = RlmOrchestrator::new(config, pool);

    if args.verbose > 0 {
        eprintln!("{DIM}Starting RLM processing...{RESET}");
        eprintln!();
    }

    // Process query
    let result = orchestrator.process(&args.query, &context).await;

    match result {
        Ok(rlm_result) => {
            // Show iteration history if verbose
            if args.verbose > 0 {
                for record in &rlm_result.history {
                    print_iteration(
                        record.step,
                        &record.llm_response,
                        &record.commands,
                        &record.output,
                        args.verbose,
                    );
                }
            }

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
            eprintln!("{BOLD}\x1b[31mError:{RESET} {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
