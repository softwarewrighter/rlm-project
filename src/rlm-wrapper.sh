#!/usr/bin/env bash
# rlm-wrapper.sh - RLM wrapper for various LLM CLIs
#
# Provides RLM capabilities on top of:
# - Claude Code CLI
# - OpenCode (Z.ai) with DeepSeek/GLM
# - Direct Ollama
# - llama.cpp server
#
# Usage:
#   ./rlm-wrapper.sh --help
#   ./rlm-wrapper.sh --query "Find all bugs" --context ./src --cli claude
#   ./rlm-wrapper.sh --query "Summarize" --context doc.txt --cli opencode
#
# Environment variables:
#   OLLAMA_HOST     - Ollama server host (default: localhost)
#   OLLAMA_PORT     - Ollama server port (default: 11434)
#   OLLAMA_MODEL    - Ollama model (default: qwen2.5-coder:32b)
#   DEEPSEEK_API_KEY - DeepSeek API key
#   ANTHROPIC_API_KEY - Claude API key
#   OPENCODE_MODEL  - OpenCode model (default: deepseek-chat)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
OLLAMA_HOST="${OLLAMA_HOST:-localhost}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5-coder:32b}"
OPENCODE_MODEL="${OPENCODE_MODEL:-deepseek-chat}"
MAX_ITERATIONS="${RLM_MAX_ITERATIONS:-30}"
WORK_DIR=""

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Cleanup on exit
cleanup() {
    if [[ -n "$WORK_DIR" && -d "$WORK_DIR" ]]; then
        rm -rf "$WORK_DIR"
    fi
}
trap cleanup EXIT

# Print usage
usage() {
    cat << EOF
RLM Wrapper - Recursive Language Model for large context processing

USAGE:
    $(basename "$0") [OPTIONS]

OPTIONS:
    -q, --query TEXT        Query to answer (required)
    -c, --context PATH      Context file or directory (required)
    -C, --cli CLI           CLI to use: claude, opencode, ollama, llama-cpp (default: ollama)
    -m, --model MODEL       Model override for the CLI
    -s, --sub-model MODEL   Model for sub-LLM calls (default: same as main)
    -o, --output FILE       Output file for results (default: stdout)
    -v, --verbose           Verbose output
    -j, --json              Output as JSON
    -h, --help              Show this help

EXAMPLES:
    # Process a codebase with Claude Code
    $(basename "$0") -q "Find security issues" -c ./src -C claude

    # Summarize documents with OpenCode + DeepSeek
    $(basename "$0") -q "Summarize key points" -c docs/ -C opencode

    # Use local Ollama
    $(basename "$0") -q "Count function definitions" -c main.rs -C ollama

    # Use distributed Ollama servers
    OLLAMA_HOST=192.168.1.10 $(basename "$0") -q "Analyze" -c data.txt

ENVIRONMENT:
    OLLAMA_HOST       Ollama server hostname
    OLLAMA_PORT       Ollama server port
    OLLAMA_MODEL      Default Ollama model
    DEEPSEEK_API_KEY  DeepSeek API key (for opencode)
    ANTHROPIC_API_KEY Claude API key (for claude cli)
    OPENCODE_MODEL    Default OpenCode model
EOF
}

# Build the RLM system prompt
build_rlm_system_prompt() {
    local context_len="$1"
    local context_type="${2:-text}"
    
    cat << 'PROMPT'
You are an RLM (Recursive Language Model) agent for processing large contexts.

Your context is available as a variable. Use code to examine and process it.

AVAILABLE IN REPL:
1. `context` - the full input text
2. `llm_query(prompt)` - call a sub-LLM for semantic analysis
3. Python standard library (re, json, etc.)

STRATEGY:
1. Probe context structure first (length, format, sample)
2. Chunk/filter based on content
3. Use llm_query() for semantic tasks
4. Use code for syntactic tasks (counting, regex)
5. Return FINAL(answer) when done

Write code in ```repl blocks. Store results in variables.
PROMPT
}

# Gather context from file or directory
gather_context() {
    local path="$1"
    local output_file="$2"
    
    if [[ -f "$path" ]]; then
        cat "$path" > "$output_file"
    elif [[ -d "$path" ]]; then
        # Gather all text files from directory
        find "$path" -type f \( \
            -name "*.py" -o -name "*.rs" -o -name "*.js" -o -name "*.ts" \
            -o -name "*.go" -o -name "*.java" -o -name "*.c" -o -name "*.h" \
            -o -name "*.cpp" -o -name "*.hpp" -o -name "*.md" -o -name "*.txt" \
            -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.toml" \
            -o -name "*.sh" -o -name "*.el" -o -name "*.lisp" \
        \) -print0 | while IFS= read -r -d '' file; do
            echo "=== FILE: $file ===" >> "$output_file"
            cat "$file" >> "$output_file"
            echo "" >> "$output_file"
        done
    else
        log_error "Path not found: $path"
        return 1
    fi
}

# Query Ollama directly
query_ollama() {
    local prompt="$1"
    local system="${2:-}"
    local model="${3:-$OLLAMA_MODEL}"
    
    local payload
    if [[ -n "$system" ]]; then
        payload=$(jq -n \
            --arg model "$model" \
            --arg prompt "$prompt" \
            --arg system "$system" \
            '{model: $model, prompt: $prompt, system: $system, stream: false}')
    else
        payload=$(jq -n \
            --arg model "$model" \
            --arg prompt "$prompt" \
            '{model: $model, prompt: $prompt, stream: false}')
    fi
    
    curl -s "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/generate" \
        -H "Content-Type: application/json" \
        -d "$payload" | jq -r '.response'
}

# Query DeepSeek API
query_deepseek() {
    local prompt="$1"
    local system="${2:-}"
    local model="${3:-deepseek-chat}"
    
    local messages
    if [[ -n "$system" ]]; then
        messages=$(jq -n \
            --arg system "$system" \
            --arg prompt "$prompt" \
            '[{role: "system", content: $system}, {role: "user", content: $prompt}]')
    else
        messages=$(jq -n \
            --arg prompt "$prompt" \
            '[{role: "user", content: $prompt}]')
    fi
    
    curl -s "https://api.deepseek.com/chat/completions" \
        -H "Authorization: Bearer ${DEEPSEEK_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg model "$model" --argjson messages "$messages" \
            '{model: $model, messages: $messages}')" \
        | jq -r '.choices[0].message.content'
}

# Query llama.cpp server (OpenAI-compatible)
query_llama_cpp() {
    local prompt="$1"
    local system="${2:-}"
    local server="${LLAMA_CPP_SERVER:-http://localhost:8080}"
    
    local messages
    if [[ -n "$system" ]]; then
        messages=$(jq -n \
            --arg system "$system" \
            --arg prompt "$prompt" \
            '[{role: "system", content: $system}, {role: "user", content: $prompt}]')
    else
        messages=$(jq -n \
            --arg prompt "$prompt" \
            '[{role: "user", content: $prompt}]')
    fi
    
    curl -s "${server}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --argjson messages "$messages" '{messages: $messages}')" \
        | jq -r '.choices[0].message.content'
}

# Run RLM loop with Claude Code CLI
run_claude_rlm() {
    local query="$1"
    local context_file="$2"
    local verbose="${3:-false}"
    
    local context
    context=$(cat "$context_file")
    local context_len=${#context}
    
    local system_prompt
    system_prompt=$(build_rlm_system_prompt "$context_len")
    
    # Create a temporary file with the full prompt
    local prompt_file
    prompt_file=$(mktemp)
    
    cat > "$prompt_file" << EOF
$system_prompt

CONTEXT (${context_len} chars):
The context is stored in a file. Use the computer tool to read and process it.
Context file: $context_file

QUERY: $query

Begin by examining the context file, then process it step by step.
When you have the answer, respond with FINAL(your_answer).
EOF
    
    # Run Claude Code with the prompt
    if command -v claude &> /dev/null; then
        claude --print < "$prompt_file"
    else
        log_error "claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
        rm "$prompt_file"
        return 1
    fi
    
    rm "$prompt_file"
}

# Run RLM loop with OpenCode
run_opencode_rlm() {
    local query="$1"
    local context_file="$2"
    local verbose="${3:-false}"
    
    local context
    context=$(cat "$context_file")
    local context_len=${#context}
    
    local system_prompt
    system_prompt=$(build_rlm_system_prompt "$context_len")
    
    # OpenCode integration
    if command -v opencode &> /dev/null; then
        # OpenCode with the prompt
        echo "$system_prompt

CONTEXT (${context_len} chars stored in file):
$context_file

QUERY: $query

Process step by step, use FINAL(answer) when done." | opencode --model "$OPENCODE_MODEL"
    else
        log_error "opencode CLI not found."
        return 1
    fi
}

# Run RLM loop directly with Ollama (Python)
run_ollama_rlm() {
    local query="$1"
    local context_file="$2"
    local verbose="${3:-false}"
    local model="${4:-$OLLAMA_MODEL}"
    
    # Check if we have the Python implementation
    local script_dir
    script_dir="$(dirname "$(readlink -f "$0")")"
    local rlm_py="${script_dir}/rlm.py"
    
    if [[ -f "$rlm_py" ]]; then
        local args=(--query "$query" --context-file "$context_file" --provider ollama)
        if [[ "$verbose" == "true" ]]; then
            args+=(--verbose)
        fi
        python3 "$rlm_py" "${args[@]}"
    else
        # Fallback: simple single-shot query
        log_warn "Full RLM script not found, using simple query"
        
        local context
        context=$(head -c 100000 "$context_file")  # Limit for single query
        
        query_ollama "Context:\n$context\n\nQuery: $query" "" "$model"
    fi
}

# Run RLM with llama.cpp server
run_llama_cpp_rlm() {
    local query="$1"
    local context_file="$2"
    local verbose="${3:-false}"
    
    # Similar to Ollama but using OpenAI-compatible API
    local script_dir
    script_dir="$(dirname "$(readlink -f "$0")")"
    local rlm_py="${script_dir}/rlm.py"
    
    if [[ -f "$rlm_py" ]]; then
        python3 "$rlm_py" --query "$query" --context-file "$context_file" --provider openai
    else
        local context
        context=$(head -c 100000 "$context_file")
        query_llama_cpp "Context:\n$context\n\nQuery: $query"
    fi
}

# Main function
main() {
    local query=""
    local context_path=""
    local cli="ollama"
    local model=""
    local sub_model=""
    local output_file=""
    local verbose=false
    local json_output=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -q|--query)
                query="$2"
                shift 2
                ;;
            -c|--context)
                context_path="$2"
                shift 2
                ;;
            -C|--cli)
                cli="$2"
                shift 2
                ;;
            -m|--model)
                model="$2"
                shift 2
                ;;
            -s|--sub-model)
                sub_model="$2"
                shift 2
                ;;
            -o|--output)
                output_file="$2"
                shift 2
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -j|--json)
                json_output=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$query" ]]; then
        log_error "Query is required (-q/--query)"
        usage
        exit 1
    fi
    
    if [[ -z "$context_path" ]]; then
        log_error "Context path is required (-c/--context)"
        usage
        exit 1
    fi
    
    # Create work directory
    WORK_DIR=$(mktemp -d)
    local context_file="$WORK_DIR/context.txt"
    
    # Gather context
    log_info "Gathering context from: $context_path"
    gather_context "$context_path" "$context_file"
    
    local context_size
    context_size=$(wc -c < "$context_file")
    log_info "Context size: $context_size bytes"
    
    # Run RLM with selected CLI
    local result
    case $cli in
        claude)
            log_info "Running with Claude Code CLI"
            result=$(run_claude_rlm "$query" "$context_file" "$verbose")
            ;;
        opencode)
            log_info "Running with OpenCode"
            result=$(run_opencode_rlm "$query" "$context_file" "$verbose")
            ;;
        ollama)
            log_info "Running with Ollama (${OLLAMA_HOST}:${OLLAMA_PORT})"
            result=$(run_ollama_rlm "$query" "$context_file" "$verbose" "${model:-$OLLAMA_MODEL}")
            ;;
        llama-cpp)
            log_info "Running with llama.cpp server"
            result=$(run_llama_cpp_rlm "$query" "$context_file" "$verbose")
            ;;
        *)
            log_error "Unknown CLI: $cli"
            exit 1
            ;;
    esac
    
    # Output result
    if [[ -n "$output_file" ]]; then
        echo "$result" > "$output_file"
        log_success "Result written to: $output_file"
    else
        echo "$result"
    fi
}

main "$@"
