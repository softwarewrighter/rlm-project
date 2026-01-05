# RLM Architecture

This document provides an overview of the Rust RLM orchestrator architecture. For detailed implementation notes, see [rlm-orchestrator-architecture.md](rlm-orchestrator-architecture.md).

## System Overview

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Client    │────▶│  RLM Server     │────▶│  Root LLM   │
│  /visualize │     │  (Rust/Axum)    │     │  (DeepSeek) │
└─────────────┘     └────────┬────────┘     └─────────────┘
                             │
                    ┌────────▼────────┐
                    │ Command Executor │
                    │  slice, find,   │
                    │  regex, count,  │
                    │  llm_query...   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Ollama   │  │ Ollama   │  │ Ollama   │
        │ (local)  │  │ (big72)  │  │ (other)  │
        └──────────┘  └──────────┘  └──────────┘
              Sub-LM Pool (for llm_query)
```

## Core Components

### API Layer (`src/api/mod.rs`)

Axum-based HTTP server with endpoints:
- `GET /health` - Health check
- `POST /query` - Simple query → answer
- `POST /debug` - Full iteration history
- `GET /visualize` - Interactive HTML UI

### Orchestrator (`src/orchestrator.rs`)

The main loop that:
1. Builds a system prompt with available commands
2. Sends query + context to root LLM
3. Parses JSON command from response
4. Executes command via CommandExecutor
5. Feeds result back to LLM
6. Repeats until `final` command or max iterations

### Command Executor (`src/commands/mod.rs`)

Parses and executes JSON commands:
- `slice`, `lines` - Extract text ranges
- `find`, `regex` - Search patterns
- `count` - Statistics
- `llm_query` - Delegate to sub-LLM
- `final` - Return answer

Manages variable storage for command chaining.

### LLM Pool (`src/llm/pool.rs`)

Manages multiple LLM providers with:
- Role-based selection (root vs sub)
- Weighted load balancing
- Provider abstraction (DeepSeek, Ollama)

### Providers (`src/llm/`)

- `deepseek.rs` - DeepSeek API client
- `ollama.rs` - Ollama API client

## Data Flow

```
Query + Context
      │
      ▼
┌─────────────────────────────────────────┐
│           Orchestrator Loop              │
│                                          │
│  1. Build prompt with context + history │
│  2. Call root LLM                        │
│  3. Parse JSON command                   │
│  4. Execute command                      │
│  5. Store result in variables           │
│  6. If not final, goto 1                │
│                                          │
└─────────────────────────────────────────┘
      │
      ▼
Final Answer
```

## Key Design Decisions

### Structured Commands (JSON)

Instead of free-form code execution, LLMs emit structured JSON commands. This provides:
- Safety (no arbitrary code execution)
- Predictability (finite command set)
- Debuggability (easy to log and replay)

### Variable Storage

Commands can store results in named variables, enabling multi-step reasoning:
```json
{"op": "find", "text": "ERROR", "store": "errors"}
{"op": "count", "what": "matches", "on": "errors"}
```

### Provider Abstraction

All LLM providers implement a common trait, making it easy to add new backends (OpenAI, Anthropic, etc.).

### Async Runtime

Uses Tokio for async HTTP. The `llm_query` command uses `block_in_place` to handle nested async calls within the sync command executor.

## File Structure

```
rlm-orchestrator/
├── Cargo.toml
├── config.toml           # Provider configuration
└── src/
    ├── main.rs           # Entry point
    ├── api/
    │   └── mod.rs        # HTTP endpoints + visualizer
    ├── commands/
    │   └── mod.rs        # Command executor
    ├── llm/
    │   ├── mod.rs        # Pool and traits
    │   ├── pool.rs       # Load balancing
    │   ├── deepseek.rs   # DeepSeek provider
    │   └── ollama.rs     # Ollama provider
    └── orchestrator.rs   # Main RLM loop
```
