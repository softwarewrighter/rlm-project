# Claude Code Instructions for RLM Project

## Project Overview

RLM (Recursive Language Models) is a Rust implementation allowing LLMs to process arbitrarily long contexts through iterative command execution.

## Critical Guidelines

### Environment Variables

When running RLM demos or the server with LiteLLM, environment variables must be loaded from `~/.env`:

**Pattern for loading env vars:**
```bash
export $(cat ~/.env | grep -v '^#' | xargs) && ./your-command.sh
```

**Required variables in `~/.env`:**
```bash
LITELLM_MASTER_KEY=sk-your-key-here
LITELLM_HOST=http://localhost:4000  # optional, defaults to localhost:4000
```

**Important:** `source ~/.env` does not work reliably in all contexts (VHS recordings, subshells). Always use the `export $(cat ...)` pattern.

### Starting the Server

```bash
# With LiteLLM (requires ~/.env with LITELLM_MASTER_KEY)
cd rlm-orchestrator
export $(cat ~/.env | grep -v '^#' | xargs) && ./target/release/rlm-server config-litellm-cli.toml

# Check health
curl http://localhost:8080/health
```

### Running CLI Demos

```bash
# L3 CLI demos require LiteLLM API key
export $(cat ~/.env | grep -v '^#' | xargs) && ./demo/l3/percentiles.sh
export $(cat ~/.env | grep -v '^#' | xargs) && ./demo/l3/error-ranking.sh
export $(cat ~/.env | grep -v '^#' | xargs) && ./demo/l3/unique-ips.sh
```

### Demo Directory Structure

```
demo/
├── l1/           # Level 1 DSL-only demos
├── l2/           # Level 2 WASM demos
├── l3/           # Level 3 CLI demos (native Rust binary execution)
└── README.md     # Demo documentation
```

### Configuration Files

| Config | Description |
|--------|-------------|
| `config-litellm-cli.toml` | LiteLLM with CLI enabled (for L3 demos) |
| `config-local-cli.toml` | Local Ollama with CLI enabled |
| `config.toml` | Default config (WASM only) |

### Level 3 CLI Notes

L3 CLI mode (`rust_cli_intent`) generates and compiles native Rust binaries:

1. **Codegen** - LLM generates Rust code (can take 10-30+ seconds)
2. **Compile** - rustc compiles the binary (~2-3 seconds)
3. **Execute** - Binary runs with stdin piped

Progress events separate these phases:
- `CliCodegenStart` / `CliCodegenComplete` - LLM code generation
- `CliCompileStart` / `CliCompileComplete` - rustc compilation
- `CliRunComplete` - binary execution

### Common Issues

1. **401 Unauthorized from LiteLLM** - API key not set. Use the `export $(cat ~/.env ...)` pattern.

2. **Server won't start** - Check if port 8080 is in use:
   ```bash
   pkill -f "rlm-server"
   ```

3. **CLI demo hangs** - Usually codegen timeout. Check server logs:
   ```bash
   tail -f /tmp/rlm-server.log
   ```

### VHS Terminal Recordings

When creating VHS recordings for demos:

```tape
# Use this pattern to load env vars
Type "export $(cat ~/.env | grep -v '^#' | xargs) && ./demo/l3/percentiles.sh"
Enter
```

### Project Structure

```
rlm-project/
├── rlm-orchestrator/     # Rust server and CLI
│   ├── src/
│   │   ├── orchestrator.rs  # Main RLM loop
│   │   ├── commands/        # Command execution
│   │   └── api/             # HTTP API + visualizer
│   └── config-*.toml        # Configuration files
├── demo/                    # Demo scripts by level
├── docs/                    # Documentation
└── CLAUDE.md               # This file
```
