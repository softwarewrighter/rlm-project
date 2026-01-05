# RLM - Recursive Language Models

A practical implementation of Recursive Language Models based on the paper ["Recursive Language Models"](https://arxiv.org/html/2512.24601v1) by Zhang, Kraska, and Khattab (MIT CSAIL).

## What is RLM?

RLM is an inference strategy that allows LLMs to process **arbitrarily long contexts** by:

1. Having the LLM output structured commands to examine context
2. Executing those commands and returning results
3. Iterating until the LLM reaches a final answer
4. Using sub-LM calls for semantic analysis of chunks

This handles inputs **2 orders of magnitude beyond model context windows**.

> **New to RLM?** Start with [docs/beginners/](docs/beginners/) for gentle introductions.

## Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| **Rust Orchestrator** | ✅ Working | Pure Rust, structured JSON commands |
| **Visualizer** | ✅ Working | Interactive HTML at `/visualize` |
| **Python Implementation** | ✅ Working | Reference implementation |
| **Multi-Provider** | ✅ Working | DeepSeek, Ollama (local/remote) |

## Quick Start

### Option 1: Rust Server (Recommended)

```bash
cd rlm-orchestrator

# Configure providers in config.toml
# Set API key if using DeepSeek
export DEEPSEEK_API_KEY="your-key"

# Run the server
cargo run --bin rlm-server

# Open visualizer
open http://localhost:8080/visualize

# Or use the API
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How many errors?", "context": "Line 1: OK\nLine 2: ERROR\nLine 3: OK"}'
```

### Option 2: Python CLI

```bash
# Install dependencies
pip install httpx rich typer pydantic

# Run with Ollama
python src/rlm.py \
  --query "Find all function definitions" \
  --context-file ./code.py

# Run with DeepSeek
export DEEPSEEK_API_KEY="your-key"
python src/rlm.py \
  --query "Summarize this" \
  --context-file doc.txt \
  --provider deepseek
```

## Documentation

### For Beginners
- [docs/beginners/what-is-rlm.md](docs/beginners/what-is-rlm.md) - What RLM is and why it matters
- [docs/beginners/how-it-works.md](docs/beginners/how-it-works.md) - Step-by-step walkthrough
- [docs/beginners/getting-started.md](docs/beginners/getting-started.md) - Your first RLM query

### Technical Documentation
- [docs/architecture.md](docs/architecture.md) - Rust orchestrator design
- [docs/commands.md](docs/commands.md) - Available JSON commands
- [docs/visualizer.md](docs/visualizer.md) - Using the web visualizer
- [docs/optimizing.md](docs/optimizing.md) - Performance optimization

### Reference
- [docs/api.md](docs/api.md) - REST API reference
- [docs/config.md](docs/config.md) - Configuration options

## Architecture

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

## Available Commands

The LLM outputs JSON commands that the executor runs:

| Command | Description | Example |
|---------|-------------|---------|
| `slice` | Get character range | `{"op": "slice", "start": 0, "end": 1000}` |
| `lines` | Get line range | `{"op": "lines", "start": 0, "end": 50}` |
| `find` | Find text occurrences | `{"op": "find", "text": "ERROR"}` |
| `regex` | Regex search | `{"op": "regex", "pattern": "def \\w+"}` |
| `count` | Count lines/chars/matches | `{"op": "count", "what": "matches"}` |
| `llm_query` | Sub-LM semantic call | `{"op": "llm_query", "prompt": "Summarize: ${chunk}"}` |
| `final` | Return answer | `{"op": "final", "answer": "Found 3 errors"}` |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Simple query → answer |
| `/debug` | POST | Full iteration history |
| `/visualize` | GET | Interactive HTML visualizer |

## Configuration

Edit `rlm-orchestrator/config.toml`:

```toml
max_iterations = 20
max_sub_calls = 50

# Root LLM (smart, for orchestration)
[[providers]]
provider_type = "deepseek"  # or "ollama"
model = "deepseek-chat"
role = "root"

# Sub LLM (fast, for llm_query)
[[providers]]
provider_type = "ollama"
base_url = "http://localhost:11434"
model = "qwen2.5-coder:14b"
role = "sub"
```

## License

MIT License

## References

- [Recursive Language Models Paper](https://arxiv.org/html/2512.24601v1)
- [Ollama](https://ollama.ai)
- [DeepSeek API](https://platform.deepseek.com)
