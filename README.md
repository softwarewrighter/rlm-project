# RLM - Recursive Language Models Implementation

A practical implementation of Recursive Language Models based on the paper ["Recursive Language Models"](https://arxiv.org/html/2512.24601v1) by Zhang, Kraska, and Khattab (MIT CSAIL).

## Overview

RLM is an inference strategy that allows LLMs to process arbitrarily long contexts by:
1. Treating prompts as external variables in a REPL environment
2. Letting the LLM write code to examine, filter, and chunk the context
3. Providing recursive sub-LM calls for semantic analysis
4. Aggregating results to produce final answers

This approach handles inputs **2 orders of magnitude beyond model context windows** while maintaining comparable or lower costs.

## Contents

```
rlm-project/
├── README.md                      # This file
├── DISCUSSION.md                  # Full conversation transcript
├── docs/
│   ├── rlm-eli5.md               # ELI5 explanation with analogies
│   ├── rlm-orchestrator-architecture.md  # Full Rust architecture
│   └── optimizing.md             # Parallelism & dogfooding guide
├── src/
│   ├── rlm.py                    # Working Python implementation
│   └── rlm-wrapper.sh            # Shell wrapper for multiple CLIs
└── config/
    └── example.toml              # Example configuration
```

## Quick Start

### Prerequisites

```bash
# Python dependencies
pip install httpx rich typer pydantic --break-system-packages

# Ensure Ollama is running
ollama serve
ollama pull qwen2.5-coder:32b  # or your preferred model
```

### Basic Usage

```bash
# Simple query with local Ollama
python src/rlm.py \
  --query "Find all function definitions" \
  --context-file ./your_codebase.py

# With DeepSeek API
export DEEPSEEK_API_KEY="your-key"
python src/rlm.py \
  --query "Summarize this document" \
  --context-file large_doc.txt \
  --provider deepseek

# Using the shell wrapper
./src/rlm-wrapper.sh \
  --query "Find security issues" \
  --context ./src \
  --cli ollama
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server hostname | `localhost` |
| `OLLAMA_PORT` | Ollama server port | `11434` |
| `OLLAMA_MODEL` | Default model | `qwen2.5-coder:32b` |
| `DEEPSEEK_API_KEY` | DeepSeek API key | - |
| `ANTHROPIC_API_KEY` | Claude API key | - |

## Implementation Options

| Option | Best For | Setup Time |
|--------|----------|------------|
| **Python + Ollama** | Learning, quick experiments | 10 minutes |
| **Shell Wrapper** | Integration with existing CLIs | 5 minutes |
| **Rust Orchestrator** | Production, high performance | 2-4 weeks |
| **MCP Server** | Claude Code integration | 1-2 days |

See `docs/rlm-eli5.md` for detailed pros/cons of each approach.

## Architecture Highlights

### Parallel Sub-LM Dispatch
Process multiple context chunks simultaneously across distributed GPU servers.

### Speculative Execution
Predict and prefetch likely-needed data while the root LLM is thinking.

### Semantic Caching
Cache not just exact matches but semantically similar queries using embeddings.

### Dogfooding
The tool can analyze and improve its own codebase through iterative cycles.

See `docs/optimizing.md` for complete optimization strategies.

## Hardware Recommendations

For optimal performance with the Rust implementation:

| Component | Recommendation |
|-----------|----------------|
| CPU | Dual Xeon (48-72 threads) |
| RAM | 256GB+ DDR4/DDR5 |
| GPU | Multiple servers with M40/RTX/P100 |
| Storage | NVMe for context caching |

## Project Status

- ✅ Python implementation (working)
- ✅ Shell wrapper (working)
- 📋 Rust orchestrator (architecture documented)
- 📋 MCP server (design documented)
- 📋 Emacs integration (elisp documented)

## References

- [Recursive Language Models Paper](https://arxiv.org/html/2512.24601v1)
- [Ollama](https://ollama.ai)
- [DeepSeek API](https://platform.deepseek.com)
- [Claude Code](https://claude.ai/code)

## License

MIT License - See individual files for details.

## Contributing

This project uses dogfooding - the RLM tool helps improve itself! See `docs/optimizing.md` for the dogfooding methodology.
