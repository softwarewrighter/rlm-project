# RLM Implementation Discussion

## Date: January 4, 2026

## Participants
- Mike (User) - Systems administrator and software developer with distributed GPU infrastructure
- Claude (Assistant) - AI assistant

---

## Summary

This conversation covered the practical implementation of Recursive Language Models (RLM) based on the paper "Recursive Language Models" by Zhang, Kraska, and Khattab from MIT CSAIL.

### Key Topics Discussed

1. **Paper Analysis**: Reviewed the RLM paper which introduces an inference strategy where:
   - Long prompts are treated as external environment variables
   - LLMs write code to examine, decompose, and process context
   - Recursive sub-LM calls enable semantic analysis of chunks
   - Results are aggregated to produce final answers

2. **Performance Results from Paper**:
   - RLM(GPT-5) achieved 91.33% on BrowseComp+ vs 0% for base model
   - 58% F1 on OOLONG-Pairs vs 0.04% for base GPT-5
   - Handles 10M+ tokens effectively
   - Comparable or lower cost than base model calls

3. **Implementation Options Designed**:
   - **Option A**: Custom Rust orchestrator (recommended for production)
   - **Option B**: OpenCode + DeepSeek API wrapper
   - **Option C**: Pure Python + Ollama (for learning/quick start)
   - **Option D**: Claude Code CLI + MCP Server
   - **Option E**: Emacs + Rust daemon integration

4. **User's Hardware Context**:
   - Distributed GPU infrastructure: Tesla M40s, RTX systems, P100s
   - Multiple Arch Linux servers
   - Dual Xeon systems with 48-72 threads
   - Substantial system RAM (256GB+)
   - Solar power at $0.45/kWh

5. **Optimization Strategies Developed**:
   - Parallel sub-LM dispatch across GPU servers
   - Speculative execution with prefetching
   - Pipeline parallelism with Tokio runtime
   - Semantic similarity caching with embeddings
   - Memory-mapped contexts for zero-copy access

6. **Dogfooding Methodology**:
   - Bootstrap cycle where RLM improves itself
   - Phases: Analysis → Improvement → Testing → Iteration
   - Human approval gates for safety
   - Continuous benchmarking framework

---

## Deliverables Created

### Documentation

1. **rlm-eli5.md** (~400 lines)
   - ELI5 explanation using "cookie jar" analogy
   - Three magic powers of RLM
   - Five implementation options with detailed pros/cons
   - Quick comparison chart
   - Recommended learning path

2. **rlm-orchestrator-architecture.md** (~1200 lines)
   - Complete Rust project structure
   - Provider trait and implementations (Ollama, DeepSeek, Claude)
   - LLM pool with load balancing strategies
   - Python REPL integration via PyO3
   - REST API with Axum
   - Emacs elisp integration
   - Configuration examples

3. **optimizing.md** (~800 lines)
   - Tokio runtime configuration for high-core systems
   - Five optimization strategies with code
   - Dogfooding orchestrator implementation
   - Benchmarking framework
   - Expected performance gains table

### Code

4. **rlm.py** (~600 lines)
   - Working Python implementation
   - Supports Ollama, DeepSeek, OpenAI-compatible, Claude
   - Rich console output with progress tracking
   - Async execution with httpx
   - CLI with argparse

5. **rlm-wrapper.sh** (~300 lines)
   - Shell wrapper for multiple CLIs
   - Support for Claude Code, OpenCode, Ollama, llama.cpp
   - Context gathering from files or directories
   - Environment variable configuration

6. **config/example.toml**
   - Example configuration for distributed setup
   - Runtime, parallelism, caching settings
   - Multiple Ollama server configuration

---

## Technical Decisions

### Why Rust for Production?
- Native async with Tokio for high parallelism
- PyO3 for Python REPL integration
- Memory safety for long-running daemon
- Excellent performance for the orchestration layer
- Mike's preferred language

### Why Python for Quick Start?
- Faster iteration during learning
- Direct REPL execution without compilation
- Rich ecosystem for HTTP clients
- Easy to modify and experiment

### Load Balancing Strategy
Chose "Adaptive" strategy combining:
- Provider priority (local > cloud)
- Current load percentage
- Error rate history
- Health check results

### Caching Approach
Two-tier caching:
1. Exact match (fast, simple)
2. Semantic similarity (embedding-based, 0.85 threshold)

Local embeddings via Ollama's embedding endpoint to avoid API costs.

---

## Recommended Next Steps

### Week 1: Quick Win
Start with Python + Ollama implementation to understand RLM behavior.

```bash
python rlm.py -q "Find all TODO items" -c ./your_project/
```

### Week 2-3: Production Path
Begin Rust orchestrator development:
1. LLM provider abstractions
2. Pool with health checking
3. Basic REPL integration
4. REST API

### Week 4+: Optimization
Add parallelism features:
1. Parallel sub-LM dispatch
2. Speculative execution
3. Semantic caching

### Parallel Track
Set up MCP server for Claude Code integration when using that workflow.

---

## Key Insights from Discussion

1. **Context as Environment Variable**: The fundamental insight - don't feed huge contexts to the LLM, let it programmatically access them.

2. **Sub-LM Calls are Cheap**: Using smaller/faster models for sub-queries keeps costs reasonable while maintaining quality.

3. **Code Execution is Powerful**: Regex filtering, chunking, and aggregation via code dramatically reduces what the LLM needs to process.

4. **Speculation Helps**: Predicting likely next requests and prefetching while the root LLM thinks saves significant latency.

5. **Dogfooding Works**: Having the tool analyze and improve itself creates a positive feedback loop for development.

---

## Questions for Future Exploration

1. How deep should recursion go? (Paper uses depth 1, but deeper might help complex tasks)

2. What's the optimal chunk size for different content types?

3. Can we train models specifically to be better RLM agents?

4. How to handle streaming responses in the RLM framework?

5. What's the best way to parallelize sub-calls while maintaining ordering where needed?

---

## References

- Paper: https://arxiv.org/html/2512.24601v1
- Ollama: https://ollama.ai
- DeepSeek: https://platform.deepseek.com
- Tokio: https://tokio.rs
- PyO3: https://pyo3.rs
