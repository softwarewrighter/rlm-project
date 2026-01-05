# RLM Testing Findings

## Summary

This document records experimental results from testing various local LLMs with the RLM orchestrator.

---

## Sub-LM Testing Results

Sub-LM models handle simple tasks: summarization, extraction, answering direct questions. They don't need to output structured JSON.

### Models Tested

| Model | Size | VRAM | Sub-Call Success | Notes |
|-------|------|------|------------------|-------|
| qwen2.5-coder:14b | 14B | ~9GB | **Works well** | Current default, reliable |
| gemma2:9b | 9B | ~5.4GB | **Works well** | Good quality, recommended |
| qwen2.5:7b-instruct | 7B | ~4.7GB | **Works** | Acceptable for simple sub-calls |
| mistral:7b | 7B | ~4.4GB | **Works** | Remote (big72), acceptable |
| llama3.2:3b | 3B | ~2GB | Not recommended | Too small for meaningful sub-calls |

### Observations

1. **7B models are sufficient for sub-LM calls** - They handle summarization and extraction well enough
2. **14B models are better** - More coherent responses, fewer edge case failures
3. **Remote sub-LMs add latency** - Consider network overhead when load balancing

---

## Root LLM Requirements

The root LLM has a harder job:
- Parse complex RLM protocol system prompts
- Output valid JSON commands consistently
- Decide which operations to use (slice, regex, lines, etc.)
- Know when to issue FINAL vs continue iterating

### Model Capability for Root Role

| Model Size | JSON Consistency | RLM Root Viable? | Notes |
|------------|------------------|------------------|-------|
| 7B | Frequently breaks | **No** | Outputs prose instead of JSON |
| 14B | Often malformed | **Unreliable** | Sometimes works, often fails |
| 32-34B | Good | **Yes** | Usable with retry logic |
| 70B+ | Excellent | **Yes (recommended)** | DeepSeek API or local Qwen2.5-72B |

### Why Small Models Fail as Root

Small models (7B-14B) typically fail because they:
- Output prose explanations instead of JSON commands
- Generate incomplete or malformed JSON
- Hallucinate non-existent operations
- Fail to issue `FINAL` command when they have the answer
- Lose track of state across iterations

---

## Smart Bypass Results

Added bypass logic to skip RLM iteration for small contexts.

| Context Size | Mode | Pass Rate | Latency | Token Savings |
|--------------|------|-----------|---------|---------------|
| 2.5K chars | Bypass (direct) | 100% (4/4) | ~1.4s | 23% |
| 41K chars | RLM iteration | 67-100% | ~11-18s | 64-74% |

### Key Insight

- **Bypass is essential** - Without it, small contexts show negative savings (-164%) due to RLM iteration overhead
- **4K char threshold is good** - Captures most small contexts without sacrificing large context benefits

---

## Large Context Testing Summary

### Context Size vs RLM Value

| Context Size | RLM Benefit | Token Savings | Notes |
|--------------|-------------|---------------|-------|
| < 4K chars | **None** (bypass) | ~23% | Use bypass - direct LLM is better |
| 4-15K chars | **Negative** | -400% to -900% | RLM overhead exceeds benefit |
| 15-50K chars | **Marginal** | ~0-30% | Breakeven zone |
| 50K+ chars | **Significant** | 60-86% | RLM shows clear value |

### niah_deep Test (248K chars) with gemma2:9b sub-LM

| Query | Iterations | Latency | Tokens | Savings |
|-------|------------|---------|--------|---------|
| Agent codename | 7 | 37s | 9,149 | **85%** |
| Coordinates | 5 | 83s | 8,508 | **86%** |

**Key Result:** On 248K char contexts (~62K tokens), RLM achieves **85-86% token savings** while finding needles in the haystack.

### Multi-Source Experiment (Rust HTTP Crates)

| Context | Queries | Pass Rate | Avg Savings |
|---------|---------|-----------|-------------|
| 11K chars (4 crates, truncated) | 4/4 | 100% | **-727%** (worse) |
| 48K chars (8 crates, full) | In progress | - | - |

**Insight:** RLM performs worse on small-to-medium contexts due to iteration overhead. The 4K bypass threshold should potentially be raised to ~15K chars.

---

## Benchmark Results (2026-01-05)

### code_analysis (2.5K chars) - BYPASS mode
| Query | Result | Latency | Tokens | Savings |
|-------|--------|---------|--------|---------|
| Class count | PASS | 1.7s | 644 | 23% |
| Method names | PASS | 1.5s | 662 | 21% |
| Inheritance | PASS | 1.6s | 643 | 23% |
| Function count | PASS | 1.5s | 648 | 23% |

**Summary:** 4/4 passed, bypass working correctly for small contexts.

### log_counting (41K chars) - RLM iteration
| Query | Result | Iterations | Latency | Tokens | Savings |
|-------|--------|------------|---------|--------|---------|
| ERROR count | PASS | 2 | 12.9s | 2759 | 74% |
| FATAL count | PASS | 2 | 8.7s | 2577 | 76% |
| ERROR/WARN ratio | FAIL | - | - | - | - |

**Summary:** 2/3 passed. Ratio question returns verbose response instead of just "40".

### niah_deep (248K chars) - RLM iteration (Needle in a Haystack)
| Query | Result | Iterations | Latency | Tokens | Savings |
|-------|--------|------------|---------|--------|---------|
| Agent codename | PASS | 7 | 37.3s | 9149 | 85% |
| Coordinates | PASS | 5 | 83.2s | 8508 | 86% |

**Summary:** 2/2 passed. RLM successfully finds needles in 248K char haystack with 85-86% token savings!

---

## Configuration Recommendations

### Development/Testing
```toml
# Use DeepSeek API for root (cheap, fast, reliable)
[[providers]]
provider_type = "deepseek"
model = "deepseek-chat"
role = "root"

# Local 7-14B for sub-calls (free)
[[providers]]
provider_type = "ollama"
model = "qwen2.5-coder:14b"  # or qwen2.5:7b-instruct
role = "sub"
```

### Production (Privacy-focused)
```toml
# Local 70B+ for root (requires dual P40 or better)
[[providers]]
provider_type = "ollama"
model = "qwen2.5:72b-instruct-q4_K_M"
role = "root"

# Same or smaller model for sub-calls
[[providers]]
provider_type = "ollama"
model = "qwen2.5:7b-instruct"
role = "sub"
```

---

## Known Issues

1. **Ratio/calculation queries** - RLM sometimes returns full analysis instead of just the answer (e.g., "100/1" instead of "100")
2. **14B root instability** - Works sometimes but not reliable enough for production

---

## What Worked

1. **Smart bypass** - Essential for small contexts, eliminates overhead penalty
2. **DeepSeek API as root** - Reliable JSON output, fast, cheap
3. **Local 7-14B models for sub-calls** - All tested models work acceptably
4. **gemma2:9b** - Good balance of quality and size for sub-LM
5. **RLM on large contexts** - 85-86% token savings on 248K char documents
6. **Helper scripts** - Repeatable builds and runs

## What Didn't Work

1. **Small models as root** - 7B-14B models fail to output consistent JSON
2. **llama3.2:3b** - Too small even for sub-calls (same model as llama3.2:latest)
3. **Calculation queries** - Need better prompting for numeric-only answers

---

## Next Steps

- [ ] Add retry logic for root LLM JSON parsing failures
- [ ] Implement streaming for long-running queries
- [ ] Add more benchmark types (multi-document, narrative reasoning)
- [ ] Test phi-3 models for sub-calls
- [ ] Add answer normalization for ratio queries
