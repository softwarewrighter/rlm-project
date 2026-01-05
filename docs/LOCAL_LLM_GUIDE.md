# Local LLM Guide for RLM

## Smart Bypass for Small Contexts

RLM now automatically bypasses iteration for small contexts (< 4000 chars by default):

```toml
# config.toml
bypass_enabled = true
bypass_threshold = 4000  # chars (~1000 tokens)
```

| Context Size | Mode | Latency | Token Savings |
|--------------|------|---------|---------------|
| < 4K chars | Direct (bypass) | ~1-2s | ~20-25% |
| > 4K chars | RLM iteration | ~10-30s | ~60-70% |

This eliminates the overhead penalty for small contexts while preserving RLM benefits for large ones.

---

## Why Model Size Matters for RLM

The **root LLM** in RLM must:
1. Parse complex system prompts explaining the RLM protocol
2. Output **valid JSON** commands consistently
3. Decide which operations to use (slice, regex, lines, etc.)
4. Know when to recurse vs when to answer
5. Track state across iterations

Small models (7B-14B) typically fail because they:
- Output prose instead of JSON
- Generate incomplete/malformed commands
- Hallucinate non-existent operations
- Fail to issue `FINAL` when they have the answer

**Sub-LM role is easier** - local models work fine for sub-calls since those are simple natural language tasks (summarize, extract facts) with no JSON protocol required.

## Model Capability Matrix

| Model Size | JSON Consistency | Instruction Following | RLM Root Viable? |
|------------|------------------|----------------------|------------------|
| 7B | ❌ Frequently breaks | ❌ Weak | No |
| 14B | ⚠️ Often malformed | ⚠️ Moderate | Unreliable |
| 32-34B | ✅ Good | ✅ Good | Yes |
| 70B+ | ✅ Excellent | ✅ Excellent | Yes (recommended) |

## Small Models for Sub-Calls (2-12GB VRAM)

Small models work well for **sub-LM calls** (simple summarization, extraction) since they don't need JSON protocol:

| Model | Size | VRAM (Q4) | Sub-Call Quality |
|-------|------|-----------|------------------|
| Llama-3.2-3B | 3B | ~2GB | ⭐⭐⭐ Good |
| Phi-3-mini | 3.8B | ~2.5GB | ⭐⭐⭐ Good |
| Qwen2.5-7B-Instruct | 7B | ~4.5GB | ⭐⭐⭐⭐ Very Good |
| Gemma-2-9B | 9B | ~6GB | ⭐⭐⭐⭐ Very Good |
| Mistral-7B-Instruct | 7B | ~4.5GB | ⭐⭐⭐ Good |

**Example config (API root + small local sub):**
```toml
# DeepSeek API for root (reliable JSON)
[[providers]]
provider_type = "deepseek"
model = "deepseek-chat"
role = "root"

# Small local model for sub-calls (free)
[[providers]]
provider_type = "ollama"
base_url = "http://localhost:11434"
model = "llama3.2:3b"
role = "sub"
```

## Hardware Configurations

### Dual P40 (48GB VRAM total)

Best setup for running 70B+ models locally.

**Recommended Models:**

| Model | Quantization | VRAM | Quality |
|-------|--------------|------|---------|
| **Qwen2.5-72B-Instruct** | Q4_K_M | ~42GB | ⭐⭐⭐⭐⭐ Best for JSON |
| **Llama-3.1-70B-Instruct** | Q4_K_M | ~40GB | ⭐⭐⭐⭐⭐ Excellent |
| Qwen2.5-32B-Instruct | Q6_K | ~26GB | ⭐⭐⭐⭐ Very Good |
| DeepSeek-Coder-V2-Lite-16B | Q8 | ~17GB | ⭐⭐⭐ Good |

**llama.cpp configuration:**
```bash
./llama-server -m qwen2.5-72b-instruct-q4_k_m.gguf \
  --n-gpu-layers 80 \
  --tensor-split 24,24 \
  --main-gpu 0 \
  --host 0.0.0.0 \
  --port 11434
```

**Ollama setup:**
```bash
# Pull the model
ollama pull qwen2.5:72b-instruct-q4_K_M

# Or for Llama
ollama pull llama3.1:70b-instruct-q4_K_M
```

### Single 24GB GPU (RTX 3090/4090, A5000, P40)

Limited to 32B models or heavily quantized 70B.

| Model | Quantization | VRAM | Notes |
|-------|--------------|------|-------|
| Qwen2.5-32B-Instruct | Q4_K_M | ~20GB | Best option |
| DeepSeek-Coder-33B | Q4_K_M | ~20GB | Good for code |
| Llama-3.1-70B | Q2_K | ~24GB | Quality loss |

### CPU + RAM Only (No GPU)

Possible but slow. Need 64GB+ RAM.

```bash
./llama-server -m qwen2.5-32b-instruct-q4_k_m.gguf \
  --n-gpu-layers 0 \
  --threads 16
```

Expect ~1-3 tokens/sec on modern CPU.

## RLM Config Examples

### Local 70B as Root (Recommended for P40 setup)

```toml
# config.toml
max_iterations = 20
max_sub_calls = 50

# Local Qwen 72B as root LLM
[[providers]]
provider_type = "ollama"
base_url = "http://p40-server:11434"
model = "qwen2.5:72b-instruct-q4_K_M"
role = "root"
weight = 1

# Local smaller model for sub-calls
[[providers]]
provider_type = "ollama"
base_url = "http://localhost:11434"
model = "qwen2.5-coder:14b"
role = "sub"
weight = 1
```

### Hybrid: API Root + Local Sub

Best cost/performance balance - use API for complex reasoning, local for simple sub-queries.

```toml
# DeepSeek API for root (cheap, reliable)
[[providers]]
provider_type = "deepseek"
base_url = "https://api.deepseek.com"
model = "deepseek-chat"
role = "root"
weight = 1

# Local Ollama for sub-calls (free)
[[providers]]
provider_type = "ollama"
base_url = "http://localhost:11434"
model = "qwen2.5-coder:14b"
role = "sub"
weight = 1
```

### Fully Local (Air-gapped)

```toml
# Local 70B for root
[[providers]]
provider_type = "ollama"
base_url = "http://localhost:11434"
model = "qwen2.5:72b-instruct-q4_K_M"
role = "root"
weight = 1

# Same or different model for sub-calls
[[providers]]
provider_type = "ollama"
base_url = "http://localhost:11434"
model = "qwen2.5:72b-instruct-q4_K_M"
role = "sub"
weight = 1
```

## Performance Expectations

### P40 (Pascal, no tensor cores)

| Model | Quantization | Tokens/sec | Latency per iteration |
|-------|--------------|------------|----------------------|
| 72B | Q4_K_M | ~10-15 t/s | ~30-60s |
| 32B | Q4_K_M | ~20-30 t/s | ~15-30s |
| 14B | Q4_K_M | ~40-60 t/s | ~5-15s |

### Modern GPUs (RTX 4090, A100)

| Model | Quantization | Tokens/sec | Latency per iteration |
|-------|--------------|------------|----------------------|
| 72B | Q4_K_M | ~30-50 t/s | ~10-20s |
| 32B | Q4_K_M | ~60-80 t/s | ~5-10s |

## Troubleshooting

### Model outputs prose instead of JSON

**Cause:** Model too small or wrong prompt format.

**Fix:** Use 32B+ model, or add few-shot examples to system prompt.

### JSON parse errors (negative indices, etc.)

**Cause:** Model generates Python-style indices.

**Status:** Fixed in orchestrator - now supports negative indices.

### Model ignores FINAL command

**Cause:** Model doesn't understand when to stop iterating.

**Fix:** Use larger model or adjust `max_iterations` in config.

### Out of VRAM

**Fix:** Use smaller quantization or enable RAM spill:
```bash
# Offload some layers to CPU
--n-gpu-layers 60  # Instead of 80
```

## Cost Comparison

| Setup | Cost | Latency | Quality |
|-------|------|---------|---------|
| DeepSeek API | ~$0.001/query | ~5-10s | ⭐⭐⭐⭐⭐ |
| Local 72B (P40) | Electricity only | ~30-60s | ⭐⭐⭐⭐⭐ |
| Local 32B | Electricity only | ~15-30s | ⭐⭐⭐⭐ |
| Local 14B | Electricity only | ~5-15s | ⭐⭐ (unreliable) |

For development/testing: DeepSeek API is cheapest and fastest.
For production/privacy: Local 70B+ is viable with proper hardware.
