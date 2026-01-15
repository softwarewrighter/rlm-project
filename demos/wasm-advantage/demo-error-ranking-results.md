# Error Ranking Demo Results

This demo compares RLM performance on a complex analytical query using different LLM backends.

**Query:** "rank the errors from most often to least often found in the logs"

**Log file:** `sample.log` (9,701 chars, 117 lines)

## Results Comparison

| Metric | llama3.2:3b (Ollama) | DeepSeek (LiteLLM) |
|--------|----------------------|--------------------|
| Iterations | 20 (no answer) | 4 ✓ |
| Total Time | ~6 min | **44 sec** |
| Tokens | 67,600 | 13,250 |
| WASM used | No | Yes |
| Success | ❌ | ✅ |

## DeepSeek Output

```
╭──────────────────────────────────────────────────────────────╮
│  RLM CLI - Recursive Language Model Query                   │
├──────────────────────────────────────────────────────────────┤
│  File:   ../demos/wasm-advantage/sample.log
│  Size:   9701 chars (117 lines, ~2425 tokens)
│  Model:  deepseek/deepseek-chat (via LiteLLM @ http://localhost:4000)
│  Query:  rank the errors from most often to least often ...
╰──────────────────────────────────────────────────────────────╯

Starting RLM processing...

┌─ Iteration 1
│ ⏳ Calling LLM...                              │ ⏱  LLM: 4426ms
│ ◀ Exec: 0ms
└────────────────────────────────────────
┌─ Iteration 2
│ ⏳ Calling LLM...                              │ ⏱  LLM: 17808ms
│ 🔧 Compiling WASM... done (713ms)
│ ◀ Exec: 721ms
└────────────────────────────────────────
┌─ Iteration 3
│ ⏳ Calling LLM...                              │ ⏱  LLM: 16607ms
│ 🔧 Compiling WASM...│ ◀ Exec: 10ms
└────────────────────────────────────────
┌─ Iteration 4
│ ⏳ Calling LLM...                              │ ⏱  LLM: 3825ms
│ ◀ Exec: 0ms
└────────────────────────────────────────
✓ Final: Error Type Rankings (most to least frequent):
============================================
1. Authen
Completed in 4 iteration(s)


╭──────────────────────────────────────────────────────────────╮
│  Results                                                     │
├──────────────────────────────────────────────────────────────┤
│  Iterations:     4
│  Sub-LM calls:   0
│  Tokens used:    11972 prompt + 1278 completion
╰──────────────────────────────────────────────────────────────╯

Answer:
════════════════════════════════════════════════════════════════
Error Type Rankings (most to least frequent):
============================================
1. AuthenticationFailed: 13 occurrences
2. RequestFailed: 11 occurrences
3. ConnectionTimeout: 10 occurrences
4. ValidationError: 9 occurrences
5. OtherError: 8 occurrences

════════════════════════════════════════════════════════════════
```

**Total time:** 44.331 seconds

## Command Used

```bash
cargo run --release --bin rlm -- \
    ../demos/wasm-advantage/sample.log \
    "rank the errors from most often to least often found in the logs" \
    --litellm \
    --litellm-url http://localhost:4000 \
    --litellm-key $LITELLM_KEY \
    -m deepseek/deepseek-chat \
    -v
```

## Key Observations

1. **Model capability matters**: DeepSeek immediately understood to use `rust_wasm` for counting and categorizing, while llama3.2:3b struggled with the analytical query.

2. **WASM advantage**: DeepSeek compiled custom Rust code to count error types in a single pass, taking 713ms to compile and 721ms to execute.

3. **Token efficiency**: DeepSeek used ~5x fewer tokens by solving the problem in fewer iterations.

4. **Real-time progress**: The `-v` flag shows LLM calls and WASM compilation as they happen, providing visibility during the ~44 second run.
