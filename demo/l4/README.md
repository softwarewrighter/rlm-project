# Level 4: LLM Delegation Demos

Level 4 introduces **recursive LLM delegation** - the ability to create nested RLM instances that have full tool access (L1-L3) for analyzing subsets of data.

## Key Concepts

### llm_query vs llm_delegate

| Aspect | llm_query | llm_delegate |
|--------|-----------|--------------|
| Context | None (prompt only) | Full via `on` parameter |
| Tools | None | L1-L3 (configurable) |
| Iterations | 1 (single call) | Multiple (nested RLM) |
| Use case | Quick semantic check | Complex chunk analysis |

### When to Use llm_delegate

- Semantic analysis that needs computation (counting, filtering)
- Cross-referencing information across sections
- Tasks requiring multi-step reasoning with tool access
- Analysis where DSL/WASM alone can't handle the semantics

## Demo: Detective Mystery

A murder mystery case file with 7 witnesses, physical evidence, and background information. The LLM must cross-reference testimonies and evidence to identify the killer.

### Usage

**Direct CLI Command:**
```bash
# Load environment and run directly
export $(cat ~/.env | grep -v '^#' | xargs)
./rlm-orchestrator/target/release/rlm \
    demo/l4/data/detective-mystery.txt \
    "Who murdered Lord Ashford? Analyze ALL witness statements and ALL physical evidence. Identify the killer." \
    --enable-llm-delegation \
    --coordinator-mode \
    --litellm \
    -m deepseek-coder \
    --max-iterations 15 \
    -v
```

**Flags explained:**
- `--enable-llm-delegation`: Enable L4 LLM commands (llm_reduce, llm_query, llm_delegate)
- `--coordinator-mode`: Use llm_reduce for chunk-based extraction (preferred for large contexts)
- `--litellm`: Use LiteLLM gateway for API access
- `-m deepseek-coder`: Model to use
- `--max-iterations 15`: Sufficient for this demo (typically completes in 3-5)

**Using the Demo Script:**
```bash
export $(cat ~/.env | grep -v '^#' | xargs)
./demo/l4/detective-demo.sh
```

### Expected Answer

**Colonel Arthur Pemberton** - based on:
- Motive: Fraud exposure from 1998 incident
- Opportunity: Admitted presence in study; gardener saw him leaving
- Physical evidence: Footprints matching his distinctive limp
- Timeline contradiction: Claims he left at 10:20 but was seen at 10:30+

See `data/answer.txt` for the full solution (human reference only - not provided to LLM).

### Data Structure

```
demo/l4/
  data/
    detective-mystery.txt   # Case file (no answer - LLM must deduce)
    answer.txt              # Solution for human reference only
  detective-demo.sh         # CLI demo script
  README.md                 # This file
```

**Important:** The mystery file does NOT contain the answer. The LLM must deduce it from the evidence.

## Future: War & Peace (Planned)

Character relationship extraction from the full novel:
- Split into chapters (L3 CLI)
- Extract characters from each chapter (L4 delegate)
- Merge and deduplicate (L3 CLI)
- Build family tree (L4 delegate)

## Configuration

Enable LLM delegation in config:

```toml
[llm_delegation]
enabled = true
max_recursion_depth = 3
nested_levels = ["dsl", "wasm"]
```

## Technical Notes

- Nested RLM instances do NOT have access to `llm_delegate` (prevents infinite recursion)
- Maximum recursion depth defaults to 3
- Nested instances use a simplified system prompt
- Progress events track delegation start/complete

## Future Ideas

### Video Description Demo (Planned)

A more advanced L4 demo using multi-modal capabilities:

1. **Input**: Large video (90 minutes)
2. **Chunking**: Use L3 CLI to call ffmpeg to extract stills every 0.5 seconds
3. **Delegation**: For each chunk (1-5 minutes of stills):
   - `llm_delegate` to vision model with stills
   - Generate description of each frame
   - Detect scene boundaries within chunk
4. **Synthesis**: Combine frame descriptions into chunk summaries
5. **Output**: Full video description with scene markers

This would demonstrate:
- L3 CLI for native tool execution (ffmpeg)
- L4 delegation for multi-modal analysis
- Hierarchical summarization (frames → scenes → video)
