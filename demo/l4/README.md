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

**File:** `detective-demo.sh`

A murder mystery case file with 7 witnesses, physical evidence, and background information. The LLM must:

1. Extract witness statements (L1 regex)
2. Analyze each witness for key claims and contradictions (L4 delegate)
3. Cross-reference with physical evidence
4. Identify the murderer with reasoning

**Running:**
```bash
# Requires LLM delegation enabled
export $(cat ~/.env | grep -v '^#' | xargs)
./demo/l4/detective-demo.sh
```

**Expected Answer:** Colonel Arthur Pemberton

### Data Structure

```
demo/l4/
  data/
    detective-mystery.txt   # 30KB case file
  detective-demo.sh         # CLI demo script
  README.md                 # This file
```

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
