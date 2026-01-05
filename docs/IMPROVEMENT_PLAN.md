# RLM Improvement Plan

**Status**: Planning complete, ready for implementation
**Created**: 2026-01-04
**Resume**: Tomorrow

## Summary
Enhance the RLM orchestrator with smart bypass for small contexts, align benchmarks with the paper, and update documentation with results.

---

## Decision Points (To Discuss)

### Priority Options
1. **Smart Bypass + README** - Quick wins first
2. **All three in sequence** - Bypass → README → Benchmarks
3. **Benchmarks first** - Validate paper alignment before optimization

### Bypass Threshold Options
1. **4000 chars (~1K tokens)** - Conservative, bypass only tiny contexts
2. **8000 chars (~2K tokens)** - Moderate, skip for single-page docs
3. **16000 chars (~4K tokens)** - Aggressive, direct path for anything small

---

---

## Part 1: Smart RLM Bypass

**Goal**: Skip RLM iteration overhead for contexts below a threshold

### Implementation Location
**Option A (Recommended)**: API handler level (`src/api/mod.rs:144-152`)
- Cleanest separation of concerns
- Easy to toggle via config
- No orchestrator changes needed

**Option B**: Orchestrator level (`src/orchestrator.rs:133-143`)
- More integrated approach
- Requires config struct changes

### Changes Required

1. **Config Extension** (`src/lib.rs`):
```rust
pub struct RlmConfig {
    // existing fields...
    pub bypass_threshold: usize,  // e.g., 4000 chars
    pub bypass_enabled: bool,
}
```

2. **API Handler** (`src/api/mod.rs`):
```rust
const DEFAULT_BYPASS_THRESHOLD: usize = 4000;  // ~1000 tokens

async fn process_query(...) {
    if state.config.bypass_enabled && request.context.len() < BYPASS_THRESHOLD {
        return direct_llm_call(&state, &request).await;
    }
    // existing RLM path...
}
```

3. **Direct LLM Helper**:
```rust
async fn direct_llm_call(state: &ApiState, request: &QueryRequest) -> QueryResponse {
    let prompt = format!("Context:\n{}\n\nQuestion: {}", request.context, request.query);
    let response = state.pool.get_root_provider().complete(&prompt).await;
    QueryResponse { answer, iterations: 1, bypassed: true }
}
```

---

## Part 2: Paper-Aligned Benchmarks

**Current Coverage**: 1/4 paper benchmarks (S-NIAH only)

### Missing Benchmarks

| Paper Benchmark | Gap | Priority |
|-----------------|-----|----------|
| BrowseComp-Plus | Multi-hop QA across documents | HIGH |
| OOLONG | Long narrative reasoning | HIGH |
| OOLONG-Pairs | Pairwise document comparison | HIGH |

### New Test Cases to Add

1. **browsecomp_minimal.json** - Two-document lookup
2. **distributed_count.json** - Count across multiple log segments
3. **oolong_narrative.json** - Complex reasoning over narrative
4. **document_comparison.json** - Pairwise aggregation

### New Context Generators Needed

```rust
// In test_runner.rs
"multi_document" => generate_multi_document(&params),  // Multiple related docs
"narrative" => generate_narrative(&params),            // Story/paper with reasoning
```

---

## Part 3: README Updates

### Add Benchmark Results Section

```markdown
## Benchmark Results

### Token Savings (RLM vs Direct)

| Context Size | Queries | Pass Rate | Avg Iterations | Token Savings |
|--------------|---------|-----------|----------------|---------------|
| 41K chars    | 3       | 100%      | 2.7            | 65%           |
| 2.5K chars   | 4       | 75%       | 3.0            | -164% (overhead) |

### Key Findings
- RLM is most effective for contexts >10K chars
- Small contexts (<4K) should use direct LLM (smart bypass)
- Expected savings: 40-70% for large contexts

### Comparison to Paper
| Metric | Paper Claims | Our Results |
|--------|--------------|-------------|
| S-NIAH | 100% (simple) | 100% |
| Token Savings | "comparable or cheaper" | 65% savings on 41K |
```

---

## Implementation Order

1. **Smart Bypass** (fastest impact)
   - Add config fields
   - Implement threshold check
   - Add direct LLM path
   - Add `bypassed` flag to response

2. **README Updates** (quick win)
   - Document current results
   - Explain when RLM helps vs hurts
   - Compare to paper claims

3. **New Benchmarks** (deeper alignment)
   - Add multi-document generator
   - Create browsecomp_minimal test
   - Create distributed_count test

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/lib.rs` | Add bypass config fields |
| `src/api/mod.rs` | Add bypass check + direct path |
| `src/bin/test_runner.rs` | Add multi-document generator |
| `benchmarks/README.md` | Add results section |
| `benchmarks/*.json` | New test cases |
| `config.toml` | Add bypass_threshold setting |

---

## References

- [RLM Paper (arXiv)](https://arxiv.org/html/2512.24601v1) - Original MIT paper by Alex Zhang & Omar Khattab
- [RLM Blog Post](https://alexzhang13.github.io/blog/2025/rlm/) - Author's explanation
- [Prime Intellect RLMEnv](https://www.primeintellect.ai/blog/rlm) - Production implementation

### Paper Key Results
| Benchmark | Base Model | RLM | Improvement |
|-----------|------------|-----|-------------|
| BrowseComp-Plus | 0% | 91.33% | ∞ (didn't fit) |
| OOLONG | 44% | 56.5% | +28% |
| OOLONG-Pairs | 0.04% F1 | 58% F1 | +1,450% |

### Our Current Results (log_counting, 41K chars)
- Pass rate: 100% (3/3)
- Avg iterations: 2.7
- Token savings: 65%
