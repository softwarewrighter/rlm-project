# War and Peace Family Tree - Efficient Approach

## The Problem

**Input:** 3.3MB text, 65,660 lines
**Current approach:** Split into 300+ chunks, process each with LLM sequentially
**Result:** 50+ chunks taking 3385+ seconds before hitting iteration limit = FAILURE

## Why Current Approach Fails

1. **Blind chunking** - treats all text equally, wastes LLM calls on irrelevant content
2. **Sequential processing** - O(n) LLM calls where n=300+
3. **No pre-filtering** - sends descriptions of battles, scenery, etc. to LLM
4. **Iteration limits** - can't complete 300 chunks in 50 iterations

## Efficient Approach: Extract-Filter-Analyze

### Phase 1: Deterministic Extraction (L3 CLI - FAST)

Use Rust code to extract ONLY relevant text:

```rust
// Pattern 1: Lines with relationship words + capitalized names
// "Prince Andrew was the son of old Prince Bolkonsky"
// "Natasha was Princess Mary's sister-in-law"

let relationship_words = ["father", "mother", "son", "daughter",
    "brother", "sister", "wife", "husband", "married", "family",
    "uncle", "aunt", "nephew", "niece", "cousin", "child"];

// Pattern 2: Lines with Russian noble titles + names
let titles = ["Prince", "Princess", "Count", "Countess",
    "Duke", "Duchess", "Baron", "General"];

// Extract lines containing: (title OR relationship_word) AND capitalized_name
```

**Expected reduction:** 65,000 lines → ~2,000-5,000 relevant lines (~95% reduction)

### Phase 2: Name Extraction & Frequency (L3 CLI - FAST)

```rust
// Extract all capitalized word sequences (potential names)
// Count frequencies
// Filter to names appearing 10+ times (main characters)
// Group aliases: "Natasha" = "Natásha" = "Natasha Rostova"
```

**Output:** List of ~50-100 main characters with frequency counts

### Phase 3: Relationship Sentence Extraction (L3 CLI - FAST)

For each main character:
```rust
// Find sentences mentioning character + relationship word
// "Natasha" + "married" → "Natasha married Pierre"
// "Prince Andrew" + "son" → "Prince Andrew was the son of..."
```

**Output:** ~500-1000 relationship sentences (maybe 50KB instead of 3.3MB)

### Phase 4: LLM Analysis (L4 - Semantic)

NOW use LLM on the filtered data:

```json
{
  "op": "llm_reduce",
  "directive": "Extract family relationships from these sentences. Format: NAME1 -[relationship]-> NAME2",
  "on": "relationship_sentences",
  "store": "relationships"
}
```

**Expected:** 3-5 LLM calls on ~50KB instead of 300+ calls on 3.3MB

### Phase 5: Family Tree Synthesis (L4 - Semantic)

```json
{
  "op": "llm_query",
  "prompt": "Build family trees for the Rostov, Bolkonsky, Kuragin, and Bezukhov families from: ${relationships}",
  "store": "family_tree"
}
```

## Implementation Options

### Option A: Pre-processing Script (Recommended for Demo)
Create a Rust CLI tool that pre-processes War and Peace:
```bash
# Run once to create filtered version
./tools/extract-characters war-and-peace.txt > war-peace-characters.txt
```
Then RLM only processes the filtered file (~50KB).

### Option B: RLM rust_cli_intent (Dynamic)
Let RLM generate the extraction code on-the-fly:
```json
{"op": "rust_cli_intent", "task": "Extract lines containing relationship words (father, mother, son, daughter, married, etc.) near capitalized names", "store": "relevant_lines"}
```

### Option C: Hybrid (Best UX)
1. RLM uses L3 CLI to extract character names (fast)
2. RLM uses L3 CLI to extract relationship sentences (fast)
3. RLM uses L4 LLM only on filtered data (semantic)

## Expected Performance

| Approach | LLM Calls | Time | Success |
|----------|-----------|------|---------|
| Current (blind chunking) | 300+ | 3385s+ | FAIL |
| New (extract-filter-analyze) | 5-10 | ~60-120s | SUCCESS |

## Key Insight

**Don't send noise to LLMs. Use deterministic code to filter first.**

The family tree task is ~95% extraction/filtering (deterministic) and ~5% semantic reasoning (LLM). Current approach inverts this ratio.

## Demo Files Needed

```
demo/l4/
  data/
    war-peace-characters.txt    # Pre-extracted character data (~50KB)
  family-tree-demo.sh           # Demo script
  tools/
    extract-characters.rs       # Rust extraction tool
```

## Sanity Checks Needed

Before starting llm_reduce:
1. Calculate number of chunks
2. If chunks > max_iterations, FAIL FAST with helpful error:
   "Context too large: 300 chunks would exceed 50 iteration limit.
    Use L3 CLI to filter relevant content first."
