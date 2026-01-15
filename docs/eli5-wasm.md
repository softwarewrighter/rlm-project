# ELI5: Teaching LLMs to Write rust_wasm Commands

This guide helps LLM prompt engineers understand how to teach language models to generate `rust_wasm` commands for RLM.

## What is rust_wasm?

When analyzing large documents, sometimes the built-in commands (`find`, `regex`, `count`) aren't enough. The `rust_wasm` command lets the LLM write custom Rust code that gets compiled and run safely in a sandbox.

**Think of it like this:** The LLM is a detective investigating a huge crime scene (the document). Built-in commands are like standard tools (magnifying glass, fingerprint kit). But sometimes the detective needs a custom gadget - that's `rust_wasm`.

## The Basic Pattern

Every `rust_wasm` command follows this pattern:

```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(input: &str) -> String { /* your code */ }",
  "on": "$variable_or_context",
  "store": "result_variable"
}
```

**The rules are simple:**
1. Write a function called `analyze`
2. It takes a string input (`&str`)
3. It returns a String
4. That's it!

## Example 1: Count Specific Patterns

**Task:** Count lines containing both "ERROR" and "timeout"

Built-in commands can't do AND logic easily. But Rust can:

```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(input: &str) -> String { input.lines().filter(|l| l.contains(\"ERROR\") && l.contains(\"timeout\")).count().to_string() }",
  "store": "timeout_errors"
}
```

**What the code does:**
- `input.lines()` - splits text into lines
- `.filter(|l| ...)` - keeps only lines matching our condition
- `.count()` - counts how many
- `.to_string()` - converts number to text (required!)

## Example 2: Word Frequency

**Task:** Find the 5 most common words

```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(input: &str) -> String { let mut counts: HashMap<&str, usize> = HashMap::new(); for word in input.split_whitespace() { *counts.entry(word).or_insert(0) += 1; } let mut pairs: Vec<_> = counts.into_iter().collect(); pairs.sort_by(|a, b| b.1.cmp(&a.1)); pairs.iter().take(5).map(|(w, c)| format!(\"{}: {}\", w, c)).collect::<Vec<_>>().join(\", \") }",
  "store": "top_words"
}
```

**What the code does:**
- Creates a HashMap to count words
- Loops through each word, incrementing its count
- Sorts by count (highest first)
- Takes top 5 and formats as "word: count"

## Example 3: Sum Numbers

**Task:** Add up all numbers in the text

```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(input: &str) -> String { let sum: i64 = input.split_whitespace().filter_map(|s| s.parse::<i64>().ok()).sum(); sum.to_string() }",
  "on": "$data",
  "store": "total"
}
```

**What the code does:**
- Splits into words
- Tries to parse each as a number (ignores non-numbers)
- Sums them up

## Example 4: Extract Structured Data

**Task:** Extract all IP addresses

```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(input: &str) -> String { let mut ips = Vec::new(); for word in input.split_whitespace() { let parts: Vec<&str> = word.split('.').collect(); if parts.len() == 4 && parts.iter().all(|p| p.parse::<u8>().is_ok()) { ips.push(word.to_string()); } } ips.join(\", \") }",
  "store": "ip_addresses"
}
```

## Example 5: Custom Aggregation

**Task:** Calculate average line length

```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(input: &str) -> String { let lines: Vec<&str> = input.lines().collect(); if lines.is_empty() { return \"0\".to_string(); } let total: usize = lines.iter().map(|l| l.len()).sum(); let avg = total / lines.len(); format!(\"{} chars average across {} lines\", avg, lines.len()) }",
  "store": "avg_length"
}
```

## When to Use rust_wasm

**USE rust_wasm when:**
- You need AND/OR logic across multiple conditions
- You need to aggregate data (sum, average, frequency)
- You need custom parsing or extraction
- Built-in commands would require many steps

**DON'T use rust_wasm when:**
- Simple `find` or `regex` works
- You just need to count lines/words/chars (use `count`)
- You just need a slice of text (use `slice` or `lines`)

## What's Available in Your Code

**You CAN use:**
- All basic Rust: loops, conditionals, iterators
- String operations: `split`, `contains`, `lines`, `trim`, etc.
- Collections: `HashMap`, `HashSet`, `Vec`, `BTreeMap`, `BTreeSet`, `VecDeque`
- Number types: `i32`, `i64`, `f64`, `usize`, etc.
- Formatting: `format!`, `to_string()`

**You CANNOT use:**
- File I/O (`std::fs`)
- Network (`std::net`)
- External crates
- System calls

## Common Patterns for LLMs

### Pattern: Filter and Count
```rust
input.lines().filter(|l| /* condition */).count().to_string()
```

### Pattern: Transform and Join
```rust
input.lines().map(|l| /* transform */).collect::<Vec<_>>().join("\n")
```

### Pattern: Aggregate with HashMap
```rust
let mut map: HashMap<K, V> = HashMap::new();
for item in input.split(...) {
    // update map
}
// format result from map
```

### Pattern: Parse and Calculate
```rust
let numbers: Vec<i64> = input.split_whitespace()
    .filter_map(|s| s.parse().ok())
    .collect();
// calculate with numbers
```

## Tips for Prompt Engineers

1. **Show examples in your system prompt** - LLMs learn patterns from examples
2. **Emphasize the signature** - `pub fn analyze(input: &str) -> String` is mandatory
3. **Remind about String return** - Numbers must be converted with `.to_string()`
4. **Keep code on one line** - JSON doesn't handle newlines well in strings
5. **Use `store`** - Always store results for later use

## Debugging Tips

If the LLM generates code that fails:

1. **Compilation error** - Check syntax, missing semicolons, type mismatches
2. **Runtime error** - Check for division by zero, empty iterators
3. **Wrong output** - The code ran but logic was wrong

The error message will be shown to the LLM so it can fix and retry.
