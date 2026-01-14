# VHS Tape - RLM Demo: Failure vs Success
# Run with: vhs demo/rlm-demo.tape

Output demo/rlm-demo.gif
Output demo/rlm-demo.mp4

Set FontSize 13
Set Width 1200
Set Height 900
Set Theme "Dracula"
Set Padding 20
Set TypingSpeed 50ms

# Title
Type "# RLM Demo: How Small LLMs Handle Large Documents"
Enter
Sleep 1s
Enter

# Show the document
Type "# Our test document has a hidden secret code..."
Enter
Sleep 500ms

Type "wc -lc demo/context_too_big.txt"
Enter
Sleep 2s

Type "grep -n NEPTUNE demo/context_too_big.txt"
Enter
Sleep 2s

Type "# Secret code NEPTUNE-FALCON-7749 is at line 235 (middle of doc)"
Enter
Sleep 1500ms
Enter

# PART 1: Direct query - FAILS
Type "# ═══════════════════════════════════════════════════════════════"
Enter
Type "# TEST 1: Direct query (truncated context) - FAILS"
Enter
Type "# ═══════════════════════════════════════════════════════════════"
Enter
Sleep 1s

Type "./demo/direct-query.sh"
Enter
Sleep 35s

Type "# FAILED: The secret is beyond the truncated context window!"
Enter
Sleep 2s
Enter

# PART 2: RLM query - SUCCEEDS
Type "# ═══════════════════════════════════════════════════════════════"
Enter
Type "# TEST 2: RLM query (iterative search) - SUCCEEDS"
Enter
Type "# ═══════════════════════════════════════════════════════════════"
Enter
Sleep 1s

Type "cd rlm-orchestrator"
Enter
Sleep 500ms

Type "./target/debug/rlm ../demo/context_too_big.txt 'Find the secret launch code' -v -m qwen3:14b -u http://big72.local:11434"
Enter
Sleep 50s

Enter
Type "# SUCCESS! RLM used 'find' to search - found the code!"
Enter
Sleep 3s
