# VHS Tape - RLM Demo: Finding a Needle in War and Peace
# Run with: vhs demo/war-and-peace-demo.tape

Output demo/war-and-peace-demo.gif
Output demo/war-and-peace-demo.mp4

Set FontSize 12
Set Width 1400
Set Height 900
Set Theme "Dracula"
Set Padding 20
Set TypingSpeed 40ms

# Title
Type "# RLM Demo: Finding a Needle in War and Peace (3.2MB)"
Enter
Sleep 1s

# Show the document stats
Type "wc -lc demo/war-and-peace-needle.txt"
Enter
Sleep 2s

Type "# 65,666 lines, 3.2MB - way too big for any LLM context window!"
Enter
Sleep 1500ms
Enter

# Run RLM
Type "cd rlm-orchestrator"
Enter
Sleep 500ms

Type "./target/release/rlm ../demo/war-and-peace-needle.txt 'What is the password to Prince Andrei secret vault?' -v -m qwen3:14b -u http://big72.local:11434"
Enter
Sleep 90s

Enter
Type "# Found it! RLM searched 3.2MB using just a few targeted queries."
Enter
Sleep 3s
