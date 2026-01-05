#!/usr/bin/env python3
"""
rlm.py - Recursive Language Model Implementation

A practical RLM implementation supporting:
- Local Ollama servers (single or distributed)
- DeepSeek API
- OpenAI-compatible APIs (including local llama.cpp servers)
- Claude API (for comparison)

Usage:
    python rlm.py --query "Find all TODO items" --context-file large_codebase.txt
    python rlm.py --query "Summarize this document" --context-file doc.md --provider ollama
    python rlm.py --interactive  # REPL mode

Requirements:
    pip install httpx rich typer pydantic --break-system-packages
"""

import asyncio
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

console = Console()

# ============================================================================
# Configuration
# ============================================================================

class ProviderType(str, Enum):
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    OPENAI_COMPATIBLE = "openai"
    CLAUDE = "claude"
    LLAMA_CPP = "llama_cpp"


@dataclass
class OllamaConfig:
    """Configuration for an Ollama server."""
    host: str = "localhost"
    port: int = 11434
    model: str = "qwen2.5-coder:32b"
    name: str = "local"
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek API."""
    api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"


@dataclass
class OpenAICompatibleConfig:
    """Configuration for OpenAI-compatible APIs (including local llama.cpp)."""
    base_url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"  # Or local model name


@dataclass
class ClaudeConfig:
    """Configuration for Claude API."""
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    model: str = "claude-sonnet-4-20250514"


@dataclass
class RLMConfig:
    """Main RLM configuration."""
    max_iterations: int = 30
    max_output_chars: int = 50000
    timeout_seconds: int = 300
    sub_call_limit: int = 100
    
    # Provider configs
    ollama_servers: list[OllamaConfig] = field(default_factory=list)
    deepseek: Optional[DeepSeekConfig] = None
    openai_compatible: Optional[OpenAICompatibleConfig] = None
    claude: Optional[ClaudeConfig] = None
    
    # Which provider to use for root vs sub calls
    root_provider: ProviderType = ProviderType.OLLAMA
    sub_provider: ProviderType = ProviderType.OLLAMA


# ============================================================================
# LLM Providers
# ============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def query(self, prompt: str, system: Optional[str] = None) -> str:
        """Query the LLM and return the response text."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is available."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama API provider."""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def query(self, prompt: str, system: Optional[str] = None) -> str:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        
        response = await self.client.post(
            f"{self.config.base_url}/api/generate",
            json=payload
        )
        response.raise_for_status()
        return response.json()["response"]
    
    async def health_check(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.config.base_url}/api/tags",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False
    
    @property
    def name(self) -> str:
        return f"Ollama ({self.config.name}: {self.config.model})"


class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider."""
    
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def query(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.post(
            f"{self.config.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.config.model,
                "messages": messages,
                "max_tokens": 4096,
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    async def health_check(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.config.base_url}/models",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False
    
    @property
    def name(self) -> str:
        return f"DeepSeek ({self.config.model})"


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible API provider (works with llama.cpp server, vLLM, etc.)."""
    
    def __init__(self, config: OpenAICompatibleConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def query(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        response = await self.client.post(
            f"{self.config.base_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": self.config.model,
                "messages": messages,
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    async def health_check(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.config.base_url}/v1/models",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False
    
    @property
    def name(self) -> str:
        return f"OpenAI-Compatible ({self.config.base_url})"


class ClaudeProvider(LLMProvider):
    """Claude API provider."""
    
    def __init__(self, config: ClaudeConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def query(self, prompt: str, system: Optional[str] = None) -> str:
        payload = {
            "model": self.config.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system:
            payload["system"] = system
        
        response = await self.client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.config.api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            json=payload
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]
    
    async def health_check(self) -> bool:
        # Claude doesn't have a simple health endpoint
        return bool(self.config.api_key)
    
    @property
    def name(self) -> str:
        return f"Claude ({self.config.model})"


# ============================================================================
# RLM Core
# ============================================================================

@dataclass
class IterationRecord:
    """Record of a single RLM iteration."""
    step: int
    code: Optional[str]
    output: str
    sub_calls: int
    error: Optional[str] = None


@dataclass
class RLMResult:
    """Result of an RLM query."""
    answer: str
    iterations: int
    history: list[IterationRecord]
    total_sub_calls: int
    success: bool = True
    error: Optional[str] = None


class RLMOrchestrator:
    """
    Recursive Language Model Orchestrator.
    
    Implements the RLM pattern from the paper:
    1. Load context as a variable in a REPL environment
    2. Let the LLM write code to examine/transform the context
    3. Provide llm_query() for recursive sub-LM calls
    4. Iterate until FINAL() answer is produced
    """
    
    CODE_PATTERN = re.compile(r"```(?:repl|python)\n([\s\S]*?)```")
    FINAL_PATTERN = re.compile(r"FINAL\(([\s\S]*?)\)")
    FINAL_VAR_PATTERN = re.compile(r"FINAL_VAR\((\w+)\)")
    
    def __init__(
        self,
        config: RLMConfig,
        root_provider: LLMProvider,
        sub_provider: LLMProvider
    ):
        self.config = config
        self.root_provider = root_provider
        self.sub_provider = sub_provider
        
        # REPL state
        self.context_store: dict[str, Any] = {}
        self.sub_call_count = 0
    
    def _build_system_prompt(self, context_len: int, context_type: str = "text") -> str:
        """Build the RLM system prompt."""
        return f"""You are an RLM (Recursive Language Model) agent tasked with answering queries over large contexts.

Your context is a {context_type} with {context_len:,} total characters.

The REPL environment provides:
1. `context` - the full input (may be huge, use programmatic access)
2. `llm_query(prompt)` - recursive sub-LM call for semantic analysis (can handle ~500K chars)
3. Standard Python: re, json, collections, itertools, etc.

STRATEGY:
1. First, probe the context structure (print samples, check length, understand format)
2. Filter/chunk based on content type (lines, paragraphs, JSON objects, etc.)
3. Use llm_query() for semantic analysis that can't be done with regex/code
4. Aggregate results in variables
5. Return FINAL(your_answer) or FINAL_VAR(variable_name) when ready

CRITICAL RULES:
- Write Python code in ```repl or ```python blocks
- You will see TRUNCATED outputs - store important data in variables
- For semantic tasks (summarization, classification), USE llm_query()
- For syntactic tasks (counting, filtering), USE code
- Don't try to print the entire context - it will be truncated!

EXAMPLE:
```repl
# Step 1: Understand the structure
print(f"Context length: {{len(context)}} chars")
print(f"First 500 chars: {{context[:500]}}")
print(f"Number of lines: {{context.count(chr(10))}}")
```

Then based on output:
```repl
# Step 2: Process chunks with sub-LM
lines = context.split('\\n')
results = []
for i in range(0, len(lines), 100):  # Process 100 lines at a time
    chunk = '\\n'.join(lines[i:i+100])
    answer = llm_query(f"In this text, find mentions of X: {{chunk}}")
    results.append(answer)
print(f"Processed {{len(results)}} chunks")
```

Finally:
```repl
# Step 3: Aggregate and conclude
final_answer = llm_query(f"Combine these findings into a final answer: {{results}}")
print(final_answer)
```
FINAL(final_answer)

Or if the answer is in a variable:
FINAL_VAR(final_answer)"""

    def _build_iteration_prompt(
        self,
        query: str,
        history: list[IterationRecord]
    ) -> str:
        """Build the prompt for this iteration."""
        prompt_parts = [f"QUERY: {query}\n"]
        
        if history:
            prompt_parts.append("PREVIOUS STEPS:")
            # Show last 5 iterations to avoid context overflow
            for record in history[-5:]:
                prompt_parts.append(f"\n[Step {record.step}]")
                if record.code:
                    prompt_parts.append(f"Code:\n```python\n{record.code}\n```")
                if record.error:
                    prompt_parts.append(f"Error: {record.error}")
                else:
                    prompt_parts.append(f"Output:\n{record.output}")
        
        prompt_parts.append("\nWhat's your next step? Write code in ```repl blocks, or provide FINAL(answer).")
        return "\n".join(prompt_parts)
    
    def _create_llm_query_function(self) -> Callable[[str], str]:
        """Create the llm_query function for the REPL."""
        async def _async_query(prompt: str) -> str:
            self.sub_call_count += 1
            if self.sub_call_count > self.config.sub_call_limit:
                return f"ERROR: Sub-call limit ({self.config.sub_call_limit}) exceeded"
            return await self.sub_provider.query(prompt)
        
        def llm_query(prompt: str) -> str:
            """Query a sub-LLM with the given prompt."""
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, need to run in executor
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(
                            asyncio.run,
                            _async_query(prompt)
                        )
                        return future.result(timeout=self.config.timeout_seconds)
                else:
                    return loop.run_until_complete(_async_query(prompt))
            except Exception as e:
                return f"ERROR in llm_query: {e}"
        
        return llm_query
    
    def _execute_code(self, code: str) -> tuple[str, Optional[str]]:
        """Execute Python code in the REPL environment."""
        # Build execution environment
        env = {
            "context": self.context_store.get("context", ""),
            "llm_query": self._create_llm_query_function(),
            "re": __import__("re"),
            "json": __import__("json"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            **self.context_store
        }
        
        stdout_capture = StringIO()
        error = None
        
        try:
            with redirect_stdout(stdout_capture):
                exec(code, env)
            
            # Update context store with new variables
            for key, value in env.items():
                if not key.startswith("_") and key not in ("context", "llm_query", "re", "json", "collections", "itertools"):
                    self.context_store[key] = value
                    
        except Exception as e:
            error = str(e)
        
        output = stdout_capture.getvalue()
        return output, error
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract code from ```repl or ```python blocks."""
        match = self.CODE_PATTERN.search(response)
        return match.group(1) if match else None
    
    def _extract_final(self, response: str) -> Optional[str]:
        """Check for FINAL() or FINAL_VAR() answer."""
        # Check for FINAL(answer)
        if match := self.FINAL_PATTERN.search(response):
            return match.group(1).strip()
        
        # Check for FINAL_VAR(variable_name)
        if match := self.FINAL_VAR_PATTERN.search(response):
            var_name = match.group(1).strip()
            if var_name in self.context_store:
                value = self.context_store[var_name]
                return str(value) if not isinstance(value, str) else value
            return f"Variable '{var_name}' not found"
        
        return None
    
    def _truncate_output(self, output: str) -> str:
        """Truncate output to configured maximum."""
        max_chars = self.config.max_output_chars
        if len(output) <= max_chars:
            return output
        
        half = max_chars // 2
        return (
            f"{output[:half]}\n"
            f"... [TRUNCATED {len(output) - max_chars:,} chars] ...\n"
            f"{output[-half:]}"
        )
    
    async def process(
        self,
        query: str,
        context: str,
        context_type: str = "text",
        verbose: bool = True
    ) -> RLMResult:
        """
        Process an RLM query.
        
        Args:
            query: The question to answer
            context: The input context (can be huge)
            context_type: Description of context type
            verbose: Whether to print progress
        
        Returns:
            RLMResult with the answer and execution history
        """
        # Reset state
        self.context_store = {"context": context}
        self.sub_call_count = 0
        history: list[IterationRecord] = []
        
        system_prompt = self._build_system_prompt(len(context), context_type)
        
        if verbose:
            console.print(Panel(
                f"[bold]Query:[/bold] {query}\n"
                f"[bold]Context:[/bold] {len(context):,} chars ({context_type})\n"
                f"[bold]Root LLM:[/bold] {self.root_provider.name}\n"
                f"[bold]Sub LLM:[/bold] {self.sub_provider.name}",
                title="RLM Starting",
                border_style="blue"
            ))
        
        for iteration in range(self.config.max_iterations):
            if verbose:
                console.print(f"\n[bold cyan]━━━ Iteration {iteration + 1} ━━━[/bold cyan]")
            
            # Build prompt for this iteration
            prompt = self._build_iteration_prompt(query, history)
            
            # Query root LLM
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
                ) as progress:
                    if verbose:
                        progress.add_task("Querying root LLM...", total=None)
                    response = await self.root_provider.query(prompt, system_prompt)
            except Exception as e:
                return RLMResult(
                    answer="",
                    iterations=iteration + 1,
                    history=history,
                    total_sub_calls=self.sub_call_count,
                    success=False,
                    error=f"Root LLM query failed: {e}"
                )
            
            # Extract code first - only check for FINAL if no code block
            code = self._extract_code(response)

            # If no code block, check for final answer
            if not code:
                if final_answer := self._extract_final(response):
                    if verbose:
                        console.print(Panel(
                            final_answer,
                            title="[bold green]FINAL ANSWER[/bold green]",
                            border_style="green"
                        ))
                    return RLMResult(
                        answer=final_answer,
                        iterations=iteration + 1,
                        history=history,
                        total_sub_calls=self.sub_call_count,
                        success=True
                    )
            
            if code:
                if verbose:
                    console.print(Syntax(code, "python", theme="monokai", line_numbers=True))
                
                sub_calls_before = self.sub_call_count
                output, error = self._execute_code(code)
                sub_calls_made = self.sub_call_count - sub_calls_before
                
                truncated_output = self._truncate_output(output)
                
                if verbose:
                    if error:
                        console.print(f"[bold red]Error:[/bold red] {error}")
                    else:
                        console.print(Panel(
                            truncated_output or "(no output)",
                            title=f"Output (sub-calls: {sub_calls_made})",
                            border_style="yellow"
                        ))
                
                history.append(IterationRecord(
                    step=iteration + 1,
                    code=code,
                    output=truncated_output,
                    sub_calls=sub_calls_made,
                    error=error
                ))
            else:
                if verbose:
                    console.print("[yellow]No code block found in response[/yellow]")
                    console.print(Panel(response[:1000], title="Raw Response"))
                
                history.append(IterationRecord(
                    step=iteration + 1,
                    code=None,
                    output="No code block found",
                    sub_calls=0
                ))
        
        # Max iterations reached
        return RLMResult(
            answer="",
            iterations=self.config.max_iterations,
            history=history,
            total_sub_calls=self.sub_call_count,
            success=False,
            error=f"Max iterations ({self.config.max_iterations}) reached without FINAL answer"
        )


# ============================================================================
# CLI Interface
# ============================================================================

def create_default_config() -> RLMConfig:
    """Create a default configuration from environment."""
    config = RLMConfig()
    
    # Default Ollama server
    config.ollama_servers = [
        OllamaConfig(
            host=os.environ.get("OLLAMA_HOST", "localhost"),
            port=int(os.environ.get("OLLAMA_PORT", "11434")),
            model=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:32b"),
            name="local"
        )
    ]
    
    # DeepSeek if API key is set
    if os.environ.get("DEEPSEEK_API_KEY"):
        config.deepseek = DeepSeekConfig()
    
    # Claude if API key is set
    if os.environ.get("ANTHROPIC_API_KEY"):
        config.claude = ClaudeConfig()
    
    return config


def get_provider(config: RLMConfig, provider_type: ProviderType) -> LLMProvider:
    """Get a provider instance by type."""
    if provider_type == ProviderType.OLLAMA:
        if not config.ollama_servers:
            raise ValueError("No Ollama servers configured")
        return OllamaProvider(config.ollama_servers[0])
    
    elif provider_type == ProviderType.DEEPSEEK:
        if not config.deepseek:
            raise ValueError("DeepSeek not configured (set DEEPSEEK_API_KEY)")
        return DeepSeekProvider(config.deepseek)
    
    elif provider_type == ProviderType.OPENAI_COMPATIBLE:
        if not config.openai_compatible:
            config.openai_compatible = OpenAICompatibleConfig()
        return OpenAICompatibleProvider(config.openai_compatible)
    
    elif provider_type == ProviderType.CLAUDE:
        if not config.claude:
            raise ValueError("Claude not configured (set ANTHROPIC_API_KEY)")
        return ClaudeProvider(config.claude)
    
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


async def main_async(
    query: str,
    context_file: Path,
    provider: str = "ollama",
    sub_provider: Optional[str] = None,
    verbose: bool = True
) -> RLMResult:
    """Main async entry point."""
    # Load context
    context = context_file.read_text()
    
    # Create config and providers
    config = create_default_config()
    
    root = get_provider(config, ProviderType(provider))
    sub = get_provider(config, ProviderType(sub_provider or provider))
    
    # Create orchestrator
    rlm = RLMOrchestrator(config, root, sub)
    
    # Process query
    return await rlm.process(query, context, verbose=verbose)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RLM - Recursive Language Model for processing large contexts"
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Query to answer"
    )
    parser.add_argument(
        "--context-file", "-c",
        type=Path,
        required=True,
        help="Path to context file"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["ollama", "deepseek", "openai", "claude"],
        default="ollama",
        help="Root LLM provider (default: ollama)"
    )
    parser.add_argument(
        "--sub-provider", "-s",
        choices=["ollama", "deepseek", "openai", "claude"],
        help="Sub-LLM provider (default: same as root)"
    )
    parser.add_argument(
        "--quiet", "-Q",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )
    
    args = parser.parse_args()
    
    if not args.context_file.exists():
        console.print(f"[red]Error: Context file not found: {args.context_file}[/red]")
        sys.exit(1)
    
    result = asyncio.run(main_async(
        query=args.query,
        context_file=args.context_file,
        provider=args.provider,
        sub_provider=args.sub_provider,
        verbose=not args.quiet
    ))
    
    if args.json:
        output = {
            "success": result.success,
            "answer": result.answer,
            "iterations": result.iterations,
            "total_sub_calls": result.total_sub_calls,
            "error": result.error,
            "history": [
                {
                    "step": h.step,
                    "code": h.code,
                    "output": h.output[:500] if h.output else None,
                    "sub_calls": h.sub_calls,
                    "error": h.error
                }
                for h in result.history
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        if result.success:
            console.print("\n[bold green]═══ SUCCESS ═══[/bold green]")
            console.print(f"Answer: {result.answer}")
            console.print(f"Iterations: {result.iterations}")
            console.print(f"Sub-calls: {result.total_sub_calls}")
        else:
            console.print("\n[bold red]═══ FAILED ═══[/bold red]")
            console.print(f"Error: {result.error}")
    
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
