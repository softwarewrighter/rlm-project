#!/usr/bin/env python3
"""
RLM Benchmark Runner

Runs benchmark tests against the RLM server and measures:
- Accuracy (correct answers)
- Iterations (steps to answer)
- Latency (time to answer)
- Token usage (via response metadata)
"""

import json
import re
import time
import random
import httpx
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Configuration
RLM_SERVER = "http://localhost:8080"
TIMEOUT = 120.0  # seconds


@dataclass
class QueryResult:
    query: str
    expected: str
    actual: str
    match_type: str
    passed: bool
    iterations: int
    latency_ms: float
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    name: str
    category: str
    queries: list = field(default_factory=list)
    total_queries: int = 0
    passed: int = 0
    failed: int = 0
    avg_iterations: float = 0.0
    avg_latency_ms: float = 0.0


def generate_context(spec: dict) -> str:
    """Generate context based on spec."""
    generator = spec.get("context_generator")
    params = spec.get("context_params", {})

    if generator == "repeat_with_needle":
        lines = []
        for i in range(params["total_lines"]):
            if i == params["needle_line"] - 1:
                lines.append(params["needle"])
            else:
                lines.append(params["filler_line"].format(n=i + 1))
        return "\n".join(lines)

    elif generator == "generate_log":
        lines = []
        dist = params["distribution"]
        total = params["total_lines"]

        # Create log entries based on distribution
        log_entries = []
        for level, count in dist.items():
            for _ in range(count):
                timestamp = f"2024-01-{random.randint(1,28):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
                messages = {
                    "INFO": ["Request processed successfully", "User logged in", "Cache hit", "Connection established"],
                    "DEBUG": ["Variable value: x=42", "Entering function process()", "Loop iteration 5", "Memory usage: 45%"],
                    "WARN": ["High memory usage detected", "Slow query: 2.5s", "Deprecated API called", "Rate limit approaching"],
                    "ERROR": ["Connection refused", "Query timeout", "Invalid input", "Authentication failed"],
                    "FATAL": ["Out of memory", "Database unreachable", "Critical failure", "System shutdown"]
                }
                msg = random.choice(messages.get(level, ["Unknown event"]))
                log_entries.append(f"[{timestamp}] {level}: {msg}")

        random.shuffle(log_entries)
        return "\n".join(log_entries)

    else:
        # Use static context from spec
        return spec.get("context", "")


def check_match(actual: str, expected, match_type: str) -> bool:
    """Check if actual answer matches expected."""
    actual_lower = actual.lower()

    if match_type == "exact":
        return actual.strip() == expected.strip()

    elif match_type == "contains":
        if isinstance(expected, str):
            return expected.lower() in actual_lower
        return False

    elif match_type == "contains_all":
        if isinstance(expected, list):
            return all(e.lower() in actual_lower for e in expected)
        return False

    elif match_type == "regex":
        pattern = expected if isinstance(expected, str) else str(expected)
        return bool(re.search(pattern, actual, re.IGNORECASE))

    return False


def run_query(query: str, context: str) -> tuple[str, int, float, Optional[str]]:
    """Run a single query against the RLM server."""
    start = time.time()

    try:
        response = httpx.post(
            f"{RLM_SERVER}/debug",
            json={"query": query, "context": context},
            timeout=TIMEOUT
        )
        latency_ms = (time.time() - start) * 1000

        if response.status_code != 200:
            return "", 0, latency_ms, f"HTTP {response.status_code}"

        data = response.json()
        answer = data.get("answer", "")
        iterations = len(data.get("iterations", []))

        if "error" in data:
            return answer, iterations, latency_ms, data["error"]

        return answer, iterations, latency_ms, None

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        return "", 0, latency_ms, str(e)


def run_benchmark(benchmark_file: Path) -> BenchmarkResult:
    """Run a single benchmark file."""
    with open(benchmark_file) as f:
        spec = json.load(f)

    result = BenchmarkResult(
        name=spec["name"],
        category=spec.get("category", "unknown")
    )

    # Generate or load context
    context = generate_context(spec)

    print(f"\n{'='*60}")
    print(f"Benchmark: {spec['name']}")
    print(f"Category: {spec.get('category', 'unknown')}")
    print(f"Context size: {len(context)} chars")
    print(f"{'='*60}")

    for q in spec.get("queries", []):
        query = q["query"]
        expected = q["expected"]
        match_type = q.get("match_type", "contains")

        print(f"\nQuery: {query[:60]}...")

        actual, iterations, latency_ms, error = run_query(query, context)

        passed = check_match(actual, expected, match_type) if not error else False

        qr = QueryResult(
            query=query,
            expected=str(expected),
            actual=actual[:200] + "..." if len(actual) > 200 else actual,
            match_type=match_type,
            passed=passed,
            iterations=iterations,
            latency_ms=latency_ms,
            error=error
        )
        result.queries.append(qr)
        result.total_queries += 1

        if passed:
            result.passed += 1
            print(f"  ✓ PASSED ({iterations} iterations, {latency_ms:.0f}ms)")
        else:
            result.failed += 1
            print(f"  ✗ FAILED")
            if error:
                print(f"    Error: {error}")
            else:
                print(f"    Expected: {expected}")
                print(f"    Got: {qr.actual}")

    # Calculate averages
    if result.queries:
        result.avg_iterations = sum(q.iterations for q in result.queries) / len(result.queries)
        result.avg_latency_ms = sum(q.latency_ms for q in result.queries) / len(result.queries)

    return result


def main():
    """Run all benchmarks and print summary."""
    benchmark_dir = Path(__file__).parent
    benchmark_files = list(benchmark_dir.glob("*.json"))

    if not benchmark_files:
        print("No benchmark files found!")
        return

    print(f"Found {len(benchmark_files)} benchmark files")

    # Check server
    try:
        r = httpx.get(f"{RLM_SERVER}/health", timeout=5.0)
        if r.status_code != 200:
            print(f"Server not healthy: {r.status_code}")
            return
    except Exception as e:
        print(f"Cannot connect to RLM server at {RLM_SERVER}: {e}")
        print("Start the server first: cargo run --bin rlm-server")
        return

    print(f"Connected to RLM server at {RLM_SERVER}")

    results = []
    for bf in sorted(benchmark_files):
        try:
            result = run_benchmark(bf)
            results.append(result)
        except Exception as e:
            print(f"Error running {bf.name}: {e}")

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    total_passed = sum(r.passed for r in results)
    total_queries = sum(r.total_queries for r in results)

    print(f"\n{'Benchmark':<30} {'Pass':<6} {'Fail':<6} {'Iters':<8} {'Latency':<10}")
    print("-"*60)

    for r in results:
        print(f"{r.name:<30} {r.passed:<6} {r.failed:<6} {r.avg_iterations:<8.1f} {r.avg_latency_ms:<10.0f}ms")

    print("-"*60)
    accuracy = (total_passed / total_queries * 100) if total_queries else 0
    print(f"\nOverall Accuracy: {total_passed}/{total_queries} ({accuracy:.1f}%)")

    if results:
        avg_iters = sum(r.avg_iterations for r in results) / len(results)
        avg_lat = sum(r.avg_latency_ms for r in results) / len(results)
        print(f"Average Iterations: {avg_iters:.1f}")
        print(f"Average Latency: {avg_lat:.0f}ms")


if __name__ == "__main__":
    main()
