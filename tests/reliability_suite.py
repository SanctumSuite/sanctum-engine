#!/usr/bin/env python3
"""Reliability suite: tests that the engine produces valid JSON at >= 95% rate.

Runs 100 varied extract_json tasks and reports success/failure rates.
Requires Ollama running with ministral-3:latest.

Usage:
    python tests/reliability_suite.py [--tasks 100] [--model ministral-3:latest]
"""
import argparse
import asyncio
import json
import sys
import time

# Add parent dir to path
sys.path.insert(0, ".")

from app.runtimes.ollama import OllamaRuntime
from app.core.output_validator import validate_json_output
from app.core.context_manager import count_tokens

# Test prompts of varying complexity
TEST_CASES = [
    # --- Short prompts, small JSON ---
    {
        "name": "simple_key_value",
        "system": "Respond with valid JSON only.",
        "user": 'Return a JSON object with keys "color" and "shape".',
        "schema": {"type": "object", "required": ["color", "shape"]},
    },
    {
        "name": "yes_no_judgment",
        "system": "You are an analyst. Respond with JSON only.",
        "user": 'Is water wet? Return {"answer": "yes" or "no", "confidence": "high/medium/low"}.',
        "schema": {"type": "object", "required": ["answer", "confidence"]},
    },
    {
        "name": "three_items_list",
        "system": "Respond with valid JSON only.",
        "user": 'List 3 countries in the Middle East. Return {"countries": ["..."]}.',
        "schema": {"type": "object", "required": ["countries"]},
    },
    # --- Medium prompts, structured output ---
    {
        "name": "evidence_extraction_short",
        "system": "You are an intelligence analyst. Extract evidence items from the text. Respond with JSON only.",
        "user": '''Extract evidence from this text:
"The Treasury Secretary announced new sanctions on Iranian oil exports. The sanctions target 50 vessels carrying Iranian crude."
Return: {"evidence_items": [{"content": "...", "context": "..."}]}''',
        "schema": {"type": "object", "required": ["evidence_items"]},
    },
    {
        "name": "source_rating",
        "system": "Rate this source using the Admiralty system. Respond with JSON only.",
        "user": '''Source: Reuters news article dated today about US-Iran negotiations.
Return: {"reliability": "A-F letter", "credibility": "1-6 number", "justification": "brief reason"}''',
        "schema": {"type": "object", "required": ["reliability", "credibility", "justification"]},
    },
    {
        "name": "hypothesis_generation",
        "system": "You are an intelligence analyst. Generate competing hypotheses. Respond with JSON only.",
        "user": '''Given: "Iran has increased uranium enrichment to 60%". Generate 3 competing hypotheses.
Return: {"hypotheses": [{"title": "...", "description": "..."}]}''',
        "schema": {"type": "object", "required": ["hypotheses"]},
    },
    # --- Longer prompts ---
    {
        "name": "document_summary",
        "system": "Summarize this document for intelligence analysis. Respond with JSON only.",
        "user": '''Document: The ongoing negotiations between the United States and Iran regarding the nuclear program have entered a new phase. Recent reports indicate that both sides have shown willingness to return to the table after months of stalemate. The European Union has offered to mediate, with the French and German foreign ministers expressing optimism about reaching a preliminary agreement by year's end. However, hardliners in both countries remain opposed to any deal that doesn't address ballistic missile development and regional proxy activities. Israel has repeatedly warned that any agreement must include stringent verification mechanisms, citing Iran's history of covert nuclear activities at previously undisclosed facilities.

Return: {"summary": "...", "key_topics": ["..."], "entities_mentioned": ["..."]}''',
        "schema": {"type": "object", "required": ["summary", "key_topics"]},
    },
    {
        "name": "matrix_rating",
        "system": "Rate the consistency of this evidence with the hypothesis. Respond with JSON only.",
        "user": '''Evidence: "Iran has increased uranium enrichment to 60%"
Hypothesis: "Iran is pursuing nuclear weapons capability"
Rate as CC (very consistent), C (consistent), N (neutral), I (inconsistent), II (very inconsistent).
Return: {"rating": "...", "reasoning": "..."}''',
        "schema": {"type": "object", "required": ["rating", "reasoning"]},
    },
    {
        "name": "search_queries",
        "system": "Generate search queries for OSINT collection. Respond with JSON only.",
        "user": '''Analytic question: "What is the likelihood of a US-Iran military confrontation in the next 6 months?"
Generate 5 diverse search queries covering different perspectives.
Return: {"queries": ["..."]}''',
        "schema": {"type": "object", "required": ["queries"]},
    },
    {
        "name": "assumptions_check",
        "system": "Identify key assumptions. Respond with JSON only.",
        "user": '''Hypothesis: "Iran will agree to a new nuclear deal within 6 months."
Identify 3 key assumptions underlying this hypothesis.
Return: {"assumptions": [{"content": "...", "assessment": "solid/questionable/unsupported"}]}''',
        "schema": {"type": "object", "required": ["assumptions"]},
    },
]


async def run_suite(num_tasks: int, model: str):
    runtime = OllamaRuntime(host="http://localhost:11434")

    # Verify connectivity
    if not await runtime.is_healthy():
        print("ERROR: Ollama is not running at localhost:11434")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Sanctum Engine Reliability Suite")
    print(f"  Model: {model}")
    print(f"  Tasks: {num_tasks}")
    print(f"{'='*60}\n")

    results = {
        "total": 0,
        "success": 0,
        "repaired": 0,
        "schema_valid": 0,
        "failed": 0,
        "errors": [],
        "latencies": [],
    }

    # Cycle through test cases to reach num_tasks
    task_index = 0
    for i in range(num_tasks):
        case = TEST_CASES[task_index % len(TEST_CASES)]
        task_index += 1

        results["total"] += 1
        label = f"[{i+1:3d}/{num_tasks}] {case['name']:30s}"

        start = time.monotonic()
        try:
            response = await runtime.generate(
                model=model,
                messages=[
                    {"role": "system", "content": case["system"] + "\n\nYou MUST respond with valid JSON only. No markdown, no code fences."},
                    {"role": "user", "content": case["user"]},
                ],
                temperature=0.3,
                max_tokens=2048,
                num_ctx=8192,
            )
            elapsed = int((time.monotonic() - start) * 1000)
            results["latencies"].append(elapsed)

            validation = validate_json_output(response.content, schema=case.get("schema"))

            if validation.valid:
                results["success"] += 1
                if validation.was_repaired:
                    results["repaired"] += 1
                results["schema_valid"] += 1
                status = "PASS (repaired)" if validation.was_repaired else "PASS"
                print(f"  {label} {status:18s} {elapsed:6d}ms  {response.tokens_out:4d} tok")
            else:
                results["failed"] += 1
                results["errors"].append({
                    "case": case["name"],
                    "error": validation.error_code,
                    "message": validation.error_message[:100] if validation.error_message else "",
                })
                print(f"  {label} {'FAIL':18s} {elapsed:6d}ms  {validation.error_code}")

        except Exception as e:
            elapsed = int((time.monotonic() - start) * 1000)
            results["failed"] += 1
            results["errors"].append({
                "case": case["name"],
                "error": type(e).__name__,
                "message": str(e)[:100],
            })
            print(f"  {label} {'ERROR':18s} {elapsed:6d}ms  {type(e).__name__}: {str(e)[:60]}")

    # Report
    success_rate = results["success"] / results["total"] if results["total"] > 0 else 0
    avg_latency = sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total:          {results['total']}")
    print(f"  Success:        {results['success']} ({success_rate:.1%})")
    print(f"  Repaired:       {results['repaired']}")
    print(f"  Failed:         {results['failed']}")
    print(f"  Avg latency:    {avg_latency:.0f}ms")
    print(f"")
    print(f"  Target: >= 95% success rate")
    print(f"  Result: {'PASS' if success_rate >= 0.95 else 'FAIL'}")
    print(f"{'='*60}\n")

    if results["errors"]:
        print("  Failures:")
        for err in results["errors"]:
            print(f"    - {err['case']}: {err['error']} — {err['message']}")
        print()

    return success_rate >= 0.95


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanctum Engine Reliability Suite")
    parser.add_argument("--tasks", type=int, default=100, help="Number of tasks to run")
    parser.add_argument("--model", type=str, default="ministral-3:latest", help="Model to test")
    args = parser.parse_args()

    passed = asyncio.run(run_suite(args.tasks, args.model))
    sys.exit(0 if passed else 1)
