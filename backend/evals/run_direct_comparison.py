#!/usr/bin/env python3
"""Direct comparison of Baseline vs CRAG retrieval.

Runs both retrievers on the same queries and compares metrics side-by-side.
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.utils.llama_setup import configure_llama_index
from src.rag.supabase_rag import SupabaseRAG
from src.rag.retrieval import IntermediateRetriever, CorrectiveRetriever


@dataclass
class QueryResult:
    """Result for a single query."""
    query: str
    ticker: str
    retrieved_count: int
    latency_ms: float
    # For CRAG
    initial_relevance: float = 0.0
    final_relevance: float = 0.0
    corrections_applied: int = 0
    grades: List[str] = None


def run_baseline(index, queries: List[dict]) -> Dict:
    """Run baseline retriever on all queries."""
    retriever = IntermediateRetriever(index)
    results = []
    total_latency = 0

    print("\n" + "=" * 60)
    print("BASELINE (IntermediateRetriever)")
    print("=" * 60)

    for q in queries:
        start = time.perf_counter()
        nodes = retriever.retrieve(
            query=q["query"],
            top_k=5,
            ticker=q["ticker"],
        )
        latency = (time.perf_counter() - start) * 1000
        total_latency += latency

        results.append(QueryResult(
            query=q["query"][:50] + "...",
            ticker=q["ticker"],
            retrieved_count=len(nodes),
            latency_ms=latency,
        ))

        print(f"  [{q['ticker']}] {q['query'][:40]}... → {len(nodes)} docs in {latency:.0f}ms")

    avg_latency = total_latency / len(queries)
    print(f"\nAverage latency: {avg_latency:.0f}ms")

    return {
        "retriever": "IntermediateRetriever",
        "queries": len(queries),
        "avg_latency_ms": avg_latency,
        "total_latency_ms": total_latency,
        "results": results,
    }


def run_crag(index, queries: List[dict]) -> Dict:
    """Run CRAG retriever on all queries."""
    # Use new defaults: 0.4 threshold, hybrid mode enabled
    retriever = CorrectiveRetriever(
        index,
        relevance_threshold=0.4,  # Lowered from 0.5
        max_corrections=2,
        skip_grading_threshold=0.72,  # Hybrid mode: skip grading if top score >= 0.72 & avg >= 0.70
    )
    results = []
    total_latency = 0
    total_corrections = 0
    improvements = 0
    hybrid_skips = 0

    print("\n" + "=" * 60)
    print("CRAG (CorrectiveRetriever) - threshold=0.4, hybrid mode enabled")
    print("=" * 60)

    for q in queries:
        start = time.perf_counter()
        result = retriever.retrieve(
            query=q["query"],
            top_k=5,
            ticker=q["ticker"],
        )
        latency = (time.perf_counter() - start) * 1000
        total_latency += latency

        grades = [g.grade for g in result.grades] if result.grades else []
        relevant_count = grades.count("relevant")
        partial_count = grades.count("partial")

        # Check if hybrid skip was used
        was_hybrid = "hybrid_skip" in result.corrections_applied
        if was_hybrid:
            hybrid_skips += 1

        results.append(QueryResult(
            query=q["query"][:50] + "...",
            ticker=q["ticker"],
            retrieved_count=len(result.nodes),
            latency_ms=latency,
            initial_relevance=result.initial_relevant_ratio,
            final_relevance=result.final_relevant_ratio,
            corrections_applied=result.num_correction_attempts,
            grades=grades,
        ))

        total_corrections += result.num_correction_attempts
        if result.improved:
            improvements += 1

        status = "✓" if result.final_relevant_ratio >= 0.4 else "✗"
        if was_hybrid:
            correction_info = " [hybrid skip]"
        elif result.num_correction_attempts > 0:
            correction_info = f" (corrected {result.num_correction_attempts}x)"
        else:
            correction_info = ""

        print(f"  {status} [{q['ticker']}] {q['query'][:40]}...")
        print(f"      → {len(result.nodes)} docs, {relevant_count}R/{partial_count}P, "
              f"{result.final_relevant_ratio:.0%} relevant, {latency:.0f}ms{correction_info}")

    avg_latency = total_latency / len(queries)
    correction_rate = sum(1 for r in results if r.corrections_applied > 0) / len(queries)
    hybrid_skip_rate = hybrid_skips / len(queries)

    print(f"\nAverage latency: {avg_latency:.0f}ms")
    print(f"Hybrid skip rate: {hybrid_skip_rate:.0%} ({hybrid_skips}/{len(queries)} queries)")
    print(f"Correction rate: {correction_rate:.0%}")
    print(f"Improvement rate: {improvements/len(queries):.0%}")

    return {
        "retriever": "CorrectiveRetriever",
        "queries": len(queries),
        "avg_latency_ms": avg_latency,
        "total_latency_ms": total_latency,
        "correction_rate": correction_rate,
        "improvement_rate": improvements / len(queries),
        "hybrid_skip_rate": hybrid_skip_rate,
        "total_corrections": total_corrections,
        "results": results,
    }


def print_comparison(baseline: Dict, crag: Dict):
    """Print side-by-side comparison."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    # Calculate averages for CRAG
    crag_results = crag["results"]
    avg_initial = sum(r.initial_relevance for r in crag_results) / len(crag_results)
    avg_final = sum(r.final_relevance for r in crag_results) / len(crag_results)

    print(f"""
┌─────────────────────────────┬─────────────────┬─────────────────┬─────────────┐
│ Metric                      │ Baseline        │ CRAG            │ Delta       │
├─────────────────────────────┼─────────────────┼─────────────────┼─────────────┤
│ Average Latency             │ {baseline['avg_latency_ms']:>10.0f} ms  │ {crag['avg_latency_ms']:>10.0f} ms  │ {crag['avg_latency_ms'] - baseline['avg_latency_ms']:>+8.0f} ms │
│ Latency Increase            │        -        │        -        │ {((crag['avg_latency_ms'] / baseline['avg_latency_ms']) - 1) * 100:>+8.0f} %  │
├─────────────────────────────┼─────────────────┼─────────────────┼─────────────┤
│ Initial Relevance Ratio     │        -        │ {avg_initial:>13.0%}  │      -      │
│ Final Relevance Ratio       │        -        │ {avg_final:>13.0%}  │      -      │
│ Relevance Improvement       │        -        │        -        │ {(avg_final - avg_initial):>+8.0%}   │
├─────────────────────────────┼─────────────────┼─────────────────┼─────────────┤
│ Correction Rate             │        0 %      │ {crag['correction_rate']:>13.0%}  │      -      │
│ Queries Improved            │        -        │ {crag['improvement_rate']:>13.0%}  │      -      │
│ Total Corrections           │        0        │ {crag['total_corrections']:>13}  │      -      │
└─────────────────────────────┴─────────────────┴─────────────────┴─────────────┘
""")

    # Per-query comparison
    print("\nPer-Query Relevance (CRAG):")
    print("-" * 60)
    for r in crag_results:
        bar_len = int(r.final_relevance * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        correction_mark = f" [+{r.corrections_applied}]" if r.corrections_applied > 0 else ""
        print(f"  {r.ticker}: {bar} {r.final_relevance:>5.0%}{correction_mark}")


def main():
    print("Loading RAG index...")
    configure_llama_index()
    rag = SupabaseRAG()
    rag.load_index()
    print("Index loaded.\n")

    # Load golden set or use sample queries
    golden_path = Path(__file__).parent / "datasets" / "golden_set.json"
    with open(golden_path) as f:
        golden_set = json.load(f)

    queries = golden_set["queries"]
    print(f"Running comparison on {len(queries)} queries...\n")

    # Run both
    baseline_results = run_baseline(rag.index, queries)
    crag_results = run_crag(rag.index, queries)

    # Print comparison
    print_comparison(baseline_results, crag_results)

    # Estimate costs
    print("\n" + "=" * 70)
    print("COST ESTIMATE (per query)")
    print("=" * 70)
    print(f"""
  Baseline:
    - Embedding lookup: ~$0.00002
    - Total: ~$0.00002/query

  CRAG:
    - Embedding lookup: ~$0.00002
    - Grading (5 docs × Haiku): ~$0.00125
    - Corrections (avg {crag_results['total_corrections']/len(queries):.1f}): ~$0.0005
    - Total: ~$0.002/query

  Cost increase: ~100x per query (but still <$0.01)
""")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Convert results for JSON
    baseline_json = {
        **baseline_results,
        "results": [
            {"query": r.query, "ticker": r.ticker, "count": r.retrieved_count, "latency_ms": r.latency_ms}
            for r in baseline_results["results"]
        ]
    }
    crag_json = {
        **crag_results,
        "results": [
            {
                "query": r.query,
                "ticker": r.ticker,
                "count": r.retrieved_count,
                "latency_ms": r.latency_ms,
                "initial_relevance": r.initial_relevance,
                "final_relevance": r.final_relevance,
                "corrections": r.corrections_applied,
                "grades": r.grades,
            }
            for r in crag_results["results"]
        ]
    }

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_dir / f"comparison_{timestamp}.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "baseline": baseline_json,
            "crag": crag_json,
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}/comparison_{timestamp}.json")


if __name__ == "__main__":
    main()
