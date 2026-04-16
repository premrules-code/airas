#!/usr/bin/env python3
"""Generate comparison report between baseline and CRAG evaluation results.

Usage:
    python evals/run_comparison.py \
        --baseline evals/results/baseline_aggregate_*.json \
        --crag evals/results/crag_aggregate_*.json \
        --output evals/results/comparison_report.md
"""

import argparse
import json
import glob
from datetime import datetime
from pathlib import Path


def load_latest_result(pattern: str) -> dict:
    """Load the most recent result file matching pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {pattern}")

    latest = files[-1]
    print(f"Loading: {latest}")

    with open(latest) as f:
        return json.load(f)


def compute_delta(baseline: float, crag: float) -> dict:
    """Compute absolute and percentage change."""
    delta = crag - baseline
    pct = (delta / baseline * 100) if baseline != 0 else 0
    return {
        "baseline": baseline,
        "crag": crag,
        "delta": delta,
        "pct_change": pct,
    }


def generate_report(baseline: dict, crag: dict) -> str:
    """Generate markdown comparison report."""

    # Extract results (handle both aggregate and detailed formats)
    baseline_ret = baseline.get("results", baseline).get("retrieval", {})
    crag_ret = crag.get("results", crag).get("retrieval", {})
    crag_crag = crag.get("results", crag).get("crag", {})

    baseline_lat = baseline.get("results", baseline).get("latency", {})
    crag_lat = crag.get("results", crag).get("latency", {})

    # Build comparison data
    comparisons = {}

    # Retrieval metrics
    retrieval_metrics = ["precision_at_k", "recall_at_k", "f1_at_k", "mrr", "ndcg_at_k", "hit_rate"]
    for metric in retrieval_metrics:
        b_val = baseline_ret.get(metric, {}).get("mean", 0)
        c_val = crag_ret.get(metric, {}).get("mean", 0)
        comparisons[metric] = compute_delta(b_val, c_val)

    # Latency
    b_lat = baseline_lat.get("latency", baseline_lat).get("p50_ms", baseline_lat.get("mean_ms", 0))
    c_lat = crag_lat.get("latency", crag_lat).get("p50_ms", crag_lat.get("mean_ms", 0))
    comparisons["latency_p50"] = compute_delta(b_lat, c_lat)

    b_lat95 = baseline_lat.get("latency", baseline_lat).get("p95_ms", 0)
    c_lat95 = crag_lat.get("latency", crag_lat).get("p95_ms", 0)
    comparisons["latency_p95"] = compute_delta(b_lat95, c_lat95)

    # Cost (if available)
    b_cost = baseline_lat.get("cost", {}).get("total", 0)
    c_cost = crag_lat.get("cost", {}).get("total", 0)
    comparisons["cost"] = compute_delta(b_cost, c_cost)

    # Generate markdown
    report = f"""# RAG Evaluation Comparison Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

| Metric | Baseline | CRAG | Delta | Change |
|--------|----------|------|-------|--------|
| Precision@5 | {comparisons['precision_at_k']['baseline']:.3f} | {comparisons['precision_at_k']['crag']:.3f} | {comparisons['precision_at_k']['delta']:+.3f} | {comparisons['precision_at_k']['pct_change']:+.1f}% |
| Recall@5 | {comparisons['recall_at_k']['baseline']:.3f} | {comparisons['recall_at_k']['crag']:.3f} | {comparisons['recall_at_k']['delta']:+.3f} | {comparisons['recall_at_k']['pct_change']:+.1f}% |
| F1@5 | {comparisons['f1_at_k']['baseline']:.3f} | {comparisons['f1_at_k']['crag']:.3f} | {comparisons['f1_at_k']['delta']:+.3f} | {comparisons['f1_at_k']['pct_change']:+.1f}% |
| MRR | {comparisons['mrr']['baseline']:.3f} | {comparisons['mrr']['crag']:.3f} | {comparisons['mrr']['delta']:+.3f} | {comparisons['mrr']['pct_change']:+.1f}% |
| NDCG@5 | {comparisons['ndcg_at_k']['baseline']:.3f} | {comparisons['ndcg_at_k']['crag']:.3f} | {comparisons['ndcg_at_k']['delta']:+.3f} | {comparisons['ndcg_at_k']['pct_change']:+.1f}% |
| Hit Rate | {comparisons['hit_rate']['baseline']:.1%} | {comparisons['hit_rate']['crag']:.1%} | {comparisons['hit_rate']['delta']:+.1%} | {comparisons['hit_rate']['pct_change']:+.1f}% |
| Latency (p50) | {comparisons['latency_p50']['baseline']:.0f}ms | {comparisons['latency_p50']['crag']:.0f}ms | {comparisons['latency_p50']['delta']:+.0f}ms | {comparisons['latency_p50']['pct_change']:+.1f}% |
| Latency (p95) | {comparisons['latency_p95']['baseline']:.0f}ms | {comparisons['latency_p95']['crag']:.0f}ms | {comparisons['latency_p95']['delta']:+.0f}ms | {comparisons['latency_p95']['pct_change']:+.1f}% |

## CRAG-Specific Metrics

"""

    if crag_crag:
        report += f"""| Metric | Value |
|--------|-------|
| Initial Relevance Ratio | {crag_crag.get('initial_relevant_ratio', {}).get('mean', 0):.1%} |
| Final Relevance Ratio | {crag_crag.get('final_relevant_ratio', {}).get('mean', 0):.1%} |
| Relevance Improvement | {crag_crag.get('improvement', {}).get('mean', 0):+.1%} |
| Correction Rate | {crag_crag.get('correction_rate', 0):.1%} |
| Improvement Rate | {crag_crag.get('improvement_rate', 0):.1%} |
| Avg Corrections/Query | {crag_crag.get('avg_corrections', 0):.2f} |

"""

    # Verdict
    recall_improved = comparisons['recall_at_k']['delta'] > 0.05
    precision_improved = comparisons['precision_at_k']['delta'] > 0
    latency_acceptable = comparisons['latency_p95']['delta'] < 1000  # <1s increase

    report += """## Verdict

"""

    verdicts = []
    if recall_improved:
        verdicts.append(f"✅ Recall improved by {comparisons['recall_at_k']['pct_change']:+.1f}%")
    else:
        verdicts.append(f"❌ Recall did not improve significantly ({comparisons['recall_at_k']['pct_change']:+.1f}%)")

    if precision_improved:
        verdicts.append(f"✅ Precision improved by {comparisons['precision_at_k']['pct_change']:+.1f}%")
    else:
        verdicts.append(f"⚠️ Precision decreased by {comparisons['precision_at_k']['pct_change']:+.1f}%")

    if latency_acceptable:
        verdicts.append(f"✅ Latency increase acceptable ({comparisons['latency_p95']['delta']:+.0f}ms)")
    else:
        verdicts.append(f"⚠️ Latency increase significant ({comparisons['latency_p95']['delta']:+.0f}ms)")

    for v in verdicts:
        report += f"- {v}\n"

    # Overall recommendation
    if recall_improved and latency_acceptable:
        report += "\n**Recommendation: ✅ Deploy CRAG** - Quality improvement justifies latency cost.\n"
    elif recall_improved:
        report += "\n**Recommendation: ⚠️ Consider CRAG with tuning** - Quality improved but latency may be an issue.\n"
    else:
        report += "\n**Recommendation: ❌ Do not deploy CRAG** - Insufficient quality improvement.\n"

    return report


def main():
    parser = argparse.ArgumentParser(description="Generate RAG evaluation comparison report")
    parser.add_argument(
        "--baseline",
        default="evals/results/baseline_aggregate_*.json",
        help="Glob pattern for baseline results"
    )
    parser.add_argument(
        "--crag",
        default="evals/results/crag_aggregate_*.json",
        help="Glob pattern for CRAG results"
    )
    parser.add_argument(
        "--output",
        default="evals/results/comparison_report.md",
        help="Output path for markdown report"
    )

    args = parser.parse_args()

    try:
        baseline = load_latest_result(args.baseline)
        crag = load_latest_result(args.crag)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nRun evaluation tests first:")
        print("  pytest evals/test_baseline.py -v")
        print("  pytest evals/test_crag.py -v")
        return 1

    report = generate_report(baseline, crag)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")
    print("\n" + "=" * 60)
    print(report)

    return 0


if __name__ == "__main__":
    exit(main())
