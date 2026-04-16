#!/usr/bin/env python3
"""Run Galileo evaluations standalone.

Usage:
    python evals/run_galileo_evals.py
    python evals/run_galileo_evals.py --output results/my_eval.json
    python evals/run_galileo_evals.py --verbose
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.utils.galileo_setup import init_galileo, is_galileo_available
from evals.metrics.galileo_metrics import GalileoEvaluator


def print_banner(text: str):
    """Print formatted banner."""
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")


def print_result_table(headers: list[str], rows: list[list]):
    """Print formatted table."""
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

    # Header
    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_row)
    print("-" * len(header_row))

    # Rows
    for row in rows:
        print(" | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)))


def run_evals(eval_path: str, verbose: bool = False) -> dict:
    """Run Galileo evaluations."""

    # Initialize Galileo
    print_banner("AIRAS Galileo Evaluation")

    if not init_galileo():
        print("WARNING: Galileo not available. Using fallback scoring.")
        print("Set GALILEO_API_KEY in .env for full evaluation.")
    else:
        print("Galileo initialized successfully.")

    # Load dataset
    print(f"\nLoading eval dataset: {eval_path}")
    with open(eval_path) as f:
        dataset = json.load(f)

    print(f"  - Queries: {len(dataset['queries'])}")
    print(f"  - Hallucination tests: {len(dataset.get('hallucination_tests', []))}")
    print(f"  - PII tests: {len(dataset.get('pii_tests', []))}")

    # Run evaluations
    evaluator = GalileoEvaluator()
    results = evaluator.evaluate_batch(dataset)

    # Print results
    print_banner("GROUNDEDNESS RESULTS")

    if verbose:
        headers = ["Query ID", "Score", "Expected", "Passed", "Flagged"]
        rows = [
            [
                r.query_id,
                f"{r.groundedness_score:.2f}",
                f"{r.expected_groundedness:.2f}",
                "✓" if r.passed else "✗",
                len(r.flagged_claims),
            ]
            for r in results["results"]["groundedness"]
        ]
        print_result_table(headers, rows)
        print()

    summary = results["summary"]
    print(f"Mean groundedness: {summary['groundedness']['mean']:.3f}")
    print(f"Std deviation: {summary['groundedness']['std']:.3f}")
    print(f"Pass rate: {summary['groundedness']['pass_rate']:.1%}")

    print_banner("HALLUCINATION DETECTION")

    hal_results = results["results"]["hallucination_detection"]
    if hal_results:
        detected = sum(1 for r in hal_results if r["detection_success"])
        print(f"Tests run: {len(hal_results)}")
        print(f"Detected: {detected}")
        print(f"Detection rate: {summary['hallucination_detection']['rate']:.1%}")

        if verbose:
            print("\nDetails:")
            for r in hal_results:
                status = "✓" if r["detection_success"] else "✗"
                print(f"  {status} {r['test_id']}: bad={r['bad_answer_score']:.2f}, good={r['good_answer_score']:.2f}")

    print_banner("PII DETECTION")

    pii_results = results["results"]["pii"]
    if pii_results:
        passed = sum(1 for r in pii_results if r.passed)
        print(f"Tests run: {len(pii_results)}")
        print(f"Passed: {passed}")
        print(f"Accuracy: {summary['pii_detection']['accuracy']:.1%}")

        if verbose:
            print("\nDetails:")
            for r in pii_results:
                status = "✓" if r.passed else "✗"
                pii_info = f" ({', '.join(r.pii_types)})" if r.pii_types else ""
                print(f"  {status} {r.text_id}: has_pii={r.has_pii}{pii_info}")

    print_banner("SUMMARY")

    print(f"Total queries evaluated: {summary['total_queries']}")
    print(f"Groundedness pass rate: {summary['groundedness']['pass_rate']:.1%}")
    print(f"Hallucination detection: {summary['hallucination_detection']['rate']:.1%}")
    print(f"PII detection accuracy: {summary['pii_detection']['accuracy']:.1%}")

    return results


def save_results(results: dict, output_path: str):
    """Save results to JSON file."""
    # Convert dataclass results to dicts
    serializable = {
        "timestamp": datetime.now().isoformat(),
        "summary": results["summary"],
        "groundedness": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "score": r.groundedness_score,
                "expected": r.expected_groundedness,
                "passed": r.passed,
                "flagged_claims": r.flagged_claims,
            }
            for r in results["results"]["groundedness"]
        ],
        "hallucination_detection": results["results"]["hallucination_detection"],
        "pii": [
            {
                "text_id": r.text_id,
                "has_pii": r.has_pii,
                "pii_types": r.pii_types,
                "expected": r.expected_pii,
                "passed": r.passed,
            }
            for r in results["results"]["pii"]
        ],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Galileo evaluations for AIRAS")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evals/datasets/galileo_eval_set.json",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save results (default: evals/results/galileo_eval_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed results",
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    eval_path = script_dir.parent / args.dataset

    if not eval_path.exists():
        print(f"Error: Dataset not found: {eval_path}")
        sys.exit(1)

    # Run evaluations
    results = run_evals(str(eval_path), verbose=args.verbose)

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = script_dir / "results" / f"galileo_eval_{timestamp}.json"

    save_results(results, str(output_path))

    # Exit with status based on pass rate
    pass_rate = results["summary"]["groundedness"]["pass_rate"]
    if pass_rate < 0.5:
        print(f"\n⚠️  Low pass rate: {pass_rate:.1%}")
        sys.exit(1)
    else:
        print(f"\n✓ Evaluation complete. Pass rate: {pass_rate:.1%}")
        sys.exit(0)


if __name__ == "__main__":
    main()
