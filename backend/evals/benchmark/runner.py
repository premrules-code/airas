"""Benchmark runner — CLI for model/prompt comparison.

Runs all scenarios against different models, computes P(success) with error bars,
and outputs a comparison table.

Usage:
    # Compare two models
    python -m evals.benchmark.runner --compare claude-sonnet-4-20250514 claude-opus-4-20250514

    # Structural replay (no API calls, tests routing + structure)
    python -m evals.benchmark.runner --structural

    # Run with N repeats for probability estimation
    python -m evals.benchmark.runner --repeats 5

    # Specific scenario
    python -m evals.benchmark.runner --scenario sc_aapl_full_abc12345
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from evals.benchmark.recorder import ScenarioRecorder
from evals.benchmark.replay import ScenarioReplayer
from evals.benchmark.scorer import BenchmarkScorer, ScenarioScore

logger = logging.getLogger(__name__)

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_all_scenarios(scenarios_dir: Optional[Path] = None) -> list[dict]:
    """Load all scenario JSON files from the scenarios directory."""
    d = scenarios_dir or SCENARIOS_DIR
    scenarios = []
    for f in sorted(d.glob("*.json")):
        try:
            scenario = ScenarioRecorder.load(str(f))
            scenarios.append(scenario)
        except Exception as e:
            logger.warning(f"Failed to load scenario {f}: {e}")
    return scenarios


def run_structural_benchmark(
    scenarios: list[dict],
    repeats: int = 1,
) -> dict:
    """Run structural benchmark: mock everything, test routing + patterns.

    Fast (~100ms per scenario), no API cost.
    """
    scorer = BenchmarkScorer()
    all_scenario_scores = []

    for scenario in scenarios:
        replayer = ScenarioReplayer(scenario)

        if repeats > 1:
            # Probabilistic mode: run N times, estimate P(success)
            results = [replayer.replay_structural().to_dict() for _ in range(repeats)]
            repeated = scorer.score_repeated_runs(scenario, results)
            print(
                f"  {scenario['scenario_id']}: "
                f"P(success) = {repeated['mean']:.2f} ±{repeated['error_bar']:.2f} "
                f"(n={repeated['n']})"
            )
            # Use mean probability for overall scoring
            scenario_score = ScenarioScore(
                scenario_id=scenario["scenario_id"],
                stages=scorer.score_scenario(scenario, results[0]).stages,
                num_turns=len(scenario.get("agent_executions", [])) + 2,
            )
        else:
            result = replayer.replay_structural()
            scenario_score = scorer.score_scenario(scenario, result.to_dict())
            print(
                f"  {scenario['scenario_id']}: "
                f"P(success) = {scenario_score.probability:.2f}"
            )

        all_scenario_scores.append(scenario_score)

    benchmark = scorer.score_benchmark(all_scenario_scores)
    return benchmark.to_dict()


def run_live_benchmark(
    scenarios: list[dict],
    model: str,
    repeats: int = 1,
) -> dict:
    """Run live benchmark: real Claude calls with mocked tools.

    Costs ~$0.50-2.00 per scenario per run.
    """
    scorer = BenchmarkScorer()
    all_scenario_scores = []

    for scenario in scenarios:
        replayer = ScenarioReplayer(scenario)

        if repeats > 1:
            results = [
                replayer.replay_live(model=model).to_dict()
                for _ in range(repeats)
            ]
            repeated = scorer.score_repeated_runs(scenario, results)
            print(
                f"  {scenario['scenario_id']}: "
                f"P(success) = {repeated['mean']:.2f} ±{repeated['error_bar']:.2f}"
            )
            scenario_score = ScenarioScore(
                scenario_id=scenario["scenario_id"],
                stages=scorer.score_scenario(scenario, results[0]).stages,
                num_turns=len(scenario.get("agent_executions", [])) + 2,
            )
        else:
            result = replayer.replay_live(model=model)
            scenario_score = scorer.score_scenario(scenario, result.to_dict())
            print(
                f"  {scenario['scenario_id']}: "
                f"P(success) = {scenario_score.probability:.2f}"
            )

        all_scenario_scores.append(scenario_score)

    benchmark = scorer.score_benchmark(all_scenario_scores)
    return benchmark.to_dict()


def run_comparison(
    scenarios: list[dict],
    models: list[str],
    repeats: int = 3,
) -> dict:
    """Compare multiple models on the same scenarios.

    Outputs a comparison table with error bars.
    """
    results = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        model_result = run_live_benchmark(scenarios, model=model, repeats=repeats)
        results[model] = model_result

    # Print comparison table
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")

    # Header
    header = f"{'Scenario':<30}"
    for model in models:
        short = model.split("-")[1] if "-" in model else model
        header += f" {'|':>2} {short:<14}"
    header += f" {'|':>2} {'Delta':<10}"
    print(header)
    print("-" * 70)

    # Per-scenario rows
    for i, scenario in enumerate(scenarios):
        sid = scenario["scenario_id"][:28]
        row = f"{sid:<30}"

        scores = []
        for model in models:
            model_scenarios = results[model].get("scenarios", [])
            if i < len(model_scenarios):
                p = model_scenarios[i]["probability"]
                scores.append(p)
                row += f" {'|':>2} {p:>6.2f}       "
            else:
                scores.append(0.0)
                row += f" {'|':>2} {'N/A':>6}       "

        if len(scores) >= 2:
            delta = scores[-1] - scores[0]
            sign = "+" if delta >= 0 else ""
            row += f" {'|':>2} {sign}{delta:.2f}"

        print(row)

    # Overall row
    print("-" * 70)
    row = f"{'OVERALL':<30}"
    overall_scores = []
    for model in models:
        p = results[model].get("overall_probability", 0.0)
        u = results[model].get("uncertainty", 0.0)
        overall_scores.append(p)
        short = f"{p:.2f}±{u:.2f}"
        row += f" {'|':>2} {short:<14}"

    if len(overall_scores) >= 2:
        delta = overall_scores[-1] - overall_scores[0]
        sign = "+" if delta >= 0 else ""
        row += f" {'|':>2} {sign}{delta:.2f}"

    print(row)
    print(f"{'='*70}")

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models": models,
        "repeats": repeats,
        "num_scenarios": len(scenarios),
        "results": results,
    }


def save_results(results: dict, name: str) -> Path:
    """Save benchmark results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RESULTS_DIR / f"benchmark_{name}_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="AIRAS Pipeline Benchmark — scenario-based evaluation"
    )
    parser.add_argument(
        "--structural", action="store_true",
        help="Run structural replay (no API calls, tests routing + patterns)",
    )
    parser.add_argument(
        "--compare", nargs="+", metavar="MODEL",
        help="Compare models (e.g., --compare claude-sonnet-4-20250514 claude-opus-4-20250514)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model for live benchmark (default: uses config)",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Number of repeats per scenario for probability estimation (default: 3)",
    )
    parser.add_argument(
        "--scenario", default=None,
        help="Run a specific scenario by ID",
    )
    parser.add_argument(
        "--scenarios-dir", default=None,
        help="Path to scenarios directory",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    # Load scenarios
    scenarios_dir = Path(args.scenarios_dir) if args.scenarios_dir else SCENARIOS_DIR
    scenarios = load_all_scenarios(scenarios_dir)

    if not scenarios:
        print(f"No scenarios found in {scenarios_dir}")
        print("Record scenarios first using ScenarioRecorder, or create sample scenarios.")
        sys.exit(1)

    if args.scenario:
        scenarios = [s for s in scenarios if s["scenario_id"] == args.scenario]
        if not scenarios:
            print(f"Scenario '{args.scenario}' not found")
            sys.exit(1)

    print(f"Loaded {len(scenarios)} scenarios from {scenarios_dir}")

    # Run benchmark
    if args.compare:
        results = run_comparison(scenarios, args.compare, repeats=args.repeats)
        save_results(results, "comparison")
    elif args.structural:
        print(f"\nStructural Benchmark (repeats={args.repeats})")
        print("-" * 40)
        results = run_structural_benchmark(scenarios, repeats=args.repeats)
        print(f"\nOverall: P(success) = {results['overall_probability']:.2f} "
              f"±{results['uncertainty']:.2f}")
        save_results(results, "structural")
    else:
        model = args.model or "default"
        print(f"\nLive Benchmark: model={model}, repeats={args.repeats}")
        print("-" * 40)
        results = run_live_benchmark(scenarios, model=model, repeats=args.repeats)
        print(f"\nOverall: P(success) = {results['overall_probability']:.2f} "
              f"±{results['uncertainty']:.2f}")
        save_results(results, f"live_{model}")


if __name__ == "__main__":
    main()
