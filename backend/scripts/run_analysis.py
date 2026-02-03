#!/usr/bin/env python3
"""Run AIRAS analysis for a stock ticker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()  # Export .env vars to os.environ so Anthropic SDK, etc. can read them

import argparse
import json
import logging
import time

from config.settings import get_settings
from config.logging_config import setup_logging
from src.utils.llama_setup import configure_llama_index
from src.utils.langfuse_setup import setup_langfuse
from src.agents.graph import build_analysis_graph
from src.agents.state import AnalysisState


def main():
    parser = argparse.ArgumentParser(description="AIRAS Investment Analysis")
    parser.add_argument("--ticker", required=True, help="Stock ticker (e.g., AAPL)")
    parser.add_argument(
        "--query", type=str, help="Specific question (routes to relevant agents)"
    )
    parser.add_argument(
        "--full", action="store_true", help="Force all 10 agents even with --query"
    )
    parser.add_argument(
        "--rag-level",
        choices=["basic", "intermediate", "advanced"],
        default="intermediate",
        help="RAG retrieval level",
    )
    parser.add_argument(
        "--sequential", action="store_true", help="Run agents sequentially (debug)"
    )
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    # Setup
    settings = get_settings()
    setup_logging(settings.log_level)
    configure_llama_index()
    setup_langfuse()

    logger = logging.getLogger(__name__)

    print(f"\n{'='*70}")
    print(f"AIRAS V3 — Investment Analysis for {args.ticker.upper()}")
    print(f"{'='*70}")
    print(f"RAG Level: {args.rag_level}")
    if args.query:
        print(f"Query: {args.query}")
    print()

    # Build and run graph
    start_time = time.time()
    graph = build_analysis_graph()

    initial_state: AnalysisState = {
        "ticker": args.ticker.upper(),
        "query": args.query if not args.full else None,
        "active_agents": [],
        "mode": "full",
        "rag_level": args.rag_level,
        "rag_context": {},
        "agent_outputs": [],
        "recommendation": None,
        "trace_id": None,
        "errors": [],
    }

    logger.info(f"Starting analysis for {args.ticker.upper()}...")
    result = graph.invoke(initial_state)
    elapsed = time.time() - start_time

    # Display results
    rec = result.get("recommendation")
    if rec:
        print(f"\n{'='*70}")
        print(f"  RECOMMENDATION: {rec.recommendation}")
        print(f"{'='*70}")
        print(f"  Overall Score:  {rec.overall_score:+.3f}")
        print(f"  Confidence:     {rec.confidence:.1%}")
        print(f"  Agents Run:     {rec.num_agents}")
        print(f"  Time:           {elapsed:.1f}s")
        print(f"\n  Category Scores:")
        print(f"    Financial:    {rec.financial_score:+.3f}")
        print(f"    Technical:    {rec.technical_score:+.3f}")
        print(f"    Sentiment:    {rec.sentiment_score:+.3f}")
        print(f"    Risk:         {rec.risk_score:+.3f}")
        print(f"\n  Agent Scores:")
        for name, score in rec.agent_scores.items():
            print(f"    {name:25s} {score:+.3f}")
        print(f"\n  Thesis:")
        print(f"    {rec.thesis}")
        if rec.bullish_factors:
            print(f"\n  Bullish Factors:")
            for f in rec.bullish_factors:
                print(f"    + {f}")
        if rec.bearish_factors:
            print(f"\n  Bearish Factors:")
            for f in rec.bearish_factors:
                print(f"    - {f}")
        if rec.risks:
            print(f"\n  Risks:")
            for r in rec.risks:
                print(f"    ! {r}")
        print(f"{'='*70}\n")

    if result.get("errors"):
        print(f"  Errors:")
        for err in result["errors"]:
            print(f"    * {err}")
        print()

    # Save to file
    if args.output and rec:
        output_data = rec.model_dump()
        output_data["analysis_time_seconds"] = round(elapsed, 2)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
