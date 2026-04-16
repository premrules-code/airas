#!/usr/bin/env python3
"""Generate and populate golden dataset for RAG evaluation.

This script helps create evaluation queries and find ground truth chunks.

Usage:
    # Generate candidate queries for a ticker
    python generate_golden_set.py --ticker AAPL --generate-queries 20

    # Find relevant chunks for a query (helps identify ground truth)
    python generate_golden_set.py --ticker AAPL --find-chunks "What was Apple's revenue?"

    # Validate golden set (check all queries have ground truth chunks)
    python generate_golden_set.py --validate
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import anthropic
from config.settings import get_settings
from src.rag.supabase_rag import SupabaseRAG
from src.rag.retrieval import IntermediateRetriever
from src.utils.llama_setup import configure_llama_index


def generate_candidate_queries(ticker: str, num_queries: int = 20) -> list[dict]:
    """Use Claude to generate diverse financial queries for evaluation."""
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    prompt = f"""Generate {num_queries} diverse financial analysis questions about {ticker} that could be answered from SEC 10-K filings.

Cover these categories:
1. Financial metrics (revenue, profit, margins, EPS) - 6 questions
2. Balance sheet items (cash, debt, assets, equity) - 4 questions
3. Risk factors - 3 questions
4. Business segments/geographic breakdown - 3 questions
5. Capital allocation (dividends, buybacks) - 2 questions
6. Multi-year trends or comparisons - 2 questions

For each question, provide:
- The question text
- Expected answer format (number, list, explanation)
- Difficulty (easy = single fact lookup, medium = requires finding multiple facts, hard = requires calculation or inference)
- Which SEC filing section likely contains the answer

Return as a JSON array with objects containing: question, expected_format, difficulty, likely_section"""

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    # Extract JSON from response
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        queries = json.loads(text)
        return queries
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}")
        print(f"Raw response:\n{text}")
        return []


def find_relevant_chunks(ticker: str, query: str, top_k: int = 10) -> list[dict]:
    """Retrieve chunks that might be relevant to a query.

    Returns chunks with their IDs so humans can identify ground truth.
    """
    configure_llama_index()

    rag = SupabaseRAG()
    rag.load_index()

    retriever = IntermediateRetriever(rag.index)
    nodes = retriever.retrieve(query, top_k=top_k, ticker=ticker)

    results = []
    for i, node in enumerate(nodes):
        results.append({
            "rank": i + 1,
            "node_id": node.node.node_id,
            "score": node.score,
            "text_preview": node.text[:500] + "..." if len(node.text) > 500 else node.text,
            "metadata": node.node.metadata,
        })

    return results


def validate_golden_set(golden_set_path: str) -> dict:
    """Validate that golden set has all required fields."""
    with open(golden_set_path) as f:
        data = json.load(f)

    issues = []
    stats = {
        "total_queries": len(data["queries"]),
        "with_ground_truth_chunks": 0,
        "without_ground_truth_chunks": 0,
        "with_answer": 0,
        "categories": {},
    }

    for q in data["queries"]:
        # Check required fields
        required = ["id", "ticker", "query", "ground_truth_answer", "ground_truth_chunks"]
        for field in required:
            if field not in q:
                issues.append(f"Query {q.get('id', 'UNKNOWN')} missing field: {field}")

        # Check ground truth chunks
        if q.get("ground_truth_chunks"):
            stats["with_ground_truth_chunks"] += 1
        else:
            stats["without_ground_truth_chunks"] += 1
            issues.append(f"Query {q['id']} has no ground truth chunks")

        if q.get("ground_truth_answer"):
            stats["with_answer"] += 1

        # Count categories
        cat = q.get("category", "uncategorized")
        stats["categories"][cat] = stats["categories"].get(cat, 0) + 1

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Golden dataset management")
    parser.add_argument("--ticker", type=str, help="Stock ticker")
    parser.add_argument("--generate-queries", type=int, help="Generate N candidate queries")
    parser.add_argument("--find-chunks", type=str, help="Find relevant chunks for a query")
    parser.add_argument("--validate", action="store_true", help="Validate golden set")
    parser.add_argument("--golden-set", type=str,
                       default="evals/datasets/golden_set.json",
                       help="Path to golden set file")

    args = parser.parse_args()

    if args.generate_queries:
        if not args.ticker:
            print("Error: --ticker required with --generate-queries")
            sys.exit(1)

        print(f"Generating {args.generate_queries} candidate queries for {args.ticker}...")
        queries = generate_candidate_queries(args.ticker, args.generate_queries)

        print(json.dumps(queries, indent=2))
        print(f"\n✅ Generated {len(queries)} candidate queries")
        print("Review these and add to golden_set.json with ground truth answers")

    elif args.find_chunks:
        if not args.ticker:
            print("Error: --ticker required with --find-chunks")
            sys.exit(1)

        print(f"Finding relevant chunks for: {args.find_chunks}")
        chunks = find_relevant_chunks(args.ticker, args.find_chunks)

        print("\n" + "=" * 80)
        for chunk in chunks:
            print(f"\n[Rank {chunk['rank']}] Score: {chunk['score']:.4f}")
            print(f"Node ID: {chunk['node_id']}")
            print(f"Metadata: {chunk['metadata']}")
            print(f"Text: {chunk['text_preview']}")
            print("-" * 80)

        print(f"\n✅ Found {len(chunks)} chunks")
        print("Mark relevant chunk IDs as ground_truth_chunks in golden_set.json")

    elif args.validate:
        print(f"Validating golden set: {args.golden_set}")
        result = validate_golden_set(args.golden_set)

        print(f"\nStats:")
        print(f"  Total queries: {result['stats']['total_queries']}")
        print(f"  With ground truth chunks: {result['stats']['with_ground_truth_chunks']}")
        print(f"  Without ground truth chunks: {result['stats']['without_ground_truth_chunks']}")
        print(f"  Categories: {result['stats']['categories']}")

        if result["valid"]:
            print("\n✅ Golden set is valid!")
        else:
            print(f"\n❌ Found {len(result['issues'])} issues:")
            for issue in result["issues"]:
                print(f"  - {issue}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
