#!/usr/bin/env python3
"""
Demo script showing exactly what Galileo does for AIRAS.

Run: python scripts/demo_galileo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.galileo_setup import init_galileo, is_galileo_available
from src.guardrails.galileo_guardrails import (
    check_hallucination,
    check_pii,
    check_context_relevance,
)


def print_header(text: str):
    print(f"\n{'='*70}")
    print(f" {text}")
    print(f"{'='*70}")


def print_result(label: str, value, good: bool = True):
    icon = "✓" if good else "✗"
    print(f"  {icon} {label}: {value}")


def demo_hallucination_detection():
    """Show how Galileo catches made-up financial data."""

    print_header("DEMO 1: Hallucination Detection")

    # The ground truth from SEC filing
    context = """
    Apple Inc. reported total net sales of $383,285 million ($383.3 billion)
    for the fiscal year ended September 30, 2023. This represents a decrease
    of 2.8% compared to fiscal year 2022 revenue of $394,328 million.
    Services revenue was $85.2 billion, representing 22% of total revenue.
    """

    print("\n📄 CONTEXT (from SEC filing):")
    print(f"   {context.strip()[:200]}...")

    # BAD: Hallucinated answer with wrong numbers
    bad_answer = """
    Apple reported revenue of $450 billion in FY2023, representing 25% growth
    year-over-year. This was driven by the launch of Apple Car and AR glasses,
    which generated $50 billion in their first year.
    """

    print("\n❌ BAD ANSWER (hallucinated):")
    print(f"   {bad_answer.strip()}")

    bad_result = check_hallucination(bad_answer, [context])

    print("\n   Galileo Analysis:")
    print_result("Groundedness Score", f"{bad_result['score']:.0%}", bad_result['score'] > 0.7)
    print_result("Is Grounded?", bad_result['is_grounded'], bad_result['is_grounded'])
    if bad_result['flagged_claims']:
        print_result("Flagged Claims", bad_result['flagged_claims'], False)

    # GOOD: Accurate answer from context
    good_answer = """
    Apple reported total net sales of $383.3 billion for fiscal year 2023,
    a decrease of 2.8% from the prior year. Services revenue reached $85.2 billion.
    """

    print("\n✓ GOOD ANSWER (grounded):")
    print(f"   {good_answer.strip()}")

    good_result = check_hallucination(good_answer, [context])

    print("\n   Galileo Analysis:")
    print_result("Groundedness Score", f"{good_result['score']:.0%}", good_result['score'] > 0.7)
    print_result("Is Grounded?", good_result['is_grounded'], good_result['is_grounded'])

    print("\n   📊 COMPARISON:")
    print(f"      Bad answer:  {bad_result['score']:.0%} groundedness")
    print(f"      Good answer: {good_result['score']:.0%} groundedness")
    print(f"      Difference:  {(good_result['score'] - bad_result['score']):.0%}")

    if bad_result['score'] < good_result['score']:
        print("\n   ✓ Galileo correctly identified the hallucinated answer!")
    else:
        print("\n   ⚠ Detection inconclusive (may need Galileo API)")


def demo_pii_detection():
    """Show how Galileo catches PII in outputs."""

    print_header("DEMO 2: PII Detection")

    # Text with PII that shouldn't be in financial output
    text_with_pii = """
    Based on insider trading data, John Smith (john.smith@apple.com) sold
    10,000 shares. Contact investor relations at 555-123-4567 for details.
    His SSN 123-45-6789 was used for verification.
    """

    print("\n❌ TEXT WITH PII:")
    print(f"   {text_with_pii.strip()}")

    pii_result = check_pii(text_with_pii)

    print("\n   Galileo Analysis:")
    print_result("PII Detected?", pii_result['has_pii'], not pii_result['has_pii'])
    if pii_result['pii_types']:
        print_result("PII Types Found", pii_result['pii_types'], False)
    print(f"\n   Redacted Version:")
    print(f"   {pii_result['redacted_text'][:200]}...")

    # Clean financial text
    clean_text = """
    Based on Form 4 filings, company executives sold shares as part of
    pre-planned 10b5-1 trading plans. Total insider selling was $2.5 million
    in the quarter, representing 0.01% of outstanding shares.
    """

    print("\n✓ CLEAN TEXT (no PII):")
    print(f"   {clean_text.strip()}")

    clean_result = check_pii(clean_text)

    print("\n   Galileo Analysis:")
    print_result("PII Detected?", clean_result['has_pii'], not clean_result['has_pii'])

    if pii_result['has_pii'] and not clean_result['has_pii']:
        print("\n   ✓ Galileo correctly identified PII and clean text!")


def demo_context_relevance():
    """Show how Galileo scores RAG chunk relevance."""

    print_header("DEMO 3: Context Relevance Scoring")

    query = "What was Apple's revenue in 2023?"

    print(f"\n🔍 QUERY: {query}")

    # Mix of relevant and irrelevant chunks
    chunks = [
        # Relevant
        "Apple Inc. reported total net sales of $383.3 billion for fiscal year 2023.",
        # Relevant
        "Revenue breakdown: iPhone $200.6B, Services $85.2B, Mac $29.4B, iPad $28.3B.",
        # Somewhat relevant
        "The company's gross margin was 44.1% in FY2023.",
        # Irrelevant
        "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
        # Irrelevant
        "The iPhone uses the A17 Pro chip manufactured using 3nm process technology.",
    ]

    print("\n📚 RETRIEVED CHUNKS:")
    for i, chunk in enumerate(chunks):
        print(f"   [{i+1}] {chunk[:70]}...")

    result = check_context_relevance(query, chunks)

    print("\n   Galileo Relevance Scores:")
    for i, (chunk, score) in enumerate(zip(chunks, result['scores'])):
        relevance = "HIGH" if score >= 0.7 else "MEDIUM" if score >= 0.4 else "LOW"
        icon = "✓" if score >= 0.5 else "✗"
        print(f"   {icon} Chunk {i+1}: {score:.0%} ({relevance})")

    print(f"\n   Average Relevance: {result['avg_score']:.0%}")
    print(f"   Relevant Chunks: {len(result['relevant_indices'])}/{len(chunks)}")

    if result['avg_score'] > 0:
        print("\n   ✓ Galileo scored chunk relevance!")


def demo_real_agent_scenario():
    """Show a realistic AIRAS agent scenario."""

    print_header("DEMO 4: Real AIRAS Agent Scenario")

    print("""
    Scenario: Financial Analyst agent analyzing AAPL

    1. RAG retrieves SEC filing chunks
    2. Agent generates analysis
    3. Galileo validates before returning to user
    """)

    # Simulated RAG context
    rag_context = """
    From AAPL 10-K FY2023:
    - Total net sales: $383,285 million
    - Net income: $96,995 million
    - iPhone revenue: $200,583 million (52% of total)
    - Services revenue: $85,200 million
    - R&D expenses: $29,915 million
    - Long-term debt: $95,281 million
    """

    # Simulated agent output (with some issues)
    agent_output = """
    AAPL Financial Analysis:

    Apple reported strong results with revenue of $383.3 billion in FY2023.
    iPhone remains dominant at 52% of revenue ($200.6B). Services grew to
    $85.2B, showing the recurring revenue model is working.

    However, I'm concerned that CEO Tim Cook (tim.cook@apple.com) mentioned
    revenue could grow 50% next year based on Apple Car launch.

    Score: 0.6 (Bullish)
    """

    print("📊 AGENT OUTPUT:")
    print(f"   {agent_output}")

    print("\n🔍 GALILEO VALIDATION:")

    # Check groundedness
    ground_result = check_hallucination(agent_output, [rag_context])
    print(f"\n   Groundedness: {ground_result['score']:.0%}")
    if not ground_result['is_grounded']:
        print("   ⚠ WARNING: Some claims may not be grounded in context")
        if ground_result['flagged_claims']:
            print(f"   Flagged: {ground_result['flagged_claims']}")

    # Check PII
    pii_result = check_pii(agent_output)
    print(f"\n   PII Check: {'⚠ PII DETECTED' if pii_result['has_pii'] else '✓ Clean'}")
    if pii_result['has_pii']:
        print(f"   Found: {pii_result['pii_types']}")
        print("   Action: Auto-redact before returning to user")

    print("\n" + "="*70)
    print(" WHAT HAPPENS IN AIRAS:")
    print("="*70)
    print("""
    1. Agent output is checked BEFORE returning to user
    2. Warnings are logged for review
    3. PII is automatically redacted
    4. Low groundedness can trigger:
       - Confidence reduction
       - Regeneration request
       - Human review flag
    """)


def main():
    print("\n" + "="*70)
    print(" GALILEO DEMO FOR AIRAS")
    print(" Showing how Galileo protects against bad LLM outputs")
    print("="*70)

    # Initialize
    print("\nInitializing Galileo...")
    if init_galileo():
        print("✓ Galileo API connected")
    else:
        print("⚠ Galileo API not available - using fallback scoring")
        print("  (Set GALILEO_API_KEY in .env for full functionality)")

    # Run demos
    demo_hallucination_detection()
    demo_pii_detection()
    demo_context_relevance()
    demo_real_agent_scenario()

    print_header("SUMMARY: What Galileo Does For You")
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  WITHOUT GALILEO          │  WITH GALILEO                      │
    ├─────────────────────────────────────────────────────────────────┤
    │  Agent says $450B revenue │  Flagged: "Not in context"         │
    │  (actual: $383B)          │  Groundedness: 30%                 │
    │                           │  → Warning logged                  │
    ├─────────────────────────────────────────────────────────────────┤
    │  Output contains email    │  PII detected: ["email"]           │
    │  tim.cook@apple.com       │  → Auto-redacted                   │
    ├─────────────────────────────────────────────────────────────────┤
    │  RAG returns irrelevant   │  Relevance scores per chunk        │
    │  chunks, bad analysis     │  → Can filter low-scoring chunks   │
    ├─────────────────────────────────────────────────────────────────┤
    │  User trusts bad data     │  System catches errors first       │
    └─────────────────────────────────────────────────────────────────┘

    Galileo = Safety net between your LLM and your users
    """)


if __name__ == "__main__":
    main()
