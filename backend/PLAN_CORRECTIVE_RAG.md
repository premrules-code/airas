# Corrective RAG (CRAG) Implementation Plan

## Executive Summary

Corrective RAG adds a **relevance grading** step after retrieval. If retrieved documents are irrelevant or ambiguous, the system self-corrects by:
1. Transforming the query and re-retrieving
2. Falling back to web search for external knowledge
3. Decomposing the query into sub-questions

This addresses a key weakness in your current pipeline: agents receive RAG context without knowing if it's actually relevant to their question.

---

## Current State Analysis

### What You Have Now

```
Query → [Basic|Intermediate|Advanced Retriever] → Top-K Documents → Agent
```

**Strengths:**
- Three retrieval levels (basic, intermediate, advanced)
- Advanced already has HyDE, multi-query, and reranking
- Metadata filtering for ticker/section

**Weaknesses:**
1. **No relevance validation** — If top-K docs are off-topic, agent hallucinates
2. **No self-correction** — Failed retrieval returns "No relevant data found" with no recovery
3. **No external fallback** — If SEC filings don't have the answer, there's no web search
4. **No query decomposition** — Complex multi-part questions aren't broken down

### Evidence of the Problem

From `context.py:46-48`:
```python
except Exception as e:
    logger.warning(f"RAG query failed for {agent_cls.AGENT_NAME}: {e}")
    result = "No relevant data found."
```

The agent proceeds with empty context — no attempt to recover.

---

## Corrective RAG Architecture

### Core Concept (from Yan et al., 2024)

```
Query → Retrieval → [GRADE each document] → Decision:
  ├─ All RELEVANT     → Use documents as-is
  ├─ Some RELEVANT    → Filter to relevant only + optional web search
  ├─ All IRRELEVANT   → Web search + query transformation + re-retrieve
  └─ AMBIGUOUS        → Decompose query into sub-questions, retrieve for each
```

### Proposed Architecture for AIRAS

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CorrectiveRetriever                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Query ──────►  IntermediateRetriever  ──────►  Top-K Docs              │
│                        │                             │                  │
│                        ▼                             ▼                  │
│               ┌────────────────┐           ┌──────────────────┐        │
│               │ RelevanceGrader │◄──────────│ Grade Each Doc   │        │
│               │ (Claude/Haiku)  │           │ RELEVANT/PARTIAL │        │
│               └────────────────┘           │ /IRRELEVANT      │        │
│                        │                   └──────────────────┘        │
│                        ▼                                                │
│               ┌────────────────────────────────────────────┐           │
│               │              Decision Router                │           │
│               ├────────────────────────────────────────────┤           │
│               │  ≥50% RELEVANT  →  Return filtered docs    │           │
│               │  <50% RELEVANT  →  Trigger corrections     │           │
│               │  0% RELEVANT    →  Full correction mode    │           │
│               └────────────────────────────────────────────┘           │
│                        │                                                │
│         ┌──────────────┼──────────────┐                                │
│         ▼              ▼              ▼                                │
│  ┌────────────┐ ┌────────────┐ ┌────────────────┐                      │
│  │Query Trans-│ │ Web Search │ │Query Decompose │                      │
│  │formation   │ │ (optional) │ │(multi-hop)     │                      │
│  └────────────┘ └────────────┘ └────────────────┘                      │
│         │              │              │                                │
│         └──────────────┴──────────────┘                                │
│                        │                                                │
│                        ▼                                                │
│               Re-retrieve + Merge + Dedupe                              │
│                        │                                                │
│                        ▼                                                │
│               Final Graded Context                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Relevance Grader (Core)

**File:** `src/rag/relevance_grader.py`

```python
class RelevanceGrader:
    """Grade retrieved documents for relevance to the query."""

    GRADES = Literal["relevant", "partial", "irrelevant"]

    def grade_document(self, query: str, document: str) -> tuple[GRADES, float]:
        """
        Returns (grade, confidence) for a single document.
        Uses Claude Haiku for speed/cost efficiency.
        """

    def grade_batch(self, query: str, documents: list[str]) -> list[GradeResult]:
        """
        Grade multiple documents in a single LLM call (batched prompt).
        More efficient than N separate calls.
        """
```

**Grading Prompt (optimized for Haiku):**
```
Given this query and document, grade the document's relevance.

Query: {query}
Document: {document[:1000]}

Respond with ONLY a JSON object:
{"grade": "relevant|partial|irrelevant", "confidence": 0.0-1.0, "reason": "brief explanation"}

Criteria:
- RELEVANT: Document directly answers or contains key information for the query
- PARTIAL: Document has some useful context but doesn't directly answer
- IRRELEVANT: Document is off-topic or wrong ticker/time period
```

**Cost/Latency Estimate:**
- Haiku: ~$0.00025 per doc grading (500 input + 50 output tokens)
- 5 docs × 10 agents = 50 gradings ≈ $0.0125 per analysis
- Latency: ~200ms per batch (parallel grading)

---

### Phase 2: Correction Strategies

**File:** `src/rag/corrections.py`

#### Strategy 1: Query Transformation

When documents are irrelevant, rewrite the query to be more specific or use different terminology.

```python
class QueryTransformer:
    """Transform failed queries into better search terms."""

    def transform(self, original_query: str, failed_docs: list[str]) -> str:
        """
        Analyze why docs failed and generate improved query.

        Techniques:
        1. Entity extraction (ticker, dates, metrics)
        2. Synonym expansion beyond EXPANSIONS dict
        3. Specificity adjustment (too broad → narrow)
        4. Temporal grounding ("recent" → "FY2023 Q4")
        """
```

**Example:**
- Original: "What is Apple's financial health?"
- Docs returned: Marketing content, product descriptions (irrelevant)
- Transformed: "Apple AAPL debt-to-equity ratio current ratio liquidity FY2023 10-K"

#### Strategy 2: Web Search Fallback (Optional)

When internal documents don't have the answer, search the web.

```python
class WebSearchFallback:
    """Search the web when internal RAG fails."""

    def search(self, query: str, ticker: str) -> list[WebResult]:
        """
        Use a web search API to find relevant financial data.

        Options:
        - Tavily API (designed for RAG, returns clean text)
        - SerpAPI + scraping
        - Bing Search API

        Filter for reputable sources:
        - SEC.gov
        - Yahoo Finance
        - Bloomberg
        - Company investor relations
        """
```

**Cost:** Tavily = $0.001 per search. Optional — can be disabled.

#### Strategy 3: Query Decomposition

For complex questions, break into sub-questions and retrieve for each.

```python
class QueryDecomposer:
    """Decompose complex queries into sub-questions."""

    def decompose(self, query: str) -> list[str]:
        """
        Break down multi-part questions.

        Example:
        "Compare Apple's revenue growth to Microsoft and analyze their debt levels"
        →
        [
          "Apple AAPL revenue growth year-over-year FY2022 FY2023",
          "Microsoft MSFT revenue growth year-over-year FY2022 FY2023",
          "Apple AAPL total debt long-term debt FY2023",
          "Microsoft MSFT total debt long-term debt FY2023"
        ]
        """
```

---

### Phase 3: Corrective Retriever Integration

**File:** `src/rag/retrieval.py` (extend existing)

```python
class CorrectiveRetriever:
    """Level 4: Corrective RAG with self-healing retrieval."""

    def __init__(self, index, enable_web_search: bool = False):
        self.index = index
        self.grader = RelevanceGrader()
        self.transformer = QueryTransformer()
        self.decomposer = QueryDecomposer()
        self.web_search = WebSearchFallback() if enable_web_search else None
        self.base_retriever = IntermediateRetriever(index)

    def retrieve(self, query: str, top_k: int = 5,
                 ticker: str = None, sections: list[str] = None,
                 max_corrections: int = 2) -> CorrectiveResult:
        """
        Retrieve with self-correction loop.

        Returns CorrectiveResult with:
        - documents: Final relevant documents
        - metadata: Grading results, corrections applied, confidence
        """

        # Step 1: Initial retrieval
        nodes = self.base_retriever.retrieve(query, top_k, ticker, sections)

        # Step 2: Grade each document
        grades = self.grader.grade_batch(query, [n.text for n in nodes])

        # Step 3: Decision based on grades
        relevant_ratio = sum(1 for g in grades if g.grade == "relevant") / len(grades)

        if relevant_ratio >= 0.5:
            # Enough relevant docs — filter and return
            return self._filter_relevant(nodes, grades)

        # Step 4: Correction loop
        for attempt in range(max_corrections):
            corrections_applied = []

            # Try query transformation
            transformed_query = self.transformer.transform(query, [n.text for n in nodes])
            new_nodes = self.base_retriever.retrieve(transformed_query, top_k, ticker, sections)
            corrections_applied.append(f"query_transform:{transformed_query[:50]}")

            # Re-grade
            new_grades = self.grader.grade_batch(query, [n.text for n in new_nodes])
            new_relevant_ratio = sum(1 for g in new_grades if g.grade == "relevant") / len(new_grades)

            if new_relevant_ratio > relevant_ratio:
                nodes, grades = new_nodes, new_grades
                relevant_ratio = new_relevant_ratio

            if relevant_ratio >= 0.5:
                break

            # Try web search if enabled and still failing
            if self.web_search and attempt == max_corrections - 1:
                web_results = self.web_search.search(query, ticker)
                # Merge web results with internal docs
                ...

        return CorrectiveResult(
            documents=self._filter_relevant(nodes, grades),
            grades=grades,
            corrections=corrections_applied,
            confidence=relevant_ratio,
        )
```

---

### Phase 4: Context Builder Integration

**File:** `src/agents/context.py` (modify existing)

```python
def _corrective_query(self, query: str, ticker: str, sections: Optional[list]) -> str:
    """Corrective RAG: grading + self-correction."""
    if self.rag.index is None:
        return self._basic_query(query)

    retriever = CorrectiveRetriever(
        self.rag.index,
        enable_web_search=self.settings.enable_web_search,
    )

    result = retriever.retrieve(
        query,
        top_k=5,
        ticker=ticker,
        sections=sections,
        max_corrections=2,
    )

    if result.documents:
        # Include confidence metadata for agent to use
        header = f"[Retrieval confidence: {result.confidence:.0%}]\n"
        if result.corrections:
            header += f"[Corrections applied: {', '.join(result.corrections)}]\n"
        return header + "\n".join(doc.text for doc in result.documents)

    return "No relevant data found after correction attempts."
```

**Add to settings.py:**
```python
rag_level: str = "intermediate"  # basic | intermediate | advanced | corrective
enable_web_search: bool = False  # Enable web fallback in corrective mode
```

---

### Phase 5: Observability & Evaluation

**File:** `src/rag/crag_metrics.py`

```python
@dataclass
class CRAGMetrics:
    """Track corrective RAG performance."""

    initial_relevant_ratio: float
    final_relevant_ratio: float
    corrections_applied: list[str]
    grading_latency_ms: float
    correction_latency_ms: float
    web_search_used: bool

    @property
    def improvement(self) -> float:
        return self.final_relevant_ratio - self.initial_relevant_ratio
```

**Langfuse Integration:**
- Log each grading decision
- Track correction attempts
- Measure retrieval quality improvement
- A/B test corrective vs. non-corrective

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/rag/relevance_grader.py` | **NEW** | Document relevance grading with Claude Haiku |
| `src/rag/corrections.py` | **NEW** | Query transformer, web search, decomposer |
| `src/rag/retrieval.py` | **MODIFY** | Add `CorrectiveRetriever` class |
| `src/agents/context.py` | **MODIFY** | Add `_corrective_query()` method |
| `config/settings.py` | **MODIFY** | Add `enable_web_search` setting |
| `src/rag/crag_metrics.py` | **NEW** | Metrics tracking |
| `requirements.txt` | **MODIFY** | Add `tavily-python` (optional) |

---

## Cost & Latency Impact

### Per-Analysis Cost (10 agents, ~50 RAG queries)

| Component | Without CRAG | With CRAG (Haiku grading) |
|-----------|--------------|---------------------------|
| RAG retrieval | ~$0.00 | ~$0.00 |
| Grading (50 docs) | $0.00 | ~$0.0125 |
| Corrections (est. 20%) | $0.00 | ~$0.003 |
| Web search (if enabled) | $0.00 | ~$0.005 |
| **Total overhead** | **$0.00** | **~$0.02** |

### Latency Impact

| Phase | Without CRAG | With CRAG |
|-------|--------------|-----------|
| Retrieval | 200ms | 200ms |
| Grading (batched) | 0ms | +150ms |
| Correction loop | 0ms | +300ms (if triggered) |
| **Total** | **200ms** | **350-650ms** |

---

## When NOT to Use CRAG

1. **Basic retrieval level** — Users who chose "basic" want speed over quality
2. **High-confidence queries** — Simple ticker + metric queries rarely fail
3. **Cached results** — If RAG results are cached, skip grading on cache hit
4. **Cost-sensitive deployments** — Haiku calls add up at scale

**Recommendation:** Make CRAG the new "advanced" level, or add as "corrective" level.

---

## Evaluation Plan

### Before Implementation: Baseline Metrics

1. Run 20 analyses with current Advanced RAG
2. Manually grade retrieved documents for relevance (human eval)
3. Measure: % relevant docs, agent confidence scores, recommendation quality

### After Implementation: A/B Test

1. Run same 20 analyses with CRAG
2. Compare:
   - % relevant docs (should increase)
   - Agent confidence scores (should increase)
   - Correction trigger rate (should be 20-40%)
   - Latency increase (should be <500ms avg)

### Success Criteria

- **Relevance improvement:** +15% relevant docs on average
- **Confidence lift:** +0.1 average agent confidence
- **Acceptable latency:** <500ms additional per query
- **Cost efficiency:** <$0.05 per full analysis

---

## Implementation Order

1. **Phase 1: RelevanceGrader** (core, test in isolation)
2. **Phase 2: QueryTransformer** (most impactful correction)
3. **Phase 3: CorrectiveRetriever** (integrate grader + transformer)
4. **Phase 4: Context Builder** (expose via rag_level="corrective")
5. **Phase 5: Metrics & Langfuse** (observability)
6. **Phase 6: Web Search** (optional, add later if needed)
7. **Phase 7: Query Decomposition** (for multi-hop questions)

---

## Questions to Decide Before Implementation

1. **Which model for grading?**
   - Claude Haiku (fastest, cheapest)
   - Claude Sonnet (better judgment, 10x cost)
   - OpenAI GPT-4o-mini (alternative)

2. **Web search provider?**
   - Tavily (RAG-optimized, $0.001/search)
   - SerpAPI (more control, more code)
   - None (disable for now)

3. **Grading threshold?**
   - 50% relevant = accept (current proposal)
   - 70% relevant = stricter (fewer false positives)
   - Adaptive based on query complexity

4. **Max correction attempts?**
   - 1 (fast, limited recovery)
   - 2 (balanced)
   - 3 (thorough, slower)

5. **Should agents see grading metadata?**
   - Yes: Agent knows retrieval confidence, adjusts own confidence
   - No: Keep agent prompts unchanged

---

---

# PART 2: Evaluation Framework

## Overview

We'll build a rigorous evaluation pipeline that measures RAG quality **before and after** CRAG implementation across multiple dimensions:

| Dimension | Metrics | Tool |
|-----------|---------|------|
| **Retrieval Quality** | Precision, Recall, MRR, NDCG | Custom + RAGAS |
| **Answer Quality** | Faithfulness, Relevancy, Correctness | RAGAS |
| **Cost** | $ per query, $ per analysis | Custom tracking |
| **Latency** | p50, p95, p99 per phase | Custom tracking |
| **Agent Impact** | Confidence scores, score variance | Custom tracking |

---

## Evaluation Dataset

### Ground Truth Construction

**File:** `evals/datasets/rag_golden_set.json`

We need human-labeled query-answer pairs from actual SEC filings:

```json
{
  "dataset_version": "1.0",
  "created_at": "2024-02-18",
  "queries": [
    {
      "id": "q001",
      "ticker": "AAPL",
      "query": "What was Apple's total revenue in FY2023?",
      "ground_truth_answer": "Apple reported total net sales of $383.3 billion for fiscal year 2023.",
      "ground_truth_chunks": [
        "AAPL_10K_2023-11-03.txt:chunk_42",
        "AAPL_10K_2023-11-03.txt:chunk_43"
      ],
      "difficulty": "easy",
      "category": "financial_metrics",
      "requires_calculation": false
    },
    {
      "id": "q002",
      "ticker": "AAPL",
      "query": "How has Apple's debt-to-equity ratio changed over the past 3 years?",
      "ground_truth_answer": "Apple's debt-to-equity ratio was 1.81 in FY2021, 1.95 in FY2022, and 1.79 in FY2023, showing slight deleveraging in the most recent year.",
      "ground_truth_chunks": [
        "AAPL_10K_2023-11-03.txt:chunk_87",
        "AAPL_10K_2022-10-28.txt:chunk_91",
        "AAPL_10K_2021-10-29.txt:chunk_85"
      ],
      "difficulty": "medium",
      "category": "financial_analysis",
      "requires_calculation": true
    },
    {
      "id": "q003",
      "ticker": "AAPL",
      "query": "What are the main risk factors Apple disclosed?",
      "ground_truth_answer": "Apple disclosed risks including: global economic conditions, supply chain disruptions, competition, regulatory changes, and cybersecurity threats.",
      "ground_truth_chunks": [
        "AAPL_10K_2023-11-03.txt:chunk_156",
        "AAPL_10K_2023-11-03.txt:chunk_157",
        "AAPL_10K_2023-11-03.txt:chunk_158"
      ],
      "difficulty": "easy",
      "category": "risk_factors",
      "requires_calculation": false
    }
  ]
}
```

**Dataset Requirements:**
- 50-100 queries across 3-5 tickers (AAPL, MSFT, GOOGL, TSLA, META)
- Categories: financial_metrics, risk_factors, competitive_analysis, earnings, guidance
- Difficulty levels: easy (single fact), medium (multi-fact), hard (reasoning required)
- Ground truth chunks manually identified from indexed documents

### Generating the Dataset

```python
# evals/scripts/generate_golden_set.py

class GoldenSetGenerator:
    """Generate evaluation dataset with human-in-the-loop verification."""

    def generate_candidate_queries(self, ticker: str, num_queries: int = 20):
        """Use Claude to generate diverse financial queries."""
        prompt = f"""
        Generate {num_queries} diverse financial analysis questions about {ticker}.

        Categories to cover:
        1. Revenue/earnings metrics (5 questions)
        2. Balance sheet items (4 questions)
        3. Cash flow analysis (3 questions)
        4. Risk factors (3 questions)
        5. Competitive positioning (3 questions)
        6. Multi-year trends (2 questions)

        For each question, estimate difficulty (easy/medium/hard).
        Return as JSON array.
        """
        # Generate candidates, then human reviews and adds ground truth

    def find_ground_truth_chunks(self, query: str, ticker: str):
        """Help human identify which chunks contain the answer."""
        # Retrieve top-20 chunks
        # Present to human for labeling
        # Human marks which chunks are truly relevant
```

---

## Retrieval Metrics

### File: `evals/metrics/retrieval_metrics.py`

```python
from dataclasses import dataclass
from typing import List, Set
import numpy as np

@dataclass
class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""

    # Core metrics
    precision_at_k: float      # Relevant retrieved / Total retrieved
    recall_at_k: float         # Relevant retrieved / Total relevant
    f1_at_k: float             # Harmonic mean of P and R

    # Ranking metrics
    mrr: float                 # Mean Reciprocal Rank (position of first relevant)
    ndcg_at_k: float           # Normalized Discounted Cumulative Gain
    map_at_k: float            # Mean Average Precision

    # Hit rate
    hit_rate: float            # % of queries with at least 1 relevant doc

    # Context quality
    avg_chunk_relevance: float # Average grader score across chunks


class RetrievalEvaluator:
    """Compute retrieval metrics against ground truth."""

    def evaluate(
        self,
        retrieved_chunk_ids: List[str],
        ground_truth_chunk_ids: List[str],
        relevance_scores: List[float] = None,  # From grader
    ) -> RetrievalMetrics:

        retrieved_set = set(retrieved_chunk_ids)
        relevant_set = set(ground_truth_chunk_ids)

        # Precision@K
        relevant_retrieved = retrieved_set & relevant_set
        precision = len(relevant_retrieved) / len(retrieved_chunk_ids) if retrieved_chunk_ids else 0

        # Recall@K
        recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0

        # F1@K
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, chunk_id in enumerate(retrieved_chunk_ids):
            if chunk_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break

        # NDCG@K
        ndcg = self._compute_ndcg(retrieved_chunk_ids, relevant_set, k=len(retrieved_chunk_ids))

        # MAP@K
        map_score = self._compute_map(retrieved_chunk_ids, relevant_set)

        # Hit rate
        hit_rate = 1.0 if relevant_retrieved else 0.0

        # Average relevance from grader
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0

        return RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            f1_at_k=f1,
            mrr=mrr,
            ndcg_at_k=ndcg,
            map_at_k=map_score,
            hit_rate=hit_rate,
            avg_chunk_relevance=avg_relevance,
        )

    def _compute_ndcg(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain."""
        dcg = sum(
            (1.0 if chunk_id in relevant else 0.0) / np.log2(i + 2)
            for i, chunk_id in enumerate(retrieved[:k])
        )

        # Ideal DCG (all relevant docs at top)
        ideal_relevance = [1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant))
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))

        return dcg / idcg if idcg > 0 else 0.0

    def _compute_map(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Mean Average Precision."""
        precisions = []
        relevant_count = 0

        for i, chunk_id in enumerate(retrieved):
            if chunk_id in relevant:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))

        return np.mean(precisions) if precisions else 0.0
```

---

## Answer Quality Metrics (RAGAS)

### File: `evals/metrics/ragas_metrics.py`

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity,
)
from datasets import Dataset

class RAGASEvaluator:
    """Evaluate RAG answer quality using RAGAS framework."""

    METRICS = [
        faithfulness,        # Is answer grounded in context? (no hallucination)
        answer_relevancy,    # Does answer address the question?
        context_precision,   # Are retrieved docs actually useful?
        context_recall,      # Did we retrieve all needed info?
        answer_correctness,  # Is the answer factually correct?
        answer_similarity,   # Semantic similarity to ground truth
    ]

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> dict:
        """Evaluate a single RAG response."""

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(dataset, metrics=self.METRICS)
        return result.to_pandas().iloc[0].to_dict()

    def evaluate_batch(self, samples: List[dict]) -> pd.DataFrame:
        """Evaluate a batch of RAG responses."""

        data = {
            "question": [s["question"] for s in samples],
            "answer": [s["answer"] for s in samples],
            "contexts": [s["contexts"] for s in samples],
            "ground_truth": [s["ground_truth"] for s in samples],
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(dataset, metrics=self.METRICS)
        return result.to_pandas()
```

**RAGAS Metrics Explained:**

| Metric | What it Measures | Score Range |
|--------|------------------|-------------|
| **Faithfulness** | % of answer claims supported by context | 0-1 (higher = less hallucination) |
| **Answer Relevancy** | Does answer address the actual question? | 0-1 |
| **Context Precision** | % of retrieved chunks that are useful | 0-1 |
| **Context Recall** | Did we get all chunks needed for the answer? | 0-1 |
| **Answer Correctness** | Factual accuracy vs ground truth | 0-1 |
| **Answer Similarity** | Semantic similarity to ground truth | 0-1 |

---

## Cost & Latency Tracking

### File: `evals/metrics/cost_latency.py`

```python
import time
from dataclasses import dataclass, field
from typing import Optional
from contextlib import contextmanager

@dataclass
class CostBreakdown:
    """Track costs per component."""

    # Embedding costs (OpenAI)
    embedding_tokens: int = 0
    embedding_cost: float = 0.0  # $0.00002 per 1K tokens for text-embedding-3-small

    # Grading costs (Claude Haiku)
    grading_input_tokens: int = 0
    grading_output_tokens: int = 0
    grading_cost: float = 0.0  # Haiku: $0.25/1M input, $1.25/1M output

    # Correction costs (Claude Haiku)
    correction_input_tokens: int = 0
    correction_output_tokens: int = 0
    correction_cost: float = 0.0

    # Web search costs (Tavily)
    web_searches: int = 0
    web_search_cost: float = 0.0  # $0.001 per search

    @property
    def total_cost(self) -> float:
        return (
            self.embedding_cost +
            self.grading_cost +
            self.correction_cost +
            self.web_search_cost
        )


@dataclass
class LatencyBreakdown:
    """Track latency per phase in milliseconds."""

    retrieval_ms: float = 0.0
    grading_ms: float = 0.0
    correction_ms: float = 0.0
    web_search_ms: float = 0.0
    total_ms: float = 0.0

    # Percentiles (computed across batch)
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None


class CostLatencyTracker:
    """Track costs and latency during RAG execution."""

    # Pricing constants
    EMBEDDING_COST_PER_1K = 0.00002  # text-embedding-3-small
    HAIKU_INPUT_PER_1M = 0.25
    HAIKU_OUTPUT_PER_1M = 1.25
    SONNET_INPUT_PER_1M = 3.00
    SONNET_OUTPUT_PER_1M = 15.00
    TAVILY_PER_SEARCH = 0.001

    def __init__(self):
        self.cost = CostBreakdown()
        self.latency = LatencyBreakdown()
        self._phase_start: Optional[float] = None

    @contextmanager
    def track_phase(self, phase: str):
        """Context manager to track latency of a phase."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            setattr(self.latency, f"{phase}_ms", elapsed_ms)
            self.latency.total_ms += elapsed_ms

    def record_grading(self, input_tokens: int, output_tokens: int, model: str = "haiku"):
        """Record tokens and cost for grading."""
        self.cost.grading_input_tokens += input_tokens
        self.cost.grading_output_tokens += output_tokens

        if model == "haiku":
            self.cost.grading_cost += (
                input_tokens * self.HAIKU_INPUT_PER_1M / 1_000_000 +
                output_tokens * self.HAIKU_OUTPUT_PER_1M / 1_000_000
            )
        else:  # sonnet
            self.cost.grading_cost += (
                input_tokens * self.SONNET_INPUT_PER_1M / 1_000_000 +
                output_tokens * self.SONNET_OUTPUT_PER_1M / 1_000_000
            )

    def record_web_search(self, num_searches: int = 1):
        """Record web search costs."""
        self.cost.web_searches += num_searches
        self.cost.web_search_cost += num_searches * self.TAVILY_PER_SEARCH

    def to_dict(self) -> dict:
        """Export metrics as dictionary."""
        return {
            "cost": {
                "embedding": self.cost.embedding_cost,
                "grading": self.cost.grading_cost,
                "correction": self.cost.correction_cost,
                "web_search": self.cost.web_search_cost,
                "total": self.cost.total_cost,
            },
            "latency": {
                "retrieval_ms": self.latency.retrieval_ms,
                "grading_ms": self.latency.grading_ms,
                "correction_ms": self.latency.correction_ms,
                "web_search_ms": self.latency.web_search_ms,
                "total_ms": self.latency.total_ms,
            },
        }
```

---

## Agent Impact Metrics

### File: `evals/metrics/agent_metrics.py`

```python
@dataclass
class AgentImpactMetrics:
    """Measure how RAG quality affects agent outputs."""

    # Confidence
    avg_confidence: float           # Mean agent confidence
    confidence_std: float           # Variance in confidence
    low_confidence_rate: float      # % of agents with confidence < 0.5

    # Score consistency
    score_variance: float           # Variance in agent scores
    outlier_rate: float             # % of scores > 2 std from mean

    # Recommendation stability
    recommendation_flips: int       # How often does rerun change recommendation?

    # Qualitative
    uses_rag_context: float         # % of agent summaries citing RAG data
    hallucination_rate: float       # % of claims not in RAG context (estimated)


class AgentImpactEvaluator:
    """Evaluate how RAG changes affect agent behavior."""

    def compare_runs(
        self,
        baseline_outputs: List[AgentOutput],
        crag_outputs: List[AgentOutput],
    ) -> dict:
        """Compare agent outputs between baseline and CRAG."""

        baseline_conf = np.mean([o.confidence for o in baseline_outputs])
        crag_conf = np.mean([o.confidence for o in crag_outputs])

        baseline_scores = [o.score for o in baseline_outputs]
        crag_scores = [o.score for o in crag_outputs]

        return {
            "confidence_delta": crag_conf - baseline_conf,
            "confidence_baseline": baseline_conf,
            "confidence_crag": crag_conf,
            "score_variance_baseline": np.var(baseline_scores),
            "score_variance_crag": np.var(crag_scores),
            "score_correlation": np.corrcoef(baseline_scores, crag_scores)[0, 1],
        }
```

---

## Evaluation Runner

### File: `evals/run_evaluation.py`

```python
#!/usr/bin/env python3
"""Run comprehensive RAG evaluation: baseline vs CRAG."""

import json
import time
from pathlib import Path
from dataclasses import asdict

from evals.metrics.retrieval_metrics import RetrievalEvaluator
from evals.metrics.ragas_metrics import RAGASEvaluator
from evals.metrics.cost_latency import CostLatencyTracker
from evals.metrics.agent_metrics import AgentImpactEvaluator

from src.rag.supabase_rag import SupabaseRAG
from src.rag.retrieval import IntermediateRetriever, AdvancedRetriever, CorrectiveRetriever
from src.agents.context import ContextBuilder


class RAGEvaluationRunner:
    """Run full evaluation suite."""

    def __init__(self, golden_set_path: str):
        self.golden_set = self._load_golden_set(golden_set_path)
        self.retrieval_eval = RetrievalEvaluator()
        self.ragas_eval = RAGASEvaluator()
        self.agent_eval = AgentImpactEvaluator()

        # Setup RAG
        self.rag = SupabaseRAG()
        self.rag.load_index()

    def run_full_evaluation(self, output_dir: str = "evals/results"):
        """Run baseline vs CRAG comparison."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": len(self.golden_set["queries"]),
            "baseline": self._evaluate_retriever("intermediate"),
            "crag": self._evaluate_retriever("corrective"),
        }

        # Compute deltas
        results["comparison"] = self._compute_comparison(
            results["baseline"], results["crag"]
        )

        # Save results
        with open(output_path / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Generate report
        self._generate_report(results, output_path / "evaluation_report.md")

        return results

    def _evaluate_retriever(self, level: str) -> dict:
        """Evaluate a single retrieval level."""

        retrieval_metrics = []
        ragas_metrics = []
        cost_trackers = []
        latency_samples = []

        for query_data in self.golden_set["queries"]:
            tracker = CostLatencyTracker()

            # Retrieve
            with tracker.track_phase("retrieval"):
                if level == "intermediate":
                    retriever = IntermediateRetriever(self.rag.index)
                elif level == "corrective":
                    retriever = CorrectiveRetriever(self.rag.index)

                nodes = retriever.retrieve(
                    query_data["query"],
                    top_k=5,
                    ticker=query_data["ticker"],
                )

            # Get chunk IDs
            retrieved_ids = [n.node.node_id for n in nodes]
            ground_truth_ids = query_data["ground_truth_chunks"]

            # Retrieval metrics
            ret_metrics = self.retrieval_eval.evaluate(
                retrieved_ids, ground_truth_ids
            )
            retrieval_metrics.append(asdict(ret_metrics))

            # RAGAS metrics (need to generate answer first)
            contexts = [n.text for n in nodes]
            answer = self._generate_answer(query_data["query"], contexts)

            ragas_result = self.ragas_eval.evaluate_single(
                question=query_data["query"],
                answer=answer,
                contexts=contexts,
                ground_truth=query_data["ground_truth_answer"],
            )
            ragas_metrics.append(ragas_result)

            cost_trackers.append(tracker.to_dict())
            latency_samples.append(tracker.latency.total_ms)

        # Aggregate metrics
        return {
            "retrieval": self._aggregate_metrics(retrieval_metrics),
            "ragas": self._aggregate_metrics(ragas_metrics),
            "cost": self._aggregate_costs(cost_trackers),
            "latency": {
                "mean_ms": np.mean(latency_samples),
                "p50_ms": np.percentile(latency_samples, 50),
                "p95_ms": np.percentile(latency_samples, 95),
                "p99_ms": np.percentile(latency_samples, 99),
            },
        }

    def _compute_comparison(self, baseline: dict, crag: dict) -> dict:
        """Compute improvement metrics."""

        def delta(metric_path: str) -> float:
            b_val = self._get_nested(baseline, metric_path)
            c_val = self._get_nested(crag, metric_path)
            return c_val - b_val

        def pct_change(metric_path: str) -> float:
            b_val = self._get_nested(baseline, metric_path)
            c_val = self._get_nested(crag, metric_path)
            return ((c_val - b_val) / b_val * 100) if b_val != 0 else 0

        return {
            "retrieval": {
                "precision_delta": delta("retrieval.precision_at_k"),
                "recall_delta": delta("retrieval.recall_at_k"),
                "ndcg_delta": delta("retrieval.ndcg_at_k"),
                "mrr_delta": delta("retrieval.mrr"),
            },
            "ragas": {
                "faithfulness_delta": delta("ragas.faithfulness"),
                "relevancy_delta": delta("ragas.answer_relevancy"),
                "correctness_delta": delta("ragas.answer_correctness"),
                "context_precision_delta": delta("ragas.context_precision"),
            },
            "cost": {
                "cost_increase_pct": pct_change("cost.total"),
                "cost_increase_usd": delta("cost.total"),
            },
            "latency": {
                "latency_increase_pct": pct_change("latency.mean_ms"),
                "latency_increase_ms": delta("latency.mean_ms"),
            },
        }

    def _generate_report(self, results: dict, output_path: Path):
        """Generate markdown report."""

        report = f"""# RAG Evaluation Report

Generated: {results['timestamp']}
Dataset Size: {results['dataset_size']} queries

## Executive Summary

| Metric | Baseline | CRAG | Delta | % Change |
|--------|----------|------|-------|----------|
| Precision@5 | {results['baseline']['retrieval']['precision_at_k']:.3f} | {results['crag']['retrieval']['precision_at_k']:.3f} | {results['comparison']['retrieval']['precision_delta']:+.3f} | - |
| Recall@5 | {results['baseline']['retrieval']['recall_at_k']:.3f} | {results['crag']['retrieval']['recall_at_k']:.3f} | {results['comparison']['retrieval']['recall_delta']:+.3f} | - |
| NDCG@5 | {results['baseline']['retrieval']['ndcg_at_k']:.3f} | {results['crag']['retrieval']['ndcg_at_k']:.3f} | {results['comparison']['retrieval']['ndcg_delta']:+.3f} | - |
| Faithfulness | {results['baseline']['ragas']['faithfulness']:.3f} | {results['crag']['ragas']['faithfulness']:.3f} | {results['comparison']['ragas']['faithfulness_delta']:+.3f} | - |
| Answer Correctness | {results['baseline']['ragas']['answer_correctness']:.3f} | {results['crag']['ragas']['answer_correctness']:.3f} | {results['comparison']['ragas']['correctness_delta']:+.3f} | - |
| Cost per Query | ${results['baseline']['cost']['total']:.4f} | ${results['crag']['cost']['total']:.4f} | +${results['comparison']['cost']['cost_increase_usd']:.4f} | +{results['comparison']['cost']['cost_increase_pct']:.1f}% |
| Latency (p50) | {results['baseline']['latency']['p50_ms']:.0f}ms | {results['crag']['latency']['p50_ms']:.0f}ms | +{results['comparison']['latency']['latency_increase_ms']:.0f}ms | +{results['comparison']['latency']['latency_increase_pct']:.1f}% |

## Detailed Retrieval Metrics

### Baseline (Intermediate RAG)
- Precision@5: {results['baseline']['retrieval']['precision_at_k']:.3f}
- Recall@5: {results['baseline']['retrieval']['recall_at_k']:.3f}
- F1@5: {results['baseline']['retrieval']['f1_at_k']:.3f}
- MRR: {results['baseline']['retrieval']['mrr']:.3f}
- NDCG@5: {results['baseline']['retrieval']['ndcg_at_k']:.3f}
- Hit Rate: {results['baseline']['retrieval']['hit_rate']:.1%}

### CRAG (Corrective RAG)
- Precision@5: {results['crag']['retrieval']['precision_at_k']:.3f}
- Recall@5: {results['crag']['retrieval']['recall_at_k']:.3f}
- F1@5: {results['crag']['retrieval']['f1_at_k']:.3f}
- MRR: {results['crag']['retrieval']['mrr']:.3f}
- NDCG@5: {results['crag']['retrieval']['ndcg_at_k']:.3f}
- Hit Rate: {results['crag']['retrieval']['hit_rate']:.1%}

## RAGAS Answer Quality

### Baseline
- Faithfulness: {results['baseline']['ragas']['faithfulness']:.3f}
- Answer Relevancy: {results['baseline']['ragas']['answer_relevancy']:.3f}
- Context Precision: {results['baseline']['ragas']['context_precision']:.3f}
- Context Recall: {results['baseline']['ragas']['context_recall']:.3f}
- Answer Correctness: {results['baseline']['ragas']['answer_correctness']:.3f}

### CRAG
- Faithfulness: {results['crag']['ragas']['faithfulness']:.3f}
- Answer Relevancy: {results['crag']['ragas']['answer_relevancy']:.3f}
- Context Precision: {results['crag']['ragas']['context_precision']:.3f}
- Context Recall: {results['crag']['ragas']['context_recall']:.3f}
- Answer Correctness: {results['crag']['ragas']['answer_correctness']:.3f}

## Cost Analysis

| Component | Baseline | CRAG |
|-----------|----------|------|
| Embedding | ${results['baseline']['cost'].get('embedding', 0):.5f} | ${results['crag']['cost'].get('embedding', 0):.5f} |
| Grading | $0.00000 | ${results['crag']['cost'].get('grading', 0):.5f} |
| Correction | $0.00000 | ${results['crag']['cost'].get('correction', 0):.5f} |
| Web Search | $0.00000 | ${results['crag']['cost'].get('web_search', 0):.5f} |
| **Total** | **${results['baseline']['cost']['total']:.5f}** | **${results['crag']['cost']['total']:.5f}** |

## Latency Analysis

| Percentile | Baseline | CRAG |
|------------|----------|------|
| p50 | {results['baseline']['latency']['p50_ms']:.0f}ms | {results['crag']['latency']['p50_ms']:.0f}ms |
| p95 | {results['baseline']['latency']['p95_ms']:.0f}ms | {results['crag']['latency']['p95_ms']:.0f}ms |
| p99 | {results['baseline']['latency']['p99_ms']:.0f}ms | {results['crag']['latency']['p99_ms']:.0f}ms |

## Recommendations

Based on the evaluation results:

1. **Quality vs Cost Trade-off**: CRAG improves [X] at a cost of [Y]
2. **When to Use CRAG**: Recommended for [use cases]
3. **Configuration Tuning**: Consider adjusting [parameters]
"""

        with open(output_path, "w") as f:
            f.write(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--golden-set", default="evals/datasets/rag_golden_set.json")
    parser.add_argument("--output-dir", default="evals/results")
    args = parser.parse_args()

    runner = RAGEvaluationRunner(args.golden_set)
    results = runner.run_full_evaluation(args.output_dir)

    print(f"Evaluation complete. Results saved to {args.output_dir}")
```

---

## Directory Structure

```
evals/
├── datasets/
│   ├── rag_golden_set.json       # Human-labeled ground truth
│   └── generate_golden_set.py    # Tool to create dataset
├── metrics/
│   ├── __init__.py
│   ├── retrieval_metrics.py      # Precision, Recall, MRR, NDCG
│   ├── ragas_metrics.py          # RAGAS wrapper
│   ├── cost_latency.py           # Cost and latency tracking
│   └── agent_metrics.py          # Agent impact metrics
├── results/
│   ├── evaluation_results.json   # Raw results
│   └── evaluation_report.md      # Human-readable report
├── run_evaluation.py             # Main evaluation script
└── README.md                     # Evaluation documentation
```

---

## Running the Evaluation

### Step 1: Generate Golden Set (One-time)

```bash
# Generate candidate queries
python evals/datasets/generate_golden_set.py --ticker AAPL --output evals/datasets/aapl_candidates.json

# Human review and label ground truth chunks (manual step)
# Edit the JSON to add ground_truth_answer and ground_truth_chunks

# Combine into final golden set
python evals/datasets/combine_golden_sets.py --output evals/datasets/rag_golden_set.json
```

### Step 2: Run Baseline Evaluation

```bash
# Evaluate current intermediate retriever
python evals/run_evaluation.py --mode baseline --output evals/results/baseline_$(date +%Y%m%d).json
```

### Step 3: Implement CRAG

```bash
# After implementing corrective retriever...
```

### Step 4: Run CRAG Evaluation

```bash
# Evaluate corrective retriever
python evals/run_evaluation.py --mode crag --output evals/results/crag_$(date +%Y%m%d).json
```

### Step 5: Compare Results

```bash
# Generate comparison report
python evals/run_evaluation.py --mode compare \
  --baseline evals/results/baseline_20240218.json \
  --crag evals/results/crag_20240218.json \
  --output evals/results/comparison_report.md
```

---

## Success Criteria

| Metric | Target | Fail if |
|--------|--------|---------|
| Precision@5 improvement | ≥+10% | <+5% |
| Recall@5 improvement | ≥+15% | <+5% |
| Faithfulness improvement | ≥+0.05 | <0 |
| Answer Correctness improvement | ≥+0.05 | <0 |
| Cost increase | <$0.03/query | >$0.10/query |
| Latency increase (p95) | <500ms | >1000ms |

---

## References

- [Corrective Retrieval Augmented Generation (Yan et al., 2024)](https://arxiv.org/abs/2401.15884)
- [Self-RAG: Learning to Retrieve, Generate, and Critique (Asai et al., 2023)](https://arxiv.org/abs/2310.11511)
- [LangGraph CRAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [LlamaIndex Evaluation](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/)
