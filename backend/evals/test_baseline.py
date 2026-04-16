"""Baseline RAG evaluation using IntermediateRetriever.

Run with: pytest evals/test_baseline.py -v
"""

import json
import time
from datetime import datetime
from pathlib import Path

import pytest

from src.rag.retrieval import IntermediateRetriever
from evals.metrics.retrieval_metrics import RetrievalEvaluator
from evals.metrics.cost_latency import CostLatencyTracker, aggregate_trackers


class TestBaselineRetrieval:
    """Evaluate baseline (IntermediateRetriever) performance."""

    @pytest.fixture(autouse=True)
    def setup(self, rag_index, golden_set, retrieval_evaluator):
        """Setup test fixtures."""
        self.index = rag_index
        self.golden_set = golden_set
        self.evaluator = retrieval_evaluator
        self.retriever = IntermediateRetriever(rag_index)

    def test_retrieval_metrics_per_query(self):
        """Test retrieval metrics for each query in golden set."""
        results = []

        for query_data in self.golden_set["queries"]:
            tracker = CostLatencyTracker()

            # Retrieve
            with tracker.track_phase("retrieval"):
                nodes = self.retriever.retrieve(
                    query=query_data["query"],
                    top_k=5,
                    ticker=query_data["ticker"],
                )

            # Get IDs
            retrieved_ids = [n.node.node_id for n in nodes]
            ground_truth_ids = query_data.get("ground_truth_chunks", [])

            # Skip if no ground truth chunks defined
            if not ground_truth_ids:
                continue

            # Evaluate
            metrics = self.evaluator.evaluate(retrieved_ids, ground_truth_ids)

            results.append({
                "query_id": query_data["id"],
                "query": query_data["query"],
                "ticker": query_data["ticker"],
                "category": query_data.get("category"),
                "difficulty": query_data.get("difficulty"),
                "metrics": metrics.to_dict(),
                "latency": tracker.latency.to_dict(),
                "retrieved_count": len(nodes),
                "ground_truth_count": len(ground_truth_ids),
            })

        # Save results
        if results:
            self._save_results(results, "baseline_retrieval")

        # At least one query should have been evaluated
        assert len(results) > 0 or len(self.golden_set["queries"]) == 0

    def test_aggregate_metrics(self):
        """Test aggregate metrics across all queries."""
        all_results = []
        trackers = []

        for query_data in self.golden_set["queries"]:
            tracker = CostLatencyTracker()

            with tracker.track_phase("retrieval"):
                nodes = self.retriever.retrieve(
                    query=query_data["query"],
                    top_k=5,
                    ticker=query_data["ticker"],
                )

            retrieved_ids = [n.node.node_id for n in nodes]
            ground_truth_ids = query_data.get("ground_truth_chunks", [])

            # For queries without ground truth, use empty list
            all_results.append({
                "retrieved_ids": retrieved_ids,
                "ground_truth_ids": ground_truth_ids,
            })
            trackers.append(tracker)

        # Filter to queries with ground truth
        results_with_gt = [r for r in all_results if r["ground_truth_ids"]]

        if results_with_gt:
            aggregated = self.evaluator.evaluate_batch(results_with_gt)
            latency_stats = aggregate_trackers(trackers)

            print("\n" + "=" * 60)
            print("BASELINE AGGREGATE METRICS")
            print("=" * 60)
            print(f"Queries evaluated: {aggregated['num_queries']}")
            print(f"Precision@5: {aggregated['precision_at_k']['mean']:.3f} (±{aggregated['precision_at_k']['std']:.3f})")
            print(f"Recall@5: {aggregated['recall_at_k']['mean']:.3f} (±{aggregated['recall_at_k']['std']:.3f})")
            print(f"MRR: {aggregated['mrr']['mean']:.3f} (±{aggregated['mrr']['std']:.3f})")
            print(f"NDCG@5: {aggregated['ndcg_at_k']['mean']:.3f} (±{aggregated['ndcg_at_k']['std']:.3f})")
            print(f"Hit Rate: {aggregated['hit_rate']['mean']:.1%}")
            print(f"Latency p50: {latency_stats['latency']['p50_ms']:.0f}ms")
            print(f"Latency p95: {latency_stats['latency']['p95_ms']:.0f}ms")
            print("=" * 60)

            # Save aggregated results
            self._save_results({
                "retrieval": aggregated,
                "latency": latency_stats,
            }, "baseline_aggregate")

    def _save_results(self, results, name: str):
        """Save results to JSON file."""
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{name}_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "retriever": "IntermediateRetriever",
                "level": "baseline",
                "results": results,
            }, f, indent=2)

        print(f"\nResults saved to: {output_path}")


class TestBaselineQuality:
    """Test answer quality using RAGAS (slower, requires LLM calls)."""

    @pytest.fixture(autouse=True)
    def setup(self, rag_index, golden_set, ragas_evaluator, settings):
        """Setup test fixtures."""
        self.index = rag_index
        self.golden_set = golden_set
        self.ragas_evaluator = ragas_evaluator
        self.retriever = IntermediateRetriever(rag_index)

        # Simple answer generator using Claude
        import anthropic
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model

    def _generate_answer(self, question: str, contexts: list[str]) -> str:
        """Generate answer from contexts using Claude."""
        context_text = "\n\n".join(contexts)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": f"""Based on the following context, answer the question.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context_text}

Question: {question}

Answer:"""
            }],
        )
        return response.content[0].text

    @pytest.mark.slow
    @pytest.mark.requires_api
    def test_ragas_metrics(self):
        """Test RAGAS answer quality metrics (slow - makes LLM calls)."""
        samples = []

        for query_data in self.golden_set["queries"][:5]:  # Limit for cost
            # Retrieve
            nodes = self.retriever.retrieve(
                query=query_data["query"],
                top_k=5,
                ticker=query_data["ticker"],
            )

            contexts = [n.text for n in nodes]

            # Generate answer
            answer = self._generate_answer(query_data["query"], contexts)

            samples.append({
                "question": query_data["query"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": query_data["ground_truth_answer"],
            })

        if samples:
            results = self.ragas_evaluator.evaluate_batch(samples)
            aggregated = self.ragas_evaluator.aggregate_results(results)

            print("\n" + "=" * 60)
            print("BASELINE RAGAS METRICS")
            print("=" * 60)
            print(f"Faithfulness: {aggregated['faithfulness']['mean']:.3f}")
            print(f"Answer Relevancy: {aggregated['answer_relevancy']['mean']:.3f}")
            print(f"Context Precision: {aggregated['context_precision']['mean']:.3f}")
            print(f"Context Recall: {aggregated['context_recall']['mean']:.3f}")
            print(f"Answer Correctness: {aggregated['answer_correctness']['mean']:.3f}")
            print("=" * 60)

            # Save
            output_dir = Path(__file__).parent / "results"
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            with open(output_dir / f"baseline_ragas_{timestamp}.json", "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "retriever": "IntermediateRetriever",
                    "level": "baseline",
                    "ragas": aggregated,
                    "num_samples": len(samples),
                }, f, indent=2)
