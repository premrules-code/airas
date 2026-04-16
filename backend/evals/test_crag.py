"""Corrective RAG evaluation using CorrectiveRetriever.

Run with: pytest evals/test_crag.py -v
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.rag.retrieval import CorrectiveRetriever
from evals.metrics.retrieval_metrics import RetrievalEvaluator
from evals.metrics.cost_latency import CostLatencyTracker, aggregate_trackers


class TestCorrectiveRetrieval:
    """Evaluate CorrectiveRetriever (CRAG) performance."""

    @pytest.fixture(autouse=True)
    def setup(self, rag_index, golden_set, retrieval_evaluator):
        """Setup test fixtures."""
        self.index = rag_index
        self.golden_set = golden_set
        self.evaluator = retrieval_evaluator
        self.retriever = CorrectiveRetriever(
            rag_index,
            relevance_threshold=0.5,
            max_corrections=2,
        )

    def test_retrieval_metrics_per_query(self):
        """Test CRAG retrieval metrics for each query in golden set."""
        results = []

        for query_data in self.golden_set["queries"]:
            tracker = CostLatencyTracker()

            # Retrieve with correction
            with tracker.track_phase("retrieval"):
                result = self.retriever.retrieve(
                    query=query_data["query"],
                    top_k=5,
                    ticker=query_data["ticker"],
                )

            # Get IDs
            retrieved_ids = [n.node.node_id for n in result.nodes]
            ground_truth_ids = query_data.get("ground_truth_chunks", [])

            # Get relevance scores from grader
            relevance_scores = [g.relevance_score for g in result.grades]

            # Skip if no ground truth chunks defined
            if not ground_truth_ids:
                continue

            # Evaluate
            metrics = self.evaluator.evaluate(
                retrieved_ids, ground_truth_ids, relevance_scores
            )

            results.append({
                "query_id": query_data["id"],
                "query": query_data["query"],
                "ticker": query_data["ticker"],
                "category": query_data.get("category"),
                "difficulty": query_data.get("difficulty"),
                "metrics": metrics.to_dict(),
                "latency": tracker.latency.to_dict(),
                "retrieved_count": len(result.nodes),
                "ground_truth_count": len(ground_truth_ids),
                # CRAG-specific metrics
                "crag": {
                    "initial_relevant_ratio": result.initial_relevant_ratio,
                    "final_relevant_ratio": result.final_relevant_ratio,
                    "corrections_applied": result.corrections_applied,
                    "num_corrections": result.num_correction_attempts,
                    "improved": result.improved,
                },
            })

        # Save results
        if results:
            self._save_results(results, "crag_retrieval")

            # Print summary
            improved_count = sum(1 for r in results if r["crag"]["improved"])
            correction_count = sum(1 for r in results if r["crag"]["num_corrections"] > 0)

            print(f"\nCRAG Results:")
            print(f"  Queries with corrections: {correction_count}/{len(results)}")
            print(f"  Queries improved: {improved_count}/{len(results)}")

    def test_aggregate_metrics(self):
        """Test aggregate metrics across all queries."""
        all_results = []
        trackers = []
        crag_stats = {
            "initial_ratios": [],
            "final_ratios": [],
            "corrections_count": [],
            "improved_count": 0,
        }

        for query_data in self.golden_set["queries"]:
            tracker = CostLatencyTracker()

            with tracker.track_phase("retrieval"):
                result = self.retriever.retrieve(
                    query=query_data["query"],
                    top_k=5,
                    ticker=query_data["ticker"],
                )

            # Track grading latency separately
            # (In real impl, this would be inside retriever)

            retrieved_ids = [n.node.node_id for n in result.nodes]
            ground_truth_ids = query_data.get("ground_truth_ids", [])
            relevance_scores = [g.relevance_score for g in result.grades]

            all_results.append({
                "retrieved_ids": retrieved_ids,
                "ground_truth_ids": ground_truth_ids,
                "relevance_scores": relevance_scores,
            })
            trackers.append(tracker)

            # CRAG stats
            crag_stats["initial_ratios"].append(result.initial_relevant_ratio)
            crag_stats["final_ratios"].append(result.final_relevant_ratio)
            crag_stats["corrections_count"].append(result.num_correction_attempts)
            if result.improved:
                crag_stats["improved_count"] += 1

        # Filter to queries with ground truth
        results_with_gt = [r for r in all_results if r["ground_truth_ids"]]

        if results_with_gt:
            aggregated = self.evaluator.evaluate_batch(results_with_gt)
        else:
            aggregated = {"num_queries": 0}

        latency_stats = aggregate_trackers(trackers)

        # Calculate CRAG-specific aggregates
        import numpy as np
        crag_aggregated = {
            "initial_relevant_ratio": {
                "mean": float(np.mean(crag_stats["initial_ratios"])),
                "std": float(np.std(crag_stats["initial_ratios"])),
            },
            "final_relevant_ratio": {
                "mean": float(np.mean(crag_stats["final_ratios"])),
                "std": float(np.std(crag_stats["final_ratios"])),
            },
            "improvement": {
                "mean": float(np.mean([f - i for i, f in zip(crag_stats["initial_ratios"], crag_stats["final_ratios"])])),
            },
            "correction_rate": sum(1 for c in crag_stats["corrections_count"] if c > 0) / len(crag_stats["corrections_count"]),
            "improvement_rate": crag_stats["improved_count"] / len(crag_stats["initial_ratios"]),
            "avg_corrections": float(np.mean(crag_stats["corrections_count"])),
        }

        print("\n" + "=" * 60)
        print("CRAG AGGREGATE METRICS")
        print("=" * 60)
        print(f"Queries evaluated: {len(all_results)}")
        if aggregated.get("num_queries", 0) > 0:
            print(f"Precision@5: {aggregated['precision_at_k']['mean']:.3f} (±{aggregated['precision_at_k']['std']:.3f})")
            print(f"Recall@5: {aggregated['recall_at_k']['mean']:.3f} (±{aggregated['recall_at_k']['std']:.3f})")
            print(f"MRR: {aggregated['mrr']['mean']:.3f} (±{aggregated['mrr']['std']:.3f})")
            print(f"NDCG@5: {aggregated['ndcg_at_k']['mean']:.3f} (±{aggregated['ndcg_at_k']['std']:.3f})")
            print(f"Hit Rate: {aggregated['hit_rate']['mean']:.1%}")
        print("-" * 60)
        print("CRAG-Specific Metrics:")
        print(f"Initial Relevance: {crag_aggregated['initial_relevant_ratio']['mean']:.1%}")
        print(f"Final Relevance: {crag_aggregated['final_relevant_ratio']['mean']:.1%}")
        print(f"Relevance Improvement: {crag_aggregated['improvement']['mean']:+.1%}")
        print(f"Correction Rate: {crag_aggregated['correction_rate']:.1%}")
        print(f"Improvement Rate: {crag_aggregated['improvement_rate']:.1%}")
        print(f"Avg Corrections/Query: {crag_aggregated['avg_corrections']:.2f}")
        print("-" * 60)
        print(f"Latency p50: {latency_stats['latency']['p50_ms']:.0f}ms")
        print(f"Latency p95: {latency_stats['latency']['p95_ms']:.0f}ms")
        print("=" * 60)

        # Save aggregated results
        self._save_results({
            "retrieval": aggregated,
            "crag": crag_aggregated,
            "latency": latency_stats,
        }, "crag_aggregate")

    def _save_results(self, results, name: str):
        """Save results to JSON file."""
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{name}_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "retriever": "CorrectiveRetriever",
                "level": "corrective",
                "config": {
                    "relevance_threshold": 0.5,
                    "max_corrections": 2,
                },
                "results": results,
            }, f, indent=2)

        print(f"\nResults saved to: {output_path}")


class TestCRAGQuality:
    """Test CRAG answer quality using RAGAS."""

    @pytest.fixture(autouse=True)
    def setup(self, rag_index, golden_set, ragas_evaluator, settings):
        """Setup test fixtures."""
        self.index = rag_index
        self.golden_set = golden_set
        self.ragas_evaluator = ragas_evaluator
        self.retriever = CorrectiveRetriever(
            rag_index,
            relevance_threshold=0.5,
            max_corrections=2,
        )

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
        """Test RAGAS answer quality metrics with CRAG."""
        samples = []

        for query_data in self.golden_set["queries"][:5]:  # Limit for cost
            # Retrieve with CRAG
            result = self.retriever.retrieve(
                query=query_data["query"],
                top_k=5,
                ticker=query_data["ticker"],
            )

            contexts = [n.text for n in result.nodes]

            # Generate answer
            answer = self._generate_answer(query_data["query"], contexts)

            samples.append({
                "question": query_data["query"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": query_data["ground_truth_answer"],
                "crag_confidence": result.confidence,
                "crag_corrections": result.num_correction_attempts,
            })

        if samples:
            results = self.ragas_evaluator.evaluate_batch(samples)
            aggregated = self.ragas_evaluator.aggregate_results(results)

            print("\n" + "=" * 60)
            print("CRAG RAGAS METRICS")
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

            with open(output_dir / f"crag_ragas_{timestamp}.json", "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "retriever": "CorrectiveRetriever",
                    "level": "corrective",
                    "ragas": aggregated,
                    "num_samples": len(samples),
                }, f, indent=2)
