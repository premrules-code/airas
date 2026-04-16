"""Retrieval quality metrics for RAG evaluation.

Metrics implemented:
- Precision@K: Fraction of retrieved documents that are relevant
- Recall@K: Fraction of relevant documents that were retrieved
- F1@K: Harmonic mean of precision and recall
- MRR (Mean Reciprocal Rank): Position of first relevant document
- NDCG@K (Normalized Discounted Cumulative Gain): Ranking quality
- MAP@K (Mean Average Precision): Average precision at each relevant doc
- Hit Rate: Whether at least one relevant document was retrieved
"""

import logging
from dataclasses import dataclass
from typing import List, Set, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Container for all retrieval metrics."""

    # Core metrics
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float

    # Ranking metrics
    mrr: float
    ndcg_at_k: float
    map_at_k: float

    # Hit rate
    hit_rate: float

    # Optional: average relevance score from grader
    avg_relevance_score: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "f1_at_k": self.f1_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "map_at_k": self.map_at_k,
            "hit_rate": self.hit_rate,
            "avg_relevance_score": self.avg_relevance_score,
        }


class RetrievalEvaluator:
    """Compute retrieval metrics against ground truth."""

    def evaluate(
        self,
        retrieved_ids: List[str],
        ground_truth_ids: List[str],
        relevance_scores: Optional[List[float]] = None,
    ) -> RetrievalMetrics:
        """Evaluate retrieval quality.

        Args:
            retrieved_ids: List of retrieved document/chunk IDs in rank order
            ground_truth_ids: List of IDs that are actually relevant
            relevance_scores: Optional relevance scores (0-1) for each retrieved doc

        Returns:
            RetrievalMetrics with all computed metrics
        """
        if not retrieved_ids:
            return RetrievalMetrics(
                precision_at_k=0.0,
                recall_at_k=0.0,
                f1_at_k=0.0,
                mrr=0.0,
                ndcg_at_k=0.0,
                map_at_k=0.0,
                hit_rate=0.0,
                avg_relevance_score=0.0,
            )

        retrieved_set = set(retrieved_ids)
        relevant_set = set(ground_truth_ids)

        # Precision@K
        relevant_retrieved = retrieved_set & relevant_set
        precision = len(relevant_retrieved) / len(retrieved_ids)

        # Recall@K
        recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0.0

        # F1@K
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # MRR (Mean Reciprocal Rank)
        mrr = self._compute_mrr(retrieved_ids, relevant_set)

        # NDCG@K
        ndcg = self._compute_ndcg(retrieved_ids, relevant_set, k=len(retrieved_ids))

        # MAP@K
        map_score = self._compute_map(retrieved_ids, relevant_set)

        # Hit rate
        hit_rate = 1.0 if relevant_retrieved else 0.0

        # Average relevance score (if provided)
        avg_relevance = np.mean(relevance_scores) if relevance_scores else None

        return RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            f1_at_k=f1,
            mrr=mrr,
            ndcg_at_k=ndcg,
            map_at_k=map_score,
            hit_rate=hit_rate,
            avg_relevance_score=avg_relevance,
        )

    def _compute_mrr(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Compute Mean Reciprocal Rank.

        MRR = 1 / position of first relevant document (1-indexed)
        """
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    def _compute_ndcg(
        self, retrieved: List[str], relevant: Set[str], k: int
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain.

        DCG = sum(rel_i / log2(i + 2)) for i in 0..k-1
        NDCG = DCG / IDCG (ideal DCG)
        """
        # DCG: Discounted Cumulative Gain
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            rel = 1.0 if doc_id in relevant else 0.0
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0

        # IDCG: Ideal DCG (all relevant docs at top)
        num_relevant = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))

        return dcg / idcg if idcg > 0 else 0.0

    def _compute_map(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Compute Mean Average Precision.

        AP = (1/|relevant|) * sum(P(k) * rel(k)) for k in 1..n
        where P(k) is precision at position k
        """
        if not relevant:
            return 0.0

        precisions = []
        relevant_count = 0

        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precisions.append(precision_at_i)

        return np.mean(precisions) if precisions else 0.0

    def evaluate_batch(
        self, results: List[dict]
    ) -> dict:
        """Evaluate a batch of queries and aggregate metrics.

        Args:
            results: List of dicts with 'retrieved_ids' and 'ground_truth_ids'

        Returns:
            Aggregated metrics with mean and std
        """
        all_metrics = []

        for r in results:
            metrics = self.evaluate(
                retrieved_ids=r["retrieved_ids"],
                ground_truth_ids=r["ground_truth_ids"],
                relevance_scores=r.get("relevance_scores"),
            )
            all_metrics.append(metrics)

        # Aggregate
        metric_names = [
            "precision_at_k",
            "recall_at_k",
            "f1_at_k",
            "mrr",
            "ndcg_at_k",
            "map_at_k",
            "hit_rate",
        ]

        aggregated = {}
        for name in metric_names:
            values = [getattr(m, name) for m in all_metrics]
            aggregated[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        aggregated["num_queries"] = len(results)

        return aggregated
