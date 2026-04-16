"""Galileo-based evaluation metrics for AIRAS.

Provides evaluation functions using Galileo AI for:
- Groundedness (hallucination detection)
- Context relevance
- PII detection
- Answer quality
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
import statistics

from src.utils.galileo_setup import is_galileo_available
from src.guardrails.galileo_guardrails import (
    check_hallucination,
    check_pii,
    check_context_relevance,
)

logger = logging.getLogger(__name__)


@dataclass
class GroundednessResult:
    """Result of groundedness evaluation."""
    query_id: str
    query: str
    answer: str
    context: str
    groundedness_score: float
    is_grounded: bool
    flagged_claims: list[str] = field(default_factory=list)
    expected_groundedness: float = 0.8
    passed: bool = False

    def __post_init__(self):
        self.passed = self.groundedness_score >= self.expected_groundedness


@dataclass
class ContextRelevanceResult:
    """Result of context relevance evaluation."""
    query_id: str
    query: str
    chunks: list[str]
    relevance_scores: list[float]
    avg_relevance: float
    relevant_chunk_count: int
    total_chunk_count: int


@dataclass
class PIIResult:
    """Result of PII detection evaluation."""
    text_id: str
    text: str
    has_pii: bool
    pii_types: list[str]
    expected_pii: bool
    passed: bool = False

    def __post_init__(self):
        self.passed = self.has_pii == self.expected_pii


@dataclass
class GalileoEvalSummary:
    """Summary of Galileo evaluation results."""
    total_queries: int = 0
    groundedness_mean: float = 0.0
    groundedness_std: float = 0.0
    groundedness_pass_rate: float = 0.0
    context_relevance_mean: float = 0.0
    pii_detection_accuracy: float = 0.0
    hallucination_detection_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "groundedness": {
                "mean": round(self.groundedness_mean, 3),
                "std": round(self.groundedness_std, 3),
                "pass_rate": round(self.groundedness_pass_rate, 3),
            },
            "context_relevance": {
                "mean": round(self.context_relevance_mean, 3),
            },
            "pii_detection": {
                "accuracy": round(self.pii_detection_accuracy, 3),
            },
            "hallucination_detection": {
                "rate": round(self.hallucination_detection_rate, 3),
            },
        }


class GalileoEvaluator:
    """Evaluator using Galileo AI metrics."""

    def __init__(self, groundedness_threshold: float = 0.7):
        self.groundedness_threshold = groundedness_threshold
        self._galileo_available = is_galileo_available()

        if not self._galileo_available:
            logger.warning(
                "Galileo not available. Evaluations will use fallback scoring."
            )

    def evaluate_groundedness(
        self,
        query_id: str,
        query: str,
        answer: str,
        context: str,
        expected_groundedness: float = 0.8,
    ) -> GroundednessResult:
        """
        Evaluate if an answer is grounded in the provided context.

        Args:
            query_id: Unique identifier for the query
            query: The original question
            answer: The generated answer
            context: The source context/documents
            expected_groundedness: Minimum expected score to pass

        Returns:
            GroundednessResult with score and pass/fail status
        """
        result = check_hallucination(answer, [context], threshold=self.groundedness_threshold)

        return GroundednessResult(
            query_id=query_id,
            query=query,
            answer=answer,
            context=context[:500] + "..." if len(context) > 500 else context,
            groundedness_score=result["score"],
            is_grounded=result["is_grounded"],
            flagged_claims=result["flagged_claims"],
            expected_groundedness=expected_groundedness,
        )

    def evaluate_context_relevance(
        self,
        query_id: str,
        query: str,
        chunks: list[str],
    ) -> ContextRelevanceResult:
        """
        Evaluate relevance of retrieved chunks to the query.

        Args:
            query_id: Unique identifier
            query: The search query
            chunks: Retrieved document chunks

        Returns:
            ContextRelevanceResult with per-chunk scores
        """
        result = check_context_relevance(query, chunks)

        return ContextRelevanceResult(
            query_id=query_id,
            query=query,
            chunks=chunks,
            relevance_scores=result["scores"],
            avg_relevance=result["avg_score"],
            relevant_chunk_count=len(result["relevant_indices"]),
            total_chunk_count=len(chunks),
        )

    def evaluate_pii(
        self,
        text_id: str,
        text: str,
        expected_has_pii: bool,
    ) -> PIIResult:
        """
        Evaluate PII detection accuracy.

        Args:
            text_id: Unique identifier
            text: Text to check for PII
            expected_has_pii: Whether PII is expected in the text

        Returns:
            PIIResult with detection status
        """
        result = check_pii(text)

        return PIIResult(
            text_id=text_id,
            text=text[:200] + "..." if len(text) > 200 else text,
            has_pii=result["has_pii"],
            pii_types=result["pii_types"],
            expected_pii=expected_has_pii,
        )

    def evaluate_hallucination_detection(
        self,
        bad_answer: str,
        good_answer: str,
        context: str,
    ) -> dict:
        """
        Test hallucination detection by comparing bad vs good answers.

        Args:
            bad_answer: Answer with hallucinated content
            good_answer: Answer grounded in context
            context: Source context

        Returns:
            Dict with scores for both answers and detection success
        """
        bad_result = check_hallucination(bad_answer, [context])
        good_result = check_hallucination(good_answer, [context])

        # Detection is successful if bad answer has lower groundedness
        detection_success = bad_result["score"] < good_result["score"]

        return {
            "bad_answer_score": bad_result["score"],
            "good_answer_score": good_result["score"],
            "bad_flagged_claims": bad_result["flagged_claims"],
            "detection_success": detection_success,
            "score_difference": good_result["score"] - bad_result["score"],
        }

    def evaluate_batch(
        self,
        eval_dataset: dict,
    ) -> dict:
        """
        Run full evaluation on a Galileo eval dataset.

        Args:
            eval_dataset: Dataset loaded from galileo_eval_set.json

        Returns:
            Dict with all evaluation results and summary
        """
        results = {
            "groundedness": [],
            "context_relevance": [],
            "pii": [],
            "hallucination_detection": [],
        }

        # Evaluate queries for groundedness
        for query_data in eval_dataset.get("queries", []):
            if "context" in query_data and "expected_answer" in query_data:
                g_result = self.evaluate_groundedness(
                    query_id=query_data["id"],
                    query=query_data["query"],
                    answer=query_data["expected_answer"],
                    context=query_data["context"],
                    expected_groundedness=query_data.get("eval_criteria", {}).get(
                        "expected_groundedness", 0.8
                    ),
                )
                results["groundedness"].append(g_result)

        # Evaluate hallucination tests
        for hal_test in eval_dataset.get("hallucination_tests", []):
            hal_result = self.evaluate_hallucination_detection(
                bad_answer=hal_test["bad_answer"],
                good_answer=hal_test["good_answer"],
                context=hal_test["context"],
            )
            hal_result["test_id"] = hal_test["id"]
            hal_result["description"] = hal_test["description"]
            results["hallucination_detection"].append(hal_result)

        # Evaluate PII tests
        for pii_test in eval_dataset.get("pii_tests", []):
            # Test text with PII
            pii_result = self.evaluate_pii(
                text_id=pii_test["id"] + "_with_pii",
                text=pii_test["text_with_pii"],
                expected_has_pii=True,
            )
            results["pii"].append(pii_result)

            # Test text without PII
            no_pii_result = self.evaluate_pii(
                text_id=pii_test["id"] + "_without_pii",
                text=pii_test["text_without_pii"],
                expected_has_pii=False,
            )
            results["pii"].append(no_pii_result)

        # Calculate summary
        summary = self._calculate_summary(results)

        return {
            "results": results,
            "summary": summary.to_dict(),
        }

    def _calculate_summary(self, results: dict) -> GalileoEvalSummary:
        """Calculate summary statistics from results."""
        summary = GalileoEvalSummary()

        # Groundedness stats
        g_scores = [r.groundedness_score for r in results["groundedness"]]
        g_passed = [r for r in results["groundedness"] if r.passed]

        if g_scores:
            summary.total_queries = len(g_scores)
            summary.groundedness_mean = statistics.mean(g_scores)
            summary.groundedness_std = statistics.stdev(g_scores) if len(g_scores) > 1 else 0.0
            summary.groundedness_pass_rate = len(g_passed) / len(g_scores)

        # Context relevance stats
        cr_scores = [r.avg_relevance for r in results["context_relevance"]]
        if cr_scores:
            summary.context_relevance_mean = statistics.mean(cr_scores)

        # PII detection accuracy
        pii_results = results["pii"]
        if pii_results:
            pii_correct = sum(1 for r in pii_results if r.passed)
            summary.pii_detection_accuracy = pii_correct / len(pii_results)

        # Hallucination detection rate
        hal_results = results["hallucination_detection"]
        if hal_results:
            hal_detected = sum(1 for r in hal_results if r["detection_success"])
            summary.hallucination_detection_rate = hal_detected / len(hal_results)

        return summary


def run_galileo_evals(eval_dataset_path: str) -> dict:
    """
    Convenience function to run Galileo evals from a JSON file.

    Args:
        eval_dataset_path: Path to galileo_eval_set.json

    Returns:
        Evaluation results dict
    """
    import json

    with open(eval_dataset_path) as f:
        dataset = json.load(f)

    evaluator = GalileoEvaluator()
    return evaluator.evaluate_batch(dataset)
