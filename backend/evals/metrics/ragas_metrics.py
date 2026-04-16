"""RAGAS metrics wrapper for answer quality evaluation.

Integrates RAGAS with Langfuse for score storage.

Metrics:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are retrieved docs actually useful?
- Context Recall: Did we retrieve all needed info?
- Answer Correctness: Is the answer factually correct vs ground truth?
"""

import logging
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RAGASResult:
    """Container for RAGAS evaluation results."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_correctness": self.answer_correctness,
        }


class RAGASEvaluator:
    """Evaluate RAG answer quality using RAGAS framework.

    Usage:
        evaluator = RAGASEvaluator()

        result = evaluator.evaluate_single(
            question="What was Apple's revenue?",
            answer="Apple's revenue was $383.3 billion",
            contexts=["Apple reported net sales of $383,285 million..."],
            ground_truth="Apple reported $383.3 billion in revenue for FY2023"
        )

        print(result.faithfulness)  # 0.95
    """

    def __init__(self, langfuse_client=None):
        """Initialize RAGAS evaluator.

        Args:
            langfuse_client: Optional Langfuse client for score storage
        """
        self.langfuse = langfuse_client
        self._metrics = None
        self._initialized = False

    def _init_metrics(self):
        """Lazy initialization of RAGAS metrics."""
        if self._initialized:
            return

        try:
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            )

            self._metrics = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "answer_correctness": answer_correctness,
            }
            self._initialized = True
            logger.info("RAGAS metrics initialized")

        except ImportError as e:
            logger.error(f"Failed to import RAGAS: {e}")
            logger.error("Install with: pip install ragas")
            raise

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
        trace_id: Optional[str] = None,
    ) -> RAGASResult:
        """Evaluate a single RAG response.

        Args:
            question: The input question
            answer: The generated answer
            contexts: List of retrieved context strings
            ground_truth: The expected/correct answer
            trace_id: Optional Langfuse trace ID for score storage

        Returns:
            RAGASResult with all metric scores
        """
        self._init_metrics()

        try:
            from datasets import Dataset
            from ragas import evaluate

            # Prepare data in RAGAS format
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            }
            dataset = Dataset.from_dict(data)

            # Run evaluation
            result = evaluate(
                dataset,
                metrics=list(self._metrics.values()),
            )

            # Extract scores
            df = result.to_pandas()
            row = df.iloc[0]

            scores = RAGASResult(
                faithfulness=float(row.get("faithfulness", 0)),
                answer_relevancy=float(row.get("answer_relevancy", 0)),
                context_precision=float(row.get("context_precision", 0)),
                context_recall=float(row.get("context_recall", 0)),
                answer_correctness=float(row.get("answer_correctness", 0)),
            )

            # Store in Langfuse if available
            if self.langfuse and trace_id:
                self._store_scores(trace_id, scores)

            return scores

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            # Return zeros on failure
            return RAGASResult(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
                answer_correctness=0.0,
            )

    def evaluate_batch(
        self,
        samples: List[dict],
    ) -> List[RAGASResult]:
        """Evaluate a batch of RAG responses.

        Args:
            samples: List of dicts with keys:
                - question: str
                - answer: str
                - contexts: List[str]
                - ground_truth: str
                - trace_id: Optional[str]

        Returns:
            List of RAGASResult for each sample
        """
        self._init_metrics()

        try:
            from datasets import Dataset
            from ragas import evaluate

            # Prepare batch data
            data = {
                "question": [s["question"] for s in samples],
                "answer": [s["answer"] for s in samples],
                "contexts": [s["contexts"] for s in samples],
                "ground_truth": [s["ground_truth"] for s in samples],
            }
            dataset = Dataset.from_dict(data)

            # Run batch evaluation
            result = evaluate(
                dataset,
                metrics=list(self._metrics.values()),
            )

            df = result.to_pandas()

            results = []
            for i, row in df.iterrows():
                scores = RAGASResult(
                    faithfulness=float(row.get("faithfulness", 0)),
                    answer_relevancy=float(row.get("answer_relevancy", 0)),
                    context_precision=float(row.get("context_precision", 0)),
                    context_recall=float(row.get("context_recall", 0)),
                    answer_correctness=float(row.get("answer_correctness", 0)),
                )
                results.append(scores)

                # Store in Langfuse
                trace_id = samples[i].get("trace_id")
                if self.langfuse and trace_id:
                    self._store_scores(trace_id, scores)

            return results

        except Exception as e:
            logger.error(f"RAGAS batch evaluation failed: {e}")
            return [
                RAGASResult(
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_precision=0.0,
                    context_recall=0.0,
                    answer_correctness=0.0,
                )
                for _ in samples
            ]

    def _store_scores(self, trace_id: str, scores: RAGASResult):
        """Store scores in Langfuse."""
        if not self.langfuse:
            return

        try:
            for metric_name, value in scores.to_dict().items():
                self.langfuse.score(
                    trace_id=trace_id,
                    name=f"ragas_{metric_name}",
                    value=value,
                )
            logger.debug(f"Stored RAGAS scores for trace {trace_id}")
        except Exception as e:
            logger.warning(f"Failed to store scores in Langfuse: {e}")

    @staticmethod
    def aggregate_results(results: List[RAGASResult]) -> dict:
        """Aggregate multiple RAGAS results into summary statistics.

        Args:
            results: List of RAGASResult instances

        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        import numpy as np

        if not results:
            return {}

        metrics = ["faithfulness", "answer_relevancy", "context_precision",
                   "context_recall", "answer_correctness"]

        aggregated = {}
        for metric in metrics:
            values = [getattr(r, metric) for r in results]
            aggregated[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        aggregated["num_samples"] = len(results)
        return aggregated
