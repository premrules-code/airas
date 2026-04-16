"""Cost and latency tracking for RAG evaluation.

Tracks:
- Token usage and costs for different models
- Latency per phase (retrieval, grading, correction, etc.)
- Provides context managers for easy timing
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Track costs per component."""

    # Embedding costs (OpenAI text-embedding-3-small)
    embedding_tokens: int = 0
    embedding_cost: float = 0.0

    # Grading costs (Claude Haiku or Sonnet)
    grading_input_tokens: int = 0
    grading_output_tokens: int = 0
    grading_cost: float = 0.0

    # Correction costs (query transformation, re-retrieval)
    correction_input_tokens: int = 0
    correction_output_tokens: int = 0
    correction_cost: float = 0.0

    # Answer generation costs
    generation_input_tokens: int = 0
    generation_output_tokens: int = 0
    generation_cost: float = 0.0

    # Web search costs (if using Tavily or similar)
    web_searches: int = 0
    web_search_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        """Total cost across all components."""
        return (
            self.embedding_cost
            + self.grading_cost
            + self.correction_cost
            + self.generation_cost
            + self.web_search_cost
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return (
            self.embedding_tokens
            + self.grading_input_tokens
            + self.grading_output_tokens
            + self.correction_input_tokens
            + self.correction_output_tokens
            + self.generation_input_tokens
            + self.generation_output_tokens
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "embedding": {
                "tokens": self.embedding_tokens,
                "cost": self.embedding_cost,
            },
            "grading": {
                "input_tokens": self.grading_input_tokens,
                "output_tokens": self.grading_output_tokens,
                "cost": self.grading_cost,
            },
            "correction": {
                "input_tokens": self.correction_input_tokens,
                "output_tokens": self.correction_output_tokens,
                "cost": self.correction_cost,
            },
            "generation": {
                "input_tokens": self.generation_input_tokens,
                "output_tokens": self.generation_output_tokens,
                "cost": self.generation_cost,
            },
            "web_search": {
                "searches": self.web_searches,
                "cost": self.web_search_cost,
            },
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LatencyBreakdown:
    """Track latency per phase in milliseconds."""

    retrieval_ms: float = 0.0
    grading_ms: float = 0.0
    correction_ms: float = 0.0
    generation_ms: float = 0.0
    web_search_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "retrieval_ms": self.retrieval_ms,
            "grading_ms": self.grading_ms,
            "correction_ms": self.correction_ms,
            "generation_ms": self.generation_ms,
            "web_search_ms": self.web_search_ms,
            "total_ms": self.total_ms,
        }


class CostLatencyTracker:
    """Track costs and latency during RAG execution.

    Usage:
        tracker = CostLatencyTracker()

        with tracker.track_phase("retrieval"):
            results = retriever.retrieve(query)

        tracker.record_grading(input_tokens=500, output_tokens=50)

        print(tracker.to_dict())
    """

    # Pricing constants (as of Feb 2024)
    # OpenAI
    EMBEDDING_COST_PER_1K = 0.00002  # text-embedding-3-small

    # Anthropic Claude
    HAIKU_INPUT_PER_1M = 0.25
    HAIKU_OUTPUT_PER_1M = 1.25
    SONNET_INPUT_PER_1M = 3.00
    SONNET_OUTPUT_PER_1M = 15.00
    OPUS_INPUT_PER_1M = 15.00
    OPUS_OUTPUT_PER_1M = 75.00

    # Web search
    TAVILY_PER_SEARCH = 0.001

    def __init__(self):
        self.cost = CostBreakdown()
        self.latency = LatencyBreakdown()
        self._phase_times: dict[str, float] = {}

    @contextmanager
    def track_phase(self, phase: str):
        """Context manager to track latency of a phase.

        Args:
            phase: One of 'retrieval', 'grading', 'correction', 'generation', 'web_search'
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._phase_times[phase] = elapsed_ms

            # Update the appropriate latency field
            if hasattr(self.latency, f"{phase}_ms"):
                setattr(self.latency, f"{phase}_ms", elapsed_ms)
                self.latency.total_ms += elapsed_ms

    def record_embedding(self, tokens: int):
        """Record embedding token usage."""
        self.cost.embedding_tokens += tokens
        self.cost.embedding_cost += tokens * self.EMBEDDING_COST_PER_1K / 1000

    def record_grading(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "haiku",
    ):
        """Record grading LLM usage."""
        self.cost.grading_input_tokens += input_tokens
        self.cost.grading_output_tokens += output_tokens

        if model == "haiku":
            self.cost.grading_cost += (
                input_tokens * self.HAIKU_INPUT_PER_1M / 1_000_000
                + output_tokens * self.HAIKU_OUTPUT_PER_1M / 1_000_000
            )
        elif model == "sonnet":
            self.cost.grading_cost += (
                input_tokens * self.SONNET_INPUT_PER_1M / 1_000_000
                + output_tokens * self.SONNET_OUTPUT_PER_1M / 1_000_000
            )
        elif model == "opus":
            self.cost.grading_cost += (
                input_tokens * self.OPUS_INPUT_PER_1M / 1_000_000
                + output_tokens * self.OPUS_OUTPUT_PER_1M / 1_000_000
            )

    def record_correction(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "haiku",
    ):
        """Record correction/transformation LLM usage."""
        self.cost.correction_input_tokens += input_tokens
        self.cost.correction_output_tokens += output_tokens

        if model == "haiku":
            self.cost.correction_cost += (
                input_tokens * self.HAIKU_INPUT_PER_1M / 1_000_000
                + output_tokens * self.HAIKU_OUTPUT_PER_1M / 1_000_000
            )
        elif model == "sonnet":
            self.cost.correction_cost += (
                input_tokens * self.SONNET_INPUT_PER_1M / 1_000_000
                + output_tokens * self.SONNET_OUTPUT_PER_1M / 1_000_000
            )

    def record_generation(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "sonnet",
    ):
        """Record answer generation LLM usage."""
        self.cost.generation_input_tokens += input_tokens
        self.cost.generation_output_tokens += output_tokens

        if model == "haiku":
            self.cost.generation_cost += (
                input_tokens * self.HAIKU_INPUT_PER_1M / 1_000_000
                + output_tokens * self.HAIKU_OUTPUT_PER_1M / 1_000_000
            )
        elif model == "sonnet":
            self.cost.generation_cost += (
                input_tokens * self.SONNET_INPUT_PER_1M / 1_000_000
                + output_tokens * self.SONNET_OUTPUT_PER_1M / 1_000_000
            )
        elif model == "opus":
            self.cost.generation_cost += (
                input_tokens * self.OPUS_INPUT_PER_1M / 1_000_000
                + output_tokens * self.OPUS_OUTPUT_PER_1M / 1_000_000
            )

    def record_web_search(self, num_searches: int = 1):
        """Record web search usage."""
        self.cost.web_searches += num_searches
        self.cost.web_search_cost += num_searches * self.TAVILY_PER_SEARCH

    def to_dict(self) -> dict:
        """Export all metrics as dictionary."""
        return {
            "cost": self.cost.to_dict(),
            "latency": self.latency.to_dict(),
        }

    def reset(self):
        """Reset all trackers."""
        self.cost = CostBreakdown()
        self.latency = LatencyBreakdown()
        self._phase_times = {}


def aggregate_trackers(trackers: list[CostLatencyTracker]) -> dict:
    """Aggregate multiple trackers into summary statistics.

    Args:
        trackers: List of CostLatencyTracker instances

    Returns:
        Dictionary with mean, std, min, max for cost and latency
    """
    import numpy as np

    if not trackers:
        return {}

    costs = [t.cost.total_cost for t in trackers]
    latencies = [t.latency.total_ms for t in trackers]

    return {
        "cost": {
            "mean": float(np.mean(costs)),
            "std": float(np.std(costs)),
            "min": float(np.min(costs)),
            "max": float(np.max(costs)),
            "total": float(np.sum(costs)),
        },
        "latency": {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
        },
        "num_samples": len(trackers),
    }
