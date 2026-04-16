"""Scenario recorder — hooks into the LangGraph pipeline to capture replayable scenarios.

Records routing decisions, agent tool calls, agent outputs, and synthesis results
as JSON scenario files for later replay and benchmarking.

Usage:
    from evals.benchmark.recorder import ScenarioRecorder

    # Wrap a pipeline run
    recorder = ScenarioRecorder(ticker="AAPL", query="Is AAPL overvalued?")
    recorder.record_routing(active_agents=["financial_analyst", ...], mode="focused")
    recorder.record_agent_execution("financial_analyst", tool_calls=[...], output=agent_output)
    recorder.record_synthesis(recommendation)
    recorder.save("evals/benchmark/scenarios/")
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.structured_outputs import AgentOutput, InvestmentRecommendation
from src.agents.router import CATEGORY_AGENTS, ALL_AGENT_NAMES

logger = logging.getLogger(__name__)

# Schema version — bump when scenario format changes to auto-invalidate old scenarios
SCHEMA_VERSION = "1.0"


class ScenarioRecorder:
    """Records a full pipeline execution as a replayable JSON scenario."""

    def __init__(self, ticker: str, query: Optional[str] = None):
        self.ticker = ticker
        self.query = query
        self.recorded_at = datetime.utcnow().isoformat() + "Z"
        self.scenario_id = self._generate_id(ticker, query)

        self._routing: Optional[dict] = None
        self._agent_executions: list[dict] = []
        self._synthesis: Optional[dict] = None
        self._rag_context: dict[str, str] = {}

    def _generate_id(self, ticker: str, query: Optional[str]) -> str:
        """Generate deterministic scenario ID from ticker + query."""
        seed = f"{ticker}:{query or 'full'}"
        h = hashlib.sha256(seed.encode()).hexdigest()[:8]
        slug = "full" if not query else query.lower().replace(" ", "_")[:30]
        return f"sc_{ticker.lower()}_{slug}_{h}"

    def record_routing(
        self,
        active_agents: list[str],
        mode: str,
        categories: Optional[list[str]] = None,
    ) -> None:
        """Record the routing decision."""
        acceptable_sets = self._compute_acceptable_sets(categories)
        self._routing = {
            "mode": mode,
            "active_agents": active_agents,
            "categories": categories or [],
            "acceptable_agent_sets": acceptable_sets,
        }

    def _compute_acceptable_sets(
        self, categories: Optional[list[str]] = None
    ) -> list[list[str]]:
        """Compute acceptable agent sets for the given categories.

        Multiple agent combinations can be valid for a query. For example,
        "Is AAPL overvalued?" could validly route to financial+technical
        or financial+earnings+competitive.
        """
        if not categories or "full_analysis" in (categories or []):
            return [ALL_AGENT_NAMES]

        # The recorded set is always acceptable
        base_agents = set()
        for cat in categories:
            cat_agents = CATEGORY_AGENTS.get(cat)
            if cat_agents is not None:
                base_agents.update(cat_agents)

        acceptable = [sorted(base_agents)]

        # Adjacent categories are also acceptable
        # e.g., "financial" query could reasonably include "risk" agents
        adjacent = {
            "financial": ["risk"],
            "risk": ["financial"],
            "technical": ["financial"],
            "sentiment": ["insider"],
            "insider": ["sentiment"],
            "competitive": ["financial", "risk"],
        }

        for cat in categories:
            for adj_cat in adjacent.get(cat, []):
                expanded = set(base_agents)
                adj_agents = CATEGORY_AGENTS.get(adj_cat)
                if adj_agents:
                    expanded.update(adj_agents)
                expanded_sorted = sorted(expanded)
                if expanded_sorted not in acceptable:
                    acceptable.append(expanded_sorted)

        return acceptable

    def record_rag_context(self, agent_name: str, context: str) -> None:
        """Record RAG context provided to an agent."""
        self._rag_context[agent_name] = context

    def record_agent_execution(
        self,
        agent_name: str,
        tool_calls: list[dict],
        output: AgentOutput,
        llm_calls: Optional[list[dict]] = None,
    ) -> None:
        """Record a single agent's execution.

        Args:
            agent_name: Name of the agent.
            tool_calls: List of {"tool": str, "input": dict, "output": dict}.
            output: The agent's final AgentOutput.
            llm_calls: Optional list of LLM call records for replay caching.
        """
        self._agent_executions.append({
            "agent_name": agent_name,
            "tool_calls": tool_calls,
            "output": {
                "score": output.score,
                "confidence": output.confidence,
                "summary": output.summary,
                "metrics": output.metrics,
                "strengths": output.strengths,
                "weaknesses": output.weaknesses,
                "sources": output.sources,
            },
            "llm_calls": llm_calls or [],
        })

    def record_synthesis(self, recommendation: InvestmentRecommendation) -> None:
        """Record the synthesis result."""
        self._synthesis = {
            "overall_score": recommendation.overall_score,
            "recommendation": recommendation.recommendation,
            "confidence": recommendation.confidence,
            "financial_score": recommendation.financial_score,
            "technical_score": recommendation.technical_score,
            "sentiment_score": recommendation.sentiment_score,
            "risk_score": recommendation.risk_score,
            "agent_scores": recommendation.agent_scores,
        }

    def to_dict(self) -> dict:
        """Serialize the scenario to a dict."""
        return {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "ticker": self.ticker,
            "query": self.query,
            "recorded_at": self.recorded_at,
            "routing": self._routing,
            "rag_context": self._rag_context,
            "agent_executions": self._agent_executions,
            "synthesis": self._synthesis,
        }

    def save(self, output_dir: str) -> Path:
        """Save the scenario to a JSON file.

        Returns the path to the saved file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"{self.scenario_id}.json"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        logger.info(f"Scenario saved: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str) -> dict:
        """Load a scenario from a JSON file."""
        with open(filepath) as f:
            scenario = json.load(f)

        if scenario.get("schema_version") != SCHEMA_VERSION:
            logger.warning(
                f"Scenario {filepath} has schema version "
                f"{scenario.get('schema_version')}, expected {SCHEMA_VERSION}. "
                f"Results may be unreliable."
            )

        return scenario
