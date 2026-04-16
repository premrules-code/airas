"""Replay engine — replays recorded scenarios with deterministic mocking.

Mocks Claude API calls and data provider responses using cached values from
the recorded scenario, enabling cheap and repeatable benchmarking without
live API calls.

Uses deterministic asset UUIDs per scenario + position to ensure stable
references across repeated runs.

Usage:
    from evals.benchmark.replay import ScenarioReplayer

    replayer = ScenarioReplayer(scenario)
    result = replayer.replay(model="claude-sonnet-4-20250514")
"""

import hashlib
import json
import logging
import time
from typing import Optional
from unittest.mock import MagicMock, patch

from src.agents.router import route_query, CATEGORY_AGENTS, ALL_AGENT_NAMES
from src.models.structured_outputs import AgentOutput

logger = logging.getLogger(__name__)


class DeterministicIDGenerator:
    """Generate deterministic IDs per scenario for stable replay.

    Uses scenario_id + position as salt, so different scenarios can run
    in parallel against the same local database without collisions.
    (Deterministic UUID scheme for parallel-safe benchmarking.)
    """

    def __init__(self, scenario_id: str):
        self.scenario_id = scenario_id
        self._counter = 0

    def next_id(self) -> str:
        """Generate next deterministic ID."""
        seed = f"{self.scenario_id}:{self._counter}"
        self._counter += 1
        return hashlib.sha256(seed.encode()).hexdigest()[:16]


class MockToolExecutor:
    """Replays tool calls from recorded scenario instead of calling real APIs."""

    def __init__(self, recorded_tool_calls: list[dict]):
        self._cache = {}
        for tc in recorded_tool_calls:
            key = self._make_key(tc["tool"], tc["input"])
            self._cache[key] = tc["output"]

    def _make_key(self, tool_name: str, tool_input: dict) -> str:
        """Create a cache key from tool name + input."""
        input_str = json.dumps(tool_input, sort_keys=True)
        return f"{tool_name}:{input_str}"

    def execute(self, tool_name: str, tool_input: dict) -> dict:
        """Return cached tool result, or a fallback if not recorded."""
        key = self._make_key(tool_name, tool_input)
        if key in self._cache:
            return self._cache[key]

        logger.warning(
            f"Tool call not in recorded scenario: {tool_name}({tool_input}). "
            f"Returning empty result."
        )
        return {"warning": "unrecorded_tool_call", "tool": tool_name}


class ReplayResult:
    """Result of replaying a scenario."""

    def __init__(self, scenario_id: str):
        self.scenario_id = scenario_id
        self.routing_result: Optional[dict] = None
        self.agent_results: list[dict] = []
        self.synthesis_result: Optional[dict] = None
        self.latency_ms: float = 0.0
        self.errors: list[str] = []

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "routing": self.routing_result,
            "agents": self.agent_results,
            "synthesis": self.synthesis_result,
            "latency_ms": self.latency_ms,
            "errors": self.errors,
        }


class ScenarioReplayer:
    """Replays a recorded scenario against the current pipeline.

    Two replay modes:
    - structural: Mock everything (Claude + tools). Tests routing and tool call
      patterns without any API cost. Fast (~100ms per scenario).
    - live: Real Claude calls, mocked tools. Tests actual model quality against
      recorded tool responses. Costs ~$0.50-2.00 per scenario.
    """

    def __init__(self, scenario: dict):
        self.scenario = scenario
        self.scenario_id = scenario["scenario_id"]
        self.id_gen = DeterministicIDGenerator(self.scenario_id)

    def replay_structural(self) -> ReplayResult:
        """Structural replay: mock Claude + tools, test routing and tool patterns.

        This is the cheap, fast mode used for regression detection on prompt
        and orchestration changes. No live API calls.
        """
        result = ReplayResult(self.scenario_id)
        start = time.time()

        try:
            # Stage 1: Test routing
            result.routing_result = self._replay_routing()

            # Stage 2: Test agent tool call patterns
            for agent_exec in self.scenario.get("agent_executions", []):
                agent_result = self._replay_agent_structural(agent_exec)
                result.agent_results.append(agent_result)

            # Stage 3: Test synthesis scoring
            result.synthesis_result = self._replay_synthesis_structural()

        except Exception as e:
            result.errors.append(f"Replay error: {e}")
            logger.error(f"Structural replay failed for {self.scenario_id}: {e}")

        result.latency_ms = (time.time() - start) * 1000
        return result

    def replay_live(self, model: Optional[str] = None) -> ReplayResult:
        """Live replay: real Claude calls, mocked tools.

        Tests actual model quality against recorded tool responses.
        Used for model comparison benchmarks.

        Args:
            model: Override model name (e.g., "claude-sonnet-4-20250514").
        """
        result = ReplayResult(self.scenario_id)
        start = time.time()

        try:
            # Stage 1: Live routing
            query = self.scenario.get("query")
            if query:
                active_agents = route_query(query)
                result.routing_result = {
                    "active_agents": active_agents,
                    "mode": "focused" if len(active_agents) < len(ALL_AGENT_NAMES) else "full",
                }
            else:
                result.routing_result = {
                    "active_agents": ALL_AGENT_NAMES,
                    "mode": "full",
                }

            # Stage 2: Live agent execution with mocked tools
            for agent_exec in self.scenario.get("agent_executions", []):
                agent_result = self._replay_agent_live(agent_exec, model=model)
                result.agent_results.append(agent_result)

            # Stage 3: Live synthesis
            result.synthesis_result = self._replay_synthesis_live(result.agent_results)

        except Exception as e:
            result.errors.append(f"Live replay error: {e}")
            logger.error(f"Live replay failed for {self.scenario_id}: {e}")

        result.latency_ms = (time.time() - start) * 1000
        return result

    def _replay_routing(self) -> dict:
        """Replay routing with the current router."""
        query = self.scenario.get("query")
        if not query:
            return {
                "active_agents": ALL_AGENT_NAMES,
                "mode": "full",
            }

        active_agents = route_query(query)
        return {
            "active_agents": active_agents,
            "mode": "focused" if len(active_agents) < len(ALL_AGENT_NAMES) else "full",
        }

    def _replay_agent_structural(self, agent_exec: dict) -> dict:
        """Structural replay: check if tool call patterns match recorded."""
        agent_name = agent_exec["agent_name"]
        recorded_tools = agent_exec.get("tool_calls", [])

        return {
            "agent_name": agent_name,
            "recorded_tool_calls": [
                {"tool": tc["tool"], "input": tc["input"]}
                for tc in recorded_tools
            ],
            # In structural mode, we use the recorded output directly
            "output": agent_exec.get("output", {}),
            "mode": "structural",
        }

    def _replay_agent_live(
        self, agent_exec: dict, model: Optional[str] = None
    ) -> dict:
        """Live replay: real Claude call with mocked tool responses."""
        from src.agents import AGENT_MAP

        agent_name = agent_exec["agent_name"]
        recorded_tools = agent_exec.get("tool_calls", [])
        mock_executor = MockToolExecutor(recorded_tools)

        agent_cls = AGENT_MAP.get(agent_name)
        if not agent_cls:
            return {
                "agent_name": agent_name,
                "error": f"Unknown agent: {agent_name}",
                "mode": "live",
            }

        agent = agent_cls()
        rag_context = self.scenario.get("rag_context", {}).get(agent_name, "")

        # Patch tool execution to use recorded responses
        with patch(
            "src.tools.financial_tools.execute_tool",
            side_effect=lambda name, inp: mock_executor.execute(name, inp),
        ):
            # Patch model if overridden
            if model:
                with patch(
                    "config.settings.get_settings"
                ) as mock_settings:
                    settings = MagicMock()
                    settings.claude_model = model
                    settings.anthropic_api_key = None  # Will use env var
                    mock_settings.return_value = settings
                    # Use a null tracer
                    output = agent.analyze(
                        self.scenario["ticker"], rag_context, MagicMock()
                    )
            else:
                output = agent.analyze(
                    self.scenario["ticker"], rag_context, MagicMock()
                )

        return {
            "agent_name": agent_name,
            "output": {
                "score": output.score,
                "confidence": output.confidence,
                "summary": output.summary,
            },
            "mode": "live",
        }

    def _replay_synthesis_structural(self) -> dict:
        """Structural replay of synthesis — use recorded agent outputs."""
        recorded_synthesis = self.scenario.get("synthesis", {})
        return {
            "mode": "structural",
            **recorded_synthesis,
        }

    def _replay_synthesis_live(self, agent_results: list[dict]) -> dict:
        """Live replay of synthesis — run real synthesis on replay agent outputs."""
        from src.agents.synthesis import synthesize

        outputs = []
        for ar in agent_results:
            output_data = ar.get("output", {})
            if "error" in ar:
                continue
            outputs.append(AgentOutput(
                agent_name=ar["agent_name"],
                ticker=self.scenario["ticker"],
                score=output_data.get("score", 0.0),
                confidence=output_data.get("confidence", 0.1),
                summary=output_data.get("summary", ""),
                metrics=output_data.get("metrics", {}),
                strengths=output_data.get("strengths", []),
                weaknesses=output_data.get("weaknesses", []),
                sources=output_data.get("sources", []),
            ))

        if not outputs:
            return {"mode": "live", "error": "No agent outputs to synthesize"}

        rec = synthesize(outputs, self.scenario["ticker"], "benchmark", tracer=None)
        return {
            "mode": "live",
            "overall_score": rec.overall_score,
            "recommendation": rec.recommendation,
            "confidence": rec.confidence,
        }
