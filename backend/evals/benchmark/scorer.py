"""Two-gate benchmark scorer — probabilistic scoring.

Implements a two-gate scoring formula (see https://www.hedra.com/blog/hedra-agent-evaluation):

    P(scenario) = ∏_t P(valid_plan_t) × P(correct_execution_t | valid_plan_t)

For AIRAS, the pipeline stages are:
    Gate 1: Routing correctness (did router pick a valid agent set?)
    Gate 2: Agent execution (did agents make correct tool calls with correct params?)
    Gate 3: Synthesis correctness (did synthesis produce a reasonable recommendation?)

The overall benchmark score is a weighted average of scenario scores,
weighted by number of pipeline stages (longer scenarios count more).

Usage:
    from evals.benchmark.scorer import BenchmarkScorer

    scorer = BenchmarkScorer()
    score = scorer.score_scenario(scenario, replay_result)
    overall = scorer.score_benchmark(scenario_scores)
"""

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StageScore:
    """Score for a single pipeline stage."""
    stage_name: str
    plan_correct: bool          # Gate 1: was the plan valid?
    execution_score: float      # Gate 2: conditional on valid plan, how well executed?
    details: dict = field(default_factory=dict)

    @property
    def combined(self) -> float:
        """P(stage) = P(valid_plan) × P(correct_execution | valid_plan)"""
        return (1.0 if self.plan_correct else 0.0) * self.execution_score


@dataclass
class ScenarioScore:
    """Full score for a single scenario across all stages."""
    scenario_id: str
    stages: list[StageScore]
    num_turns: int  # Number of pipeline stages (for weighting)

    @property
    def probability(self) -> float:
        """P(scenario) = ∏ P(stage_t) across all turns."""
        if not self.stages:
            return 0.0
        p = 1.0
        for stage in self.stages:
            p *= stage.combined
        return p

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "probability": self.probability,
            "num_turns": self.num_turns,
            "stages": [
                {
                    "stage": s.stage_name,
                    "plan_correct": s.plan_correct,
                    "execution_score": s.execution_score,
                    "combined": s.combined,
                    "details": s.details,
                }
                for s in self.stages
            ],
        }


@dataclass
class BenchmarkResult:
    """Overall benchmark result across all scenarios."""
    scenario_scores: list[ScenarioScore]
    overall_probability: float
    uncertainty: float  # Standard error

    def to_dict(self) -> dict:
        return {
            "overall_probability": self.overall_probability,
            "uncertainty": self.uncertainty,
            "num_scenarios": len(self.scenario_scores),
            "scenarios": [s.to_dict() for s in self.scenario_scores],
        }


class BenchmarkScorer:
    """Scores replay results using the two-gate metric."""

    # Thresholds
    SCORE_TOLERANCE = 0.3       # How far off agent scores can be from recorded
    CONFIDENCE_TOLERANCE = 0.2  # How far off confidence can be from recorded
    RECOMMENDATION_LEVELS = {
        "STRONG BUY": 2, "BUY": 1, "HOLD": 0, "SELL": -1, "STRONG SELL": -2,
    }

    def score_scenario(self, scenario: dict, replay_result: dict) -> ScenarioScore:
        """Score a single replayed scenario.

        Args:
            scenario: The original recorded scenario.
            replay_result: The ReplayResult.to_dict() from replaying.

        Returns:
            ScenarioScore with per-stage breakdown.
        """
        stages = []

        # Stage 1: Routing
        routing_score = self._score_routing(
            scenario.get("routing", {}),
            replay_result.get("routing", {}),
        )
        stages.append(routing_score)

        # Stage 2: Per-agent execution
        for recorded_agent in scenario.get("agent_executions", []):
            agent_name = recorded_agent["agent_name"]
            replayed_agent = self._find_agent_result(
                agent_name, replay_result.get("agents", [])
            )
            agent_score = self._score_agent(recorded_agent, replayed_agent)
            stages.append(agent_score)

        # Stage 3: Synthesis
        synthesis_score = self._score_synthesis(
            scenario.get("synthesis", {}),
            replay_result.get("synthesis", {}),
        )
        stages.append(synthesis_score)

        return ScenarioScore(
            scenario_id=scenario["scenario_id"],
            stages=stages,
            num_turns=len(stages),
        )

    def score_benchmark(self, scenario_scores: list[ScenarioScore]) -> BenchmarkResult:
        """Compute overall benchmark score across all scenarios.

        Weighted average by number of turns (longer scenarios count more).
        Uncertainty accumulated in quadrature (each scenario independent).
        """
        if not scenario_scores:
            return BenchmarkResult(
                scenario_scores=[],
                overall_probability=0.0,
                uncertainty=0.0,
            )

        total_weight = sum(s.num_turns for s in scenario_scores)
        if total_weight == 0:
            return BenchmarkResult(
                scenario_scores=scenario_scores,
                overall_probability=0.0,
                uncertainty=0.0,
            )

        # Weighted average
        weighted_sum = sum(
            s.probability * s.num_turns for s in scenario_scores
        )
        overall = weighted_sum / total_weight

        # Uncertainty in quadrature (each scenario treated as independent measurement)
        variance_sum = sum(
            ((s.probability - overall) * s.num_turns / total_weight) ** 2
            for s in scenario_scores
        )
        uncertainty = math.sqrt(variance_sum) if variance_sum > 0 else 0.0

        return BenchmarkResult(
            scenario_scores=scenario_scores,
            overall_probability=overall,
            uncertainty=uncertainty,
        )

    def score_repeated_runs(
        self, scenario: dict, replay_results: list[dict]
    ) -> dict:
        """Score N repeated runs of the same scenario to estimate P(success).

        Probabilistic scoring:
        run M repeats, estimate P(success) = successful_runs / M.

        Args:
            scenario: The original recorded scenario.
            replay_results: List of ReplayResult.to_dict() from N runs.

        Returns:
            Dict with mean probability, std dev, and per-run scores.
        """
        scores = [
            self.score_scenario(scenario, result)
            for result in replay_results
        ]
        probabilities = [s.probability for s in scores]
        n = len(probabilities)

        if n == 0:
            return {"mean": 0.0, "std": 0.0, "n": 0, "runs": []}

        mean_p = sum(probabilities) / n
        variance = sum((p - mean_p) ** 2 for p in probabilities) / max(n - 1, 1)
        std_p = math.sqrt(variance)

        return {
            "mean": mean_p,
            "std": std_p,
            "n": n,
            "error_bar": 1.96 * std_p / math.sqrt(n) if n > 1 else 0.0,
            "runs": [s.to_dict() for s in scores],
        }

    # --- Stage scorers ---

    def _score_routing(self, recorded: dict, replayed: dict) -> StageScore:
        """Score routing: did the replayed router pick a valid agent set?

        Uses acceptable sets (not exact match) because multiple agent
        combinations can be valid for the same query.
        """
        replayed_agents = set(replayed.get("active_agents", []))
        acceptable_sets = recorded.get("acceptable_agent_sets", [])

        # Check if replayed agents match any acceptable set
        plan_correct = False
        best_overlap = 0.0

        for acceptable in acceptable_sets:
            acceptable_set = set(acceptable)
            if replayed_agents == acceptable_set:
                plan_correct = True
                best_overlap = 1.0
                break
            # Partial credit: Jaccard similarity
            intersection = len(replayed_agents & acceptable_set)
            union = len(replayed_agents | acceptable_set)
            overlap = intersection / union if union > 0 else 0.0
            best_overlap = max(best_overlap, overlap)

        # If no exact match but overlap > 0.7, still count as valid plan
        if not plan_correct and best_overlap >= 0.7:
            plan_correct = True

        return StageScore(
            stage_name="routing",
            plan_correct=plan_correct,
            execution_score=best_overlap,
            details={
                "replayed_agents": sorted(replayed_agents),
                "acceptable_sets": acceptable_sets,
                "best_overlap": best_overlap,
            },
        )

    def _score_agent(self, recorded: dict, replayed: dict) -> StageScore:
        """Score a single agent: valid plan (right tools) + correct execution."""
        agent_name = recorded["agent_name"]

        if not replayed or "error" in replayed:
            return StageScore(
                stage_name=f"agent:{agent_name}",
                plan_correct=False,
                execution_score=0.0,
                details={"error": replayed.get("error", "Agent not found in replay")},
            )

        recorded_tools = recorded.get("tool_calls", [])
        replayed_output = replayed.get("output", {})
        recorded_output = recorded.get("output", {})

        # Gate 1: Did agent choose a valid plan (right tool calls)?
        # For structural replay, we compare tool call patterns.
        # For live replay, the agent may choose different but valid tools.
        plan_correct = True  # Default: trust the agent chose a valid plan

        if replayed.get("mode") == "structural":
            # In structural mode, tools should match recorded
            replayed_tools = replayed.get("recorded_tool_calls", [])
            recorded_tool_names = {tc["tool"] for tc in recorded_tools}
            replayed_tool_names = {tc["tool"] for tc in replayed_tools}
            plan_correct = recorded_tool_names == replayed_tool_names

        # Gate 2: Execution quality — how close are outputs?
        score_diff = abs(
            replayed_output.get("score", 0.0) - recorded_output.get("score", 0.0)
        )
        conf_diff = abs(
            replayed_output.get("confidence", 0.0) - recorded_output.get("confidence", 0.0)
        )

        score_ok = score_diff <= self.SCORE_TOLERANCE
        conf_ok = conf_diff <= self.CONFIDENCE_TOLERANCE

        # Execution score: weighted average of score accuracy and confidence accuracy
        execution_score = 0.0
        if score_ok and conf_ok:
            execution_score = 1.0
        elif score_ok or conf_ok:
            execution_score = 0.5
        else:
            # Partial credit based on how close
            execution_score = max(
                0.0,
                1.0 - (score_diff / 2.0 + conf_diff / 2.0),
            )

        return StageScore(
            stage_name=f"agent:{agent_name}",
            plan_correct=plan_correct,
            execution_score=execution_score,
            details={
                "score_diff": score_diff,
                "confidence_diff": conf_diff,
                "score_ok": score_ok,
                "confidence_ok": conf_ok,
            },
        )

    def _score_synthesis(self, recorded: dict, replayed: dict) -> StageScore:
        """Score synthesis: did we get a reasonable recommendation?"""
        if not replayed or "error" in replayed:
            return StageScore(
                stage_name="synthesis",
                plan_correct=False,
                execution_score=0.0,
                details={"error": replayed.get("error", "No synthesis result")},
            )

        recorded_rec = recorded.get("recommendation", "HOLD")
        replayed_rec = replayed.get("recommendation", "HOLD")

        # Gate 1: Plan correct = recommendation within 1 level
        rec_diff = abs(
            self.RECOMMENDATION_LEVELS.get(recorded_rec, 0)
            - self.RECOMMENDATION_LEVELS.get(replayed_rec, 0)
        )
        plan_correct = rec_diff <= 1  # Within one level is acceptable

        # Gate 2: Execution score based on overall score proximity
        score_diff = abs(
            replayed.get("overall_score", 0.0) - recorded.get("overall_score", 0.0)
        )
        execution_score = max(0.0, 1.0 - score_diff)

        return StageScore(
            stage_name="synthesis",
            plan_correct=plan_correct,
            execution_score=execution_score,
            details={
                "recorded_recommendation": recorded_rec,
                "replayed_recommendation": replayed_rec,
                "recommendation_diff": rec_diff,
                "score_diff": score_diff,
            },
        )

    def _find_agent_result(self, agent_name: str, agent_results: list[dict]) -> dict:
        """Find an agent's result in the replay results."""
        for ar in agent_results:
            if ar.get("agent_name") == agent_name:
                return ar
        return {}
