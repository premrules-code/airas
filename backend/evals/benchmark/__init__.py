"""Pipeline-level benchmarking for agent evaluation.

Scenario-based benchmarking: record real pipeline runs, replay with deterministic
mocking, score with a two-gate metric, compare across models/prompts.

Inspired by: https://www.hedra.com/blog/hedra-agent-evaluation
"""

from evals.benchmark.recorder import ScenarioRecorder
from evals.benchmark.scorer import BenchmarkScorer
from evals.benchmark.replay import ScenarioReplayer

__all__ = ["ScenarioRecorder", "BenchmarkScorer", "ScenarioReplayer"]
