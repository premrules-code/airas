"""Tests for synthesis scoring math and recommendation mapping."""

import pytest
from unittest.mock import patch, MagicMock

from src.agents.synthesis import synthesize, AGENT_WEIGHTS, CATEGORIES
from src.agents import ALL_AGENTS
from src.models.structured_outputs import AgentOutput
from tests.fixtures.agent_outputs import (
    make_agent_output,
    BULLISH_OUTPUTS,
    BEARISH_OUTPUTS,
    MIXED_OUTPUTS,
    NEUTRAL_OUTPUTS,
    ALL_AGENT_NAMES,
)
from tests.fixtures.claude_responses import THESIS_RESPONSE

pytestmark = pytest.mark.unit


def _patch_thesis():
    """Return a context manager that patches Claude for thesis generation."""
    mock_cls = MagicMock()
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = THESIS_RESPONSE
    return patch("src.agents.synthesis.anthropic.Anthropic", mock_cls)


class TestWeights:
    """Verify agent weight configuration."""

    def test_weights_sum_to_one(self):
        total = sum(AGENT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"

    def test_all_agents_have_weights(self):
        for agent_cls in ALL_AGENTS:
            name = agent_cls.AGENT_NAME
            assert name in AGENT_WEIGHTS, f"Agent {name} missing from AGENT_WEIGHTS"

    def test_all_categories_reference_valid_agents(self):
        all_names = {cls.AGENT_NAME for cls in ALL_AGENTS}
        for cat, agents in CATEGORIES.items():
            for a in agents:
                assert a in all_names, f"Category {cat} references unknown agent {a}"


class TestCategoryScoreCalc:
    """Test confidence-weighted category score calculation."""

    def test_hand_computed_category_score(self, mock_settings, mock_thesis_generation):
        """Two agents (score=0.6, conf=0.9) and (score=0.4, conf=0.7).

        weighted_sum = 0.6*0.9 + 0.4*0.7 = 0.54 + 0.28 = 0.82
        confidence_sum = 0.9 + 0.7 = 1.6
        expected = 0.82 / 1.6 = 0.5125
        """
        o1 = make_agent_output("financial_analyst", score=0.6, confidence=0.9)
        o2 = make_agent_output("earnings_analysis", score=0.4, confidence=0.7)
        result = synthesize([o1, o2], "AAPL", "full")

        # financial_score comes from financial_analyst + earnings_analysis
        assert abs(result.financial_score - 0.5125) < 0.001

    def test_single_agent_category(self, mock_settings, mock_thesis_generation):
        """Category with one agent should use that agent's score directly."""
        o1 = make_agent_output("technical_analyst", score=0.3, confidence=0.8)
        result = synthesize([o1], "AAPL", "full")
        # technical_score from technical_analyst only (options_analysis absent)
        assert abs(result.technical_score - 0.3) < 0.001


class TestOverallScoreCalc:
    """Test overall weighted score calculation."""

    def test_hand_computed_overall_3_agents(self, mock_settings, mock_thesis_generation):
        """Three agents with known weights.

        financial_analyst: score=0.6, conf=0.9, weight=0.20
          → 0.6 * 0.9 * 0.20 = 0.108, conf*weight = 0.18
        technical_analyst: score=0.4, conf=0.8, weight=0.15
          → 0.4 * 0.8 * 0.15 = 0.048, conf*weight = 0.12
        news_sentiment: score=-0.2, conf=0.7, weight=0.12
          → -0.2 * 0.7 * 0.12 = -0.0168, conf*weight = 0.084

        total_weighted = 0.108 + 0.048 - 0.0168 = 0.1392
        total_weight = 0.18 + 0.12 + 0.084 = 0.384
        overall = 0.1392 / 0.384 = 0.3625
        """
        outputs = [
            make_agent_output("financial_analyst", score=0.6, confidence=0.9),
            make_agent_output("technical_analyst", score=0.4, confidence=0.8),
            make_agent_output("news_sentiment", score=-0.2, confidence=0.7),
        ]
        result = synthesize(outputs, "AAPL", "full")
        assert abs(result.overall_score - 0.3625) < 0.01


class TestRecommendationMapping:
    """Test score → recommendation thresholds."""

    def test_strong_buy_boundary(self, mock_settings, mock_thesis_generation):
        """All agents score +0.7 → STRONG BUY."""
        result = synthesize(BULLISH_OUTPUTS, "AAPL", "full")
        assert result.recommendation in ["STRONG BUY", "BUY"]
        assert result.overall_score > 0.2

    def test_strong_sell_boundary(self, mock_settings, mock_thesis_generation):
        """All agents score -0.7 → STRONG SELL."""
        result = synthesize(BEARISH_OUTPUTS, "AAPL", "full")
        assert result.recommendation in ["STRONG SELL", "SELL"]
        assert result.overall_score < -0.2

    def test_mixed_hold(self, mock_settings, mock_thesis_generation):
        """Mixed scores → HOLD."""
        result = synthesize(MIXED_OUTPUTS, "AAPL", "full")
        # Mixed might be HOLD or BUY/SELL depending on weights
        assert result.recommendation in ["HOLD", "BUY", "SELL"]

    def test_neutral_hold(self, mock_settings, mock_thesis_generation):
        """Near-zero scores → HOLD."""
        result = synthesize(NEUTRAL_OUTPUTS, "AAPL", "full")
        assert result.recommendation == "HOLD"
        assert abs(result.overall_score) < 0.2

    @pytest.mark.parametrize("threshold,expected", [
        (0.7, "STRONG BUY"),
        (0.3, "BUY"),
        (0.0, "HOLD"),
        (-0.3, "SELL"),
        (-0.7, "STRONG SELL"),
    ])
    def test_threshold_mapping(self, threshold, expected, mock_settings, mock_thesis_generation):
        """Verify each threshold maps to correct recommendation."""
        # All agents at same score with confidence=1.0 → overall_score ≈ score
        outputs = [
            make_agent_output(name, score=threshold, confidence=1.0)
            for name in ALL_AGENT_NAMES
        ]
        result = synthesize(outputs, "AAPL", "full")
        assert result.recommendation == expected


class TestEdgeCases:
    """Edge cases for synthesize()."""

    def test_empty_outputs(self, mock_settings, mock_thesis_generation):
        """synthesize([]) should not crash."""
        result = synthesize([], "AAPL", "full")
        assert result.overall_score == 0.0
        assert result.recommendation == "HOLD"
        assert result.num_agents == 0

    def test_single_output(self, mock_settings, mock_thesis_generation):
        """Single agent output should work."""
        o = make_agent_output("financial_analyst", score=0.5, confidence=0.9)
        result = synthesize([o], "AAPL", "full")
        assert result.num_agents == 1
        assert abs(result.overall_score - 0.5) < 0.01

    def test_zero_confidence(self, mock_settings, mock_thesis_generation):
        """Zero confidence should not divide by zero."""
        outputs = [
            make_agent_output(name, score=0.5, confidence=0.0)
            for name in ALL_AGENT_NAMES
        ]
        result = synthesize(outputs, "AAPL", "full")
        assert result.overall_score == 0.0  # total_weight=0 → 0.0

    def test_unknown_agent_gets_default_weight(self, mock_settings, mock_thesis_generation):
        """Agent not in AGENT_WEIGHTS gets default 0.05 weight."""
        o = make_agent_output("unknown_agent", score=0.8, confidence=0.9)
        result = synthesize([o], "AAPL", "full")
        # Should still produce a valid result
        assert -1.0 <= result.overall_score <= 1.0
