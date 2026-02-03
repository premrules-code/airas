"""Golden scenario tests: known data → expected score ranges."""

from unittest.mock import patch, MagicMock

import pytest

from src.agents.synthesis import synthesize
from src.models.structured_outputs import AgentOutput
from tests.fixtures.agent_outputs import make_agent_output, ALL_AGENT_NAMES
from tests.fixtures.claude_responses import THESIS_RESPONSE

pytestmark = pytest.mark.scoring


@pytest.fixture(autouse=True)
def _patch_synthesis_claude():
    """Auto-patch Claude for thesis generation in all scoring tests."""
    mock_cls = MagicMock()
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = THESIS_RESPONSE
    with patch("src.agents.synthesis.anthropic.Anthropic", mock_cls):
        yield


def _bullish_scenario_outputs() -> list[AgentOutput]:
    """Bullish: strong fundamentals, trending up, insider buying."""
    return [
        make_agent_output("financial_analyst", score=0.65, confidence=0.85,
                          summary="Strong margins, PE=12, ROE=35%."),
        make_agent_output("news_sentiment", score=0.40, confidence=0.70,
                          summary="Positive news coverage."),
        make_agent_output("technical_analyst", score=0.55, confidence=0.80,
                          summary="Price above SMA50, RSI 58."),
        make_agent_output("risk_assessment", score=0.30, confidence=0.65,
                          summary="Low D/E of 0.5, manageable risks."),
        make_agent_output("competitive_analysis", score=0.40, confidence=0.70,
                          summary="Market leader with strong moat."),
        make_agent_output("insider_activity", score=0.35, confidence=0.60,
                          summary="Net insider buying."),
        make_agent_output("options_analysis", score=0.10, confidence=0.40,
                          summary="Slightly bullish options flow."),
        make_agent_output("social_sentiment", score=0.25, confidence=0.35,
                          summary="Positive social buzz."),
        make_agent_output("earnings_analysis", score=0.50, confidence=0.80,
                          summary="Consistent earnings beats."),
        make_agent_output("analyst_ratings", score=0.55, confidence=0.85,
                          summary="Strong buy consensus, 13% upside."),
    ]


def _bearish_scenario_outputs() -> list[AgentOutput]:
    """Bearish: weak fundamentals, declining price, insider selling."""
    return [
        make_agent_output("financial_analyst", score=-0.55, confidence=0.80,
                          summary="PE=55, ROE=3%, margin=2%."),
        make_agent_output("news_sentiment", score=-0.40, confidence=0.70,
                          summary="Negative headlines dominating."),
        make_agent_output("technical_analyst", score=-0.50, confidence=0.80,
                          summary="Price below SMA200, RSI 32."),
        make_agent_output("risk_assessment", score=-0.45, confidence=0.70,
                          summary="D/E=5.0, high leverage risk."),
        make_agent_output("competitive_analysis", score=-0.35, confidence=0.65,
                          summary="Losing market share to competitors."),
        make_agent_output("insider_activity", score=-0.30, confidence=0.55,
                          summary="Net insider selling."),
        make_agent_output("options_analysis", score=-0.15, confidence=0.40,
                          summary="High put/call ratio."),
        make_agent_output("social_sentiment", score=-0.20, confidence=0.30,
                          summary="Negative social sentiment."),
        make_agent_output("earnings_analysis", score=-0.45, confidence=0.75,
                          summary="Consecutive earnings misses."),
        make_agent_output("analyst_ratings", score=-0.50, confidence=0.80,
                          summary="Majority sell ratings, downside risk."),
    ]


def _neutral_scenario_outputs() -> list[AgentOutput]:
    """Neutral: average fundamentals, sideways price action."""
    return [
        make_agent_output("financial_analyst", score=0.05, confidence=0.65,
                          summary="PE=22, ROE=12%, margin=10%."),
        make_agent_output("news_sentiment", score=0.00, confidence=0.55,
                          summary="Mixed news, no clear direction."),
        make_agent_output("technical_analyst", score=0.10, confidence=0.60,
                          summary="Price near SMA50, RSI 50."),
        make_agent_output("risk_assessment", score=-0.05, confidence=0.55,
                          summary="D/E=1.5, moderate risk."),
        make_agent_output("competitive_analysis", score=0.05, confidence=0.50,
                          summary="Average competitive position."),
        make_agent_output("insider_activity", score=0.00, confidence=0.40,
                          summary="Balanced insider activity."),
        make_agent_output("options_analysis", score=0.00, confidence=0.30,
                          summary="Neutral options sentiment."),
        make_agent_output("social_sentiment", score=0.05, confidence=0.25,
                          summary="Low social media activity."),
        make_agent_output("earnings_analysis", score=0.10, confidence=0.60,
                          summary="In-line earnings."),
        make_agent_output("analyst_ratings", score=0.00, confidence=0.65,
                          summary="Hold consensus."),
    ]


class TestBullishScenario:
    """Known bullish data → BUY or STRONG BUY."""

    def test_overall_score_positive(self, mock_settings):
        outputs = _bullish_scenario_outputs()
        result = synthesize(outputs, "AAPL", "full")
        assert result.overall_score > 0.2, f"Expected > 0.2, got {result.overall_score}"

    def test_recommendation_bullish(self, mock_settings):
        outputs = _bullish_scenario_outputs()
        result = synthesize(outputs, "AAPL", "full")
        assert result.recommendation in ["BUY", "STRONG BUY"], \
            f"Expected BUY/STRONG BUY, got {result.recommendation}"

    def test_financial_score_positive(self, mock_settings):
        outputs = _bullish_scenario_outputs()
        result = synthesize(outputs, "AAPL", "full")
        assert result.financial_score > 0.0

    def test_technical_score_positive(self, mock_settings):
        outputs = _bullish_scenario_outputs()
        result = synthesize(outputs, "AAPL", "full")
        assert result.technical_score > 0.0


class TestBearishScenario:
    """Known bearish data → SELL or STRONG SELL."""

    def test_overall_score_negative(self, mock_settings):
        outputs = _bearish_scenario_outputs()
        result = synthesize(outputs, "AAPL", "full")
        assert result.overall_score < -0.2, f"Expected < -0.2, got {result.overall_score}"

    def test_recommendation_bearish(self, mock_settings):
        outputs = _bearish_scenario_outputs()
        result = synthesize(outputs, "AAPL", "full")
        assert result.recommendation in ["SELL", "STRONG SELL"], \
            f"Expected SELL/STRONG SELL, got {result.recommendation}"

    def test_financial_score_negative(self, mock_settings):
        outputs = _bearish_scenario_outputs()
        result = synthesize(outputs, "AAPL", "full")
        assert result.financial_score < 0.0

    def test_risk_score_negative(self, mock_settings):
        outputs = _bearish_scenario_outputs()
        result = synthesize(outputs, "AAPL", "full")
        assert result.risk_score < 0.0


class TestNeutralScenario:
    """Known neutral data → HOLD."""

    def test_overall_score_near_zero(self, mock_settings):
        outputs = _neutral_scenario_outputs()
        result = synthesize(outputs, "AAPL", "full")
        assert -0.2 <= result.overall_score <= 0.2, \
            f"Expected between -0.2 and 0.2, got {result.overall_score}"

    def test_recommendation_hold(self, mock_settings):
        outputs = _neutral_scenario_outputs()
        result = synthesize(outputs, "AAPL", "full")
        assert result.recommendation == "HOLD", \
            f"Expected HOLD, got {result.recommendation}"


class TestCrossScenarioConsistency:
    """Verify relative ordering between scenarios."""

    def test_bullish_beats_neutral(self, mock_settings):
        bullish = synthesize(_bullish_scenario_outputs(), "AAPL", "full")
        neutral = synthesize(_neutral_scenario_outputs(), "AAPL", "full")
        assert bullish.overall_score > neutral.overall_score

    def test_neutral_beats_bearish(self, mock_settings):
        neutral = synthesize(_neutral_scenario_outputs(), "AAPL", "full")
        bearish = synthesize(_bearish_scenario_outputs(), "AAPL", "full")
        assert neutral.overall_score > bearish.overall_score

    def test_bullish_beats_bearish(self, mock_settings):
        bullish = synthesize(_bullish_scenario_outputs(), "AAPL", "full")
        bearish = synthesize(_bearish_scenario_outputs(), "AAPL", "full")
        assert bullish.overall_score > bearish.overall_score

    def test_confidence_reasonable(self, mock_settings):
        """All scenarios should have reasonable confidence values."""
        for outputs_fn in [_bullish_scenario_outputs, _bearish_scenario_outputs, _neutral_scenario_outputs]:
            result = synthesize(outputs_fn(), "AAPL", "full")
            assert 0.0 < result.confidence < 1.0, \
                f"Confidence {result.confidence} outside reasonable range"
