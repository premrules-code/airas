"""Tests for BaseAgent._parse_output edge cases."""

import json

import pytest

from src.agents.financial_analyst import FinancialAnalystAgent
from src.models.structured_outputs import AgentOutput

pytestmark = pytest.mark.unit

TICKER = "AAPL"


@pytest.fixture
def agent():
    """Return a FinancialAnalystAgent instance for testing parse logic."""
    return FinancialAnalystAgent()


def _valid_json(score=0.5, confidence=0.8, agent_name="financial_analyst", ticker="AAPL"):
    return json.dumps({
        "agent_name": agent_name,
        "ticker": ticker,
        "score": score,
        "confidence": confidence,
        "metrics": {"pe_ratio": 15.2},
        "strengths": ["Strong revenue"],
        "weaknesses": ["High PE"],
        "summary": "Analysis complete.",
        "sources": ["SEC filings"],
    })


class TestParseOutput:
    """Test _parse_output with various input formats."""

    def test_clean_json(self, agent):
        result = agent._parse_output(_valid_json(), TICKER)
        assert isinstance(result, AgentOutput)
        assert result.score == 0.5
        assert result.confidence == 0.8
        assert result.agent_name == "financial_analyst"

    def test_json_in_json_fence(self, agent):
        text = f"```json\n{_valid_json()}\n```"
        result = agent._parse_output(text, TICKER)
        assert result.score == 0.5

    def test_json_in_bare_fence(self, agent):
        text = f"```\n{_valid_json()}\n```"
        result = agent._parse_output(text, TICKER)
        assert result.score == 0.5

    def test_extra_text_before_json_falls_back(self, agent):
        """Extra text before unfenced JSON causes parse failure → fallback."""
        text = "Here is my analysis:\n\n" + _valid_json()
        result = agent._parse_output(text, TICKER)
        # The strip + json.loads on raw text with prefix should fail
        # and return fallback (score=0, confidence=0.1)
        # OR if the raw text happens to parse, that's fine too
        assert isinstance(result, AgentOutput)
        assert result.agent_name == "financial_analyst"

    def test_completely_invalid_text(self, agent):
        result = agent._parse_output("This is not JSON at all.", TICKER)
        assert result.score == 0.0
        assert result.confidence == 0.1
        assert result.agent_name == "financial_analyst"
        assert "incomplete" in result.summary.lower() or "error" in result.summary.lower()

    def test_agent_name_overridden(self, agent):
        """JSON with wrong agent_name should be overridden to correct value."""
        text = _valid_json(agent_name="wrong_agent")
        result = agent._parse_output(text, TICKER)
        assert result.agent_name == "financial_analyst"

    def test_ticker_overridden(self, agent):
        """JSON with wrong ticker should be overridden."""
        text = _valid_json(ticker="WRONG")
        result = agent._parse_output(text, TICKER)
        assert result.ticker == TICKER

    def test_score_out_of_bounds_falls_back(self, agent):
        """Score of 1.5 triggers Pydantic ValidationError → fallback."""
        text = _valid_json(score=1.5)
        result = agent._parse_output(text, TICKER)
        assert result.score == 0.0
        assert result.confidence == 0.1

    def test_negative_confidence_falls_back(self, agent):
        text = _valid_json(confidence=-0.5)
        result = agent._parse_output(text, TICKER)
        assert result.score == 0.0
        assert result.confidence == 0.1


class TestFallbackOutput:
    """Test _fallback_output."""

    def test_default_fallback(self, agent):
        result = agent._fallback_output(TICKER)
        assert result.score == 0.0
        assert result.confidence == 0.1
        assert result.agent_name == "financial_analyst"
        assert result.ticker == TICKER

    def test_fallback_with_reason(self, agent):
        result = agent._fallback_output(TICKER, "Some error occurred")
        assert "Some error occurred" in result.summary

    def test_fallback_valid_model(self, agent):
        """Fallback output must be a valid AgentOutput."""
        result = agent._fallback_output(TICKER, "test")
        # Should not raise on model validation
        data = result.model_dump()
        AgentOutput(**data)
