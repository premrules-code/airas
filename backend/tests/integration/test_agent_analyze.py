"""Integration tests for BaseAgent.analyze() with mocked Claude + tools."""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.agents.financial_analyst import FinancialAnalystAgent
from src.agents.tracing import TracingManager
from src.models.structured_outputs import AgentOutput
from tests.fixtures.claude_responses import (
    make_claude_response,
    make_tool_use_response,
    MockMessage,
    MockTextBlock,
    MockToolUseBlock,
    MockUsage,
    FINANCIAL_ANALYST_BULLISH,
)
from tests.fixtures.tool_results import STOCK_PRICE_RESULT, RATIO_RESULTS

pytestmark = pytest.mark.integration

TICKER = "AAPL"


@pytest.fixture
def agent():
    return FinancialAnalystAgent()


@pytest.fixture
def tracer():
    t = TracingManager()
    t.langfuse = None
    return t


class TestAnalyzeBasicFlow:
    """Test agent.analyze() returns valid AgentOutput with mocked Claude."""

    def test_direct_json_response(self, agent, tracer, mock_settings):
        """Claude returns valid JSON on first call → valid AgentOutput."""
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = FINANCIAL_ANALYST_BULLISH

            result = agent.analyze(TICKER, "Some RAG context", tracer)

        assert isinstance(result, AgentOutput)
        assert result.agent_name == "financial_analyst"
        assert result.ticker == TICKER
        assert -1.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_rag_context(self, agent, tracer, mock_settings):
        """Empty RAG context should still work."""
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = FINANCIAL_ANALYST_BULLISH

            result = agent.analyze(TICKER, "", tracer)

        assert isinstance(result, AgentOutput)


class TestToolUseFlow:
    """Test the tool-use loop in analyze()."""

    def test_tool_use_then_final_response(self, agent, tracer, mock_settings):
        """Claude returns tool_use → tool result → final JSON response."""
        tool_use_msg = make_tool_use_response(
            "get_stock_price", {"ticker": "AAPL"}
        )
        final_msg = FINANCIAL_ANALYST_BULLISH

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls, \
             patch("src.agents.base_agent.execute_tool") as mock_exec:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.side_effect = [tool_use_msg, final_msg]
            mock_exec.return_value = STOCK_PRICE_RESULT

            result = agent.analyze(TICKER, "context", tracer)

        assert isinstance(result, AgentOutput)
        assert result.score == 0.65
        mock_exec.assert_called_once_with("get_stock_price", {"ticker": "AAPL"})

    def test_tool_error_handled(self, agent, tracer, mock_settings):
        """Tool raising exception → error dict sent back → Claude still responds."""
        tool_use_msg = make_tool_use_response(
            "get_stock_price", {"ticker": "AAPL"}
        )
        final_msg = FINANCIAL_ANALYST_BULLISH

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls, \
             patch("src.agents.base_agent.execute_tool") as mock_exec:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.side_effect = [tool_use_msg, final_msg]
            mock_exec.side_effect = RuntimeError("API down")

            result = agent.analyze(TICKER, "context", tracer)

        assert isinstance(result, AgentOutput)


class TestRetryBehavior:
    """Test rate limit retry logic."""

    def test_rate_limit_retry_succeeds(self, agent, tracer, mock_settings):
        """RateLimitError on first call, success on second."""
        import anthropic as anthropic_mod

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls, \
             patch("src.agents.base_agent.time.sleep"):  # skip actual sleep
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.side_effect = [
                anthropic_mod.RateLimitError(
                    message="rate limited",
                    response=MagicMock(status_code=429, headers={}),
                    body={"error": {"message": "rate limited", "type": "rate_limit_error"}},
                ),
                FINANCIAL_ANALYST_BULLISH,
            ]

            result = agent.analyze(TICKER, "context", tracer)

        assert isinstance(result, AgentOutput)
        assert result.score == 0.65


class TestMaxIterations:
    """Test max tool iterations safeguard."""

    def test_max_iterations_returns_fallback(self, agent, tracer, mock_settings):
        """Infinite tool_use responses → hits MAX_TOOL_ITERATIONS → fallback."""
        tool_use_msg = make_tool_use_response(
            "get_stock_price", {"ticker": "AAPL"}
        )

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls, \
             patch("src.agents.base_agent.execute_tool") as mock_exec:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            # Always return tool_use, never end_turn
            mock_client.messages.create.return_value = tool_use_msg
            mock_exec.return_value = STOCK_PRICE_RESULT

            result = agent.analyze(TICKER, "context", tracer)

        assert isinstance(result, AgentOutput)
        assert result.score == 0.0
        assert result.confidence == 0.1
        assert "max" in result.summary.lower() or "iteration" in result.summary.lower()
