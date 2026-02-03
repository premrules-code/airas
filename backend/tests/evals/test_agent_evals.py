"""Per-agent evals: configuration, tool-use flow, score direction, fallback behavior.

Each of the 10 agents is tested for:
- Configuration correctness (AGENT_NAME, WEIGHT, TOOLS, TEMPERATURE)
- Direct JSON response → valid AgentOutput with correct agent_name
- Tool-use flow → agent dispatches its expected tool, returns valid output
- Bullish data → positive score
- Bearish data → negative score
- Score polarity flips between bullish/bearish scenarios
- Fallback on Claude error → neutral output
"""

from unittest.mock import patch, MagicMock

import pytest

from src.agents import ALL_AGENTS, AGENT_MAP
from src.agents.base_agent import BaseAgent
from src.agents.synthesis import AGENT_WEIGHTS
from src.agents.tracing import TracingManager
from src.models.structured_outputs import AgentOutput
from tests.fixtures.per_agent_responses import AGENT_FIXTURES
from tests.fixtures.claude_responses import make_claude_response, make_tool_use_response

pytestmark = pytest.mark.scoring

TICKER = "AAPL"

# All agent names for parametrization
AGENT_NAMES = [cls.AGENT_NAME for cls in ALL_AGENTS]


@pytest.fixture
def tracer():
    t = TracingManager()
    t.langfuse = None
    return t


# ============================================================================
# 1. Configuration validation — every agent
# ============================================================================


class TestAgentConfiguration:
    """Verify each agent's class attributes are correctly set."""

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_agent_name_matches_class(self, agent_name):
        cls = AGENT_MAP[agent_name]
        instance = cls()
        assert instance.AGENT_NAME == agent_name

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_agent_is_base_agent_subclass(self, agent_name):
        cls = AGENT_MAP[agent_name]
        assert issubclass(cls, BaseAgent)

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_agent_has_weight_in_registry(self, agent_name):
        assert agent_name in AGENT_WEIGHTS, f"{agent_name} missing from AGENT_WEIGHTS"

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_agent_weight_matches_synthesis(self, agent_name):
        cls = AGENT_MAP[agent_name]
        instance = cls()
        assert instance.WEIGHT == AGENT_WEIGHTS[agent_name], \
            f"{agent_name} WEIGHT={instance.WEIGHT} != AGENT_WEIGHTS={AGENT_WEIGHTS[agent_name]}"

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_agent_has_system_prompt(self, agent_name):
        cls = AGENT_MAP[agent_name]
        instance = cls()
        assert len(instance.SYSTEM_PROMPT) > 50, f"{agent_name} has suspiciously short prompt"

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_agent_temperature_in_range(self, agent_name):
        cls = AGENT_MAP[agent_name]
        instance = cls()
        assert 0.0 <= instance.TEMPERATURE <= 1.0

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_agent_tools_is_list(self, agent_name):
        cls = AGENT_MAP[agent_name]
        instance = cls()
        assert isinstance(instance.TOOLS, list)

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_agent_rag_queries_is_list(self, agent_name):
        cls = AGENT_MAP[agent_name]
        instance = cls()
        assert isinstance(instance.RAG_QUERIES, list)


# ============================================================================
# Agent-specific tool mappings
# ============================================================================

EXPECTED_TOOLS = {
    "financial_analyst": ["calculate_financial_ratio", "compare_companies", "get_stock_price"],
    "news_sentiment": ["get_stock_price"],
    "technical_analyst": ["get_technical_indicators", "get_stock_price"],
    "risk_assessment": ["calculate_financial_ratio", "get_stock_price", "get_technical_indicators"],
    "competitive_analysis": ["compare_companies", "get_stock_price"],
    "insider_activity": ["get_insider_trades", "get_stock_price"],
    "options_analysis": ["get_options_data", "get_stock_price"],
    "social_sentiment": ["get_social_sentiment", "get_stock_price"],
    "earnings_analysis": ["calculate_financial_ratio", "compare_companies"],
    "analyst_ratings": ["get_analyst_ratings", "get_stock_price"],
}

AGENTS_WITH_RAG = {
    "financial_analyst", "news_sentiment", "competitive_analysis", "earnings_analysis",
}

AGENTS_WITHOUT_RAG = {
    "technical_analyst", "risk_assessment", "insider_activity",
    "options_analysis", "social_sentiment", "analyst_ratings",
}


class TestAgentToolConfig:
    """Verify each agent declares the correct tools."""

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_expected_tools(self, agent_name):
        cls = AGENT_MAP[agent_name]
        instance = cls()
        assert sorted(instance.TOOLS) == sorted(EXPECTED_TOOLS[agent_name]), \
            f"{agent_name} tools mismatch: {instance.TOOLS} != {EXPECTED_TOOLS[agent_name]}"

    @pytest.mark.parametrize("agent_name", sorted(AGENTS_WITH_RAG))
    def test_rag_agents_have_queries(self, agent_name):
        cls = AGENT_MAP[agent_name]
        instance = cls()
        assert len(instance.RAG_QUERIES) > 0, f"{agent_name} should have RAG queries"

    @pytest.mark.parametrize("agent_name", sorted(AGENTS_WITHOUT_RAG))
    def test_non_rag_agents_empty_queries(self, agent_name):
        cls = AGENT_MAP[agent_name]
        instance = cls()
        assert len(instance.RAG_QUERIES) == 0, f"{agent_name} should have no RAG queries"


# ============================================================================
# 2. Direct response eval — each agent parses JSON correctly
# ============================================================================


class TestDirectResponse:
    """Each agent returns valid AgentOutput when Claude responds with JSON directly."""

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_bullish_direct_response(self, agent_name, tracer, mock_settings):
        cls = AGENT_MAP[agent_name]
        agent = cls()
        data = AGENT_FIXTURES[agent_name]["bullish"]

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]

            result = agent.analyze(TICKER, "Some RAG context", tracer)

        assert isinstance(result, AgentOutput)
        assert result.agent_name == agent_name
        assert result.ticker == TICKER
        assert -1.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_bearish_direct_response(self, agent_name, tracer, mock_settings):
        cls = AGENT_MAP[agent_name]
        agent = cls()
        data = AGENT_FIXTURES[agent_name]["bearish"]

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]

            result = agent.analyze(TICKER, "", tracer)

        assert isinstance(result, AgentOutput)
        assert result.agent_name == agent_name
        assert result.score < 0.0, f"{agent_name} bearish should have negative score"


# ============================================================================
# 3. Tool-use flow eval — each agent dispatches its tool, then responds
# ============================================================================


class TestToolUseFlow:
    """Each agent's tool-use loop works: tool_use → tool result → final JSON."""

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_tool_use_then_response(self, agent_name, tracer, mock_settings):
        cls = AGENT_MAP[agent_name]
        agent = cls()
        data = AGENT_FIXTURES[agent_name]["bullish"]

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls, \
             patch("src.agents.base_agent.execute_tool") as mock_exec:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.side_effect = [
                data["tool_use"],
                data["response"],
            ]
            mock_exec.return_value = data["tool_result"]

            result = agent.analyze(TICKER, "context", tracer)

        assert isinstance(result, AgentOutput)
        assert result.agent_name == agent_name
        # Verify the agent called its expected tool
        called_tool = mock_exec.call_args[0][0]
        assert called_tool in EXPECTED_TOOLS[agent_name], \
            f"{agent_name} called {called_tool}, expected one of {EXPECTED_TOOLS[agent_name]}"


# ============================================================================
# 4. Score direction eval — bullish vs bearish polarity
# ============================================================================


class TestScoreDirection:
    """Bullish data produces higher score than bearish data for each agent."""

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_bullish_beats_bearish(self, agent_name, tracer, mock_settings):
        cls = AGENT_MAP[agent_name]
        agent = cls()
        bullish_data = AGENT_FIXTURES[agent_name]["bullish"]
        bearish_data = AGENT_FIXTURES[agent_name]["bearish"]

        # Bullish run
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = bullish_data["response"]
            bullish_result = agent.analyze(TICKER, "context", tracer)

        # Bearish run
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = bearish_data["response"]
            bearish_result = agent.analyze(TICKER, "context", tracer)

        assert bullish_result.score > bearish_result.score, \
            f"{agent_name}: bullish={bullish_result.score} should > bearish={bearish_result.score}"

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_bullish_score_positive(self, agent_name, tracer, mock_settings):
        cls = AGENT_MAP[agent_name]
        agent = cls()
        data = AGENT_FIXTURES[agent_name]["bullish"]

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "context", tracer)

        assert result.score > 0.0, \
            f"{agent_name} bullish score should be positive, got {result.score}"

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_bearish_score_negative(self, agent_name, tracer, mock_settings):
        cls = AGENT_MAP[agent_name]
        agent = cls()
        data = AGENT_FIXTURES[agent_name]["bearish"]

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "context", tracer)

        assert result.score < 0.0, \
            f"{agent_name} bearish score should be negative, got {result.score}"


# ============================================================================
# 5. Confidence eval — confidence within expected ranges
# ============================================================================


class TestConfidenceLevels:
    """Each agent reports reasonable confidence for its data quality."""

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_confidence_bounded(self, agent_name, tracer, mock_settings):
        cls = AGENT_MAP[agent_name]
        agent = cls()
        data = AGENT_FIXTURES[agent_name]["bullish"]

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "context", tracer)

        assert 0.0 <= result.confidence <= 1.0
        # Agents with tools+RAG should have moderate-to-high confidence
        # Agents with limited data (options, social) often have lower confidence
        if agent_name in ("options_analysis", "social_sentiment"):
            assert result.confidence <= 0.80, \
                f"{agent_name} shouldn't be over-confident with limited data"


# ============================================================================
# 6. Fallback eval — each agent handles errors gracefully
# ============================================================================


class TestFallbackBehavior:
    """Each agent falls back to neutral output on Claude errors."""

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_fallback_on_exception(self, agent_name, tracer, mock_settings):
        """Claude raises an exception → agent returns fallback."""
        cls = AGENT_MAP[agent_name]
        agent = cls()

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.side_effect = RuntimeError("API down")

            result = agent.analyze(TICKER, "context", tracer)

        assert isinstance(result, AgentOutput)
        assert result.agent_name == agent_name
        assert result.score == 0.0
        assert result.confidence == 0.1

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_fallback_on_malformed_json(self, agent_name, tracer, mock_settings):
        """Claude returns garbage text → agent falls back."""
        cls = AGENT_MAP[agent_name]
        agent = cls()

        malformed = make_claude_response("This is not valid JSON at all {{{")

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = malformed

            result = agent.analyze(TICKER, "context", tracer)

        assert result.score == 0.0
        assert result.confidence == 0.1
        assert result.agent_name == agent_name

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_fallback_on_tool_error_loop(self, agent_name, tracer, mock_settings):
        """Infinite tool-use loop → hits MAX_TOOL_ITERATIONS → fallback."""
        cls = AGENT_MAP[agent_name]
        agent = cls()
        data = AGENT_FIXTURES[agent_name]["bullish"]

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls, \
             patch("src.agents.base_agent.execute_tool") as mock_exec:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            # Always return tool_use, never end_turn
            mock_client.messages.create.return_value = data["tool_use"]
            mock_exec.return_value = data["tool_result"]

            result = agent.analyze(TICKER, "context", tracer)

        assert result.score == 0.0
        assert result.confidence == 0.1


# ============================================================================
# 7. Agent-specific scoring rubric evals
# ============================================================================


class TestFinancialAnalystScoring:
    """Financial analyst score direction matches fundamentals."""

    def test_low_pe_high_roe_is_bullish(self, tracer, mock_settings):
        from src.agents.financial_analyst import FinancialAnalystAgent
        agent = FinancialAnalystAgent()
        data = AGENT_FIXTURES["financial_analyst"]["bullish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "context", tracer)
        assert result.score >= 0.5, "Low PE + high ROE should be strongly bullish"

    def test_high_pe_low_roe_is_bearish(self, tracer, mock_settings):
        from src.agents.financial_analyst import FinancialAnalystAgent
        agent = FinancialAnalystAgent()
        data = AGENT_FIXTURES["financial_analyst"]["bearish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "context", tracer)
        assert result.score <= -0.3, "High PE + low ROE should be bearish"


class TestTechnicalAnalystScoring:
    """Technical analyst score matches indicator signals."""

    def test_above_smas_macd_bullish(self, tracer, mock_settings):
        from src.agents.technical_analyst import TechnicalAnalystAgent
        agent = TechnicalAnalystAgent()
        data = AGENT_FIXTURES["technical_analyst"]["bullish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "", tracer)
        assert result.score >= 0.3, "Price above SMAs + bullish MACD should be bullish"

    def test_below_smas_macd_bearish(self, tracer, mock_settings):
        from src.agents.technical_analyst import TechnicalAnalystAgent
        agent = TechnicalAnalystAgent()
        data = AGENT_FIXTURES["technical_analyst"]["bearish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "", tracer)
        assert result.score <= -0.3, "Price below SMAs + bearish MACD should be bearish"


class TestRiskAssessmentScoring:
    """Risk assessment inverted scale: high score = low risk."""

    def test_low_leverage_is_positive(self, tracer, mock_settings):
        from src.agents.risk_assessment import RiskAssessmentAgent
        agent = RiskAssessmentAgent()
        data = AGENT_FIXTURES["risk_assessment"]["bullish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "", tracer)
        assert result.score > 0.0, "Low risk profile should produce positive score"

    def test_high_leverage_is_negative(self, tracer, mock_settings):
        from src.agents.risk_assessment import RiskAssessmentAgent
        agent = RiskAssessmentAgent()
        data = AGENT_FIXTURES["risk_assessment"]["bearish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "", tracer)
        assert result.score < 0.0, "High risk profile should produce negative score"


class TestInsiderActivityScoring:
    """Insider activity: buying = bullish, selling = bearish."""

    def test_cluster_buying_is_bullish(self, tracer, mock_settings):
        from src.agents.insider_activity import InsiderActivityAgent
        agent = InsiderActivityAgent()
        data = AGENT_FIXTURES["insider_activity"]["bullish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "", tracer)
        assert result.score >= 0.3, "Cluster insider buying should be bullish"

    def test_heavy_selling_is_bearish(self, tracer, mock_settings):
        from src.agents.insider_activity import InsiderActivityAgent
        agent = InsiderActivityAgent()
        data = AGENT_FIXTURES["insider_activity"]["bearish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "", tracer)
        assert result.score <= -0.2, "Heavy insider selling should be bearish"


class TestCompetitiveAnalysisScoring:
    """Competitive analysis: wide moat = bullish, no moat = bearish."""

    def test_wide_moat_is_bullish(self, tracer, mock_settings):
        from src.agents.competitive_analysis import CompetitiveAnalysisAgent
        agent = CompetitiveAnalysisAgent()
        data = AGENT_FIXTURES["competitive_analysis"]["bullish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "context", tracer)
        assert result.score >= 0.3, "Wide moat should be bullish"

    def test_narrow_moat_is_bearish(self, tracer, mock_settings):
        from src.agents.competitive_analysis import CompetitiveAnalysisAgent
        agent = CompetitiveAnalysisAgent()
        data = AGENT_FIXTURES["competitive_analysis"]["bearish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "context", tracer)
        assert result.score < 0.0, "Narrowing moat should be bearish"


class TestNewsSentimentScoring:
    """News sentiment: optimistic filing tone = bullish."""

    def test_optimistic_tone_bullish(self, tracer, mock_settings):
        from src.agents.news_sentiment import NewsSentimentAgent
        agent = NewsSentimentAgent()
        data = AGENT_FIXTURES["news_sentiment"]["bullish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "Positive MD&A text", tracer)
        assert result.score > 0.0, "Optimistic filing tone should be bullish"

    def test_cautious_tone_bearish(self, tracer, mock_settings):
        from src.agents.news_sentiment import NewsSentimentAgent
        agent = NewsSentimentAgent()
        data = AGENT_FIXTURES["news_sentiment"]["bearish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "Cautious risk factors", tracer)
        assert result.score < 0.0, "Cautious filing tone should be bearish"


class TestOptionsAnalysisScoring:
    """Options: low PCR = bullish, high PCR = bearish."""

    def test_low_pcr_bullish(self, tracer, mock_settings):
        from src.agents.options_analysis import OptionsAnalysisAgent
        agent = OptionsAnalysisAgent()
        data = AGENT_FIXTURES["options_analysis"]["bullish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "", tracer)
        assert result.score > 0.0, "Low put/call ratio should be bullish"

    def test_high_pcr_bearish(self, tracer, mock_settings):
        from src.agents.options_analysis import OptionsAnalysisAgent
        agent = OptionsAnalysisAgent()
        data = AGENT_FIXTURES["options_analysis"]["bearish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "", tracer)
        assert result.score < 0.0, "High put/call ratio should be bearish"


class TestSocialSentimentScoring:
    """Social sentiment: bullish crowd = positive, bearish crowd = negative."""

    def test_bullish_social(self, tracer, mock_settings):
        from src.agents.social_sentiment import SocialSentimentAgent
        agent = SocialSentimentAgent()
        data = AGENT_FIXTURES["social_sentiment"]["bullish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "", tracer)
        assert result.score > 0.0

    def test_bearish_social(self, tracer, mock_settings):
        from src.agents.social_sentiment import SocialSentimentAgent
        agent = SocialSentimentAgent()
        data = AGENT_FIXTURES["social_sentiment"]["bearish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "", tracer)
        assert result.score < 0.0


class TestEarningsAnalysisScoring:
    """Earnings: growing EPS = bullish, declining = bearish."""

    def test_growing_eps_bullish(self, tracer, mock_settings):
        from src.agents.earnings_analysis import EarningsAnalysisAgent
        agent = EarningsAnalysisAgent()
        data = AGENT_FIXTURES["earnings_analysis"]["bullish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "EPS up 15%", tracer)
        assert result.score >= 0.3, "Growing EPS should be bullish"

    def test_declining_eps_bearish(self, tracer, mock_settings):
        from src.agents.earnings_analysis import EarningsAnalysisAgent
        agent = EarningsAnalysisAgent()
        data = AGENT_FIXTURES["earnings_analysis"]["bearish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "EPS down 12%", tracer)
        assert result.score <= -0.2, "Declining EPS should be bearish"


class TestAnalystRatingsScoring:
    """Analyst ratings: strong buy consensus = bullish, sell = bearish."""

    def test_strong_buy_consensus_bullish(self, tracer, mock_settings):
        from src.agents.analyst_ratings import AnalystRatingsAgent
        agent = AnalystRatingsAgent()
        data = AGENT_FIXTURES["analyst_ratings"]["bullish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "", tracer)
        assert result.score >= 0.3, "Strong Buy consensus should be bullish"

    def test_sell_consensus_bearish(self, tracer, mock_settings):
        from src.agents.analyst_ratings import AnalystRatingsAgent
        agent = AnalystRatingsAgent()
        data = AGENT_FIXTURES["analyst_ratings"]["bearish"]
        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "", tracer)
        assert result.score <= -0.2, "Sell consensus should be bearish"


# ============================================================================
# 8. Multi-tool sequence eval — agents that call multiple tools
# ============================================================================


class TestMultiToolAgents:
    """Agents that use multiple tools in sequence."""

    def test_financial_analyst_multi_tool(self, tracer, mock_settings):
        """Financial analyst calls ratio, then compare, then final response."""
        from src.agents.financial_analyst import FinancialAnalystAgent
        agent = FinancialAnalystAgent()
        data = AGENT_FIXTURES["financial_analyst"]["bullish"]

        ratio_call = make_tool_use_response(
            "calculate_financial_ratio", {"ratio_type": "pe_ratio", "ticker": "AAPL"}, "toolu_01"
        )
        compare_call = make_tool_use_response(
            "compare_companies", {"tickers": ["AAPL", "MSFT"], "metric": "pe_ratio"}, "toolu_02"
        )
        final = data["response"]

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls, \
             patch("src.agents.base_agent.execute_tool") as mock_exec:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.side_effect = [ratio_call, compare_call, final]
            mock_exec.side_effect = [
                {"ratio_name": "pe_ratio", "value": 12.0, "interpretation": "Fairly valued"},
                {"metric": "pe_ratio", "winner": "AAPL", "values": [12.0, 28.5]},
            ]

            result = agent.analyze(TICKER, "context", tracer)

        assert isinstance(result, AgentOutput)
        assert mock_exec.call_count == 2
        called_tools = [call[0][0] for call in mock_exec.call_args_list]
        assert "calculate_financial_ratio" in called_tools
        assert "compare_companies" in called_tools

    def test_risk_assessment_multi_tool(self, tracer, mock_settings):
        """Risk assessment calls ratio, then technicals, then final response."""
        from src.agents.risk_assessment import RiskAssessmentAgent
        agent = RiskAssessmentAgent()

        ratio_call = make_tool_use_response(
            "calculate_financial_ratio", {"ratio_type": "debt_to_equity", "ticker": "AAPL"}, "toolu_01"
        )
        tech_call = make_tool_use_response(
            "get_technical_indicators", {"ticker": "AAPL"}, "toolu_02"
        )
        final = AGENT_FIXTURES["risk_assessment"]["bullish"]["response"]

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls, \
             patch("src.agents.base_agent.execute_tool") as mock_exec:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.side_effect = [ratio_call, tech_call, final]
            mock_exec.side_effect = [
                {"ratio_name": "debt_to_equity", "value": 0.8, "interpretation": "Low leverage"},
                {"ticker": "AAPL", "rsi_14": 55.0, "trend": "bullish"},
            ]

            result = agent.analyze(TICKER, "", tracer)

        assert isinstance(result, AgentOutput)
        assert mock_exec.call_count == 2


# ============================================================================
# 9. Output quality eval — all fields populated
# ============================================================================


class TestOutputQuality:
    """Each agent populates all expected fields in AgentOutput."""

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_output_has_summary(self, agent_name, tracer, mock_settings):
        cls = AGENT_MAP[agent_name]
        agent = cls()
        data = AGENT_FIXTURES[agent_name]["bullish"]

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "context", tracer)

        assert len(result.summary) > 0, f"{agent_name} should have non-empty summary"
        assert len(result.strengths) > 0, f"{agent_name} should have strengths"
        assert len(result.weaknesses) > 0, f"{agent_name} should have weaknesses"

    @pytest.mark.parametrize("agent_name", AGENT_NAMES)
    def test_output_roundtrip(self, agent_name, tracer, mock_settings):
        """AgentOutput from each agent can be serialized and restored."""
        cls = AGENT_MAP[agent_name]
        agent = cls()
        data = AGENT_FIXTURES[agent_name]["bullish"]

        with patch("src.agents.base_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = data["response"]
            result = agent.analyze(TICKER, "context", tracer)

        dumped = result.model_dump()
        restored = AgentOutput(**dumped)
        assert restored.score == result.score
        assert restored.agent_name == result.agent_name
