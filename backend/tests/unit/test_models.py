"""Tests for Pydantic model validation."""

import pytest
from pydantic import ValidationError

from src.models.structured_outputs import (
    AgentOutput,
    InvestmentRecommendation,
    FinancialMetrics,
    RatioResult,
    CompanyComparison,
)

pytestmark = pytest.mark.unit


class TestAgentOutput:
    """AgentOutput model validation."""

    def test_valid_construction(self):
        out = AgentOutput(
            agent_name="financial_analyst",
            ticker="AAPL",
            score=0.5,
            confidence=0.8,
            summary="Good outlook.",
        )
        assert out.agent_name == "financial_analyst"
        assert out.ticker == "AAPL"
        assert out.score == 0.5
        assert out.confidence == 0.8

    def test_score_too_high(self):
        with pytest.raises(ValidationError):
            AgentOutput(
                agent_name="test", ticker="AAPL",
                score=1.5, confidence=0.8, summary="Bad.",
            )

    def test_score_too_low(self):
        with pytest.raises(ValidationError):
            AgentOutput(
                agent_name="test", ticker="AAPL",
                score=-1.5, confidence=0.8, summary="Bad.",
            )

    def test_confidence_too_high(self):
        with pytest.raises(ValidationError):
            AgentOutput(
                agent_name="test", ticker="AAPL",
                score=0.5, confidence=1.5, summary="Bad.",
            )

    def test_confidence_too_low(self):
        with pytest.raises(ValidationError):
            AgentOutput(
                agent_name="test", ticker="AAPL",
                score=0.5, confidence=-0.1, summary="Bad.",
            )

    def test_boundary_score_values(self):
        """Exact boundary values -1 and 1 should be accepted."""
        out_min = AgentOutput(
            agent_name="test", ticker="T",
            score=-1.0, confidence=0.0, summary="Min.",
        )
        out_max = AgentOutput(
            agent_name="test", ticker="T",
            score=1.0, confidence=1.0, summary="Max.",
        )
        assert out_min.score == -1.0
        assert out_max.score == 1.0

    def test_default_timestamp_populated(self):
        out = AgentOutput(
            agent_name="test", ticker="AAPL",
            score=0.0, confidence=0.5, summary="Test.",
        )
        assert out.timestamp is not None
        assert len(out.timestamp) > 0

    def test_default_empty_lists(self):
        out = AgentOutput(
            agent_name="test", ticker="AAPL",
            score=0.0, confidence=0.5, summary="Test.",
        )
        assert out.metrics == {}
        assert out.strengths == []
        assert out.weaknesses == []
        assert out.sources == []

    def test_model_dump_roundtrip(self):
        out = AgentOutput(
            agent_name="test", ticker="AAPL",
            score=0.3, confidence=0.7, summary="Roundtrip.",
            metrics={"pe": 15}, strengths=["a"], weaknesses=["b"],
        )
        data = out.model_dump()
        restored = AgentOutput(**data)
        assert restored.score == out.score
        assert restored.agent_name == out.agent_name


class TestInvestmentRecommendation:
    """InvestmentRecommendation model validation."""

    def test_valid_construction(self):
        rec = InvestmentRecommendation(
            ticker="AAPL",
            recommendation="BUY",
            confidence=0.8,
            overall_score=0.45,
            financial_score=0.5,
            technical_score=0.4,
            sentiment_score=0.3,
            risk_score=-0.1,
            bullish_factors=["Strong revenue"],
            bearish_factors=["High PE"],
            risks=["Macro risk"],
            thesis="AAPL is a good investment.",
        )
        assert rec.recommendation == "BUY"
        assert rec.overall_score == 0.45

    def test_invalid_recommendation_string(self):
        with pytest.raises(ValidationError):
            InvestmentRecommendation(
                ticker="AAPL",
                recommendation="MAYBE BUY",
                confidence=0.8,
                overall_score=0.3,
                financial_score=0.0,
                technical_score=0.0,
                sentiment_score=0.0,
                risk_score=0.0,
                bullish_factors=[],
                bearish_factors=[],
                risks=[],
                thesis="Test.",
            )

    def test_score_fields_bounded(self):
        """overall_score, financial_score, etc. must be in [-1, 1]."""
        with pytest.raises(ValidationError):
            InvestmentRecommendation(
                ticker="AAPL",
                recommendation="HOLD",
                confidence=0.5,
                overall_score=1.5,  # out of bounds
                financial_score=0.0,
                technical_score=0.0,
                sentiment_score=0.0,
                risk_score=0.0,
                bullish_factors=[],
                bearish_factors=[],
                risks=[],
                thesis="Test.",
            )

    def test_all_valid_recommendations(self):
        for rec_str in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]:
            rec = InvestmentRecommendation(
                ticker="AAPL",
                recommendation=rec_str,
                confidence=0.5,
                overall_score=0.0,
                financial_score=0.0,
                technical_score=0.0,
                sentiment_score=0.0,
                risk_score=0.0,
                bullish_factors=[],
                bearish_factors=[],
                risks=[],
                thesis="Test.",
            )
            assert rec.recommendation == rec_str

    def test_model_dump_roundtrip(self):
        rec = InvestmentRecommendation(
            ticker="AAPL",
            recommendation="HOLD",
            confidence=0.6,
            overall_score=0.1,
            financial_score=0.2,
            technical_score=-0.1,
            sentiment_score=0.0,
            risk_score=-0.05,
            bullish_factors=["a"],
            bearish_factors=["b"],
            risks=["c"],
            thesis="Roundtrip test.",
        )
        data = rec.model_dump()
        restored = InvestmentRecommendation(**data)
        assert restored.recommendation == rec.recommendation
        assert restored.overall_score == rec.overall_score


class TestFinancialMetrics:
    """FinancialMetrics model validation."""

    def test_minimal_construction(self):
        fm = FinancialMetrics(ticker="AAPL", fiscal_period="FY2023")
        assert fm.ticker == "AAPL"
        assert fm.revenue is None

    def test_full_construction(self):
        fm = FinancialMetrics(
            ticker="AAPL", fiscal_period="FY2023",
            revenue=394.3, revenue_growth=8.0, gross_margin=44.1,
            pe_ratio=28.5, debt_to_equity=1.8,
        )
        assert fm.revenue == 394.3


class TestRatioResult:
    """RatioResult model validation."""

    def test_construction(self):
        r = RatioResult(
            ratio_name="pe_ratio", ticker="AAPL",
            value=28.5, interpretation="Overvalued",
        )
        assert r.value == 28.5

    def test_none_value(self):
        r = RatioResult(
            ratio_name="roe", ticker="AAPL",
            value=None, interpretation="N/A",
        )
        assert r.value is None


class TestCompanyComparison:
    """CompanyComparison model validation."""

    def test_construction(self):
        c = CompanyComparison(
            metric="pe_ratio",
            companies=["AAPL", "MSFT"],
            values=[28.5, 32.0],
            winner="MSFT",
            analysis="MSFT leads.",
        )
        assert c.winner == "MSFT"
