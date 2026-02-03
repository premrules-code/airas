"""Pre-built AgentOutput objects for testing."""

from src.models.structured_outputs import AgentOutput

ALL_AGENT_NAMES = [
    "financial_analyst",
    "news_sentiment",
    "technical_analyst",
    "risk_assessment",
    "competitive_analysis",
    "insider_activity",
    "options_analysis",
    "social_sentiment",
    "earnings_analysis",
    "analyst_ratings",
]


def make_agent_output(
    agent_name: str = "financial_analyst",
    ticker: str = "AAPL",
    score: float = 0.5,
    confidence: float = 0.8,
    summary: str = "Test summary.",
    metrics: dict | None = None,
    strengths: list[str] | None = None,
    weaknesses: list[str] | None = None,
    sources: list[str] | None = None,
) -> AgentOutput:
    """Factory for AgentOutput with sane defaults."""
    return AgentOutput(
        agent_name=agent_name,
        ticker=ticker,
        score=score,
        confidence=confidence,
        summary=summary,
        metrics=metrics or {},
        strengths=strengths or ["Strong revenue growth"],
        weaknesses=weaknesses or ["High valuation"],
        sources=sources or ["SEC filings"],
    )


def _make_set(scores: list[float], confidences: list[float], ticker: str = "AAPL") -> list[AgentOutput]:
    """Build a list of 10 AgentOutputs, one per agent."""
    outputs = []
    for i, name in enumerate(ALL_AGENT_NAMES):
        outputs.append(make_agent_output(
            agent_name=name,
            ticker=ticker,
            score=scores[i],
            confidence=confidences[i],
            summary=f"{name} analysis complete.",
        ))
    return outputs


BULLISH_OUTPUTS = _make_set(
    scores=[0.7, 0.5, 0.6, 0.4, 0.5, 0.3, 0.4, 0.3, 0.5, 0.6],
    confidences=[0.9, 0.8, 0.85, 0.7, 0.75, 0.6, 0.5, 0.4, 0.8, 0.85],
)

BEARISH_OUTPUTS = _make_set(
    scores=[-0.7, -0.5, -0.6, -0.4, -0.5, -0.3, -0.4, -0.3, -0.5, -0.6],
    confidences=[0.9, 0.8, 0.85, 0.7, 0.75, 0.6, 0.5, 0.4, 0.8, 0.85],
)

MIXED_OUTPUTS = _make_set(
    scores=[0.6, -0.4, 0.3, -0.2, 0.1, -0.1, 0.2, -0.3, 0.4, -0.5],
    confidences=[0.85, 0.7, 0.8, 0.6, 0.7, 0.5, 0.4, 0.3, 0.75, 0.8],
)

NEUTRAL_OUTPUTS = _make_set(
    scores=[0.05, -0.03, 0.02, -0.01, 0.04, -0.02, 0.01, 0.0, 0.03, -0.04],
    confidences=[0.65, 0.6, 0.7, 0.55, 0.6, 0.5, 0.4, 0.3, 0.65, 0.7],
)
