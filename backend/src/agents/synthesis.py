"""Synthesis — combine agent outputs into InvestmentRecommendation."""

import json
import logging
import time
from typing import Optional
import anthropic

from config.settings import get_settings
from src.models.structured_outputs import AgentOutput, InvestmentRecommendation
from src.agents.tracing import TracingManager

logger = logging.getLogger(__name__)

AGENT_WEIGHTS = {
    "financial_analyst": 0.20,
    "news_sentiment": 0.12,
    "technical_analyst": 0.15,
    "risk_assessment": 0.10,
    "competitive_analysis": 0.10,
    "insider_activity": 0.08,
    "options_analysis": 0.05,
    "social_sentiment": 0.03,
    "earnings_analysis": 0.07,
    "analyst_ratings": 0.10,
}

CATEGORIES = {
    "financial_score": ["financial_analyst", "earnings_analysis"],
    "technical_score": ["technical_analyst", "options_analysis"],
    "sentiment_score": ["news_sentiment", "social_sentiment", "analyst_ratings"],
    "risk_score": ["risk_assessment", "competitive_analysis", "insider_activity"],
}


def synthesize(
    outputs: list[AgentOutput],
    ticker: str,
    mode: str,
    tracer: Optional[TracingManager] = None,
) -> InvestmentRecommendation:
    """Combine agent outputs into InvestmentRecommendation."""
    settings = get_settings()
    synthesis_span = tracer.span_agent("synthesis") if tracer else None

    # Build agent_scores dict
    agent_scores = {o.agent_name: o.score for o in outputs}

    # Calculate category scores (confidence-weighted)
    category_scores = {}
    for category, agent_names in CATEGORIES.items():
        relevant = [o for o in outputs if o.agent_name in agent_names]
        if relevant:
            weighted_sum = sum(o.score * o.confidence for o in relevant)
            confidence_sum = sum(o.confidence for o in relevant)
            category_scores[category] = (
                weighted_sum / confidence_sum if confidence_sum > 0 else 0.0
            )
        else:
            category_scores[category] = 0.0

    # Calculate overall score (weight-adjusted, confidence-weighted)
    total_weighted = 0.0
    total_weight = 0.0
    for o in outputs:
        w = AGENT_WEIGHTS.get(o.agent_name, 0.05)
        total_weighted += o.score * o.confidence * w
        total_weight += o.confidence * w
    overall_score = total_weighted / total_weight if total_weight > 0 else 0.0

    # Clamp to [-1, 1]
    overall_score = max(-1.0, min(1.0, overall_score))

    # Determine recommendation
    if overall_score >= 0.6:
        rec = "STRONG BUY"
    elif overall_score >= 0.2:
        rec = "BUY"
    elif overall_score >= -0.2:
        rec = "HOLD"
    elif overall_score >= -0.6:
        rec = "SELL"
    else:
        rec = "STRONG SELL"

    # Generate thesis via Claude
    thesis_data = _generate_thesis(outputs, ticker, rec, settings)

    # Clamp category scores to [-1, 1]
    for key in category_scores:
        category_scores[key] = max(-1.0, min(1.0, category_scores[key]))

    recommendation = InvestmentRecommendation(
        ticker=ticker,
        company_name=ticker,
        recommendation=rec,
        confidence=(
            sum(o.confidence for o in outputs) / len(outputs) if outputs else 0.0
        ),
        overall_score=round(overall_score, 4),
        financial_score=round(category_scores.get("financial_score", 0.0), 4),
        technical_score=round(category_scores.get("technical_score", 0.0), 4),
        sentiment_score=round(category_scores.get("sentiment_score", 0.0), 4),
        risk_score=round(category_scores.get("risk_score", 0.0), 4),
        agent_scores=agent_scores,
        bullish_factors=thesis_data.get("bullish_factors", []),
        bearish_factors=thesis_data.get("bearish_factors", []),
        risks=thesis_data.get("risks", []),
        thesis=thesis_data.get("thesis", ""),
        num_agents=len(outputs),
    )

    if synthesis_span:
        synthesis_span.update(output={
            "recommendation": recommendation.recommendation,
            "overall_score": recommendation.overall_score,
            "confidence": recommendation.confidence,
        })
        synthesis_span.end()

    return recommendation


def _generate_thesis(
    outputs: list[AgentOutput], ticker: str, rec: str, settings
) -> dict:
    """Use Claude to generate thesis from all agent summaries."""
    summaries = "\n".join(
        f"- {o.agent_name} (score={o.score}, confidence={o.confidence}): {o.summary}"
        for o in outputs
    )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        # Retry on rate limits
        response = None
        for attempt in range(4):
            try:
                response = client.messages.create(
                    model=settings.claude_model,
                    max_tokens=500,
                    temperature=0.3,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Based on these agent analyses for {ticker} "
                                f"(recommendation: {rec}):\n\n{summaries}\n\n"
                                "Generate a JSON with:\n"
                                '- "thesis": 2-3 sentence investment thesis\n'
                                '- "bullish_factors": 3 key bullish factors (strings)\n'
                                '- "bearish_factors": 3 key bearish factors (strings)\n'
                                '- "risks": 3 key risks (strings)\n\n'
                                "Respond with ONLY valid JSON."
                            ),
                        }
                    ],
                )
                break  # Success
            except anthropic.RateLimitError:
                if attempt < 3:
                    delay = 30 * (2 ** attempt)
                    logger.info(f"Synthesis rate limited, retrying in {delay}s")
                    time.sleep(delay)
                else:
                    raise

        text = response.content[0].text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Thesis generation failed: {e}")
        return {
            "thesis": f"{ticker} is rated {rec} based on multi-factor analysis.",
            "bullish_factors": [],
            "bearish_factors": [],
            "risks": [],
        }
