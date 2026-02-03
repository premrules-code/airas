"""Smart query router — classifies query and selects relevant agents."""

import json
import logging
import time
import anthropic
from config.settings import get_settings

logger = logging.getLogger(__name__)

ROUTER_PROMPT = """Classify this financial analysis query into relevant categories.

Categories:
- "full_analysis": Comprehensive investment analysis, "should I buy/invest", overall assessment
- "financial": Balance sheet, revenue, margins, debt, earnings, cash flow
- "technical": Stock price trends, charts, moving averages, RSI, momentum
- "risk": Risk factors, volatility, downside, debt risk
- "sentiment": News tone, analyst ratings, social media mood
- "insider": Insider trading, executive buying/selling
- "competitive": Competitive advantages, moat, market position, market share

Query: "{query}"

Respond with ONLY a JSON array of category names. Example: ["financial", "technical"]
If the query implies a full investment recommendation, respond: ["full_analysis"]"""


CATEGORY_AGENTS = {
    "full_analysis": None,  # Means ALL agents
    "financial": ["financial_analyst", "earnings_analysis"],
    "technical": ["technical_analyst", "options_analysis"],
    "risk": ["risk_assessment", "competitive_analysis"],
    "sentiment": ["news_sentiment", "social_sentiment", "analyst_ratings"],
    "insider": ["insider_activity"],
    "competitive": ["competitive_analysis"],
}

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


def route_query(query: str) -> list[str]:
    """Classify query and return list of agent names to activate."""
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    try:
        response = None
        for attempt in range(4):
            try:
                response = client.messages.create(
                    model=settings.claude_model,
                    max_tokens=100,
                    temperature=0.0,
                    messages=[{"role": "user", "content": ROUTER_PROMPT.format(query=query)}],
                )
                break
            except anthropic.RateLimitError:
                if attempt < 3:
                    delay = 30 * (2 ** attempt)
                    logger.info(f"Router rate limited, retrying in {delay}s")
                    time.sleep(delay)
                else:
                    raise

        text = response.content[0].text.strip()
        # Handle potential markdown wrapping
        if "```" in text:
            text = text.split("```")[1].split("```")[0]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        categories = json.loads(text)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Router parse error, defaulting to full analysis: {e}")
        return ALL_AGENT_NAMES

    if "full_analysis" in categories:
        return ALL_AGENT_NAMES

    agents = set()
    for cat in categories:
        cat_agents = CATEGORY_AGENTS.get(cat)
        if cat_agents is not None:
            agents.update(cat_agents)

    return list(agents) if agents else ALL_AGENT_NAMES
