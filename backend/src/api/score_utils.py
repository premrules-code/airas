"""Score conversion and agent display metadata."""


def to_display_score(internal: float) -> int:
    """Convert internal score (-1..+1) to display score (0..100)."""
    return int((internal + 1) * 50)


def score_to_color(display: int) -> str:
    """Map display score to hex color."""
    if display >= 75:
        return "#10b981"
    if display >= 60:
        return "#3b82f6"
    if display >= 45:
        return "#f59e0b"
    return "#ef4444"


def score_to_signal(display: int) -> str:
    """Map display score to signal label."""
    if display >= 60:
        return "BUY"
    if display <= 40:
        return "SELL"
    return "HOLD"


AGENT_DISPLAY_META = {
    "financial_analyst": {
        "label": "Financial Analyst",
        "icon": "chart-bar",
        "description": "Revenue, margins, balance sheet health",
    },
    "news_sentiment": {
        "label": "News Sentiment",
        "icon": "newspaper",
        "description": "Recent news tone and impact",
    },
    "technical_analyst": {
        "label": "Technical Analyst",
        "icon": "chart-line",
        "description": "Price trends, indicators, patterns",
    },
    "risk_assessment": {
        "label": "Risk Assessment",
        "icon": "shield",
        "description": "Volatility, downside risk factors",
    },
    "competitive_analysis": {
        "label": "Competitive Analysis",
        "icon": "users",
        "description": "Market position vs peers",
    },
    "insider_activity": {
        "label": "Insider Activity",
        "icon": "user-check",
        "description": "Executive buying/selling patterns",
    },
    "options_analysis": {
        "label": "Options Analysis",
        "icon": "layers",
        "description": "Options flow, implied volatility",
    },
    "social_sentiment": {
        "label": "Social Sentiment",
        "icon": "message-circle",
        "description": "Social media buzz and tone",
    },
    "earnings_analysis": {
        "label": "Earnings Analysis",
        "icon": "dollar-sign",
        "description": "Earnings quality, beat/miss history",
    },
    "analyst_ratings": {
        "label": "Analyst Ratings",
        "icon": "star",
        "description": "Wall Street consensus and targets",
    },
    "synthesis": {
        "label": "Enhanced Synthesis",
        "icon": "zap",
        "description": "Weighted composite score",
    },
}
