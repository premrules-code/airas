"""Social Sentiment agent — StockTwits + Reddit + yfinance news."""

from src.agents.base_agent import BaseAgent
from src.agents.prompts import SOCIAL_SENTIMENT_PROMPT


class SocialSentimentAgent(BaseAgent):
    AGENT_NAME = "social_sentiment"
    WEIGHT = 0.03
    TEMPERATURE = 0.3
    SECTIONS = None  # No RAG

    SYSTEM_PROMPT = SOCIAL_SENTIMENT_PROMPT

    RAG_QUERIES = []  # Uses tools only

    TOOLS = [
        "get_social_sentiment",
        "get_stock_price",
    ]
