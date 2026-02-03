"""News Sentiment agent — management tone, MD&A language, forward guidance."""

from src.agents.base_agent import BaseAgent
from src.agents.prompts import NEWS_SENTIMENT_PROMPT


class NewsSentimentAgent(BaseAgent):
    AGENT_NAME = "news_sentiment"
    WEIGHT = 0.12
    TEMPERATURE = 0.4
    SECTIONS = None  # All sections — needs MD&A narrative

    SYSTEM_PROMPT = NEWS_SENTIMENT_PROMPT

    RAG_QUERIES = [
        "{ticker} management discussion and analysis business outlook",
        "{ticker} forward looking statements guidance expectations",
        "{ticker} risk factors significant risks uncertainties",
        "{ticker} results of operations compared to prior year performance",
    ]

    TOOLS = [
        "get_stock_price",
    ]
