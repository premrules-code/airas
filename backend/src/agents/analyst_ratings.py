"""Analyst Ratings agent — consensus rating, price targets."""

from src.agents.base_agent import BaseAgent
from src.agents.prompts import ANALYST_RATINGS_PROMPT


class AnalystRatingsAgent(BaseAgent):
    AGENT_NAME = "analyst_ratings"
    WEIGHT = 0.10
    TEMPERATURE = 0.2
    SECTIONS = None  # No RAG

    SYSTEM_PROMPT = ANALYST_RATINGS_PROMPT

    RAG_QUERIES = []  # Uses tools only

    TOOLS = [
        "get_analyst_ratings",
        "get_stock_price",
    ]
