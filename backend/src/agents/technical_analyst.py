"""Technical Analyst agent — price trends, momentum indicators."""

from src.agents.base_agent import BaseAgent
from src.agents.prompts import TECHNICAL_ANALYST_PROMPT


class TechnicalAnalystAgent(BaseAgent):
    AGENT_NAME = "technical_analyst"
    WEIGHT = 0.15
    TEMPERATURE = 0.2
    SECTIONS = None  # No RAG — pure tools

    SYSTEM_PROMPT = TECHNICAL_ANALYST_PROMPT

    RAG_QUERIES = []  # No RAG queries

    TOOLS = [
        "get_technical_indicators",
        "get_stock_price",
    ]
