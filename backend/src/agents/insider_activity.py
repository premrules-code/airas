"""Insider Activity agent — insider buys/sells, transaction patterns."""

from src.agents.base_agent import BaseAgent
from src.agents.prompts import INSIDER_ACTIVITY_PROMPT


class InsiderActivityAgent(BaseAgent):
    AGENT_NAME = "insider_activity"
    WEIGHT = 0.08
    TEMPERATURE = 0.2
    SECTIONS = None  # No RAG

    SYSTEM_PROMPT = INSIDER_ACTIVITY_PROMPT

    RAG_QUERIES = []  # Uses tools only

    TOOLS = [
        "get_insider_trades",
        "get_stock_price",
    ]
