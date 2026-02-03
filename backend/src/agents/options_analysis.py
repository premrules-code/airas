"""Options Analysis agent — options chain, implied volatility."""

from src.agents.base_agent import BaseAgent
from src.agents.prompts import OPTIONS_ANALYSIS_PROMPT


class OptionsAnalysisAgent(BaseAgent):
    AGENT_NAME = "options_analysis"
    WEIGHT = 0.05
    TEMPERATURE = 0.2
    SECTIONS = None  # No RAG

    SYSTEM_PROMPT = OPTIONS_ANALYSIS_PROMPT

    RAG_QUERIES = []  # Uses tools only

    TOOLS = [
        "get_options_data",
        "get_stock_price",
    ]
