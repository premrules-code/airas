"""Earnings Analysis agent — EPS trends, earnings quality, revenue growth."""

from src.agents.base_agent import BaseAgent
from src.agents.prompts import EARNINGS_ANALYSIS_PROMPT


class EarningsAnalysisAgent(BaseAgent):
    AGENT_NAME = "earnings_analysis"
    WEIGHT = 0.07
    TEMPERATURE = 0.2
    SECTIONS = ["income_statement"]

    SYSTEM_PROMPT = EARNINGS_ANALYSIS_PROMPT

    RAG_QUERIES = [
        "{ticker} earnings per share diluted EPS trend",
        "{ticker} revenue growth year over year quarterly",
        "{ticker} segment revenue product services breakdown",
        "{ticker} operating expenses cost structure efficiency",
    ]

    TOOLS = [
        "calculate_financial_ratio",
        "compare_companies",
    ]
