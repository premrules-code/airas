"""Risk Assessment agent — volatility, beta, debt risk."""

from src.agents.base_agent import BaseAgent
from src.agents.prompts import RISK_ASSESSMENT_PROMPT


class RiskAssessmentAgent(BaseAgent):
    AGENT_NAME = "risk_assessment"
    WEIGHT = 0.10
    TEMPERATURE = 0.3
    SECTIONS = None  # No RAG

    SYSTEM_PROMPT = RISK_ASSESSMENT_PROMPT

    RAG_QUERIES = []  # Uses tools only

    TOOLS = [
        "calculate_financial_ratio",
        "get_stock_price",
        "get_technical_indicators",
    ]
