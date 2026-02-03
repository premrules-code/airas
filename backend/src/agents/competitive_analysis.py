"""Competitive Analysis agent — moat, market share, competitive advantages."""

from src.agents.base_agent import BaseAgent
from src.agents.prompts import COMPETITIVE_ANALYSIS_PROMPT


class CompetitiveAnalysisAgent(BaseAgent):
    AGENT_NAME = "competitive_analysis"
    WEIGHT = 0.10
    TEMPERATURE = 0.4
    SECTIONS = None  # All sections

    SYSTEM_PROMPT = COMPETITIVE_ANALYSIS_PROMPT

    RAG_QUERIES = [
        "{ticker} competitive landscape market position market share",
        "{ticker} competitive advantages intellectual property patents",
        "{ticker} barriers to entry brand recognition customer loyalty",
        "{ticker} industry competition risk factors competitive threats",
    ]

    TOOLS = [
        "compare_companies",
        "get_stock_price",
    ]
