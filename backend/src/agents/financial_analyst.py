"""Financial Analyst agent — balance sheet, margins, ratios."""

from src.agents.base_agent import BaseAgent
from src.agents.prompts import FINANCIAL_ANALYST_PROMPT


class FinancialAnalystAgent(BaseAgent):
    AGENT_NAME = "financial_analyst"
    WEIGHT = 0.20
    TEMPERATURE = 0.2
    SECTIONS = ["balance_sheet", "income_statement", "cash_flow"]

    SYSTEM_PROMPT = FINANCIAL_ANALYST_PROMPT

    RAG_QUERIES = [
        "{ticker} revenue gross profit operating income net income margins",
        "{ticker} balance sheet total assets liabilities shareholders equity",
        "{ticker} cash flow from operations free cash flow capital expenditures",
        "{ticker} debt structure interest expense debt maturity",
    ]

    TOOLS = [
        "calculate_financial_ratio",
        "compare_companies",
        "get_stock_price",
    ]
