"""Agent package — exports all agents and mappings."""

from src.agents.financial_analyst import FinancialAnalystAgent
from src.agents.news_sentiment import NewsSentimentAgent
from src.agents.technical_analyst import TechnicalAnalystAgent
from src.agents.risk_assessment import RiskAssessmentAgent
from src.agents.competitive_analysis import CompetitiveAnalysisAgent
from src.agents.insider_activity import InsiderActivityAgent
from src.agents.options_analysis import OptionsAnalysisAgent
from src.agents.social_sentiment import SocialSentimentAgent
from src.agents.earnings_analysis import EarningsAnalysisAgent
from src.agents.analyst_ratings import AnalystRatingsAgent

ALL_AGENTS = [
    FinancialAnalystAgent,
    NewsSentimentAgent,
    TechnicalAnalystAgent,
    RiskAssessmentAgent,
    CompetitiveAnalysisAgent,
    InsiderActivityAgent,
    OptionsAnalysisAgent,
    SocialSentimentAgent,
    EarningsAnalysisAgent,
    AnalystRatingsAgent,
]

AGENT_MAP = {cls.AGENT_NAME: cls for cls in ALL_AGENTS}
