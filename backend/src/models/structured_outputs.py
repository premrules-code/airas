# backend/src/models/structured_outputs.py
"""
Pydantic models for structured outputs.
All agents return typed, validated data.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from datetime import datetime


class AgentOutput(BaseModel):
    """Standard output from any agent."""
    
    agent_name: str = Field(description="Name of the agent")
    ticker: str = Field(description="Stock ticker analyzed")
    score: float = Field(ge=-1, le=1, description="Score from -1 (bearish) to +1 (bullish)")
    confidence: float = Field(ge=0, le=1, description="Confidence level 0-1")
    
    # Detailed results
    metrics: Dict = Field(default_factory=dict, description="Key metrics analyzed")
    strengths: List[str] = Field(default_factory=list, description="Top strengths (max 3)")
    weaknesses: List[str] = Field(default_factory=list, description="Top weaknesses (max 3)")
    
    # Summary
    summary: str = Field(description="One-sentence summary")
    
    # Metadata
    sources: List[str] = Field(default_factory=list, description="Data sources used")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class FinancialMetrics(BaseModel):
    """Structured financial data."""
    
    ticker: str
    fiscal_period: str
    
    # Income Statement
    revenue: Optional[float] = Field(None, description="Revenue in billions")
    revenue_growth: Optional[float] = Field(None, description="YoY growth %")
    gross_margin: Optional[float] = Field(None, description="Gross margin %")
    operating_margin: Optional[float] = Field(None, description="Operating margin %")
    net_margin: Optional[float] = Field(None, description="Net margin %")
    net_income: Optional[float] = Field(None, description="Net income in billions")
    
    # Balance Sheet
    total_assets: Optional[float] = None
    total_debt: Optional[float] = None
    total_equity: Optional[float] = None
    cash: Optional[float] = None
    
    # Cash Flow
    operating_cashflow: Optional[float] = None
    free_cashflow: Optional[float] = None
    
    # Ratios
    pe_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    roe: Optional[float] = Field(None, description="Return on Equity %")
    roa: Optional[float] = Field(None, description="Return on Assets %")


class InvestmentRecommendation(BaseModel):
    """Complete investment recommendation from synthesis agent."""
    
    ticker: str
    company_name: Optional[str] = None
    
    # Recommendation
    recommendation: Literal["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
    confidence: float = Field(ge=0, le=1)
    
    # Scores
    overall_score: float = Field(ge=-1, le=1, description="Combined weighted score")
    
    # Category scores
    financial_score: float = Field(ge=-1, le=1)
    technical_score: float = Field(ge=-1, le=1)
    sentiment_score: float = Field(ge=-1, le=1)
    risk_score: float = Field(ge=-1, le=1)
    
    # All agent scores
    agent_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Analysis
    bullish_factors: List[str] = Field(description="Top 3-5 bullish points")
    bearish_factors: List[str] = Field(description="Top 3-5 bearish points")
    risks: List[str] = Field(description="Key investment risks")
    
    # Thesis
    thesis: str = Field(description="Investment thesis (2-4 sentences)")
    
    # Metadata
    analysis_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    analysis_time_seconds: Optional[float] = None
    num_agents: int = Field(default=11)


class RatioResult(BaseModel):
    """Result from financial ratio calculation."""
    
    ratio_name: str
    ticker: str
    value: Optional[float]
    components: Dict = Field(default_factory=dict)
    interpretation: str


class CompanyComparison(BaseModel):
    """Compare multiple companies on a metric."""
    
    metric: str
    companies: List[str]
    values: List[float]
    winner: str
    analysis: str
