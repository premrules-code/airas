"""Request/response Pydantic models for the API."""

from pydantic import BaseModel, Field
from typing import Optional


# ── Requests ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Unified smart input — backend auto-detects analysis vs Q&A."""
    input: str = Field(..., min_length=1, description="Ticker or natural language question")


class AnalysisRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    query: Optional[str] = None
    rag_level: str = "intermediate"


class QARequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    question: str = Field(..., min_length=1)


class SECDownloadRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    filing_types: list[str] = ["10-K"]
    num_filings: int = 3


# ── Responses ─────────────────────────────────────────────────────────────

class AgentScoreDisplay(BaseModel):
    agent_name: str
    label: str
    icon: str
    description: str
    internal_score: float
    display_score: int
    color: str
    signal: str
    summary: str
    confidence: float
    strengths: list[str] = []
    weaknesses: list[str] = []


class RecommendationDisplay(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    recommendation: str
    confidence: float
    overall_score: int
    overall_color: str
    financial_score: int
    technical_score: int
    sentiment_score: int
    risk_score: int
    bullish_factors: list[str] = []
    bearish_factors: list[str] = []
    risks: list[str] = []
    thesis: str = ""
    num_agents: int = 0


class AnalysisJobResponse(BaseModel):
    job_id: str
    status: str
    ticker: str


class AnalysisResultResponse(BaseModel):
    job_id: str
    status: str
    ticker: str
    agents: list[AgentScoreDisplay] = []
    recommendation: Optional[RecommendationDisplay] = None
    errors: list[str] = []


class QAResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    ticker: str


class QueryResponse(BaseModel):
    mode: str  # "analysis" or "qa"
    # Analysis fields (present when mode == "analysis")
    job_id: Optional[str] = None
    status: Optional[str] = None
    ticker: Optional[str] = None
    # QA fields (present when mode == "qa")
    answer: Optional[str] = None
    sources: Optional[list[dict]] = None


class SECStatusResponse(BaseModel):
    ticker: str
    download_status: str  # "idle" | "running" | "done" | "error"
    index_status: str
    files_count: int = 0
    error: Optional[str] = None


class CompanyInfo(BaseModel):
    ticker: str
    files_count: int
    last_indexed: Optional[str] = None
