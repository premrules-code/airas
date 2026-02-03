"""Smart query route — auto-detects analysis vs Q&A from user input."""

import re
import logging

from fastapi import APIRouter

from src.api.schemas import QueryRequest, QueryResponse
from src.api.jobs import start_analysis
from src.api.routes.qa import _run_qa

logger = logging.getLogger(__name__)
router = APIRouter()

_BARE_TICKER = re.compile(r"^[A-Za-z]{1,5}$")

# Matches uppercase ticker symbols in text
_TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5})\b")

# Also match potential lowercase tickers (e.g. "aapl", "nvda")
_LOWER_TICKER_PATTERN = re.compile(r"\b([a-z]{1,5})\b")

# Words that signal a question rather than a bare ticker
_QUESTION_SIGNALS = {
    "what", "how", "why", "when", "is", "are", "does", "do", "can", "will",
    "should", "which", "compare", "tell", "explain", "show", "describe",
    "analyze", "analysis", "revenue", "debt", "profit", "margin", "growth",
    "risk", "price", "earnings", "dividend", "outlook", "forecast",
    "valuation", "ratio", "performance", "trend",
}

_TICKER_STOPWORDS = {
    "A", "I", "AN", "AT", "BY", "DO", "GO", "IF", "IN", "IS", "IT",
    "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE", "THE",
    "AND", "FOR", "NOT", "BUT", "ARE", "WAS", "HAS", "HAD", "HOW",
    "ITS", "MAY", "NEW", "NOW", "OLD", "OUR", "OWN", "SAY", "SHE",
    "ALL", "CAN", "HER", "HIM", "HIS", "LET", "OUT", "RUN", "SET",
    "TRY", "USE", "WAY", "WHO", "DID", "GET", "GOT",
    "ANY", "BIG", "DAY", "END", "FAR", "FEW", "VS",
}

# Company name → ticker mapping (lowercase keys for case-insensitive matching)
_COMPANY_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "adobe": "ADBE",
    "salesforce": "CRM",
    "intel": "INTC",
    "amd": "AMD",
    "ibm": "IBM",
    "oracle": "ORCL",
    "cisco": "CSCO",
    "qualcomm": "QCOM",
    "paypal": "PYPL",
    "shopify": "SHOP",
    "uber": "UBER",
    "airbnb": "ABNB",
    "spotify": "SPOT",
    "snap": "SNAP",
    "snapchat": "SNAP",
    "twitter": "TWTR",
    "disney": "DIS",
    "walmart": "WMT",
    "costco": "COST",
    "starbucks": "SBUX",
    "nike": "NKE",
    "boeing": "BA",
    "jpmorgan": "JPM",
    "goldman": "GS",
    "berkshire": "BRK-B",
    "visa": "V",
    "mastercard": "MA",
    "palantir": "PLTR",
    "rivian": "RIVN",
    "coinbase": "COIN",
    "robinhood": "HOOD",
    "snowflake": "SNOW",
    "databricks": "DBX",
    "dropbox": "DBX",
    "crowdstrike": "CRWD",
    "palo alto": "PANW",
    "zoom": "ZM",
    "roku": "ROKU",
    "pinterest": "PINS",
    "moderna": "MRNA",
    "pfizer": "PFE",
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "exxon": "XOM",
    "chevron": "CVX",
    "cocacola": "KO",
    "coca-cola": "KO",
    "coca cola": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
    "procter": "PG",
    "procter & gamble": "PG",
    "broadcom": "AVGO",
    "arm": "ARM",
    "servicenow": "NOW",
}

# Known valid tickers (to confirm lowercase matches like "amd", "ibm")
_KNOWN_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "NFLX", "ADBE", "CRM", "INTC", "AMD", "IBM", "ORCL", "CSCO",
    "QCOM", "PYPL", "SHOP", "UBER", "ABNB", "SPOT", "SNAP", "DIS",
    "WMT", "COST", "SBUX", "NKE", "BA", "JPM", "GS", "V", "MA",
    "PLTR", "RIVN", "COIN", "HOOD", "SNOW", "DBX", "CRWD", "PANW",
    "ZM", "ROKU", "PINS", "MRNA", "PFE", "JNJ", "XOM", "CVX",
    "KO", "PEP", "PG", "AVGO", "ARM", "NOW", "TWTR",
}


def _extract_tickers(text: str) -> list[str]:
    """Extract all ticker symbols from a natural language query.

    Handles:
    - Uppercase tickers: AAPL, NVDA
    - Lowercase tickers: aapl, nvda
    - Company names: Apple, nvidia, Microsoft
    - Multi-word names: Palo Alto, Coca-Cola
    """
    seen = set()
    tickers = []

    def _add(ticker: str):
        t = ticker.upper()
        if t not in seen:
            seen.add(t)
            tickers.append(t)

    # 1) Check multi-word company names first (longest match)
    text_lower = text.lower()
    for name, ticker in sorted(_COMPANY_TO_TICKER.items(), key=lambda x: -len(x[0])):
        if name in text_lower:
            _add(ticker)
            # Remove the matched name to avoid re-matching parts of it
            text_lower = text_lower.replace(name, " ")

    # 2) Match uppercase ticker symbols (AAPL, NVDA)
    for m in _TICKER_PATTERN.findall(text):
        if m not in _TICKER_STOPWORDS:
            _add(m)

    # 3) Match lowercase words that are known tickers (aapl, amd, ibm)
    if not tickers:
        for m in _LOWER_TICKER_PATTERN.findall(text):
            upper = m.upper()
            if upper in _KNOWN_TICKERS and upper not in _TICKER_STOPWORDS:
                _add(upper)

    return tickers


@router.post("/api/query")
def smart_query(req: QueryRequest) -> QueryResponse:
    """Detect whether input is a bare ticker (analysis) or a question (Q&A)."""
    text = req.input.strip()

    # Bare ticker like "AAPL" or "aapl"
    if _BARE_TICKER.match(text):
        upper = text.upper()
        # If it's a known company name, resolve it
        resolved = _COMPANY_TO_TICKER.get(text.lower(), upper)
        job = start_analysis(resolved)
        return QueryResponse(
            mode="analysis",
            job_id=job.job_id,
            status=job.status,
            ticker=job.ticker,
        )

    # Natural language — extract tickers and run Q&A
    tickers = _extract_tickers(text)
    if not tickers:
        return QueryResponse(
            mode="qa",
            answer="Could not identify a ticker symbol in your question. Please include a ticker (e.g., AAPL) or company name (e.g., Apple) for analysis.",
            sources=[],
        )

    # Single ticker with no question signals → full analysis
    if len(tickers) == 1:
        words = set(text.lower().split())
        if not words & _QUESTION_SIGNALS:
            job = start_analysis(tickers[0])
            return QueryResponse(
                mode="analysis",
                job_id=job.job_id,
                status=job.status,
                ticker=job.ticker,
            )

    result = _run_qa(tickers, text)
    return QueryResponse(
        mode="qa",
        ticker=tickers[0],
        answer=result["answer"],
        sources=result["sources"],
    )
