"""Finnhub API client — primary data provider (60 calls/min free tier).

Covers: quotes, basic financials, historical candles, insider trades,
company news, analyst recommendations, price targets, earnings.
"""

import logging
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from config.settings import get_settings

logger = logging.getLogger(__name__)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Response cache (30-min TTL — data doesn't change during an analysis run)
_cache: Dict[str, tuple] = {}
CACHE_TTL = 1800


def _finnhub_get(path: str, params: dict = None) -> Optional[dict | list]:
    """Make authenticated Finnhub GET request with caching."""
    settings = get_settings()
    if not settings.finnhub_api_key:
        return None

    if params is None:
        params = {}
    params["token"] = settings.finnhub_api_key

    url = f"{FINNHUB_BASE_URL}/{path}"
    cache_key = f"{url}?{'&'.join(f'{k}={v}' for k, v in sorted(params.items()) if k != 'token')}"

    now = time.time()
    if cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if now - cached_time < CACHE_TTL:
            return cached_data

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            _cache[cache_key] = (time.time(), data)
            return data
        elif resp.status_code == 429:
            logger.warning("Finnhub rate limited")
            return None
        else:
            logger.warning(f"Finnhub {resp.status_code} for {path}: {resp.text[:200]}")
            return None
    except Exception as e:
        logger.warning(f"Finnhub request error for {path}: {e}")
        return None


# --- Quote & Price ---


def get_quote(symbol: str) -> Optional[Dict]:
    """Get real-time quote.

    Returns: {c: current, h: high, l: low, o: open, pc: prev close, t: timestamp}
    """
    data = _finnhub_get("quote", {"symbol": symbol.upper()})
    if data and data.get("c"):  # c=0 means no data
        return data
    return None


def get_candles(symbol: str, resolution: str = "D", from_ts: int = None, to_ts: int = None) -> Optional[Dict]:
    """Get historical OHLCV candles.

    Args:
        symbol: Stock ticker
        resolution: D=daily, W=weekly, M=monthly
        from_ts: Unix timestamp start (defaults to 1 year ago)
        to_ts: Unix timestamp end (defaults to now)

    Returns: {c: [closes], h: [highs], l: [lows], o: [opens], v: [volumes], t: [timestamps], s: "ok"}
    """
    now = int(time.time())
    if not from_ts:
        from_ts = now - 365 * 86400  # 1 year
    if not to_ts:
        to_ts = now

    data = _finnhub_get("stock/candle", {
        "symbol": symbol.upper(),
        "resolution": resolution,
        "from": from_ts,
        "to": to_ts,
    })
    if data and data.get("s") == "ok":
        return data
    return None


# --- Fundamentals ---


def get_basic_financials(symbol: str) -> Optional[Dict]:
    """Get basic financial metrics (PE, PB, debt/equity, margins, etc.).

    Returns dict with 'metric' key containing all ratios and 'series' for quarterly data.
    """
    data = _finnhub_get("stock/metric", {"symbol": symbol.upper(), "metric": "all"})
    if data and data.get("metric"):
        return data
    return None


def get_company_profile(symbol: str) -> Optional[Dict]:
    """Get company profile.

    Returns: {name, ticker, exchange, ipo, marketCapitalization, shareOutstanding, ...}
    """
    data = _finnhub_get("stock/profile2", {"symbol": symbol.upper()})
    if data and data.get("name"):
        return data
    return None


# --- Analyst Data ---


def get_recommendation_trends(symbol: str) -> Optional[List[Dict]]:
    """Get analyst recommendation trends.

    Returns list of monthly summaries: [{buy, hold, sell, strongBuy, strongSell, period}, ...]
    """
    data = _finnhub_get("stock/recommendation", {"symbol": symbol.upper()})
    if data and isinstance(data, list) and len(data) > 0:
        return data
    return None


def get_price_target(symbol: str) -> Optional[Dict]:
    """Get analyst price target consensus.

    Returns: {targetHigh, targetLow, targetMean, targetMedian, lastUpdated}
    """
    data = _finnhub_get("stock/price-target", {"symbol": symbol.upper()})
    if data and data.get("targetMean"):
        return data
    return None


# --- Earnings ---


def get_earnings(symbol: str, limit: int = 4) -> Optional[List[Dict]]:
    """Get historical earnings surprises.

    Returns list: [{actual, estimate, surprise, surprisePercent, period, symbol}, ...]
    """
    data = _finnhub_get("stock/earnings", {"symbol": symbol.upper(), "limit": limit})
    if data and isinstance(data, list) and len(data) > 0:
        return data
    return None


# --- Peers ---


def get_peers(symbol: str) -> Optional[List[str]]:
    """Get list of peer company tickers."""
    data = _finnhub_get("stock/peers", {"symbol": symbol.upper()})
    if data and isinstance(data, list) and len(data) > 0:
        return data
    return None


# --- Existing endpoints ---


def get_insider_transactions(symbol: str) -> Optional[List[Dict]]:
    """Get insider transactions for a symbol.

    Returns list of transactions with fields:
    name, share, change, filingDate, transactionDate, transactionCode, transactionPrice
    """
    data = _finnhub_get("stock/insider-transactions", {"symbol": symbol.upper()})
    if data and isinstance(data, dict):
        return data.get("data", [])
    return None


def get_company_news(symbol: str, from_date: str = None, to_date: str = None) -> Optional[List[Dict]]:
    """Get company news articles.

    Returns list of articles with fields:
    headline, summary, source, datetime, url
    """
    if not from_date:
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")

    return _finnhub_get("company-news", {
        "symbol": symbol.upper(),
        "from": from_date,
        "to": to_date,
    })
