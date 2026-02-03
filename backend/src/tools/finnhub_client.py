"""Finnhub API client for insider trades and company news."""

import logging
import time
import requests
from typing import Dict, Optional, List
from config.settings import get_settings

logger = logging.getLogger(__name__)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Response cache (5-min TTL)
_cache: Dict[str, tuple] = {}
CACHE_TTL = 300


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
            _cache[cache_key] = (now, data)
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

    Args:
        symbol: Stock ticker
        from_date: Start date YYYY-MM-DD (defaults to 7 days ago)
        to_date: End date YYYY-MM-DD (defaults to today)

    Returns list of articles with fields:
    headline, summary, source, datetime, url
    """
    from datetime import datetime, timedelta
    if not from_date:
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")

    return _finnhub_get("company-news", {
        "symbol": symbol.upper(),
        "from": from_date,
        "to": to_date,
    })
