"""Financial Modeling Prep API client with caching and retry."""

import logging
import time
import requests
from typing import Dict, Optional, List
from config.settings import get_settings

logger = logging.getLogger(__name__)

FMP_BASE_URL = "https://financialmodelingprep.com/stable"

# Response cache (5-min TTL)
_cache: Dict[str, tuple] = {}  # cache_key -> (timestamp, data)
CACHE_TTL = 300


def _fmp_get(path: str, params: dict = None, max_retries: int = 2) -> Optional[dict | list]:
    """Make authenticated FMP API GET request with caching and retry."""
    settings = get_settings()
    if not settings.fmp_api_key:
        return None

    if params is None:
        params = {}
    params["apikey"] = settings.fmp_api_key

    url = f"{FMP_BASE_URL}/{path}"
    cache_key = f"{url}?{'&'.join(f'{k}={v}' for k, v in sorted(params.items()) if k != 'apikey')}"

    # Check cache
    now = time.time()
    if cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if now - cached_time < CACHE_TTL:
            return cached_data

    # Make request with retry
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                _cache[cache_key] = (now, data)
                return data
            elif resp.status_code == 429:
                delay = 5 * (2 ** attempt)
                logger.warning(f"FMP rate limited, retrying in {delay}s")
                time.sleep(delay)
            else:
                logger.warning(f"FMP {resp.status_code} for {path}: {resp.text[:200]}")
                return None
        except Exception as e:
            logger.warning(f"FMP request error for {path}: {e}")
            if attempt < max_retries:
                time.sleep(2)

    return None


# --- Specific endpoint wrappers ---


def get_quote(symbol: str) -> Optional[Dict]:
    """Get real-time stock quote."""
    data = _fmp_get("quote", {"symbol": symbol})
    return data[0] if data and isinstance(data, list) and len(data) > 0 else None


def get_profile(symbol: str) -> Optional[Dict]:
    """Get company profile."""
    data = _fmp_get("profile", {"symbol": symbol})
    return data[0] if data and isinstance(data, list) and len(data) > 0 else None


def get_ratios(symbol: str) -> Optional[Dict]:
    """Get financial ratios (most recent annual)."""
    data = _fmp_get("ratios", {"symbol": symbol, "period": "annual", "limit": 1})
    return data[0] if data and isinstance(data, list) and len(data) > 0 else None


def get_key_metrics(symbol: str) -> Optional[Dict]:
    """Get key financial metrics."""
    data = _fmp_get("key-metrics", {"symbol": symbol, "period": "annual", "limit": 1})
    return data[0] if data and isinstance(data, list) and len(data) > 0 else None


def get_historical_prices(symbol: str, from_date: str = None, to_date: str = None) -> Optional[List]:
    """Get historical EOD prices."""
    params = {"symbol": symbol}
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    return _fmp_get("historical-price-eod/light", params)


def get_technical_indicator(symbol: str, indicator: str, period: int = 14, timeframe: str = "1day") -> Optional[List]:
    """Get technical indicator (sma, ema, rsi, etc.)."""
    return _fmp_get(f"technical-indicators/{indicator}", {
        "symbol": symbol, "periodLength": period, "timeframe": timeframe
    })


def get_insider_trades_fmp(symbol: str, limit: int = 20) -> Optional[List]:
    """Get latest insider trades for symbol."""
    return _fmp_get("insider-trading/latest", {"symbol": symbol, "limit": limit})


def get_insider_stats(symbol: str) -> Optional[Dict]:
    """Get insider trading statistics."""
    data = _fmp_get("insider-trading/statistics", {"symbol": symbol})
    return data[0] if data and isinstance(data, list) and len(data) > 0 else None


def get_price_target_summary(symbol: str) -> Optional[Dict]:
    """Get analyst price target summary."""
    data = _fmp_get("price-target-summary", {"symbol": symbol})
    return data[0] if data and isinstance(data, list) and len(data) > 0 else None


def get_stock_grades(symbol: str, limit: int = 10) -> Optional[List]:
    """Get analyst stock grades."""
    return _fmp_get("grades", {"symbol": symbol, "limit": limit})


def get_stock_news(symbol: str, limit: int = 10) -> Optional[List]:
    """Get latest news for stock."""
    return _fmp_get("news/stock", {"symbols": symbol, "limit": limit})
