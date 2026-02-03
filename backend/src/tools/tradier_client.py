"""Tradier API client for options data (sandbox is free)."""

import logging
import time
import requests
from typing import Dict, Optional, List
from config.settings import get_settings

logger = logging.getLogger(__name__)

TRADIER_SANDBOX_URL = "https://sandbox.tradier.com/v1"
TRADIER_PRODUCTION_URL = "https://api.tradier.com/v1"

# Response cache (5-min TTL)
_cache: Dict[str, tuple] = {}
CACHE_TTL = 300


def _get_base_url() -> str:
    settings = get_settings()
    return TRADIER_SANDBOX_URL if settings.tradier_sandbox else TRADIER_PRODUCTION_URL


def _tradier_get(path: str, params: dict = None) -> Optional[dict]:
    """Make authenticated Tradier GET request with caching."""
    settings = get_settings()
    if not settings.tradier_api_token:
        return None

    base_url = _get_base_url()
    url = f"{base_url}/{path}"
    cache_key = f"{url}?{'&'.join(f'{k}={v}' for k, v in sorted((params or {}).items()))}"

    now = time.time()
    if cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if now - cached_time < CACHE_TTL:
            return cached_data

    headers = {
        "Authorization": f"Bearer {settings.tradier_api_token}",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            _cache[cache_key] = (now, data)
            return data
        else:
            logger.warning(f"Tradier {resp.status_code} for {path}: {resp.text[:200]}")
            return None
    except Exception as e:
        logger.warning(f"Tradier request error for {path}: {e}")
        return None


def get_option_expirations(symbol: str) -> Optional[List[str]]:
    """Get available option expiration dates for a symbol."""
    data = _tradier_get("markets/options/expirations", {"symbol": symbol.upper()})
    if data and "expirations" in data:
        exp = data["expirations"]
        if exp and "date" in exp:
            dates = exp["date"]
            return dates if isinstance(dates, list) else [dates]
    return None


def get_option_chain(symbol: str, expiration: str) -> Optional[List[Dict]]:
    """Get options chain for a symbol and expiration date.

    Returns list of option contracts with fields:
    symbol, option_type (call/put), strike, bid, ask, last, volume,
    open_interest, implied_volatility, greeks, etc.
    """
    data = _tradier_get("markets/options/chains", {
        "symbol": symbol.upper(),
        "expiration": expiration,
        "greeks": "true",
    })
    if data and "options" in data:
        opts = data["options"]
        if opts and "option" in opts:
            options = opts["option"]
            return options if isinstance(options, list) else [options]
    return None
