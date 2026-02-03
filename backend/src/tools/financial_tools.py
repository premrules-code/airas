# backend/src/tools/financial_tools.py
"""
Financial tools for agents.
Can be called via Claude function calling.

Strategy: Try FMP (Financial Modeling Prep) first for all tools except options.
Fall back to yfinance if FMP key is not configured or FMP returns None.
Options data stays yfinance-only (FMP doesn't offer it).
"""

import logging
import time
from typing import Dict, List, Optional
import yfinance as yf
import requests
import numpy as np
from datetime import datetime, timedelta

from src.models.structured_outputs import RatioResult, CompanyComparison
from src.tools import fmp_client
from src.tools import finnhub_client
from src.tools import tradier_client

logger = logging.getLogger(__name__)

# --- yfinance Ticker cache + retry (kept for fallback + options) ---
_ticker_cache: Dict[str, yf.Ticker] = {}
_info_cache: Dict[str, Dict] = {}
_info_cache_time: Dict[str, float] = {}
INFO_CACHE_TTL = 1800  # 30 minutes (data doesn't change during an analysis)


def _get_ticker(symbol: str) -> yf.Ticker:
    """Get a cached yfinance Ticker object."""
    if symbol not in _ticker_cache:
        _ticker_cache[symbol] = yf.Ticker(symbol)
    return _ticker_cache[symbol]


def _get_info_with_retry(symbol: str, max_retries: int = 3) -> Dict:
    """Get ticker info with caching and retry on 429."""
    now = time.time()
    if symbol in _info_cache and (now - _info_cache_time.get(symbol, 0)) < INFO_CACHE_TTL:
        return _info_cache[symbol]

    stock = _get_ticker(symbol)
    for attempt in range(max_retries):
        try:
            info = stock.info
            if info and info.get("symbol"):
                _info_cache[symbol] = info
                _info_cache_time[symbol] = now
                return info
        except Exception:
            pass
        if attempt < max_retries - 1:
            delay = 10 * (2 ** attempt)
            logger.info(f"yfinance retry for {symbol} in {delay}s (attempt {attempt + 1})")
            time.sleep(delay)

    return _info_cache.get(symbol, {})


# ============================================================================
# Tool 1: get_stock_price — FMP quote → yfinance fallback
# ============================================================================


def get_stock_price(ticker: str, date: Optional[str] = None) -> Dict:
    """Get current or historical stock price."""
    logger.info(f"Getting price for {ticker}")

    # Historical date requests always use yfinance (FMP historical needs different handling)
    if date:
        return _yfinance_get_stock_price(ticker, date)

    # Try FMP first for current price
    quote = fmp_client.get_quote(ticker)
    if quote:
        return {
            "ticker": ticker,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "current_price": quote.get("price"),
            "open": quote.get("open"),
            "high": quote.get("dayHigh"),
            "low": quote.get("dayLow"),
            "volume": quote.get("volume"),
            "market_cap": quote.get("marketCap"),
            "pe_ratio": quote.get("pe"),
            "52_week_high": quote.get("yearHigh"),
            "52_week_low": quote.get("yearLow"),
            "source": "fmp",
        }

    # Fallback to yfinance
    return _yfinance_get_stock_price(ticker, date)


def _yfinance_get_stock_price(ticker: str, date: Optional[str] = None) -> Dict:
    """yfinance fallback for get_stock_price."""
    try:
        stock = _get_ticker(ticker)
        if date:
            end = datetime.strptime(date, "%Y-%m-%d")
            start = end - timedelta(days=1)
            hist = stock.history(start=start, end=end + timedelta(days=1))
            if not hist.empty:
                row = hist.iloc[0]
                return {
                    "ticker": ticker,
                    "date": date,
                    "open": round(float(row['Open']), 2),
                    "high": round(float(row['High']), 2),
                    "low": round(float(row['Low']), 2),
                    "close": round(float(row['Close']), 2),
                    "volume": int(row['Volume']),
                    "source": "yfinance",
                }
        info = _get_info_with_retry(ticker)
        return {
            "ticker": ticker,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "current_price": info.get('currentPrice'),
            "open": info.get('open'),
            "high": info.get('dayHigh'),
            "low": info.get('dayLow'),
            "volume": info.get('volume'),
            "market_cap": info.get('marketCap'),
            "pe_ratio": info.get('trailingPE'),
            "52_week_high": info.get('fiftyTwoWeekHigh'),
            "52_week_low": info.get('fiftyTwoWeekLow'),
            "source": "yfinance",
        }
    except Exception as e:
        logger.error(f"get_stock_price error: {e}")
        return {"ticker": ticker, "error": str(e)}


# ============================================================================
# Tool 2: calculate_financial_ratio — FMP ratios → yfinance fallback
# ============================================================================


def _interpret_ratio(ratio_type: str, value: Optional[float]) -> str:
    """Return interpretation string for a ratio value."""
    if value is None:
        return "N/A"
    if ratio_type == "pe_ratio":
        return 'Overvalued' if value > 25 else 'Fairly valued'
    elif ratio_type == "roe":
        return 'Excellent' if value > 15 else 'Good' if value > 10 else 'Fair'
    elif ratio_type == "debt_to_equity":
        return 'High leverage' if value > 2 else 'Moderate' if value > 1 else 'Low leverage'
    elif ratio_type == "profit_margin":
        return 'Excellent' if value > 20 else 'Good' if value > 10 else 'Fair'
    elif ratio_type == "current_ratio":
        return 'Strong liquidity' if value > 2 else 'Adequate' if value > 1 else 'Weak'
    return "N/A"


def calculate_financial_ratio(
    ratio_type: str,
    ticker: str,
    period: str = "FY2023"
) -> Dict:
    """Calculate financial ratios using FMP or yfinance."""
    logger.info(f"Calculating {ratio_type} for {ticker}")

    # Try FMP first
    value = _fmp_get_ratio(ratio_type, ticker)
    if value is not None:
        interpretation = _interpret_ratio(ratio_type, value)
        result = RatioResult(
            ratio_name=ratio_type,
            ticker=ticker,
            value=value,
            components={"period": period, "source": "fmp"},
            interpretation=interpretation,
        )
        logger.info(f"  Result (FMP): {value} ({interpretation})")
        return result.model_dump()

    # Fallback to yfinance
    return _yfinance_calculate_ratio(ratio_type, ticker, period)


def _fmp_get_ratio(ratio_type: str, ticker: str) -> Optional[float]:
    """Extract a specific ratio from FMP data."""
    ratios = fmp_client.get_ratios(ticker)
    if not ratios:
        return None

    # FMP field names in /stable/ratios response
    ratio_mapping = {
        "pe_ratio": "priceToEarningsRatio",
        "debt_to_equity": "debtToEquityRatio",
        "profit_margin": "netProfitMargin",
        "current_ratio": "currentRatio",
    }
    # ROE is in key-metrics, not ratios
    metrics_mapping = {
        "roe": "returnOnEquity",
    }

    if ratio_type in ratio_mapping:
        raw = ratios.get(ratio_mapping[ratio_type])
    elif ratio_type in metrics_mapping:
        metrics = fmp_client.get_key_metrics(ticker)
        raw = metrics.get(metrics_mapping[ratio_type]) if metrics else None
    else:
        return None

    if raw is None:
        return None

    # FMP returns roe/margins as decimals (0.25 = 25%), convert to percentage
    if ratio_type in ("roe", "profit_margin"):
        return round(float(raw) * 100, 2)

    return round(float(raw), 2)


def _yfinance_calculate_ratio(ratio_type: str, ticker: str, period: str) -> Dict:
    """yfinance fallback for calculate_financial_ratio."""
    try:
        info = _get_info_with_retry(ticker)
        if ratio_type == "pe_ratio":
            price = info.get('currentPrice', 0)
            eps = info.get('trailingEps', 1)
            value = round(price / eps, 2) if eps else None
        elif ratio_type == "roe":
            roe = info.get('returnOnEquity')
            value = round(roe * 100, 2) if roe else None
        elif ratio_type == "debt_to_equity":
            de = info.get('debtToEquity')
            value = round(de, 2) if de else None
        elif ratio_type == "profit_margin":
            margin = info.get('profitMargins')
            value = round(margin * 100, 2) if margin else None
        elif ratio_type == "current_ratio":
            ca = info.get('totalCurrentAssets', 0)
            cl = info.get('totalCurrentLiabilities', 1)
            value = round(ca / cl, 2) if cl else None
        else:
            raise ValueError(f"Unknown ratio: {ratio_type}")

        interpretation = _interpret_ratio(ratio_type, value)
        result = RatioResult(
            ratio_name=ratio_type,
            ticker=ticker,
            value=value,
            components={"period": period, "source": "yfinance"},
            interpretation=interpretation,
        )
        logger.info(f"  Result (yfinance): {value} ({interpretation})")
        return result.model_dump()
    except Exception as e:
        logger.error(f"  Error: {e}")
        return {"ratio_name": ratio_type, "ticker": ticker, "value": None, "interpretation": f"Error: {e}"}


# ============================================================================
# Tool 3: compare_companies — FMP quote per ticker → yfinance fallback
# ============================================================================


def compare_companies(
    tickers: List[str],
    metric: str,
    period: str = "FY2023"
) -> Dict:
    """Compare metric across multiple companies."""
    logger.info(f"Comparing {metric} for {', '.join(tickers)}")
    try:
        results = []
        for ticker in tickers:
            value = _get_comparison_value_fmp(ticker, metric)
            if value is None:
                value = _get_comparison_value_yfinance(ticker, metric)
            results.append({'ticker': ticker, 'value': round(value or 0, 2)})

        results.sort(key=lambda x: x['value'], reverse=True)
        comparison = CompanyComparison(
            metric=metric,
            companies=[r['ticker'] for r in results],
            values=[r['value'] for r in results],
            winner=results[0]['ticker'] if results else "None",
            analysis=f"{results[0]['ticker']} leads with {results[0]['value']}",
        )
        logger.info(f"  Winner: {comparison.winner}")
        return comparison.model_dump()
    except Exception as e:
        logger.error(f"  Error: {e}")
        return {"metric": metric, "companies": tickers, "values": [], "winner": "Error", "analysis": str(e)}


def _get_comparison_value_fmp(ticker: str, metric: str) -> Optional[float]:
    """Get a comparison metric value from FMP."""
    quote = fmp_client.get_quote(ticker)
    if not quote:
        return None

    if metric == "market_cap":
        mc = quote.get("marketCap")
        return mc / 1e9 if mc else None
    elif metric == "pe_ratio":
        return quote.get("pe")
    elif metric == "revenue":
        # Quote doesn't have revenue; try ratios/key-metrics
        metrics = fmp_client.get_key_metrics(ticker)
        if metrics and metrics.get("revenuePerShare"):
            # Approximate: not a direct match, fall through to yfinance
            return None
        return None
    elif metric == "profit_margin":
        ratios = fmp_client.get_ratios(ticker)
        if ratios and ratios.get("netProfitMargin") is not None:
            return round(float(ratios["netProfitMargin"]) * 100, 2)
        return None
    elif metric == "roe":
        # ROE is in key-metrics, not ratios
        metrics = fmp_client.get_key_metrics(ticker)
        if metrics and metrics.get("returnOnEquity") is not None:
            return round(float(metrics["returnOnEquity"]) * 100, 2)
        return None

    return None


def _get_comparison_value_yfinance(ticker: str, metric: str) -> Optional[float]:
    """Get a comparison metric value from yfinance."""
    info = _get_info_with_retry(ticker)
    if metric == "revenue":
        return info.get('totalRevenue', 0) / 1e9
    elif metric == "profit_margin":
        return (info.get('profitMargins', 0) or 0) * 100
    elif metric == "roe":
        return (info.get('returnOnEquity', 0) or 0) * 100
    elif metric == "pe_ratio":
        return info.get('trailingPE', 0) or 0
    elif metric == "market_cap":
        return info.get('marketCap', 0) / 1e9
    return 0


# ============================================================================
# Technical indicator helpers (kept for yfinance fallback calculations)
# ============================================================================


def _calculate_rsi(close_prices, period: int = 14) -> Optional[float]:
    """Calculate RSI from close prices."""
    if len(close_prices) < period + 1:
        return None
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(float(val), 2) if not np.isnan(val) else None


def _calculate_macd(close_prices) -> Dict:
    """Calculate MACD indicator."""
    if len(close_prices) < 26:
        return {"macd": None, "signal": None, "histogram": None}
    ema_12 = close_prices.ewm(span=12, adjust=False).mean()
    ema_26 = close_prices.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": round(float(macd_line.iloc[-1]), 4),
        "signal": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
    }


def _calculate_bollinger(close_prices, period: int = 20) -> Dict:
    """Calculate Bollinger Bands."""
    if len(close_prices) < period:
        return {"upper": None, "middle": None, "lower": None}
    sma = close_prices.rolling(window=period).mean()
    std = close_prices.rolling(window=period).std()
    return {
        "upper": round(float((sma + 2 * std).iloc[-1]), 2),
        "middle": round(float(sma.iloc[-1]), 2),
        "lower": round(float((sma - 2 * std).iloc[-1]), 2),
    }


# ============================================================================
# Tool 4: get_technical_indicators — FMP indicators → yfinance fallback
# ============================================================================


def get_technical_indicators(ticker: str, period: str = "6mo") -> Dict:
    """Calculate technical indicators from FMP or price history."""
    logger.info(f"Getting technical indicators for {ticker}")

    # Try FMP first
    result = _fmp_get_technical_indicators(ticker)
    if result:
        return result

    # Fallback to yfinance
    return _yfinance_get_technical_indicators(ticker, period)


def _fmp_get_technical_indicators(ticker: str) -> Optional[Dict]:
    """Calculate technical indicators from FMP historical prices using ta library.

    Uses FMP Starter-tier historical prices (no Premium needed) and computes
    SMA, RSI, MACD, Bollinger Bands locally.
    """
    import pandas as pd

    hist = fmp_client.get_historical_prices(ticker)
    if not hist or len(hist) < 26:
        return None

    # FMP returns newest first, reverse for chronological order
    prices = list(reversed(hist))
    close = pd.Series(
        [float(p.get("price", p.get("close", 0))) for p in prices],
        name="Close",
    )
    current_price = float(close.iloc[-1])

    # Try ta library for indicators; fall back to manual calculation
    try:
        from ta.trend import SMAIndicator, MACD as TAmacd
        from ta.momentum import RSIIndicator
        from ta.volatility import BollingerBands

        sma_20_ind = SMAIndicator(close, window=20)
        sma_50_ind = SMAIndicator(close, window=50)
        rsi_ind = RSIIndicator(close, window=14)
        macd_ind = TAmacd(close)
        bb_ind = BollingerBands(close, window=20)

        sma_20_val = round(float(sma_20_ind.sma_indicator().iloc[-1]), 2) if len(close) >= 20 else None
        sma_50_val = round(float(sma_50_ind.sma_indicator().iloc[-1]), 2) if len(close) >= 50 else None
        sma_200_val = None
        if len(close) >= 200:
            sma_200_ind = SMAIndicator(close, window=200)
            sma_200_val = round(float(sma_200_ind.sma_indicator().iloc[-1]), 2)

        rsi_val = round(float(rsi_ind.rsi().iloc[-1]), 2)

        macd = {
            "macd": round(float(macd_ind.macd().iloc[-1]), 4),
            "signal": round(float(macd_ind.macd_signal().iloc[-1]), 4),
            "histogram": round(float(macd_ind.macd_diff().iloc[-1]), 4),
        }

        bollinger = {
            "upper": round(float(bb_ind.bollinger_hband().iloc[-1]), 2),
            "middle": round(float(bb_ind.bollinger_mavg().iloc[-1]), 2),
            "lower": round(float(bb_ind.bollinger_lband().iloc[-1]), 2),
        }

    except ImportError:
        logger.info("ta library not installed, using manual calculations")
        sma_20_val = round(float(close.rolling(20).mean().iloc[-1]), 2) if len(close) >= 20 else None
        sma_50_val = round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None
        sma_200_val = round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None
        rsi_val = _calculate_rsi(close, 14)
        macd = _calculate_macd(close)
        bollinger = _calculate_bollinger(close)

    price_1m = None
    price_3m = None
    if len(close) >= 21:
        price_1m = round(float((close.iloc[-1] / close.iloc[-21] - 1) * 100), 2)
    if len(close) >= 63:
        price_3m = round(float((close.iloc[-1] / close.iloc[-63] - 1) * 100), 2)

    volumes = [int(p.get("volume", 0)) for p in prices[-20:]]
    vol_avg = int(sum(volumes) / max(len(volumes), 1))

    return {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "sma_20": sma_20_val,
        "sma_50": sma_50_val,
        "sma_200": sma_200_val,
        "rsi_14": rsi_val,
        "macd": macd,
        "bollinger_bands": bollinger,
        "price_change_1m": price_1m,
        "price_change_3m": price_3m,
        "volume_avg_20d": vol_avg,
        "trend": "bullish" if current_price > (sma_50_val or current_price) else "bearish",
        "source": "fmp+ta",
    }


def _yfinance_get_technical_indicators(ticker: str, period: str = "6mo") -> Dict:
    """yfinance fallback for get_technical_indicators."""
    try:
        stock = _get_ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return {"ticker": ticker, "error": "No price history available"}

        close = hist['Close']
        sma_200 = None
        if len(close) >= 200:
            sma_200 = round(float(close.rolling(200).mean().iloc[-1]), 2)

        sma_50_val = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        current = float(close.iloc[-1])

        price_1m = None
        price_3m = None
        if len(close) >= 21:
            price_1m = round(float((close.iloc[-1] / close.iloc[-21] - 1) * 100), 2)
        if len(close) >= 63:
            price_3m = round(float((close.iloc[-1] / close.iloc[-63] - 1) * 100), 2)

        return {
            "ticker": ticker,
            "current_price": round(current, 2),
            "sma_20": round(float(close.rolling(20).mean().iloc[-1]), 2) if len(close) >= 20 else None,
            "sma_50": round(float(sma_50_val), 2) if len(close) >= 50 else None,
            "sma_200": sma_200,
            "rsi_14": _calculate_rsi(close, 14),
            "macd": _calculate_macd(close),
            "bollinger_bands": _calculate_bollinger(close),
            "price_change_1m": price_1m,
            "price_change_3m": price_3m,
            "volume_avg_20d": int(hist['Volume'].tail(20).mean()) if len(hist) >= 20 else int(hist['Volume'].mean()),
            "trend": "bullish" if current > float(sma_50_val) else "bearish",
            "source": "yfinance",
        }
    except Exception as e:
        logger.error(f"Technical indicators error: {e}")
        return {"ticker": ticker, "error": str(e)}


# ============================================================================
# Tool 5: get_insider_trades — FMP → yfinance fallback
# ============================================================================


def get_insider_trades(ticker: str) -> Dict:
    """Get recent insider trading activity."""
    logger.info(f"Getting insider trades for {ticker}")

    # Try FMP first
    result = _fmp_get_insider_trades(ticker)
    if result:
        return result

    # Try Finnhub
    result = _finnhub_get_insider_trades(ticker)
    if result:
        return result

    # Fallback to yfinance
    return _yfinance_get_insider_trades(ticker)


def _fmp_get_insider_trades(ticker: str) -> Optional[Dict]:
    """Get insider trades from FMP."""
    trades_data = fmp_client.get_insider_trades_fmp(ticker, limit=50)
    if not trades_data:
        return None

    # FMP endpoint may return trades for other symbols — filter client-side
    filtered = [t for t in trades_data if (t.get("symbol") or "").upper() == ticker.upper()]
    if not filtered:
        return None  # No trades for this ticker, fall back to yfinance

    buys = 0
    sells = 0
    trades = []
    for t in filtered[:20]:
        tx_type = (t.get("transactionType") or "").lower()
        acq = (t.get("acquisitionOrDisposition") or "").upper()
        if "purchase" in tx_type or "buy" in tx_type or acq == "A":
            buys += 1
        elif "sale" in tx_type or "sell" in tx_type or acq == "D":
            sells += 1

        trades.append({
            "insider": t.get("reportingName", ""),
            "type": t.get("transactionType", "") or t.get("acquisitionOrDisposition", ""),
            "shares": str(t.get("securitiesTransacted", "")),
            "value": str(t.get("price", "")),
            "date": str(t.get("filingDate", "")),
        })

    return {
        "ticker": ticker,
        "total_trades": len(filtered[:20]),
        "buys": buys,
        "sells": sells,
        "buy_sell_ratio": round(buys / max(sells, 1), 2),
        "net_signal": "bullish" if buys > sells else "bearish" if sells > buys else "neutral",
        "recent_trades": trades[:10],
        "source": "fmp",
    }


def _finnhub_get_insider_trades(ticker: str) -> Optional[Dict]:
    """Get insider trades from Finnhub."""
    transactions = finnhub_client.get_insider_transactions(ticker)
    if not transactions:
        return None

    buys = 0
    sells = 0
    trades = []
    for t in transactions[:20]:
        # Finnhub transactionCode: P=Purchase, S=Sale, A=Grant/Award, M=Exercise
        code = (t.get("transactionCode") or "").upper()
        if code == "P":
            buys += 1
        elif code == "S":
            sells += 1

        trades.append({
            "insider": t.get("name", ""),
            "type": "Purchase" if code == "P" else "Sale" if code == "S" else code,
            "shares": str(t.get("change", "")),
            "value": str(t.get("transactionPrice", "")),
            "date": str(t.get("transactionDate", t.get("filingDate", ""))),
        })

    if not trades:
        return None

    return {
        "ticker": ticker,
        "total_trades": len(transactions[:20]),
        "buys": buys,
        "sells": sells,
        "buy_sell_ratio": round(buys / max(sells, 1), 2),
        "net_signal": "bullish" if buys > sells else "bearish" if sells > buys else "neutral",
        "recent_trades": trades[:10],
        "source": "finnhub",
    }


def _yfinance_get_insider_trades(ticker: str) -> Dict:
    """yfinance fallback for get_insider_trades."""
    try:
        stock = _get_ticker(ticker)
        insiders = stock.insider_transactions

        if insiders is None or insiders.empty:
            return {"ticker": ticker, "trades": [], "signal": "neutral", "summary": "No recent insider data"}

        recent = insiders.head(20)
        buys = len(recent[recent['Text'].str.contains('Purchase|Buy|Acquisition', case=False, na=False)])
        sells = len(recent[recent['Text'].str.contains('Sale|Sell|Disposition', case=False, na=False)])

        trades = []
        for _, row in recent.head(10).iterrows():
            trades.append({
                "insider": str(row.get('Insider Trading', row.get('Insider', ''))),
                "type": str(row.get('Text', '')),
                "shares": str(row.get('Shares', '')),
                "value": str(row.get('Value', '')),
                "date": str(row.get('Start Date', row.get('Date', ''))),
            })

        return {
            "ticker": ticker,
            "total_trades": len(recent),
            "buys": buys,
            "sells": sells,
            "buy_sell_ratio": round(buys / max(sells, 1), 2),
            "net_signal": "bullish" if buys > sells else "bearish" if sells > buys else "neutral",
            "recent_trades": trades,
            "source": "yfinance",
        }
    except Exception as e:
        logger.error(f"Insider trades error: {e}")
        return {"ticker": ticker, "error": str(e), "signal": "neutral"}


# ============================================================================
# Tool 6: get_options_data — yfinance ONLY (FMP doesn't have options)
# ============================================================================


def get_options_data(ticker: str) -> Dict:
    """Get options chain summary and implied volatility.

    Currently disabled — no reliable free provider available.
    Returns neutral placeholder so the options agent scores with low confidence.
    """
    logger.info(f"Options data skipped for {ticker} (no reliable provider)")
    return {
        "ticker": ticker,
        "available": False,
        "summary": "Options data unavailable — no reliable provider configured",
        "signal": "neutral",
    }


def _tradier_get_options_data(ticker: str) -> Optional[Dict]:
    """Get options data from Tradier."""
    expirations = tradier_client.get_option_expirations(ticker)
    if not expirations:
        return None

    # Use nearest expiration
    chain = tradier_client.get_option_chain(ticker, expirations[0])
    if not chain:
        return None

    calls = [o for o in chain if o.get("option_type") == "call"]
    puts = [o for o in chain if o.get("option_type") == "put"]

    calls_vol = sum(int(c.get("volume") or 0) for c in calls)
    puts_vol = sum(int(p.get("volume") or 0) for p in puts)

    call_ivs = [float(c.get("greeks", {}).get("mid_iv") or 0) for c in calls if c.get("greeks", {}).get("mid_iv")]
    put_ivs = [float(p.get("greeks", {}).get("mid_iv") or 0) for p in puts if p.get("greeks", {}).get("mid_iv")]

    avg_iv_calls = round(sum(call_ivs) / max(len(call_ivs), 1) * 100, 2) if call_ivs else None
    avg_iv_puts = round(sum(put_ivs) / max(len(put_ivs), 1) * 100, 2) if put_ivs else None

    pcr = round(puts_vol / max(calls_vol, 1), 2)

    if puts_vol > calls_vol * 1.5:
        signal = "bearish"
    elif calls_vol > puts_vol * 1.5:
        signal = "bullish"
    else:
        signal = "neutral"

    return {
        "ticker": ticker,
        "nearest_expiration": expirations[0],
        "num_expirations": len(expirations),
        "calls_volume": calls_vol,
        "puts_volume": puts_vol,
        "put_call_ratio": pcr,
        "avg_implied_vol_calls": avg_iv_calls,
        "avg_implied_vol_puts": avg_iv_puts,
        "signal": signal,
        "source": "tradier",
    }


def _yfinance_get_options_data(ticker: str, max_retries: int = 3) -> Dict:
    """yfinance fallback for get_options_data with dedicated retry.

    Since other tools now use FMP/Finnhub, yfinance rate limit pressure
    is much lower — options is the only yfinance-exclusive tool.
    """
    for attempt in range(max_retries):
        try:
            stock = _get_ticker(ticker)
            expirations = stock.options

            if not expirations:
                return {"ticker": ticker, "available": False, "summary": "No options data"}

            chain = stock.option_chain(expirations[0])
            calls, puts = chain.calls, chain.puts

            calls_vol = int(calls['volume'].sum()) if 'volume' in calls.columns and not calls['volume'].isna().all() else 0
            puts_vol = int(puts['volume'].sum()) if 'volume' in puts.columns and not puts['volume'].isna().all() else 0

            avg_iv_calls = round(float(calls['impliedVolatility'].mean() * 100), 2) if 'impliedVolatility' in calls.columns else None
            avg_iv_puts = round(float(puts['impliedVolatility'].mean() * 100), 2) if 'impliedVolatility' in puts.columns else None

            pcr = round(puts_vol / max(calls_vol, 1), 2)

            if puts_vol > calls_vol * 1.5:
                signal = "bearish"
            elif calls_vol > puts_vol * 1.5:
                signal = "bullish"
            else:
                signal = "neutral"

            return {
                "ticker": ticker,
                "nearest_expiration": expirations[0],
                "num_expirations": len(expirations),
                "calls_volume": calls_vol,
                "puts_volume": puts_vol,
                "put_call_ratio": pcr,
                "avg_implied_vol_calls": avg_iv_calls,
                "avg_implied_vol_puts": avg_iv_puts,
                "signal": signal,
                "source": "yfinance",
            }
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                delay = 15 * (2 ** attempt)
                logger.info(f"yfinance options 429, retry in {delay}s (attempt {attempt + 1})")
                time.sleep(delay)
            else:
                logger.error(f"Options data error: {e}")
                return {"ticker": ticker, "error": str(e), "available": False}

    return {"ticker": ticker, "error": "Max retries exceeded", "available": False}


# ============================================================================
# Tool 7: get_analyst_ratings — FMP → yfinance fallback
# ============================================================================


def _get_recommendation_trend(stock) -> List[Dict]:
    """Get analyst recommendation trend over last 4 months (yfinance)."""
    try:
        trend = stock.recommendations
        if trend is None or trend.empty:
            return []
        recent = trend.tail(4)
        return [
            {
                "period": str(row.get('period', idx)),
                "strongBuy": int(row.get('strongBuy', 0)),
                "buy": int(row.get('buy', 0)),
                "hold": int(row.get('hold', 0)),
                "sell": int(row.get('sell', 0)),
                "strongSell": int(row.get('strongSell', 0)),
            }
            for idx, row in recent.iterrows()
        ]
    except Exception:
        return []


def get_analyst_ratings(ticker: str) -> Dict:
    """Get analyst consensus and price targets."""
    logger.info(f"Getting analyst ratings for {ticker}")

    # Try FMP first
    result = _fmp_get_analyst_ratings(ticker)
    if result:
        return result

    # Fallback to yfinance
    return _yfinance_get_analyst_ratings(ticker)


def _fmp_get_analyst_ratings(ticker: str) -> Optional[Dict]:
    """Get analyst ratings from FMP."""
    summary = fmp_client.get_price_target_summary(ticker)
    if not summary:
        return None

    current_price = None
    quote = fmp_client.get_quote(ticker)
    if quote:
        current_price = quote.get("price")

    # FMP field names: lastQuarterAvgPriceTarget, lastYearAvgPriceTarget, allTimeAvgPriceTarget
    target_mean = (
        summary.get("lastQuarterAvgPriceTarget")
        or summary.get("lastYearAvgPriceTarget")
        or summary.get("allTimeAvgPriceTarget")
    )
    num_analysts = (
        summary.get("lastQuarterCount")
        or summary.get("lastYearCount")
        or summary.get("allTimeCount")
        or 0
    )

    upside = None
    if target_mean and current_price:
        upside = round(((target_mean / max(current_price, 1)) - 1) * 100, 2)

    # Get recent grades for trend data
    grades = fmp_client.get_stock_grades(ticker, limit=10) or []
    grade_trend = []
    for g in grades[:5]:
        grade_trend.append({
            "date": g.get("date", ""),
            "company": g.get("gradingCompany", ""),
            "action": g.get("newGrade", ""),
            "previous": g.get("previousGrade", ""),
        })

    # Map consensus to recommendation key based on upside
    rec_key = "none"
    if target_mean and current_price:
        pct = (target_mean / max(current_price, 1) - 1) * 100
        if pct > 20:
            rec_key = "strong_buy"
        elif pct > 10:
            rec_key = "buy"
        elif pct > -5:
            rec_key = "hold"
        elif pct > -15:
            rec_key = "sell"
        else:
            rec_key = "strong_sell"

    return {
        "ticker": ticker,
        "recommendation": rec_key,
        "num_analysts": num_analysts,
        "target_price_mean": target_mean,
        "target_price_high": summary.get("lastMonthAvgPriceTarget"),
        "target_price_low": summary.get("allTimeAvgPriceTarget"),
        "target_price_median": target_mean,
        "current_price": current_price,
        "upside_pct": upside,
        "recent_grades": grade_trend,
        "source": "fmp",
    }


def _yfinance_get_analyst_ratings(ticker: str) -> Dict:
    """yfinance fallback for get_analyst_ratings."""
    try:
        stock = _get_ticker(ticker)
        info = _get_info_with_retry(ticker)

        current_price = info.get('currentPrice', 0)
        target_mean = info.get('targetMeanPrice')
        upside = None
        if target_mean and current_price:
            upside = round(((target_mean / max(current_price, 1)) - 1) * 100, 2)

        return {
            "ticker": ticker,
            "recommendation": info.get('recommendationKey', 'none'),
            "num_analysts": info.get('numberOfAnalystOpinions', 0),
            "target_price_mean": target_mean,
            "target_price_high": info.get('targetHighPrice'),
            "target_price_low": info.get('targetLowPrice'),
            "target_price_median": info.get('targetMedianPrice'),
            "current_price": current_price,
            "upside_pct": upside,
            "recommendation_trend": _get_recommendation_trend(stock),
            "source": "yfinance",
        }
    except Exception as e:
        logger.error(f"Analyst ratings error: {e}")
        return {"ticker": ticker, "error": str(e)}


# ============================================================================
# Tool 8: get_social_sentiment — FMP news → yfinance news fallback
# ============================================================================


def _vader_score(text: str) -> float:
    """Return compound sentiment score: -1 (negative) to +1 (positive)."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text)["compound"]
    except Exception:
        return 0.0


def get_social_sentiment(ticker: str) -> Dict:
    """Aggregate social sentiment from StockTwits, Reddit, and news."""
    logger.info(f"Getting social sentiment for {ticker}")
    results = {"ticker": ticker, "sources": {}}

    # Source 1: StockTwits (free, no auth)
    try:
        resp = requests.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            messages = data.get("messages", [])
            bullish = sum(
                1 for m in messages
                if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bullish"
            )
            bearish = sum(
                1 for m in messages
                if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bearish"
            )
            total = bullish + bearish
            results["sources"]["stocktwits"] = {
                "total_messages": len(messages),
                "bullish": bullish,
                "bearish": bearish,
                "bull_ratio": round(bullish / max(total, 1), 2),
                "signal": "bullish" if bullish > bearish else "bearish" if bearish > bullish else "neutral",
                "sample_messages": [m.get("body", "")[:200] for m in messages[:5]],
            }
        else:
            results["sources"]["stocktwits"] = {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        results["sources"]["stocktwits"] = {"error": str(e)}

    # Source 2: Reddit (requires PRAW credentials in .env)
    try:
        import praw
        from config.settings import get_settings
        settings = get_settings()
        if settings.reddit_client_id and settings.reddit_client_secret:
            reddit = praw.Reddit(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent="airas-v3-sentiment",
            )
            posts = []
            for sub_name in ["wallstreetbets", "stocks", "investing"]:
                try:
                    subreddit = reddit.subreddit(sub_name)
                    for post in subreddit.search(ticker, limit=10, time_filter="week"):
                        sentiment = _vader_score(post.title)
                        posts.append({
                            "title": post.title[:200],
                            "subreddit": sub_name,
                            "score": post.score,
                            "sentiment": sentiment,
                        })
                except Exception:
                    continue

            avg_sentiment = sum(p["sentiment"] for p in posts) / max(len(posts), 1)
            results["sources"]["reddit"] = {
                "total_posts": len(posts),
                "avg_sentiment": round(avg_sentiment, 3),
                "signal": "bullish" if avg_sentiment > 0.1 else "bearish" if avg_sentiment < -0.1 else "neutral",
                "top_posts": sorted(posts, key=lambda p: p["score"], reverse=True)[:5],
            }
        else:
            results["sources"]["reddit"] = {"error": "Reddit credentials not configured"}
    except ImportError:
        results["sources"]["reddit"] = {"error": "PRAW not installed"}
    except Exception as e:
        results["sources"]["reddit"] = {"error": str(e)}

    # Source 3: News — try FMP → Finnhub → yfinance
    news_result = _fmp_get_news_sentiment(ticker)
    if not news_result:
        news_result = _finnhub_get_news_sentiment(ticker)
    if not news_result:
        news_result = _yfinance_get_news_sentiment(ticker)
    results["sources"]["news"] = news_result

    # Aggregate
    signals = []
    for source_data in results["sources"].values():
        if isinstance(source_data, dict) and "signal" in source_data:
            signals.append(source_data["signal"])

    bullish_count = signals.count("bullish")
    bearish_count = signals.count("bearish")
    results["aggregate_signal"] = (
        "bullish" if bullish_count > bearish_count
        else "bearish" if bearish_count > bullish_count
        else "neutral"
    )
    results["confidence"] = round(max(bullish_count, bearish_count) / max(len(signals), 1), 2)

    return results


def _fmp_get_news_sentiment(ticker: str) -> Optional[Dict]:
    """Get news sentiment from FMP."""
    news = fmp_client.get_stock_news(ticker, limit=10)
    if not news:
        return None

    headlines = []
    for item in news[:10]:
        title = item.get("title", "")
        sentiment = _vader_score(title)
        headlines.append({
            "title": title,
            "publisher": item.get("site", item.get("publishedDate", "")),
            "sentiment": sentiment,
        })

    if not headlines:
        return None

    avg_news = sum(h["sentiment"] for h in headlines) / max(len(headlines), 1)
    return {
        "total_articles": len(headlines),
        "avg_sentiment": round(avg_news, 3),
        "signal": "bullish" if avg_news > 0.1 else "bearish" if avg_news < -0.1 else "neutral",
        "headlines": headlines[:5],
        "source": "fmp",
    }


def _finnhub_get_news_sentiment(ticker: str) -> Optional[Dict]:
    """Get news sentiment from Finnhub."""
    articles = finnhub_client.get_company_news(ticker)
    if not articles:
        return None

    headlines = []
    for item in articles[:10]:
        title = item.get("headline", "")
        sentiment = _vader_score(title)
        headlines.append({
            "title": title,
            "publisher": item.get("source", ""),
            "sentiment": sentiment,
        })

    if not headlines:
        return None

    avg_news = sum(h["sentiment"] for h in headlines) / max(len(headlines), 1)
    return {
        "total_articles": len(headlines),
        "avg_sentiment": round(avg_news, 3),
        "signal": "bullish" if avg_news > 0.1 else "bearish" if avg_news < -0.1 else "neutral",
        "headlines": headlines[:5],
        "source": "finnhub",
    }


def _yfinance_get_news_sentiment(ticker: str) -> Dict:
    """yfinance fallback for news sentiment."""
    try:
        stock = _get_ticker(ticker)
        news = stock.news or []
        headlines = []
        for item in news[:10]:
            sentiment = _vader_score(item.get("title", ""))
            headlines.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "sentiment": sentiment,
            })
        avg_news = sum(h["sentiment"] for h in headlines) / max(len(headlines), 1)
        return {
            "total_articles": len(headlines),
            "avg_sentiment": round(avg_news, 3),
            "signal": "bullish" if avg_news > 0.1 else "bearish" if avg_news < -0.1 else "neutral",
            "headlines": headlines[:5],
            "source": "yfinance",
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Cache pre-warming — call once before agents start so they all read from cache
# ============================================================================


def warm_cache(ticker: str):
    """Best-effort pre-fetch of common FMP and yfinance data for a ticker.

    Each call tries once (no retries on rate limit) — whatever succeeds
    goes into cache. Agents will fetch anything that was missed, with
    per-key lock dedup preventing concurrent duplicate requests.
    """
    logger.info(f"Pre-warming cache for {ticker}...")
    calls = [
        ("quote", "quote", {"symbol": ticker}),
        ("ratios", "ratios", {"symbol": ticker, "period": "annual", "limit": 1}),
        ("key-metrics", "key-metrics", {"symbol": ticker, "period": "annual", "limit": 1}),
        ("historical-prices", "historical-price-eod/light", {"symbol": ticker}),
        ("insider-trades", "insider-trading/latest", {"symbol": ticker, "limit": 20}),
        ("price-target", "price-target-summary", {"symbol": ticker}),
        ("grades", "grades", {"symbol": ticker, "limit": 10}),
        ("news", "news/stock", {"symbols": ticker, "limit": 10}),
    ]
    cached = 0
    for name, path, params in calls:
        try:
            result = fmp_client._fmp_get(path, params, _warm=True)
            if result is not None:
                cached += 1
                logger.info(f"  Cached {name} for {ticker}")
        except Exception as e:
            logger.debug(f"  Warm skip {name}/{ticker}: {e}")

    # Also warm yfinance info (used as fallback)
    try:
        _get_info_with_retry(ticker, max_retries=1)
        cached += 1
        logger.info(f"  Cached yfinance info for {ticker}")
    except Exception:
        pass

    logger.info(f"Cache pre-warm done for {ticker}: {cached}/{len(calls)+1} endpoints cached")


# ============================================================================
# Tool registry for Claude function calling
# ============================================================================

FINANCIAL_TOOLS = [
    {
        "name": "calculate_financial_ratio",
        "description": "Calculate financial ratios (P/E, ROE, D/E, profit margin, current ratio)",
        "input_schema": {
            "type": "object",
            "properties": {
                "ratio_type": {
                    "type": "string",
                    "enum": ["pe_ratio", "roe", "debt_to_equity", "profit_margin", "current_ratio"]
                },
                "ticker": {"type": "string"},
                "period": {"type": "string", "default": "FY2023"}
            },
            "required": ["ratio_type", "ticker"]
        }
    },
    {
        "name": "compare_companies",
        "description": "Compare metric across multiple companies",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}},
                "metric": {
                    "type": "string",
                    "enum": ["revenue", "profit_margin", "roe", "pe_ratio", "market_cap"]
                },
                "period": {"type": "string", "default": "FY2023"}
            },
            "required": ["tickers", "metric"]
        }
    },
    {
        "name": "get_stock_price",
        "description": "Get current or historical stock price",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "date": {"type": "string", "description": "Optional YYYY-MM-DD"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_technical_indicators",
        "description": "Calculate technical indicators (SMA, RSI, MACD, Bollinger Bands) from price history",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "period": {"type": "string", "default": "6mo", "description": "Price history period (1mo, 3mo, 6mo, 1y, 2y)"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_insider_trades",
        "description": "Get recent insider trading activity (buys, sells, transaction patterns)",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_options_data",
        "description": "Get options chain summary (put/call ratio, implied volatility, volume)",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_analyst_ratings",
        "description": "Get analyst consensus rating, price targets, and recommendation trends",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_social_sentiment",
        "description": "Get aggregated social sentiment from StockTwits, Reddit, and news headlines",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"]
        }
    },
]


def execute_tool(tool_name: str, tool_input: Dict) -> Dict:
    """Execute a tool by name."""
    tools = {
        "calculate_financial_ratio": calculate_financial_ratio,
        "compare_companies": compare_companies,
        "get_stock_price": get_stock_price,
        "get_technical_indicators": get_technical_indicators,
        "get_insider_trades": get_insider_trades,
        "get_options_data": get_options_data,
        "get_analyst_ratings": get_analyst_ratings,
        "get_social_sentiment": get_social_sentiment,
    }
    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")
    return tools[tool_name](**tool_input)
