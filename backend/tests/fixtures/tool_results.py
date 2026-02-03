"""Mock tool result data for testing."""

STOCK_PRICE_RESULT = {
    "ticker": "AAPL",
    "date": "2024-01-15",
    "current_price": 185.50,
    "open": 184.20,
    "high": 186.80,
    "low": 183.90,
    "volume": 55000000,
    "market_cap": 2900000000000,
    "pe_ratio": 28.5,
    "52_week_high": 199.62,
    "52_week_low": 143.90,
    "source": "fmp",
}

RATIO_RESULTS = {
    "pe_ratio": {
        "bullish": {
            "ratio_name": "pe_ratio",
            "ticker": "AAPL",
            "value": 12.0,
            "components": {"period": "FY2023", "source": "fmp"},
            "interpretation": "Fairly valued",
        },
        "bearish": {
            "ratio_name": "pe_ratio",
            "ticker": "AAPL",
            "value": 55.0,
            "components": {"period": "FY2023", "source": "fmp"},
            "interpretation": "Overvalued",
        },
    },
    "roe": {
        "bullish": {
            "ratio_name": "roe",
            "ticker": "AAPL",
            "value": 35.0,
            "components": {"period": "FY2023", "source": "fmp"},
            "interpretation": "Excellent",
        },
        "bearish": {
            "ratio_name": "roe",
            "ticker": "AAPL",
            "value": 3.0,
            "components": {"period": "FY2023", "source": "fmp"},
            "interpretation": "Fair",
        },
    },
    "profit_margin": {
        "bullish": {
            "ratio_name": "profit_margin",
            "ticker": "AAPL",
            "value": 30.0,
            "components": {"period": "FY2023", "source": "fmp"},
            "interpretation": "Excellent",
        },
        "bearish": {
            "ratio_name": "profit_margin",
            "ticker": "AAPL",
            "value": 2.0,
            "components": {"period": "FY2023", "source": "fmp"},
            "interpretation": "Fair",
        },
    },
    "debt_to_equity": {
        "bullish": {
            "ratio_name": "debt_to_equity",
            "ticker": "AAPL",
            "value": 0.5,
            "components": {"period": "FY2023", "source": "fmp"},
            "interpretation": "Low leverage",
        },
        "bearish": {
            "ratio_name": "debt_to_equity",
            "ticker": "AAPL",
            "value": 5.0,
            "components": {"period": "FY2023", "source": "fmp"},
            "interpretation": "High leverage",
        },
    },
}

TECHNICAL_INDICATORS_RESULT = {
    "ticker": "AAPL",
    "current_price": 185.50,
    "sma_20": 183.00,
    "sma_50": 180.00,
    "sma_200": 170.00,
    "rsi_14": 58.5,
    "macd": {"macd": 1.25, "signal": 0.80, "histogram": 0.45},
    "bollinger_bands": {"upper": 190.00, "middle": 183.00, "lower": 176.00},
    "price_change_1m": 3.5,
    "price_change_3m": 8.2,
    "volume_avg_20d": 52000000,
    "trend": "bullish",
    "source": "fmp+ta",
}

INSIDER_TRADES_RESULT = {
    "ticker": "AAPL",
    "total_trades": 15,
    "buys": 8,
    "sells": 7,
    "buy_sell_ratio": 1.14,
    "net_signal": "bullish",
    "recent_trades": [
        {
            "insider": "Tim Cook",
            "type": "Purchase",
            "shares": "50000",
            "value": "9275000",
            "date": "2024-01-10",
        },
        {
            "insider": "Luca Maestri",
            "type": "Sale",
            "shares": "20000",
            "value": "3710000",
            "date": "2024-01-08",
        },
    ],
    "source": "fmp",
}

OPTIONS_RESULT = {
    "ticker": "AAPL",
    "available": False,
    "summary": "Options data unavailable",
    "signal": "neutral",
}

ANALYST_RATINGS_RESULT = {
    "ticker": "AAPL",
    "recommendation": "buy",
    "num_analysts": 35,
    "target_price_mean": 210.0,
    "target_price_high": 250.0,
    "target_price_low": 170.0,
    "target_price_median": 215.0,
    "current_price": 185.50,
    "upside_pct": 13.21,
    "source": "fmp",
}

SOCIAL_SENTIMENT_RESULT = {
    "ticker": "AAPL",
    "sources": {
        "stocktwits": {
            "total_messages": 30,
            "bullish": 18,
            "bearish": 12,
            "bull_ratio": 0.60,
            "signal": "bullish",
        },
        "reddit": {"error": "Reddit credentials not configured"},
        "news": {
            "total_articles": 10,
            "avg_sentiment": 0.15,
            "signal": "bullish",
            "source": "fmp",
        },
    },
    "aggregate_signal": "bullish",
    "confidence": 0.67,
}

COMPARE_RESULT = {
    "metric": "pe_ratio",
    "companies": ["MSFT", "AAPL", "GOOGL"],
    "values": [32.5, 28.5, 25.0],
    "winner": "MSFT",
    "analysis": "MSFT leads with 32.5",
}
