"""Per-agent bullish/bearish Claude responses and tool results for evals."""

import json
from tests.fixtures.claude_responses import make_claude_response, make_tool_use_response


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _agent_json(agent_name, ticker, score, confidence, summary, metrics=None):
    return json.dumps({
        "agent_name": agent_name,
        "ticker": ticker,
        "score": score,
        "confidence": confidence,
        "metrics": metrics or {},
        "strengths": ["Strength A", "Strength B"],
        "weaknesses": ["Weakness A"],
        "summary": summary,
        "sources": ["test data"],
    })


# ---------------------------------------------------------------------------
# 1. Financial Analyst  (tools: calculate_financial_ratio, compare_companies, get_stock_price)
# ---------------------------------------------------------------------------

FINANCIAL_ANALYST = {
    "bullish": {
        "response": make_claude_response(_agent_json(
            "financial_analyst", "AAPL", 0.65, 0.85,
            "Strong margins, low PE, excellent ROE.",
            {"pe_ratio": 12.0, "roe": 35.0, "profit_margin": 30.0, "debt_to_equity": 0.5},
        )),
        "tool_use": make_tool_use_response("calculate_financial_ratio", {"ratio_type": "pe_ratio", "ticker": "AAPL"}),
        "tool_result": {
            "ratio_name": "pe_ratio", "ticker": "AAPL", "value": 12.0,
            "components": {"period": "FY2023", "source": "fmp"}, "interpretation": "Fairly valued",
        },
    },
    "bearish": {
        "response": make_claude_response(_agent_json(
            "financial_analyst", "AAPL", -0.55, 0.80,
            "Weak margins, high PE, poor ROE.",
            {"pe_ratio": 55.0, "roe": 3.0, "profit_margin": 2.0, "debt_to_equity": 5.0},
        )),
        "tool_use": make_tool_use_response("calculate_financial_ratio", {"ratio_type": "pe_ratio", "ticker": "AAPL"}),
        "tool_result": {
            "ratio_name": "pe_ratio", "ticker": "AAPL", "value": 55.0,
            "components": {"period": "FY2023", "source": "fmp"}, "interpretation": "Overvalued",
        },
    },
}

# ---------------------------------------------------------------------------
# 2. News Sentiment  (tools: get_stock_price)
# ---------------------------------------------------------------------------

NEWS_SENTIMENT = {
    "bullish": {
        "response": make_claude_response(_agent_json(
            "news_sentiment", "AAPL", 0.45, 0.70,
            "Optimistic management tone with specific revenue guidance.",
            {"tone": "optimistic", "guidance_specificity": "high", "risk_severity": "low"},
        )),
        "tool_use": make_tool_use_response("get_stock_price", {"ticker": "AAPL"}),
        "tool_result": {
            "ticker": "AAPL", "current_price": 185.50, "source": "fmp",
        },
    },
    "bearish": {
        "response": make_claude_response(_agent_json(
            "news_sentiment", "AAPL", -0.40, 0.65,
            "Cautious management tone, expanded risk factors, vague guidance.",
            {"tone": "cautious", "guidance_specificity": "low", "risk_severity": "high"},
        )),
        "tool_use": make_tool_use_response("get_stock_price", {"ticker": "AAPL"}),
        "tool_result": {
            "ticker": "AAPL", "current_price": 145.00, "source": "fmp",
        },
    },
}

# ---------------------------------------------------------------------------
# 3. Technical Analyst  (tools: get_technical_indicators, get_stock_price)
# ---------------------------------------------------------------------------

TECHNICAL_ANALYST = {
    "bullish": {
        "response": make_claude_response(_agent_json(
            "technical_analyst", "AAPL", 0.50, 0.80,
            "Bullish setup: price above all SMAs, MACD bullish crossover.",
            {"rsi_14": 58, "sma_50": 180.0, "sma_200": 170.0, "trend": "bullish"},
        )),
        "tool_use": make_tool_use_response("get_technical_indicators", {"ticker": "AAPL"}),
        "tool_result": {
            "ticker": "AAPL", "current_price": 185.50, "sma_20": 183.0,
            "sma_50": 180.0, "sma_200": 170.0, "rsi_14": 58.5,
            "macd": {"macd": 1.25, "signal": 0.80, "histogram": 0.45},
            "bollinger_bands": {"upper": 190.0, "middle": 183.0, "lower": 176.0},
            "trend": "bullish", "source": "fmp+ta",
        },
    },
    "bearish": {
        "response": make_claude_response(_agent_json(
            "technical_analyst", "AAPL", -0.50, 0.75,
            "Bearish: price below SMA200, RSI oversold, MACD bearish.",
            {"rsi_14": 28, "sma_50": 190.0, "sma_200": 200.0, "trend": "bearish"},
        )),
        "tool_use": make_tool_use_response("get_technical_indicators", {"ticker": "AAPL"}),
        "tool_result": {
            "ticker": "AAPL", "current_price": 155.00, "sma_20": 162.0,
            "sma_50": 170.0, "sma_200": 180.0, "rsi_14": 28.0,
            "macd": {"macd": -2.10, "signal": -0.90, "histogram": -1.20},
            "bollinger_bands": {"upper": 175.0, "middle": 162.0, "lower": 149.0},
            "trend": "bearish", "source": "fmp+ta",
        },
    },
}

# ---------------------------------------------------------------------------
# 4. Risk Assessment  (tools: calculate_financial_ratio, get_stock_price, get_technical_indicators)
# ---------------------------------------------------------------------------

RISK_ASSESSMENT = {
    "bullish": {
        "response": make_claude_response(_agent_json(
            "risk_assessment", "AAPL", 0.40, 0.75,
            "Low risk profile: manageable leverage, moderate volatility.",
            {"beta": 1.1, "debt_to_equity": 0.8, "volatility": "low"},
        )),
        "tool_use": make_tool_use_response("calculate_financial_ratio", {"ratio_type": "debt_to_equity", "ticker": "AAPL"}),
        "tool_result": {
            "ratio_name": "debt_to_equity", "ticker": "AAPL", "value": 0.8,
            "components": {"period": "FY2023", "source": "fmp"}, "interpretation": "Low leverage",
        },
    },
    "bearish": {
        "response": make_claude_response(_agent_json(
            "risk_assessment", "AAPL", -0.45, 0.70,
            "High risk: excessive leverage, high volatility, extreme valuation.",
            {"beta": 2.1, "debt_to_equity": 5.0, "volatility": "high"},
        )),
        "tool_use": make_tool_use_response("calculate_financial_ratio", {"ratio_type": "debt_to_equity", "ticker": "AAPL"}),
        "tool_result": {
            "ratio_name": "debt_to_equity", "ticker": "AAPL", "value": 5.0,
            "components": {"period": "FY2023", "source": "fmp"}, "interpretation": "High leverage",
        },
    },
}

# ---------------------------------------------------------------------------
# 5. Competitive Analysis  (tools: compare_companies, get_stock_price)
# ---------------------------------------------------------------------------

COMPETITIVE_ANALYSIS = {
    "bullish": {
        "response": make_claude_response(_agent_json(
            "competitive_analysis", "AAPL", 0.60, 0.80,
            "Wide moat: ecosystem lock-in, brand, switching costs.",
            {"moat_type": "wide", "market_position": "leader"},
        )),
        "tool_use": make_tool_use_response("compare_companies", {"tickers": ["AAPL", "MSFT", "GOOGL"], "metric": "market_cap"}),
        "tool_result": {
            "metric": "market_cap", "companies": ["AAPL", "MSFT", "GOOGL"],
            "values": [2900.0, 2800.0, 1700.0], "winner": "AAPL",
            "analysis": "AAPL leads with 2900.0",
        },
    },
    "bearish": {
        "response": make_claude_response(_agent_json(
            "competitive_analysis", "AAPL", -0.35, 0.65,
            "Narrowing moat: market maturing, losing share to competitors.",
            {"moat_type": "narrow", "market_position": "declining"},
        )),
        "tool_use": make_tool_use_response("compare_companies", {"tickers": ["AAPL", "MSFT", "GOOGL"], "metric": "revenue"}),
        "tool_result": {
            "metric": "revenue", "companies": ["MSFT", "GOOGL", "AAPL"],
            "values": [220.0, 300.0, 180.0], "winner": "GOOGL",
            "analysis": "GOOGL leads with 300.0",
        },
    },
}

# ---------------------------------------------------------------------------
# 6. Insider Activity  (tools: get_insider_trades, get_stock_price)
# ---------------------------------------------------------------------------

INSIDER_ACTIVITY = {
    "bullish": {
        "response": make_claude_response(_agent_json(
            "insider_activity", "AAPL", 0.45, 0.70,
            "Cluster buying by multiple C-suite executives.",
            {"buys": 12, "sells": 3, "buy_sell_ratio": 4.0, "net_signal": "bullish"},
        )),
        "tool_use": make_tool_use_response("get_insider_trades", {"ticker": "AAPL"}),
        "tool_result": {
            "ticker": "AAPL", "total_trades": 15, "buys": 12, "sells": 3,
            "buy_sell_ratio": 4.0, "net_signal": "bullish",
            "recent_trades": [
                {"insider": "Tim Cook", "type": "Purchase", "shares": "100000",
                 "value": "18550000", "date": "2024-01-10"},
                {"insider": "Jeff Williams", "type": "Purchase", "shares": "50000",
                 "value": "9275000", "date": "2024-01-09"},
            ],
            "source": "fmp",
        },
    },
    "bearish": {
        "response": make_claude_response(_agent_json(
            "insider_activity", "AAPL", -0.40, 0.65,
            "Heavy selling by CEO and CFO, unusual volume.",
            {"buys": 1, "sells": 14, "buy_sell_ratio": 0.07, "net_signal": "bearish"},
        )),
        "tool_use": make_tool_use_response("get_insider_trades", {"ticker": "AAPL"}),
        "tool_result": {
            "ticker": "AAPL", "total_trades": 15, "buys": 1, "sells": 14,
            "buy_sell_ratio": 0.07, "net_signal": "bearish",
            "recent_trades": [
                {"insider": "Tim Cook", "type": "Sale", "shares": "200000",
                 "value": "37100000", "date": "2024-01-10"},
            ],
            "source": "fmp",
        },
    },
}

# ---------------------------------------------------------------------------
# 7. Options Analysis  (tools: get_options_data, get_stock_price)
# ---------------------------------------------------------------------------

OPTIONS_ANALYSIS = {
    "bullish": {
        "response": make_claude_response(_agent_json(
            "options_analysis", "AAPL", 0.30, 0.55,
            "Low PCR, call volume surge, moderate IV.",
            {"put_call_ratio": 0.55, "avg_iv_calls": 22.0, "signal": "bullish"},
        )),
        "tool_use": make_tool_use_response("get_options_data", {"ticker": "AAPL"}),
        "tool_result": {
            "ticker": "AAPL", "nearest_expiration": "2024-01-19",
            "calls_volume": 850000, "puts_volume": 467500,
            "put_call_ratio": 0.55, "avg_implied_vol_calls": 22.0,
            "avg_implied_vol_puts": 25.0, "signal": "bullish", "source": "tradier",
        },
    },
    "bearish": {
        "response": make_claude_response(_agent_json(
            "options_analysis", "AAPL", -0.25, 0.50,
            "High PCR, elevated put IV, protective put buying.",
            {"put_call_ratio": 1.8, "avg_iv_puts": 45.0, "signal": "bearish"},
        )),
        "tool_use": make_tool_use_response("get_options_data", {"ticker": "AAPL"}),
        "tool_result": {
            "ticker": "AAPL", "nearest_expiration": "2024-01-19",
            "calls_volume": 300000, "puts_volume": 540000,
            "put_call_ratio": 1.8, "avg_implied_vol_calls": 30.0,
            "avg_implied_vol_puts": 45.0, "signal": "bearish", "source": "tradier",
        },
    },
}

# ---------------------------------------------------------------------------
# 8. Social Sentiment  (tools: get_social_sentiment, get_stock_price)
# ---------------------------------------------------------------------------

SOCIAL_SENTIMENT = {
    "bullish": {
        "response": make_claude_response(_agent_json(
            "social_sentiment", "AAPL", 0.35, 0.50,
            "Positive sentiment across StockTwits and news.",
            {"stocktwits_bull_ratio": 0.72, "news_avg_sentiment": 0.25, "aggregate": "bullish"},
        )),
        "tool_use": make_tool_use_response("get_social_sentiment", {"ticker": "AAPL"}),
        "tool_result": {
            "ticker": "AAPL",
            "sources": {
                "stocktwits": {"total_messages": 50, "bullish": 36, "bearish": 14,
                               "bull_ratio": 0.72, "signal": "bullish"},
                "reddit": {"error": "not configured"},
                "news": {"total_articles": 10, "avg_sentiment": 0.25,
                         "signal": "bullish", "source": "fmp"},
            },
            "aggregate_signal": "bullish", "confidence": 1.0,
        },
    },
    "bearish": {
        "response": make_claude_response(_agent_json(
            "social_sentiment", "AAPL", -0.30, 0.45,
            "Negative sentiment: bearish StockTwits, negative news.",
            {"stocktwits_bull_ratio": 0.25, "news_avg_sentiment": -0.30, "aggregate": "bearish"},
        )),
        "tool_use": make_tool_use_response("get_social_sentiment", {"ticker": "AAPL"}),
        "tool_result": {
            "ticker": "AAPL",
            "sources": {
                "stocktwits": {"total_messages": 50, "bullish": 12, "bearish": 38,
                               "bull_ratio": 0.24, "signal": "bearish"},
                "reddit": {"error": "not configured"},
                "news": {"total_articles": 10, "avg_sentiment": -0.30,
                         "signal": "bearish", "source": "fmp"},
            },
            "aggregate_signal": "bearish", "confidence": 1.0,
        },
    },
}

# ---------------------------------------------------------------------------
# 9. Earnings Analysis  (tools: calculate_financial_ratio, compare_companies)
# ---------------------------------------------------------------------------

EARNINGS_ANALYSIS = {
    "bullish": {
        "response": make_claude_response(_agent_json(
            "earnings_analysis", "AAPL", 0.50, 0.80,
            "Growing EPS, diversified revenue, strong operating leverage.",
            {"eps_growth_yoy": 15.0, "services_growth": 18.0, "margin_trend": "expanding"},
        )),
        "tool_use": make_tool_use_response("calculate_financial_ratio", {"ratio_type": "profit_margin", "ticker": "AAPL"}),
        "tool_result": {
            "ratio_name": "profit_margin", "ticker": "AAPL", "value": 28.0,
            "components": {"period": "FY2023", "source": "fmp"}, "interpretation": "Excellent",
        },
    },
    "bearish": {
        "response": make_claude_response(_agent_json(
            "earnings_analysis", "AAPL", -0.45, 0.75,
            "Declining EPS, shrinking margins, one-time items.",
            {"eps_growth_yoy": -12.0, "margin_trend": "contracting"},
        )),
        "tool_use": make_tool_use_response("calculate_financial_ratio", {"ratio_type": "profit_margin", "ticker": "AAPL"}),
        "tool_result": {
            "ratio_name": "profit_margin", "ticker": "AAPL", "value": 4.0,
            "components": {"period": "FY2023", "source": "fmp"}, "interpretation": "Fair",
        },
    },
}

# ---------------------------------------------------------------------------
# 10. Analyst Ratings  (tools: get_analyst_ratings, get_stock_price)
# ---------------------------------------------------------------------------

ANALYST_RATINGS = {
    "bullish": {
        "response": make_claude_response(_agent_json(
            "analyst_ratings", "AAPL", 0.55, 0.85,
            "Strong Buy consensus, 18% upside to mean target.",
            {"consensus": "strong_buy", "target_upside_pct": 18.0, "num_analysts": 40},
        )),
        "tool_use": make_tool_use_response("get_analyst_ratings", {"ticker": "AAPL"}),
        "tool_result": {
            "ticker": "AAPL", "recommendation": "strong_buy", "num_analysts": 40,
            "target_price_mean": 220.0, "target_price_high": 260.0,
            "target_price_low": 190.0, "target_price_median": 225.0,
            "current_price": 185.50, "upside_pct": 18.6, "source": "fmp",
        },
    },
    "bearish": {
        "response": make_claude_response(_agent_json(
            "analyst_ratings", "AAPL", -0.45, 0.80,
            "Sell consensus, 15% downside to mean target, recent downgrades.",
            {"consensus": "sell", "target_upside_pct": -15.0, "num_analysts": 30},
        )),
        "tool_use": make_tool_use_response("get_analyst_ratings", {"ticker": "AAPL"}),
        "tool_result": {
            "ticker": "AAPL", "recommendation": "sell", "num_analysts": 30,
            "target_price_mean": 158.0, "target_price_high": 180.0,
            "target_price_low": 130.0, "target_price_median": 155.0,
            "current_price": 185.50, "upside_pct": -14.8, "source": "fmp",
        },
    },
}


# ---------------------------------------------------------------------------
# Registry: agent_name → fixture data (for parametrized tests)
# ---------------------------------------------------------------------------

AGENT_FIXTURES = {
    "financial_analyst": FINANCIAL_ANALYST,
    "news_sentiment": NEWS_SENTIMENT,
    "technical_analyst": TECHNICAL_ANALYST,
    "risk_assessment": RISK_ASSESSMENT,
    "competitive_analysis": COMPETITIVE_ANALYSIS,
    "insider_activity": INSIDER_ACTIVITY,
    "options_analysis": OPTIONS_ANALYSIS,
    "social_sentiment": SOCIAL_SENTIMENT,
    "earnings_analysis": EARNINGS_ANALYSIS,
    "analyst_ratings": ANALYST_RATINGS,
}
