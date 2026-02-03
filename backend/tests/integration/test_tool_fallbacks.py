"""Integration tests for tool fallback chains with mocked providers."""

from unittest.mock import patch, MagicMock

import pytest

from src.tools.financial_tools import (
    get_stock_price,
    calculate_financial_ratio,
    get_insider_trades,
    get_technical_indicators,
)

pytestmark = pytest.mark.integration


class TestGetStockPriceFallback:
    """get_stock_price: FMP → yfinance fallback chain."""

    def test_fmp_success(self):
        """FMP returns data → use it."""
        with patch("src.tools.financial_tools.fmp_client.get_quote") as mock_fmp:
            mock_fmp.return_value = {
                "price": 185.50,
                "open": 184.20,
                "dayHigh": 186.80,
                "dayLow": 183.90,
                "volume": 55000000,
                "marketCap": 2900000000000,
                "pe": 28.5,
                "yearHigh": 199.62,
                "yearLow": 143.90,
            }
            result = get_stock_price("AAPL")

        assert result["source"] == "fmp"
        assert result["current_price"] == 185.50

    def test_fmp_none_falls_back_to_yfinance(self):
        """FMP returns None → yfinance fallback."""
        mock_info = {
            "symbol": "AAPL",
            "currentPrice": 185.50,
            "open": 184.20,
            "dayHigh": 186.80,
            "dayLow": 183.90,
            "volume": 55000000,
            "marketCap": 2900000000000,
            "trailingPE": 28.5,
            "fiftyTwoWeekHigh": 199.62,
            "fiftyTwoWeekLow": 143.90,
        }
        with patch("src.tools.financial_tools.fmp_client.get_quote", return_value=None), \
             patch("src.tools.financial_tools._get_info_with_retry", return_value=mock_info):
            result = get_stock_price("AAPL")

        assert result["source"] == "yfinance"
        assert result["current_price"] == 185.50


class TestCalculateFinancialRatioFallback:
    """calculate_financial_ratio: FMP → yfinance fallback chain."""

    def test_fmp_ratio_success(self):
        """FMP returns ratio data → use it."""
        with patch("src.tools.financial_tools.fmp_client.get_ratios") as mock_ratios:
            mock_ratios.return_value = {"priceToEarningsRatio": 28.5}
            result = calculate_financial_ratio("pe_ratio", "AAPL")

        assert result["components"]["source"] == "fmp"
        assert result["value"] == 28.5

    def test_fmp_none_falls_back_to_yfinance(self):
        """FMP returns None → yfinance."""
        mock_info = {
            "symbol": "AAPL",
            "currentPrice": 185.50,
            "trailingEps": 6.5,
        }
        with patch("src.tools.financial_tools.fmp_client.get_ratios", return_value=None), \
             patch("src.tools.financial_tools._get_info_with_retry", return_value=mock_info):
            result = calculate_financial_ratio("pe_ratio", "AAPL")

        assert result["components"]["source"] == "yfinance"
        assert result["value"] == round(185.50 / 6.5, 2)


class TestGetInsiderTradesFallback:
    """get_insider_trades: FMP → Finnhub → yfinance fallback chain."""

    def test_fmp_success(self):
        """FMP returns insider trades → use it."""
        with patch("src.tools.financial_tools.fmp_client.get_insider_trades_fmp") as mock_fmp:
            mock_fmp.return_value = [
                {
                    "symbol": "AAPL",
                    "reportingName": "Tim Cook",
                    "transactionType": "Purchase",
                    "securitiesTransacted": 50000,
                    "price": 185.0,
                    "filingDate": "2024-01-10",
                },
            ]
            result = get_insider_trades("AAPL")

        assert result["source"] == "fmp"
        assert result["buys"] == 1

    def test_fmp_none_finnhub_none_yfinance_fallback(self):
        """FMP None → Finnhub None → yfinance fallback."""
        import pandas as pd

        mock_df = pd.DataFrame({
            "Text": ["Purchase at price", "Sale of shares"],
            "Insider Trading": ["Tim Cook", "Luca Maestri"],
            "Shares": [50000, 20000],
            "Value": [9275000, 3710000],
            "Start Date": ["2024-01-10", "2024-01-08"],
        })

        mock_stock = MagicMock()
        mock_stock.insider_transactions = mock_df

        with patch("src.tools.financial_tools.fmp_client.get_insider_trades_fmp", return_value=None), \
             patch("src.tools.financial_tools.finnhub_client.get_insider_transactions", return_value=None), \
             patch("src.tools.financial_tools._get_ticker", return_value=mock_stock):
            result = get_insider_trades("AAPL")

        assert result["source"] == "yfinance"
        assert result["buys"] == 1
        assert result["sells"] == 1

    def test_fmp_none_finnhub_success(self):
        """FMP None → Finnhub returns data → use Finnhub."""
        with patch("src.tools.financial_tools.fmp_client.get_insider_trades_fmp", return_value=None), \
             patch("src.tools.financial_tools.finnhub_client.get_insider_transactions") as mock_fh:
            mock_fh.return_value = [
                {
                    "name": "Tim Cook",
                    "transactionCode": "P",
                    "change": 50000,
                    "transactionPrice": 185.0,
                    "transactionDate": "2024-01-10",
                },
            ]
            result = get_insider_trades("AAPL")

        assert result["source"] == "finnhub"
        assert result["buys"] == 1


class TestGetTechnicalIndicatorsFallback:
    """get_technical_indicators: FMP → yfinance fallback chain."""

    def test_fmp_historical_prices_used(self):
        """FMP returns historical prices → computed with ta/manual."""
        import pandas as pd
        import numpy as np

        # Generate 100 days of fake prices
        prices = [{"price": float(150 + i * 0.5), "volume": 50000000} for i in range(100)]

        with patch("src.tools.financial_tools.fmp_client.get_historical_prices") as mock_fmp:
            mock_fmp.return_value = list(reversed(prices))  # FMP returns newest first
            result = get_technical_indicators("AAPL")

        assert result["source"] == "fmp+ta"
        assert result["ticker"] == "AAPL"
        assert result["current_price"] is not None
        assert result["sma_20"] is not None
        assert result["sma_50"] is not None
        assert result["rsi_14"] is not None

    def test_fmp_none_falls_back_to_yfinance(self):
        """FMP returns None → yfinance fallback."""
        import pandas as pd
        import numpy as np

        # Create mock yfinance history
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        close = pd.Series(np.linspace(150, 185, 100), index=dates, name="Close")
        volume = pd.Series([50000000] * 100, index=dates, name="Volume")
        hist = pd.DataFrame({"Close": close, "Volume": volume})

        mock_stock = MagicMock()
        mock_stock.history.return_value = hist

        with patch("src.tools.financial_tools.fmp_client.get_historical_prices", return_value=None), \
             patch("src.tools.financial_tools._get_ticker", return_value=mock_stock):
            result = get_technical_indicators("AAPL")

        assert result["source"] == "yfinance"
        assert result["ticker"] == "AAPL"
