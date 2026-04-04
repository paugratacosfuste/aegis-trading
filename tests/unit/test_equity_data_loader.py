"""Tests for equity data loading via yfinance in backtest data_loader."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aegis.backtest.data_loader import download_from_yfinance, is_crypto_symbol


class TestIsCryptoSymbol:
    def test_btc_usdt_is_crypto(self):
        assert is_crypto_symbol("BTC/USDT") is True

    def test_eth_usdt_is_crypto(self):
        assert is_crypto_symbol("ETH/USDT") is True

    def test_sol_usdt_is_crypto(self):
        assert is_crypto_symbol("SOL/USDT") is True

    def test_aapl_is_equity(self):
        assert is_crypto_symbol("AAPL") is False

    def test_msft_is_equity(self):
        assert is_crypto_symbol("MSFT") is False

    def test_spy_is_equity(self):
        assert is_crypto_symbol("SPY") is False

    def test_jpm_is_equity(self):
        assert is_crypto_symbol("JPM") is False

    def test_btcusdt_no_slash_is_crypto(self):
        assert is_crypto_symbol("BTCUSDT") is True

    def test_googl_is_equity(self):
        assert is_crypto_symbol("GOOGL") is False


class TestDownloadFromYfinance:
    @patch("aegis.backtest.data_loader.yf")
    def test_returns_market_data_points(self, mock_yf):
        """download_from_yfinance returns MarketDataPoint list with equity asset class."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        index = pd.DatetimeIndex([
            datetime(2025, 4, 1),
            datetime(2025, 4, 2),
            datetime(2025, 4, 3),
        ])
        mock_ticker.history.return_value = pd.DataFrame(
            {
                "Open": [150.0, 151.0, 152.0],
                "High": [155.0, 156.0, 157.0],
                "Low": [149.0, 150.0, 151.0],
                "Close": [153.0, 154.0, 155.0],
                "Volume": [1e6, 1.1e6, 1.2e6],
            },
            index=index,
        )

        candles = download_from_yfinance("AAPL", start="2025-04-01", end="2025-04-04")

        assert len(candles) == 3
        assert candles[0].symbol == "AAPL"
        assert candles[0].asset_class == "equity"
        assert candles[0].timeframe == "1d"
        assert candles[0].source == "yfinance"
        assert candles[0].open == 150.0
        assert candles[0].close == 153.0

    @patch("aegis.backtest.data_loader.yf")
    def test_empty_dataframe_returns_empty_list(self, mock_yf):
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        candles = download_from_yfinance("INVALID", start="2025-04-01", end="2025-04-04")
        assert candles == []

    @patch("aegis.backtest.data_loader.yf")
    def test_interval_passed_to_yfinance(self, mock_yf):
        """Verify interval parameter is forwarded to yfinance."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        download_from_yfinance("AAPL", start="2025-04-01", end="2025-04-04", interval="1wk")
        mock_ticker.history.assert_called_once_with(
            start="2025-04-01", end="2025-04-04", interval="1wk"
        )

    @patch("aegis.backtest.data_loader.yf")
    def test_timestamps_are_utc(self, mock_yf):
        """Timestamps should be UTC even if yfinance returns naive datetimes."""
        from datetime import timezone

        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        index = pd.DatetimeIndex([datetime(2025, 6, 1)])
        mock_ticker.history.return_value = pd.DataFrame(
            {
                "Open": [100.0], "High": [105.0], "Low": [99.0],
                "Close": [103.0], "Volume": [5e5],
            },
            index=index,
        )

        candles = download_from_yfinance("MSFT", start="2025-06-01", end="2025-06-02")
        assert candles[0].timestamp.tzinfo is not None
