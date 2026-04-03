"""Tests for yfinance collector."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestYFinanceCollector:
    @pytest.fixture
    def mock_repo(self):
        return MagicMock()

    @pytest.fixture
    def collector(self, mock_repo):
        from aegis.data.yfinance_collector import YFinanceCollector

        return YFinanceCollector(
            repository=mock_repo,
            symbols=["SPY", "QQQ"],
        )

    def test_symbol_list(self, collector):
        assert collector._symbols == ["SPY", "QQQ"]

    @patch("aegis.data.yfinance_collector.yf")
    def test_collect_daily(self, mock_yf, collector, mock_repo):
        mock_ticker = MagicMock()
        mock_df = pd.DataFrame(
            {
                "Open": [400.0],
                "High": [405.0],
                "Low": [398.0],
                "Close": [402.0],
                "Volume": [1000000],
            },
            index=pd.DatetimeIndex([datetime(2025, 6, 1, tzinfo=timezone.utc)]),
        )
        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        total = collector.collect_daily(period="5d")
        assert total == 2  # 1 candle per symbol * 2 symbols
        assert mock_yf.Ticker.call_count == 2
        assert mock_repo.insert_candles_batch.call_count == 2

    @patch("aegis.data.yfinance_collector.yf")
    def test_collect_daily_empty(self, mock_yf, collector):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        total = collector.collect_daily(period="5d")
        assert total == 0

    @patch("aegis.data.yfinance_collector.yf")
    def test_collect_historical(self, mock_yf, collector, mock_repo):
        mock_ticker = MagicMock()
        mock_df = pd.DataFrame(
            {
                "Open": [400.0, 401.0],
                "High": [405.0, 406.0],
                "Low": [398.0, 399.0],
                "Close": [402.0, 403.0],
                "Volume": [1000000, 1100000],
            },
            index=pd.DatetimeIndex(
                [
                    datetime(2025, 6, 1, tzinfo=timezone.utc),
                    datetime(2025, 6, 2, tzinfo=timezone.utc),
                ]
            ),
        )
        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        count = collector.collect_historical("SPY", "2025-06-01", "2025-06-02")
        assert count == 2
        mock_repo.insert_candles_batch.assert_called_once()

    @patch("aegis.data.yfinance_collector.yf")
    def test_collect_daily_error_continues(self, mock_yf, collector):
        mock_yf.Ticker.side_effect = Exception("API error")
        total = collector.collect_daily(period="5d")
        assert total == 0
