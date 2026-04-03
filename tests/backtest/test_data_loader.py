"""Tests for backtest data loader."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from aegis.backtest.data_loader import _to_aegis_symbol, load_from_db


class TestDataLoader:
    def test_aegis_symbol_conversion(self):
        assert _to_aegis_symbol("BTCUSDT") == "BTC/USDT"
        assert _to_aegis_symbol("ETHUSDT") == "ETH/USDT"
        assert _to_aegis_symbol("SOLUSDT") == "SOL/USDT"

    def test_load_from_db(self):
        mock_db = MagicMock()
        mock_db.fetch_all.return_value = [
            {
                "symbol": "BTC/USDT",
                "asset_class": "crypto",
                "timestamp": datetime(2025, 6, 1, tzinfo=timezone.utc),
                "timeframe": "1h",
                "open": 42000.0,
                "high": 42500.0,
                "low": 41500.0,
                "close": 42200.0,
                "volume": 100.0,
                "source": "binance",
            }
        ]
        candles = load_from_db(
            mock_db,
            "BTC/USDT",
            "1h",
            datetime(2025, 1, 1),
            datetime(2025, 12, 31),
        )
        assert len(candles) == 1
        assert candles[0].symbol == "BTC/USDT"
        assert candles[0].close == 42200.0

    def test_load_from_db_empty(self):
        mock_db = MagicMock()
        mock_db.fetch_all.return_value = []
        candles = load_from_db(
            mock_db, "BTC/USDT", "1h", datetime(2025, 1, 1), datetime(2025, 12, 31)
        )
        assert candles == []
