"""Integration tests for the data pipeline. Tests WS parsing and DB read/write."""

from datetime import datetime, timezone

import pytest

from aegis.common.types import MarketDataPoint
from aegis.data.binance_ws import _parse_kline, _symbol_to_aegis


class TestBinanceWsParsing:
    def test_symbol_conversion(self):
        assert _symbol_to_aegis("btcusdt") == "BTC/USDT"
        assert _symbol_to_aegis("ethusdt") == "ETH/USDT"
        assert _symbol_to_aegis("solusdt") == "SOL/USDT"

    def test_parse_closed_kline(self):
        """Closed kline should produce a MarketDataPoint."""
        msg = {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {
                "t": 1704067200000,  # 2024-01-01 00:00:00 UTC
                "s": "BTCUSDT",
                "i": "1m",
                "o": "42000.00",
                "h": "42100.00",
                "l": "41900.00",
                "c": "42050.00",
                "v": "150.5",
                "x": True,  # closed
            },
        }
        point = _parse_kline(msg)
        assert point is not None
        assert point.symbol == "BTC/USDT"
        assert point.timeframe == "1m"
        assert point.open == 42000.0
        assert point.close == 42050.0
        assert point.volume == 150.5
        assert point.source == "binance"
        assert point.asset_class == "crypto"

    def test_parse_unclosed_kline_returns_none(self):
        """Unclosed kline should return None."""
        msg = {
            "k": {
                "t": 1704067200000,
                "s": "BTCUSDT",
                "i": "1m",
                "o": "42000.00",
                "h": "42100.00",
                "l": "41900.00",
                "c": "42050.00",
                "v": "150.5",
                "x": False,
            },
        }
        point = _parse_kline(msg)
        assert point is None


class TestMarketDataRepository:
    """Integration tests requiring a real PostgreSQL connection."""

    @pytest.fixture
    def db_pool(self):
        """Connect to the test database."""
        from aegis.common.db import DatabasePool

        try:
            pool = DatabasePool(
                host="localhost",
                port=5432,
                dbname="aegis",
                user="aegis",
                password="aegis",
                min_connections=1,
                max_connections=3,
            )
            # Clean up test data before/after
            pool.execute(
                "DELETE FROM market_data WHERE symbol = %s", ("TEST/USDT",)
            )
            yield pool
            pool.execute(
                "DELETE FROM market_data WHERE symbol = %s", ("TEST/USDT",)
            )
            pool.close()
        except Exception:
            pytest.skip("PostgreSQL not available")

    @pytest.fixture
    def repo(self, db_pool):
        from aegis.data.repository import MarketDataRepository

        return MarketDataRepository(db_pool)

    @pytest.mark.integration
    def test_insert_and_read_candle(self, repo):
        point = MarketDataPoint(
            symbol="TEST/USDT",
            asset_class="crypto",
            timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
            timeframe="1h",
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=1000.0,
            source="test",
        )
        repo.insert_candle(point)

        candles = repo.get_candles(
            "TEST/USDT", "1h",
            datetime(2025, 6, 1, tzinfo=timezone.utc),
            datetime(2025, 6, 2, tzinfo=timezone.utc),
        )
        assert len(candles) == 1
        assert candles[0].close == 105.0

    @pytest.mark.integration
    def test_duplicate_insert_ignored(self, repo):
        point = MarketDataPoint(
            symbol="TEST/USDT",
            asset_class="crypto",
            timestamp=datetime(2025, 6, 1, 13, 0, tzinfo=timezone.utc),
            timeframe="1h",
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=1000.0,
            source="test",
        )
        repo.insert_candle(point)
        repo.insert_candle(point)  # duplicate

        candles = repo.get_latest_candles("TEST/USDT", "1h", limit=10)
        timestamps = [c.timestamp for c in candles]
        assert timestamps.count(point.timestamp) == 1

    @pytest.mark.integration
    def test_get_latest_price(self, repo):
        point = MarketDataPoint(
            symbol="TEST/USDT",
            asset_class="crypto",
            timestamp=datetime(2025, 6, 1, 14, 0, tzinfo=timezone.utc),
            timeframe="1m",
            open=200.0,
            high=210.0,
            low=190.0,
            close=205.0,
            volume=500.0,
            source="test",
        )
        repo.insert_candle(point)

        price = repo.get_latest_price("TEST/USDT")
        assert price == 205.0

    @pytest.mark.integration
    def test_batch_insert(self, repo):
        points = [
            MarketDataPoint(
                symbol="TEST/USDT",
                asset_class="crypto",
                timestamp=datetime(2025, 6, 1, 15, i, tzinfo=timezone.utc),
                timeframe="1m",
                open=100.0 + i,
                high=110.0 + i,
                low=90.0 + i,
                close=105.0 + i,
                volume=100.0,
                source="test",
            )
            for i in range(5)
        ]
        repo.insert_candles_batch(points)

        candles = repo.get_latest_candles("TEST/USDT", "1m", limit=10)
        assert len(candles) >= 5
