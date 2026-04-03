"""Tests for z-score statistical agent. Written FIRST per TDD."""

from datetime import datetime, timedelta, timezone

import pytest

from aegis.common.types import MarketDataPoint


def _make_candles(prices: list[float]) -> list[MarketDataPoint]:
    base = datetime(2025, 6, 1, tzinfo=timezone.utc)
    return [
        MarketDataPoint(
            symbol="BTC/USDT",
            asset_class="crypto",
            timestamp=base + timedelta(hours=i),
            timeframe="1h",
            open=p,
            high=p + 10,
            low=p - 10,
            close=p,
            volume=100.0,
            source="binance",
        )
        for i, p in enumerate(prices)
    ]


class TestZScoreAgent:
    def test_strong_buy_when_below_mean(self):
        """Price 2+ std devs below mean should produce long signal."""
        from aegis.agents.statistical.zscore import ZScoreAgent

        # 19 candles at 100, then one at 80 (2 std devs below if std ~ 10)
        prices = [100.0] * 15 + [110.0, 90.0, 110.0, 90.0, 70.0]
        candles = _make_candles(prices)

        agent = ZScoreAgent("stat_01", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.direction > 0.5
        assert signal.confidence > 0.5

    def test_strong_sell_when_above_mean(self):
        """Price 2+ std devs above mean should produce short signal."""
        from aegis.agents.statistical.zscore import ZScoreAgent

        prices = [100.0] * 15 + [90.0, 110.0, 90.0, 110.0, 130.0]
        candles = _make_candles(prices)

        agent = ZScoreAgent("stat_01", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.direction < -0.5
        assert signal.confidence > 0.5

    def test_neutral_at_mean(self):
        """Price near the mean should produce near-zero direction."""
        from aegis.agents.statistical.zscore import ZScoreAgent

        prices = [100.0] * 20
        candles = _make_candles(prices)

        agent = ZScoreAgent("stat_01", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert abs(signal.direction) < 0.1
        assert signal.confidence < 0.1

    def test_zero_std_returns_neutral(self):
        """All identical prices (std=0) should not crash."""
        from aegis.agents.statistical.zscore import ZScoreAgent

        prices = [50.0] * 20
        candles = _make_candles(prices)

        agent = ZScoreAgent("stat_01", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.direction == 0.0
        assert signal.confidence == 0.0

    def test_insufficient_data_returns_neutral(self):
        """Fewer than lookback candles should return neutral."""
        from aegis.agents.statistical.zscore import ZScoreAgent

        prices = [100.0] * 5
        candles = _make_candles(prices)

        agent = ZScoreAgent("stat_01", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.direction == 0.0
        assert signal.confidence == 0.0

    def test_direction_clipped(self):
        """Extreme z-scores should not produce direction outside [-1, 1]."""
        from aegis.agents.statistical.zscore import ZScoreAgent

        # Huge outlier
        prices = [100.0] * 19 + [50.0]
        candles = _make_candles(prices)

        agent = ZScoreAgent("stat_01", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert -1.0 <= signal.direction <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
