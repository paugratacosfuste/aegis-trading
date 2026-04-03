"""Tests for RSI+EMA technical agent. Written FIRST per TDD."""

from datetime import datetime, timedelta, timezone

import pytest

from aegis.common.types import MarketDataPoint


def _make_candles(prices: list[float], timeframe: str = "1h") -> list[MarketDataPoint]:
    """Create candles from a list of close prices."""
    base = datetime(2025, 6, 1, tzinfo=timezone.utc)
    return [
        MarketDataPoint(
            symbol="BTC/USDT",
            asset_class="crypto",
            timestamp=base + timedelta(hours=i),
            timeframe=timeframe,
            open=p - 10,
            high=p + 20,
            low=p - 20,
            close=p,
            volume=100.0,
            source="binance",
        )
        for i, p in enumerate(prices)
    ]


class TestRsiEmaAgent:
    def test_bullish_when_rsi_low_and_ema_bullish(self):
        """Steady uptrend should give bullish EMA crossover + moderate RSI."""
        from aegis.agents.technical.rsi_ema import RsiEmaAgent

        # Consistent uptrend: EMA9 will be above EMA21
        prices = [40000 + i * 150 for i in range(30)]
        candles = _make_candles(prices)

        agent = RsiEmaAgent("tech_03", {})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.direction > 0
        assert signal.agent_id == "tech_03"
        assert signal.symbol == "BTC/USDT"

    def test_bearish_when_rsi_high_and_ema_bearish(self):
        """Steady downtrend should give bearish EMA crossover."""
        from aegis.agents.technical.rsi_ema import RsiEmaAgent

        # Consistent downtrend: EMA9 will be below EMA21
        prices = [45000 - i * 150 for i in range(30)]
        candles = _make_candles(prices)

        agent = RsiEmaAgent("tech_03", {})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.direction < 0

    def test_neutral_when_flat(self):
        """Flat prices should produce near-zero direction."""
        from aegis.agents.technical.rsi_ema import RsiEmaAgent

        prices = [42000.0 + (i % 3 - 1) * 5 for i in range(30)]
        candles = _make_candles(prices)

        agent = RsiEmaAgent("tech_03", {})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert abs(signal.direction) < 0.5

    def test_insufficient_data_returns_neutral(self):
        """Fewer than 21 candles should return neutral signal."""
        from aegis.agents.technical.rsi_ema import RsiEmaAgent

        prices = [42000.0 + i * 10 for i in range(10)]
        candles = _make_candles(prices)

        agent = RsiEmaAgent("tech_03", {})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.direction == 0.0
        assert signal.confidence == 0.0

    def test_direction_within_bounds(self):
        """Direction must always be in [-1, 1]."""
        from aegis.agents.technical.rsi_ema import RsiEmaAgent

        prices = [42000 + i * 500 for i in range(30)]  # strong trend
        candles = _make_candles(prices)

        agent = RsiEmaAgent("tech_03", {})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert -1.0 <= signal.direction <= 1.0
        assert 0.0 <= signal.confidence <= 1.0

    def test_signal_metadata(self):
        """Signal should have correct agent_type and timeframe."""
        from aegis.agents.technical.rsi_ema import RsiEmaAgent

        prices = [42000.0 + i * 50 for i in range(30)]
        candles = _make_candles(prices)

        agent = RsiEmaAgent("tech_03", {})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.agent_type == "technical"
        assert signal.timeframe == "1h"
