"""Tests for momentum agent. Written FIRST per TDD."""

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


class TestMomentumAgent:
    def test_strong_long_on_uptrend(self):
        """10% price increase over 20 periods should give strong long."""
        from aegis.agents.momentum.timeseries import MomentumAgent

        prices = [100.0 + i * 0.55 for i in range(21)]  # ~10% rise
        candles = _make_candles(prices)

        agent = MomentumAgent("mom_03", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.direction > 0.5
        assert signal.confidence > 0.5

    def test_strong_short_on_downtrend(self):
        """10% price decrease over 20 periods should give strong short."""
        from aegis.agents.momentum.timeseries import MomentumAgent

        prices = [100.0 - i * 0.55 for i in range(21)]  # ~10% drop
        candles = _make_candles(prices)

        agent = MomentumAgent("mom_03", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.direction < -0.5
        assert signal.confidence > 0.5

    def test_neutral_on_flat(self):
        """No change should produce neutral."""
        from aegis.agents.momentum.timeseries import MomentumAgent

        prices = [100.0] * 21
        candles = _make_candles(prices)

        agent = MomentumAgent("mom_03", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert abs(signal.direction) < 0.1
        assert signal.confidence < 0.1

    def test_insufficient_data_neutral(self):
        """Fewer than lookback+1 candles returns neutral."""
        from aegis.agents.momentum.timeseries import MomentumAgent

        prices = [100.0] * 10
        candles = _make_candles(prices)

        agent = MomentumAgent("mom_03", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.direction == 0.0
        assert signal.confidence == 0.0

    def test_direction_clipped(self):
        """Extreme returns should not exceed [-1, 1]."""
        from aegis.agents.momentum.timeseries import MomentumAgent

        prices = [100.0] * 20 + [200.0]  # 100% return
        candles = _make_candles(prices)

        agent = MomentumAgent("mom_03", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert -1.0 <= signal.direction <= 1.0
        assert 0.0 <= signal.confidence <= 1.0

    def test_signal_metadata(self):
        from aegis.agents.momentum.timeseries import MomentumAgent

        prices = [100.0 + i for i in range(21)]
        candles = _make_candles(prices)

        agent = MomentumAgent("mom_03", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.agent_type == "momentum"
        assert signal.agent_id == "mom_03"
