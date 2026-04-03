"""Tests for AsianRangeAgent."""

from datetime import datetime, timezone

import pytest

from aegis.agents.technical.asian_range import AsianRangeAgent
from aegis.common.types import MarketDataPoint


def _make_session_candles(
    asian_high: float = 42500.0,
    asian_low: float = 42000.0,
    sweep_high: bool = False,
    sweep_low: bool = False,
    price_re_enters: bool = True,
) -> list[MarketDataPoint]:
    """Create 15m candles spanning Asian + London sessions."""
    candles = []
    base_price = 42250.0

    # Asian session: 00:00-08:00 GMT (32 candles at 15m)
    for i in range(32):
        hour = i // 4
        minute = (i % 4) * 15
        mid = (asian_high + asian_low) / 2
        candles.append(MarketDataPoint(
            symbol="BTC/USDT", asset_class="crypto",
            timestamp=datetime(2025, 6, 1, hour, minute, tzinfo=timezone.utc),
            timeframe="15m",
            open=mid - 20, high=min(asian_high, mid + 100),
            low=max(asian_low, mid - 100), close=mid + 10,
            volume=100.0, source="binance",
        ))

    # London session: 08:00-10:00 GMT (8 candles at 15m)
    asian_range = asian_high - asian_low
    for i in range(8):
        hour = 8 + i // 4
        minute = (i % 4) * 15

        if sweep_high and i == 2:
            # Sweep candle that breaks above asian high
            h = asian_high + asian_range * 0.3
            candles.append(MarketDataPoint(
                symbol="BTC/USDT", asset_class="crypto",
                timestamp=datetime(2025, 6, 1, hour, minute, tzinfo=timezone.utc),
                timeframe="15m",
                open=asian_high - 20, high=h, low=asian_high - 50,
                close=asian_high - 30 if price_re_enters else h - 10,
                volume=300.0, source="binance",
            ))
        elif sweep_low and i == 2:
            l = asian_low - asian_range * 0.3
            candles.append(MarketDataPoint(
                symbol="BTC/USDT", asset_class="crypto",
                timestamp=datetime(2025, 6, 1, hour, minute, tzinfo=timezone.utc),
                timeframe="15m",
                open=asian_low + 20, high=asian_low + 50, low=l,
                close=asian_low + 30 if price_re_enters else l + 10,
                volume=300.0, source="binance",
            ))
        else:
            mid = (asian_high + asian_low) / 2
            candles.append(MarketDataPoint(
                symbol="BTC/USDT", asset_class="crypto",
                timestamp=datetime(2025, 6, 1, hour, minute, tzinfo=timezone.utc),
                timeframe="15m",
                open=mid, high=mid + 50, low=mid - 50, close=mid + 10,
                volume=100.0, source="binance",
            ))

    # Post-London candles (fill up to MIN_CANDLES)
    for i in range(8):
        hour = 10 + i // 4
        minute = (i % 4) * 15
        mid = (asian_high + asian_low) / 2
        final_close = mid if price_re_enters else (
            asian_high + asian_range * 0.2 if sweep_high else asian_low - asian_range * 0.2
        )
        candles.append(MarketDataPoint(
            symbol="BTC/USDT", asset_class="crypto",
            timestamp=datetime(2025, 6, 1, hour, minute, tzinfo=timezone.utc),
            timeframe="15m",
            open=final_close - 10, high=final_close + 30,
            low=final_close - 30, close=final_close,
            volume=100.0, source="binance",
        ))

    return candles


class TestAsianRangeAgent:
    def test_high_swept_generates_short(self):
        candles = _make_session_candles(sweep_high=True, price_re_enters=True)
        agent = AsianRangeAgent("tech_11", {"market": "crypto"})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.direction < 0  # Short after high sweep

    def test_low_swept_generates_long(self):
        candles = _make_session_candles(sweep_low=True, price_re_enters=True)
        agent = AsianRangeAgent("tech_11", {"market": "crypto"})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.direction > 0  # Long after low sweep

    def test_no_sweep_neutral(self):
        candles = _make_session_candles(sweep_high=False, sweep_low=False)
        agent = AsianRangeAgent("tech_11", {"market": "crypto"})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert abs(signal.direction) < 0.1

    def test_insufficient_data_neutral(self):
        agent = AsianRangeAgent("tech_11", {"market": "crypto"})
        candles = _make_session_candles()[:10]
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.direction == 0.0

    def test_agent_type_is_technical(self):
        agent = AsianRangeAgent("tech_11", {"market": "crypto"})
        assert agent.agent_type == "technical"

    def test_registered_in_registry(self):
        from aegis.agents.registry import get_agent_class
        cls = get_agent_class("technical", "asian_range")
        assert cls is AsianRangeAgent
