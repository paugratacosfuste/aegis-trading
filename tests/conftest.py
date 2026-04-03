"""Shared test fixtures for the Aegis Trading System."""

from datetime import datetime, timezone

import pytest

from aegis.common.types import AgentSignal, MarketDataPoint


@pytest.fixture
def sample_candles_uptrend() -> list[MarketDataPoint]:
    """30 1h candles with steady uptrend from 40000 to 43000."""
    candles = []
    base_price = 40000.0
    step = 100.0
    for i in range(30):
        price = base_price + i * step
        candles.append(
            MarketDataPoint(
                symbol="BTC/USDT",
                asset_class="crypto",
                timestamp=datetime(2025, 6, 1, i, 0, tzinfo=timezone.utc),
                timeframe="1h",
                open=price,
                high=price + 50,
                low=price - 30,
                close=price + 30,
                volume=100.0 + i * 5,
                source="binance",
            )
        )
    return candles


@pytest.fixture
def sample_candles_downtrend() -> list[MarketDataPoint]:
    """30 1h candles with steady downtrend from 43000 to 40000."""
    candles = []
    base_price = 43000.0
    step = -100.0
    for i in range(30):
        price = base_price + i * step
        candles.append(
            MarketDataPoint(
                symbol="BTC/USDT",
                asset_class="crypto",
                timestamp=datetime(2025, 6, 1, i, 0, tzinfo=timezone.utc),
                timeframe="1h",
                open=price,
                high=price + 30,
                low=price - 50,
                close=price - 30,
                volume=100.0 + i * 5,
                source="binance",
            )
        )
    return candles


@pytest.fixture
def sample_candles_flat() -> list[MarketDataPoint]:
    """30 1h candles at roughly constant price ~42000."""
    candles = []
    for i in range(30):
        candles.append(
            MarketDataPoint(
                symbol="BTC/USDT",
                asset_class="crypto",
                timestamp=datetime(2025, 6, 1, i, 0, tzinfo=timezone.utc),
                timeframe="1h",
                open=42000.0,
                high=42020.0,
                low=41980.0,
                close=42000.0,
                volume=100.0,
                source="binance",
            )
        )
    return candles


@pytest.fixture
def sample_candles_volatile() -> list[MarketDataPoint]:
    """30 1h candles with high variance oscillating around 42000."""
    import math

    candles = []
    for i in range(30):
        offset = 1500 * math.sin(i * 0.5)
        price = 42000.0 + offset
        candles.append(
            MarketDataPoint(
                symbol="BTC/USDT",
                asset_class="crypto",
                timestamp=datetime(2025, 6, 1, i, 0, tzinfo=timezone.utc),
                timeframe="1h",
                open=price - 200,
                high=price + 500,
                low=price - 500,
                close=price,
                volume=500.0 + abs(offset) / 3,
                source="binance",
            )
        )
    return candles


@pytest.fixture
def sample_agent_signal() -> AgentSignal:
    """A valid bullish AgentSignal for downstream testing."""
    return AgentSignal(
        agent_id="tech_03",
        agent_type="technical",
        symbol="BTC/USDT",
        timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        direction=0.7,
        confidence=0.8,
        timeframe="1h",
        expected_holding_period="hours",
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        reasoning={"rsi": 28},
        features_used={"rsi_14": 28.0},
        metadata={},
    )
