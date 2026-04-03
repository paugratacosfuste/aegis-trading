"""Shared test fixtures for the Aegis Trading System."""

from datetime import datetime, timedelta, timezone

import pytest

from aegis.common.types import (
    AgentSignal,
    CryptoMetrics,
    FundamentalScore,
    GeopoliticalEvent,
    MacroDataPoint,
    MarketDataPoint,
)

_BASE_TIME = datetime(2025, 6, 1, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_candles_uptrend() -> list[MarketDataPoint]:
    """30 1h candles with steady uptrend from 40000 to 43000."""
    candles = []
    base_price = 40000.0
    step = 100.0
    for i in range(30):
        price = base_price + i * step
        ts = _BASE_TIME + timedelta(hours=i)
        candles.append(
            MarketDataPoint(
                symbol="BTC/USDT",
                asset_class="crypto",
                timestamp=ts,
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
        ts = _BASE_TIME + timedelta(hours=i)
        candles.append(
            MarketDataPoint(
                symbol="BTC/USDT",
                asset_class="crypto",
                timestamp=ts,
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
        ts = _BASE_TIME + timedelta(hours=i)
        candles.append(
            MarketDataPoint(
                symbol="BTC/USDT",
                asset_class="crypto",
                timestamp=ts,
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
        ts = _BASE_TIME + timedelta(hours=i)
        candles.append(
            MarketDataPoint(
                symbol="BTC/USDT",
                asset_class="crypto",
                timestamp=ts,
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


@pytest.fixture
def sample_macro_data() -> MacroDataPoint:
    return MacroDataPoint(
        timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        yield_10y=4.25, yield_2y=4.05, yield_spread=0.20,
        vix=18.5, vix_regime="normal", dxy=104.5,
        fed_rate=5.25, cpi_latest=3.2,
    )


@pytest.fixture
def sample_geo_event() -> GeopoliticalEvent:
    return GeopoliticalEvent(
        event_id="geo_test_01",
        timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        source="gdelt", category="conflict", severity=0.6,
        affected_sectors=("energy", "defense"),
        affected_regions=("middle_east",),
        raw_text="Test geopolitical event",
        sentiment_score=-0.4, half_life_hours=24,
    )


@pytest.fixture
def sample_fundamental_score() -> FundamentalScore:
    return FundamentalScore(
        symbol="AAPL",
        timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
        sector="tech", market_cap_tier="large",
        quality_score=0.82, value_score=0.65, growth_score=0.90,
        pe_zscore=-0.3, revenue_growth=0.12, source="yahoo",
    )


@pytest.fixture
def sample_crypto_metrics() -> CryptoMetrics:
    return CryptoMetrics(
        symbol="BTC/USDT",
        timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        funding_rate=0.005, open_interest=5e9, btc_dominance=55.0,
        fear_greed_index=65, tvl=50e9, tvl_change_24h=1.5,
        liquidations_24h=80e6, source="binance",
    )


@pytest.fixture
def sample_veto_signal() -> AgentSignal:
    """A signal with VETO metadata set."""
    return AgentSignal(
        agent_id="geo_01", agent_type="geopolitical", symbol="BTC/USDT",
        timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        direction=-0.5, confidence=0.9, timeframe="1h",
        expected_holding_period="hours", entry_price=None,
        stop_loss=None, take_profit=None,
        reasoning={"risk_score": 0.85},
        features_used={},
        metadata={"veto": True, "risk_score": 0.85},
    )
