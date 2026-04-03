"""Tests for signal decay."""

from datetime import datetime, timedelta, timezone

import pytest

from aegis.common.types import AgentSignal
from aegis.ensemble.decay import SIGNAL_HALF_LIVES, apply_decay


def _make_signal(agent_type: str = "technical", confidence: float = 1.0,
                 timestamp: datetime | None = None) -> AgentSignal:
    ts = timestamp or datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    return AgentSignal(
        agent_id="test_01",
        agent_type=agent_type,
        symbol="BTC/USDT",
        timestamp=ts,
        direction=0.5,
        confidence=confidence,
        timeframe="1h",
        expected_holding_period="hours",
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        reasoning={},
        features_used={},
        metadata={},
    )


class TestDecay:
    def test_no_decay_at_zero_hours(self):
        signal = _make_signal()
        result = apply_decay(signal, signal.timestamp)
        assert result.confidence == pytest.approx(1.0)

    def test_half_confidence_at_half_life(self):
        signal = _make_signal(agent_type="technical")
        half_life = SIGNAL_HALF_LIVES["technical"]  # 4 hours
        future = signal.timestamp + timedelta(hours=half_life)
        result = apply_decay(signal, future)
        assert result.confidence == pytest.approx(0.5, abs=0.01)

    def test_quarter_confidence_at_two_half_lives(self):
        signal = _make_signal(agent_type="technical")
        half_life = SIGNAL_HALF_LIVES["technical"]
        future = signal.timestamp + timedelta(hours=half_life * 2)
        result = apply_decay(signal, future)
        assert result.confidence == pytest.approx(0.25, abs=0.01)

    def test_statistical_decays_slower(self):
        tech = _make_signal(agent_type="technical")
        stat = _make_signal(agent_type="statistical")
        future = tech.timestamp + timedelta(hours=24)
        tech_decayed = apply_decay(tech, future)
        stat_decayed = apply_decay(stat, future)
        assert stat_decayed.confidence > tech_decayed.confidence

    def test_unknown_type_uses_default(self):
        signal = _make_signal(agent_type="unknown_type")
        future = signal.timestamp + timedelta(hours=24)
        result = apply_decay(signal, future)
        # Default half-life is 24h, so at 24h: 0.5
        assert result.confidence == pytest.approx(0.5, abs=0.01)

    def test_direction_unchanged(self):
        signal = _make_signal()
        future = signal.timestamp + timedelta(hours=10)
        result = apply_decay(signal, future)
        assert result.direction == signal.direction

    def test_future_timestamp_no_decay(self):
        signal = _make_signal()
        past = signal.timestamp - timedelta(hours=1)
        result = apply_decay(signal, past)
        assert result.confidence == signal.confidence
