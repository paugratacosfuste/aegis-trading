"""Tests for intra-type aggregator."""

from datetime import datetime, timezone

import pytest

from aegis.common.types import AgentSignal
from aegis.ensemble.aggregator import aggregate_intra_type


def _make_signal(
    agent_id: str = "tech_01",
    agent_type: str = "technical",
    direction: float = 0.5,
    confidence: float = 0.8,
    timestamp: datetime | None = None,
) -> AgentSignal:
    ts = timestamp or datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    return AgentSignal(
        agent_id=agent_id,
        agent_type=agent_type,
        symbol="BTC/USDT",
        timestamp=ts,
        direction=direction,
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


class TestAggregator:
    def test_empty_signals_returns_none(self):
        assert aggregate_intra_type([]) is None

    def test_single_signal_passes_through(self):
        sig = _make_signal(direction=0.7, confidence=0.9)
        result = aggregate_intra_type([sig])
        assert result is not None
        assert result.direction == pytest.approx(0.7, abs=0.01)

    def test_all_agree_boosted_confidence(self):
        signals = [
            _make_signal(agent_id=f"tech_{i:02d}", direction=0.6, confidence=0.7)
            for i in range(5)
        ]
        result = aggregate_intra_type(signals)
        assert result is not None
        # All agree -> agreement bonus boosts confidence
        # agreement = 1 - std([0.6]*5) = 1.0
        # confidence = avg(0.7) * (0.7 + 0.3*1.0) = 0.7
        assert result.confidence >= 0.65

    def test_split_signals_reduced_confidence(self):
        signals = [
            _make_signal(agent_id="tech_01", direction=0.8, confidence=0.7),
            _make_signal(agent_id="tech_02", direction=0.8, confidence=0.7),
            _make_signal(agent_id="tech_03", direction=-0.8, confidence=0.7),
            _make_signal(agent_id="tech_04", direction=-0.8, confidence=0.7),
        ]
        result = aggregate_intra_type(signals)
        assert result is not None
        # Split 50/50 -> low agreement, direction near 0
        assert abs(result.direction) < 0.1

    def test_weight_affects_direction(self):
        signals = [
            _make_signal(agent_id="tech_01", direction=0.8, confidence=0.8),
            _make_signal(agent_id="tech_02", direction=-0.5, confidence=0.8),
        ]
        weights = {"tech_01": 3.0, "tech_02": 1.0}
        result = aggregate_intra_type(signals, agent_weights=weights)
        assert result is not None
        assert result.direction > 0  # tech_01 dominates

    def test_all_zero_confidence_returns_neutral(self):
        signals = [
            _make_signal(agent_id="tech_01", direction=0.8, confidence=0.0),
            _make_signal(agent_id="tech_02", direction=-0.5, confidence=0.0),
        ]
        result = aggregate_intra_type(signals)
        assert result is not None
        assert result.direction == 0.0
        assert result.confidence == 0.0

    def test_direction_bounds(self):
        signals = [
            _make_signal(agent_id=f"tech_{i:02d}", direction=1.0, confidence=1.0)
            for i in range(10)
        ]
        result = aggregate_intra_type(signals)
        assert result is not None
        assert -1.0 <= result.direction <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_agent_type_preserved(self):
        signals = [_make_signal(agent_type="statistical")]
        result = aggregate_intra_type(signals)
        assert result.agent_type == "statistical"
