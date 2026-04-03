"""Tests for ensemble voter. Written FIRST per TDD."""

from datetime import datetime, timezone

import pytest

from aegis.common.types import AgentSignal


def _make_signal(direction: float, confidence: float, agent_id: str = "test") -> AgentSignal:
    return AgentSignal(
        agent_id=agent_id,
        agent_type="technical",
        symbol="BTC/USDT",
        timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
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


class TestEnsembleVoter:
    def test_all_bullish_produces_long(self):
        """Three bullish signals should produce LONG."""
        from aegis.ensemble.voter import vote

        signals = [
            _make_signal(0.8, 0.9, "tech_03"),
            _make_signal(0.6, 0.7, "stat_01"),
            _make_signal(0.7, 0.8, "mom_03"),
        ]
        decision = vote(signals, confidence_threshold=0.45)
        assert decision.action == "LONG"
        assert decision.direction > 0
        assert decision.confidence > 0.45

    def test_all_bearish_produces_short(self):
        """Three bearish signals should produce SHORT."""
        from aegis.ensemble.voter import vote

        signals = [
            _make_signal(-0.8, 0.9, "tech_03"),
            _make_signal(-0.6, 0.7, "stat_01"),
            _make_signal(-0.7, 0.8, "mom_03"),
        ]
        decision = vote(signals, confidence_threshold=0.45)
        assert decision.action == "SHORT"
        assert decision.direction < 0

    def test_mixed_signals_no_trade(self):
        """Conflicting signals should produce NO_TRADE."""
        from aegis.ensemble.voter import vote

        signals = [
            _make_signal(0.8, 0.9, "tech_03"),
            _make_signal(-0.8, 0.9, "stat_01"),
            _make_signal(0.0, 0.5, "mom_03"),
        ]
        decision = vote(signals, confidence_threshold=0.45)
        # Signals cancel out, direction too weak
        assert decision.action == "NO_TRADE"

    def test_below_threshold_no_trade(self):
        """Low confidence signals should produce NO_TRADE."""
        from aegis.ensemble.voter import vote

        signals = [
            _make_signal(0.5, 0.1, "tech_03"),
            _make_signal(0.3, 0.15, "stat_01"),
            _make_signal(0.4, 0.1, "mom_03"),
        ]
        decision = vote(signals, confidence_threshold=0.45)
        assert decision.action == "NO_TRADE"

    def test_empty_signals_no_trade(self):
        """No signals should produce NO_TRADE."""
        from aegis.ensemble.voter import vote

        decision = vote([], confidence_threshold=0.45)
        assert decision.action == "NO_TRADE"

    def test_higher_threshold_more_selective(self):
        """Production threshold (0.70) should reject marginal signals."""
        from aegis.ensemble.voter import vote

        signals = [
            _make_signal(0.5, 0.5, "tech_03"),
            _make_signal(0.4, 0.5, "stat_01"),
            _make_signal(0.3, 0.5, "mom_03"),
        ]
        # Should pass at 0.45
        lab = vote(signals, confidence_threshold=0.45)
        # Should fail at 0.70
        prod = vote(signals, confidence_threshold=0.70)
        assert lab.action in ("LONG", "NO_TRADE")
        assert prod.action == "NO_TRADE"

    def test_symbol_propagated(self):
        """Decision should carry the symbol from the signals."""
        from aegis.ensemble.voter import vote

        signals = [_make_signal(0.8, 0.9, "tech_03")]
        decision = vote(signals, confidence_threshold=0.45)
        assert decision.symbol == "BTC/USDT"
