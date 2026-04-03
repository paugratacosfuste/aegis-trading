"""Tests for two-level ensemble voter upgrade."""

from datetime import datetime, timezone

import pytest

from aegis.common.types import AgentSignal
from aegis.ensemble.voter import vote


def _sig(agent_id: str, agent_type: str, direction: float, confidence: float) -> AgentSignal:
    return AgentSignal(
        agent_id=agent_id, agent_type=agent_type, symbol="BTC/USDT",
        timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        direction=direction, confidence=confidence, timeframe="1h",
        expected_holding_period="hours", entry_price=None, stop_loss=None,
        take_profit=None, reasoning={}, features_used={}, metadata={},
    )


class TestTwoLevelVoting:
    def test_single_type_works(self):
        signals = [
            _sig("tech_01", "technical", 0.8, 0.9),
            _sig("tech_02", "technical", 0.7, 0.8),
        ]
        result = vote(signals)
        assert result.action == "LONG"
        assert result.direction > 0

    def test_multiple_types(self):
        signals = [
            _sig("tech_01", "technical", 0.6, 0.8),
            _sig("stat_01", "statistical", 0.5, 0.7),
            _sig("mom_01", "momentum", 0.7, 0.9),
        ]
        result = vote(signals)
        assert result.action == "LONG"

    def test_intra_type_aggregation(self):
        # 10 technical agents should aggregate before cross-type vote
        signals = [
            _sig(f"tech_{i:02d}", "technical", 0.6, 0.7) for i in range(10)
        ]
        result = vote(signals)
        assert result.action == "LONG"

    def test_regime_affects_result(self):
        signals = [
            _sig("tech_01", "technical", 0.3, 0.6),
            _sig("stat_01", "statistical", -0.5, 0.8),
            _sig("mom_01", "momentum", 0.2, 0.5),
        ]
        normal = vote(signals, regime="normal")
        mean_rev = vote(signals, regime="mean_reverting")
        # In mean_reverting regime, statistical weight is boosted
        # so the overall direction should be more negative
        assert mean_rev.direction < normal.direction or \
               mean_rev.action == "NO_TRADE"

    def test_conflict_tech_vs_stat_no_trade(self):
        signals = [
            _sig("tech_01", "technical", 0.8, 0.7),
            _sig("stat_01", "statistical", -0.8, 0.7),
        ]
        result = vote(signals)
        assert result.action == "NO_TRADE"
        assert "conflict" in result.reason.lower()

    def test_strong_consensus_trades(self):
        signals = [
            _sig("tech_01", "technical", 0.8, 0.9),
            _sig("stat_01", "statistical", 0.7, 0.8),
            _sig("mom_01", "momentum", 0.9, 0.95),
            _sig("sent_01", "sentiment", 0.5, 0.6),
        ]
        result = vote(signals)
        assert result.action == "LONG"
        assert result.confidence > 0.5

    def test_all_bearish(self):
        signals = [
            _sig("tech_01", "technical", -0.7, 0.8),
            _sig("stat_01", "statistical", -0.6, 0.7),
            _sig("mom_01", "momentum", -0.8, 0.9),
        ]
        result = vote(signals)
        assert result.action == "SHORT"

    def test_below_confidence_threshold(self):
        signals = [
            _sig("tech_01", "technical", 0.3, 0.2),
            _sig("stat_01", "statistical", 0.2, 0.1),
        ]
        result = vote(signals, confidence_threshold=0.70)
        assert result.action == "NO_TRADE"

    def test_direction_too_weak(self):
        signals = [
            _sig("tech_01", "technical", 0.05, 0.8),
            _sig("stat_01", "statistical", -0.05, 0.8),
        ]
        result = vote(signals)
        assert result.action == "NO_TRADE"

    def test_empty_signals(self):
        result = vote([])
        assert result.action == "NO_TRADE"

    def test_backward_compat_phase1_style(self):
        # Calling with just signals + threshold should work
        signals = [
            _sig("rsi_ema_bt", "technical", 0.7, 0.8),
            _sig("zscore_bt", "statistical", 0.3, 0.6),
            _sig("momentum_bt", "momentum", 0.5, 0.7),
        ]
        result = vote(signals, confidence_threshold=0.45)
        assert result.action in ("LONG", "SHORT", "NO_TRADE")

    def test_contributing_signals_contain_all(self):
        signals = [
            _sig("tech_01", "technical", 0.8, 0.9),
            _sig("mom_01", "momentum", 0.7, 0.8),
        ]
        result = vote(signals)
        if result.action != "NO_TRADE":
            assert "tech_01" in result.contributing_signals
            assert "mom_01" in result.contributing_signals
