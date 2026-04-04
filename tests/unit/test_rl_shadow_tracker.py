"""Tests for shadow tracker."""

import numpy as np
import pytest

from aegis.rl.shadow.tracker import ShadowTracker


class TestShadowTracker:
    def test_all_none_no_errors(self):
        """Tracker with no components should be no-op."""
        tracker = ShadowTracker()
        result = tracker.on_ensemble_vote([], "normal", 10000.0)
        assert result is None
        result = tracker.on_position_sized(np.zeros(25), 0.03, 0.03)
        assert result is None
        result = tracker.on_exit_check(np.zeros(20), "hold")
        assert result is None
        tracker.on_trade_closed({"symbol": "BTC/USDT", "net_pnl": 100.0})
        assert tracker.get_summary()["total_predictions"] == 1  # trade_outcome only

    def test_weight_bandit_tracking(self, sample_agent_signal):
        from aegis.rl.weight_allocator.bandit import WeightAllocatorBandit

        bandit = WeightAllocatorBandit(exploration_rate=1.0)
        tracker = ShadowTracker(weight_bandit=bandit)

        weights = tracker.on_ensemble_vote(
            [sample_agent_signal], "bull", 10000.0,
            baseline_weights={"technical": 0.25},
        )
        assert weights is not None
        assert isinstance(weights, dict)
        summary = tracker.get_summary()
        assert summary["by_component"]["weight_allocator"] == 1

    def test_exception_safety(self):
        """Tracker must never raise, even with broken components."""

        class BrokenBandit:
            def predict(self, ctx):
                raise RuntimeError("Intentional test error")

        tracker = ShadowTracker(weight_bandit=BrokenBandit())
        result = tracker.on_ensemble_vote([], "normal", 10000.0)
        assert result is None  # Swallowed exception

    def test_get_summary(self, sample_agent_signal):
        from aegis.rl.weight_allocator.bandit import WeightAllocatorBandit

        bandit = WeightAllocatorBandit(exploration_rate=1.0)
        tracker = ShadowTracker(weight_bandit=bandit)

        for _ in range(5):
            tracker.on_ensemble_vote([sample_agent_signal], "normal", 10000.0)

        summary = tracker.get_summary()
        assert summary["total_predictions"] == 5
        assert summary["by_component"]["weight_allocator"] == 5
