"""Tests for agent performance tracking (M6.5)."""

from datetime import datetime, timezone

import pytest

from aegis.ensemble.performance import AgentPerformanceTracker


@pytest.fixture
def tracker():
    return AgentPerformanceTracker()


class TestAgentPerformanceTracker:
    def test_initial_weight_default(self, tracker):
        assert tracker.get_weight("tech_01") == 1.0

    def test_record_correct_prediction(self, tracker):
        tracker.record_outcome("tech_01", predicted_direction=0.8, actual_return=0.05)
        stats = tracker.get_stats("tech_01")
        assert stats["n_predictions"] == 1
        assert stats["n_correct"] == 1
        assert stats["hit_rate"] == 1.0

    def test_record_wrong_prediction(self, tracker):
        tracker.record_outcome("tech_01", predicted_direction=0.8, actual_return=-0.05)
        stats = tracker.get_stats("tech_01")
        assert stats["n_predictions"] == 1
        assert stats["n_correct"] == 0
        assert stats["hit_rate"] == 0.0

    def test_bayesian_weight_update_correct(self, tracker):
        # Multiple correct predictions should increase weight
        for _ in range(10):
            tracker.record_outcome("tech_01", predicted_direction=0.8, actual_return=0.03)
        w = tracker.get_weight("tech_01")
        assert w > 1.0

    def test_bayesian_weight_update_wrong(self, tracker):
        # Multiple wrong predictions should decrease weight
        for _ in range(10):
            tracker.record_outcome("tech_01", predicted_direction=0.8, actual_return=-0.03)
        w = tracker.get_weight("tech_01")
        assert w < 1.0

    def test_weight_bounded(self, tracker):
        # Even extreme performance shouldn't go outside bounds
        for _ in range(100):
            tracker.record_outcome("tech_01", predicted_direction=0.9, actual_return=0.10)
        w = tracker.get_weight("tech_01")
        assert 0.1 <= w <= 3.0

    def test_multiple_agents_independent(self, tracker):
        tracker.record_outcome("tech_01", predicted_direction=0.8, actual_return=0.05)
        tracker.record_outcome("tech_02", predicted_direction=0.8, actual_return=-0.05)
        assert tracker.get_weight("tech_01") > tracker.get_weight("tech_02")

    def test_information_coefficient(self, tracker):
        # Perfect positive correlation
        for i in range(20):
            d = 0.5 + i * 0.02
            r = 0.01 + i * 0.001
            tracker.record_outcome("tech_01", predicted_direction=d, actual_return=r)
        stats = tracker.get_stats("tech_01")
        assert stats["ic"] > 0.5

    def test_no_predictions_default_stats(self, tracker):
        stats = tracker.get_stats("tech_01")
        assert stats["n_predictions"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["ic"] == 0.0

    def test_get_all_weights(self, tracker):
        tracker.record_outcome("tech_01", predicted_direction=0.8, actual_return=0.05)
        tracker.record_outcome("stat_01", predicted_direction=-0.5, actual_return=-0.03)
        weights = tracker.get_all_weights()
        assert "tech_01" in weights
        assert "stat_01" in weights

    def test_near_zero_return_ignored(self, tracker):
        # Very small returns shouldn't count as correct or wrong
        tracker.record_outcome("tech_01", predicted_direction=0.8, actual_return=0.0001)
        stats = tracker.get_stats("tech_01")
        assert stats["n_predictions"] == 0  # Ignored

    def test_reset_agent(self, tracker):
        tracker.record_outcome("tech_01", predicted_direction=0.8, actual_return=0.05)
        tracker.reset("tech_01")
        stats = tracker.get_stats("tech_01")
        assert stats["n_predictions"] == 0
        assert tracker.get_weight("tech_01") == 1.0
