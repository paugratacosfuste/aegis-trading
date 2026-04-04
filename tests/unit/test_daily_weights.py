"""Tests for daily Bayesian weight updates."""

import pytest

from aegis.feedback.daily_weights import (
    compute_composite_score,
    compute_ic,
    ema_update,
    normalize_within_type,
    run_daily_update,
)


class TestComputeIC:
    def test_strong_positive_correlation(self):
        predictions = [1.0, 1.0, -1.0, -1.0]
        actuals = [0.05, 0.03, -0.04, -0.02]
        ic = compute_ic(predictions, actuals)
        assert ic > 0.9  # Strong positive correlation

    def test_strong_negative_correlation(self):
        predictions = [1.0, 1.0, -1.0, -1.0]
        actuals = [-0.05, -0.03, 0.04, 0.02]
        ic = compute_ic(predictions, actuals)
        assert ic < -0.9  # Strong negative correlation

    def test_zero_variance_predictions(self):
        predictions = [1.0, 1.0, 1.0]
        actuals = [0.01, -0.01, 0.02]
        ic = compute_ic(predictions, actuals)
        assert ic == 0.0

    def test_too_few_samples(self):
        ic = compute_ic([1.0, -1.0], [0.01, -0.01])
        assert ic == 0.0

    def test_empty(self):
        assert compute_ic([], []) == 0.0


class TestComputeCompositeScore:
    def test_standard(self):
        # 0.5 * 0.6 + 0.5 * 0.3 = 0.45
        score = compute_composite_score(hit_rate=0.6, ic=0.3)
        assert score == pytest.approx(0.45)

    def test_negative_ic_clamped_to_zero(self):
        # 0.5 * 0.4 + 0.5 * max(-0.2, 0) = 0.2
        score = compute_composite_score(hit_rate=0.4, ic=-0.2)
        assert score == pytest.approx(0.20)

    def test_perfect_scores(self):
        score = compute_composite_score(hit_rate=1.0, ic=1.0)
        assert score == pytest.approx(1.0)

    def test_zero_scores(self):
        score = compute_composite_score(hit_rate=0.0, ic=0.0)
        assert score == pytest.approx(0.0)


class TestEmaUpdate:
    def test_standard(self):
        # 0.20 * 0.9 + 0.45 * 0.1 = 0.225
        result = ema_update(old_weight=0.20, score=0.45, learning_rate=0.10)
        assert result == pytest.approx(0.225)

    def test_clipped_below(self):
        result = ema_update(old_weight=0.05, score=0.0, learning_rate=0.10,
                            weight_min=0.05)
        assert result >= 0.05

    def test_clipped_above(self):
        result = ema_update(old_weight=0.30, score=1.0, learning_rate=0.10,
                            weight_max=0.30)
        assert result <= 0.30

    def test_high_learning_rate(self):
        result = ema_update(old_weight=0.10, score=0.80, learning_rate=0.50)
        # 0.10 * 0.5 + 0.80 * 0.5 = 0.45 -> clipped to 0.30
        assert result == pytest.approx(0.30)


class TestNormalizeWithinType:
    def test_renormalize(self):
        weights = {
            "tech_01": {"weight": 0.30, "type": "technical"},
            "tech_02": {"weight": 0.20, "type": "technical"},
            "tech_03": {"weight": 0.10, "type": "technical"},
        }
        result = normalize_within_type(weights)
        total = sum(v["weight"] for v in result.values())
        assert total == pytest.approx(1.0)
        # Proportions preserved
        assert result["tech_01"]["weight"] > result["tech_02"]["weight"]
        assert result["tech_02"]["weight"] > result["tech_03"]["weight"]

    def test_multiple_types(self):
        weights = {
            "tech_01": {"weight": 0.20, "type": "technical"},
            "tech_02": {"weight": 0.10, "type": "technical"},
            "stat_01": {"weight": 0.15, "type": "statistical"},
            "stat_02": {"weight": 0.05, "type": "statistical"},
        }
        result = normalize_within_type(weights)
        tech_total = sum(
            v["weight"] for v in result.values() if v["type"] == "technical"
        )
        stat_total = sum(
            v["weight"] for v in result.values() if v["type"] == "statistical"
        )
        assert tech_total == pytest.approx(1.0)
        assert stat_total == pytest.approx(1.0)

    def test_single_agent_type(self):
        weights = {"tech_01": {"weight": 0.15, "type": "technical"}}
        result = normalize_within_type(weights)
        assert result["tech_01"]["weight"] == pytest.approx(1.0)


class TestRunDailyUpdate:
    def test_with_signal_outcomes(self):
        """Integration-style: given signal outcomes, produces correct weight updates."""
        signal_outcomes = [
            # tech_01: 4 correct, 1 wrong = 0.8 hit_rate
            {"agent_id": "tech_01", "agent_type": "technical",
             "predicted_direction": 1.0, "actual_return": 0.02, "is_correct": True},
            {"agent_id": "tech_01", "agent_type": "technical",
             "predicted_direction": -1.0, "actual_return": -0.01, "is_correct": True},
            {"agent_id": "tech_01", "agent_type": "technical",
             "predicted_direction": 1.0, "actual_return": 0.03, "is_correct": True},
            {"agent_id": "tech_01", "agent_type": "technical",
             "predicted_direction": 1.0, "actual_return": 0.01, "is_correct": True},
            {"agent_id": "tech_01", "agent_type": "technical",
             "predicted_direction": 1.0, "actual_return": -0.01, "is_correct": False},
            # tech_02: 2 correct, 3 wrong = 0.4 hit_rate
            {"agent_id": "tech_02", "agent_type": "technical",
             "predicted_direction": 1.0, "actual_return": 0.01, "is_correct": True},
            {"agent_id": "tech_02", "agent_type": "technical",
             "predicted_direction": 1.0, "actual_return": -0.02, "is_correct": False},
            {"agent_id": "tech_02", "agent_type": "technical",
             "predicted_direction": -1.0, "actual_return": 0.01, "is_correct": False},
            {"agent_id": "tech_02", "agent_type": "technical",
             "predicted_direction": 1.0, "actual_return": -0.01, "is_correct": False},
            {"agent_id": "tech_02", "agent_type": "technical",
             "predicted_direction": -1.0, "actual_return": -0.02, "is_correct": True},
        ]
        current_weights = {
            "tech_01": {"agent_type": "technical", "weight": 0.15},
            "tech_02": {"agent_type": "technical", "weight": 0.15},
        }
        logs = run_daily_update(
            signal_outcomes=signal_outcomes,
            current_weights=current_weights,
            learning_rate=0.10,
            min_signals=5,
            weight_min=0.05,
            weight_max=0.30,
        )
        assert len(logs) == 2
        # tech_01 should get higher weight (better performer)
        tech_01_log = next(l for l in logs if l.agent_id == "tech_01")
        tech_02_log = next(l for l in logs if l.agent_id == "tech_02")
        assert tech_01_log.new_weight > tech_02_log.new_weight
        # Both should sum to 1.0 (normalized within type)
        total = tech_01_log.new_weight + tech_02_log.new_weight
        assert total == pytest.approx(1.0)

    def test_skip_sparse_agents(self):
        """Agents with fewer than min_signals are skipped."""
        signal_outcomes = [
            {"agent_id": "tech_01", "agent_type": "technical",
             "predicted_direction": 1.0, "actual_return": 0.02, "is_correct": True},
            {"agent_id": "tech_01", "agent_type": "technical",
             "predicted_direction": 1.0, "actual_return": 0.01, "is_correct": True},
        ]
        current_weights = {
            "tech_01": {"agent_type": "technical", "weight": 0.20},
        }
        logs = run_daily_update(
            signal_outcomes=signal_outcomes,
            current_weights=current_weights,
            min_signals=5,
        )
        assert len(logs) == 0

    def test_no_signal_outcomes(self):
        logs = run_daily_update(
            signal_outcomes=[],
            current_weights={},
        )
        assert logs == []

    def test_default_weight_for_new_agent(self):
        """Agent not in current_weights gets default weight."""
        signal_outcomes = [
            {"agent_id": "tech_01", "agent_type": "technical",
             "predicted_direction": 1.0, "actual_return": 0.02, "is_correct": True},
        ] * 5
        logs = run_daily_update(
            signal_outcomes=signal_outcomes,
            current_weights={},
            min_signals=5,
        )
        assert len(logs) == 1
        # Should use default 1/N weight as old_weight
        assert logs[0].old_weight > 0
