"""Tests for anti-overfitting safeguards."""

import numpy as np
import pytest

from aegis.feedback.safeguards import (
    bonferroni_significance,
    check_minimum_samples,
    check_minimum_training_data,
    check_prediction_stability,
    compute_regime_conditioned_metrics,
)


class TestMinimumSamples:
    def test_below_threshold(self):
        assert check_minimum_samples(29) is False

    def test_at_threshold(self):
        assert check_minimum_samples(30) is True

    def test_above_threshold(self):
        assert check_minimum_samples(100) is True

    def test_custom_threshold(self):
        assert check_minimum_samples(4, required=5) is False
        assert check_minimum_samples(5, required=5) is True


class TestMinimumTrainingData:
    def test_below(self):
        assert check_minimum_training_data(499) is False

    def test_at(self):
        assert check_minimum_training_data(500) is True

    def test_custom(self):
        assert check_minimum_training_data(99, required=100) is False
        assert check_minimum_training_data(100, required=100) is True


class TestBonferroniSignificance:
    def test_significant(self):
        # p=0.004, n=10, alpha=0.05 -> threshold=0.005 -> 0.004 < 0.005 -> True
        assert bonferroni_significance(0.004, 10) is True

    def test_not_significant(self):
        # p=0.01, n=10, alpha=0.05 -> threshold=0.005 -> 0.01 > 0.005 -> False
        assert bonferroni_significance(0.01, 10) is False

    def test_single_comparison(self):
        # p=0.04, n=1, alpha=0.05 -> threshold=0.05 -> 0.04 < 0.05 -> True
        assert bonferroni_significance(0.04, 1) is True

    def test_zero_comparisons(self):
        assert bonferroni_significance(0.001, 0) is False

    def test_many_comparisons(self):
        # p=0.001, n=100, alpha=0.05 -> threshold=0.0005 -> 0.001 > 0.0005 -> False
        assert bonferroni_significance(0.001, 100) is False

    def test_custom_alpha(self):
        # p=0.001, n=10, alpha=0.10 -> threshold=0.01 -> 0.001 < 0.01 -> True
        assert bonferroni_significance(0.001, 10, alpha=0.10) is True


class TestPredictionStability:
    def test_stable(self):
        old = np.array([0.6, 0.7, 0.3, 0.8, 0.2])
        new = np.array([0.65, 0.72, 0.28, 0.81, 0.19])  # Same classes
        is_stable, frac = check_prediction_stability(old, new)
        assert is_stable is True
        assert frac == 0.0

    def test_unstable(self):
        old = np.array([0.6, 0.7, 0.3, 0.8, 0.2])
        new = np.array([0.4, 0.3, 0.7, 0.2, 0.8])  # All flipped
        is_stable, frac = check_prediction_stability(old, new)
        assert is_stable is False
        assert frac == 1.0

    def test_partial_change(self):
        old = np.array([0.6, 0.7, 0.3, 0.8, 0.2, 0.6, 0.7, 0.3, 0.8, 0.2])
        new = np.array([0.4, 0.7, 0.7, 0.8, 0.2, 0.6, 0.7, 0.3, 0.8, 0.2])
        # 2/10 flipped = 20% change
        is_stable, frac = check_prediction_stability(old, new)
        assert is_stable is True
        assert abs(frac - 0.2) < 1e-9

    def test_empty_arrays(self):
        is_stable, frac = check_prediction_stability(np.array([]), np.array([]))
        assert is_stable is True
        assert frac == 0.0

    def test_mismatched_length(self):
        is_stable, frac = check_prediction_stability(np.array([0.5]), np.array([0.5, 0.6]))
        assert is_stable is False
        assert frac == 1.0

    def test_custom_threshold(self):
        old = np.array([0.6, 0.3, 0.7, 0.8, 0.2])
        new = np.array([0.4, 0.3, 0.7, 0.8, 0.2])  # 1/5 = 20%
        is_stable, frac = check_prediction_stability(old, new, threshold=0.10)
        assert is_stable is False
        assert abs(frac - 0.2) < 1e-9


class TestRegimeConditionedMetrics:
    def test_single_regime(self):
        trades = [
            {"regime_at_entry": "bull", "net_pnl": 100.0},
            {"regime_at_entry": "bull", "net_pnl": -50.0},
            {"regime_at_entry": "bull", "net_pnl": 75.0},
        ]
        result = compute_regime_conditioned_metrics(trades)
        assert "bull" in result
        assert result["bull"]["n_trades"] == 3.0
        assert result["bull"]["win_rate"] == pytest.approx(2 / 3)

    def test_multiple_regimes(self):
        trades = [
            {"regime_at_entry": "bull", "net_pnl": 100.0},
            {"regime_at_entry": "bear", "net_pnl": -80.0},
            {"regime_at_entry": "bull", "net_pnl": 50.0},
            {"regime_at_entry": "bear", "net_pnl": -20.0},
        ]
        result = compute_regime_conditioned_metrics(trades)
        assert len(result) == 2
        assert result["bull"]["win_rate"] == 1.0
        assert result["bear"]["win_rate"] == 0.0

    def test_empty_trades(self):
        result = compute_regime_conditioned_metrics([])
        assert result == {}

    def test_missing_regime_key(self):
        trades = [{"net_pnl": 100.0}]
        result = compute_regime_conditioned_metrics(trades)
        assert "unknown" in result
