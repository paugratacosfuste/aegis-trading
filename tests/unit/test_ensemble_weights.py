"""Tests for ensemble weight system."""

import pytest

from aegis.ensemble.weights import (
    BASE_TYPE_WEIGHTS,
    REGIME_WEIGHT_ADJUSTMENTS,
    apply_regime_weights,
)


class TestWeights:
    def test_base_weights_sum_to_one(self):
        total = sum(BASE_TYPE_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_normal_regime_no_change(self):
        result = apply_regime_weights(BASE_TYPE_WEIGHTS, "normal")
        for k in BASE_TYPE_WEIGHTS:
            assert result[k] == pytest.approx(BASE_TYPE_WEIGHTS[k], abs=0.001)

    def test_trend_up_boosts_momentum(self):
        normal = apply_regime_weights(BASE_TYPE_WEIGHTS, "normal")
        trend = apply_regime_weights(BASE_TYPE_WEIGHTS, "strong_trend_up")
        assert trend["momentum"] > normal["momentum"]

    def test_mean_reverting_boosts_statistical(self):
        normal = apply_regime_weights(BASE_TYPE_WEIGHTS, "normal")
        mr = apply_regime_weights(BASE_TYPE_WEIGHTS, "mean_reverting")
        assert mr["statistical"] > normal["statistical"]

    def test_crisis_reduces_everything(self):
        crisis = apply_regime_weights(BASE_TYPE_WEIGHTS, "crisis")
        # Weights are re-normalized to sum to 1.0
        total = sum(crisis.values())
        assert total == pytest.approx(1.0)

    def test_weights_always_renormalize(self):
        for regime in REGIME_WEIGHT_ADJUSTMENTS:
            result = apply_regime_weights(BASE_TYPE_WEIGHTS, regime)
            total = sum(result.values())
            assert total == pytest.approx(1.0), f"Regime {regime} doesn't sum to 1.0"

    def test_unknown_regime_acts_as_normal(self):
        result = apply_regime_weights(BASE_TYPE_WEIGHTS, "nonexistent_regime")
        for k in BASE_TYPE_WEIGHTS:
            assert result[k] == pytest.approx(BASE_TYPE_WEIGHTS[k], abs=0.001)

    def test_all_regimes_exist(self):
        expected = {
            # Phase 2 regimes
            "strong_trend_up", "strong_trend_down", "mean_reverting",
            "high_volatility", "low_volatility", "crisis",
            # Phase 3 macro regimes
            "risk_on", "risk_off", "recession_risk",
            "early_cycle", "mid_cycle", "late_cycle",
            "expansion", "contraction",
            "deflationary", "low", "moderate", "high", "very_high",
            "bull", "bear", "transition", "recovery",
        }
        assert set(REGIME_WEIGHT_ADJUSTMENTS.keys()) == expected

    def test_zero_weights_normal_regime(self):
        result = apply_regime_weights({"a": 0.0, "b": 0.0}, "normal")
        assert result == {"a": 0.0, "b": 0.0}

    def test_zero_weights_regime(self):
        result = apply_regime_weights({"a": 0.0, "b": 0.0}, "crisis")
        assert result == {"a": 0.0, "b": 0.0}

    def test_risk_on_boosts_momentum(self):
        normal = apply_regime_weights(BASE_TYPE_WEIGHTS, "normal")
        risk_on = apply_regime_weights(BASE_TYPE_WEIGHTS, "risk_on")
        assert risk_on["momentum"] > normal["momentum"]

    def test_risk_off_boosts_geopolitical(self):
        normal = apply_regime_weights(BASE_TYPE_WEIGHTS, "normal")
        risk_off = apply_regime_weights(BASE_TYPE_WEIGHTS, "risk_off")
        assert risk_off["geopolitical"] > normal["geopolitical"]

    def test_recession_risk_reduces_momentum(self):
        normal = apply_regime_weights(BASE_TYPE_WEIGHTS, "normal")
        recession = apply_regime_weights(BASE_TYPE_WEIGHTS, "recession_risk")
        assert recession["momentum"] < normal["momentum"]
