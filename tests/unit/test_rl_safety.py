"""Tests for RL safety module."""

import pytest

from aegis.rl.common.safety import (
    clamp_position_size,
    enforce_circuit_breaker,
    validate_weight_config,
)
from aegis.rl.constants import AGENT_TYPES, MAX_POSITION_SIZE, MIN_POSITION_SIZE
from aegis.rl.types import WeightConfig


class TestClampPositionSize:
    def test_within_bounds_unchanged(self):
        result = clamp_position_size(0.05, kelly_size=0.05, portfolio_value=10000)
        assert result == 0.05

    def test_clamps_to_max(self):
        result = clamp_position_size(0.20, kelly_size=0.20, portfolio_value=10000)
        assert result == MAX_POSITION_SIZE

    def test_clamps_to_min(self):
        result = clamp_position_size(0.001, kelly_size=0.001, portfolio_value=10000)
        assert result == MIN_POSITION_SIZE

    def test_kelly_divergence_upper(self):
        # RL wants 0.10, Kelly says 0.04 -> max allowed = 0.04 * 1.5 = 0.06
        result = clamp_position_size(0.10, kelly_size=0.04, portfolio_value=10000)
        assert result == pytest.approx(0.06, abs=0.001)

    def test_kelly_divergence_lower(self):
        # RL wants 0.01, Kelly says 0.06 -> min allowed = 0.06 * 0.5 = 0.03
        result = clamp_position_size(0.01, kelly_size=0.06, portfolio_value=10000)
        assert result == pytest.approx(0.03, abs=0.001)

    def test_zero_kelly_no_divergence_check(self):
        # When Kelly is 0, skip divergence check, just apply absolute bounds
        result = clamp_position_size(0.05, kelly_size=0.0, portfolio_value=10000)
        assert result == 0.05


class TestValidateWeightConfig:
    def _valid_weights(self) -> dict[str, float]:
        n = len(AGENT_TYPES)
        return {t: 1.0 / n for t in AGENT_TYPES}

    def test_valid_config(self):
        wc = WeightConfig(config_id=0, name="test", weights=self._valid_weights())
        assert validate_weight_config(wc) is True

    def test_rejects_missing_type(self):
        weights = self._valid_weights()
        del weights["crypto"]
        weights["technical"] += weights.get("crypto", 0)
        # Need to re-normalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        wc = WeightConfig(config_id=1, name="incomplete", weights=weights)
        assert validate_weight_config(wc) is False

    def test_rejects_bad_sum(self):
        weights = {t: 0.1 for t in AGENT_TYPES}  # sums to 0.7
        # WeightConfig itself raises, so test validate on a manually constructed one
        # Use object.__new__ to bypass __post_init__
        wc = object.__new__(WeightConfig)
        object.__setattr__(wc, "config_id", 2)
        object.__setattr__(wc, "name", "bad_sum")
        object.__setattr__(wc, "weights", weights)
        assert validate_weight_config(wc) is False


class TestEnforceCircuitBreaker:
    def test_normal_allows(self):
        assert enforce_circuit_breaker(9800, 10000) is True

    def test_halt_on_drawdown(self):
        # -6% daily
        assert enforce_circuit_breaker(9400, 10000) is False

    def test_exact_threshold_halts(self):
        # -5% exactly
        assert enforce_circuit_breaker(9500, 10000) is False

    def test_zero_sod_halts(self):
        assert enforce_circuit_breaker(10000, 0) is False

    def test_custom_threshold(self):
        # -3% with custom -2% threshold
        assert enforce_circuit_breaker(9700, 10000, daily_halt_threshold=-0.02) is False
