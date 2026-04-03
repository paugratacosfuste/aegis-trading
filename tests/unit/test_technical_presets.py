"""Tests for technical indicator presets."""

import pytest

from aegis.agents.technical.presets import INDICATOR_PRESETS, PERIOD_STYLES


class TestPresets:
    def test_all_presets_exist(self):
        expected = {"momentum_fast", "volume_confirmed", "trend_following",
                    "mean_reversion", "full_suite", "multi_confirmation"}
        assert set(INDICATOR_PRESETS.keys()) == expected

    def test_weights_sum_to_one(self):
        for name, preset in INDICATOR_PRESETS.items():
            total = sum(preset.values())
            assert total == pytest.approx(1.0, abs=0.01), f"{name} weights sum to {total}"

    def test_all_period_styles_exist(self):
        expected = {"fast", "standard", "slow", "mixed"}
        assert set(PERIOD_STYLES.keys()) == expected

    def test_fast_periods_shorter_than_slow(self):
        assert PERIOD_STYLES["fast"]["rsi_period"] < PERIOD_STYLES["slow"]["rsi_period"]
        assert PERIOD_STYLES["fast"]["ema_short"] < PERIOD_STYLES["slow"]["ema_short"]

    def test_presets_are_immutable(self):
        with pytest.raises(TypeError):
            INDICATOR_PRESETS["full_suite"]["rsi"] = 0.99
