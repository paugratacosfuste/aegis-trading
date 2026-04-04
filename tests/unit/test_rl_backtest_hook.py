"""Tests for backtest shadow hook integration."""

import pytest

from aegis.rl.integration.backtest_hook import BacktestShadowHook


class TestBacktestShadowHook:
    def test_disabled_by_default(self):
        hook = BacktestShadowHook()
        hook.setup()
        assert hook.tracker is None
        assert hook.get_summary() == {}

    def test_enabled_creates_tracker(self):
        config = {
            "enabled": True,
            "components": {
                "weight_allocator": {"enabled": True, "exploration_rate": 0.2},
            },
        }
        hook = BacktestShadowHook(rl_config=config)
        hook.setup()
        assert hook.tracker is not None

    def test_all_components_enabled(self):
        config = {
            "enabled": True,
            "components": {
                "weight_allocator": {"enabled": True, "exploration_rate": 0.1},
                "position_sizer": {"enabled": True},
                "exit_manager": {"enabled": True},
            },
        }
        hook = BacktestShadowHook(rl_config=config)
        hook.setup()
        tracker = hook.tracker
        assert tracker is not None
        assert tracker._weight_bandit is not None
        assert tracker._position_sizer is not None
        assert tracker._exit_manager is not None

    def test_engine_with_shadow_hook(self, sample_candles_uptrend):
        """BacktestEngine runs with shadow hook without errors."""
        from aegis.backtest.engine import BacktestEngine

        config = {
            "enabled": True,
            "components": {
                "weight_allocator": {"enabled": True, "exploration_rate": 1.0},
            },
        }
        hook = BacktestShadowHook(rl_config=config)
        hook.setup()

        engine = BacktestEngine(
            initial_capital=10000.0,
            confidence_threshold=0.1,
            shadow_hook=hook,
        )
        results = engine.run(sample_candles_uptrend)

        # Should have shadow_summary in results
        assert "shadow_summary" in results
        assert "total_predictions" in results["shadow_summary"]

    def test_engine_without_hook_no_regression(self, sample_candles_uptrend):
        """Engine without shadow hook still works identically."""
        from aegis.backtest.engine import BacktestEngine

        engine = BacktestEngine(initial_capital=10000.0)
        results = engine.run(sample_candles_uptrend)
        assert "shadow_summary" not in results
        assert "metrics" in results
