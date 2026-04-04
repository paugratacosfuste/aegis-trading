"""Tests for RL feature builder."""

import numpy as np
import pytest

from aegis.rl.common.feature_builder import (
    build_exit_obs,
    build_position_obs,
    build_weight_context,
)
from aegis.rl.constants import EXIT_OBS_DIM, POSITION_OBS_DIM, WEIGHT_CONTEXT_DIM


class TestBuildWeightContext:
    def test_shape_and_dtype(self, sample_agent_signal):
        obs = build_weight_context(
            signals=[sample_agent_signal],
            regime="bull",
            portfolio_value=10000.0,
        )
        assert obs.shape == (WEIGHT_CONTEXT_DIM,)
        assert obs.dtype == np.float32

    def test_empty_signals(self):
        obs = build_weight_context([], "normal", 10000.0)
        assert np.all(obs == 0.0)

    def test_regime_one_hot(self, sample_agent_signal):
        for regime, idx in [("bull", 4), ("bear", 5), ("transition", 6), ("recovery", 7)]:
            obs = build_weight_context([sample_agent_signal], regime, 10000.0)
            assert obs[idx] == 1.0
            # Other regime slots should be 0
            other_idxs = [4, 5, 6, 7]
            other_idxs.remove(idx)
            for oi in other_idxs:
                assert obs[oi] == 0.0

    def test_normal_regime_no_one_hot(self, sample_agent_signal):
        obs = build_weight_context([sample_agent_signal], "normal", 10000.0)
        assert all(obs[i] == 0.0 for i in range(4, 8))

    def test_no_nan(self, sample_agent_signal):
        obs = build_weight_context(
            [sample_agent_signal], "bull", 10000.0,
            equity_curve=[10000, 10100, 10050, 9900],
        )
        assert not np.any(np.isnan(obs))

    def test_drawdown_computed(self, sample_agent_signal):
        curve = list(range(10000, 10100)) + [9900]  # Peak 10099, current 9900
        obs = build_weight_context([sample_agent_signal], "normal", 9900.0, equity_curve=curve)
        assert obs[8] > 0.0  # Drawdown should be positive


class TestBuildPositionObs:
    def test_shape_and_dtype(self, sample_candles_uptrend):
        obs = build_position_obs(
            candles=sample_candles_uptrend,
            signal_confidence=0.7,
            signal_direction=0.5,
            portfolio_value=10000.0,
            open_positions=2,
            regime="bull",
            atr_14=500.0,
        )
        assert obs.shape == (POSITION_OBS_DIM,)
        assert obs.dtype == np.float32

    def test_confidence_and_direction(self, sample_candles_uptrend):
        obs = build_position_obs(
            sample_candles_uptrend, 0.85, -0.6, 10000.0, 0, "normal", 500.0,
        )
        assert obs[0] == pytest.approx(0.85)
        assert obs[1] == pytest.approx(-0.6)

    def test_no_nan(self, sample_candles_uptrend):
        obs = build_position_obs(
            sample_candles_uptrend, 0.7, 0.5, 10000.0, 2, "bear", 500.0,
        )
        assert not np.any(np.isnan(obs))

    def test_empty_candles(self):
        obs = build_position_obs([], 0.5, 0.5, 10000.0, 0, "normal", 500.0)
        assert obs[0] == 0.5
        assert obs[2] == 0.0  # No return data


class TestBuildExitObs:
    def test_shape_and_dtype(self, sample_candles_uptrend):
        pos = {
            "entry_price": 40000.0, "stop_loss": 39000.0, "risk_amount": 1000.0,
            "direction": "LONG", "entry_index": 0, "partial_taken": False,
        }
        obs = build_exit_obs(sample_candles_uptrend, pos, atr_14=500.0)
        assert obs.shape == (EXIT_OBS_DIM,)
        assert obs.dtype == np.float32

    def test_r_multiple_positive_long(self, sample_candles_uptrend):
        # Uptrend candles end around 42900 (40000 + 29*100 + 30)
        pos = {
            "entry_price": 40000.0, "stop_loss": 39000.0, "risk_amount": 1000.0,
            "direction": "LONG", "entry_index": 0, "partial_taken": False,
        }
        obs = build_exit_obs(sample_candles_uptrend, pos, atr_14=500.0)
        assert obs[0] > 0.0  # Positive R-multiple (price above entry)

    def test_partial_taken_flag(self, sample_candles_uptrend):
        pos = {
            "entry_price": 40000.0, "stop_loss": 39000.0, "risk_amount": 1000.0,
            "direction": "LONG", "entry_index": 0, "partial_taken": True,
        }
        obs = build_exit_obs(sample_candles_uptrend, pos, atr_14=500.0)
        assert obs[3] == 1.0

    def test_empty_candles(self):
        pos = {"entry_price": 100.0, "stop_loss": 90.0, "risk_amount": 10.0,
               "direction": "LONG", "entry_index": 0, "partial_taken": False}
        obs = build_exit_obs([], pos, atr_14=5.0)
        assert obs.shape == (EXIT_OBS_DIM,)
        assert not np.any(np.isnan(obs))

    def test_no_nan(self, sample_candles_uptrend):
        pos = {
            "entry_price": 40000.0, "stop_loss": 39000.0, "risk_amount": 1000.0,
            "direction": "SHORT", "entry_index": 0, "partial_taken": False,
        }
        obs = build_exit_obs(sample_candles_uptrend, pos, atr_14=500.0)
        assert not np.any(np.isnan(obs))
