"""Tests for position sizing Gym environment."""

import numpy as np
import pytest

from aegis.rl.constants import MAX_POSITION_SIZE, POSITION_OBS_DIM
from aegis.rl.position_sizer.env import PositionSizingEnv


def _make_trade_history(n: int = 20) -> list[dict]:
    rng = np.random.default_rng(42)
    return [
        {
            "obs": rng.standard_normal(POSITION_OBS_DIM).astype(np.float32),
            "pnl": rng.normal(0.0, 50.0),
            "kelly_size": 0.03,
        }
        for _ in range(n)
    ]


class TestPositionSizingEnv:
    def test_check_env(self):
        """Gymnasium env checker passes."""
        from gymnasium.utils.env_checker import check_env

        env = PositionSizingEnv(trade_history=_make_trade_history())
        check_env(env, skip_render_check=True)

    def test_observation_space_shape(self):
        env = PositionSizingEnv()
        obs, _ = env.reset()
        assert obs.shape == (POSITION_OBS_DIM,)
        assert obs.dtype == np.float32

    def test_action_space_shape(self):
        env = PositionSizingEnv()
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == 0.0
        assert env.action_space.high[0] == MAX_POSITION_SIZE

    def test_single_step_episode(self):
        env = PositionSizingEnv(trade_history=_make_trade_history())
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs_next, reward, terminated, truncated, info = env.step(action)
        assert terminated is True
        assert truncated is False
        assert isinstance(reward, float)

    def test_reward_sign_winning_trade(self):
        """Positive PnL trade should give positive reward."""
        history = [{"obs": np.zeros(POSITION_OBS_DIM, dtype=np.float32), "pnl": 100.0, "kelly_size": 0.03}]
        env = PositionSizingEnv(trade_history=history)
        env.reset()
        _, reward, _, _, _ = env.step(np.array([0.03], dtype=np.float32))
        assert reward > 0

    def test_reward_sign_losing_trade(self):
        """Negative PnL trade should give negative reward."""
        history = [{"obs": np.zeros(POSITION_OBS_DIM, dtype=np.float32), "pnl": -100.0, "kelly_size": 0.03}]
        env = PositionSizingEnv(trade_history=history)
        env.reset()
        _, reward, _, _, _ = env.step(np.array([0.03], dtype=np.float32))
        assert reward < 0

    def test_empty_history_works(self):
        env = PositionSizingEnv()
        obs, _ = env.reset()
        assert obs.shape == (POSITION_OBS_DIM,)
        _, reward, _, _, _ = env.step(np.array([0.05], dtype=np.float32))
        assert isinstance(reward, float)
