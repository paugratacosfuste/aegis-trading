"""Tests for exit management Gym environment."""

import numpy as np
import pytest

from aegis.rl.constants import EXIT_OBS_DIM, NUM_EXIT_ACTIONS
from aegis.rl.exit_manager.env import ExitManagementEnv
from aegis.rl.types import ExitAction


def _make_sequences(n_trades: int = 5, bars_per: int = 30) -> list[list[dict]]:
    rng = np.random.default_rng(42)
    sequences = []
    for _ in range(n_trades):
        bars = []
        cum_r = 0.0
        for b in range(bars_per):
            pnl_change = rng.normal(0.0, 0.01)
            cum_r += pnl_change * 10
            bars.append({
                "obs": rng.standard_normal(EXIT_OBS_DIM).astype(np.float32),
                "r_multiple": cum_r,
                "pnl_change": pnl_change,
            })
        sequences.append(bars)
    return sequences


class TestExitManagementEnv:
    def test_check_env(self):
        """Gymnasium env checker passes."""
        from gymnasium.utils.env_checker import check_env

        env = ExitManagementEnv(trade_sequences=_make_sequences())
        check_env(env, skip_render_check=True)

    def test_observation_space(self):
        env = ExitManagementEnv()
        obs, _ = env.reset()
        assert obs.shape == (EXIT_OBS_DIM,)
        assert obs.dtype == np.float32

    def test_action_space(self):
        env = ExitManagementEnv()
        assert env.action_space.n == NUM_EXIT_ACTIONS

    def test_hold_continues_episode(self):
        env = ExitManagementEnv(trade_sequences=_make_sequences())
        env.reset()
        _, _, terminated, _, _ = env.step(ExitAction.HOLD)
        assert terminated is False  # Should continue

    def test_full_exit_terminates(self):
        env = ExitManagementEnv(trade_sequences=_make_sequences())
        env.reset()
        _, _, terminated, _, _ = env.step(ExitAction.FULL_EXIT)
        assert terminated is True

    def test_partial_reduces_position(self):
        env = ExitManagementEnv(trade_sequences=_make_sequences())
        env.reset()
        env.step(ExitAction.PARTIAL_50)
        assert env._remaining_pct == pytest.approx(0.5)
        env.step(ExitAction.PARTIAL_50)
        assert env._remaining_pct == pytest.approx(0.0)

    def test_episode_lifecycle(self):
        """Run a full episode with various actions."""
        env = ExitManagementEnv(trade_sequences=_make_sequences(n_trades=3, bars_per=10))
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
            if steps > 200:
                break

        assert steps >= 1
        assert isinstance(total_reward, float)

    def test_max_bars_terminates(self):
        """Episode terminates at MAX_BARS."""
        long_seq = [{
            "obs": np.zeros(EXIT_OBS_DIM, dtype=np.float32),
            "r_multiple": 0.0, "pnl_change": 0.0,
        } for _ in range(300)]
        env = ExitManagementEnv(trade_sequences=[long_seq])
        env.reset()

        for _ in range(200):
            _, _, terminated, _, _ = env.step(ExitAction.HOLD)
            if terminated:
                break

        assert terminated is True
