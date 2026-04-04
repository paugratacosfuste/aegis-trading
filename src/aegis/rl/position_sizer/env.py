"""Gymnasium environment for position sizing.

Each episode = one trade decision. The agent observes market context
and outputs a position size (fraction of portfolio, 0 to MAX_POSITION_SIZE).
Reward is based on risk-adjusted PnL of the resulting trade.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aegis.rl.constants import MAX_POSITION_SIZE, POSITION_OBS_DIM


class PositionSizingEnv(gym.Env):
    """Position sizing environment.

    Observation: Box(POSITION_OBS_DIM,) — market features
    Action: Box(1,) — position size as fraction of portfolio [0, MAX_POSITION_SIZE]
    Episode: single step (one trade decision → reward)
    """

    metadata = {"render_modes": []}

    def __init__(self, trade_history: list[dict] | None = None):
        """Initialize with optional trade history for replay.

        trade_history: list of dicts with keys:
            obs: np.ndarray (POSITION_OBS_DIM,)
            pnl: float (trade PnL)
            kelly_size: float (what Kelly formula suggested)
        """
        super().__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(POSITION_OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=MAX_POSITION_SIZE,
            shape=(1,), dtype=np.float32,
        )

        self._trade_history = trade_history or []
        self._current_idx = 0
        self._current_obs: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self._trade_history:
            self._current_idx = self.np_random.integers(0, len(self._trade_history))
            self._current_obs = self._trade_history[self._current_idx]["obs"].astype(np.float32)
        else:
            self._current_obs = np.zeros(POSITION_OBS_DIM, dtype=np.float32)

        return self._current_obs, {}

    def step(self, action: np.ndarray):
        position_size = float(np.clip(action[0], 0.0, MAX_POSITION_SIZE))

        # Get trade outcome
        if self._trade_history and self._current_idx < len(self._trade_history):
            trade = self._trade_history[self._current_idx]
            pnl = trade["pnl"]
            kelly_size = trade.get("kelly_size", 0.03)
        else:
            pnl = 0.0
            kelly_size = 0.03

        # Reward: PnL scaled by chosen size, minus risk penalty
        reward = pnl * (position_size / kelly_size) if kelly_size > 0 else 0.0

        # Penalty for deviating too far from Kelly
        deviation = abs(position_size - kelly_size) / kelly_size if kelly_size > 0 else 0.0
        if deviation > 0.5:
            reward -= deviation * 0.1

        terminated = True  # Single-step episode
        truncated = False

        return self._current_obs, float(reward), terminated, truncated, {}
