"""Gymnasium environment for exit management.

Multi-step episode per trade. The agent observes position state each bar
and chooses an exit action (HOLD, TIGHTEN_STOP, PARTIAL_25, PARTIAL_50, FULL_EXIT).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aegis.rl.constants import EXIT_OBS_DIM, NUM_EXIT_ACTIONS
from aegis.rl.types import ExitAction


class ExitManagementEnv(gym.Env):
    """Exit management environment.

    Observation: Box(EXIT_OBS_DIM,) — position + market features
    Action: Discrete(5) — {HOLD, TIGHTEN_STOP, PARTIAL_25, PARTIAL_50, FULL_EXIT}
    Episode: multiple steps (one per bar until exit or max bars)
    """

    metadata = {"render_modes": []}

    MAX_BARS = 200

    def __init__(self, trade_sequences: list[list[dict]] | None = None):
        """Initialize with optional trade sequences for replay.

        trade_sequences: list of trades, each trade is a list of bar dicts:
            obs: np.ndarray (EXIT_OBS_DIM,)
            r_multiple: float
            pnl_change: float (PnL change this bar)
        """
        super().__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(EXIT_OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_EXIT_ACTIONS)

        self._sequences = trade_sequences or []
        self._current_seq: list[dict] = []
        self._step_idx = 0
        self._remaining_pct = 1.0  # Fraction of position still open
        self._cumulative_reward = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self._sequences:
            seq_idx = self.np_random.integers(0, len(self._sequences))
            self._current_seq = self._sequences[seq_idx]
        else:
            # Generate a dummy sequence
            self._current_seq = [
                {
                    "obs": np.zeros(EXIT_OBS_DIM, dtype=np.float32),
                    "r_multiple": 0.0,
                    "pnl_change": 0.0,
                }
                for _ in range(50)
            ]

        self._step_idx = 0
        self._remaining_pct = 1.0
        self._cumulative_reward = 0.0

        return self._get_obs(), {}

    def step(self, action: int):
        bar = self._current_seq[self._step_idx] if self._step_idx < len(self._current_seq) else self._current_seq[-1]
        r_multiple = bar["r_multiple"]
        pnl_change = bar["pnl_change"] * self._remaining_pct

        # Base reward: PnL change for remaining position
        reward = pnl_change
        hold_cost = -0.001

        if action == ExitAction.FULL_EXIT:
            # Bonus for exiting at good R
            if r_multiple >= 2.0:
                reward += 0.5
            elif r_multiple >= 1.0:
                reward += 0.2
            elif r_multiple <= -1.0:
                reward -= 0.1
            self._remaining_pct = 0.0

        elif action == ExitAction.PARTIAL_50:
            exit_pct = min(0.50, self._remaining_pct)
            if r_multiple >= 1.5:
                reward += 0.3 * exit_pct
            else:
                reward -= 0.05
            self._remaining_pct -= exit_pct

        elif action == ExitAction.PARTIAL_25:
            exit_pct = min(0.25, self._remaining_pct)
            if r_multiple >= 1.0:
                reward += 0.15 * exit_pct
            else:
                reward -= 0.03
            self._remaining_pct -= exit_pct

        elif action == ExitAction.TIGHTEN_STOP:
            if r_multiple > 0.5:
                reward += 0.05
            else:
                reward -= 0.02
            reward += hold_cost

        else:  # HOLD
            # Penalty for holding losers too long
            if r_multiple < -0.5 and self._step_idx > 48:
                reward -= 0.1
            reward += hold_cost

        self._step_idx += 1
        self._cumulative_reward += reward

        # Episode ends on full exit, position depleted, or max bars
        terminated = (
            self._remaining_pct <= 0.01
            or self._step_idx >= len(self._current_seq)
            or self._step_idx >= self.MAX_BARS
        )
        truncated = False

        return self._get_obs(), float(reward), terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        if self._step_idx < len(self._current_seq):
            return self._current_seq[self._step_idx]["obs"].astype(np.float32)
        return np.zeros(EXIT_OBS_DIM, dtype=np.float32)
