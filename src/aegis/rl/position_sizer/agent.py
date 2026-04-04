"""PPO agent wrapper for position sizing."""

import logging
from pathlib import Path

import numpy as np

from aegis.rl.common.safety import clamp_position_size
from aegis.rl.position_sizer.env import PositionSizingEnv

logger = logging.getLogger(__name__)


class PositionSizerAgent:
    """Wraps Stable-Baselines3 PPO for position sizing."""

    def __init__(self, trade_history: list[dict] | None = None):
        self._env = PositionSizingEnv(trade_history=trade_history)
        self._model = None

    def train(self, total_timesteps: int = 10000) -> None:
        """Train PPO on trade history."""
        from stable_baselines3 import PPO

        self._model = PPO(
            "MlpPolicy",
            self._env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=10,
            verbose=0,
        )
        self._model.learn(total_timesteps=total_timesteps)
        logger.info("Position sizer trained for %d timesteps", total_timesteps)

    def predict(self, obs: np.ndarray, kelly_size: float = 0.03) -> float:
        """Predict position size from observation.

        Returns clamped position size (fraction of portfolio).
        """
        if self._model is None:
            return kelly_size  # Fallback to Kelly before training

        action, _ = self._model.predict(obs, deterministic=True)
        raw_size = float(action[0])
        return clamp_position_size(raw_size, kelly_size, portfolio_value=1.0)

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        if self._model is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._model.save(str(path))
            logger.info("Position sizer saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        from stable_baselines3 import PPO

        self._model = PPO.load(str(path), env=self._env)
        logger.info("Position sizer loaded from %s", path)
