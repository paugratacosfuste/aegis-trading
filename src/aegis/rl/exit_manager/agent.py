"""DQN agent wrapper for exit management."""

import logging
from pathlib import Path

import numpy as np

from aegis.rl.exit_manager.env import ExitManagementEnv
from aegis.rl.types import ExitAction

logger = logging.getLogger(__name__)


class ExitManagerAgent:
    """Wraps Stable-Baselines3 DQN for exit management."""

    def __init__(self, trade_sequences: list[list[dict]] | None = None):
        self._env = ExitManagementEnv(trade_sequences=trade_sequences)
        self._model = None

    def train(self, total_timesteps: int = 10000) -> None:
        """Train DQN on trade sequences."""
        from stable_baselines3 import DQN

        self._model = DQN(
            "MlpPolicy",
            self._env,
            learning_rate=1e-3,
            buffer_size=5000,
            learning_starts=100,
            batch_size=32,
            exploration_fraction=0.3,
            verbose=0,
        )
        self._model.learn(total_timesteps=total_timesteps)
        logger.info("Exit manager trained for %d timesteps", total_timesteps)

    def predict(self, obs: np.ndarray) -> ExitAction:
        """Predict exit action from observation."""
        if self._model is None:
            return ExitAction.HOLD  # Fallback before training

        action, _ = self._model.predict(obs, deterministic=True)
        return ExitAction(int(action))

    def save(self, path: str | Path) -> None:
        if self._model is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._model.save(str(path))
            logger.info("Exit manager saved to %s", path)

    def load(self, path: str | Path) -> None:
        from stable_baselines3 import DQN

        self._model = DQN.load(str(path), env=self._env)
        logger.info("Exit manager loaded from %s", path)
