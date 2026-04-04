"""Contextual bandit for weight allocation using SGDClassifier.

Uses scikit-learn's SGDClassifier with partial_fit for online learning.
Protocol interface allows swapping to VowpalWabbit later.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

from aegis.rl.constants import DEFAULT_EXPLORATION_RATE, NUM_WEIGHT_CONFIGS
from aegis.rl.types import WeightConfig
from aegis.rl.weight_allocator.configs import WEIGHT_CONFIG_MAP, WEIGHT_CONFIGS

logger = logging.getLogger(__name__)


class WeightAllocatorBandit:
    """Contextual bandit that selects weight configurations.

    Uses epsilon-greedy exploration with SGDClassifier as the policy.
    """

    def __init__(
        self,
        exploration_rate: float = DEFAULT_EXPLORATION_RATE,
        random_seed: int = 42,
    ):
        self._exploration_rate = exploration_rate
        self._rng = np.random.default_rng(random_seed)
        self._classes = np.arange(NUM_WEIGHT_CONFIGS)

        # SGDClassifier with log_loss for probabilistic predictions
        self._model = SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=0.001,
            random_state=random_seed,
            warm_start=True,
        )
        self._fitted = False

    def predict(self, context: np.ndarray) -> WeightConfig:
        """Select a weight configuration given context features.

        Uses epsilon-greedy: explore with probability epsilon,
        exploit (use model) otherwise.
        """
        if self._rng.random() < self._exploration_rate or not self._fitted:
            # Explore: random config
            config_id = int(self._rng.integers(0, NUM_WEIGHT_CONFIGS))
        else:
            # Exploit: use model prediction
            x = context.reshape(1, -1)
            config_id = int(self._model.predict(x)[0])

        return WEIGHT_CONFIG_MAP[config_id]

    def update(
        self,
        context: np.ndarray,
        config_id: int,
        reward: float,
    ) -> None:
        """Update the model with observed reward.

        For contextual bandits, we train on (context, best_action) pairs.
        We use the reward to weight the sample: positive rewards reinforce
        the chosen action, negative rewards suppress it.
        """
        x = context.reshape(1, -1)
        y = np.array([config_id])

        # Use sample_weight to incorporate reward signal
        # Positive reward = higher weight for this action
        # Negative reward = we still train (SGD handles it via loss)
        weight = np.array([max(0.1, 1.0 + reward)])

        self._model.partial_fit(x, y, classes=self._classes, sample_weight=weight)
        self._fitted = True

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self._model,
            "fitted": self._fitted,
            "exploration_rate": self._exploration_rate,
        }
        joblib.dump(state, path)
        logger.info("Weight allocator bandit saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        state = joblib.load(path)
        self._model = state["model"]
        self._fitted = state["fitted"]
        self._exploration_rate = state.get("exploration_rate", self._exploration_rate)
        logger.info("Weight allocator bandit loaded from %s", path)

    @property
    def exploration_rate(self) -> float:
        return self._exploration_rate

    @exploration_rate.setter
    def exploration_rate(self, value: float) -> None:
        self._exploration_rate = max(0.0, min(1.0, value))
