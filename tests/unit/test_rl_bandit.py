"""Tests for weight allocator contextual bandit."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aegis.rl.constants import WEIGHT_CONTEXT_DIM
from aegis.rl.types import WeightConfig
from aegis.rl.weight_allocator.bandit import WeightAllocatorBandit


class TestWeightAllocatorBandit:
    def _random_context(self) -> np.ndarray:
        return np.random.randn(WEIGHT_CONTEXT_DIM).astype(np.float32)

    def test_predict_returns_weight_config(self):
        bandit = WeightAllocatorBandit(exploration_rate=1.0)
        config = bandit.predict(self._random_context())
        assert isinstance(config, WeightConfig)
        assert 0 <= config.config_id < 50

    def test_predict_with_exploitation(self):
        bandit = WeightAllocatorBandit(exploration_rate=0.0, random_seed=42)
        ctx = self._random_context()
        # Train on a few examples
        for _ in range(10):
            bandit.update(ctx, config_id=5, reward=1.0)
        # Should exploit learned preference
        config = bandit.predict(ctx)
        assert isinstance(config, WeightConfig)

    def test_update_makes_model_fitted(self):
        bandit = WeightAllocatorBandit()
        assert not bandit._fitted
        bandit.update(self._random_context(), config_id=0, reward=0.5)
        assert bandit._fitted

    def test_exploration_rate_bounds(self):
        bandit = WeightAllocatorBandit()
        bandit.exploration_rate = 1.5
        assert bandit.exploration_rate == 1.0
        bandit.exploration_rate = -0.5
        assert bandit.exploration_rate == 0.0

    def test_save_load_roundtrip(self):
        bandit = WeightAllocatorBandit(exploration_rate=0.2)
        ctx = self._random_context()
        bandit.update(ctx, config_id=3, reward=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bandit.joblib"
            bandit.save(path)

            loaded = WeightAllocatorBandit()
            loaded.load(path)

            assert loaded._fitted is True
            assert loaded.exploration_rate == 0.2

            # Both should make the same prediction with exploitation
            loaded.exploration_rate = 0.0
            bandit.exploration_rate = 0.0
            assert bandit.predict(ctx).config_id == loaded.predict(ctx).config_id

    def test_many_predictions_cover_range(self):
        """With full exploration, should sample diverse configs."""
        bandit = WeightAllocatorBandit(exploration_rate=1.0, random_seed=42)
        ids = set()
        for _ in range(200):
            config = bandit.predict(self._random_context())
            ids.add(config.config_id)
        assert len(ids) > 20  # Should cover a good range
