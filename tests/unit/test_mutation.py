"""Tests for cohort mutation."""

import numpy as np
import pytest

from aegis.lab.mutation import generate_random_cohort, mutate_cohort
from aegis.lab.types import CohortConfig, CohortStatus, StrategyCohort


def _parent():
    return StrategyCohort(
        cohort_id="cohort_A", name="Baseline",
        status=CohortStatus.PROMOTED,
        config=CohortConfig(
            agent_weights={
                "technical": 0.25, "statistical": 0.20, "momentum": 0.20,
                "sentiment": 0.15, "geopolitical": 0.05, "world_leader": 0.05,
                "crypto": 0.10,
            },
            confidence_threshold=0.45,
            risk_params={"max_risk_per_trade": 0.05, "kelly_fraction": 0.5},
            universe=("BTC/USDT",),
            invert_sentiment=True,
        ),
        generation=1,
    )


class TestMutateCohort:
    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        child = mutate_cohort(_parent(), rng=rng)
        total = sum(child.config.agent_weights.values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_generation_incremented(self):
        child = mutate_cohort(_parent(), rng=np.random.default_rng(42))
        assert child.generation == 2

    def test_parent_id_set(self):
        child = mutate_cohort(_parent(), rng=np.random.default_rng(42))
        assert child.parent_cohort_id == "cohort_A"

    def test_status_is_created(self):
        child = mutate_cohort(_parent(), rng=np.random.default_rng(42))
        assert child.status == CohortStatus.CREATED

    def test_weights_bounded(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            child = mutate_cohort(_parent(), rng=rng)
            for w in child.config.agent_weights.values():
                assert 0.0 < w <= 0.5  # After renormalization

    def test_threshold_bounded(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            child = mutate_cohort(_parent(), rng=rng)
            assert 0.25 <= child.config.confidence_threshold <= 0.75

    def test_risk_bounded(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            child = mutate_cohort(_parent(), rng=rng)
            mr = child.config.risk_params.get("max_risk_per_trade", 0.05)
            assert 0.005 <= mr <= 0.10

    def test_preserves_flags(self):
        child = mutate_cohort(_parent(), rng=np.random.default_rng(42))
        assert child.config.invert_sentiment is True
        assert child.config.universe == ("BTC/USDT",)

    def test_deterministic_with_seed(self):
        c1 = mutate_cohort(_parent(), rng=np.random.default_rng(42))
        c2 = mutate_cohort(_parent(), rng=np.random.default_rng(42))
        assert c1.config.confidence_threshold == c2.config.confidence_threshold
        assert c1.config.agent_weights == c2.config.agent_weights

    def test_unique_id(self):
        c1 = mutate_cohort(_parent(), rng=np.random.default_rng(42))
        c2 = mutate_cohort(_parent(), rng=np.random.default_rng(99))
        assert c1.cohort_id != c2.cohort_id


class TestGenerateRandomCohort:
    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        cohort = generate_random_cohort(rng=rng)
        total = sum(cohort.config.agent_weights.values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_seven_weight_types(self):
        cohort = generate_random_cohort(rng=np.random.default_rng(42))
        assert len(cohort.config.agent_weights) == 7

    def test_threshold_bounded(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            cohort = generate_random_cohort(rng=rng)
            assert 0.30 <= cohort.config.confidence_threshold <= 0.65

    def test_status_created(self):
        cohort = generate_random_cohort(rng=np.random.default_rng(42))
        assert cohort.status == CohortStatus.CREATED

    def test_universe_passed_through(self):
        cohort = generate_random_cohort(
            universe=("BTC/USDT",), rng=np.random.default_rng(42)
        )
        assert cohort.config.universe == ("BTC/USDT",)
