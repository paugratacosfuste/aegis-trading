"""Tests for RL types: frozen dataclasses, enums, validation."""

from datetime import datetime, timezone

import pytest

from aegis.rl.types import (
    ExitAction,
    PromotionStage,
    RLPrediction,
    ShadowResult,
    ShadowSummary,
    WeightConfig,
)


class TestPromotionStage:
    def test_all_stages_exist(self):
        assert set(PromotionStage) == {
            PromotionStage.TRAINING,
            PromotionStage.SHADOW,
            PromotionStage.CANDIDATE,
            PromotionStage.ACTIVE,
            PromotionStage.RETIRED,
        }

    def test_string_values(self):
        assert PromotionStage.SHADOW == "shadow"
        assert PromotionStage.ACTIVE == "active"


class TestExitAction:
    def test_all_actions_exist(self):
        assert len(ExitAction) == 5

    def test_integer_values(self):
        assert ExitAction.HOLD == 0
        assert ExitAction.FULL_EXIT == 4


class TestWeightConfig:
    def test_valid_config(self):
        wc = WeightConfig(
            config_id=0,
            name="baseline",
            weights={
                "technical": 0.25,
                "statistical": 0.20,
                "momentum": 0.20,
                "sentiment": 0.15,
                "geopolitical": 0.05,
                "world_leader": 0.05,
                "crypto": 0.10,
            },
        )
        assert wc.config_id == 0
        assert abs(sum(wc.weights.values()) - 1.0) < 0.001

    def test_frozen(self):
        wc = WeightConfig(config_id=0, name="test", weights={"a": 0.5, "b": 0.5})
        with pytest.raises(AttributeError):
            wc.config_id = 1  # type: ignore[misc]

    def test_rejects_bad_sum(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            WeightConfig(config_id=0, name="bad", weights={"a": 0.5})

    def test_rejects_negative_weights(self):
        with pytest.raises(ValueError, match="non-negative"):
            WeightConfig(config_id=0, name="bad", weights={"a": 1.5, "b": -0.5})


class TestRLPrediction:
    def test_creation(self):
        ts = datetime(2025, 6, 1, tzinfo=timezone.utc)
        pred = RLPrediction(
            component="weight_allocator",
            timestamp=ts,
            symbol="BTC/USDT",
            prediction={"config_id": 5},
            context_features={"vol": 0.3},
            model_version="v1",
        )
        assert pred.mode == "shadow"
        assert pred.component == "weight_allocator"

    def test_frozen(self):
        ts = datetime(2025, 6, 1, tzinfo=timezone.utc)
        pred = RLPrediction(
            component="test", timestamp=ts, symbol="X",
            prediction={}, context_features={}, model_version="v1",
        )
        with pytest.raises(AttributeError):
            pred.mode = "active"  # type: ignore[misc]


class TestShadowResult:
    def test_creation(self):
        ts = datetime(2025, 6, 1, tzinfo=timezone.utc)
        pred = RLPrediction(
            component="position_sizer", timestamp=ts, symbol="BTC/USDT",
            prediction={"size": 0.05}, context_features={}, model_version="v1",
        )
        result = ShadowResult(
            component="position_sizer",
            prediction=pred,
            baseline_value=0.03,
            rl_value=0.05,
            actual_outcome=100.0,
            rl_would_have_been=120.0,
            timestamp=ts,
        )
        assert result.rl_would_have_been > result.actual_outcome


class TestShadowSummary:
    def test_creation(self):
        summary = ShadowSummary(
            component="weight_allocator",
            total_predictions=1000,
            baseline_cumulative=500.0,
            rl_cumulative=600.0,
            baseline_sharpe=0.5,
            rl_sharpe=0.7,
            outperformance_days=70,
            total_days=90,
            stage=PromotionStage.SHADOW,
        )
        assert summary.rl_cumulative > summary.baseline_cumulative
