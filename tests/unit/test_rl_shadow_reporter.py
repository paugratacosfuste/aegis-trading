"""Tests for shadow reporter and promotion criteria."""

import pytest

from aegis.rl.shadow.reporter import check_promotion_criteria, generate_report
from aegis.rl.types import PromotionStage, ShadowSummary


class TestGenerateReport:
    def test_empty_predictions(self):
        report = generate_report([])
        assert report == {}

    def test_groups_by_component(self):
        preds = [
            {"component": "weight_allocator", "prediction": {"config_id": 1}},
            {"component": "weight_allocator", "prediction": {"config_id": 2}},
            {"component": "position_sizer", "prediction": {"size": 0.05}},
            {"component": "trade_outcome", "trade": {}},  # Excluded
        ]
        report = generate_report(preds)
        assert "weight_allocator" in report
        assert report["weight_allocator"]["total_predictions"] == 2
        assert "position_sizer" in report
        assert "trade_outcome" not in report


class TestCheckPromotionCriteria:
    def test_meets_all_criteria(self):
        summary = ShadowSummary(
            component="weight_allocator",
            total_predictions=1000,
            baseline_cumulative=500.0,
            rl_cumulative=700.0,
            baseline_sharpe=0.3,
            rl_sharpe=0.5,  # Margin = 0.2 > 0.1 threshold
            outperformance_days=70,
            total_days=95,  # > 90 days
            stage=PromotionStage.SHADOW,
        )
        assert check_promotion_criteria(summary) is True

    def test_not_enough_days(self):
        summary = ShadowSummary(
            component="weight_allocator",
            total_predictions=500,
            baseline_cumulative=500.0, rl_cumulative=700.0,
            baseline_sharpe=0.3, rl_sharpe=0.5,
            outperformance_days=50,
            total_days=60,  # < 90
            stage=PromotionStage.SHADOW,
        )
        assert check_promotion_criteria(summary) is False

    def test_insufficient_sharpe_margin(self):
        summary = ShadowSummary(
            component="position_sizer",
            total_predictions=1000,
            baseline_cumulative=500.0, rl_cumulative=510.0,
            baseline_sharpe=0.4, rl_sharpe=0.42,  # Margin = 0.02 < 0.1
            outperformance_days=60,
            total_days=100,
            stage=PromotionStage.SHADOW,
        )
        assert check_promotion_criteria(summary) is False

    def test_wrong_stage(self):
        summary = ShadowSummary(
            component="exit_manager",
            total_predictions=1000,
            baseline_cumulative=500.0, rl_cumulative=800.0,
            baseline_sharpe=0.3, rl_sharpe=0.6,
            outperformance_days=80,
            total_days=100,
            stage=PromotionStage.TRAINING,  # Not in shadow
        )
        assert check_promotion_criteria(summary) is False
