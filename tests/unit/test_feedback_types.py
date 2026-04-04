"""Tests for feedback loop types."""

from datetime import date, datetime, timezone
from types import MappingProxyType

from aegis.feedback.types import (
    AgentWeightSnapshot,
    FeatureImportance,
    ModelVersion,
    MonthlyRetrospective,
    RegimePlaybookEntry,
    WeightUpdateLog,
)


class TestWeightUpdateLog:
    def test_frozen(self):
        log = WeightUpdateLog(
            agent_id="tech_01", agent_type="technical",
            old_weight=0.20, new_weight=0.22,
            hit_rate=0.55, ic=0.12, composite_score=0.335,
            n_signals=15, update_date=date(2026, 4, 1),
        )
        assert log.agent_id == "tech_01"
        assert log.new_weight == 0.22

    def test_immutable(self):
        log = WeightUpdateLog(
            agent_id="tech_01", agent_type="technical",
            old_weight=0.20, new_weight=0.22,
            hit_rate=0.55, ic=0.12, composite_score=0.335,
            n_signals=15, update_date=date(2026, 4, 1),
        )
        try:
            log.new_weight = 0.99
            assert False, "Should raise"
        except AttributeError:
            pass


class TestAgentWeightSnapshot:
    def test_defaults(self):
        snap = AgentWeightSnapshot(
            agent_id="stat_01", agent_type="statistical",
            weight=0.15, hit_rate=0.50, ic=0.05,
        )
        assert snap.rolling_sharpe is None
        assert snap.updated_at is None


class TestModelVersion:
    def test_fields(self):
        mv = ModelVersion(
            model_id="mv_001", model_type="momentum_predictor",
            version="2026-04-01",
            train_start=date(2023, 4, 1), train_end=date(2026, 3, 1),
            val_start=date(2026, 4, 1), val_end=date(2026, 7, 1),
            train_samples=5000, val_samples=500,
            val_auc=0.62, previous_auc=0.60,
            accepted=True,
            feature_names=("rsi_14", "ema_20", "volume_ratio"),
            model_path="models/feedback/momentum_2026-04-01.json",
        )
        assert mv.accepted is True
        assert len(mv.feature_names) == 3
        assert mv.created_at is None


class TestFeatureImportance:
    def test_default_status(self):
        fi = FeatureImportance(
            model_id="mv_001", retrain_date=date(2026, 4, 1),
            feature_name="rsi_14", shap_importance=0.15, rank=1,
        )
        assert fi.status == "stable"

    def test_custom_status(self):
        fi = FeatureImportance(
            model_id="mv_001", retrain_date=date(2026, 4, 1),
            feature_name="new_feat", shap_importance=0.08, rank=5,
            status="new",
        )
        assert fi.status == "new"


class TestRegimePlaybookEntry:
    def test_weights_become_mapping_proxy(self):
        entry = RegimePlaybookEntry(
            regime="bull",
            last_updated=date(2026, 4, 1),
            total_observations=100,
            best_agent_weights={"technical": 0.3, "momentum": 0.7},
            best_cohort_ids=("cohort_A",),
            avg_sharpe=1.5, avg_win_rate=0.55,
            worst_agent_types=("sentiment",),
        )
        assert isinstance(entry.best_agent_weights, MappingProxyType)
        assert entry.recommended_position_size_mult == 1.0

    def test_frozen(self):
        entry = RegimePlaybookEntry(
            regime="crisis",
            last_updated=date(2026, 4, 1),
            total_observations=50,
            best_agent_weights={"geopolitical": 1.0},
            best_cohort_ids=(),
            avg_sharpe=-0.5, avg_win_rate=0.30,
            worst_agent_types=("momentum", "technical"),
        )
        try:
            entry.regime = "bull"
            assert False, "Should raise"
        except AttributeError:
            pass


class TestMonthlyRetrospective:
    def test_defaults(self):
        retro = MonthlyRetrospective(
            month="2026-04",
            production_return=0.05,
            production_sharpe=1.2,
            benchmark_return=0.03,
            alpha=0.02,
            best_lab_cohort="cohort_B",
            worst_lab_cohort="cohort_E",
            cohorts_promoted=("cohort_B",),
            cohorts_relegated=("cohort_E",),
            most_improved_agent="tech_03",
            most_degraded_agent="stat_07",
            agents_evolved=("mom_09", "stat_10"),
        )
        assert retro.alpha == 0.02
        assert len(retro.recommendations) == 0
        assert isinstance(retro.regimes_encountered, MappingProxyType)

    def test_with_recommendations(self):
        retro = MonthlyRetrospective(
            month="2026-04",
            production_return=0.05,
            production_sharpe=1.2,
            benchmark_return=0.03,
            alpha=0.02,
            best_lab_cohort=None,
            worst_lab_cohort=None,
            cohorts_promoted=(),
            cohorts_relegated=(),
            most_improved_agent=None,
            most_degraded_agent=None,
            agents_evolved=(),
            recommendations=("Review agent stat_03", "Feature rsi_14 stable"),
        )
        assert len(retro.recommendations) == 2
