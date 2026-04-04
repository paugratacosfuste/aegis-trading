"""Tests for strategy tournament."""

from datetime import date, datetime, timezone

import pytest

from aegis.lab.tournament import (
    _normalize,
    identify_promotion_candidates,
    identify_relegation_candidates,
    run_weekly,
)
from aegis.lab.types import CohortConfig, CohortPerformance, CohortStatus, StrategyCohort


def _perf(cid, sharpe=1.0, win_rate=0.5, max_dd=-0.10, pf=1.5, trades=20, pnl=1000.0):
    return CohortPerformance(
        cohort_id=cid, sharpe=sharpe, win_rate=win_rate,
        max_drawdown=max_dd, profit_factor=pf,
        total_trades=trades, net_pnl=pnl,
    )


def _cohort(cid, status=CohortStatus.EVALUATING, evaluation_start=None):
    return StrategyCohort(
        cohort_id=cid, name=f"Test {cid}", status=status,
        config=CohortConfig(
            agent_weights={"technical": 1.0}, confidence_threshold=0.45,
            risk_params={}, universe=(),
        ),
        created_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
        evaluation_start=evaluation_start or datetime(2025, 3, 1, tzinfo=timezone.utc),
    )


class TestNormalize:
    def test_basic(self):
        result = _normalize([1.0, 2.0, 3.0])
        assert result == [0.0, 0.5, 1.0]

    def test_identical_values(self):
        result = _normalize([5.0, 5.0, 5.0])
        assert result == [0.5, 0.5, 0.5]

    def test_empty(self):
        assert _normalize([]) == []

    def test_two_values(self):
        result = _normalize([10.0, 20.0])
        assert result == [0.0, 1.0]

    def test_negative_values(self):
        result = _normalize([-2.0, 0.0, 2.0])
        assert result == [0.0, 0.5, 1.0]


class TestRunWeekly:
    def test_ranking_order(self):
        perfs = {
            "A": _perf("A", sharpe=2.0, win_rate=0.6, max_dd=-0.05, pf=3.0),
            "B": _perf("B", sharpe=0.5, win_rate=0.4, max_dd=-0.20, pf=1.0),
            "C": _perf("C", sharpe=1.0, win_rate=0.5, max_dd=-0.10, pf=1.5),
        }
        results = run_weekly(perfs, date(2025, 6, 1))
        assert len(results) == 3
        assert results[0].cohort_id == "A"  # Best
        assert results[0].rank == 1
        assert results[2].cohort_id == "B"  # Worst
        assert results[2].rank == 3

    def test_composite_score_range(self):
        perfs = {
            "A": _perf("A", sharpe=2.0, win_rate=0.6, max_dd=-0.05, pf=3.0),
            "B": _perf("B", sharpe=-0.5, win_rate=0.3, max_dd=-0.30, pf=0.5),
        }
        results = run_weekly(perfs, date(2025, 6, 1))
        for r in results:
            assert 0.0 <= r.composite_score <= 1.0

    def test_single_cohort(self):
        perfs = {"A": _perf("A")}
        results = run_weekly(perfs, date(2025, 6, 1))
        assert len(results) == 1
        assert results[0].rank == 1
        # All normalized to 0.5 (single value)
        assert results[0].composite_score == pytest.approx(0.5)

    def test_empty(self):
        assert run_weekly({}, date(2025, 6, 1)) == []

    def test_all_tied(self):
        perfs = {
            "A": _perf("A", sharpe=1.0, win_rate=0.5, max_dd=-0.10, pf=1.5),
            "B": _perf("B", sharpe=1.0, win_rate=0.5, max_dd=-0.10, pf=1.5),
        }
        results = run_weekly(perfs, date(2025, 6, 1))
        assert results[0].composite_score == results[1].composite_score

    def test_inf_profit_factor_capped(self):
        perfs = {"A": _perf("A", pf=float("inf"))}
        results = run_weekly(perfs, date(2025, 6, 1))
        assert results[0].profit_factor == 10.0


class TestPromotionCandidates:
    def test_meets_all_criteria(self):
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        perfs = {"A": _perf("A", sharpe=1.5, win_rate=0.55, max_dd=-0.10)}
        cohorts = {"A": _cohort("A")}
        candidates = identify_promotion_candidates(perfs, cohorts, now)
        assert candidates == ["A"]

    def test_sharpe_too_low(self):
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        perfs = {"A": _perf("A", sharpe=0.8, win_rate=0.55, max_dd=-0.10)}
        cohorts = {"A": _cohort("A")}
        assert identify_promotion_candidates(perfs, cohorts, now) == []

    def test_win_rate_too_low(self):
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        perfs = {"A": _perf("A", sharpe=1.5, win_rate=0.35, max_dd=-0.10)}
        cohorts = {"A": _cohort("A")}
        assert identify_promotion_candidates(perfs, cohorts, now) == []

    def test_drawdown_too_high(self):
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        perfs = {"A": _perf("A", sharpe=1.5, win_rate=0.55, max_dd=-0.25)}
        cohorts = {"A": _cohort("A")}
        assert identify_promotion_candidates(perfs, cohorts, now) == []

    def test_too_young(self):
        now = datetime(2025, 3, 20, tzinfo=timezone.utc)  # Only 19 days old
        perfs = {"A": _perf("A", sharpe=1.5, win_rate=0.55, max_dd=-0.10)}
        cohorts = {"A": _cohort("A")}
        assert identify_promotion_candidates(perfs, cohorts, now) == []

    def test_only_evaluating_cohorts(self):
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        perfs = {"A": _perf("A", sharpe=1.5, win_rate=0.55, max_dd=-0.10)}
        cohorts = {"A": _cohort("A", status=CohortStatus.PROMOTED)}
        assert identify_promotion_candidates(perfs, cohorts, now) == []


class TestRelegationCandidates:
    def test_negative_sharpe(self):
        perfs = {"A": _perf("A", sharpe=-0.5, max_dd=-0.10)}
        cohorts = {"A": _cohort("A")}
        assert identify_relegation_candidates(perfs, cohorts) == ["A"]

    def test_bad_drawdown(self):
        perfs = {"A": _perf("A", sharpe=0.5, max_dd=-0.35)}
        cohorts = {"A": _cohort("A")}
        assert identify_relegation_candidates(perfs, cohorts) == ["A"]

    def test_both_criteria(self):
        perfs = {"A": _perf("A", sharpe=-0.5, max_dd=-0.40)}
        cohorts = {"A": _cohort("A")}
        assert identify_relegation_candidates(perfs, cohorts) == ["A"]

    def test_no_relegation(self):
        perfs = {"A": _perf("A", sharpe=0.5, max_dd=-0.10)}
        cohorts = {"A": _cohort("A")}
        assert identify_relegation_candidates(perfs, cohorts) == []

    def test_only_active_cohorts(self):
        perfs = {"A": _perf("A", sharpe=-1.0)}
        cohorts = {"A": _cohort("A", status=CohortStatus.RETIRED)}
        assert identify_relegation_candidates(perfs, cohorts) == []
