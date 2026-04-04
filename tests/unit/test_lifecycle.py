"""Tests for cohort lifecycle state machine."""

from datetime import datetime, timedelta, timezone

import pytest

from aegis.common.exceptions import AegisError
from aegis.lab.lifecycle import (
    advance_status,
    apply_demotion,
    apply_promotion,
    apply_relegation,
    can_be_evaluated,
)
from aegis.lab.types import CohortConfig, CohortStatus, StrategyCohort


def _cohort(status=CohortStatus.CREATED, burn_in_start=None, created_at=None, evaluation_start=None):
    return StrategyCohort(
        cohort_id="test_A", name="Test",
        status=status,
        config=CohortConfig(
            agent_weights={"technical": 1.0}, confidence_threshold=0.45,
            risk_params={}, universe=(),
        ),
        created_at=created_at or datetime(2025, 3, 1, tzinfo=timezone.utc),
        burn_in_start=burn_in_start,
        evaluation_start=evaluation_start,
    )


class TestAdvanceStatus:
    def test_created_to_burn_in(self):
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        c = _cohort(CohortStatus.CREATED)
        result = advance_status(c, now)
        assert result.status == CohortStatus.BURN_IN
        assert result.burn_in_start == now

    def test_burn_in_to_evaluating(self):
        start = datetime(2025, 5, 1, tzinfo=timezone.utc)
        now = datetime(2025, 5, 20, tzinfo=timezone.utc)  # 19 days, > 14
        c = _cohort(CohortStatus.BURN_IN, burn_in_start=start)
        result = advance_status(c, now)
        assert result.status == CohortStatus.EVALUATING
        assert result.evaluation_start == now

    def test_burn_in_stays_if_too_early(self):
        start = datetime(2025, 5, 1, tzinfo=timezone.utc)
        now = datetime(2025, 5, 10, tzinfo=timezone.utc)  # Only 9 days
        c = _cohort(CohortStatus.BURN_IN, burn_in_start=start)
        result = advance_status(c, now)
        assert result.status == CohortStatus.BURN_IN

    def test_evaluating_stays(self):
        c = _cohort(CohortStatus.EVALUATING)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        result = advance_status(c, now)
        assert result.status == CohortStatus.EVALUATING

    def test_no_burn_in_start_stays(self):
        c = _cohort(CohortStatus.BURN_IN, burn_in_start=None)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        result = advance_status(c, now)
        assert result.status == CohortStatus.BURN_IN


class TestPromotion:
    def test_promote_evaluating(self):
        c = _cohort(CohortStatus.EVALUATING)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        result = apply_promotion(c, now)
        assert result.status == CohortStatus.PROMOTED

    def test_promote_invalid_from_created(self):
        c = _cohort(CohortStatus.CREATED)
        with pytest.raises(AegisError):
            apply_promotion(c, datetime.now(timezone.utc))


class TestRelegation:
    def test_relegate_evaluating(self):
        c = _cohort(CohortStatus.EVALUATING)
        result = apply_relegation(c, datetime.now(timezone.utc))
        assert result.status == CohortStatus.RELEGATED
        assert result.relegation_count == 1

    def test_second_relegation_retires(self):
        c = StrategyCohort(
            cohort_id="test_A", name="Test",
            status=CohortStatus.EVALUATING,
            config=CohortConfig(
                agent_weights={"technical": 1.0}, confidence_threshold=0.45,
                risk_params={}, universe=(),
            ),
            relegation_count=1,
        )
        result = apply_relegation(c, datetime.now(timezone.utc))
        assert result.status == CohortStatus.RETIRED
        assert result.relegation_count == 2

    def test_relegate_invalid_from_created(self):
        c = _cohort(CohortStatus.CREATED)
        with pytest.raises(AegisError):
            apply_relegation(c, datetime.now(timezone.utc))

    def test_relegate_promoted(self):
        c = _cohort(CohortStatus.PROMOTED)
        result = apply_relegation(c, datetime.now(timezone.utc))
        assert result.status == CohortStatus.RELEGATED


class TestDemotion:
    def test_demote_promoted(self):
        c = _cohort(CohortStatus.PROMOTED)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        result = apply_demotion(c, now)
        assert result.status == CohortStatus.EVALUATING
        assert result.evaluation_start == now

    def test_demote_invalid_from_evaluating(self):
        c = _cohort(CohortStatus.EVALUATING)
        with pytest.raises(AegisError):
            apply_demotion(c, datetime.now(timezone.utc))


class TestCanBeEvaluated:
    def test_old_enough(self):
        c = _cohort(CohortStatus.EVALUATING,
                     evaluation_start=datetime(2025, 3, 1, tzinfo=timezone.utc))
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        assert can_be_evaluated(c, now) is True

    def test_too_young(self):
        c = _cohort(CohortStatus.EVALUATING,
                     evaluation_start=datetime(2025, 5, 15, tzinfo=timezone.utc))
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        assert can_be_evaluated(c, now) is False

    def test_wrong_status(self):
        c = _cohort(CohortStatus.BURN_IN)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        assert can_be_evaluated(c, now) is False

    def test_no_evaluation_start(self):
        c = _cohort(CohortStatus.EVALUATING, evaluation_start=None)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        assert can_be_evaluated(c, now) is False
