"""Cohort lifecycle state machine: status transitions and time-based advancement."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from aegis.common.exceptions import AegisError
from aegis.lab.types import CohortStatus, StrategyCohort

DEFAULT_BURN_IN_DAYS = 14
DEFAULT_EVALUATION_DAYS = 60

# Valid transitions
_VALID_TRANSITIONS: dict[str, set[str]] = {
    CohortStatus.CREATED: {CohortStatus.BURN_IN},
    CohortStatus.BURN_IN: {CohortStatus.EVALUATING},
    CohortStatus.EVALUATING: {CohortStatus.PROMOTED, CohortStatus.RELEGATED},
    CohortStatus.PROMOTED: {CohortStatus.EVALUATING, CohortStatus.RELEGATED},
    CohortStatus.RELEGATED: {CohortStatus.RETIRED, CohortStatus.BURN_IN},
    CohortStatus.RETIRED: set(),
}


def _validate_transition(current: str, target: str) -> None:
    valid = _VALID_TRANSITIONS.get(current, set())
    if target not in valid:
        raise AegisError(f"Invalid cohort transition: {current} -> {target}")


def advance_status(
    cohort: StrategyCohort,
    current_time: datetime,
    burn_in_days: int = DEFAULT_BURN_IN_DAYS,
) -> StrategyCohort:
    """Time-based status advancement. Returns new cohort if advanced, else same."""
    if cohort.status == CohortStatus.CREATED:
        _validate_transition(CohortStatus.CREATED, CohortStatus.BURN_IN)
        return replace(
            cohort,
            status=CohortStatus.BURN_IN,
            burn_in_start=current_time,
        )

    if cohort.status == CohortStatus.BURN_IN:
        if cohort.burn_in_start is None:
            return cohort
        elapsed = (current_time - cohort.burn_in_start).days
        if elapsed >= burn_in_days:
            _validate_transition(CohortStatus.BURN_IN, CohortStatus.EVALUATING)
            return replace(
                cohort,
                status=CohortStatus.EVALUATING,
                evaluation_start=current_time,
            )

    return cohort


def apply_promotion(
    cohort: StrategyCohort,
    current_time: datetime,
) -> StrategyCohort:
    """Promote a cohort. Raises on invalid transition."""
    _validate_transition(cohort.status, CohortStatus.PROMOTED)
    return replace(cohort, status=CohortStatus.PROMOTED)


def apply_relegation(
    cohort: StrategyCohort,
    current_time: datetime,
) -> StrategyCohort:
    """Relegate a cohort. If relegation_count >= 1, retire instead."""
    _validate_transition(cohort.status, CohortStatus.RELEGATED)
    new_count = cohort.relegation_count + 1
    if new_count >= 2:
        _validate_transition(CohortStatus.RELEGATED, CohortStatus.RETIRED)
        return replace(
            cohort,
            status=CohortStatus.RETIRED,
            relegation_count=new_count,
        )
    return replace(
        cohort,
        status=CohortStatus.RELEGATED,
        relegation_count=new_count,
    )


def apply_demotion(
    cohort: StrategyCohort,
    current_time: datetime,
) -> StrategyCohort:
    """Demote a promoted cohort back to evaluating."""
    _validate_transition(cohort.status, CohortStatus.EVALUATING)
    return replace(
        cohort,
        status=CohortStatus.EVALUATING,
        evaluation_start=current_time,
    )


def can_be_evaluated(
    cohort: StrategyCohort,
    current_time: datetime,
    evaluation_days: int = DEFAULT_EVALUATION_DAYS,
) -> bool:
    """True if the cohort is evaluating and has been for >= evaluation_days."""
    if cohort.status != CohortStatus.EVALUATING:
        return False
    if cohort.evaluation_start is None:
        return False
    age_days = (current_time - cohort.evaluation_start).days
    return age_days >= evaluation_days
