"""Strategy tournament: weekly cohort comparison and ranking."""

from datetime import date, datetime, timedelta, timezone

from aegis.lab.types import CohortPerformance, CohortStatus, StrategyCohort, TournamentResult

# Composite score weights
_W_SHARPE = 0.40
_W_DRAWDOWN = 0.25
_W_PROFIT_FACTOR = 0.20
_W_WIN_RATE = 0.15

# Promotion criteria
PROMOTION_MIN_SHARPE = 1.0
PROMOTION_MIN_WIN_RATE = 0.40
PROMOTION_MAX_DRAWDOWN = -0.20  # Negative fraction
PROMOTION_MIN_DAYS = 60

# Relegation criteria
RELEGATION_MAX_SHARPE = 0.0
RELEGATION_MIN_DRAWDOWN = -0.30  # Worse than -30%


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]. Returns 0.5 for all if identical."""
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [0.5] * len(values)
    return [(v - min_v) / (max_v - min_v) for v in values]


def run_weekly(
    performances: dict[str, CohortPerformance],
    week_start: date,
) -> list[TournamentResult]:
    """Score and rank all cohorts for a tournament week.

    Returns sorted TournamentResult list (rank 1 = best).
    """
    if not performances:
        return []

    cohort_ids = list(performances.keys())
    sharpes = [performances[c].sharpe for c in cohort_ids]
    drawdowns = [performances[c].max_drawdown for c in cohort_ids]
    profit_factors = [performances[c].profit_factor for c in cohort_ids]
    win_rates = [performances[c].win_rate for c in cohort_ids]

    # Cap profit_factor for normalization (inf -> 10.0)
    profit_factors = [min(pf, 10.0) for pf in profit_factors]

    norm_sharpe = _normalize(sharpes)
    # For drawdown, less negative is better, so negate for normalization
    norm_dd = _normalize([-d for d in drawdowns])  # Higher = less drawdown = better
    norm_pf = _normalize(profit_factors)
    norm_wr = _normalize(win_rates)

    scored: list[tuple[str, float]] = []
    for i, cid in enumerate(cohort_ids):
        composite = (
            _W_SHARPE * norm_sharpe[i]
            + _W_DRAWDOWN * norm_dd[i]
            + _W_PROFIT_FACTOR * norm_pf[i]
            + _W_WIN_RATE * norm_wr[i]
        )
        scored.append((cid, composite))

    # Sort by composite score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for rank, (cid, composite) in enumerate(scored, start=1):
        perf = performances[cid]
        results.append(TournamentResult(
            cohort_id=cid,
            week_start=week_start,
            sharpe=perf.sharpe,
            win_rate=perf.win_rate,
            max_drawdown=perf.max_drawdown,
            profit_factor=min(perf.profit_factor, 10.0),
            total_trades=perf.total_trades,
            net_pnl=perf.net_pnl,
            composite_score=composite,
            rank=rank,
        ))

    return results


def identify_promotion_candidates(
    performances: dict[str, CohortPerformance],
    cohorts: dict[str, StrategyCohort],
    current_time: datetime | None = None,
) -> list[str]:
    """Cohorts meeting ALL promotion criteria after 60+ days."""
    now = current_time or datetime.now(timezone.utc)
    candidates = []
    for cid, perf in performances.items():
        cohort = cohorts.get(cid)
        if cohort is None or cohort.status != CohortStatus.EVALUATING:
            continue
        # Age check
        if cohort.created_at is None:
            continue
        age_days = (now - cohort.created_at).days
        if age_days < PROMOTION_MIN_DAYS:
            continue
        # Performance criteria
        if (
            perf.sharpe >= PROMOTION_MIN_SHARPE
            and perf.win_rate >= PROMOTION_MIN_WIN_RATE
            and perf.max_drawdown >= PROMOTION_MAX_DRAWDOWN  # Less negative = better
        ):
            candidates.append(cid)
    return candidates


def identify_relegation_candidates(
    performances: dict[str, CohortPerformance],
    cohorts: dict[str, StrategyCohort],
) -> list[str]:
    """Cohorts with Sharpe < 0 OR max_drawdown > 30%."""
    candidates = []
    for cid, perf in performances.items():
        cohort = cohorts.get(cid)
        if cohort is None or cohort.status not in (
            CohortStatus.EVALUATING, CohortStatus.PROMOTED
        ):
            continue
        if perf.sharpe < RELEGATION_MAX_SHARPE or perf.max_drawdown < RELEGATION_MIN_DRAWDOWN:
            candidates.append(cid)
    return candidates
