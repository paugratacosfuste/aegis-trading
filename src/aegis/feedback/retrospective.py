"""Monthly retrospective report: aggregates all feedback subsystem outputs.

Produces a MonthlyRetrospective covering production performance, lab summary,
agent insights, feature importance shifts, regime analysis, and recommendations.
"""

import logging
from types import MappingProxyType

from aegis.backtest.metrics import calculate_sharpe, calculate_win_rate
from aegis.feedback.types import MonthlyRetrospective

logger = logging.getLogger(__name__)


def compute_alpha(production_return: float, benchmark_return: float) -> float:
    """Alpha = production return - benchmark return."""
    return production_return - benchmark_return


def find_most_changed_agents(
    weight_deltas: dict[str, float],
) -> tuple[str | None, str | None]:
    """Find the most improved and most degraded agents by weight delta.

    Returns (most_improved_id, most_degraded_id).
    """
    if not weight_deltas:
        return None, None

    most_improved = max(weight_deltas, key=weight_deltas.get)
    most_degraded = min(weight_deltas, key=weight_deltas.get)
    return most_improved, most_degraded


def generate_recommendations(
    production_sharpe: float,
    alpha: float,
    most_degraded: str | None,
    evolved_count: int,
) -> list[str]:
    """Generate rule-based recommendations from performance data."""
    recs = []

    if production_sharpe < 0:
        recs.append(
            f"Production Sharpe is negative ({production_sharpe:.2f}). "
            "Review risk parameters and confidence thresholds."
        )

    if alpha < -0.02:
        recs.append(
            f"Underperforming benchmark by {abs(alpha)*100:.1f}%. "
            "Consider increasing confidence threshold or reducing position sizes."
        )

    if most_degraded:
        recs.append(
            f"Agent {most_degraded} showed largest weight degradation. "
            "Review its strategy and recent signal quality."
        )

    if evolved_count > 0:
        recs.append(
            f"{evolved_count} agent(s) were evolved this month. "
            "Monitor their burn-in performance."
        )

    if production_sharpe > 1.5 and alpha > 0:
        recs.append("Strong performance this month. Maintain current configuration.")

    return recs


def build_retrospective(
    month: str,
    trades_pnls: list[float],
    benchmark_returns: list[float],
    weight_deltas: dict[str, float],
    evolved_agent_ids: list[str],
    regimes_encountered: dict[str, int] | None = None,
    regime_performance: dict[str, float] | None = None,
    best_lab_cohort: str | None = None,
    worst_lab_cohort: str | None = None,
    cohorts_promoted: list[str] | None = None,
    cohorts_relegated: list[str] | None = None,
    feature_importance_shifts: dict[str, float] | None = None,
    rl_shadow_performance: dict[str, float] | None = None,
) -> MonthlyRetrospective:
    """Build a MonthlyRetrospective from aggregated data.

    This is a pure function that takes pre-fetched data and produces the report.
    The caller is responsible for querying the database.
    """
    # Production metrics
    production_return = sum(trades_pnls) if trades_pnls else 0.0
    production_sharpe = calculate_sharpe(trades_pnls) if len(trades_pnls) >= 2 else 0.0

    # Benchmark
    benchmark_return = sum(benchmark_returns) if benchmark_returns else 0.0
    alpha = compute_alpha(production_return, benchmark_return)

    # Agent insights
    most_improved, most_degraded = find_most_changed_agents(weight_deltas)

    # Recommendations
    recs = generate_recommendations(
        production_sharpe=production_sharpe,
        alpha=alpha,
        most_degraded=most_degraded,
        evolved_count=len(evolved_agent_ids),
    )

    return MonthlyRetrospective(
        month=month,
        production_return=production_return,
        production_sharpe=production_sharpe,
        benchmark_return=benchmark_return,
        alpha=alpha,
        best_lab_cohort=best_lab_cohort,
        worst_lab_cohort=worst_lab_cohort,
        cohorts_promoted=tuple(cohorts_promoted or []),
        cohorts_relegated=tuple(cohorts_relegated or []),
        most_improved_agent=most_improved,
        most_degraded_agent=most_degraded,
        agents_evolved=tuple(evolved_agent_ids),
        feature_importance_shifts=MappingProxyType(feature_importance_shifts or {}),
        regimes_encountered=MappingProxyType(regimes_encountered or {}),
        regime_performance=MappingProxyType(regime_performance or {}),
        rl_shadow_performance=MappingProxyType(rl_shadow_performance or {}),
        recommendations=tuple(recs),
    )
