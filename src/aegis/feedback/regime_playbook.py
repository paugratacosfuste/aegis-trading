"""Regime playbook: per-regime learned optimal settings.

Builds playbook entries from historical trades and agent performance,
recording which agent types and configurations work best in each regime.
"""

import logging
from collections import defaultdict
from datetime import date

from aegis.backtest.metrics import calculate_sharpe, calculate_win_rate
from aegis.feedback.types import RegimePlaybookEntry

logger = logging.getLogger(__name__)


def build_playbook_entries(
    trades: list[dict],
    agent_performances: list[dict],
    min_observations: int = 30,
    as_of_date: date | None = None,
) -> list[RegimePlaybookEntry]:
    """Build regime playbook entries from trade and agent performance data.

    Args:
        trades: List of trade dicts with regime_at_entry, net_pnl, return_pct.
        agent_performances: List of dicts with agent_id, agent_type,
            regime_at_entry, is_correct.
        min_observations: Minimum trades per regime to include.
        as_of_date: Date for the playbook entry (defaults to today).

    Returns:
        List of RegimePlaybookEntry, one per regime with enough data.
    """
    if not trades:
        return []

    if as_of_date is None:
        as_of_date = date.today()

    # Group trades by regime
    trades_by_regime: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        regime = t.get("regime_at_entry", "unknown")
        trades_by_regime[regime].append(t)

    # Group agent performances by regime
    perfs_by_regime: dict[str, list[dict]] = defaultdict(list)
    for p in agent_performances:
        regime = p.get("regime_at_entry", "unknown")
        perfs_by_regime[regime].append(p)

    entries = []
    for regime, regime_trades in trades_by_regime.items():
        if len(regime_trades) < min_observations:
            continue

        pnls = [t["net_pnl"] for t in regime_trades]
        returns = [t.get("return_pct", 0.0) for t in regime_trades]

        avg_sharpe = calculate_sharpe(returns) if len(returns) >= 2 else 0.0
        avg_win_rate = calculate_win_rate(pnls)

        # Analyze agent type performance for this regime
        best_weights, worst_types = _analyze_agent_performance(
            perfs_by_regime.get(regime, [])
        )

        # Recommended position size: scale based on win rate and regime performance
        position_mult = _compute_position_mult(avg_win_rate, avg_sharpe)

        entries.append(RegimePlaybookEntry(
            regime=regime,
            last_updated=as_of_date,
            total_observations=len(regime_trades),
            best_agent_weights=best_weights,
            best_cohort_ids=(),  # Populated when cohort data available
            avg_sharpe=avg_sharpe,
            avg_win_rate=avg_win_rate,
            worst_agent_types=tuple(worst_types),
            recommended_position_size_mult=position_mult,
            recommended_max_positions=10,
            recommended_confidence_threshold=_compute_confidence_threshold(avg_win_rate),
        ))

    return entries


def _analyze_agent_performance(
    performances: list[dict],
) -> tuple[dict[str, float], list[str]]:
    """Compute per-type hit rates and identify best/worst types.

    Returns (best_weights, worst_types).
    """
    if not performances:
        return {}, []

    # Group by agent_type
    by_type: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for p in performances:
        agent_type = p["agent_type"]
        by_type[agent_type]["total"] += 1
        if p.get("is_correct", False):
            by_type[agent_type]["correct"] += 1

    # Compute hit rates
    type_hit_rates: dict[str, float] = {}
    for agent_type, counts in by_type.items():
        if counts["total"] > 0:
            type_hit_rates[agent_type] = counts["correct"] / counts["total"]

    if not type_hit_rates:
        return {}, []

    # Best weights: proportional to hit rate
    total_hr = sum(type_hit_rates.values())
    best_weights = {}
    if total_hr > 0:
        best_weights = {k: v / total_hr for k, v in type_hit_rates.items()}

    # Worst types: below 0.4 hit rate
    worst_types = [t for t, hr in type_hit_rates.items() if hr < 0.4]

    return best_weights, worst_types


def _compute_position_mult(win_rate: float, sharpe: float) -> float:
    """Compute recommended position size multiplier.

    Lower for poor-performing regimes, higher for strong ones.
    """
    if win_rate < 0.35 or sharpe < -0.5:
        return 0.5
    if win_rate < 0.45 or sharpe < 0.0:
        return 0.75
    if win_rate > 0.55 and sharpe > 1.0:
        return 1.25
    return 1.0


def _compute_confidence_threshold(win_rate: float) -> float:
    """Higher threshold for low-win-rate regimes."""
    if win_rate < 0.40:
        return 0.60
    if win_rate < 0.50:
        return 0.50
    return 0.45
