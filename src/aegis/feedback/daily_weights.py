"""Daily Bayesian weight updates: Speed 1 of the feedback loop.

Updates every agent's weight based on recent signal accuracy using an
exponential moving average with IC/hit_rate composite scoring.
"""

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import numpy as np

from aegis.feedback.types import WeightUpdateLog

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHT = 0.15
_DEFAULT_LEARNING_RATE = 0.10
_DEFAULT_MIN_SIGNALS = 5
_DEFAULT_WEIGHT_MIN = 0.05
_DEFAULT_WEIGHT_MAX = 0.30


def compute_ic(predictions: list[float], actuals: list[float]) -> float:
    """Information coefficient: correlation between predicted directions and actual returns.

    Returns 0.0 if fewer than 3 samples or zero variance.
    """
    if len(predictions) < 3 or len(actuals) < 3:
        return 0.0
    preds = np.array(predictions)
    acts = np.array(actuals)
    if np.std(preds) == 0 or np.std(acts) == 0:
        return 0.0
    return float(np.corrcoef(preds, acts)[0, 1])


def compute_composite_score(hit_rate: float, ic: float) -> float:
    """Composite score: 0.5 * hit_rate + 0.5 * max(ic, 0)."""
    return 0.5 * hit_rate + 0.5 * max(ic, 0.0)


def ema_update(
    old_weight: float,
    score: float,
    learning_rate: float = _DEFAULT_LEARNING_RATE,
    weight_min: float = _DEFAULT_WEIGHT_MIN,
    weight_max: float = _DEFAULT_WEIGHT_MAX,
) -> float:
    """Exponential moving average weight update with clipping."""
    new_weight = old_weight * (1 - learning_rate) + score * learning_rate
    return float(np.clip(new_weight, weight_min, weight_max))


def normalize_within_type(
    weights: dict[str, dict],
) -> dict[str, dict]:
    """Renormalize weights so each agent type sums to 1.0.

    Args:
        weights: {agent_id: {"weight": float, "type": str, ...}}

    Returns:
        Same dict with updated weight values.
    """
    by_type: dict[str, list[str]] = defaultdict(list)
    for agent_id, info in weights.items():
        by_type[info["type"]].append(agent_id)

    result = {k: dict(v) for k, v in weights.items()}
    for agent_type, agent_ids in by_type.items():
        total = sum(result[aid]["weight"] for aid in agent_ids)
        if total > 0:
            for aid in agent_ids:
                result[aid]["weight"] = result[aid]["weight"] / total
    return result


def run_daily_update(
    signal_outcomes: list[dict],
    current_weights: dict[str, dict],
    learning_rate: float = _DEFAULT_LEARNING_RATE,
    min_signals: int = _DEFAULT_MIN_SIGNALS,
    weight_min: float = _DEFAULT_WEIGHT_MIN,
    weight_max: float = _DEFAULT_WEIGHT_MAX,
    update_date: date | None = None,
) -> list[WeightUpdateLog]:
    """Run the daily weight update on pre-fetched data.

    Args:
        signal_outcomes: List of dicts with agent_id, agent_type,
            predicted_direction, actual_return, is_correct.
        current_weights: {agent_id: {"agent_type": str, "weight": float}}.
        learning_rate: EMA learning rate.
        min_signals: Minimum matched signals to update an agent.
        weight_min: Minimum allowed weight.
        weight_max: Maximum allowed weight.
        update_date: Date for the log entry (defaults to today).

    Returns:
        List of WeightUpdateLog entries for all updated agents.
    """
    if not signal_outcomes:
        return []

    if update_date is None:
        update_date = date.today()

    # Group signals by agent
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for so in signal_outcomes:
        by_agent[so["agent_id"]].append(so)

    # Compute new weights
    updated: dict[str, dict] = {}
    for agent_id, outcomes in by_agent.items():
        if len(outcomes) < min_signals:
            continue

        agent_type = outcomes[0]["agent_type"]

        # Compute metrics
        predictions = [o["predicted_direction"] for o in outcomes]
        actuals = [o["actual_return"] for o in outcomes]
        n_correct = sum(1 for o in outcomes if o["is_correct"])
        hit_rate = n_correct / len(outcomes)
        ic = compute_ic(predictions, actuals)
        composite = compute_composite_score(hit_rate, ic)

        # Get old weight
        if agent_id in current_weights:
            old_weight = current_weights[agent_id].get("weight", _DEFAULT_WEIGHT)
        else:
            old_weight = _DEFAULT_WEIGHT

        # EMA update
        new_weight = ema_update(old_weight, composite, learning_rate, weight_min, weight_max)

        updated[agent_id] = {
            "type": agent_type,
            "weight": new_weight,
            "old_weight": old_weight,
            "hit_rate": hit_rate,
            "ic": ic,
            "composite": composite,
            "n_signals": len(outcomes),
        }

    if not updated:
        return []

    # Normalize within each type
    updated = normalize_within_type(updated)

    # Build log entries
    logs = []
    for agent_id, info in updated.items():
        logs.append(WeightUpdateLog(
            agent_id=agent_id,
            agent_type=info["type"],
            old_weight=info["old_weight"],
            new_weight=info["weight"],
            hit_rate=info["hit_rate"],
            ic=info["ic"],
            composite_score=info["composite"],
            n_signals=info["n_signals"],
            update_date=update_date,
        ))

    return logs
