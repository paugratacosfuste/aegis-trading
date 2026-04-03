"""Intra-type aggregator: confidence-weighted voting within one agent type.

Produces one aggregated AgentSignal per type from N instance signals.
Includes agreement bonus (consensus boosts confidence).
"""

from datetime import datetime, timezone

import numpy as np

from aegis.common.types import AgentSignal
from aegis.ensemble.decay import apply_decay


def aggregate_intra_type(
    signals: list[AgentSignal],
    agent_weights: dict[str, float] | None = None,
    current_time: datetime | None = None,
) -> AgentSignal | None:
    """Aggregate N signals of the same type into one.

    Args:
        signals: Signals from agents of the same type.
        agent_weights: Per-agent weight overrides (agent_id -> weight).
            Defaults to 1.0 for all.
        current_time: If provided, applies signal decay before aggregation.

    Returns:
        Aggregated AgentSignal, or None if no valid signals.
    """
    if not signals:
        return None

    if agent_weights is None:
        agent_weights = {}

    # Apply decay if current_time provided
    if current_time is not None:
        signals = [apply_decay(s, current_time) for s in signals]

    # Filter out zero-confidence signals
    active = [s for s in signals if s.confidence > 0.001]
    if not active:
        return _neutral_from(signals[0])

    # Confidence-weighted direction average
    total_weight = 0.0
    weighted_direction = 0.0
    weighted_confidence = 0.0

    for signal in active:
        w = agent_weights.get(signal.agent_id, 1.0) * signal.confidence
        weighted_direction += signal.direction * w
        weighted_confidence += signal.confidence * agent_weights.get(signal.agent_id, 1.0)
        total_weight += w

    if total_weight == 0:
        return _neutral_from(signals[0])

    agg_direction = weighted_direction / total_weight
    agg_confidence = weighted_confidence / len(active)

    # Agreement bonus: low variance in directions = high agreement
    directions = [s.direction for s in active]
    agreement = 1.0 - float(np.std(directions)) if len(directions) > 1 else 1.0
    agg_confidence *= 0.7 + 0.3 * max(0.0, agreement)

    agg_direction = max(-1.0, min(1.0, agg_direction))
    agg_confidence = max(0.0, min(1.0, agg_confidence))

    return AgentSignal(
        agent_id=f"{active[0].agent_type}_agg",
        agent_type=active[0].agent_type,
        symbol=active[0].symbol,
        timestamp=max(s.timestamp for s in active),
        direction=agg_direction,
        confidence=agg_confidence,
        timeframe=active[0].timeframe,
        expected_holding_period=active[0].expected_holding_period,
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        reasoning={"n_agents": len(active), "agreement": round(agreement, 3)},
        features_used={},
        metadata={"agent_ids": [s.agent_id for s in active]},
    )


def _neutral_from(signal: AgentSignal) -> AgentSignal:
    """Create a neutral aggregated signal from a reference signal."""
    return AgentSignal(
        agent_id=f"{signal.agent_type}_agg",
        agent_type=signal.agent_type,
        symbol=signal.symbol,
        timestamp=signal.timestamp,
        direction=0.0,
        confidence=0.0,
        timeframe=signal.timeframe,
        expected_holding_period=signal.expected_holding_period,
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        reasoning={"note": "no valid signals"},
        features_used={},
        metadata={},
    )
