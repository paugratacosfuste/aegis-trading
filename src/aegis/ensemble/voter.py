"""Ensemble voter: two-level aggregation from 03-ENSEMBLE-VOTING.md.

Stage 1: Intra-type aggregation (confidence-weighted + agreement bonus)
Stage 2: Cross-type weighted voting (with regime adjustments)

Backward compatible: old signature with just signals + threshold still works.
"""

from collections import defaultdict
from datetime import datetime

from aegis.common.types import AgentSignal, TradeDecision
from aegis.ensemble.aggregator import aggregate_intra_type
from aegis.ensemble.weights import BASE_TYPE_WEIGHTS, apply_regime_weights

DIRECTION_THRESHOLD = 0.1  # Minimum |direction| to generate a trade


def vote(
    signals: list[AgentSignal],
    confidence_threshold: float = 0.45,
    agent_weights: dict[str, float] | None = None,
    regime: str = "normal",
    current_time: datetime | None = None,
) -> TradeDecision:
    """Two-level ensemble vote.

    Stage 1: Group signals by agent_type, aggregate within each type.
    Stage 2: Weighted cross-type vote using base weights + regime adjustments.

    Backward compatible: called with just (signals, threshold) behaves
    like Phase 1 equal-weight voting.
    """
    if not signals:
        return _no_trade("No signals", "UNKNOWN")

    symbol = signals[0].symbol

    # Stage 1: Intra-type aggregation
    by_type: dict[str, list[AgentSignal]] = defaultdict(list)
    for s in signals:
        by_type[s.agent_type].append(s)

    type_signals: dict[str, AgentSignal] = {}
    for agent_type, type_sigs in by_type.items():
        agg = aggregate_intra_type(type_sigs, agent_weights, current_time)
        if agg is not None and agg.confidence > 0.001:
            type_signals[agent_type] = agg

    if not type_signals:
        return _no_trade("All types neutral", symbol)

    # Stage 2: Cross-type weighted vote
    # Use provided base weights or defaults, filtered to present types
    present_weights = {
        t: BASE_TYPE_WEIGHTS.get(t, 0.2)
        for t in type_signals
    }
    adjusted_weights = apply_regime_weights(present_weights, regime)

    # Weighted direction and confidence
    total_weight = 0.0
    weighted_direction = 0.0
    weighted_confidence = 0.0

    for agent_type, sig in type_signals.items():
        w = adjusted_weights.get(agent_type, 0.0)
        weighted_direction += sig.direction * w * sig.confidence
        weighted_confidence += sig.confidence * w
        total_weight += w * sig.confidence

    if total_weight == 0:
        return _no_trade("Zero total weight", symbol)

    final_direction = weighted_direction / total_weight
    final_confidence = weighted_confidence

    # Conflict resolution
    action, final_direction, final_confidence, reason = _resolve_conflicts(
        type_signals, final_direction, final_confidence
    )

    if action == "NO_TRADE":
        return _no_trade(reason, symbol)

    if final_confidence < confidence_threshold:
        return _no_trade(
            f"Confidence {final_confidence:.2f} below threshold {confidence_threshold}",
            symbol,
        )

    if abs(final_direction) < DIRECTION_THRESHOLD:
        return _no_trade(
            f"Direction {final_direction:.3f} too weak",
            symbol,
        )

    action = "LONG" if final_direction > 0 else "SHORT"

    # Collect all contributing signals (both raw and aggregated)
    contributing = {}
    for s in signals:
        contributing[s.agent_id] = s

    return TradeDecision(
        action=action,
        symbol=symbol,
        direction=final_direction,
        confidence=final_confidence,
        quantity=0.0,  # Set by risk manager
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        contributing_signals=contributing,
        reason=f"Ensemble vote: {action} dir={final_direction:.3f} conf={final_confidence:.3f}",
    )


def _resolve_conflicts(
    type_signals: dict[str, AgentSignal],
    direction: float,
    confidence: float,
) -> tuple[str, float, float, str]:
    """Apply conflict resolution rules from 03-ENSEMBLE-VOTING.md.

    Returns (action, direction, confidence, reason).
    """
    # Check for strong disagreement between technical and statistical
    tech = type_signals.get("technical")
    stat = type_signals.get("statistical")

    if tech and stat:
        if (tech.direction > 0.3 and stat.direction < -0.3) or \
           (tech.direction < -0.3 and stat.direction > 0.3):
            if abs(tech.confidence - stat.confidence) < 0.2:
                # Similar confidence, opposite direction -> no trade
                return "NO_TRADE", 0.0, 0.0, "Technical vs statistical conflict"

    return "TRADE", direction, confidence, ""


def _no_trade(reason: str, symbol: str) -> TradeDecision:
    return TradeDecision(
        action="NO_TRADE",
        symbol=symbol,
        direction=0.0,
        confidence=0.0,
        quantity=0.0,
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        contributing_signals={},
        reason=reason,
    )
