"""Ensemble voter: equal-weight average of agent signals.

Phase 1 implementation from 03-ENSEMBLE-VOTING.md.
"""

from aegis.common.types import AgentSignal, TradeDecision

DIRECTION_THRESHOLD = 0.1  # Minimum |direction| to generate a trade


def vote(
    signals: list[AgentSignal],
    confidence_threshold: float = 0.45,
) -> TradeDecision:
    """Equal-weight confidence-weighted vote across all signals.

    Returns a TradeDecision: LONG, SHORT, or NO_TRADE.
    """
    if not signals:
        return _no_trade("No signals", "UNKNOWN")

    symbol = signals[0].symbol

    # Confidence-weighted direction average
    total_weight = sum(s.confidence for s in signals)
    if total_weight == 0:
        return _no_trade("All zero confidence", symbol)

    weighted_direction = sum(s.direction * s.confidence for s in signals) / total_weight
    avg_confidence = sum(s.confidence for s in signals) / len(signals)

    if avg_confidence < confidence_threshold:
        return _no_trade(
            f"Confidence {avg_confidence:.2f} below threshold {confidence_threshold}",
            symbol,
        )

    if abs(weighted_direction) < DIRECTION_THRESHOLD:
        return _no_trade(
            f"Direction {weighted_direction:.3f} too weak",
            symbol,
        )

    action = "LONG" if weighted_direction > 0 else "SHORT"

    return TradeDecision(
        action=action,
        symbol=symbol,
        direction=weighted_direction,
        confidence=avg_confidence,
        quantity=0.0,  # Set by risk manager
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        contributing_signals={s.agent_id: s for s in signals},
        reason=f"Ensemble vote: {action} with direction={weighted_direction:.3f}",
    )


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
