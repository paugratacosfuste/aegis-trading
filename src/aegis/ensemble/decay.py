"""Signal decay: reduces confidence of stale signals based on half-life.

Applied in the ensemble layer before aggregation.
"""

from datetime import datetime, timezone

from aegis.common.types import AgentSignal

SIGNAL_HALF_LIVES: dict[str, float] = {
    "technical": 4.0,       # hours
    "statistical": 48.0,    # hours
    "momentum": 120.0,      # hours
    "sentiment": 6.0,       # hours
    "macro": 168.0,         # hours (1 week - regime shifts are slow)
    "geopolitical": 24.0,   # hours (events decay within a day)
    "world_leader": 12.0,   # hours (statements lose impact fast)
    "fundamental": 720.0,   # hours (30 days - fundamentals are sticky)
    "crypto": 4.0,          # hours (crypto moves fast)
}

_DEFAULT_HALF_LIFE = 24.0  # hours


def apply_decay(signal: AgentSignal, current_time: datetime) -> AgentSignal:
    """Return a new AgentSignal with confidence decayed by elapsed time."""
    hours_since = (current_time - signal.timestamp).total_seconds() / 3600.0
    if hours_since <= 0:
        return signal

    # Per-signal half-life override (e.g., world_leader statement types)
    half_life = signal.metadata.get("half_life_hours") if signal.metadata else None
    if half_life is None:
        half_life = SIGNAL_HALF_LIVES.get(signal.agent_type, _DEFAULT_HALF_LIFE)
    decay_factor = 0.5 ** (hours_since / half_life)

    return AgentSignal(
        agent_id=signal.agent_id,
        agent_type=signal.agent_type,
        symbol=signal.symbol,
        timestamp=signal.timestamp,
        direction=signal.direction,
        confidence=signal.confidence * decay_factor,
        timeframe=signal.timeframe,
        expected_holding_period=signal.expected_holding_period,
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        reasoning=signal.reasoning,
        features_used=signal.features_used,
        metadata=signal.metadata,
    )
