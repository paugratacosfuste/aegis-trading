"""RL safety module: hard constraints that are never overridden.

This is the single enforcement point for all RL outputs.
Clamps, validates, and vetoes before any RL recommendation is applied.
"""

import logging

from aegis.rl.constants import (
    AGENT_TYPES,
    MAX_KELLY_DIVERGENCE,
    MAX_POSITION_SIZE,
    MIN_POSITION_SIZE,
)
from aegis.rl.types import WeightConfig

logger = logging.getLogger(__name__)


def clamp_position_size(
    rl_size: float,
    kelly_size: float,
    portfolio_value: float,
) -> float:
    """Clamp RL-recommended position size within safety bounds.

    1. Absolute bounds: [MIN_POSITION_SIZE, MAX_POSITION_SIZE] of portfolio.
    2. Kelly divergence cap: can't differ from Kelly by more than 50%.

    Returns fraction of portfolio (0.0 if rejected).
    """
    # Absolute bounds
    size = max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, rl_size))

    # Kelly divergence cap
    if kelly_size > 0:
        lower = kelly_size * (1.0 - MAX_KELLY_DIVERGENCE)
        upper = kelly_size * (1.0 + MAX_KELLY_DIVERGENCE)
        size = max(lower, min(upper, size))

    # Re-apply absolute bounds after Kelly clamping
    size = max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, size))

    if size != rl_size:
        logger.debug(
            "Safety clamped position size: %.4f -> %.4f (kelly=%.4f)",
            rl_size, size, kelly_size,
        )

    return size


def validate_weight_config(config: WeightConfig) -> bool:
    """Validate a weight configuration meets safety criteria.

    1. Weights must sum to 1.0 (within tolerance).
    2. All weights must be non-negative.
    3. All 7 agent types must be present.
    """
    total = sum(config.weights.values())
    if abs(total - 1.0) > 0.01:
        logger.warning("Weight config %d: sum=%.4f (expected 1.0)", config.config_id, total)
        return False

    if any(v < 0 for v in config.weights.values()):
        logger.warning("Weight config %d: negative weight found", config.config_id)
        return False

    missing = set(AGENT_TYPES) - set(config.weights.keys())
    if missing:
        logger.warning("Weight config %d: missing types %s", config.config_id, missing)
        return False

    return True


def enforce_circuit_breaker(
    portfolio_value: float,
    start_of_day_value: float,
    daily_halt_threshold: float = -0.05,
) -> bool:
    """Check if circuit breaker should block RL actions.

    Returns True if RL actions are allowed, False if halted.
    """
    if start_of_day_value <= 0:
        return False

    daily_return = (portfolio_value - start_of_day_value) / start_of_day_value
    if daily_return <= daily_halt_threshold:
        logger.info("Circuit breaker HALT: daily return %.2f%% <= %.2f%%",
                     daily_return * 100, daily_halt_threshold * 100)
        return False

    return True
