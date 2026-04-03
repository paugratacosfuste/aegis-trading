"""Ensemble weight system: base weights and regime-conditional adjustments.

Phase 2 supports 4 signal-generating agent types.
Phase 3 will add macro, geopolitical, world_leader, fundamental.
"""

BASE_TYPE_WEIGHTS: dict[str, float] = {
    "technical": 0.30,
    "statistical": 0.25,
    "momentum": 0.25,
    "sentiment": 0.20,
}

REGIME_WEIGHT_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "strong_trend_up": {
        "technical": 1.3,
        "momentum": 1.4,
        "statistical": 0.6,
        "sentiment": 1.1,
    },
    "strong_trend_down": {
        "technical": 1.2,
        "momentum": 1.3,
        "statistical": 0.5,
        "sentiment": 1.2,
    },
    "mean_reverting": {
        "technical": 0.8,
        "momentum": 0.6,
        "statistical": 1.5,
        "sentiment": 0.9,
    },
    "high_volatility": {
        "technical": 0.7,
        "momentum": 0.5,
        "statistical": 0.8,
        "sentiment": 1.3,
    },
    "low_volatility": {
        "technical": 1.1,
        "momentum": 1.1,
        "statistical": 1.2,
        "sentiment": 0.7,
    },
    "crisis": {
        "technical": 0.3,
        "momentum": 0.2,
        "statistical": 0.3,
        "sentiment": 0.5,
    },
}


def apply_regime_weights(
    base_weights: dict[str, float],
    regime: str = "normal",
) -> dict[str, float]:
    """Apply regime adjustments to base weights and re-normalize.

    Args:
        base_weights: {agent_type: weight} base values.
        regime: Current market regime. "normal" means no adjustment.

    Returns:
        Adjusted weights that sum to 1.0.
    """
    if regime == "normal" or regime not in REGIME_WEIGHT_ADJUSTMENTS:
        total = sum(base_weights.values())
        if total == 0:
            return base_weights
        return {k: v / total for k, v in base_weights.items()}

    adjustments = REGIME_WEIGHT_ADJUSTMENTS[regime]
    adjusted = {
        k: v * adjustments.get(k, 1.0)
        for k, v in base_weights.items()
    }

    total = sum(adjusted.values())
    if total == 0:
        return adjusted
    return {k: v / total for k, v in adjusted.items()}
