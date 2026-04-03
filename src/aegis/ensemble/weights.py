"""Ensemble weight system: base weights and regime-conditional adjustments.

Voting types: technical, statistical, momentum, sentiment, geopolitical,
world_leader, crypto. Non-voting types (macro, fundamental) influence
the ensemble through regime extraction and confidence modifiers.
"""

BASE_TYPE_WEIGHTS: dict[str, float] = {
    "technical": 0.25,
    "statistical": 0.20,
    "momentum": 0.20,
    "sentiment": 0.15,
    "geopolitical": 0.05,
    "world_leader": 0.05,
    "crypto": 0.10,
}

REGIME_WEIGHT_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "strong_trend_up": {
        "technical": 1.3,
        "momentum": 1.4,
        "statistical": 0.6,
        "sentiment": 1.1,
        "geopolitical": 0.8,
        "world_leader": 0.8,
        "crypto": 1.2,
    },
    "strong_trend_down": {
        "technical": 1.2,
        "momentum": 1.3,
        "statistical": 0.5,
        "sentiment": 1.2,
        "geopolitical": 1.2,
        "world_leader": 1.0,
        "crypto": 1.1,
    },
    "mean_reverting": {
        "technical": 0.8,
        "momentum": 0.6,
        "statistical": 1.5,
        "sentiment": 0.9,
        "geopolitical": 0.8,
        "world_leader": 0.8,
        "crypto": 0.9,
    },
    "high_volatility": {
        "technical": 0.7,
        "momentum": 0.5,
        "statistical": 0.8,
        "sentiment": 1.3,
        "geopolitical": 1.3,
        "world_leader": 1.2,
        "crypto": 0.8,
    },
    "low_volatility": {
        "technical": 1.1,
        "momentum": 1.1,
        "statistical": 1.2,
        "sentiment": 0.7,
        "geopolitical": 0.7,
        "world_leader": 0.7,
        "crypto": 1.0,
    },
    "crisis": {
        "technical": 0.3,
        "momentum": 0.2,
        "statistical": 0.3,
        "sentiment": 0.5,
        "geopolitical": 1.8,
        "world_leader": 1.5,
        "crypto": 0.3,
    },
    # Macro regime types (extracted from macro agents)
    "risk_on": {
        "technical": 1.1,
        "momentum": 1.3,
        "statistical": 0.9,
        "sentiment": 1.2,
        "geopolitical": 0.7,
        "world_leader": 0.8,
        "crypto": 1.3,
    },
    "risk_off": {
        "technical": 0.8,
        "momentum": 0.6,
        "statistical": 1.2,
        "sentiment": 1.3,
        "geopolitical": 1.5,
        "world_leader": 1.3,
        "crypto": 0.5,
    },
    "recession_risk": {
        "technical": 0.5,
        "momentum": 0.3,
        "statistical": 0.4,
        "sentiment": 0.6,
        "geopolitical": 1.5,
        "world_leader": 1.3,
        "crypto": 0.3,
    },
    # Yield curve / economic cycle regimes
    "early_cycle": {
        "technical": 1.1,
        "momentum": 1.4,
        "statistical": 0.8,
        "sentiment": 1.1,
        "geopolitical": 0.7,
        "world_leader": 0.8,
        "crypto": 1.2,
    },
    "mid_cycle": {
        "technical": 1.0,
        "momentum": 1.0,
        "statistical": 1.0,
        "sentiment": 1.0,
        "geopolitical": 1.0,
        "world_leader": 1.0,
        "crypto": 1.0,
    },
    "late_cycle": {
        "technical": 0.9,
        "momentum": 0.7,
        "statistical": 1.2,
        "sentiment": 1.2,
        "geopolitical": 1.2,
        "world_leader": 1.1,
        "crypto": 0.7,
    },
    "expansion": {
        "technical": 1.1,
        "momentum": 1.3,
        "statistical": 0.9,
        "sentiment": 1.1,
        "geopolitical": 0.7,
        "world_leader": 0.8,
        "crypto": 1.2,
    },
    "contraction": {
        "technical": 0.7,
        "momentum": 0.4,
        "statistical": 0.5,
        "sentiment": 0.8,
        "geopolitical": 1.4,
        "world_leader": 1.3,
        "crypto": 0.4,
    },
    # Inflation regimes
    "deflationary": {
        "technical": 0.9,
        "momentum": 0.8,
        "statistical": 1.3,
        "sentiment": 1.0,
        "geopolitical": 1.0,
        "world_leader": 1.0,
        "crypto": 0.8,
    },
    "low": {
        "technical": 1.0,
        "momentum": 1.1,
        "statistical": 1.0,
        "sentiment": 1.0,
        "geopolitical": 0.9,
        "world_leader": 0.9,
        "crypto": 1.1,
    },
    "moderate": {
        "technical": 1.0,
        "momentum": 1.0,
        "statistical": 1.0,
        "sentiment": 1.0,
        "geopolitical": 1.0,
        "world_leader": 1.0,
        "crypto": 1.0,
    },
    "high": {
        "technical": 0.8,
        "momentum": 0.6,
        "statistical": 1.1,
        "sentiment": 1.3,
        "geopolitical": 1.3,
        "world_leader": 1.2,
        "crypto": 0.6,
    },
    "very_high": {
        "technical": 0.6,
        "momentum": 0.4,
        "statistical": 0.8,
        "sentiment": 1.4,
        "geopolitical": 1.5,
        "world_leader": 1.4,
        "crypto": 0.4,
    },
    # HMM regimes
    "bull": {
        "technical": 1.2,
        "momentum": 1.4,
        "statistical": 0.7,
        "sentiment": 1.1,
        "geopolitical": 0.7,
        "world_leader": 0.8,
        "crypto": 1.3,
    },
    "bear": {
        "technical": 0.8,
        "momentum": 0.5,
        "statistical": 0.6,
        "sentiment": 1.2,
        "geopolitical": 1.4,
        "world_leader": 1.3,
        "crypto": 0.4,
    },
    "transition": {
        "technical": 1.0,
        "momentum": 0.8,
        "statistical": 1.2,
        "sentiment": 1.1,
        "geopolitical": 1.0,
        "world_leader": 1.0,
        "crypto": 0.9,
    },
    "recovery": {
        "technical": 1.1,
        "momentum": 1.3,
        "statistical": 0.9,
        "sentiment": 1.2,
        "geopolitical": 0.8,
        "world_leader": 0.9,
        "crypto": 1.2,
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
