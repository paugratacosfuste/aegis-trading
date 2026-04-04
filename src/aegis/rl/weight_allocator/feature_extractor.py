"""Feature extraction wrapper for weight allocation bandit."""

import numpy as np

from aegis.common.types import AgentSignal
from aegis.rl.common.feature_builder import build_weight_context


def extract_context(
    signals: list[AgentSignal],
    regime: str,
    portfolio_value: float,
    equity_curve: list[float] | None = None,
) -> np.ndarray:
    """Extract context features for weight allocation bandit.

    Thin wrapper around build_weight_context.
    """
    return build_weight_context(signals, regime, portfolio_value, equity_curve)
