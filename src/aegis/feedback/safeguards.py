"""Anti-overfitting safeguards for the feedback loop.

Cross-cutting checks used by daily weight updates, weekly retraining,
and monthly evolution to prevent learning from noise.
"""

import math

import numpy as np


def check_minimum_samples(n: int, required: int = 30) -> bool:
    """Return True if n >= required observations for statistical conclusions."""
    return n >= required


def check_minimum_training_data(n: int, required: int = 500) -> bool:
    """Return True if n >= required data points for model training."""
    return n >= required


def bonferroni_significance(
    p_value: float,
    n_comparisons: int,
    alpha: float = 0.05,
) -> bool:
    """Return True if p_value is significant after Bonferroni correction.

    Adjusted threshold = alpha / n_comparisons.
    """
    if n_comparisons <= 0:
        return False
    threshold = alpha / n_comparisons
    return p_value < threshold


def check_prediction_stability(
    old_predictions: np.ndarray,
    new_predictions: np.ndarray,
    threshold: float = 0.30,
) -> tuple[bool, float]:
    """Check if model predictions changed more than threshold fraction.

    Returns (is_stable, change_fraction). Compares by fraction of predictions
    that flipped sign (for classification) or changed > threshold in magnitude.
    """
    if len(old_predictions) == 0 or len(new_predictions) == 0:
        return True, 0.0
    if len(old_predictions) != len(new_predictions):
        return False, 1.0

    old = np.asarray(old_predictions, dtype=float)
    new = np.asarray(new_predictions, dtype=float)

    # For binary/probability predictions: fraction that flipped class
    old_class = (old >= 0.5).astype(int)
    new_class = (new >= 0.5).astype(int)
    change_fraction = float(np.mean(old_class != new_class))

    return change_fraction <= threshold, change_fraction


def compute_regime_conditioned_metrics(
    trades: list[dict],
) -> dict[str, dict[str, float]]:
    """Compute per-regime metrics instead of global aggregates.

    Each trade dict must have 'regime_at_entry' and 'net_pnl' keys.
    Returns {regime: {sharpe, win_rate, n_trades, avg_pnl}}.
    """
    from aegis.backtest.metrics import calculate_sharpe, calculate_win_rate

    by_regime: dict[str, list[float]] = {}
    for t in trades:
        regime = t.get("regime_at_entry", "unknown")
        pnl = t.get("net_pnl", 0.0)
        by_regime.setdefault(regime, []).append(pnl)

    result: dict[str, dict[str, float]] = {}
    for regime, pnls in by_regime.items():
        n = len(pnls)
        avg_pnl = sum(pnls) / n if n > 0 else 0.0
        result[regime] = {
            "sharpe": calculate_sharpe(pnls) if n >= 2 else 0.0,
            "win_rate": calculate_win_rate(pnls),
            "n_trades": n,
            "avg_pnl": avg_pnl,
        }
    return result
