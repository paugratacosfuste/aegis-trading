"""Backtest performance metrics.

From 04-RISK-MANAGEMENT.md.
"""

import math


def calculate_sharpe(returns: list[float], periods_per_year: float = 252.0) -> float:
    """Annualized Sharpe ratio (risk-free rate assumed 0)."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: list[float]) -> float:
    """Maximum drawdown as a negative fraction (e.g., -0.20 for 20% drawdown)."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def calculate_win_rate(pnls: list[float]) -> float:
    """Fraction of trades with PnL >= 0."""
    if not pnls:
        return 0.0
    wins = sum(1 for p in pnls if p >= 0)
    return wins / len(pnls)


def calculate_profit_factor(pnls: list[float]) -> float:
    """Gross profit / gross loss."""
    if not pnls:
        return 0.0
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss
