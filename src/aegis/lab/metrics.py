"""Lab metrics: bridges cohort runner data to backtest metrics."""

from datetime import datetime

from aegis.backtest.metrics import (
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe,
    calculate_win_rate,
)
from aegis.lab.types import CohortPerformance


def calculate_cohort_performance(
    cohort_id: str,
    pnls: list[float],
    equity_curve: list[float],
) -> CohortPerformance:
    """Calculate performance metrics from raw PnL and equity data."""
    returns = []
    if len(equity_curve) > 1:
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                returns.append((equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1])

    return CohortPerformance(
        cohort_id=cohort_id,
        sharpe=calculate_sharpe(returns),
        win_rate=calculate_win_rate(pnls),
        max_drawdown=calculate_max_drawdown(equity_curve),
        profit_factor=calculate_profit_factor(pnls),
        total_trades=len(pnls),
        net_pnl=sum(pnls),
        equity_curve=tuple(equity_curve),
    )
