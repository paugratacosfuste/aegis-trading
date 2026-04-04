"""Shared reward functions for RL components.

Each function maps outcomes to rewards with proper incentive alignment:
- Positive rewards for good risk-adjusted returns
- Penalties for excessive risk, large drawdowns, or poor timing
"""

import numpy as np


def weight_allocation_reward(
    portfolio_returns_24h: list[float],
) -> float:
    """Reward for weight allocation: 24-hour Sharpe ratio.

    Higher is better. Penalizes volatility, not just returns.
    """
    if len(portfolio_returns_24h) < 2:
        return 0.0

    mean_ret = np.mean(portfolio_returns_24h)
    std_ret = np.std(portfolio_returns_24h)

    if std_ret < 1e-8:
        return float(mean_ret * 100)  # Scale up tiny returns

    return float(mean_ret / std_ret)


def position_sizing_reward(
    trade_pnl: float,
    position_size_pct: float,
    max_drawdown_during: float = 0.0,
) -> float:
    """Reward for position sizing: risk-adjusted PnL.

    Components:
      - PnL (primary signal)
      - Size penalty: larger positions need proportionally larger PnL
      - Drawdown penalty: penalize if max drawdown during trade was large
    """
    # Primary signal: raw PnL
    risk_adjusted = trade_pnl

    # Size penalty: larger positions with losses are penalized proportionally
    size_penalty = 0.0
    if trade_pnl < 0:
        size_penalty = trade_pnl * (position_size_pct / 0.03)  # Scale loss by relative size

    # Drawdown penalty
    dd_penalty = 0.0
    if max_drawdown_during < -0.02:
        dd_penalty = max_drawdown_during * 10  # Amplify drawdown signal

    return risk_adjusted + size_penalty + dd_penalty


def exit_management_reward(
    step_pnl_change: float,
    action_taken: int,
    bars_held: int,
    r_multiple: float,
) -> float:
    """Per-step reward for exit management.

    Incentives:
      - Holding profitable positions: small positive reward
      - Exiting at good R-multiples: bonus
      - Holding losing positions too long: penalty
      - Hold cost: small negative each step to discourage inaction
    """
    hold_cost = -0.001  # Small penalty per step

    if action_taken == 4:  # FULL_EXIT
        # Bonus for exiting at good R
        if r_multiple >= 2.0:
            return step_pnl_change + 0.5
        elif r_multiple >= 1.0:
            return step_pnl_change + 0.2
        elif r_multiple <= -1.0:
            return step_pnl_change - 0.1  # Late stop-loss penalty
        return step_pnl_change

    if action_taken == 1:  # TIGHTEN_STOP
        # Reward for tightening when profitable
        if r_multiple > 0.5:
            return step_pnl_change + 0.05
        return step_pnl_change - 0.02  # Slight penalty for premature tightening

    if action_taken in (2, 3):  # PARTIAL exits
        if r_multiple >= 1.5:
            return step_pnl_change + 0.3  # Good partial timing
        return step_pnl_change - 0.05  # Premature partial

    # HOLD (action 0)
    if r_multiple < -0.5 and bars_held > 48:
        return step_pnl_change - 0.1  # Penalty for holding losers too long

    return step_pnl_change + hold_cost
