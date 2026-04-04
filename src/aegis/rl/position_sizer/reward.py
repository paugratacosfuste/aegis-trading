"""Position sizing reward wrapper.

Thin wrapper adding consistency bonus to base position_sizing_reward.
"""

from aegis.rl.common.reward import position_sizing_reward


def compute_reward(
    trade_pnl: float,
    position_size_pct: float,
    max_drawdown_during: float = 0.0,
    prev_size_pct: float | None = None,
) -> float:
    """Compute reward with optional consistency bonus.

    Consistency bonus: small reward for sizing similarly to previous trades
    (avoids erratic position sizing).
    """
    base = position_sizing_reward(trade_pnl, position_size_pct, max_drawdown_during)

    if prev_size_pct is not None and prev_size_pct > 0:
        change_ratio = abs(position_size_pct - prev_size_pct) / prev_size_pct
        if change_ratio < 0.2:
            base += 0.05  # Small consistency bonus
        elif change_ratio > 1.0:
            base -= 0.05  # Penalty for wild swings

    return base
