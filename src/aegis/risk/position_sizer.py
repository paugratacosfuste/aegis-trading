"""Half-Kelly position sizing with hard caps.

From 04-RISK-MANAGEMENT.md.
"""


def calculate_position_size(
    portfolio_value: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    signal_confidence: float,
    max_risk_pct: float = 0.02,
) -> float:
    """Calculate position size using half-Kelly criterion.

    Returns dollar value of the position (0.0 if no edge).
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0

    b = avg_win / abs(avg_loss)  # Win/loss ratio
    p = win_rate
    q = 1 - p

    kelly = (b * p - q) / b
    if kelly <= 0:
        return 0.0

    half_kelly = kelly / 2.0
    adjusted = half_kelly * signal_confidence

    # Max position based on risk cap
    max_position_by_risk = (portfolio_value * max_risk_pct) / abs(avg_loss)

    # Never more than 10% of portfolio in one position
    max_position_by_pct = portfolio_value * 0.10

    return min(
        portfolio_value * adjusted,
        max_position_by_risk,
        max_position_by_pct,
    )
