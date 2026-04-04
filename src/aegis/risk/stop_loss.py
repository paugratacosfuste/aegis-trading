"""ATR-based stop-loss calculation.

From 04-RISK-MANAGEMENT.md.
"""

ATR_MULTIPLIERS = {
    "low": 1.5,
    "normal": 2.0,
    "high": 2.5,
    "extreme": 3.0,
}

TIMEFRAME_ADJUSTMENTS = {
    "1m": 0.5,
    "5m": 0.6,
    "15m": 0.7,
    "1h": 1.0,
    "4h": 1.3,
    "1d": 1.8,
}


def calculate_stop_loss(
    entry_price: float,
    direction: str,
    atr_14: float,
    volatility_regime: str = "normal",
    timeframe: str = "1h",
) -> float:
    """Calculate ATR-based stop-loss price.

    Returns the stop-loss price level.
    """
    multiplier = ATR_MULTIPLIERS.get(volatility_regime, 2.0)
    tf_adj = TIMEFRAME_ADJUSTMENTS.get(timeframe, 1.0)

    stop_distance = atr_14 * multiplier * tf_adj

    if direction == "LONG":
        return entry_price - stop_distance
    else:
        return entry_price + stop_distance


def update_trailing_stop(
    current_stop: float,
    current_price: float,
    direction: str,
    atr_14: float,
    unrealized_pnl_pct: float,
) -> float:
    """Trailing stop: tightens as profit grows.

    Only activates once >1% profitable. Never loosens the stop.
    Trail distance = 1.5 * ATR.
    """
    if direction == "LONG" and unrealized_pnl_pct > 0.01:
        trail_distance = atr_14 * 1.5
        new_stop = current_price - trail_distance
        return max(new_stop, current_stop)

    if direction == "SHORT" and unrealized_pnl_pct > 0.01:
        trail_distance = atr_14 * 1.5
        new_stop = current_price + trail_distance
        return min(new_stop, current_stop)

    return current_stop
