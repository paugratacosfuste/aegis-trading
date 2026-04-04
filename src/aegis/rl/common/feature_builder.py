"""Shared feature builder for RL components.

Builds observation vectors from existing Aegis types. Handles missing data
with zeros, normalizes where needed. Used for both training and inference.
"""

import numpy as np

from aegis.common.types import AgentSignal, MacroDataPoint, MarketDataPoint, Position
from aegis.rl.constants import EXIT_OBS_DIM, POSITION_OBS_DIM, WEIGHT_CONTEXT_DIM


def build_weight_context(
    signals: list[AgentSignal],
    regime: str,
    portfolio_value: float,
    equity_curve: list[float] | None = None,
) -> np.ndarray:
    """Build context vector for weight allocation bandit.

    Features (11):
      [0] num_signals (normalized 0-1, max 100)
      [1] mean_confidence
      [2] mean_abs_direction
      [3] direction_agreement (fraction same sign as majority)
      [4-7] regime one-hot (bull, bear, transition, recovery)
      [8] portfolio_drawdown (0 if no curve)
      [9] recent_volatility (std of last 24 equity changes, 0 if unavailable)
      [10] signal_type_diversity (unique types / 7)
    """
    obs = np.zeros(WEIGHT_CONTEXT_DIM, dtype=np.float32)

    if not signals:
        return obs

    obs[0] = min(len(signals) / 100.0, 1.0)
    obs[1] = np.mean([s.confidence for s in signals])
    obs[2] = np.mean([abs(s.direction) for s in signals])

    # Direction agreement
    directions = [s.direction for s in signals]
    mean_dir = np.mean(directions)
    if mean_dir != 0:
        majority_sign = 1.0 if mean_dir > 0 else -1.0
        obs[3] = sum(1 for d in directions if d * majority_sign > 0) / len(directions)

    # Regime one-hot
    regime_map = {"bull": 4, "bear": 5, "transition": 6, "recovery": 7}
    if regime in regime_map:
        obs[regime_map[regime]] = 1.0

    # Portfolio drawdown
    if equity_curve and len(equity_curve) > 1:
        peak = max(equity_curve)
        if peak > 0:
            obs[8] = (peak - equity_curve[-1]) / peak

    # Recent volatility
    if equity_curve and len(equity_curve) > 24:
        recent = equity_curve[-24:]
        changes = [(recent[i] - recent[i - 1]) / recent[i - 1]
                   for i in range(1, len(recent)) if recent[i - 1] > 0]
        if changes:
            obs[9] = float(np.std(changes))

    # Signal type diversity
    unique_types = len({s.agent_type for s in signals})
    obs[10] = unique_types / 7.0

    return obs


def build_position_obs(
    candles: list[MarketDataPoint],
    signal_confidence: float,
    signal_direction: float,
    portfolio_value: float,
    open_positions: int,
    regime: str,
    atr_14: float,
) -> np.ndarray:
    """Build observation vector for position sizing.

    Features (25):
      [0] signal_confidence
      [1] signal_direction (normalized -1 to 1)
      [2-5] price features (return_1h, return_24h, vol_24h, atr_ratio)
      [6-9] regime one-hot (bull, bear, transition, recovery)
      [10] portfolio_utilization (open_positions / 10)
      [11] drawdown (from recent equity, approximated from candle returns)
      [12-24] rolling stats (13 features from candle data)
    """
    obs = np.zeros(POSITION_OBS_DIM, dtype=np.float32)

    obs[0] = signal_confidence
    obs[1] = signal_direction

    if len(candles) >= 2:
        obs[2] = (candles[-1].close - candles[-2].close) / candles[-2].close if candles[-2].close > 0 else 0.0

    if len(candles) >= 24:
        obs[3] = (candles[-1].close - candles[-24].close) / candles[-24].close if candles[-24].close > 0 else 0.0
        closes = [c.close for c in candles[-24:]]
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
                   for i in range(1, len(closes)) if closes[i - 1] > 0]
        obs[4] = float(np.std(returns)) if returns else 0.0

    # ATR ratio (ATR / price)
    if candles and candles[-1].close > 0:
        obs[5] = atr_14 / candles[-1].close

    # Regime one-hot
    regime_map = {"bull": 6, "bear": 7, "transition": 8, "recovery": 9}
    if regime in regime_map:
        obs[regime_map[regime]] = 1.0

    obs[10] = min(open_positions / 10.0, 1.0)

    # Rolling stats from candles
    if len(candles) >= 14:
        closes = np.array([c.close for c in candles[-14:]])
        highs = np.array([c.high for c in candles[-14:]])
        lows = np.array([c.low for c in candles[-14:]])
        volumes = np.array([c.volume for c in candles[-14:]])

        if closes[0] > 0:
            obs[12] = (closes[-1] - closes[0]) / closes[0]  # 14-bar return
        obs[13] = float(np.std(closes) / np.mean(closes)) if np.mean(closes) > 0 else 0.0  # CV
        obs[14] = float(np.mean(highs - lows) / np.mean(closes)) if np.mean(closes) > 0 else 0.0  # avg range ratio
        obs[15] = float(np.std(volumes) / np.mean(volumes)) if np.mean(volumes) > 0 else 0.0  # volume CV
        obs[16] = float(volumes[-1] / np.mean(volumes)) if np.mean(volumes) > 0 else 0.0  # relative volume

        # Simple momentum: count of up bars / total
        up_bars = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
        obs[17] = up_bars / (len(closes) - 1)

    return obs


def build_exit_obs(
    candles: list[MarketDataPoint],
    position: dict,
    atr_14: float,
) -> np.ndarray:
    """Build observation vector for exit management.

    Features (20):
      [0] r_multiple (unrealized PnL / risk)
      [1] unrealized_pnl_pct
      [2] bars_held (normalized, / 200)
      [3] partial_taken (0 or 1)
      [4] stop_distance_pct (distance to stop / entry)
      [5-8] price features (recent returns, volatility)
      [9-12] regime one-hot
      [13] atr_ratio
      [14-19] reserved / candle stats
    """
    obs = np.zeros(EXIT_OBS_DIM, dtype=np.float32)

    entry = position.get("entry_price", 0.0)
    risk_amount = position.get("risk_amount", 0.0)
    direction = position.get("direction", "LONG")

    if not candles or entry <= 0:
        return obs

    current_price = candles[-1].close

    # R-multiple
    if direction == "LONG":
        pnl_per_unit = current_price - entry
    else:
        pnl_per_unit = entry - current_price

    if risk_amount > 0:
        obs[0] = pnl_per_unit / risk_amount
    obs[1] = pnl_per_unit / entry

    # Bars held
    entry_index = position.get("entry_index", 0)
    bars_held = max(0, len(candles) - 1 - entry_index) if entry_index < len(candles) else 0
    obs[2] = min(bars_held / 200.0, 1.0)

    obs[3] = 1.0 if position.get("partial_taken", False) else 0.0

    # Stop distance
    stop = position.get("stop_loss", entry)
    obs[4] = abs(current_price - stop) / entry if entry > 0 else 0.0

    # Price features
    if len(candles) >= 2:
        obs[5] = (candles[-1].close - candles[-2].close) / candles[-2].close if candles[-2].close > 0 else 0.0
    if len(candles) >= 24:
        obs[6] = (candles[-1].close - candles[-24].close) / candles[-24].close if candles[-24].close > 0 else 0.0
        closes = [c.close for c in candles[-24:]]
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
                   for i in range(1, len(closes)) if closes[i - 1] > 0]
        obs[7] = float(np.std(returns)) if returns else 0.0

    # ATR ratio
    if candles[-1].close > 0:
        obs[13] = atr_14 / candles[-1].close

    return obs
