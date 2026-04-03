"""Pairs cointegration agent: statistical arbitrage on cointegrated pairs.

Requires historical data for both symbols. Returns neutral without pair data.
Skeleton for Phase 2, fully functional when multi-symbol pipeline is available.
"""

import numpy as np

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("statistical", "pairs")
class PairsAgent(BaseAgent):

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._lookback = config.get("lookback", 90)
        self._pair_prices: list[float] | None = config.get("pair_prices")

    @property
    def agent_type(self) -> str:
        return "statistical"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        timeframe = candles[-1].timeframe if candles else "1h"

        if self._pair_prices is None or len(candles) < self._lookback:
            return self._neutral_signal(symbol, timeframe)

        prices_a = np.array([c.close for c in candles[-self._lookback:]])
        prices_b = np.array(self._pair_prices[-self._lookback:])

        if len(prices_b) < self._lookback:
            return self._neutral_signal(symbol, timeframe)

        # Simple spread: log(A) - beta * log(B)
        log_a = np.log(prices_a)
        log_b = np.log(prices_b)

        # OLS: log_a = alpha + beta * log_b
        x = np.column_stack([np.ones(len(log_b)), log_b])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(x, log_a, rcond=None)
        except np.linalg.LinAlgError:
            return self._neutral_signal(symbol, timeframe)

        alpha, beta = coeffs[0], coeffs[1]
        spread = log_a - alpha - beta * log_b

        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread))
        if spread_std == 0:
            return self._neutral_signal(symbol, timeframe)

        z = (spread[-1] - spread_mean) / spread_std

        direction = -float(z) / 2.5
        direction = max(-1.0, min(1.0, direction))
        confidence = min(abs(float(z)) / 3.0, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={
                "spread_z": round(float(z), 3), "beta": round(float(beta), 4),
            },
            features={"spread_z": float(z), "beta": float(beta)},
        )
