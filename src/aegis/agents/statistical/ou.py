"""Ornstein-Uhlenbeck agent: estimates mean-reversion speed and equilibrium.

Fits OU process via OLS: dP = theta*(mu - P)*dt + sigma*dW
Signal: distance from estimated equilibrium, weighted by reversion speed.
"""

import numpy as np

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("statistical", "ornstein_uhlenbeck")
class OUAgent(BaseAgent):

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._lookback = config.get("lookback", 60)

    @property
    def agent_type(self) -> str:
        return "statistical"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        timeframe = candles[-1].timeframe if candles else "1h"
        if len(candles) < self._lookback:
            return self._neutral_signal(symbol, timeframe)

        prices = np.array([c.close for c in candles[-self._lookback:]])
        dp = np.diff(prices)
        p_lag = prices[:-1]

        if len(p_lag) < 2:
            return self._neutral_signal(symbol, timeframe)

        # OLS: dP = a + b * P_lag
        # theta = -b, mu = -a/b
        x = np.column_stack([np.ones(len(p_lag)), p_lag])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(x, dp, rcond=None)
        except np.linalg.LinAlgError:
            return self._neutral_signal(symbol, timeframe)

        a, b = coeffs[0], coeffs[1]

        if b >= 0:
            # No mean reversion (trending or random walk)
            return self._neutral_signal(symbol, timeframe)

        theta = -b
        mu = -a / b if b != 0 else prices[-1]

        current = prices[-1]
        distance = (current - mu) / mu if mu != 0 else 0.0

        # Direction: mean reversion toward mu
        direction = -distance * min(theta * 10, 3.0)
        direction = max(-1.0, min(1.0, direction))

        # Confidence: higher theta = faster reversion = more confident
        confidence = min(theta * 5, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={
                "theta": round(theta, 4),
                "mu": round(mu, 2),
                "current": round(current, 2),
                "distance_pct": round(distance * 100, 2),
            },
            features={"theta": theta, "mu": mu, "distance": distance},
        )
