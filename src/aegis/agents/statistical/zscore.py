"""Statistical agent: 20-day z-score mean reversion.

Implements stat_01 from 02-AGENT-ARCHITECTURE.md.
"""

import statistics

from aegis.agents.base import BaseAgent
from aegis.common.types import AgentSignal, MarketDataPoint


class ZScoreAgent(BaseAgent):
    """Generates mean-reversion signals based on z-score of price vs rolling mean."""

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._lookback = config.get("lookback", 20)

    @property
    def agent_type(self) -> str:
        return "statistical"

    def generate_signal(
        self, symbol: str, candles: list[MarketDataPoint]
    ) -> AgentSignal:
        timeframe = candles[-1].timeframe if candles else "1h"

        if len(candles) < self._lookback:
            return self._neutral_signal(symbol, timeframe)

        closes = [c.close for c in candles[-self._lookback :]]
        mean = statistics.mean(closes)
        std = statistics.stdev(closes) if len(closes) > 1 else 0.0

        if std == 0:
            return self._neutral_signal(symbol, timeframe)

        z = (closes[-1] - mean) / std

        # Mean reversion: negative z (below mean) -> buy, positive z -> sell
        direction = -z / 3.0
        confidence = min(abs(z) / 3.0, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={
                "z_score": round(z, 3),
                "mean_20d": round(mean, 2),
                "std_20d": round(std, 2),
                "current_price": closes[-1],
            },
            features={
                "z_score": round(z, 4),
                "rolling_mean": round(mean, 4),
                "rolling_std": round(std, 4),
            },
        )
