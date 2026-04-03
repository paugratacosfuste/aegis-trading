"""Momentum agent: 20-day time-series momentum.

Implements mom_03 from 02-AGENT-ARCHITECTURE.md.
"""

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("momentum", "timeseries")
class MomentumAgent(BaseAgent):
    """Generates signals based on N-day price return (time-series momentum)."""

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._lookback = config.get("lookback", 20)

    @property
    def agent_type(self) -> str:
        return "momentum"

    def generate_signal(
        self, symbol: str, candles: list[MarketDataPoint]
    ) -> AgentSignal:
        timeframe = candles[-1].timeframe if candles else "1h"

        if len(candles) < self._lookback + 1:
            return self._neutral_signal(symbol, timeframe)

        price_now = candles[-1].close
        price_past = candles[-(self._lookback + 1)].close

        if price_past == 0:
            return self._neutral_signal(symbol, timeframe)

        return_nd = (price_now - price_past) / price_past

        # Direction: scaled by 10x, clipped to [-1, 1]
        direction = return_nd * 10.0

        # Confidence: 5% move = max confidence
        confidence = min(abs(return_nd) / 0.05, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={
                "return_20d": round(return_nd, 4),
                "price_now": round(price_now, 2),
                "price_20d_ago": round(price_past, 2),
            },
            features={
                "return_20d": round(return_nd, 6),
                "price_current": round(price_now, 4),
                "price_lookback": round(price_past, 4),
            },
        )
