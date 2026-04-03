"""Momentum acceleration agent: rate of change of momentum.

Compares short-window return to long-window return.
Positive acceleration = momentum increasing.
Negative acceleration = momentum fading.
"""

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("momentum", "acceleration")
class MomentumAccelerationAgent(BaseAgent):

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._short_window = config.get("short_window", 5)
        self._long_window = config.get("long_window", 20)

    @property
    def agent_type(self) -> str:
        return "momentum"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        timeframe = candles[-1].timeframe if candles else "1h"

        if len(candles) < self._long_window + 1:
            return self._neutral_signal(symbol, timeframe)

        price_now = candles[-1].close
        price_short_ago = candles[-(self._short_window + 1)].close
        price_long_ago = candles[-(self._long_window + 1)].close

        if price_short_ago == 0 or price_long_ago == 0:
            return self._neutral_signal(symbol, timeframe)

        short_return = (price_now - price_short_ago) / price_short_ago
        long_return = (price_now - price_long_ago) / price_long_ago

        # Acceleration: short momentum > long momentum = accelerating
        acceleration = short_return - long_return

        direction = acceleration * 20.0  # Scale
        direction = max(-1.0, min(1.0, direction))
        confidence = min(abs(acceleration) / 0.03, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={
                "short_return": round(short_return, 4),
                "long_return": round(long_return, 4),
                "acceleration": round(acceleration, 4),
            },
            features={
                "short_return": short_return,
                "long_return": long_return,
                "acceleration": acceleration,
            },
        )
