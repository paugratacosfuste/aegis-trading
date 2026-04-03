"""Dual momentum agent: Antonacci-style absolute + relative momentum.

Combines: is the asset going up (absolute) AND outperforming peers (relative)?
"""

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("momentum", "dual")
class DualMomentumAgent(BaseAgent):

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._lookback = config.get("lookback", 20)
        self._benchmark_return: float | None = config.get("benchmark_return")

    @property
    def agent_type(self) -> str:
        return "momentum"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        timeframe = candles[-1].timeframe if candles else "1h"

        if len(candles) < self._lookback + 1:
            return self._neutral_signal(symbol, timeframe)

        price_now = candles[-1].close
        price_past = candles[-(self._lookback + 1)].close
        if price_past == 0:
            return self._neutral_signal(symbol, timeframe)

        abs_return = (price_now - price_past) / price_past
        benchmark = self._benchmark_return if self._benchmark_return is not None else 0.0
        rel_momentum = abs_return - benchmark

        if abs_return > 0 and rel_momentum > 0:
            direction = min(rel_momentum * 10, 1.0)
        elif abs_return < 0:
            direction = max(abs_return * 5, -1.0)
        else:
            direction = 0.0  # Positive absolute but negative relative = neutral

        confidence = min(abs(abs_return) / 0.05, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={
                "abs_return": round(abs_return, 4),
                "rel_momentum": round(rel_momentum, 4),
            },
            features={"abs_return": abs_return, "rel_momentum": rel_momentum},
        )
