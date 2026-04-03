"""Cross-sectional momentum: ranks asset vs peers.

Returns neutral without peer data. Skeleton for Phase 2.
"""

import statistics

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("momentum", "cross_sectional")
class CrossSectionalMomentumAgent(BaseAgent):

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._lookback = config.get("lookback", 20)
        self._peer_returns: list[float] | None = config.get("peer_returns")

    @property
    def agent_type(self) -> str:
        return "momentum"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        timeframe = candles[-1].timeframe if candles else "1h"

        if self._peer_returns is None or len(candles) < self._lookback + 1:
            return self._neutral_signal(symbol, timeframe)

        price_now = candles[-1].close
        price_past = candles[-(self._lookback + 1)].close
        if price_past == 0:
            return self._neutral_signal(symbol, timeframe)

        asset_return = (price_now - price_past) / price_past
        peer_mean = statistics.mean(self._peer_returns) if self._peer_returns else 0.0
        rel_momentum = asset_return - peer_mean

        direction = min(rel_momentum * 10, 1.0)
        direction = max(-1.0, direction)
        confidence = min(abs(rel_momentum) / 0.05, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={"asset_return": round(asset_return, 4), "rel_momentum": round(rel_momentum, 4)},
            features={"asset_return": asset_return, "rel_momentum": rel_momentum},
        )
