"""Sector-relative z-score agent: how far from sector peers.

Requires peer returns data to function. Returns neutral without peer data.
Skeleton for Phase 2, fully functional when peer data pipeline is available.
"""

import statistics

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("statistical", "sector_relative")
class SectorRelativeAgent(BaseAgent):

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._lookback = config.get("lookback", 50)
        self._peer_returns: list[float] | None = config.get("peer_returns")

    @property
    def agent_type(self) -> str:
        return "statistical"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        timeframe = candles[-1].timeframe if candles else "1h"

        if self._peer_returns is None or len(candles) < self._lookback:
            return self._neutral_signal(symbol, timeframe)

        closes = [c.close for c in candles[-self._lookback:]]
        asset_return = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0.0

        peer_mean = statistics.mean(self._peer_returns) if self._peer_returns else 0.0
        peer_std = statistics.stdev(self._peer_returns) if len(self._peer_returns) > 1 else 0.0

        if peer_std == 0:
            return self._neutral_signal(symbol, timeframe)

        z = (asset_return - peer_mean) / peer_std
        direction = -z / 3.0
        direction = max(-1.0, min(1.0, direction))
        confidence = min(abs(z) / 3.0, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={"z_sector": round(z, 3), "asset_return": round(asset_return, 4)},
            features={"z_sector": z, "asset_return": asset_return},
        )
