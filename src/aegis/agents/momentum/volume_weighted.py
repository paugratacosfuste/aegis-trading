"""Volume-weighted momentum: only counts moves with above-average volume."""

import statistics

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("momentum", "volume_weighted")
class VolumeWeightedMomentumAgent(BaseAgent):

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._lookback = config.get("lookback", 20)

    @property
    def agent_type(self) -> str:
        return "momentum"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        timeframe = candles[-1].timeframe if candles else "1h"

        if len(candles) < self._lookback + 1:
            return self._neutral_signal(symbol, timeframe)

        window = candles[-(self._lookback + 1):]
        volumes = [c.volume for c in window]
        avg_volume = statistics.mean(volumes) if volumes else 0.0

        if avg_volume == 0:
            return self._neutral_signal(symbol, timeframe)

        # Only count returns on candles with above-average volume
        weighted_return = 0.0
        count = 0
        for i in range(1, len(window)):
            if window[i].volume >= avg_volume and window[i - 1].close != 0:
                ret = (window[i].close - window[i - 1].close) / window[i - 1].close
                weighted_return += ret
                count += 1

        if count == 0:
            return self._neutral_signal(symbol, timeframe)

        avg_return = weighted_return / count
        direction = avg_return * 100  # Scale up
        direction = max(-1.0, min(1.0, direction))
        confidence = min(abs(avg_return) / 0.003, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={
                "vol_weighted_return": round(avg_return, 6),
                "high_vol_candles": count,
            },
            features={"vol_weighted_return": avg_return, "high_vol_candles": count},
        )
