"""RSI-filtered momentum: vetoes momentum at RSI extremes.

Standard momentum signal but neutralized if RSI > threshold (for longs)
or RSI < (100-threshold) (for shorts).
"""

import pandas as pd
from ta.momentum import RSIIndicator

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("momentum", "rsi_filtered")
class RsiFilteredMomentumAgent(BaseAgent):

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._lookback = config.get("lookback", 20)
        self._rsi_threshold = config.get("rsi_threshold", 75)

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

        return_nd = (price_now - price_past) / price_past
        direction = return_nd * 10.0
        direction = max(-1.0, min(1.0, direction))

        # RSI filter
        closes = pd.Series([c.close for c in candles])
        rsi_series = RSIIndicator(close=closes, window=14).rsi()
        rsi = rsi_series.iloc[-1]

        if pd.isna(rsi):
            rsi = 50.0

        # Veto: long signal with overbought RSI
        if direction > 0 and rsi > self._rsi_threshold:
            return self._build_signal(
                symbol=symbol,
                direction=0.0,
                confidence=0.1,
                timeframe=timeframe,
                reasoning={"filtered": True, "rsi": round(rsi, 1), "return": round(return_nd, 4)},
                features={"rsi": rsi, "return": return_nd},
            )

        # Veto: short signal with oversold RSI
        if direction < 0 and rsi < (100 - self._rsi_threshold):
            return self._build_signal(
                symbol=symbol,
                direction=0.0,
                confidence=0.1,
                timeframe=timeframe,
                reasoning={"filtered": True, "rsi": round(rsi, 1), "return": round(return_nd, 4)},
                features={"rsi": rsi, "return": return_nd},
            )

        confidence = min(abs(return_nd) / 0.05, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={"rsi": round(rsi, 1), "return": round(return_nd, 4)},
            features={"rsi": rsi, "return": return_nd},
        )
