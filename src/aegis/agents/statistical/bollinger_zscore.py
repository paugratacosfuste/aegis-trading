"""Bollinger z-score agent: volatility-adjusted mean reversion.

Uses Bollinger Bands %B as a z-score proxy.
%B < 0 (below lower band) -> strong buy
%B > 1 (above upper band) -> strong sell
"""

import pandas as pd
from ta.volatility import BollingerBands

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("statistical", "bollinger_zscore")
class BollingerZScoreAgent(BaseAgent):

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._period = config.get("period", 20)
        self._std_dev = config.get("std_dev", 2.0)

    @property
    def agent_type(self) -> str:
        return "statistical"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        timeframe = candles[-1].timeframe if candles else "1h"
        if len(candles) < self._period + 1:
            return self._neutral_signal(symbol, timeframe)

        closes = pd.Series([c.close for c in candles])
        bb = BollingerBands(close=closes, window=self._period, window_dev=self._std_dev)
        pband = bb.bollinger_pband().iloc[-1]

        if pd.isna(pband):
            return self._neutral_signal(symbol, timeframe)

        # %B: 0 = at lower band, 1 = at upper band
        # Mean reversion: sell above upper band, buy below lower
        direction = -(pband - 0.5) * 2.0
        direction = max(-1.0, min(1.0, direction))

        # Confidence: higher at extremes
        confidence = min(abs(pband - 0.5) * 2.0, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={"pband": round(pband, 3)},
            features={"bollinger_pband": pband},
        )
