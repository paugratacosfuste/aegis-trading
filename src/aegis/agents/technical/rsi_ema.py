"""Technical agent: RSI(14) + EMA(9,21) signal generation.

Simplified version of tech_03 from 02-AGENT-ARCHITECTURE.md.
Uses the `ta` library (pure Python, no ta-lib C dependency).
"""

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint

MIN_CANDLES = 21  # EMA(21) needs at least 21 data points


@register_agent("technical", "rsi_ema")
class RsiEmaAgent(BaseAgent):
    """Generates signals based on RSI(14) and EMA(9) vs EMA(21) crossover."""

    @property
    def agent_type(self) -> str:
        return "technical"

    def generate_signal(
        self, symbol: str, candles: list[MarketDataPoint]
    ) -> AgentSignal:
        if len(candles) < MIN_CANDLES:
            return self._neutral_signal(symbol, candles[0].timeframe if candles else "1h")

        timeframe = candles[-1].timeframe
        closes = pd.Series([c.close for c in candles])

        # RSI(14)
        rsi_indicator = RSIIndicator(close=closes, window=14)
        rsi_series = rsi_indicator.rsi()
        rsi = rsi_series.iloc[-1]

        if pd.isna(rsi):
            return self._neutral_signal(symbol, timeframe)

        # EMA(9) and EMA(21)
        ema9 = EMAIndicator(close=closes, window=9).ema_indicator().iloc[-1]
        ema21 = EMAIndicator(close=closes, window=21).ema_indicator().iloc[-1]

        if pd.isna(ema9) or pd.isna(ema21):
            return self._neutral_signal(symbol, timeframe)

        # RSI signal: RSI < 30 -> +1.0, RSI > 70 -> -1.0, else linear scale
        if rsi < 30:
            rsi_signal = 1.0
        elif rsi > 70:
            rsi_signal = -1.0
        else:
            rsi_signal = (50 - rsi) / 20.0

        # EMA signal: EMA9 > EMA21 -> bullish, else bearish
        ema_spread = (ema9 - ema21) / ema21 if ema21 != 0 else 0.0
        ema_signal = max(-1.0, min(1.0, ema_spread * 100))  # Scale spread

        # Combined direction: RSI 40%, EMA 60%
        direction = rsi_signal * 0.4 + ema_signal * 0.6

        # Confidence: based on RSI extremity and EMA spread magnitude
        rsi_confidence = abs(rsi - 50) / 50.0
        ema_confidence = min(abs(ema_spread) * 50, 1.0)
        confidence = rsi_confidence * 0.4 + ema_confidence * 0.6

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={
                "rsi_14": round(rsi, 2),
                "rsi_signal": round(rsi_signal, 3),
                "ema_9": round(ema9, 2),
                "ema_21": round(ema21, 2),
                "ema_signal": round(ema_signal, 3),
            },
            features={
                "rsi_14": round(rsi, 4),
                "ema_9": round(ema9, 4),
                "ema_21": round(ema21, 4),
                "ema_spread": round(ema_spread, 6),
            },
        )
