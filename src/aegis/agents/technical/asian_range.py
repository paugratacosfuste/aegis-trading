"""Asian Range breakout/sweep strategy (tech_11, tech_12).

Session-based liquidity sweep strategy:
1. Identify Asian consolidation range (00:00-08:00 GMT)
2. Detect London sweep (false breakout past range)
3. Generate reversal signal after sweep confirmation
"""

from datetime import timezone

import pandas as pd
from ta.volatility import AverageTrueRange

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint

# Need at least a full day of 15m candles (96 candles)
MIN_CANDLES = 48

# Session hours (GMT)
ASIAN_START = 0
ASIAN_END = 8
LONDON_START = 7
LONDON_END = 10


@register_agent("technical", "asian_range")
class AsianRangeAgent(BaseAgent):
    """Session-based Asian Range sweep/breakout strategy."""

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._market = config.get("market", "crypto")
        # Tighter sweep thresholds for forex
        self._sweep_threshold = 0.05 if self._market == "forex" else 0.10

    @property
    def agent_type(self) -> str:
        return "technical"

    def generate_signal(
        self, symbol: str, candles: list[MarketDataPoint]
    ) -> AgentSignal:
        if len(candles) < MIN_CANDLES:
            return self._neutral_signal(symbol, "15m")

        # Ensure timestamps are UTC
        utc_candles = [c for c in candles if c.timestamp.tzinfo is not None]
        if len(utc_candles) < MIN_CANDLES:
            # Fall back to treating timestamps as UTC
            utc_candles = candles

        # Find the most recent Asian session
        asian_candles = [
            c for c in utc_candles[-96:]  # Last ~24h of 15m candles
            if ASIAN_START <= c.timestamp.hour < ASIAN_END
        ]
        if len(asian_candles) < 4:
            return self._neutral_signal(symbol, "15m")

        asian_high = max(c.high for c in asian_candles)
        asian_low = min(c.low for c in asian_candles)
        asian_range = asian_high - asian_low

        if asian_range <= 0:
            return self._neutral_signal(symbol, "15m")

        # ATR filter: range must be meaningful but not too wide
        closes = pd.Series([c.close for c in utc_candles])
        highs = pd.Series([c.high for c in utc_candles])
        lows = pd.Series([c.low for c in utc_candles])

        if len(closes) < 14:
            return self._neutral_signal(symbol, "15m")

        atr = AverageTrueRange(high=highs, low=lows, close=closes, window=14)
        atr_val = atr.average_true_range().iloc[-1]
        if pd.isna(atr_val) or atr_val <= 0:
            return self._neutral_signal(symbol, "15m")

        # Filter: range too tight or too wide
        if asian_range < 0.3 * atr_val:
            return self._neutral_signal(symbol, "15m")
        if asian_range > 2.0 * atr_val:
            return self._neutral_signal(symbol, "15m")

        # Find London session candles after Asian range
        london_candles = [
            c for c in utc_candles[-96:]
            if LONDON_START <= c.timestamp.hour < LONDON_END
            and c.timestamp > asian_candles[-1].timestamp
        ]
        if not london_candles:
            return self._neutral_signal(symbol, "15m")

        # Detect sweep
        sweep_threshold = self._sweep_threshold * asian_range
        high_swept = any(c.high > asian_high + sweep_threshold for c in london_candles)
        low_swept = any(c.low < asian_low - sweep_threshold for c in london_candles)

        # Check for re-entry (confirmation)
        last_candle = utc_candles[-1]
        price_in_range = asian_low <= last_candle.close <= asian_high

        if high_swept and low_swept:
            # Both sides swept = choppy, unclear
            return self._build_signal(
                symbol=symbol,
                direction=0.0,
                confidence=0.2,
                timeframe="15m",
                reasoning={"note": "both_sides_swept", "asian_high": asian_high,
                           "asian_low": asian_low},
                features={"asian_range": asian_range, "atr_14": atr_val},
            )

        if high_swept and price_in_range:
            # High swept + re-entered = short signal
            sweep_candle = max(
                (c for c in london_candles if c.high > asian_high + sweep_threshold),
                key=lambda c: c.high,
            )
            sweep_depth = sweep_candle.high - asian_high
            confidence = self._compute_confidence(
                sweep_depth, asian_range, atr_val, utc_candles
            )
            return self._build_signal(
                symbol=symbol,
                direction=-1.0,
                confidence=confidence,
                timeframe="15m",
                reasoning={
                    "sweep": "high_swept", "asian_high": round(asian_high, 2),
                    "asian_low": round(asian_low, 2),
                    "sweep_high": round(sweep_candle.high, 2),
                },
                features={
                    "asian_range": round(asian_range, 4),
                    "sweep_depth": round(sweep_depth, 4),
                    "atr_14": round(atr_val, 4),
                },
            )

        if low_swept and price_in_range:
            # Low swept + re-entered = long signal
            sweep_candle = min(
                (c for c in london_candles if c.low < asian_low - sweep_threshold),
                key=lambda c: c.low,
            )
            sweep_depth = asian_low - sweep_candle.low
            confidence = self._compute_confidence(
                sweep_depth, asian_range, atr_val, utc_candles
            )
            return self._build_signal(
                symbol=symbol,
                direction=1.0,
                confidence=confidence,
                timeframe="15m",
                reasoning={
                    "sweep": "low_swept", "asian_high": round(asian_high, 2),
                    "asian_low": round(asian_low, 2),
                    "sweep_low": round(sweep_candle.low, 2),
                },
                features={
                    "asian_range": round(asian_range, 4),
                    "sweep_depth": round(sweep_depth, 4),
                    "atr_14": round(atr_val, 4),
                },
            )

        return self._neutral_signal(symbol, "15m")

    def _compute_confidence(
        self,
        sweep_depth: float,
        asian_range: float,
        atr_val: float,
        candles: list[MarketDataPoint],
    ) -> float:
        """Confidence from sweep depth and volume."""
        base = 0.5
        if asian_range > 0 and sweep_depth > 0.3 * asian_range:
            base += 0.10  # Deep sweep = more liquidity taken
        # Volume check on recent candles
        volumes = [c.volume for c in candles[-10:]]
        if volumes:
            avg_vol = sum(volumes) / len(volumes)
            recent_vol = volumes[-1]
            if avg_vol > 0 and recent_vol > 1.5 * avg_vol:
                base += 0.15  # High volume confirms
        return min(base, 1.0)
