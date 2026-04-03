"""Kalman filter agent: dynamic fair value estimation.

Simple 1D Kalman filter tracking fair price.
Signal: current price vs Kalman estimate of fair value.
"""

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint

MIN_CANDLES = 10


@register_agent("statistical", "kalman")
class KalmanAgent(BaseAgent):

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._process_noise = config.get("process_noise", 1.0)
        self._measurement_noise = config.get("measurement_noise", 10.0)

    @property
    def agent_type(self) -> str:
        return "statistical"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        timeframe = candles[-1].timeframe if candles else "1h"
        if len(candles) < MIN_CANDLES:
            return self._neutral_signal(symbol, timeframe)

        # 1D Kalman filter
        q = self._process_noise  # Process noise
        r = self._measurement_noise  # Measurement noise
        x = candles[0].close  # Initial state estimate
        p = 1000.0  # Initial uncertainty

        for candle in candles:
            # Predict
            x_pred = x
            p_pred = p + q

            # Update
            k = p_pred / (p_pred + r)  # Kalman gain
            x = x_pred + k * (candle.close - x_pred)
            p = (1 - k) * p_pred

        current = candles[-1].close
        fair_value = x

        if fair_value == 0:
            return self._neutral_signal(symbol, timeframe)

        # Mean reversion: price above fair value -> sell, below -> buy
        deviation = (current - fair_value) / fair_value
        direction = -deviation * 10.0
        direction = max(-1.0, min(1.0, direction))

        # Confidence: lower filter uncertainty = more confident
        confidence = min(1.0 / (1.0 + p / r), 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={
                "fair_value": round(fair_value, 2),
                "current": round(current, 2),
                "deviation_pct": round(deviation * 100, 3),
                "kalman_gain": round(k, 4),
            },
            features={"fair_value": fair_value, "deviation": deviation, "uncertainty": p},
        )
