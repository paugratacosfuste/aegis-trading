"""Hurst-filtered z-score agent: only signals when Hurst < 0.5.

Computes Hurst exponent via rescaled range (R/S) method.
If Hurst < 0.5 (mean-reverting), applies z-score signal.
If Hurst >= 0.5 (trending/random walk), returns neutral.
"""

import statistics

import numpy as np

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint

MIN_CANDLES = 50


@register_agent("statistical", "hurst_zscore")
class HurstZScoreAgent(BaseAgent):

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._lookback = config.get("lookback", 50)

    @property
    def agent_type(self) -> str:
        return "statistical"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        timeframe = candles[-1].timeframe if candles else "1h"
        if len(candles) < max(self._lookback, MIN_CANDLES):
            return self._neutral_signal(symbol, timeframe)

        prices = [c.close for c in candles[-self._lookback:]]
        hurst = self._compute_hurst(prices)

        if hurst is None or hurst >= 0.5:
            return self._build_signal(
                symbol=symbol,
                direction=0.0,
                confidence=0.1,
                timeframe=timeframe,
                reasoning={"hurst": round(hurst, 3) if hurst else None, "filtered": True},
                features={"hurst": hurst or 0.5},
            )

        # Hurst < 0.5: mean reverting. Apply z-score.
        mean = statistics.mean(prices)
        std = statistics.stdev(prices) if len(prices) > 1 else 0.0
        if std == 0:
            return self._neutral_signal(symbol, timeframe)

        z = (prices[-1] - mean) / std
        direction = -z / 3.0
        direction = max(-1.0, min(1.0, direction))

        # Confidence boosted by how strongly mean-reverting (lower Hurst)
        z_conf = min(abs(z) / 3.0, 1.0)
        hurst_bonus = (0.5 - hurst) * 2  # 0.0 at H=0.5, 1.0 at H=0.0
        confidence = z_conf * (0.6 + 0.4 * hurst_bonus)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={
                "hurst": round(hurst, 3), "z_score": round(z, 3),
                "mean": round(mean, 2), "std": round(std, 2),
            },
            features={"hurst": hurst, "z_score": z},
        )

    @staticmethod
    def _compute_hurst(prices: list[float]) -> float | None:
        """Compute Hurst exponent via simplified R/S method."""
        n = len(prices)
        if n < 20:
            return None

        returns = [prices[i] / prices[i - 1] - 1 for i in range(1, n) if prices[i - 1] != 0]
        if len(returns) < 10:
            return None

        rs_values = []
        sizes = []
        for size in [int(n / 4), int(n / 2), n - 1]:
            if size < 5 or size > len(returns):
                continue
            chunk = returns[:size]
            mean_r = sum(chunk) / len(chunk)
            deviations = [r - mean_r for r in chunk]
            cumulative = []
            s = 0.0
            for d in deviations:
                s += d
                cumulative.append(s)
            r_range = max(cumulative) - min(cumulative)
            std_r = statistics.stdev(chunk) if len(chunk) > 1 else 0.0
            if std_r > 0:
                rs_values.append(r_range / std_r)
                sizes.append(size)

        if len(rs_values) < 2:
            return None

        # Linear regression of log(R/S) vs log(n)
        log_sizes = np.log(sizes)
        log_rs = np.log(rs_values)
        try:
            coeffs = np.polyfit(log_sizes, log_rs, 1)
            return float(coeffs[0])
        except (np.linalg.LinAlgError, ValueError):
            return None
