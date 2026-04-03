"""Multi-window consensus agent: z-scores at 20d, 50d, 100d must agree.

Only generates a signal when all three windows agree on direction.
Disagreement returns neutral.
"""

import statistics

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("statistical", "multi_window")
class MultiWindowAgent(BaseAgent):

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self._windows = config.get("windows", [20, 50, 100])

    @property
    def agent_type(self) -> str:
        return "statistical"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        timeframe = candles[-1].timeframe if candles else "1h"
        max_window = max(self._windows)
        if len(candles) < max_window:
            return self._neutral_signal(symbol, timeframe)

        zscores = {}
        for w in self._windows:
            closes = [c.close for c in candles[-w:]]
            mean = statistics.mean(closes)
            std = statistics.stdev(closes) if len(closes) > 1 else 0.0
            if std == 0:
                return self._neutral_signal(symbol, timeframe)
            zscores[w] = (closes[-1] - mean) / std

        # Check if all z-scores agree on direction
        signs = [1 if z > 0 else -1 for z in zscores.values()]
        all_agree = all(s == signs[0] for s in signs)

        if not all_agree:
            return self._build_signal(
                symbol=symbol,
                direction=0.0,
                confidence=0.1,
                timeframe=timeframe,
                reasoning={f"z_{w}d": round(z, 3) for w, z in zscores.items()},
                features={f"z_{w}d": z for w, z in zscores.items()},
            )

        # All agree: use average z-score, boost confidence
        avg_z = sum(zscores.values()) / len(zscores)
        direction = -avg_z / 3.0
        direction = max(-1.0, min(1.0, direction))
        confidence = min(abs(avg_z) / 2.5, 1.0) * 1.1  # Agreement bonus
        confidence = min(confidence, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            reasoning={f"z_{w}d": round(z, 3) for w, z in zscores.items()},
            features={f"z_{w}d": z for w, z in zscores.items()},
        )
