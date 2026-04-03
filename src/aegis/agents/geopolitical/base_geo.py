"""Base class for geopolitical agents.

Extends BaseAgent with GeopoliticalProvider dependency.
Geo agents can VETO trades when risk > threshold.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aegis.agents.base import BaseAgent
from aegis.agents.geopolitical.providers import GeopoliticalProvider, NullGeopoliticalProvider
from aegis.common.types import AgentSignal

_VETO_THRESHOLD = 0.7


class BaseGeopoliticalAgent(BaseAgent):
    """Base for geopolitical agents. Injects GeopoliticalProvider."""

    def __init__(
        self,
        agent_id: str,
        config: dict[str, Any],
        provider: GeopoliticalProvider | None = None,
    ):
        super().__init__(agent_id, config)
        self._provider = provider or NullGeopoliticalProvider()
        self._veto_threshold = config.get("veto_threshold", _VETO_THRESHOLD)

    @property
    def agent_type(self) -> str:
        return "geopolitical"

    def _build_geo_signal(
        self,
        symbol: str,
        direction: float,
        confidence: float,
        risk_score: float,
        reasoning: dict,
    ) -> AgentSignal:
        veto = risk_score > self._veto_threshold
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            direction=direction,
            confidence=confidence,
            timeframe="1d",
            expected_holding_period="days",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasoning=reasoning,
            features_used={},
            metadata={"veto": veto, "risk_score": risk_score},
        )
