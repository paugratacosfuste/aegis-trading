"""Base class for world leader agents.

Extends BaseAgent with LeaderProvider dependency.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aegis.agents.base import BaseAgent
from aegis.agents.world_leader.providers import LeaderProvider, NullLeaderProvider
from aegis.common.types import AgentSignal

# Half-life in hours by statement type
HALF_LIVES: dict[str, int] = {
    "social_media": 4,
    "official_statement": 48,
    "executive_order": 336,  # 2 weeks
    "policy_change": 720,    # 1 month
    "monetary_policy": 72,   # 3 days
    "trade": 48,
    "fiscal": 168,           # 1 week
    "regulatory": 168,
}


class BaseLeaderAgent(BaseAgent):
    """Base for world leader agents. Injects LeaderProvider."""

    def __init__(
        self,
        agent_id: str,
        config: dict[str, Any],
        provider: LeaderProvider | None = None,
    ):
        super().__init__(agent_id, config)
        self._provider = provider or NullLeaderProvider()

    @property
    def agent_type(self) -> str:
        return "world_leader"

    def _get_half_life(self, statement_type: str) -> int:
        return HALF_LIVES.get(statement_type, 12)

    def _build_leader_signal(
        self,
        symbol: str,
        direction: float,
        confidence: float,
        leader: str,
        statement_type: str,
        reasoning: dict,
    ) -> AgentSignal:
        half_life = self._get_half_life(statement_type)
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            direction=direction,
            confidence=confidence,
            timeframe="1h",
            expected_holding_period="hours",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasoning=reasoning,
            features_used={},
            metadata={
                "leader": leader,
                "statement_type": statement_type,
                "half_life_hours": half_life,
            },
        )
