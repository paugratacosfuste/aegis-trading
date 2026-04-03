"""Base class for fundamental agents.

Fundamental agents are FILTERS, not signal generators.
They output confidence_modifier and veto in metadata, direction is always 0.0.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aegis.agents.base import BaseAgent
from aegis.agents.fundamental.providers import FundamentalProvider, NullFundamentalProvider
from aegis.common.types import AgentSignal

# Ensemble modification thresholds
_BOOST_THRESHOLD = 0.7   # score > 0.7 -> boost 20%
_REDUCE_THRESHOLD = 0.3  # score < 0.3 -> reduce 30%
_VETO_THRESHOLD = 0.1    # score < 0.1 -> VETO


class BaseFundamentalAgent(BaseAgent):
    """Base for fundamental agents. Injects FundamentalProvider."""

    def __init__(
        self,
        agent_id: str,
        config: dict[str, Any],
        provider: FundamentalProvider | None = None,
    ):
        super().__init__(agent_id, config)
        self._provider = provider or NullFundamentalProvider()

    @property
    def agent_type(self) -> str:
        return "fundamental"

    def _build_fundamental_signal(
        self,
        symbol: str,
        quality_score: float,
        reasoning: dict,
    ) -> AgentSignal:
        """Build a fundamental signal with confidence modifier and veto logic.

        Direction is always 0.0 — fundamentals modify other signals, not vote.
        """
        if quality_score > _BOOST_THRESHOLD:
            modifier = 1.2
        elif quality_score < _VETO_THRESHOLD:
            modifier = 0.0
        elif quality_score < _REDUCE_THRESHOLD:
            modifier = 0.7
        else:
            modifier = 1.0

        veto = quality_score < _VETO_THRESHOLD

        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            direction=0.0,
            confidence=quality_score,
            timeframe="1d",
            expected_holding_period="weeks",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasoning=reasoning,
            features_used={},
            metadata={
                "quality_score": quality_score,
                "confidence_modifier": modifier,
                "veto": veto,
            },
        )
