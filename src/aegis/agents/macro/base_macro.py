"""Base class for all macro agents.

Extends BaseAgent with MacroProvider dependency.
Macro agents output regime classification in metadata, not trade signals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aegis.agents.base import BaseAgent
from aegis.agents.macro.providers import MacroProvider, NullMacroProvider
from aegis.common.types import AgentSignal


class BaseMacroAgent(BaseAgent):
    """Base for macro agents. Injects a MacroProvider."""

    def __init__(
        self,
        agent_id: str,
        config: dict[str, Any],
        provider: MacroProvider | None = None,
    ):
        super().__init__(agent_id, config)
        self._provider = provider or NullMacroProvider()

    @property
    def agent_type(self) -> str:
        return "macro"

    def _build_macro_signal(
        self,
        symbol: str,
        regime: str,
        regime_confidence: float,
        sector_tilts: dict[str, float] | None = None,
        asset_class_tilts: dict[str, float] | None = None,
        reasoning: dict | None = None,
    ) -> AgentSignal:
        """Build a macro signal with regime metadata.

        Macro agents always have direction=0.0 — they modify weights, not vote.
        """
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            direction=0.0,
            confidence=0.0,
            timeframe="1d",
            expected_holding_period="days",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasoning=reasoning or {},
            features_used={},
            metadata={
                "regime": regime,
                "regime_confidence": regime_confidence,
                "sector_tilts": sector_tilts or {},
                "asset_class_tilts": asset_class_tilts or {},
            },
        )
