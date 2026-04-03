"""Base agent abstract class. All agents implement this contract."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from aegis.common.types import AgentSignal, MarketDataPoint


class BaseAgent(ABC):
    """Abstract base for all signal-generating agents."""

    def __init__(self, agent_id: str, config: dict[str, Any]):
        self.agent_id = agent_id
        self.config = config

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the agent type string (e.g., 'technical', 'statistical')."""
        ...

    @abstractmethod
    def generate_signal(
        self, symbol: str, candles: list[MarketDataPoint]
    ) -> AgentSignal:
        """Generate a signal from recent candles."""
        ...

    def _neutral_signal(self, symbol: str, timeframe: str = "1h") -> AgentSignal:
        """Return a neutral (no opinion) signal."""
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            direction=0.0,
            confidence=0.0,
            timeframe=timeframe,
            expected_holding_period="hours",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasoning={"note": "insufficient data or neutral"},
            features_used={},
            metadata={},
        )

    def _build_signal(
        self,
        symbol: str,
        direction: float,
        confidence: float,
        timeframe: str,
        reasoning: dict,
        features: dict,
    ) -> AgentSignal:
        """Build a validated signal with clamped direction/confidence."""
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            expected_holding_period="hours",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasoning=reasoning,
            features_used=features,
            metadata={},
        )
