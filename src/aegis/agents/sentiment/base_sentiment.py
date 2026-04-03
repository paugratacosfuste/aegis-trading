"""Base class for all sentiment agents.

Extends BaseAgent with a SentimentProvider dependency.
"""

from __future__ import annotations

from typing import Any

from aegis.agents.base import BaseAgent
from aegis.agents.sentiment.providers import NullSentimentProvider, SentimentProvider


class BaseSentimentAgent(BaseAgent):
    """Base for sentiment agents. Injects a SentimentProvider."""

    def __init__(
        self,
        agent_id: str,
        config: dict[str, Any],
        provider: SentimentProvider | None = None,
    ):
        super().__init__(agent_id, config)
        self._provider = provider or NullSentimentProvider()

    @property
    def agent_type(self) -> str:
        return "sentiment"
