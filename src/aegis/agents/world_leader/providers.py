"""World leader data providers: protocol and concrete implementations.

Decouples leader agents from RSS feeds and social media scrapers.
"""

from __future__ import annotations

from typing import Protocol


class LeaderProvider(Protocol):
    def get_recent_statements(self, hours: int = 48) -> list[dict]:
        """Return recent classified statements.

        Each dict has: leader, text, source, timestamp, statement_type, sentiment_score.
        """
        ...


class NullLeaderProvider:
    """Always returns empty. Used in backtest without leader data."""

    def get_recent_statements(self, hours: int = 48) -> list[dict]:
        return []


class HistoricalLeaderProvider:
    """Reads from preloaded statement dicts."""

    def __init__(self, statements: list[dict] | None = None):
        self._statements = statements or []

    def get_recent_statements(self, hours: int = 48) -> list[dict]:
        return self._statements
