"""Sentiment data providers: protocol and concrete implementations.

Decouples sentiment agents from live data sources.
In backtest, use NullSentimentProvider (returns None -> agents emit neutral).
"""

from __future__ import annotations

from typing import Protocol

from aegis.common.types import SentimentDataPoint


class SentimentProvider(Protocol):
    def get_sentiment(self, symbol: str) -> SentimentDataPoint | None:
        """Return latest sentiment data, or None if unavailable."""
        ...


class NullSentimentProvider:
    """Always returns None. Used in backtest when no historical sentiment."""

    def get_sentiment(self, symbol: str) -> None:
        return None


class HistoricalSentimentProvider:
    """Reads from a preloaded list of SentimentDataPoints."""

    def __init__(self, data: list[SentimentDataPoint]):
        self._by_symbol: dict[str, list[SentimentDataPoint]] = {}
        for d in data:
            self._by_symbol.setdefault(d.symbol, []).append(d)

    def get_sentiment(self, symbol: str) -> SentimentDataPoint | None:
        points = self._by_symbol.get(symbol, [])
        return points[-1] if points else None
