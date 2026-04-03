"""Geopolitical data providers: protocol and concrete implementations.

Decouples geopolitical agents from GDELT/Finnhub/RSS feeds.
"""

from __future__ import annotations

from typing import Protocol

from aegis.common.types import GeopoliticalEvent


class GeopoliticalProvider(Protocol):
    def get_recent_events(self, hours: int = 24) -> list[GeopoliticalEvent]:
        """Return recent geopolitical events within the lookback window."""
        ...

    def get_risk_score(self) -> float:
        """Return composite geopolitical risk score [0, 1]."""
        ...


class NullGeopoliticalProvider:
    """Always returns empty. Used in backtest without geopolitical data."""

    def get_recent_events(self, hours: int = 24) -> list[GeopoliticalEvent]:
        return []

    def get_risk_score(self) -> float:
        return 0.0


class HistoricalGeopoliticalProvider:
    """Reads from preloaded GeopoliticalEvents."""

    def __init__(self, events: list[GeopoliticalEvent] | None = None, risk_score: float = 0.0):
        self._events = events or []
        self._risk_score = risk_score

    def get_recent_events(self, hours: int = 24) -> list[GeopoliticalEvent]:
        return self._events

    def get_risk_score(self) -> float:
        return self._risk_score
