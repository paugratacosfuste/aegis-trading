"""Geopolitical data providers: protocol and concrete implementations.

Decouples geopolitical agents from GDELT/Finnhub/RSS feeds.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
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


# Default window used when the caller does not specify one.
_DEFAULT_RISK_WINDOW_HOURS = 168  # one week


class BacktestGeopoliticalProvider:
    """Time-series geopolitical provider for backtest. Advances with the bar
    clock. Exposes only events with ``timestamp <= clock`` (no look-ahead)
    and derives a rolling risk score that decays each event by its
    ``half_life_hours`` attribute.
    """

    def __init__(self, events: list[GeopoliticalEvent]):
        self._events = sorted(events, key=lambda e: e.timestamp)
        # Start the clock at the minimum datetime so nothing is visible until
        # the first ``advance_to`` call.
        self._clock: datetime | None = None

    def advance_to(self, timestamp: datetime) -> None:
        self._clock = timestamp

    def _visible_events(self) -> list[GeopoliticalEvent]:
        if self._clock is None:
            return []
        return [e for e in self._events if e.timestamp <= self._clock]

    def get_recent_events(
        self, hours: int = _DEFAULT_RISK_WINDOW_HOURS
    ) -> list[GeopoliticalEvent]:
        if self._clock is None:
            return []
        window_start = self._clock - timedelta(hours=hours)
        return [
            e
            for e in self._events
            if window_start <= e.timestamp <= self._clock
        ]

    def get_risk_score(self) -> float:
        """Composite geopolitical risk in [0, 1].

        Each visible event contributes ``severity * 2^(-age / half_life)``;
        the sum is squashed to [0, 1] via ``1 - exp(-x)`` so any single
        extreme event cannot saturate the score but a cluster of moderate
        events still dominates a single stale one.
        """
        events = self._visible_events()
        if not events or self._clock is None:
            return 0.0

        total = 0.0
        for event in events:
            age_hours = (self._clock - event.timestamp).total_seconds() / 3600.0
            half_life = max(1, int(event.half_life_hours))
            decay = 0.5 ** (age_hours / half_life)
            total += max(0.0, event.severity) * decay

        # Squash to [0, 1]; 1 - exp(-total) saturates smoothly.
        score = 1.0 - math.exp(-total)
        return max(0.0, min(1.0, score))
