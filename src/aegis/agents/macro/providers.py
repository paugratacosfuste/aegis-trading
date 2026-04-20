"""Macro data providers: protocol and concrete implementations.

Decouples macro agents from live data sources (FRED, Treasury, yfinance).
In backtest, use NullMacroProvider (returns None -> agents emit neutral).
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from aegis.common.types import MacroDataPoint


class MacroProvider(Protocol):
    def get_macro_snapshot(self) -> MacroDataPoint | None:
        """Return latest macro snapshot, or None if unavailable."""
        ...

    def get_yield_curve(self) -> dict[str, float] | None:
        """Return tenor->yield mapping (e.g. {'1M': 4.5, '2Y': 4.2, '10Y': 4.0})."""
        ...

    def get_vix_history(self, lookback_days: int = 30) -> list[float] | None:
        """Return recent VIX closing values."""
        ...


class NullMacroProvider:
    """Always returns None. Used in backtest without historical macro data."""

    def get_macro_snapshot(self) -> None:
        return None

    def get_yield_curve(self) -> None:
        return None

    def get_vix_history(self, lookback_days: int = 30) -> None:
        return None


class HistoricalMacroProvider:
    """Reads from preloaded MacroDataPoints."""

    def __init__(
        self,
        snapshots: list[MacroDataPoint] | None = None,
        yield_curve: dict[str, float] | None = None,
        vix_history: list[float] | None = None,
    ):
        self._snapshots = snapshots or []
        self._yield_curve = yield_curve
        self._vix_history = vix_history

    def get_macro_snapshot(self) -> MacroDataPoint | None:
        return self._snapshots[-1] if self._snapshots else None

    def get_yield_curve(self) -> dict[str, float] | None:
        return self._yield_curve

    def get_vix_history(self, lookback_days: int = 30) -> list[float] | None:
        return self._vix_history


class BacktestMacroProvider:
    """Time-series macro provider for backtest. Advances with the bar clock.

    Call advance_to(timestamp) each bar so get_macro_snapshot() returns
    the most recent data as of that timestamp (no look-ahead).
    """

    def __init__(self, snapshots: list[MacroDataPoint]):
        self._snapshots = sorted(snapshots, key=lambda s: s.timestamp)
        self._current_idx = 0

    def advance_to(self, timestamp: datetime) -> None:
        """Advance to the latest snapshot on or before timestamp."""
        while (
            self._current_idx < len(self._snapshots) - 1
            and self._snapshots[self._current_idx + 1].timestamp <= timestamp
        ):
            self._current_idx += 1

    def get_macro_snapshot(self) -> MacroDataPoint | None:
        if not self._snapshots:
            return None
        return self._snapshots[self._current_idx]

    def get_yield_curve(self) -> dict[str, float] | None:
        snap = self.get_macro_snapshot()
        if snap is None:
            return None
        return {"10Y": snap.yield_10y, "2Y": snap.yield_2y}

    def get_vix_history(self, lookback_days: int = 30) -> list[float] | None:
        if not self._snapshots:
            return None
        end = self._current_idx + 1
        start = max(0, end - lookback_days)
        return [s.vix for s in self._snapshots[start:end]]
