"""Fundamental data providers: protocol and concrete implementations.

Decouples fundamental agents from Yahoo Finance / Financial Modeling Prep.
"""

from __future__ import annotations

from typing import Protocol

from aegis.common.types import FundamentalScore


class FundamentalProvider(Protocol):
    def get_fundamentals(self, symbol: str) -> FundamentalScore | None:
        """Return latest fundamental score for a symbol."""
        ...

    def get_sector_fundamentals(self, sector: str) -> list[FundamentalScore]:
        """Return all fundamental scores for a given sector."""
        ...


class NullFundamentalProvider:
    """Always returns None. Used in backtest without fundamental data."""

    def get_fundamentals(self, symbol: str) -> None:
        return None

    def get_sector_fundamentals(self, sector: str) -> list[FundamentalScore]:
        return []


class HistoricalFundamentalProvider:
    """Reads from preloaded FundamentalScores."""

    def __init__(self, scores: list[FundamentalScore] | None = None):
        self._by_symbol: dict[str, FundamentalScore] = {}
        self._by_sector: dict[str, list[FundamentalScore]] = {}
        for s in (scores or []):
            self._by_symbol[s.symbol] = s
            self._by_sector.setdefault(s.sector, []).append(s)

    def get_fundamentals(self, symbol: str) -> FundamentalScore | None:
        return self._by_symbol.get(symbol)

    def get_sector_fundamentals(self, sector: str) -> list[FundamentalScore]:
        return self._by_sector.get(sector, [])
