"""Crypto metrics providers: protocol and concrete implementations.

Decouples crypto agents from CoinGecko/DefiLlama/Binance APIs.
"""

from __future__ import annotations

from typing import Protocol

from aegis.common.types import CryptoMetrics


class CryptoMetricsProvider(Protocol):
    def get_metrics(self, symbol: str) -> CryptoMetrics | None:
        """Return latest crypto metrics for a symbol."""
        ...

    def get_funding_rate(self, symbol: str) -> float | None:
        """Return current funding rate for a perpetual futures symbol."""
        ...

    def get_fear_greed(self) -> int | None:
        """Return current Crypto Fear & Greed Index (0-100)."""
        ...


class NullCryptoMetricsProvider:
    """Always returns None. Used in backtest without crypto metrics."""

    def get_metrics(self, symbol: str) -> None:
        return None

    def get_funding_rate(self, symbol: str) -> None:
        return None

    def get_fear_greed(self) -> None:
        return None


class HistoricalCryptoMetricsProvider:
    """Reads from preloaded CryptoMetrics."""

    def __init__(
        self,
        metrics: list[CryptoMetrics] | None = None,
        fear_greed: int | None = None,
    ):
        self._by_symbol: dict[str, CryptoMetrics] = {}
        for m in (metrics or []):
            self._by_symbol[m.symbol] = m
        self._fear_greed = fear_greed

    def get_metrics(self, symbol: str) -> CryptoMetrics | None:
        return self._by_symbol.get(symbol)

    def get_funding_rate(self, symbol: str) -> float | None:
        m = self._by_symbol.get(symbol)
        return m.funding_rate if m else None

    def get_fear_greed(self) -> int | None:
        return self._fear_greed
