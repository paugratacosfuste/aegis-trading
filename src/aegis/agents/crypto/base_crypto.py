"""Base class for crypto-specific agents.

Crypto agents only activate for crypto symbols.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aegis.agents.base import BaseAgent
from aegis.agents.crypto.providers import CryptoMetricsProvider, NullCryptoMetricsProvider
from aegis.common.types import AgentSignal

_CRYPTO_SYMBOLS = ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC")


def _is_crypto_symbol(symbol: str) -> bool:
    base = symbol.split("/")[0].upper()
    return base in _CRYPTO_SYMBOLS or "USDT" in symbol.upper()


class BaseCryptoAgent(BaseAgent):
    """Base for crypto-specific agents. Injects CryptoMetricsProvider."""

    def __init__(
        self,
        agent_id: str,
        config: dict[str, Any],
        provider: CryptoMetricsProvider | None = None,
    ):
        super().__init__(agent_id, config)
        self._provider = provider or NullCryptoMetricsProvider()

    @property
    def agent_type(self) -> str:
        return "crypto"

    def _is_crypto(self, symbol: str) -> bool:
        return _is_crypto_symbol(symbol)
