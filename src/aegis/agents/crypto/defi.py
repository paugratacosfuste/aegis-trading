"""Crypto agent: DeFi TVL trend detection."""

from aegis.agents.crypto.base_crypto import BaseCryptoAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint

_TVL_CHANGE_THRESHOLD = 3.0  # 3% change needed for signal


@register_agent("crypto", "defi_tvl")
class DefiTvlAgent(BaseCryptoAgent):
    """DeFi Total Value Locked trend agent.

    TVL rising sharply -> bullish for DeFi tokens.
    TVL dropping sharply -> bearish / risk-off signal.
    """

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        if not self._is_crypto(symbol):
            return self._neutral_signal(symbol)

        metrics = self._provider.get_metrics(symbol)
        if metrics is None:
            return self._neutral_signal(symbol)

        change = metrics.tvl_change_24h

        if abs(change) < _TVL_CHANGE_THRESHOLD:
            return self._neutral_signal(symbol)

        # Direction proportional to TVL change, capped
        direction = change / 20.0  # 20% change = full signal
        confidence = min(abs(change) / 15.0, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=max(-1.0, min(1.0, direction)),
            confidence=confidence,
            timeframe="1d",
            reasoning={
                "tvl": round(metrics.tvl / 1e9, 2),
                "tvl_change_24h": round(change, 2),
            },
            features={"tvl_change": change},
        )
