"""Crypto agents: Fear & Greed + Liquidation cascade signals."""

from aegis.agents.crypto.base_crypto import BaseCryptoAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("crypto", "fear_greed_crypto")
class CryptoFearGreedAgent(BaseCryptoAgent):
    """Contrarian Crypto Fear & Greed Index signal.

    Extreme Fear (<20) -> bullish (buy when others are fearful).
    Extreme Greed (>80) -> bearish (sell when others are greedy).
    """

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        if not self._is_crypto(symbol):
            return self._neutral_signal(symbol)

        fg = self._provider.get_fear_greed()
        if fg is None:
            return self._neutral_signal(symbol)

        # Center around 50, scale to [-1, 1], then invert (contrarian)
        raw = (fg - 50) / 50.0  # [-1, 1]
        direction = -raw  # Contrarian: greed -> sell, fear -> buy

        # Confidence higher at extremes
        confidence = abs(raw) * 0.8
        if fg < 20 or fg > 80:
            confidence = min(confidence + 0.2, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=max(-1.0, min(1.0, direction)),
            confidence=min(confidence, 1.0),
            timeframe="1d",
            reasoning={"fear_greed_index": fg, "contrarian_direction": round(direction, 3)},
            features={"fear_greed": fg},
        )


@register_agent("crypto", "liquidations")
class LiquidationsAgent(BaseCryptoAgent):
    """Exchange liquidation cascade detection.

    Large liquidation events indicate forced selling/buying.
    Uses liquidation volume as a volatility/regime signal.
    """

    _NORMAL_LIQUIDATIONS = 100e6  # $100M/day is normal

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        if not self._is_crypto(symbol):
            return self._neutral_signal(symbol)

        metrics = self._provider.get_metrics(symbol)
        if metrics is None:
            return self._neutral_signal(symbol)

        liqs = metrics.liquidations_24h
        ratio = liqs / self._NORMAL_LIQUIDATIONS

        if ratio < 1.5:
            return self._neutral_signal(symbol)

        # Large liquidations = high volatility, bearish bias (cascading stops)
        direction = -0.3 * min(ratio / 5.0, 1.0)
        confidence = min(ratio / 5.0, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe="1h",
            reasoning={
                "liquidations_24h": round(liqs / 1e6, 1),
                "ratio_to_normal": round(ratio, 2),
            },
            features={"liquidations": liqs},
        )
