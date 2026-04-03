"""Crypto agent: BTC dominance trends and rotation signals."""

from aegis.agents.crypto.base_crypto import BaseCryptoAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint

_HIGH_DOMINANCE = 60.0  # BTC dominant
_LOW_DOMINANCE = 40.0   # Alt season


@register_agent("crypto", "dominance")
class DominanceAgent(BaseCryptoAgent):
    """BTC dominance trend detection.

    High dominance -> bullish BTC, bearish alts.
    Low dominance -> alt season, bullish alts.
    """

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        if not self._is_crypto(symbol):
            return self._neutral_signal(symbol)

        metrics = self._provider.get_metrics(symbol)
        if metrics is None:
            return self._neutral_signal(symbol)

        dominance = metrics.btc_dominance
        is_btc = "BTC" in symbol.upper()

        if dominance > _HIGH_DOMINANCE:
            # BTC dominance high -> bullish BTC, bearish alts
            direction = 0.5 if is_btc else -0.3
        elif dominance < _LOW_DOMINANCE:
            # Alt season -> bearish BTC, bullish alts
            direction = -0.3 if is_btc else 0.5
        else:
            # Neutral zone
            direction = 0.0

        confidence = abs(dominance - 50.0) / 50.0  # Further from 50 = higher confidence

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=min(confidence, 1.0),
            timeframe="1d",
            reasoning={"btc_dominance": round(dominance, 2), "is_btc": is_btc},
            features={"btc_dominance": dominance},
        )
