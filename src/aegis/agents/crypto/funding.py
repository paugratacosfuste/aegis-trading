"""Crypto agent: Funding rate reversal signals."""

from aegis.agents.crypto.base_crypto import BaseCryptoAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint

_FUNDING_THRESHOLD = 0.001  # 0.1% — typical neutral range


@register_agent("crypto", "funding_reversal")
class FundingReversalAgent(BaseCryptoAgent):
    """Contrarian funding rate signal.

    Positive funding > threshold -> crowded longs -> short signal.
    Negative funding < -threshold -> crowded shorts -> long signal.
    """

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        if not self._is_crypto(symbol):
            return self._neutral_signal(symbol)

        rate = self._provider.get_funding_rate(symbol)
        if rate is None:
            return self._neutral_signal(symbol)

        threshold = self.config.get("threshold", _FUNDING_THRESHOLD)

        if abs(rate) < threshold:
            return self._neutral_signal(symbol)

        # Contrarian: positive funding = short, negative = long
        direction = -rate * 10.0  # Scale up — typical rates are 0.01%
        confidence = min(abs(rate) * 20.0, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=max(-1.0, min(1.0, direction)),
            confidence=confidence,
            timeframe="1h",
            reasoning={"funding_rate": round(rate, 6), "contrarian_direction": round(direction, 3)},
            features={"funding_rate": rate},
        )
