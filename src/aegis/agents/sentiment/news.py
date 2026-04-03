"""News sentiment agent: directional or contrarian from news sentiment.

sent_01: directional (positive news = bullish)
sent_02: contrarian (extreme positive = bearish)
"""

from aegis.agents.registry import register_agent
from aegis.agents.sentiment.base_sentiment import BaseSentimentAgent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("sentiment", "news")
class NewsSentimentAgent(BaseSentimentAgent):

    def __init__(self, agent_id, config, provider=None):
        super().__init__(agent_id, config, provider)
        self._mode = config.get("mode", "directional")

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        data = self._provider.get_sentiment(symbol)
        if data is None:
            return self._neutral_signal(symbol)

        score = data.sentiment_score  # [-1, +1]

        if self._mode == "contrarian":
            if score > 0.7:
                direction = -0.5
            elif score < -0.7:
                direction = 0.5
            else:
                direction = score * 0.5
        else:
            direction = score

        confidence = abs(score)

        return self._build_signal(
            symbol=symbol,
            direction=max(-1.0, min(1.0, direction)),
            confidence=min(confidence, 1.0),
            timeframe="1h",
            reasoning={"sentiment_score": round(score, 3), "mode": self._mode},
            features={"sentiment_score": score},
        )
