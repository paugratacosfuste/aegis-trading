"""Fear & Greed index agent: contrarian from extreme sentiment.

sent_05: equity (CNN Fear & Greed)
sent_06: crypto (Alternative.me)

Uses sentiment_score mapped from 0-100 scale to [-1, +1].
"""

from aegis.agents.registry import register_agent
from aegis.agents.sentiment.base_sentiment import BaseSentimentAgent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("sentiment", "fear_greed")
class FearGreedAgent(BaseSentimentAgent):

    def __init__(self, agent_id, config, provider=None):
        super().__init__(agent_id, config, provider)
        self._market = config.get("market", "equity")

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        data = self._provider.get_sentiment(symbol)
        if data is None:
            return self._neutral_signal(symbol)

        # sentiment_score is in [-1, +1], map to 0-100 for F&G interpretation
        # Or use directly: -1 = extreme fear, +1 = extreme greed
        fg_score = data.sentiment_score

        # Contrarian logic
        if fg_score > 0.6:
            direction = -(fg_score - 0.5)  # Greed -> sell
        elif fg_score < -0.6:
            direction = -(fg_score + 0.5)  # Fear -> buy
        else:
            direction = -fg_score * 0.3  # Mild contrarian

        confidence = abs(fg_score)

        return self._build_signal(
            symbol=symbol,
            direction=max(-1.0, min(1.0, direction)),
            confidence=min(confidence, 1.0),
            timeframe="1d",
            reasoning={
                "fear_greed": round(fg_score, 3), "market": self._market,
            },
            features={"fear_greed": fg_score},
        )
