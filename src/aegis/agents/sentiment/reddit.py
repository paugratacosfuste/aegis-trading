"""Reddit sentiment agent: retail buzz detection.

sent_03: directional (high buzz + positive = bullish)
sent_04: contrarian (extreme buzz = top signal)
"""

from aegis.agents.registry import register_agent
from aegis.agents.sentiment.base_sentiment import BaseSentimentAgent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("sentiment", "reddit")
class RedditSentimentAgent(BaseSentimentAgent):

    def __init__(self, agent_id, config, provider=None):
        super().__init__(agent_id, config, provider)
        self._mode = config.get("mode", "directional")

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        data = self._provider.get_sentiment(symbol)
        if data is None:
            return self._neutral_signal(symbol)

        score = data.sentiment_score
        mentions = data.mention_count
        # Normalize mentions: 100+ mentions = max buzz
        buzz = min(mentions / 100.0, 1.0)

        if self._mode == "contrarian":
            # Extreme buzz = contrarian signal
            if buzz > 0.8 and score > 0.3:
                direction = -0.7  # Retail euphoria = top
            elif buzz > 0.8 and score < -0.3:
                direction = 0.5  # Retail panic = bottom
            else:
                direction = score * 0.3
        else:
            # Directional: sentiment * buzz amplifier
            direction = score * (0.5 + 0.5 * buzz)

        confidence = buzz * 0.5 + abs(score) * 0.5

        return self._build_signal(
            symbol=symbol,
            direction=max(-1.0, min(1.0, direction)),
            confidence=min(confidence, 1.0),
            timeframe="1h",
            reasoning={
                "sentiment": round(score, 3), "mentions": mentions,
                "buzz": round(buzz, 3), "mode": self._mode,
            },
            features={"sentiment": score, "mentions": mentions, "buzz": buzz},
        )
