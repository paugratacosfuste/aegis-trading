"""Combined sentiment agents: sent_07 through sent_10.

sent_07: equal-weight average of all sentiment sources
sent_08: sentiment velocity (rate of change)
sent_09: news volume anomaly detection
sent_10: sentiment divergence (news vs retail)
"""

from aegis.agents.registry import register_agent
from aegis.agents.sentiment.base_sentiment import BaseSentimentAgent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("sentiment", "combined")
class CombinedSentimentAgent(BaseSentimentAgent):
    """Equal-weight average of all available sentiment data."""

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        data = self._provider.get_sentiment(symbol)
        if data is None:
            return self._neutral_signal(symbol)

        direction = data.sentiment_score
        confidence = min(abs(direction) + 0.2, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=max(-1.0, min(1.0, direction)),
            confidence=confidence,
            timeframe="1h",
            reasoning={"combined_score": round(direction, 3)},
            features={"combined_score": direction},
        )


@register_agent("sentiment", "velocity")
class SentimentVelocityAgent(BaseSentimentAgent):
    """Rate of change of sentiment."""

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        data = self._provider.get_sentiment(symbol)
        if data is None:
            return self._neutral_signal(symbol)

        velocity = data.sentiment_velocity
        direction = velocity * 2.0  # Scale up
        confidence = min(abs(velocity) * 3, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=max(-1.0, min(1.0, direction)),
            confidence=min(confidence, 1.0),
            timeframe="1h",
            reasoning={"velocity": round(velocity, 3)},
            features={"velocity": velocity},
        )


@register_agent("sentiment", "news_volume")
class NewsVolumeAgent(BaseSentimentAgent):
    """Abnormal news volume detection."""

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        data = self._provider.get_sentiment(symbol)
        if data is None:
            return self._neutral_signal(symbol)

        mentions = data.mention_count
        # High mention count + sentiment direction = event-driven signal
        if mentions < 10:
            return self._neutral_signal(symbol)

        volume_signal = min(mentions / 50.0, 1.0)
        direction = data.sentiment_score * volume_signal
        confidence = volume_signal * 0.7

        return self._build_signal(
            symbol=symbol,
            direction=max(-1.0, min(1.0, direction)),
            confidence=min(confidence, 1.0),
            timeframe="1h",
            reasoning={"mentions": mentions, "sentiment": round(data.sentiment_score, 3)},
            features={"mentions": mentions, "sentiment": data.sentiment_score},
        )


@register_agent("sentiment", "divergence")
class SentimentDivergenceAgent(BaseSentimentAgent):
    """Divergence between news and retail sentiment.

    When news is bullish but retail is bearish (or vice versa),
    this signals a potential opportunity.
    """

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        data = self._provider.get_sentiment(symbol)
        if data is None:
            return self._neutral_signal(symbol)

        # Use sentiment_score as news, sentiment_velocity as proxy for retail shift
        news = data.sentiment_score
        retail_shift = data.sentiment_velocity

        # Divergence: news positive but retail turning negative, or vice versa
        if news > 0 and retail_shift < -0.1:
            direction = -0.5  # News bullish but retail fading -> contrarian sell
        elif news < 0 and retail_shift > 0.1:
            direction = 0.5  # News bearish but retail recovering -> contrarian buy
        else:
            direction = 0.0

        divergence = abs(news - retail_shift)
        confidence = min(divergence, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=max(-1.0, min(1.0, direction)),
            confidence=min(confidence, 1.0),
            timeframe="1h",
            reasoning={
                "news": round(news, 3), "retail_shift": round(retail_shift, 3),
                "divergence": round(divergence, 3),
            },
            features={"news": news, "retail_shift": retail_shift},
        )
