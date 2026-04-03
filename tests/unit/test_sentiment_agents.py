"""Tests for sentiment agents (M5)."""

from datetime import datetime, timezone

import pytest

from aegis.agents.sentiment.combined import (
    CombinedSentimentAgent,
    NewsVolumeAgent,
    SentimentDivergenceAgent,
    SentimentVelocityAgent,
)
from aegis.agents.sentiment.fear_greed import FearGreedAgent
from aegis.agents.sentiment.news import NewsSentimentAgent
from aegis.agents.sentiment.providers import (
    HistoricalSentimentProvider,
    NullSentimentProvider,
)
from aegis.agents.sentiment.reddit import RedditSentimentAgent
from aegis.common.types import SentimentDataPoint


def _make_provider(score: float = 0.5, mentions: int = 50, velocity: float = 0.1):
    data = SentimentDataPoint(
        symbol="BTC/USDT",
        timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        source="test",
        sentiment_score=score,
        mention_count=mentions,
        sentiment_velocity=velocity,
    )
    return HistoricalSentimentProvider([data])


class TestNullProvider:
    def test_returns_none(self):
        provider = NullSentimentProvider()
        assert provider.get_sentiment("BTC/USDT") is None


class TestNewsSentimentAgent:
    def test_no_data_neutral(self, sample_candles_uptrend):
        agent = NewsSentimentAgent("sent_01", {"mode": "directional"})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_directional_positive(self, sample_candles_uptrend):
        provider = _make_provider(score=0.8)
        agent = NewsSentimentAgent("sent_01", {"mode": "directional"}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction > 0

    def test_contrarian_extreme_positive(self, sample_candles_uptrend):
        provider = _make_provider(score=0.9)
        agent = NewsSentimentAgent("sent_02", {"mode": "contrarian"}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction < 0  # Contrarian: too bullish -> bearish

    def test_agent_type(self):
        agent = NewsSentimentAgent("sent_01", {})
        assert agent.agent_type == "sentiment"


class TestRedditSentimentAgent:
    def test_high_buzz_bullish(self, sample_candles_uptrend):
        provider = _make_provider(score=0.6, mentions=200)
        agent = RedditSentimentAgent("sent_03", {"mode": "directional"}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction > 0

    def test_contrarian_euphoria(self, sample_candles_uptrend):
        provider = _make_provider(score=0.8, mentions=150)
        agent = RedditSentimentAgent("sent_04", {"mode": "contrarian"}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction < 0  # Retail euphoria = sell


class TestFearGreedAgent:
    def test_extreme_greed_sell(self, sample_candles_uptrend):
        provider = _make_provider(score=0.9)
        agent = FearGreedAgent("sent_05", {"market": "equity"}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction < 0

    def test_extreme_fear_buy(self, sample_candles_uptrend):
        provider = _make_provider(score=-0.9)
        agent = FearGreedAgent("sent_06", {"market": "crypto"}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction > 0


class TestCombinedAgents:
    def test_combined_positive(self, sample_candles_uptrend):
        provider = _make_provider(score=0.6)
        agent = CombinedSentimentAgent("sent_07", {}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction > 0

    def test_velocity(self, sample_candles_uptrend):
        provider = _make_provider(velocity=0.3)
        agent = SentimentVelocityAgent("sent_08", {}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction > 0

    def test_news_volume_low_mentions_neutral(self, sample_candles_uptrend):
        provider = _make_provider(mentions=3)
        agent = NewsVolumeAgent("sent_09", {}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_divergence_bullish_news_fading_retail(self, sample_candles_uptrend):
        provider = _make_provider(score=0.7, velocity=-0.3)
        agent = SentimentDivergenceAgent("sent_10", {}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction < 0  # Contrarian sell


class TestRegistration:
    def test_all_registered(self):
        from aegis.agents.registry import get_agent_class
        assert get_agent_class("sentiment", "news") is NewsSentimentAgent
        assert get_agent_class("sentiment", "reddit") is RedditSentimentAgent
        assert get_agent_class("sentiment", "fear_greed") is FearGreedAgent
        assert get_agent_class("sentiment", "combined") is CombinedSentimentAgent
        assert get_agent_class("sentiment", "velocity") is SentimentVelocityAgent
        assert get_agent_class("sentiment", "news_volume") is NewsVolumeAgent
        assert get_agent_class("sentiment", "divergence") is SentimentDivergenceAgent
