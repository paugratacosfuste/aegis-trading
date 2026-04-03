"""Tests for TechnicalIndicatorAgent."""

import pytest

from aegis.agents.technical.indicator import TechnicalIndicatorAgent


class TestTechnicalIndicatorAgent:
    def test_uptrend_bullish(self, sample_candles_uptrend):
        agent = TechnicalIndicatorAgent("tech_04", {"preset": "trend_following", "period_style": "fast"})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction > 0
        assert signal.confidence > 0

    def test_downtrend_bearish(self, sample_candles_downtrend):
        agent = TechnicalIndicatorAgent("tech_04", {"preset": "trend_following", "period_style": "fast"})
        signal = agent.generate_signal("BTC/USDT", sample_candles_downtrend)
        assert signal.direction < 0

    def test_insufficient_data_neutral(self, sample_candles_uptrend):
        agent = TechnicalIndicatorAgent("tech_01", {"preset": "momentum_fast"})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend[:5])
        assert signal.direction == 0.0
        assert signal.confidence == 0.0

    def test_direction_bounds(self, sample_candles_uptrend):
        agent = TechnicalIndicatorAgent("tech_01", {"preset": "momentum_fast", "period_style": "fast"})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert -1.0 <= signal.direction <= 1.0
        assert 0.0 <= signal.confidence <= 1.0

    def test_different_presets_different_signals(self, sample_candles_volatile):
        momentum_agent = TechnicalIndicatorAgent("tech_01", {"preset": "momentum_fast"})
        trend_agent = TechnicalIndicatorAgent("tech_04", {"preset": "trend_following"})
        sig_m = momentum_agent.generate_signal("BTC/USDT", sample_candles_volatile)
        sig_t = trend_agent.generate_signal("BTC/USDT", sample_candles_volatile)
        # Different presets should generally produce different signals
        # At minimum, the reasoning keys should differ
        assert set(sig_m.reasoning.keys()) != set(sig_t.reasoning.keys())

    def test_agent_type_is_technical(self):
        agent = TechnicalIndicatorAgent("tech_01", {})
        assert agent.agent_type == "technical"

    def test_flat_market_low_confidence(self, sample_candles_flat):
        agent = TechnicalIndicatorAgent("tech_03", {"preset": "full_suite"})
        signal = agent.generate_signal("BTC/USDT", sample_candles_flat)
        # Flat market -> low ADX -> lower confidence
        assert signal.confidence < 0.8

    def test_registered_in_registry(self):
        from aegis.agents.registry import get_agent_class
        cls = get_agent_class("technical", "indicator")
        assert cls is TechnicalIndicatorAgent
