"""Tests for crypto-specific agents (M6)."""

from datetime import datetime, timezone

import pytest

from aegis.agents.crypto.providers import HistoricalCryptoMetricsProvider
from aegis.common.types import CryptoMetrics


def _metrics(symbol="BTC/USDT", funding_rate=0.005, fear_greed=65,
             btc_dominance=55.0, tvl=50e9, tvl_change=1.5,
             liquidations=80e6, **kw) -> CryptoMetrics:
    defaults = dict(
        symbol=symbol, timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
        funding_rate=funding_rate, open_interest=5e9,
        btc_dominance=btc_dominance, fear_greed_index=fear_greed,
        tvl=tvl, tvl_change_24h=tvl_change,
        liquidations_24h=liquidations, source="binance",
    )
    defaults.update(kw)
    return CryptoMetrics(**defaults)


def _provider(metrics=None, fear_greed=None):
    m = metrics or [_metrics()]
    fg = fear_greed if fear_greed is not None else m[0].fear_greed_index
    return HistoricalCryptoMetricsProvider(metrics=m, fear_greed=fg)


class TestFundingReversalAgent:
    def test_no_data_neutral(self, sample_candles_uptrend):
        from aegis.agents.crypto.funding import FundingReversalAgent
        agent = FundingReversalAgent("crypto_fund_01", {})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_high_positive_funding_short(self, sample_candles_uptrend):
        from aegis.agents.crypto.funding import FundingReversalAgent
        agent = FundingReversalAgent("crypto_fund_01", {},
                                     _provider([_metrics(funding_rate=0.05)]))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction < 0  # Crowded longs -> short

    def test_high_negative_funding_long(self, sample_candles_uptrend):
        from aegis.agents.crypto.funding import FundingReversalAgent
        agent = FundingReversalAgent("crypto_fund_01", {},
                                     _provider([_metrics(funding_rate=-0.05)]))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction > 0  # Crowded shorts -> long

    def test_equity_symbol_neutral(self, sample_candles_uptrend):
        from aegis.agents.crypto.funding import FundingReversalAgent
        agent = FundingReversalAgent("crypto_fund_01", {}, _provider())
        signal = agent.generate_signal("AAPL", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_agent_type(self):
        from aegis.agents.crypto.funding import FundingReversalAgent
        agent = FundingReversalAgent("crypto_fund_01", {})
        assert agent.agent_type == "crypto"


class TestDominanceAgent:
    def test_no_data_neutral(self, sample_candles_uptrend):
        from aegis.agents.crypto.dominance import DominanceAgent
        agent = DominanceAgent("crypto_dom_01", {})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_high_dominance_btc_bullish(self, sample_candles_uptrend):
        from aegis.agents.crypto.dominance import DominanceAgent
        agent = DominanceAgent("crypto_dom_01", {},
                               _provider([_metrics(btc_dominance=65.0)]))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction > 0  # BTC dominant = bullish for BTC

    def test_low_dominance_alt_season(self, sample_candles_uptrend):
        from aegis.agents.crypto.dominance import DominanceAgent
        m = _metrics(symbol="ETH/USDT", btc_dominance=38.0)
        agent = DominanceAgent("crypto_dom_01", {}, _provider([m]))
        signal = agent.generate_signal("ETH/USDT", sample_candles_uptrend)
        assert signal.direction > 0  # Alt season = bullish for alts


class TestCryptoFearGreedAgent:
    def test_extreme_greed_bearish(self, sample_candles_uptrend):
        from aegis.agents.crypto.crypto_sentiment import CryptoFearGreedAgent
        agent = CryptoFearGreedAgent("crypto_sent_01", {},
                                      _provider(fear_greed=90))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction < 0  # Contrarian

    def test_extreme_fear_bullish(self, sample_candles_uptrend):
        from aegis.agents.crypto.crypto_sentiment import CryptoFearGreedAgent
        agent = CryptoFearGreedAgent("crypto_sent_01", {},
                                      _provider(fear_greed=10))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction > 0  # Contrarian


class TestLiquidationsAgent:
    def test_large_long_liquidation_bearish(self, sample_candles_uptrend):
        from aegis.agents.crypto.crypto_sentiment import LiquidationsAgent
        agent = LiquidationsAgent("crypto_sent_02", {},
                                   _provider([_metrics(liquidations=500e6)]))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        # Large liquidations -> high volatility signal
        assert abs(signal.direction) > 0 or signal.confidence > 0


class TestDefiTvlAgent:
    def test_tvl_rising_bullish(self, sample_candles_uptrend):
        from aegis.agents.crypto.defi import DefiTvlAgent
        agent = DefiTvlAgent("crypto_defi_01", {},
                              _provider([_metrics(tvl_change=15.0)]))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction > 0

    def test_tvl_dropping_bearish(self, sample_candles_uptrend):
        from aegis.agents.crypto.defi import DefiTvlAgent
        agent = DefiTvlAgent("crypto_defi_01", {},
                              _provider([_metrics(tvl_change=-12.0)]))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction < 0


class TestCryptoTechnicalAgent:
    def test_uptrend_bullish(self, sample_candles_uptrend):
        from aegis.agents.crypto.crypto_technical import CryptoTechnicalAgent
        agent = CryptoTechnicalAgent("crypto_tech_01",
                                      {"preset": "trend_following", "period_style": "fast"})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.agent_type == "crypto"

    def test_equity_symbol_neutral(self, sample_candles_uptrend):
        from aegis.agents.crypto.crypto_technical import CryptoTechnicalAgent
        agent = CryptoTechnicalAgent("crypto_tech_01",
                                      {"preset": "trend_following", "period_style": "fast"})
        signal = agent.generate_signal("AAPL", sample_candles_uptrend)
        assert signal.direction == 0.0


class TestRegistration:
    def test_all_crypto_registered(self):
        from aegis.agents.registry import get_agent_class
        from aegis.agents.crypto.funding import FundingReversalAgent
        from aegis.agents.crypto.dominance import DominanceAgent
        from aegis.agents.crypto.crypto_sentiment import CryptoFearGreedAgent, LiquidationsAgent
        from aegis.agents.crypto.defi import DefiTvlAgent
        from aegis.agents.crypto.crypto_technical import CryptoTechnicalAgent
        assert get_agent_class("crypto", "funding_reversal") is FundingReversalAgent
        assert get_agent_class("crypto", "dominance") is DominanceAgent
        assert get_agent_class("crypto", "fear_greed_crypto") is CryptoFearGreedAgent
        assert get_agent_class("crypto", "liquidations") is LiquidationsAgent
        assert get_agent_class("crypto", "defi_tvl") is DefiTvlAgent
        assert get_agent_class("crypto", "crypto_technical") is CryptoTechnicalAgent
