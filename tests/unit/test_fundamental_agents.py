"""Tests for fundamental agents (M5)."""

from datetime import datetime, timezone

import pytest

from aegis.agents.fundamental.providers import HistoricalFundamentalProvider
from aegis.common.types import FundamentalScore


def _score(symbol="AAPL", sector="tech", quality=0.8, value=0.6, growth=0.9,
           market_cap_tier="large", **kw) -> FundamentalScore:
    defaults = dict(
        symbol=symbol, timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
        sector=sector, market_cap_tier=market_cap_tier,
        quality_score=quality, value_score=value, growth_score=growth,
        pe_zscore=-0.3, revenue_growth=0.12, source="yahoo",
    )
    defaults.update(kw)
    return FundamentalScore(**defaults)


def _provider(*scores):
    return HistoricalFundamentalProvider(scores=list(scores))


class TestBaseFundamentalAgent:
    def test_agent_type(self):
        from aegis.agents.fundamental.sector import SectorFundamentalAgent
        agent = SectorFundamentalAgent("fund_01", {"sector": "tech"})
        assert agent.agent_type == "fundamental"

    def test_direction_always_zero(self, sample_candles_uptrend):
        from aegis.agents.fundamental.sector import SectorFundamentalAgent
        agent = SectorFundamentalAgent("fund_01", {"sector": "tech"},
                                       _provider(_score()))
        signal = agent.generate_signal("AAPL", sample_candles_uptrend)
        assert signal.direction == 0.0


class TestSectorFundamentalAgent:
    def test_no_data_neutral(self, sample_candles_uptrend):
        from aegis.agents.fundamental.sector import SectorFundamentalAgent
        agent = SectorFundamentalAgent("fund_01", {"sector": "tech"})
        signal = agent.generate_signal("AAPL", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_high_quality_boost(self, sample_candles_uptrend):
        from aegis.agents.fundamental.sector import SectorFundamentalAgent
        agent = SectorFundamentalAgent("fund_01", {"sector": "tech"},
                                       _provider(_score(quality=0.85)))
        signal = agent.generate_signal("AAPL", sample_candles_uptrend)
        assert signal.metadata.get("confidence_modifier", 1.0) > 1.0

    def test_low_quality_reduce(self, sample_candles_uptrend):
        from aegis.agents.fundamental.sector import SectorFundamentalAgent
        agent = SectorFundamentalAgent("fund_01", {"sector": "tech"},
                                       _provider(_score(quality=0.1, value=0.1, growth=0.1)))
        signal = agent.generate_signal("AAPL", sample_candles_uptrend)
        assert signal.metadata.get("confidence_modifier", 1.0) < 1.0

    def test_very_low_quality_veto(self, sample_candles_uptrend):
        from aegis.agents.fundamental.sector import SectorFundamentalAgent
        agent = SectorFundamentalAgent("fund_01", {"sector": "tech"},
                                       _provider(_score(quality=0.05, value=0.05, growth=0.05)))
        signal = agent.generate_signal("AAPL", sample_candles_uptrend)
        assert signal.metadata.get("veto") is True

    def test_wrong_sector_neutral(self, sample_candles_uptrend):
        from aegis.agents.fundamental.sector import SectorFundamentalAgent
        # Agent is for healthcare, but stock is in tech
        agent = SectorFundamentalAgent("fund_02", {"sector": "healthcare"},
                                       _provider(_score(sector="tech")))
        signal = agent.generate_signal("AAPL", sample_candles_uptrend)
        # Should still process — sector filtering is at provider level
        assert signal.direction == 0.0


class TestMarketCapAgent:
    def test_large_cap_high_quality(self, sample_candles_uptrend):
        from aegis.agents.fundamental.market_cap import MarketCapAgent
        agent = MarketCapAgent("fund_12", {"tier": "large"},
                               _provider(_score(market_cap_tier="large", quality=0.8)))
        signal = agent.generate_signal("AAPL", sample_candles_uptrend)
        assert signal.metadata.get("confidence_modifier", 1.0) >= 1.0

    def test_wrong_tier_neutral(self, sample_candles_uptrend):
        from aegis.agents.fundamental.market_cap import MarketCapAgent
        agent = MarketCapAgent("fund_14", {"tier": "small"},
                               _provider(_score(market_cap_tier="large")))
        signal = agent.generate_signal("AAPL", sample_candles_uptrend)
        assert signal.direction == 0.0  # Tier mismatch

    def test_no_data_neutral(self, sample_candles_uptrend):
        from aegis.agents.fundamental.market_cap import MarketCapAgent
        agent = MarketCapAgent("fund_12", {"tier": "large"})
        signal = agent.generate_signal("AAPL", sample_candles_uptrend)
        assert signal.direction == 0.0


class TestEarningsSurpriseAgent:
    def test_no_data_neutral(self, sample_candles_uptrend):
        from aegis.agents.fundamental.earnings import EarningsSurpriseAgent
        agent = EarningsSurpriseAgent("fund_15", {})
        signal = agent.generate_signal("AAPL", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_positive_surprise_boost(self, sample_candles_uptrend):
        from aegis.agents.fundamental.earnings import EarningsSurpriseAgent
        provider = _provider(_score(quality=0.9, growth=0.95, revenue_growth=0.20))
        agent = EarningsSurpriseAgent("fund_15", {}, provider)
        signal = agent.generate_signal("AAPL", sample_candles_uptrend)
        assert signal.metadata.get("confidence_modifier", 1.0) >= 1.0

    def test_agent_type(self):
        from aegis.agents.fundamental.earnings import EarningsSurpriseAgent
        agent = EarningsSurpriseAgent("fund_15", {})
        assert agent.agent_type == "fundamental"


class TestRegistration:
    def test_all_fundamental_registered(self):
        from aegis.agents.registry import get_agent_class
        from aegis.agents.fundamental.sector import SectorFundamentalAgent
        from aegis.agents.fundamental.market_cap import MarketCapAgent
        from aegis.agents.fundamental.earnings import EarningsSurpriseAgent
        assert get_agent_class("fundamental", "sector") is SectorFundamentalAgent
        assert get_agent_class("fundamental", "market_cap") is MarketCapAgent
        assert get_agent_class("fundamental", "earnings_surprise") is EarningsSurpriseAgent
