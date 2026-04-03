"""Tests for Phase 3 types and providers."""

from datetime import datetime, timezone

import pytest

from aegis.common.types import CryptoMetrics, FundamentalScore


class TestFundamentalScore:
    def test_creation(self):
        fs = FundamentalScore(
            symbol="AAPL", timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
            sector="tech", market_cap_tier="large", quality_score=0.8,
            value_score=0.6, growth_score=0.9, pe_zscore=-0.5,
            revenue_growth=0.15, source="yahoo",
        )
        assert fs.symbol == "AAPL"
        assert fs.quality_score == 0.8
        assert fs.sector == "tech"

    def test_frozen(self):
        fs = FundamentalScore(
            symbol="AAPL", timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
            sector="tech", market_cap_tier="large", quality_score=0.8,
            value_score=0.6, growth_score=0.9, pe_zscore=-0.5,
            revenue_growth=0.15, source="yahoo",
        )
        with pytest.raises(AttributeError):
            fs.quality_score = 0.5  # type: ignore[misc]


class TestCryptoMetrics:
    def test_creation(self):
        cm = CryptoMetrics(
            symbol="BTC/USDT", timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
            funding_rate=0.01, open_interest=5e9, btc_dominance=55.0,
            fear_greed_index=72, tvl=50e9, tvl_change_24h=2.5,
            liquidations_24h=100e6, source="binance",
        )
        assert cm.symbol == "BTC/USDT"
        assert cm.funding_rate == 0.01
        assert cm.fear_greed_index == 72

    def test_frozen(self):
        cm = CryptoMetrics(
            symbol="BTC/USDT", timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
            funding_rate=0.01, open_interest=5e9, btc_dominance=55.0,
            fear_greed_index=72, tvl=50e9, tvl_change_24h=2.5,
            liquidations_24h=100e6, source="binance",
        )
        with pytest.raises(AttributeError):
            cm.funding_rate = 0.02  # type: ignore[misc]


class TestMacroProvider:
    def test_null_returns_none(self):
        from aegis.agents.macro.providers import NullMacroProvider
        p = NullMacroProvider()
        assert p.get_macro_snapshot() is None
        assert p.get_yield_curve() is None
        assert p.get_vix_history() is None

    def test_historical_returns_latest(self):
        from aegis.common.types import MacroDataPoint
        from aegis.agents.macro.providers import HistoricalMacroProvider
        snap = MacroDataPoint(
            timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
            yield_10y=4.0, yield_2y=3.8, yield_spread=0.2,
            vix=18.0, vix_regime="normal", dxy=104.0,
            fed_rate=5.25, cpi_latest=3.2,
        )
        p = HistoricalMacroProvider(
            snapshots=[snap],
            yield_curve={"2Y": 3.8, "10Y": 4.0},
            vix_history=[18.0, 19.0, 17.5],
        )
        assert p.get_macro_snapshot() == snap
        assert p.get_yield_curve()["10Y"] == 4.0
        assert len(p.get_vix_history()) == 3


class TestGeopoliticalProvider:
    def test_null_returns_empty(self):
        from aegis.agents.geopolitical.providers import NullGeopoliticalProvider
        p = NullGeopoliticalProvider()
        assert p.get_recent_events() == []
        assert p.get_risk_score() == 0.0

    def test_historical_returns_events(self):
        from aegis.common.types import GeopoliticalEvent
        from aegis.agents.geopolitical.providers import HistoricalGeopoliticalProvider
        event = GeopoliticalEvent(
            event_id="evt_01", timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
            source="gdelt", category="conflict", severity=0.8,
            affected_sectors=("energy", "defense"),
            affected_regions=("middle_east",), raw_text="test",
            sentiment_score=-0.6, half_life_hours=24,
        )
        p = HistoricalGeopoliticalProvider(events=[event], risk_score=0.75)
        assert len(p.get_recent_events()) == 1
        assert p.get_risk_score() == 0.75


class TestLeaderProvider:
    def test_null_returns_empty(self):
        from aegis.agents.world_leader.providers import NullLeaderProvider
        p = NullLeaderProvider()
        assert p.get_recent_statements() == []

    def test_historical_returns_statements(self):
        from aegis.agents.world_leader.providers import HistoricalLeaderProvider
        stmt = {"leader": "Powell", "text": "Rates unchanged", "sentiment_score": 0.0}
        p = HistoricalLeaderProvider(statements=[stmt])
        assert len(p.get_recent_statements()) == 1


class TestFundamentalProvider:
    def test_null_returns_none(self):
        from aegis.agents.fundamental.providers import NullFundamentalProvider
        p = NullFundamentalProvider()
        assert p.get_fundamentals("AAPL") is None
        assert p.get_sector_fundamentals("tech") == []

    def test_historical_by_symbol(self):
        from aegis.agents.fundamental.providers import HistoricalFundamentalProvider
        fs = FundamentalScore(
            symbol="AAPL", timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
            sector="tech", market_cap_tier="large", quality_score=0.8,
            value_score=0.6, growth_score=0.9, pe_zscore=-0.5,
            revenue_growth=0.15, source="yahoo",
        )
        p = HistoricalFundamentalProvider(scores=[fs])
        assert p.get_fundamentals("AAPL") == fs
        assert p.get_fundamentals("MSFT") is None
        assert len(p.get_sector_fundamentals("tech")) == 1


class TestCryptoMetricsProvider:
    def test_null_returns_none(self):
        from aegis.agents.crypto.providers import NullCryptoMetricsProvider
        p = NullCryptoMetricsProvider()
        assert p.get_metrics("BTC/USDT") is None
        assert p.get_funding_rate("BTC/USDT") is None
        assert p.get_fear_greed() is None

    def test_historical_returns_metrics(self):
        from aegis.agents.crypto.providers import HistoricalCryptoMetricsProvider
        cm = CryptoMetrics(
            symbol="BTC/USDT", timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
            funding_rate=0.01, open_interest=5e9, btc_dominance=55.0,
            fear_greed_index=72, tvl=50e9, tvl_change_24h=2.5,
            liquidations_24h=100e6, source="binance",
        )
        p = HistoricalCryptoMetricsProvider(metrics=[cm], fear_greed=72)
        assert p.get_metrics("BTC/USDT") == cm
        assert p.get_funding_rate("BTC/USDT") == 0.01
        assert p.get_fear_greed() == 72
        assert p.get_metrics("ETH/USDT") is None
