"""Tests for geopolitical agents (M4)."""

from datetime import datetime, timezone

import pytest

from aegis.common.types import GeopoliticalEvent


def _event(category="conflict", severity=0.5, **kw) -> GeopoliticalEvent:
    defaults = dict(
        event_id="evt_01", timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
        source="gdelt", category=category, severity=severity,
        affected_sectors=("energy",), affected_regions=("middle_east",),
        raw_text="test event", sentiment_score=-0.3, half_life_hours=24,
    )
    defaults.update(kw)
    return GeopoliticalEvent(**defaults)


class TestConflictSanctionsAgent:
    def test_no_data_neutral(self, sample_candles_uptrend):
        from aegis.agents.geopolitical.conflict import ConflictSanctionsAgent
        agent = ConflictSanctionsAgent("geo_01", {})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_high_risk_veto(self, sample_candles_uptrend):
        from aegis.agents.geopolitical.conflict import ConflictSanctionsAgent
        from aegis.agents.geopolitical.providers import HistoricalGeopoliticalProvider
        provider = HistoricalGeopoliticalProvider(
            events=[_event(severity=0.9)], risk_score=0.85,
        )
        agent = ConflictSanctionsAgent("geo_01", {}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata.get("veto") is True
        assert signal.direction < 0

    def test_moderate_risk_no_veto(self, sample_candles_uptrend):
        from aegis.agents.geopolitical.conflict import ConflictSanctionsAgent
        from aegis.agents.geopolitical.providers import HistoricalGeopoliticalProvider
        provider = HistoricalGeopoliticalProvider(
            events=[_event(severity=0.4)], risk_score=0.4,
        )
        agent = ConflictSanctionsAgent("geo_01", {}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata.get("veto") is not True

    def test_agent_type(self):
        from aegis.agents.geopolitical.conflict import ConflictSanctionsAgent
        agent = ConflictSanctionsAgent("geo_01", {})
        assert agent.agent_type == "geopolitical"


class TestTradePolicyAgent:
    def test_no_data_neutral(self, sample_candles_uptrend):
        from aegis.agents.geopolitical.trade_policy import TradePolicyAgent
        agent = TradePolicyAgent("geo_02", {})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_protectionist_policy_bearish(self, sample_candles_uptrend):
        from aegis.agents.geopolitical.trade_policy import TradePolicyAgent
        from aegis.agents.geopolitical.providers import HistoricalGeopoliticalProvider
        events = [
            _event(category="trade", severity=0.7, raw_text="New tariff on imports"),
            _event(category="trade", severity=0.6, raw_text="Export ban announced",
                   event_id="evt_02"),
        ]
        provider = HistoricalGeopoliticalProvider(events=events, risk_score=0.6)
        agent = TradePolicyAgent("geo_02", {}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction < 0  # Protectionist = bearish

    def test_high_severity_veto(self, sample_candles_uptrend):
        from aegis.agents.geopolitical.trade_policy import TradePolicyAgent
        from aegis.agents.geopolitical.providers import HistoricalGeopoliticalProvider
        events = [_event(category="trade", severity=0.9)]
        provider = HistoricalGeopoliticalProvider(events=events, risk_score=0.85)
        agent = TradePolicyAgent("geo_02", {}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata.get("veto") is True


class TestRegistration:
    def test_all_geo_registered(self):
        from aegis.agents.registry import get_agent_class
        from aegis.agents.geopolitical.conflict import ConflictSanctionsAgent
        from aegis.agents.geopolitical.trade_policy import TradePolicyAgent
        assert get_agent_class("geopolitical", "conflict_sanctions") is ConflictSanctionsAgent
        assert get_agent_class("geopolitical", "trade_policy") is TradePolicyAgent
