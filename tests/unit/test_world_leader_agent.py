"""Tests for world leader agent (M4)."""

from datetime import datetime, timezone

import pytest


def _statement(leader="Powell", text="Rates unchanged", stmt_type="monetary_policy",
               sentiment=0.0, **kw):
    defaults = dict(
        leader=leader, text=text, source="fed_rss",
        timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc).isoformat(),
        statement_type=stmt_type, sentiment_score=sentiment,
    )
    defaults.update(kw)
    return defaults


class TestStatementAgent:
    def test_no_data_neutral(self, sample_candles_uptrend):
        from aegis.agents.world_leader.statement_agent import StatementAgent
        agent = StatementAgent("leader_01", {})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_hawkish_monetary_bearish(self, sample_candles_uptrend):
        from aegis.agents.world_leader.statement_agent import StatementAgent
        from aegis.agents.world_leader.providers import HistoricalLeaderProvider
        stmt = _statement(text="We will raise rates further to combat inflation",
                          sentiment=-0.6, stmt_type="monetary_policy")
        provider = HistoricalLeaderProvider(statements=[stmt])
        agent = StatementAgent("leader_01", {}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction < 0

    def test_dovish_monetary_bullish(self, sample_candles_uptrend):
        from aegis.agents.world_leader.statement_agent import StatementAgent
        from aegis.agents.world_leader.providers import HistoricalLeaderProvider
        stmt = _statement(text="Rate cuts are on the table for next meeting",
                          sentiment=0.6, stmt_type="monetary_policy")
        provider = HistoricalLeaderProvider(statements=[stmt])
        agent = StatementAgent("leader_01", {}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction > 0

    def test_half_life_social_media_short(self, sample_candles_uptrend):
        from aegis.agents.world_leader.statement_agent import StatementAgent
        from aegis.agents.world_leader.providers import HistoricalLeaderProvider
        stmt = _statement(leader="Trump", text="Big trade deal coming!",
                          sentiment=0.5, stmt_type="social_media")
        provider = HistoricalLeaderProvider(statements=[stmt])
        agent = StatementAgent("leader_01", {}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata.get("half_life_hours", 0) <= 6

    def test_half_life_executive_order_long(self, sample_candles_uptrend):
        from aegis.agents.world_leader.statement_agent import StatementAgent
        from aegis.agents.world_leader.providers import HistoricalLeaderProvider
        stmt = _statement(text="Executive order on trade restrictions",
                          sentiment=-0.5, stmt_type="executive_order")
        provider = HistoricalLeaderProvider(statements=[stmt])
        agent = StatementAgent("leader_01", {}, provider)
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata.get("half_life_hours", 0) >= 168  # 1+ week

    def test_agent_type(self):
        from aegis.agents.world_leader.statement_agent import StatementAgent
        agent = StatementAgent("leader_01", {})
        assert agent.agent_type == "world_leader"

    def test_registration(self):
        from aegis.agents.registry import get_agent_class
        from aegis.agents.world_leader.statement_agent import StatementAgent
        assert get_agent_class("world_leader", "statement") is StatementAgent
