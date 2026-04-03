"""Tests for agent factory."""

import pytest

from aegis.agents.factory import create_agents_from_config, create_default_agents
from aegis.agents.registry import AgentRegistrationError


class TestCreateDefaultAgents:
    def test_returns_three_agents(self):
        agents = create_default_agents()
        assert len(agents) == 3

    def test_agent_types(self):
        agents = create_default_agents()
        types = {a.agent_type for a in agents}
        assert types == {"technical", "statistical", "momentum"}

    def test_agent_ids(self):
        agents = create_default_agents()
        ids = {a.agent_id for a in agents}
        assert ids == {"rsi_ema_bt", "zscore_bt", "momentum_bt"}


class TestCreateFromConfig:
    def test_create_existing_agents_from_config(self):
        config = {
            "technical": [
                {"id": "tech_01", "strategy": "rsi_ema", "params": {}},
            ],
            "statistical": [
                {"id": "stat_01", "strategy": "zscore", "params": {"lookback": 20}},
                {"id": "stat_02", "strategy": "zscore", "params": {"lookback": 50}},
            ],
            "momentum": [
                {"id": "mom_01", "strategy": "timeseries", "params": {"lookback": 5}},
            ],
        }
        # Ensure agents are imported/registered
        from aegis.agents.technical.rsi_ema import RsiEmaAgent  # noqa: F401
        from aegis.agents.statistical.zscore import ZScoreAgent  # noqa: F401
        from aegis.agents.momentum.timeseries import MomentumAgent  # noqa: F401

        agents = create_agents_from_config(config)
        assert len(agents) == 4
        assert agents[0].agent_id == "tech_01"
        assert agents[1].agent_id == "stat_01"
        assert agents[2].agent_id == "stat_02"

    def test_empty_config_returns_empty(self):
        agents = create_agents_from_config({})
        assert agents == []

    def test_unknown_strategy_raises(self):
        config = {
            "technical": [
                {"id": "bad", "strategy": "nonexistent", "params": {}},
            ],
        }
        with pytest.raises(AgentRegistrationError):
            create_agents_from_config(config)

    def test_params_passed_to_agent(self):
        from aegis.agents.statistical.zscore import ZScoreAgent  # noqa: F401

        config = {
            "statistical": [
                {"id": "stat_custom", "strategy": "zscore", "params": {"lookback": 100}},
            ],
        }
        agents = create_agents_from_config(config)
        assert agents[0]._lookback == 100

    def test_missing_params_uses_defaults(self):
        from aegis.agents.statistical.zscore import ZScoreAgent  # noqa: F401

        config = {
            "statistical": [
                {"id": "stat_default", "strategy": "zscore"},
            ],
        }
        agents = create_agents_from_config(config)
        assert agents[0]._lookback == 20  # default
