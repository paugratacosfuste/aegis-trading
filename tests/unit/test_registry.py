"""Tests for agent registry."""

import pytest

from aegis.agents.base import BaseAgent
from aegis.agents.registry import (
    AgentRegistrationError,
    _clear_registry,
    _REGISTRY,
    get_agent_class,
    list_registered,
    register_agent,
    restore_registry,
)
from aegis.common.types import AgentSignal, MarketDataPoint


class _DummyAgent(BaseAgent):
    @property
    def agent_type(self) -> str:
        return "test"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        return self._neutral_signal(symbol)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry around each test."""
    saved = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)
    restore_registry()


class TestRegistry:
    def test_register_and_retrieve(self):
        register_agent("test", "dummy")(_DummyAgent)
        cls = get_agent_class("test", "dummy")
        assert cls is _DummyAgent

    def test_unknown_strategy_raises(self):
        with pytest.raises(AgentRegistrationError, match="No agent registered"):
            get_agent_class("nonexistent", "nope")

    def test_duplicate_registration_same_class_ok(self):
        register_agent("test", "dup")(_DummyAgent)
        # Same class re-registration is idempotent
        register_agent("test", "dup")(_DummyAgent)
        assert get_agent_class("test", "dup") is _DummyAgent

    def test_duplicate_registration_different_class_raises(self):
        class _OtherAgent(_DummyAgent):
            pass

        register_agent("test", "dup2")(_DummyAgent)
        with pytest.raises(AgentRegistrationError, match="Duplicate"):
            register_agent("test", "dup2")(_OtherAgent)

    def test_list_registered(self):
        register_agent("test", "aaa")(_DummyAgent)
        entries = list_registered()
        assert ("test", "aaa") in entries

    def test_existing_agents_registered(self):
        # Importing the modules triggers registration
        from aegis.agents.technical.rsi_ema import RsiEmaAgent  # noqa: F401
        from aegis.agents.statistical.zscore import ZScoreAgent  # noqa: F401
        from aegis.agents.momentum.timeseries import MomentumAgent  # noqa: F401

        registered = list_registered()
        assert ("technical", "rsi_ema") in registered
        assert ("statistical", "zscore") in registered
        assert ("momentum", "timeseries") in registered

    def test_clear_registry(self):
        register_agent("test", "clearme")(_DummyAgent)
        _clear_registry()
        assert len(_REGISTRY) == 0
