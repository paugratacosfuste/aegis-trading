"""Agent registry: maps (agent_type, strategy) to agent classes.

Agents self-register at import time via the @register_agent decorator.
The factory uses the registry to create instances from config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aegis.agents.base import BaseAgent

_REGISTRY: dict[tuple[str, str], type[BaseAgent]] = {}
# Permanent record so re-registration works after _clear_registry (testing)
_ALL_REGISTERED: list[tuple[str, str, type[BaseAgent]]] = []


class AgentRegistrationError(Exception):
    """Raised when an agent lookup or registration fails."""


def register_agent(agent_type: str, strategy: str):
    """Class decorator that registers an agent class.

    Usage:
        @register_agent("technical", "rsi_ema")
        class RsiEmaAgent(BaseAgent): ...
    """

    def decorator(cls: type[BaseAgent]) -> type[BaseAgent]:
        key = (agent_type, strategy)
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise AgentRegistrationError(
                f"Duplicate registration for {key}: "
                f"{_REGISTRY[key].__name__} already registered"
            )
        _REGISTRY[key] = cls
        _ALL_REGISTERED.append((agent_type, strategy, cls))
        return cls

    return decorator


def get_agent_class(agent_type: str, strategy: str) -> type[BaseAgent]:
    """Look up a registered agent class.

    Raises AgentRegistrationError if not found.
    """
    key = (agent_type, strategy)
    if key not in _REGISTRY:
        available = ", ".join(f"{t}:{s}" for t, s in sorted(_REGISTRY.keys()))
        raise AgentRegistrationError(
            f"No agent registered for {agent_type}:{strategy}. "
            f"Available: [{available}]"
        )
    return _REGISTRY[key]


def list_registered() -> list[tuple[str, str]]:
    """Return all registered (agent_type, strategy) pairs."""
    return sorted(_REGISTRY.keys())


def restore_registry() -> None:
    """Restore all registrations from permanent record.

    Call this after _clear_registry to bring back all
    agents that were registered via @register_agent.
    """
    for agent_type, strategy, cls in _ALL_REGISTERED:
        _REGISTRY[(agent_type, strategy)] = cls


def _clear_registry() -> None:
    """Clear active registry. Only for testing.

    Use restore_registry() to bring entries back.
    """
    _REGISTRY.clear()
