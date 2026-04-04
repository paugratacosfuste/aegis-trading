"""Agent factory: creates agent instances from YAML config.

Usage:
    agents = create_agents_from_config(config.agents)
    agents = create_default_agents()  # Phase 1 backward compat
"""

from __future__ import annotations

from typing import Any

from aegis.agents.base import BaseAgent
from aegis.agents.registry import get_agent_class, restore_registry


def _ensure_registered() -> None:
    """Import all agent modules and restore registry if needed."""
    # Phase 1 agents
    import aegis.agents.technical.rsi_ema  # noqa: F401
    import aegis.agents.statistical.zscore  # noqa: F401
    import aegis.agents.momentum.timeseries  # noqa: F401
    # Phase 2 technical agents
    import aegis.agents.technical.indicator  # noqa: F401
    import aegis.agents.technical.asian_range  # noqa: F401
    # Phase 2 statistical agents
    import aegis.agents.statistical.ou  # noqa: F401
    import aegis.agents.statistical.kalman  # noqa: F401
    import aegis.agents.statistical.bollinger_zscore  # noqa: F401
    import aegis.agents.statistical.hurst  # noqa: F401
    import aegis.agents.statistical.multi_window  # noqa: F401
    import aegis.agents.statistical.sector_relative  # noqa: F401
    import aegis.agents.statistical.pairs  # noqa: F401
    # Phase 2 momentum agents
    import aegis.agents.momentum.cross_sectional  # noqa: F401
    import aegis.agents.momentum.dual  # noqa: F401
    import aegis.agents.momentum.volume_weighted  # noqa: F401
    import aegis.agents.momentum.rsi_filtered  # noqa: F401
    import aegis.agents.momentum.acceleration  # noqa: F401
    # Phase 2 sentiment agents
    import aegis.agents.sentiment.news  # noqa: F401
    import aegis.agents.sentiment.reddit  # noqa: F401
    import aegis.agents.sentiment.fear_greed  # noqa: F401
    import aegis.agents.sentiment.combined  # noqa: F401
    # Phase 3 macro agents
    import aegis.agents.macro.yield_curve  # noqa: F401
    import aegis.agents.macro.risk_regime  # noqa: F401
    import aegis.agents.macro.economic_cycle  # noqa: F401
    import aegis.agents.macro.inflation  # noqa: F401
    import aegis.agents.macro.hmm_regime  # noqa: F401
    # Phase 3 geopolitical agents
    import aegis.agents.geopolitical.conflict  # noqa: F401
    import aegis.agents.geopolitical.trade_policy  # noqa: F401
    # Phase 3 world leader agent
    import aegis.agents.world_leader.statement_agent  # noqa: F401
    # Phase 3 fundamental agents
    import aegis.agents.fundamental.sector  # noqa: F401
    import aegis.agents.fundamental.market_cap  # noqa: F401
    import aegis.agents.fundamental.earnings  # noqa: F401
    # Phase 3 crypto agents
    import aegis.agents.crypto.funding  # noqa: F401
    import aegis.agents.crypto.dominance  # noqa: F401
    import aegis.agents.crypto.crypto_sentiment  # noqa: F401
    import aegis.agents.crypto.defi  # noqa: F401
    import aegis.agents.crypto.crypto_technical  # noqa: F401
    restore_registry()


def create_agents_from_config(
    agents_config: dict[str, list[dict[str, Any]]],
    enabled_types: list[str] | None = None,
) -> list[BaseAgent]:
    """Create all agents defined in the YAML agents section.

    agents_config shape:
        {"technical": [{"id": "tech_01", "strategy": "indicator", "params": {...}}, ...],
         "statistical": [...], ...}

    If enabled_types is provided, only agent types in the list are created.
    """
    _ensure_registered()
    agents: list[BaseAgent] = []
    for agent_type, agent_defs in agents_config.items():
        if enabled_types is not None and agent_type not in enabled_types:
            continue
        for agent_def in agent_defs:
            agent_id = agent_def["id"]
            strategy = agent_def["strategy"]
            params = agent_def.get("params", {})
            cls = get_agent_class(agent_type, strategy)
            agents.append(cls(agent_id, params))
    return agents


def create_default_agents() -> list[BaseAgent]:
    """Create the original Phase 1 agent set (backward compat)."""
    # Import here to trigger registration
    from aegis.agents.momentum.timeseries import MomentumAgent
    from aegis.agents.statistical.zscore import ZScoreAgent
    from aegis.agents.technical.rsi_ema import RsiEmaAgent

    return [
        RsiEmaAgent("rsi_ema_bt", {}),
        ZScoreAgent("zscore_bt", {}),
        MomentumAgent("momentum_bt", {}),
    ]
