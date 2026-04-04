"""Cohort configuration templates: 10 strategy hypotheses (A-J)."""

from datetime import datetime, timezone

from aegis.lab.types import CohortConfig, CohortStatus, StrategyCohort

# Default risk parameters
_DEFAULT_RISK = {
    "max_risk_per_trade": 0.05,
    "kelly_fraction": 0.5,
    "stop_loss_pct": 0.05,
}

_CONSERVATIVE_RISK = {
    "max_risk_per_trade": 0.01,
    "kelly_fraction": 0.3,
    "stop_loss_pct": 0.03,
}

_AGGRESSIVE_RISK = {
    "max_risk_per_trade": 0.08,
    "kelly_fraction": 0.8,
    "stop_loss_pct": 0.08,
}

_DEFAULT_UNIVERSE = (
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "SPY", "QQQ", "AAPL", "MSFT", "GOOGL",
)


TEMPLATES: dict[str, dict] = {
    "A": {
        "name": "Baseline Production",
        "config": CohortConfig(
            agent_weights={
                "technical": 0.25, "statistical": 0.20, "momentum": 0.20,
                "sentiment": 0.15, "geopolitical": 0.05, "world_leader": 0.05,
                "crypto": 0.10,
            },
            confidence_threshold=0.40,
            risk_params=_DEFAULT_RISK,
            universe=_DEFAULT_UNIVERSE,
        ),
    },
    "B": {
        "name": "Trend-Following Heavy",
        "config": CohortConfig(
            agent_weights={
                "technical": 0.35, "statistical": 0.10, "momentum": 0.25,
                "sentiment": 0.10, "geopolitical": 0.05, "world_leader": 0.05,
                "crypto": 0.10,
            },
            confidence_threshold=0.45,
            risk_params=_DEFAULT_RISK,
            universe=_DEFAULT_UNIVERSE,
        ),
    },
    "C": {
        "name": "Mean-Reversion Heavy",
        "config": CohortConfig(
            agent_weights={
                "technical": 0.15, "statistical": 0.40, "momentum": 0.10,
                "sentiment": 0.15, "geopolitical": 0.05, "world_leader": 0.05,
                "crypto": 0.10,
            },
            confidence_threshold=0.45,
            risk_params=_DEFAULT_RISK,
            universe=_DEFAULT_UNIVERSE,
        ),
    },
    "D": {
        "name": "Sentiment-Driven",
        "config": CohortConfig(
            agent_weights={
                "technical": 0.15, "statistical": 0.10, "momentum": 0.10,
                "sentiment": 0.25, "geopolitical": 0.10, "world_leader": 0.20,
                "crypto": 0.10,
            },
            confidence_threshold=0.45,
            risk_params=_DEFAULT_RISK,
            universe=_DEFAULT_UNIVERSE,
        ),
    },
    "E": {
        "name": "Contrarian",
        "config": CohortConfig(
            agent_weights={
                "technical": 0.25, "statistical": 0.20, "momentum": 0.20,
                "sentiment": 0.15, "geopolitical": 0.05, "world_leader": 0.05,
                "crypto": 0.10,
            },
            confidence_threshold=0.45,
            risk_params=_DEFAULT_RISK,
            universe=_DEFAULT_UNIVERSE,
            invert_sentiment=True,
        ),
    },
    "F": {
        "name": "Macro-First",
        "config": CohortConfig(
            agent_weights={
                "technical": 0.25, "statistical": 0.20, "momentum": 0.20,
                "sentiment": 0.15, "geopolitical": 0.05, "world_leader": 0.05,
                "crypto": 0.10,
            },
            confidence_threshold=0.45,
            risk_params={**_DEFAULT_RISK, "kelly_fraction": 0.3},
            universe=_DEFAULT_UNIVERSE,
            macro_position_sizing=True,
        ),
    },
    "G": {
        "name": "Equal Weight",
        "config": CohortConfig(
            agent_weights={
                "technical": 1 / 7, "statistical": 1 / 7, "momentum": 1 / 7,
                "sentiment": 1 / 7, "geopolitical": 1 / 7, "world_leader": 1 / 7,
                "crypto": 1 / 7,
            },
            confidence_threshold=0.45,
            risk_params=_DEFAULT_RISK,
            universe=_DEFAULT_UNIVERSE,
        ),
    },
    "H": {
        "name": "Adaptive (RL Placeholder)",
        "config": CohortConfig(
            agent_weights={
                "technical": 1 / 7, "statistical": 1 / 7, "momentum": 1 / 7,
                "sentiment": 1 / 7, "geopolitical": 1 / 7, "world_leader": 1 / 7,
                "crypto": 1 / 7,
            },
            confidence_threshold=0.45,
            risk_params=_DEFAULT_RISK,
            universe=_DEFAULT_UNIVERSE,
        ),
    },
    "I": {
        "name": "Conservative",
        "config": CohortConfig(
            agent_weights={
                "technical": 0.25, "statistical": 0.20, "momentum": 0.20,
                "sentiment": 0.15, "geopolitical": 0.05, "world_leader": 0.05,
                "crypto": 0.10,
            },
            confidence_threshold=0.60,
            risk_params=_CONSERVATIVE_RISK,
            universe=_DEFAULT_UNIVERSE,
        ),
    },
    "J": {
        "name": "Aggressive",
        "config": CohortConfig(
            agent_weights={
                "technical": 0.25, "statistical": 0.20, "momentum": 0.20,
                "sentiment": 0.15, "geopolitical": 0.05, "world_leader": 0.05,
                "crypto": 0.10,
            },
            confidence_threshold=0.35,
            risk_params=_AGGRESSIVE_RISK,
            universe=_DEFAULT_UNIVERSE,
        ),
    },
}


def get_default_templates() -> list[StrategyCohort]:
    """Return all 10 cohort templates as StrategyCohort instances."""
    now = datetime.now(timezone.utc)
    return [
        create_cohort_from_template(tid, creation_time=now)
        for tid in TEMPLATES
    ]


def create_cohort_from_template(
    template_id: str,
    generation: int = 0,
    creation_time: datetime | None = None,
) -> StrategyCohort:
    """Create a StrategyCohort from a template ID (A-J)."""
    tmpl = TEMPLATES[template_id]
    return StrategyCohort(
        cohort_id=f"cohort_{template_id}",
        name=tmpl["name"],
        status=CohortStatus.CREATED,
        config=tmpl["config"],
        generation=generation,
        created_at=creation_time or datetime.now(timezone.utc),
    )
