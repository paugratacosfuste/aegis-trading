"""Lab types: cohort configuration, strategy cohort, tournament results."""

from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import Mapping


class CohortStatus:
    CREATED = "created"
    BURN_IN = "burn_in"
    EVALUATING = "evaluating"
    PROMOTED = "promoted"
    RELEGATED = "relegated"
    RETIRED = "retired"

    ALL = frozenset({CREATED, BURN_IN, EVALUATING, PROMOTED, RELEGATED, RETIRED})


@dataclass(frozen=True)
class CohortConfig:
    """Immutable configuration for a strategy cohort."""

    agent_weights: Mapping[str, float]
    confidence_threshold: float
    risk_params: Mapping[str, float]
    universe: tuple[str, ...]
    invert_sentiment: bool = False
    macro_position_sizing: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_weights", MappingProxyType(dict(self.agent_weights)))
        object.__setattr__(self, "risk_params", MappingProxyType(dict(self.risk_params)))

    def to_dict(self) -> dict:
        return {
            "agent_weights": dict(self.agent_weights),
            "confidence_threshold": self.confidence_threshold,
            "risk_params": dict(self.risk_params),
            "universe": list(self.universe),
            "invert_sentiment": self.invert_sentiment,
            "macro_position_sizing": self.macro_position_sizing,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CohortConfig":
        return cls(
            agent_weights=data["agent_weights"],
            confidence_threshold=float(data["confidence_threshold"]),
            risk_params=data.get("risk_params", {}),
            universe=tuple(data.get("universe", ())),
            invert_sentiment=data.get("invert_sentiment", False),
            macro_position_sizing=data.get("macro_position_sizing", False),
        )


@dataclass(frozen=True)
class StrategyCohort:
    """Immutable snapshot of a strategy cohort's state."""

    cohort_id: str
    name: str
    status: str
    config: CohortConfig
    generation: int = 0
    parent_cohort_id: str | None = None
    relegation_count: int = 0
    created_at: datetime | None = None
    burn_in_start: datetime | None = None
    evaluation_start: datetime | None = None
    virtual_capital: float = 100_000.0


@dataclass(frozen=True)
class CohortPerformance:
    """Running performance metrics snapshot for a cohort."""

    cohort_id: str
    sharpe: float
    win_rate: float
    max_drawdown: float  # Negative fraction, e.g. -0.15
    profit_factor: float
    total_trades: int
    net_pnl: float
    equity_curve: tuple[float, ...] = ()


@dataclass(frozen=True)
class TournamentResult:
    """One cohort's result in a weekly tournament."""

    cohort_id: str
    week_start: date
    sharpe: float
    win_rate: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    net_pnl: float
    composite_score: float
    rank: int
