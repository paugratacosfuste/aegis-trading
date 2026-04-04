"""Feedback loop types: frozen dataclasses for weight updates, models, playbooks."""

from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class WeightUpdateLog:
    """Audit record for a single agent weight update."""

    agent_id: str
    agent_type: str
    old_weight: float
    new_weight: float
    hit_rate: float
    ic: float
    composite_score: float
    n_signals: int
    update_date: date


@dataclass(frozen=True)
class AgentWeightSnapshot:
    """Current state of an agent's weight and performance metrics."""

    agent_id: str
    agent_type: str
    weight: float
    hit_rate: float
    ic: float
    rolling_sharpe: float | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class ModelVersion:
    """Record of a trained model version with validation metrics."""

    model_id: str
    model_type: str
    version: str
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    train_samples: int
    val_samples: int
    val_auc: float
    previous_auc: float | None
    accepted: bool
    feature_names: tuple[str, ...]
    model_path: str
    created_at: datetime | None = None


@dataclass(frozen=True)
class FeatureImportance:
    """SHAP feature importance for one feature in a model version."""

    model_id: str
    retrain_date: date
    feature_name: str
    shap_importance: float
    rank: int
    status: str = "stable"  # "stable" | "new" | "dropped"


@dataclass(frozen=True)
class RegimePlaybookEntry:
    """Learned optimal settings for a specific market regime."""

    regime: str
    last_updated: date
    total_observations: int
    best_agent_weights: Mapping[str, float]
    best_cohort_ids: tuple[str, ...]
    avg_sharpe: float
    avg_win_rate: float
    worst_agent_types: tuple[str, ...]
    recommended_position_size_mult: float = 1.0
    recommended_max_positions: int = 10
    recommended_confidence_threshold: float = 0.45

    def __post_init__(self) -> None:
        if not isinstance(self.best_agent_weights, MappingProxyType):
            object.__setattr__(
                self, "best_agent_weights",
                MappingProxyType(dict(self.best_agent_weights)),
            )


@dataclass(frozen=True)
class MonthlyRetrospective:
    """Automated monthly analysis report."""

    month: str  # "2026-04"
    production_return: float
    production_sharpe: float
    benchmark_return: float
    alpha: float
    best_lab_cohort: str | None
    worst_lab_cohort: str | None
    cohorts_promoted: tuple[str, ...]
    cohorts_relegated: tuple[str, ...]
    most_improved_agent: str | None
    most_degraded_agent: str | None
    agents_evolved: tuple[str, ...]
    feature_importance_shifts: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    regimes_encountered: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    regime_performance: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    rl_shadow_performance: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    recommendations: tuple[str, ...] = ()
    created_at: datetime | None = None
