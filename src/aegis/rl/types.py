"""RL Meta-Controller shared types.

All types are frozen dataclasses for immutability.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PromotionStage(str, Enum):
    """RL component lifecycle stages."""

    TRAINING = "training"
    SHADOW = "shadow"
    CANDIDATE = "candidate"  # Passed 90-day outperformance
    ACTIVE = "active"  # Promoted to influence decisions
    RETIRED = "retired"


class ExitAction(int, Enum):
    """Discrete exit management actions."""

    HOLD = 0
    TIGHTEN_STOP = 1
    PARTIAL_25 = 2
    PARTIAL_50 = 3
    FULL_EXIT = 4


@dataclass(frozen=True)
class WeightConfig:
    """A specific agent-type weight allocation."""

    config_id: int
    name: str
    weights: dict[str, float]  # agent_type -> weight (sums to 1.0)

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")
        if any(v < 0 for v in self.weights.values()):
            raise ValueError("All weights must be non-negative")


@dataclass(frozen=True)
class RLPrediction:
    """A single RL prediction for shadow tracking."""

    component: str  # "weight_allocator" | "position_sizer" | "exit_manager"
    timestamp: datetime
    symbol: str
    prediction: dict  # Component-specific prediction data
    context_features: dict  # Feature vector used for prediction
    model_version: str
    mode: str = "shadow"  # "shadow" | "active"


@dataclass(frozen=True)
class ShadowResult:
    """Counterfactual result comparing RL vs baseline."""

    component: str
    prediction: RLPrediction
    baseline_value: float  # What the rule-based system chose
    rl_value: float  # What the RL would have chosen
    actual_outcome: float  # What actually happened (PnL, Sharpe, etc.)
    rl_would_have_been: float  # Counterfactual outcome
    timestamp: datetime


@dataclass(frozen=True)
class ShadowSummary:
    """Aggregate shadow mode performance for a component."""

    component: str
    total_predictions: int
    baseline_cumulative: float
    rl_cumulative: float
    baseline_sharpe: float
    rl_sharpe: float
    outperformance_days: int
    total_days: int
    stage: PromotionStage
