"""Cohort mutation: generates new cohorts from successful parents."""

import uuid
from datetime import datetime, timezone

import numpy as np

from aegis.lab.types import CohortConfig, CohortStatus, StrategyCohort


def mutate_cohort(
    parent: StrategyCohort,
    mutation_rate: float = 0.15,
    rng: np.random.Generator | None = None,
) -> StrategyCohort:
    """Create a mutated child from a parent cohort.

    Mutates weights (±mutation_rate), threshold (±0.05), and max_risk (±0.01).
    All values clipped to valid ranges and weights renormalized to sum to 1.0.
    """
    gen = rng or np.random.default_rng()

    # Mutate weights
    new_weights = {}
    for key, w in parent.config.agent_weights.items():
        noise = gen.normal(0, mutation_rate * max(w, 0.05))
        new_w = np.clip(w + noise, 0.05, 0.40)
        new_weights[key] = float(new_w)

    # Renormalize
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {k: v / total for k, v in new_weights.items()}

    # Mutate threshold
    new_threshold = parent.config.confidence_threshold + gen.normal(0, 0.05)
    new_threshold = float(np.clip(new_threshold, 0.25, 0.75))

    # Mutate risk
    new_risk = dict(parent.config.risk_params)
    if "max_risk_per_trade" in new_risk:
        new_risk["max_risk_per_trade"] = float(np.clip(
            new_risk["max_risk_per_trade"] + gen.normal(0, 0.01),
            0.005, 0.10,
        ))

    child_id = f"cohort_mut_{uuid.uuid4().hex[:8]}"
    return StrategyCohort(
        cohort_id=child_id,
        name=f"Mutant of {parent.name} (gen {parent.generation + 1})",
        status=CohortStatus.CREATED,
        config=CohortConfig(
            agent_weights=new_weights,
            confidence_threshold=new_threshold,
            risk_params=new_risk,
            universe=parent.config.universe,
            invert_sentiment=parent.config.invert_sentiment,
            macro_position_sizing=parent.config.macro_position_sizing,
        ),
        generation=parent.generation + 1,
        parent_cohort_id=parent.cohort_id,
        created_at=datetime.now(timezone.utc),
    )


def generate_random_cohort(
    universe: tuple[str, ...] = (),
    rng: np.random.Generator | None = None,
) -> StrategyCohort:
    """Generate a cohort with Dirichlet-distributed random weights."""
    gen = rng or np.random.default_rng()

    types = ["technical", "statistical", "momentum", "sentiment",
             "geopolitical", "world_leader", "crypto"]
    weights = gen.dirichlet(np.ones(len(types)))
    agent_weights = {t: float(w) for t, w in zip(types, weights)}

    threshold = float(gen.uniform(0.30, 0.65))
    max_risk = float(gen.uniform(0.01, 0.08))

    child_id = f"cohort_rnd_{uuid.uuid4().hex[:8]}"
    return StrategyCohort(
        cohort_id=child_id,
        name=f"Random {child_id[-8:]}",
        status=CohortStatus.CREATED,
        config=CohortConfig(
            agent_weights=agent_weights,
            confidence_threshold=threshold,
            risk_params={"max_risk_per_trade": max_risk, "kelly_fraction": 0.5,
                         "stop_loss_pct": 0.05},
            universe=universe,
        ),
        created_at=datetime.now(timezone.utc),
    )
