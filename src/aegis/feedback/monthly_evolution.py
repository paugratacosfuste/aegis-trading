"""Monthly agent evolution: Speed 3 of the feedback loop.

Replaces the bottom 20% of agents within each type with mutations
from top performers, random new parameters, or regime specialists.
"""

import copy
import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# Per-strategy mutation schemas: {param_name: (min, max, mutation_std)}
MUTATION_SCHEMAS: dict[str, dict[str, tuple[float, float, float]]] = {
    "timeseries": {"lookback": (3, 200, 5.0)},
    "zscore": {"lookback": (10, 200, 10.0)},
    "ornstein_uhlenbeck": {"lookback": (20, 200, 10.0)},
    "kalman": {
        "process_noise": (0.001, 0.1, 0.005),
        "measurement_noise": (0.1, 10.0, 0.5),
    },
    "bollinger_zscore": {"period": (10, 50, 5.0)},
    "hurst_zscore": {"lookback": (50, 200, 15.0)},
    "cross_sectional": {},
    "dual": {
        "lookback": (5, 100, 5.0),
        "benchmark_lookback": (20, 200, 10.0),
    },
    "volume_weighted": {"lookback": (5, 100, 5.0)},
    "rsi_filtered": {
        "lookback": (5, 100, 5.0),
        "rsi_threshold": (50, 90, 5.0),
    },
    "acceleration": {
        "short_lookback": (3, 20, 2.0),
        "long_lookback": (10, 100, 5.0),
    },
}


def identify_bottom_agents(
    sharpes: dict[str, float],
    bottom_pct: float = 0.20,
) -> list[str]:
    """Identify bottom performers by rolling Sharpe.

    Returns agent IDs of the bottom bottom_pct fraction.
    Single agents cannot be evolved.
    """
    n = len(sharpes)
    if n <= 1:
        return []

    n_bottom = int(n * bottom_pct)
    if n_bottom == 0:
        return []

    sorted_agents = sorted(sharpes.items(), key=lambda x: x[1])
    return [agent_id for agent_id, _ in sorted_agents[:n_bottom]]


def mutate_agent_params(
    parent_params: dict,
    schema: dict[str, tuple[float, float, float]],
    rng: np.random.Generator,
) -> dict:
    """Mutate numeric params using Gaussian noise from schema.

    Non-schema params are preserved unchanged.
    """
    result = dict(parent_params)
    for param, (lo, hi, std) in schema.items():
        if param in result:
            current = float(result[param])
            noise = rng.normal(0, std)
            new_val = float(np.clip(current + noise, lo, hi))
            # Preserve int type if original was int
            if isinstance(parent_params[param], int):
                new_val = int(round(new_val))
            result[param] = new_val
    return result


def random_agent_params(
    schema: dict[str, tuple[float, float, float]],
    rng: np.random.Generator,
) -> dict:
    """Generate random params within schema bounds.

    Integer params (where both bounds are whole numbers) are rounded to int.
    """
    result = {}
    for param, (lo, hi, _std) in schema.items():
        val = rng.uniform(lo, hi)
        # If both bounds are whole numbers, treat as integer param
        if lo == int(lo) and hi == int(hi):
            val = int(round(val))
        result[param] = val
    return result


def run_monthly_evolution(
    agent_configs: dict[str, list[dict]],
    agent_sharpes: dict[str, float],
    agent_types: dict[str, str],
    bottom_pct: float = 0.20,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, list[dict]], list[str]]:
    """Run monthly agent evolution.

    Args:
        agent_configs: {agent_type: [{"id": ..., "strategy": ..., "params": ...}]}
        agent_sharpes: {agent_id: rolling_sharpe_90d}
        agent_types: {agent_id: agent_type}
        bottom_pct: Fraction of agents to evolve per type.
        rng: Random number generator for reproducibility.

    Returns:
        (updated_agent_configs, list_of_evolved_agent_ids)
    """
    if rng is None:
        rng = np.random.default_rng()

    updated_configs = copy.deepcopy(agent_configs)
    all_evolved: list[str] = []

    # Group agents by type
    by_type: dict[str, dict[str, float]] = defaultdict(dict)
    for agent_id, sharpe in agent_sharpes.items():
        agent_type = agent_types.get(agent_id)
        if agent_type:
            by_type[agent_type][agent_id] = sharpe

    for agent_type, type_sharpes in by_type.items():
        if agent_type not in updated_configs:
            continue

        bottom_ids = identify_bottom_agents(type_sharpes, bottom_pct)
        if not bottom_ids:
            continue

        # Identify top performers for mutation source
        sorted_agents = sorted(type_sharpes.items(), key=lambda x: x[1], reverse=True)
        n_top = max(1, int(len(sorted_agents) * 0.20))
        top_ids = [aid for aid, _ in sorted_agents[:n_top]]

        # Build lookup: agent_id -> config index
        config_list = updated_configs[agent_type]
        id_to_idx = {cfg["id"]: i for i, cfg in enumerate(config_list)}

        for agent_id in bottom_ids:
            if agent_id not in id_to_idx:
                continue

            idx = id_to_idx[agent_id]
            cfg = config_list[idx]
            strategy = cfg.get("strategy", "")
            schema = MUTATION_SCHEMAS.get(strategy, {})

            # Choose evolution strategy
            roll = rng.random()
            if roll < 0.50 and top_ids:
                # Mutate from top performer
                parent_id = rng.choice(top_ids)
                parent_idx = id_to_idx.get(parent_id)
                if parent_idx is not None:
                    parent_params = config_list[parent_idx]["params"]
                    new_params = mutate_agent_params(parent_params, schema, rng)
                else:
                    new_params = _evolve_random_or_copy(cfg["params"], schema, rng)
            elif roll < 0.80:
                # Random new params
                new_params = _evolve_random_or_copy(cfg["params"], schema, rng)
            else:
                # Regime specialist (falls back to random if no playbook)
                new_params = _evolve_random_or_copy(cfg["params"], schema, rng)

            config_list[idx] = {**cfg, "params": new_params}
            all_evolved.append(agent_id)

    return updated_configs, all_evolved


def _evolve_random_or_copy(
    current_params: dict,
    schema: dict[str, tuple[float, float, float]],
    rng: np.random.Generator,
) -> dict:
    """Generate random params for schema keys, preserve non-schema keys."""
    if not schema:
        # No numeric params to mutate; return params unchanged
        return dict(current_params)
    random_vals = random_agent_params(schema, rng)
    result = dict(current_params)
    result.update(random_vals)
    return result
