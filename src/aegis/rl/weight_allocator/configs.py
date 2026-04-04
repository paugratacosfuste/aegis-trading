"""50 pre-defined weight configurations for the contextual bandit.

Config #0 = current BASE_TYPE_WEIGHTS from ensemble/weights.py.
Others explore different allocations for various market conditions.
All configs have all 7 agent types and sum to 1.0.
"""

from aegis.rl.constants import AGENT_TYPES
from aegis.rl.types import WeightConfig

_T = AGENT_TYPES  # shorthand: technical, statistical, momentum, sentiment, geopolitical, world_leader, crypto


def _cfg(cid: int, name: str, w: tuple[float, ...]) -> WeightConfig:
    """Create WeightConfig from ordered tuple matching AGENT_TYPES."""
    weights = dict(zip(_T, w))
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}  # Normalize
    return WeightConfig(config_id=cid, name=name, weights=weights)


# fmt: off
WEIGHT_CONFIGS: list[WeightConfig] = [
    # --- Baseline ---
    _cfg(0,  "baseline",            (0.25, 0.20, 0.20, 0.15, 0.05, 0.05, 0.10)),
    # --- Momentum-heavy ---
    _cfg(1,  "momentum_heavy",      (0.15, 0.10, 0.40, 0.15, 0.05, 0.05, 0.10)),
    _cfg(2,  "momentum_dominant",   (0.10, 0.05, 0.50, 0.15, 0.05, 0.05, 0.10)),
    _cfg(3,  "momentum_tech",       (0.30, 0.05, 0.35, 0.10, 0.05, 0.05, 0.10)),
    # --- Mean-reversion ---
    _cfg(4,  "stat_heavy",          (0.15, 0.40, 0.10, 0.15, 0.05, 0.05, 0.10)),
    _cfg(5,  "stat_dominant",       (0.10, 0.50, 0.05, 0.15, 0.05, 0.05, 0.10)),
    _cfg(6,  "stat_tech",           (0.25, 0.35, 0.10, 0.10, 0.05, 0.05, 0.10)),
    # --- Technical-focused ---
    _cfg(7,  "tech_heavy",          (0.40, 0.15, 0.15, 0.10, 0.05, 0.05, 0.10)),
    _cfg(8,  "tech_dominant",       (0.50, 0.10, 0.15, 0.10, 0.03, 0.02, 0.10)),
    _cfg(9,  "tech_crypto",         (0.35, 0.10, 0.10, 0.10, 0.05, 0.05, 0.25)),
    # --- Sentiment-focused ---
    _cfg(10, "sentiment_heavy",     (0.15, 0.15, 0.15, 0.35, 0.05, 0.05, 0.10)),
    _cfg(11, "sentiment_geo",       (0.10, 0.10, 0.10, 0.30, 0.20, 0.10, 0.10)),
    _cfg(12, "sentiment_balanced",  (0.20, 0.15, 0.15, 0.25, 0.10, 0.05, 0.10)),
    # --- Defensive ---
    _cfg(13, "defensive_geo",       (0.10, 0.10, 0.05, 0.20, 0.30, 0.15, 0.10)),
    _cfg(14, "defensive_balanced",  (0.15, 0.20, 0.05, 0.20, 0.20, 0.10, 0.10)),
    _cfg(15, "ultra_defensive",     (0.05, 0.15, 0.05, 0.15, 0.30, 0.20, 0.10)),
    # --- Risk-on ---
    _cfg(16, "risk_on_momentum",    (0.20, 0.10, 0.35, 0.10, 0.03, 0.02, 0.20)),
    _cfg(17, "risk_on_crypto",      (0.15, 0.10, 0.15, 0.10, 0.05, 0.05, 0.40)),
    _cfg(18, "risk_on_aggressive",  (0.25, 0.05, 0.30, 0.10, 0.03, 0.02, 0.25)),
    # --- Balanced variations ---
    _cfg(19, "balanced_equal",      (0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16)),
    _cfg(20, "balanced_no_geo",     (0.25, 0.25, 0.25, 0.15, 0.02, 0.02, 0.06)),
    _cfg(21, "balanced_quant",      (0.20, 0.30, 0.25, 0.05, 0.05, 0.05, 0.10)),
    # --- Crypto-heavy ---
    _cfg(22, "crypto_heavy",        (0.15, 0.10, 0.15, 0.10, 0.05, 0.05, 0.40)),
    _cfg(23, "crypto_momentum",     (0.10, 0.05, 0.25, 0.10, 0.05, 0.05, 0.40)),
    _cfg(24, "crypto_tech",         (0.25, 0.05, 0.10, 0.10, 0.05, 0.05, 0.40)),
    # --- Bull regime ---
    _cfg(25, "bull_momentum",       (0.20, 0.05, 0.40, 0.15, 0.03, 0.02, 0.15)),
    _cfg(26, "bull_tech",           (0.35, 0.10, 0.25, 0.10, 0.05, 0.05, 0.10)),
    _cfg(27, "bull_crypto",         (0.20, 0.05, 0.25, 0.10, 0.05, 0.05, 0.30)),
    # --- Bear regime ---
    _cfg(28, "bear_defensive",      (0.10, 0.15, 0.05, 0.25, 0.25, 0.10, 0.10)),
    _cfg(29, "bear_stat",           (0.10, 0.35, 0.05, 0.20, 0.15, 0.10, 0.05)),
    _cfg(30, "bear_geo",            (0.10, 0.10, 0.05, 0.20, 0.30, 0.20, 0.05)),
    # --- Transition regime ---
    _cfg(31, "transition_balanced", (0.18, 0.22, 0.18, 0.18, 0.08, 0.08, 0.08)),
    _cfg(32, "transition_stat",     (0.15, 0.30, 0.15, 0.15, 0.10, 0.05, 0.10)),
    _cfg(33, "transition_cautious", (0.15, 0.25, 0.10, 0.20, 0.15, 0.05, 0.10)),
    # --- Recovery regime ---
    _cfg(34, "recovery_momentum",   (0.20, 0.10, 0.35, 0.15, 0.05, 0.05, 0.10)),
    _cfg(35, "recovery_tech",       (0.30, 0.10, 0.25, 0.15, 0.05, 0.05, 0.10)),
    _cfg(36, "recovery_balanced",   (0.22, 0.18, 0.22, 0.18, 0.05, 0.05, 0.10)),
    # --- Extreme configurations ---
    _cfg(37, "pure_quant",          (0.30, 0.40, 0.20, 0.02, 0.02, 0.02, 0.04)),
    _cfg(38, "pure_discretionary",  (0.05, 0.05, 0.05, 0.40, 0.20, 0.15, 0.10)),
    _cfg(39, "min_variance",        (0.20, 0.25, 0.15, 0.10, 0.10, 0.10, 0.10)),
    # --- Combinations ---
    _cfg(40, "tech_stat_equal",     (0.30, 0.30, 0.10, 0.10, 0.05, 0.05, 0.10)),
    _cfg(41, "momentum_sentiment",  (0.10, 0.10, 0.30, 0.30, 0.05, 0.05, 0.10)),
    _cfg(42, "crypto_sentiment",    (0.10, 0.10, 0.10, 0.25, 0.05, 0.05, 0.35)),
    _cfg(43, "tech_momentum_equal", (0.30, 0.10, 0.30, 0.10, 0.05, 0.05, 0.10)),
    _cfg(44, "stat_momentum",       (0.10, 0.30, 0.30, 0.10, 0.05, 0.05, 0.10)),
    # --- Adaptive tilt ---
    _cfg(45, "high_vol_tilt",       (0.10, 0.20, 0.10, 0.20, 0.20, 0.10, 0.10)),
    _cfg(46, "low_vol_tilt",        (0.25, 0.20, 0.25, 0.10, 0.05, 0.05, 0.10)),
    _cfg(47, "crisis_tilt",         (0.05, 0.10, 0.05, 0.15, 0.35, 0.25, 0.05)),
    _cfg(48, "expansion_tilt",      (0.20, 0.10, 0.30, 0.10, 0.05, 0.05, 0.20)),
    _cfg(49, "contraction_tilt",    (0.10, 0.25, 0.05, 0.20, 0.20, 0.10, 0.10)),
]
# fmt: on

WEIGHT_CONFIG_MAP: dict[int, WeightConfig] = {wc.config_id: wc for wc in WEIGHT_CONFIGS}
