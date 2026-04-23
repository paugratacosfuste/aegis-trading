"""RL hybrid action space + decoder — Aegis 2.0 Phase 3.3.

Plan §3.3::

    action_space = Tuple((
        Discrete(5),              # size bucket: (0, 0.25, 0.5, 0.75, 1.0)
        Box(low=1.0, high=4.0)    # stop distance in ATR multiples
    ))

Bucket ``0`` is the **veto**: the RL executor is allowed to override the
thesis layer and skip a trade entirely. This capability is essential —
without it the RL is a slave to the voter, not a judge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

# Canonical sizing grid: fractions of the per-trade risk budget.
SIZE_BUCKETS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

_STOP_MIN = 1.0
_STOP_MAX = 4.0


@dataclass(frozen=True)
class RLAction:
    """Decoded, ready-to-execute RL action."""

    size_fraction: float
    stop_atr_mult: float
    is_veto: bool


def build_action_space() -> spaces.Tuple:
    return spaces.Tuple(
        (
            spaces.Discrete(len(SIZE_BUCKETS)),
            spaces.Box(
                low=np.array([_STOP_MIN], dtype=np.float32),
                high=np.array([_STOP_MAX], dtype=np.float32),
                dtype=np.float32,
            ),
        )
    )


def decode_action(raw_action) -> RLAction:
    """Turn a raw Gym action into an executable RLAction.

    Tolerates scalar floats and 1-D arrays for the stop multiplier; clips
    to ``[1.0, 4.0]`` in case the policy emitted something slightly OOB
    before SB3 clipping.
    """
    size_bucket, stop_raw = raw_action
    size_bucket_int = int(size_bucket)
    if not 0 <= size_bucket_int < len(SIZE_BUCKETS):
        raise ValueError(
            f"size_bucket must be in [0, {len(SIZE_BUCKETS)}); got {size_bucket_int}"
        )

    stop_arr = np.asarray(stop_raw, dtype=np.float64).reshape(-1)
    stop_val = float(stop_arr[0]) if stop_arr.size else _STOP_MIN
    stop_clipped = float(np.clip(stop_val, _STOP_MIN, _STOP_MAX))

    return RLAction(
        size_fraction=SIZE_BUCKETS[size_bucket_int],
        stop_atr_mult=stop_clipped,
        is_veto=(size_bucket_int == 0),
    )
