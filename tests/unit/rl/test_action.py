"""Unit tests for the RL hybrid action decoder — Aegis 2.0 Phase 3.3.

Action space (plan §3.3):
    Tuple(
        Discrete(5),                       # size bucket 0..4
        Box(low=1.0, high=4.0, shape=(1,)) # stop distance in ATR multiples
    )

Size bucket semantics: ``(0.0, 0.25, 0.5, 0.75, 1.0)`` of the per-trade
risk budget. Bucket ``0`` is the **veto** — RL overrides the thesis and
skips the trade. This is a critical capability.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegis.rl.envs.action import (
    SIZE_BUCKETS,
    RLAction,
    build_action_space,
    decode_action,
)


# ---------------------------------------------------------------------------
# Action space shape
# ---------------------------------------------------------------------------


def test_build_action_space_is_tuple_of_discrete_and_box():
    import gymnasium as gym

    space = build_action_space()
    assert isinstance(space, gym.spaces.Tuple)
    discrete, box = space.spaces
    assert isinstance(discrete, gym.spaces.Discrete)
    assert discrete.n == 5
    assert isinstance(box, gym.spaces.Box)
    assert box.low.shape == (1,)
    assert box.low[0] == pytest.approx(1.0)
    assert box.high[0] == pytest.approx(4.0)


def test_action_space_sample_is_decodable():
    # Sanity: SB3 will sample from the space; decode_action must accept it.
    space = build_action_space()
    space.seed(0)
    raw = space.sample()
    action = decode_action(raw)
    assert 0.0 <= action.size_fraction <= 1.0
    assert 1.0 <= action.stop_atr_mult <= 4.0


# ---------------------------------------------------------------------------
# decode_action: size bucket semantics
# ---------------------------------------------------------------------------


def test_bucket_0_is_veto():
    action = decode_action((0, np.array([2.0], dtype=np.float32)))
    assert action.is_veto is True
    assert action.size_fraction == 0.0


def test_bucket_4_is_full_size():
    action = decode_action((4, np.array([3.0], dtype=np.float32)))
    assert action.size_fraction == pytest.approx(1.0)
    assert action.is_veto is False


def test_mid_buckets_map_to_quarter_steps():
    assert decode_action((1, np.array([2.0], dtype=np.float32))).size_fraction == pytest.approx(0.25)
    assert decode_action((2, np.array([2.0], dtype=np.float32))).size_fraction == pytest.approx(0.50)
    assert decode_action((3, np.array([2.0], dtype=np.float32))).size_fraction == pytest.approx(0.75)


def test_invalid_bucket_raises():
    with pytest.raises(ValueError):
        decode_action((-1, np.array([2.0], dtype=np.float32)))
    with pytest.raises(ValueError):
        decode_action((5, np.array([2.0], dtype=np.float32)))


# ---------------------------------------------------------------------------
# Stop multiplier handling
# ---------------------------------------------------------------------------


def test_stop_multiplier_passes_through_in_range():
    a = decode_action((2, np.array([2.5], dtype=np.float32)))
    assert a.stop_atr_mult == pytest.approx(2.5, rel=1e-6)


def test_stop_multiplier_clipped_below_minimum():
    # PPO can propose actions slightly outside the Box range before
    # clipping; decoder must defend against this.
    a = decode_action((2, np.array([0.2], dtype=np.float32)))
    assert a.stop_atr_mult == pytest.approx(1.0)


def test_stop_multiplier_clipped_above_maximum():
    a = decode_action((2, np.array([7.0], dtype=np.float32)))
    assert a.stop_atr_mult == pytest.approx(4.0)


def test_stop_multiplier_accepts_scalar_float():
    # Some callers will hand us a plain float. Decoder must handle it.
    a = decode_action((2, 2.5))
    assert a.stop_atr_mult == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# SIZE_BUCKETS constant
# ---------------------------------------------------------------------------


def test_size_buckets_are_canonical_quarter_steps():
    assert SIZE_BUCKETS == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_rl_action_is_frozen_dataclass():
    a = RLAction(size_fraction=0.5, stop_atr_mult=2.0, is_veto=False)
    with pytest.raises(Exception):
        a.size_fraction = 0.1  # type: ignore[misc]
