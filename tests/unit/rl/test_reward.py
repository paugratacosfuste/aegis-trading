"""Unit tests for DifferentialSharpeReward — Aegis 2.0 Phase 3.2.

Implements Moody & Saffell (1998) differential Sharpe ratio as a per-bar
reward signal. Maintains EMA of returns (A) and squared returns (B) with
memory rate ``eta``.

  delta_A = r_t - A_prev
  delta_B = r_t^2 - B_prev
  A_new = A_prev + eta * delta_A
  B_new = B_prev + eta * delta_B
  denom = (B_prev - A_prev^2) ** 1.5
  D_t   = (B_prev * delta_A - 0.5 * A_prev * delta_B) / denom   (0 if denom≈0)

Tests pin the math and confirm that shape_reward() is a passthrough stub
until Pau fills it in (plan §3.4, Appendix B).
"""

from __future__ import annotations

import math

import pytest

from aegis.rl.envs.reward import DifferentialSharpeReward, shape_reward


# ---------------------------------------------------------------------------
# DifferentialSharpeReward
# ---------------------------------------------------------------------------


def test_first_step_returns_zero_reward():
    # A=B=0 at construction, so denom = 0 — the very first bar has no
    # Sharpe information yet. Returning 0 prevents NaN / inf explosions.
    rew = DifferentialSharpeReward(eta=0.01)
    r = rew.step(0.01)
    assert r == 0.0


def test_state_updates_after_first_step():
    rew = DifferentialSharpeReward(eta=0.01)
    rew.step(0.05)
    # A_new = 0 + 0.01 * (0.05 - 0) = 0.0005
    assert rew.ema_return == pytest.approx(0.0005)
    # B_new = 0 + 0.01 * (0.0025 - 0) = 0.000025
    assert rew.ema_squared_return == pytest.approx(0.000025)


def test_constant_returns_keep_differential_near_zero_once_stable():
    # If every bar returns the same r, the Sharpe estimate has no new
    # information and D_t should stay small (not blow up).
    rew = DifferentialSharpeReward(eta=0.1)
    for _ in range(500):
        rew.step(0.01)
    # After convergence, D_t near zero (machine-epsilon or tiny).
    r = rew.step(0.01)
    assert abs(r) < 0.5


def test_upside_surprise_produces_positive_reward():
    # Seed state with a noisy positive-drift return stream (realistic for
    # markets) so variance is non-degenerate, then a continued positive
    # surprise should raise the running Sharpe ⇒ positive differential.
    import random
    rng = random.Random(42)
    rew = DifferentialSharpeReward(eta=0.1)
    for _ in range(200):
        rew.step(rng.gauss(mu=0.005, sigma=0.02))
    # A big positive return on top of noisy drift.
    r_big = rew.step(0.05)
    assert r_big > 0


def test_downside_surprise_produces_negative_reward():
    import random
    rng = random.Random(42)
    rew = DifferentialSharpeReward(eta=0.1)
    for _ in range(200):
        rew.step(rng.gauss(mu=0.005, sigma=0.02))
    r_drop = rew.step(-0.05)
    assert r_drop < 0


def test_explicit_math_matches_formula():
    # Pin one specific step against the hand-computed formula.
    rew = DifferentialSharpeReward(eta=0.1)
    # Seed A_prev = 0.02, B_prev = 0.001 by directly loading state.
    rew._A = 0.02  # noqa: SLF001
    rew._B = 0.001  # noqa: SLF001

    r_t = 0.03
    delta_A = r_t - 0.02
    delta_B = r_t ** 2 - 0.001
    denom = (0.001 - 0.02 ** 2) ** 1.5
    expected = (0.001 * delta_A - 0.5 * 0.02 * delta_B) / denom

    got = rew.step(r_t)
    assert got == pytest.approx(expected, rel=1e-9)


def test_zero_variance_returns_zero_not_nan():
    # When B - A^2 collapses to 0 (degenerate), reward must be 0, not NaN.
    rew = DifferentialSharpeReward(eta=0.1)
    rew._A = 0.02  # noqa: SLF001
    rew._B = 0.02 ** 2  # exact match → denom = 0
    r = rew.step(0.03)
    assert r == 0.0
    assert not math.isnan(r)


def test_reset_clears_state():
    rew = DifferentialSharpeReward(eta=0.01)
    for _ in range(10):
        rew.step(0.02)
    assert rew.ema_return != 0
    rew.reset()
    assert rew.ema_return == 0.0
    assert rew.ema_squared_return == 0.0
    # first step after reset returns 0
    assert rew.step(0.01) == 0.0


def test_invalid_eta_raises():
    with pytest.raises(ValueError):
        DifferentialSharpeReward(eta=0.0)
    with pytest.raises(ValueError):
        DifferentialSharpeReward(eta=1.5)
    with pytest.raises(ValueError):
        DifferentialSharpeReward(eta=-0.1)


# ---------------------------------------------------------------------------
# shape_reward stub (Pau's 10-line contribution slot)
# ---------------------------------------------------------------------------


def test_shape_reward_is_passthrough_by_default():
    # Until Pau fills in the stub, shaping must be identity so that
    # differential Sharpe alone drives learning.
    assert shape_reward(0.0, state={}, action={}, market={}) == 0.0
    assert shape_reward(1.23, state={}, action={}, market={}) == 1.23
    assert shape_reward(-0.5, state={}, action={}, market={}) == -0.5
