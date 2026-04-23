"""Unit tests for the Aegis 2.0 TradingEnv — Phase 3.5.

The env composes the three Phase-3 primitives:
  * ObservationBuilder (obs_v1)
  * action decoder (Tuple(Discrete(5), Box(1,4)))
  * DifferentialSharpeReward

For Phase 3 the env is deliberately minimal: single symbol, at most one
open position, thesis signal pre-aligned with candles. PPO training,
multi-symbol, and walk-forward plumbing belong to Phase 4.

These tests pin the Gym contract and make sure the env is deterministic
given a seed (plan §3.6 acceptance criterion).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import gymnasium as gym

from aegis.common.types import MarketDataPoint, ThesisSignal
from aegis.rl.envs import OBS_VERSION
from aegis.rl.envs.observation import OBS_V1_DIM
from aegis.rl.envs.trading_env import TradingEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _candles(n: int, *, seed: int = 17) -> list[MarketDataPoint]:
    """Stable geometric noise candles — enough bars for ATR to warm up."""
    rng = random.Random(seed)
    start_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = 100.0
    out = []
    for i in range(n):
        price *= 1.0 + rng.gauss(0.001, 0.01)
        out.append(
            MarketDataPoint(
                symbol="BTC/USDT",
                asset_class="crypto",
                timestamp=start_ts + timedelta(days=7 * i),
                timeframe="1w",
                open=price * 0.999,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000.0,
                source="binance",
            )
        )
    return out


def _theses(candles: list[MarketDataPoint], *, direction: str = "long") -> list[ThesisSignal]:
    return [
        ThesisSignal(
            symbol=c.symbol,
            timestamp=c.timestamp,
            direction=direction,
            conviction=0.7,
            contributing_agents=("tech_09",),
            metadata={},
        )
        for c in candles
    ]


def _env(**overrides) -> TradingEnv:
    candles = overrides.pop("candles", None) or _candles(80)
    theses = overrides.pop("theses", None) or _theses(candles, direction="long")
    return TradingEnv(
        candles=candles,
        theses=theses,
        initial_capital=overrides.pop("initial_capital", 10_000.0),
        commission_pct=overrides.pop("commission_pct", 0.001),
        training=overrides.pop("training", False),
        seed=overrides.pop("seed", 0),
        **overrides,
    )


# ---------------------------------------------------------------------------
# Gym contract
# ---------------------------------------------------------------------------


def test_env_is_gymnasium_env():
    env = _env()
    assert isinstance(env, gym.Env)


def test_observation_space_matches_obs_v1():
    env = _env()
    assert env.observation_space.shape == (OBS_V1_DIM,)
    assert env.observation_space.dtype == np.float32


def test_action_space_is_tuple_of_discrete_and_box():
    env = _env()
    assert isinstance(env.action_space, gym.spaces.Tuple)
    discrete, box = env.action_space.spaces
    assert isinstance(discrete, gym.spaces.Discrete)
    assert discrete.n == 5
    assert isinstance(box, gym.spaces.Box)
    assert box.low[0] == pytest.approx(1.0)
    assert box.high[0] == pytest.approx(4.0)


def test_reset_returns_obs_and_info():
    env = _env()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_V1_DIM,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)
    # Plan §6 guardrail #12: schema version must be exposed.
    assert info.get("obs_version") == OBS_VERSION


def test_step_returns_five_tuple():
    env = _env()
    env.reset(seed=0)
    out = env.step((0, np.array([2.0], dtype=np.float32)))
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_V1_DIM,)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_before_reset_raises():
    env = _env()
    with pytest.raises(RuntimeError):
        env.step((0, np.array([2.0], dtype=np.float32)))


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------


def test_episode_terminates_at_end_of_candles():
    candles = _candles(40)
    env = _env(candles=candles, theses=_theses(candles))
    env.reset(seed=0)
    terminated = False
    steps = 0
    while not terminated and steps < 200:
        _, _, terminated, truncated, _ = env.step((0, np.array([2.0], dtype=np.float32)))
        steps += 1
        if truncated:
            break
    assert terminated is True
    # Should not take more bars than candles provides after warmup.
    assert steps <= len(candles)


# ---------------------------------------------------------------------------
# Determinism (plan §3.6 acceptance criterion)
# ---------------------------------------------------------------------------


def test_reset_is_deterministic_given_seed():
    env = _env()
    obs_a, _ = env.reset(seed=123)
    obs_b, _ = env.reset(seed=123)
    np.testing.assert_array_equal(obs_a, obs_b)


def test_rollout_is_deterministic_given_seed():
    # Two identical envs with the same seed and the same scripted actions
    # must produce identical trajectories.
    def run(env):
        obs, _ = env.reset(seed=7)
        rewards = []
        for i in range(30):
            action = ((i % 5), np.array([2.0], dtype=np.float32))
            obs, r, term, trunc, _ = env.step(action)
            rewards.append(r)
            if term or trunc:
                break
        return rewards

    r1 = run(_env())
    r2 = run(_env())
    assert r1 == r2


# ---------------------------------------------------------------------------
# Veto behavior — bucket 0 should not open a trade.
# ---------------------------------------------------------------------------


def test_veto_action_does_not_open_trade():
    env = _env()
    env.reset(seed=0)
    # Always veto.
    info_final = {}
    for _ in range(20):
        _, _, term, trunc, info_final = env.step((0, np.array([2.0], dtype=np.float32)))
        if term or trunc:
            break
    assert info_final["trades_opened"] == 0


def test_non_veto_action_opens_trade_when_thesis_non_flat():
    candles = _candles(80)
    env = _env(candles=candles, theses=_theses(candles, direction="long"))
    env.reset(seed=0)
    # Force a real trade on the first openable bar.
    info_final = {}
    for _ in range(len(candles)):
        _, _, term, trunc, info_final = env.step((4, np.array([2.0], dtype=np.float32)))
        if term or trunc:
            break
    assert info_final["trades_opened"] >= 1


# ---------------------------------------------------------------------------
# Training-time guardrails
# ---------------------------------------------------------------------------


def test_training_mode_applies_guardrails_to_observations():
    env_eval = _env(training=False, seed=0)
    env_train = _env(training=True, seed=0)
    obs_eval, _ = env_eval.reset(seed=0)
    obs_train, _ = env_train.reset(seed=0)
    # With dropout + noise, training obs should differ from eval obs.
    assert not np.array_equal(obs_eval, obs_train)
    # Schema dim preserved.
    assert obs_train.shape == obs_eval.shape
