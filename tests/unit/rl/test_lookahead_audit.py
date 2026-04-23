"""Look-ahead audit for the RL env — plan §6 guardrail #13.

Promises under test:

  1. ``ObservationBuilder.build`` must only consume candles at or before
     the current step. A peek at ``candles[t+1]`` would be catastrophic
     for training (the policy learns to cheat).
  2. The env never advances more bars than it is supposed to per step.
  3. A 100-step random rollout does not crash, does not produce NaN
     rewards, and does not grow equity to infinity.

Implementation note: we wrap the underlying candle list in a sentinel
list that raises on any access above a recorded high-water mark. Any
accidental forward read surfaces as an ``IndexError`` caught by the
test.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from aegis.common.types import MarketDataPoint, ThesisSignal
from aegis.rl.envs.trading_env import TradingEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _candles(n: int) -> list[MarketDataPoint]:
    rng = random.Random(17)
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


def _theses(candles: list[MarketDataPoint]) -> list[ThesisSignal]:
    return [
        ThesisSignal(
            symbol=c.symbol,
            timestamp=c.timestamp,
            direction="long",
            conviction=0.7,
            contributing_agents=("tech_09",),
            metadata={},
        )
        for c in candles
    ]


class _FutureReadGuard(list):
    """A list that raises on any read past the current high-water mark.

    The env is allowed to read ``candles[0:step+1]``; reading beyond is a
    look-ahead leak. The guard tracks the high-water mark via
    :meth:`set_limit` and inspects every ``__getitem__``.
    """

    def __init__(self, items):
        super().__init__(items)
        self._limit: int | None = None

    def set_limit(self, limit: int) -> None:
        self._limit = limit

    def __getitem__(self, key):  # type: ignore[override]
        limit = self._limit
        if limit is not None:
            if isinstance(key, slice):
                stop = key.stop if key.stop is not None else len(self)
                if stop - 1 > limit:
                    raise AssertionError(
                        f"look-ahead leak: slice up to {stop - 1} > limit {limit}"
                    )
            elif isinstance(key, int):
                idx = key if key >= 0 else len(self) + key
                if idx > limit:
                    raise AssertionError(
                        f"look-ahead leak: index {idx} > limit {limit}"
                    )
        return super().__getitem__(key)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_observation_never_reads_future_candles():
    """Plan §6 guardrail #13 — 100 random bars, no forward reads."""
    candles = _FutureReadGuard(_candles(120))
    theses = _theses(list(candles))
    env = TradingEnv(candles=candles, theses=theses, seed=0)
    env.reset(seed=0)
    # The env exposes _step_idx; on each step we clamp the guard's view
    # to the bar the env thinks it's currently *observing*.
    for _ in range(100):
        candles.set_limit(env._step_idx)  # noqa: SLF001
        action = (random.randrange(5), np.array([2.0], dtype=np.float32))
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break


def test_random_rollout_never_produces_nan_reward():
    env = TradingEnv(
        candles=_candles(200), theses=_theses(_candles(200)), seed=0
    )
    env.reset(seed=0)
    rng = random.Random(42)
    for _ in range(100):
        action = (rng.randrange(5), np.array([rng.uniform(1.0, 4.0)], dtype=np.float32))
        _, reward, term, trunc, _ = env.step(action)
        assert not math.isnan(reward), "reward must never be NaN"
        assert math.isfinite(reward), "reward must stay finite"
        if term or trunc:
            break


def test_random_rollout_equity_stays_finite():
    env = TradingEnv(
        candles=_candles(200), theses=_theses(_candles(200)), seed=0
    )
    env.reset(seed=0)
    rng = random.Random(99)
    info = {}
    for _ in range(150):
        action = (rng.randrange(5), np.array([rng.uniform(1.0, 4.0)], dtype=np.float32))
        _, _, term, trunc, info = env.step(action)
        assert math.isfinite(info["equity"]), "equity blew up to inf"
        assert info["equity"] > 0, f"equity went negative: {info['equity']}"
        if term or trunc:
            break


def test_random_rollout_trades_logged():
    """Sanity: a rollout that isn't all-vetoes should open at least one trade."""
    env = TradingEnv(
        candles=_candles(200), theses=_theses(_candles(200)), seed=0
    )
    env.reset(seed=0)
    rng = random.Random(123)
    info = {}
    for _ in range(200):
        # Avoid the veto bucket half the time so we actually trade.
        bucket = rng.choice([1, 2, 3, 4])
        action = (bucket, np.array([2.0], dtype=np.float32))
        _, _, term, trunc, info = env.step(action)
        if term or trunc:
            break
    assert info["trades_opened"] >= 1


def test_step_index_advances_exactly_one_per_step():
    env = TradingEnv(
        candles=_candles(100), theses=_theses(_candles(100)), seed=0
    )
    env.reset(seed=0)
    previous = env._step_idx  # noqa: SLF001
    for _ in range(30):
        _, _, term, trunc, _ = env.step((0, np.array([2.0], dtype=np.float32)))
        current = env._step_idx  # noqa: SLF001
        assert current - previous == 1, "env must advance exactly one bar per step"
        previous = current
        if term or trunc:
            break
