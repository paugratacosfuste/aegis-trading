"""Unit tests for the RL observation builder — Aegis 2.0 Phase 3.4.

``obs_v1`` is a frozen 42-feature schema. Any change invalidates all
trained policies (plan §6 guardrail #12). Features not wired to real
providers yet (macro / geopolitical / crypto / regime) default to 0.0
so the shape stays stable; they'll fill in when providers land.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from aegis.common.types import MarketDataPoint, ThesisSignal
from aegis.rl.envs import OBS_VERSION
from aegis.rl.envs.observation import (
    OBS_V1_DIM,
    OBS_V1_FEATURE_NAMES,
    ObservationBuilder,
    PortfolioSnapshot,
)


def _candles(n: int, start_price: float = 100.0, *,
             start_ts: datetime | None = None) -> list[MarketDataPoint]:
    """Geometric-noise price series — just enough for indicators to run."""
    import random
    rng = random.Random(17)
    if start_ts is None:
        start_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = start_price
    out = []
    for i in range(n):
        price *= 1.0 + rng.gauss(0.001, 0.01)
        high = price * 1.01
        low = price * 0.99
        out.append(
            MarketDataPoint(
                symbol="BTC/USDT",
                asset_class="crypto",
                timestamp=start_ts + timedelta(days=7 * i),
                timeframe="1w",
                open=price * 0.999,
                high=high,
                low=low,
                close=price,
                volume=1000.0,
                source="binance",
            )
        )
    return out


def _thesis(direction: str = "long", conviction: float = 0.7) -> ThesisSignal:
    return ThesisSignal(
        symbol="BTC/USDT",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        direction=direction,
        conviction=conviction,
        contributing_agents=("tech_09",),
        metadata={},
    )


def _portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        equity=10_000.0,
        initial_capital=10_000.0,
        exposure_notional=0.0,
        unrealized_pnl=0.0,
        initial_risk=1.0,
        bars_since_entry=0,
        max_drawdown_pct=0.0,
        open_positions_count=0,
        max_positions=3,
    )


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


def test_obs_version_is_v1():
    assert OBS_VERSION == "obs_v1"


def test_obs_v1_feature_names_length_matches_dim():
    assert len(OBS_V1_FEATURE_NAMES) == OBS_V1_DIM


def test_obs_v1_dim_is_42():
    assert OBS_V1_DIM == 42


# ---------------------------------------------------------------------------
# Shape / dtype / finiteness
# ---------------------------------------------------------------------------


def test_build_returns_correct_shape_and_dtype():
    builder = ObservationBuilder()
    obs = builder.build(
        thesis=_thesis(),
        signal_age_bars=0,
        candles=_candles(60),
        portfolio=_portfolio(),
    )
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_V1_DIM,)
    assert obs.dtype == np.float32


def test_all_features_are_finite():
    builder = ObservationBuilder()
    obs = builder.build(
        thesis=_thesis(),
        signal_age_bars=3,
        candles=_candles(60),
        portfolio=_portfolio(),
    )
    assert np.isfinite(obs).all()


# ---------------------------------------------------------------------------
# Thesis section
# ---------------------------------------------------------------------------


def test_thesis_long_one_hot():
    builder = ObservationBuilder()
    obs = builder.build(
        thesis=_thesis("long", 0.6),
        signal_age_bars=0,
        candles=_candles(60),
        portfolio=_portfolio(),
    )
    # First 3 slots = one-hot (long, short, flat); slot 3 = conviction.
    assert obs[0] == 1.0 and obs[1] == 0.0 and obs[2] == 0.0
    assert obs[3] == pytest.approx(0.6)


def test_thesis_short_one_hot():
    builder = ObservationBuilder()
    obs = builder.build(
        thesis=_thesis("short", 0.8),
        signal_age_bars=0,
        candles=_candles(60),
        portfolio=_portfolio(),
    )
    assert obs[0] == 0.0 and obs[1] == 1.0 and obs[2] == 0.0
    assert obs[3] == pytest.approx(0.8)


def test_thesis_flat_one_hot():
    builder = ObservationBuilder()
    obs = builder.build(
        thesis=_thesis("flat", 0.0),
        signal_age_bars=0,
        candles=_candles(60),
        portfolio=_portfolio(),
    )
    assert obs[0] == 0.0 and obs[1] == 0.0 and obs[2] == 1.0


def test_signal_age_clipped_to_unit_interval():
    builder = ObservationBuilder()
    obs = builder.build(
        thesis=_thesis(),
        signal_age_bars=1000,  # very old
        candles=_candles(60),
        portfolio=_portfolio(),
    )
    # Normalized to [0, 1].
    assert 0.0 <= obs[4] <= 1.0


# ---------------------------------------------------------------------------
# Unwired sections (regime/macro/geo/crypto) default to 0
# ---------------------------------------------------------------------------


def test_regime_macro_geo_crypto_default_zero_when_no_provider():
    builder = ObservationBuilder()
    obs = builder.build(
        thesis=_thesis(),
        signal_age_bars=0,
        candles=_candles(60),
        portfolio=_portfolio(),
    )
    # Slots 5..33 = regime(4) + macro(10) + geo(3) + crypto(4) = 21.
    # Layout pinned in OBS_V1_FEATURE_NAMES; assert the whole block is 0.
    assert np.all(obs[5:34] == 0.0)


# ---------------------------------------------------------------------------
# Portfolio section
# ---------------------------------------------------------------------------


def test_portfolio_zero_exposure_when_no_position():
    builder = ObservationBuilder()
    port = _portfolio()
    obs = builder.build(
        thesis=_thesis(),
        signal_age_bars=0,
        candles=_candles(60),
        portfolio=port,
    )
    # Portfolio block is slots 34..39. Exposure is the first portfolio slot.
    assert obs[34] == 0.0


def test_portfolio_exposure_populated_when_in_trade():
    builder = ObservationBuilder()
    port = PortfolioSnapshot(
        equity=10_000.0,
        initial_capital=10_000.0,
        exposure_notional=3_000.0,
        unrealized_pnl=150.0,
        initial_risk=100.0,
        bars_since_entry=3,
        max_drawdown_pct=-5.0,
        open_positions_count=1,
        max_positions=3,
    )
    obs = builder.build(
        thesis=_thesis(),
        signal_age_bars=1,
        candles=_candles(60),
        portfolio=port,
    )
    assert obs[34] == pytest.approx(0.3)  # 3000/10000
    # Slot 39: open_positions_count / max_positions
    assert obs[39] == pytest.approx(1.0 / 3.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Meta section — deterministic math
# ---------------------------------------------------------------------------


def test_meta_day_of_week_sin_populated():
    # Sunday 2024-01-07 → dow=6, sin(2*pi*6/7) ≈ -0.7818
    builder = ObservationBuilder()
    ts = datetime(2024, 1, 7, tzinfo=timezone.utc)
    candles = _candles(60, start_ts=ts)
    obs = builder.build(
        thesis=_thesis(),
        signal_age_bars=0,
        candles=candles,
        portfolio=_portfolio(),
    )
    # Last candle's timestamp is 59 weeks after start — still Sunday.
    expected = math.sin(2 * math.pi * 6 / 7)
    assert obs[40] == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Guardrails: training-only dropout + noise
# ---------------------------------------------------------------------------


def test_guardrails_identity_when_not_training():
    builder = ObservationBuilder()
    rng = np.random.default_rng(0)
    obs = builder.build(
        thesis=_thesis(),
        signal_age_bars=0,
        candles=_candles(60),
        portfolio=_portfolio(),
    )
    guarded = builder.apply_guardrails(obs, training=False, rng=rng)
    np.testing.assert_array_equal(obs, guarded)


def test_guardrails_apply_when_training():
    builder = ObservationBuilder(dropout_rate=0.2, noise_sigma=0.05)
    rng = np.random.default_rng(42)
    obs = builder.build(
        thesis=_thesis(),
        signal_age_bars=0,
        candles=_candles(60),
        portfolio=_portfolio(),
    )
    guarded = builder.apply_guardrails(obs, training=True, rng=rng)
    # Something must have changed (very high probability given seed 42).
    assert not np.array_equal(obs, guarded)
    # Shape and dtype preserved.
    assert guarded.shape == obs.shape
    assert guarded.dtype == obs.dtype


def test_guardrails_deterministic_with_same_seed():
    builder = ObservationBuilder(dropout_rate=0.2, noise_sigma=0.05)
    obs = builder.build(
        thesis=_thesis(),
        signal_age_bars=0,
        candles=_candles(60),
        portfolio=_portfolio(),
    )
    g1 = builder.apply_guardrails(obs, training=True, rng=np.random.default_rng(7))
    g2 = builder.apply_guardrails(obs, training=True, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(g1, g2)


# ---------------------------------------------------------------------------
# Provider overrides
# ---------------------------------------------------------------------------


def test_macro_snapshot_overrides_defaults():
    builder = ObservationBuilder()
    macro = {"vix": 2.5}  # already-normalized value
    obs = builder.build(
        thesis=_thesis(),
        signal_age_bars=0,
        candles=_candles(60),
        portfolio=_portfolio(),
        macro_snapshot=macro,
    )
    vix_idx = OBS_V1_FEATURE_NAMES.index("vix")
    assert obs[vix_idx] == pytest.approx(2.5)
