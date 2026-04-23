"""Observation builder for the Aegis 2.0 RL executor — frozen ``obs_v1`` schema.

Plan §3.4 + §6 guardrail #12: the observation layout is **frozen**. Any
change to ``OBS_V1_DIM`` or ``OBS_V1_FEATURE_NAMES`` invalidates every
trained policy — so the schema reserves slots for providers that aren't
wired yet (regime, macro, geopolitical, crypto). Those default to 0.0
and fill in later without bumping the schema version.

Layout (42 slots)::

    0..4   thesis       : dir_long, dir_short, dir_flat, conviction, signal_age
    5..8   regime       : risk_on, risk_off, neutral, volatility   (provider)
    9..18  macro        : yield_10y, yield_2y, spread, vix,
                          dxy, fed_funds, cpi_yoy, m2_yoy,
                          epu, gold                                (provider)
    19..21 geopolitical : severity, velocity, category             (provider)
    22..25 crypto       : btc_dominance, fear_greed, funding,
                          liquidations                             (provider)
    26..33 reserved     : future providers (zero-filled for now)
    34..39 portfolio    : exposure, unrealized_r, bars_since_entry,
                          drawdown, capital_fraction, positions_frac
    40..41 meta         : day_of_week_sin, quarter_end_flag

Guardrails (§6 #12, training-only):
  * feature dropout — zero out a random subset to simulate provider gaps
  * Gaussian noise — σ×z perturbation to harden against data drift
Never applied at inference; `apply_guardrails(obs, training=False, ...)`
is the identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from aegis.common.types import MarketDataPoint, ThesisSignal


# ---------------------------------------------------------------------------
# Frozen schema — DO NOT EDIT after policies are trained.
# ---------------------------------------------------------------------------

OBS_V1_FEATURE_NAMES: tuple[str, ...] = (
    # 0..4 thesis
    "dir_long",
    "dir_short",
    "dir_flat",
    "conviction",
    "signal_age_norm",
    # 5..8 regime
    "regime_risk_on",
    "regime_risk_off",
    "regime_neutral",
    "regime_volatility",
    # 9..18 macro
    "yield_10y",
    "yield_2y",
    "yield_spread",
    "vix",
    "dxy",
    "fed_funds",
    "cpi_yoy",
    "m2_yoy",
    "epu",
    "gold",
    # 19..21 geopolitical
    "geo_severity",
    "geo_velocity",
    "geo_category",
    # 22..25 crypto
    "btc_dominance",
    "fear_greed",
    "funding_rate",
    "liquidations",
    # 26..33 reserved (future providers)
    "reserved_0",
    "reserved_1",
    "reserved_2",
    "reserved_3",
    "reserved_4",
    "reserved_5",
    "reserved_6",
    "reserved_7",
    # 34..39 portfolio
    "portfolio_exposure",
    "portfolio_unrealized_r",
    "portfolio_bars_since_entry",
    "portfolio_drawdown",
    "portfolio_capital_fraction",
    "portfolio_positions_fraction",
    # 40..41 meta
    "meta_day_of_week_sin",
    "meta_quarter_end_flag",
)

OBS_V1_DIM: int = len(OBS_V1_FEATURE_NAMES)

# Slot ranges for provider-backed blocks — used by overrides.
_REGIME_SLOT = OBS_V1_FEATURE_NAMES.index("regime_risk_on")
_MACRO_SLOT = OBS_V1_FEATURE_NAMES.index("yield_10y")
_GEO_SLOT = OBS_V1_FEATURE_NAMES.index("geo_severity")
_CRYPTO_SLOT = OBS_V1_FEATURE_NAMES.index("btc_dominance")

# Max signal age used for normalization (weeks). Beyond this, clipped to 1.0.
_SIGNAL_AGE_CAP_BARS: float = 52.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Minimal portfolio view the RL executor sees at each step."""

    equity: float
    initial_capital: float
    exposure_notional: float
    unrealized_pnl: float
    initial_risk: float  # dollar risk at entry (1R)
    bars_since_entry: int
    max_drawdown_pct: float  # negative or zero, percentage (e.g. -5.0)
    open_positions_count: int
    max_positions: int


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class ObservationBuilder:
    """Assemble the 42-dim ``obs_v1`` feature vector.

    Parameters
    ----------
    dropout_rate : float
        Probability that each non-one-hot feature is zeroed during
        training-time guardrails. Ignored at inference.
    noise_sigma : float
        Stddev of additive Gaussian noise during training-time
        guardrails. Ignored at inference.
    """

    def __init__(self, dropout_rate: float = 0.2, noise_sigma: float = 0.05) -> None:
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0.0, 1.0)")
        if noise_sigma < 0.0:
            raise ValueError("noise_sigma must be >= 0")
        self._dropout_rate = dropout_rate
        self._noise_sigma = noise_sigma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        thesis: ThesisSignal,
        signal_age_bars: int,
        candles: Sequence[MarketDataPoint],
        portfolio: PortfolioSnapshot,
        regime_snapshot: Mapping[str, float] | None = None,
        macro_snapshot: Mapping[str, float] | None = None,
        geo_snapshot: Mapping[str, float] | None = None,
        crypto_snapshot: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Return the 42-dim observation as ``float32``."""

        obs = np.zeros(OBS_V1_DIM, dtype=np.float32)

        # Thesis block (0..4).
        self._fill_thesis(obs, thesis, signal_age_bars)

        # Provider blocks (5..33) — zero by default, override by name.
        if regime_snapshot:
            self._override_by_name(obs, regime_snapshot)
        if macro_snapshot:
            self._override_by_name(obs, macro_snapshot)
        if geo_snapshot:
            self._override_by_name(obs, geo_snapshot)
        if crypto_snapshot:
            self._override_by_name(obs, crypto_snapshot)

        # Portfolio block (34..39).
        self._fill_portfolio(obs, portfolio)

        # Meta block (40..41) — sourced from last candle's timestamp.
        if candles:
            self._fill_meta(obs, candles[-1])

        # Defence in depth: any NaN/inf from providers becomes 0.
        np.nan_to_num(obs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        return obs

    def apply_guardrails(
        self,
        obs: np.ndarray,
        *,
        training: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply training-only feature dropout + Gaussian noise.

        At inference (``training=False``) this is the identity — the
        policy sees the real observation.
        """
        if not training:
            return obs

        guarded = obs.copy()

        # Dropout on non-one-hot features (spare direction slots 0..2).
        if self._dropout_rate > 0.0:
            mask = rng.random(OBS_V1_DIM) < self._dropout_rate
            mask[0:3] = False  # preserve direction one-hot
            guarded[mask] = 0.0

        # Gaussian noise on all continuous features (also spare one-hot).
        if self._noise_sigma > 0.0:
            noise = rng.normal(0.0, self._noise_sigma, OBS_V1_DIM).astype(np.float32)
            noise[0:3] = 0.0
            guarded = guarded + noise

        return guarded.astype(obs.dtype, copy=False)

    # ------------------------------------------------------------------
    # Block fillers
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_thesis(
        obs: np.ndarray, thesis: ThesisSignal, signal_age_bars: int
    ) -> None:
        direction = thesis.direction.lower()
        if direction == "long":
            obs[0] = 1.0
        elif direction == "short":
            obs[1] = 1.0
        else:  # flat / unknown
            obs[2] = 1.0
        obs[3] = float(thesis.conviction)
        age = max(0, int(signal_age_bars))
        obs[4] = min(age / _SIGNAL_AGE_CAP_BARS, 1.0)

    @staticmethod
    def _fill_portfolio(obs: np.ndarray, portfolio: PortfolioSnapshot) -> None:
        equity = portfolio.equity if portfolio.equity > 0 else 1.0
        initial_capital = (
            portfolio.initial_capital if portfolio.initial_capital > 0 else 1.0
        )
        initial_risk = portfolio.initial_risk if portfolio.initial_risk > 0 else 1.0
        max_positions = portfolio.max_positions if portfolio.max_positions > 0 else 1

        # 34 exposure: notional / equity
        obs[34] = float(portfolio.exposure_notional) / equity
        # 35 unrealized_r: pnl in R units
        obs[35] = float(portfolio.unrealized_pnl) / initial_risk
        # 36 bars since entry, scaled to weeks
        obs[36] = min(float(portfolio.bars_since_entry) / _SIGNAL_AGE_CAP_BARS, 1.0)
        # 37 drawdown (already negative pct; convert to [-1, 0])
        obs[37] = float(portfolio.max_drawdown_pct) / 100.0
        # 38 capital fraction: equity vs starting
        obs[38] = float(portfolio.equity) / initial_capital
        # 39 open positions as fraction of max
        obs[39] = float(portfolio.open_positions_count) / max_positions

    @staticmethod
    def _fill_meta(obs: np.ndarray, last_candle: MarketDataPoint) -> None:
        ts = last_candle.timestamp
        # day of week: Mon=0..Sun=6; encoded as sin(2πk/7) for cyclic continuity.
        dow = ts.weekday()
        obs[40] = math.sin(2.0 * math.pi * dow / 7.0)
        # quarter end = last month of a quarter (3,6,9,12)
        obs[41] = 1.0 if ts.month in (3, 6, 9, 12) else 0.0

    @staticmethod
    def _override_by_name(obs: np.ndarray, snapshot: Mapping[str, float]) -> None:
        for name, value in snapshot.items():
            try:
                idx = OBS_V1_FEATURE_NAMES.index(name)
            except ValueError:
                # Unknown key — silently skip. Providers may send extras.
                continue
            obs[idx] = float(value)
