"""Gymnasium environment for the Aegis 2.0 RL executor.

Composes the three Phase-3 primitives:

    obs_v1 ObservationBuilder ─┐
    hybrid action decoder ─────┤──▶ TradingEnv ──▶ PPO
    DifferentialSharpeReward ──┘

The env is deliberately minimal for Phase 3:
  * single symbol
  * at most one open position
  * thesis signal pre-aligned with candles (one per bar)
  * ATR-based stop from the action's ``stop_atr_mult``
  * bar-return based equity accounting with round-trip commission

Walk-forward training, multi-symbol envs, and regime stratification are
Phase 4 concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import gymnasium as gym
from ta.volatility import AverageTrueRange

from aegis.common.types import MarketDataPoint, ThesisSignal

from . import OBS_VERSION
from .action import RLAction, build_action_space, decode_action
from .observation import (
    OBS_V1_DIM,
    ObservationBuilder,
    PortfolioSnapshot,
)
from .reward import DifferentialSharpeReward, shape_reward


_ATR_PERIOD = 14
_MIN_WARMUP_BARS = _ATR_PERIOD + 1


@dataclass
class _OpenPosition:
    direction: int  # +1 long, -1 short
    entry_price: float
    quantity: float
    stop_price: float
    entry_bar: int


class TradingEnv(gym.Env):
    """Single-symbol Gym environment for the Aegis 2.0 RL executor."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        candles: Sequence[MarketDataPoint],
        theses: Sequence[ThesisSignal],
        initial_capital: float = 10_000.0,
        commission_pct: float = 0.001,
        size_risk_fraction: float = 0.02,
        training: bool = False,
        seed: int = 0,
        dropout_rate: float = 0.2,
        noise_sigma: float = 0.05,
        reward_eta: float = 0.01,
    ) -> None:
        super().__init__()
        if len(candles) < _MIN_WARMUP_BARS + 2:
            raise ValueError(
                f"need at least {_MIN_WARMUP_BARS + 2} candles; got {len(candles)}"
            )
        if len(candles) != len(theses):
            raise ValueError("candles and theses must be the same length")
        if initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if not 0.0 <= commission_pct < 1.0:
            raise ValueError("commission_pct must be in [0, 1)")

        self._candles = list(candles)
        self._theses = list(theses)
        self._initial_capital = float(initial_capital)
        self._commission_pct = float(commission_pct)
        self._size_risk_fraction = float(size_risk_fraction)
        self._training = bool(training)

        self._builder = ObservationBuilder(
            dropout_rate=dropout_rate, noise_sigma=noise_sigma
        )
        self._reward_engine = DifferentialSharpeReward(eta=reward_eta)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_V1_DIM,),
            dtype=np.float32,
        )
        self.action_space = build_action_space()

        self._default_seed = seed
        self._rng = np.random.default_rng(seed)
        self._atr_series = self._precompute_atr()

        # Runtime state — populated by reset().
        self._step_idx: int = -1
        self._cash: float = 0.0
        self._peak_equity: float = 0.0
        self._open_position: _OpenPosition | None = None
        self._trades_opened: int = 0
        self._trades_closed: int = 0
        self._reset_called: bool = False

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):  # noqa: D401
        if seed is not None:
            self._default_seed = seed
        self._rng = np.random.default_rng(self._default_seed)
        self._step_idx = _MIN_WARMUP_BARS
        self._cash = self._initial_capital
        self._peak_equity = self._initial_capital
        self._open_position = None
        self._trades_opened = 0
        self._trades_closed = 0
        self._reward_engine.reset()
        self._reset_called = True

        obs = self._build_observation()
        info = {
            "obs_version": OBS_VERSION,
            "equity": self._equity_at(self._candles[self._step_idx].close),
            "step": self._step_idx,
        }
        return obs, info

    def step(self, action):
        if not self._reset_called:
            raise RuntimeError("must call reset() before step()")

        decoded = decode_action(action)
        candle = self._candles[self._step_idx]
        thesis = self._theses[self._step_idx]

        # Snapshot equity at the close of the *current* bar — this is the
        # baseline for the bar return after we advance one step.
        equity_before = self._equity_at(candle.close)

        # Intra-bar stop check first (can only trigger if we already hold).
        if self._open_position is not None:
            self._check_stop_and_exit(candle)

        # Act on the decoded action at this bar's close.
        if self._open_position is None:
            self._maybe_open(decoded, thesis, candle)
        else:
            # RL veto closes an existing position.
            if decoded.is_veto:
                self._close_position(candle.close)

        # Advance one bar; mark-to-market with the next bar's close.
        next_idx = self._step_idx + 1
        terminated = next_idx >= len(self._candles) - 1
        next_price = self._candles[
            min(next_idx, len(self._candles) - 1)
        ].close

        equity_after = self._equity_at(next_price)
        bar_return = (
            (equity_after - equity_before) / equity_before
            if equity_before > 0
            else 0.0
        )

        base_reward = self._reward_engine.step(bar_return)
        reward = shape_reward(base_reward, state={}, action={}, market={})

        self._peak_equity = max(self._peak_equity, equity_after)
        self._step_idx = next_idx

        # Force-close at episode end so final equity is clean.
        if terminated and self._open_position is not None:
            self._close_position(next_price)

        obs = self._build_observation()
        info = {
            "obs_version": OBS_VERSION,
            "equity": self._equity_at(next_price),
            "bar_return": bar_return,
            "trades_opened": self._trades_opened,
            "trades_closed": self._trades_closed,
            "step": self._step_idx,
        }
        return obs, float(reward), bool(terminated), False, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _equity_at(self, mark_price: float) -> float:
        """Mark-to-market equity: cash + unrealized PnL."""
        if self._open_position is None:
            return self._cash
        pos = self._open_position
        unrealized = pos.direction * (mark_price - pos.entry_price) * pos.quantity
        return self._cash + unrealized

    def _build_observation(self) -> np.ndarray:
        # Window ≤ step_idx — never read beyond (plan §6 guardrail #13).
        idx = min(self._step_idx, len(self._candles) - 1)
        window = self._candles[: idx + 1]
        thesis = self._theses[idx]
        portfolio = self._portfolio_snapshot(self._candles[idx].close)

        obs = self._builder.build(
            thesis=thesis,
            signal_age_bars=0,
            candles=window,
            portfolio=portfolio,
        )
        return self._builder.apply_guardrails(
            obs, training=self._training, rng=self._rng
        )

    def _portfolio_snapshot(self, mark_price: float) -> PortfolioSnapshot:
        pos = self._open_position
        equity = self._equity_at(mark_price)
        exposure = 0.0
        unrealized = 0.0
        initial_risk = 1.0
        bars_since = 0
        if pos is not None:
            exposure = pos.quantity * mark_price
            unrealized = pos.direction * (mark_price - pos.entry_price) * pos.quantity
            initial_risk = max(
                1e-8, abs(pos.entry_price - pos.stop_price) * pos.quantity
            )
            bars_since = max(0, self._step_idx - pos.entry_bar)
        dd_pct = 0.0
        if self._peak_equity > 0:
            dd_pct = (equity - self._peak_equity) / self._peak_equity * 100.0

        return PortfolioSnapshot(
            equity=equity,
            initial_capital=self._initial_capital,
            exposure_notional=exposure,
            unrealized_pnl=unrealized,
            initial_risk=initial_risk,
            bars_since_entry=bars_since,
            max_drawdown_pct=min(0.0, dd_pct),
            open_positions_count=1 if pos else 0,
            max_positions=1,
        )

    def _maybe_open(
        self,
        action: RLAction,
        thesis: ThesisSignal,
        candle: MarketDataPoint,
    ) -> None:
        if action.is_veto or action.size_fraction <= 0:
            return
        direction = {"long": 1, "short": -1}.get(thesis.direction.lower(), 0)
        if direction == 0:
            return
        atr = self._atr_series[self._step_idx]
        if not np.isfinite(atr) or atr <= 0:
            return

        risk_budget = self._cash * self._size_risk_fraction
        risk_per_trade = action.size_fraction * risk_budget
        stop_distance = action.stop_atr_mult * atr
        if stop_distance <= 0:
            return
        quantity = risk_per_trade / stop_distance
        if quantity <= 0:
            return

        entry_price = candle.close
        stop_price = entry_price - direction * stop_distance

        self._cash -= self._commission_pct * quantity * entry_price
        self._open_position = _OpenPosition(
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            stop_price=stop_price,
            entry_bar=self._step_idx,
        )
        self._trades_opened += 1

    def _check_stop_and_exit(self, candle: MarketDataPoint) -> None:
        pos = self._open_position
        if pos is None:
            return
        if pos.direction == 1 and candle.low <= pos.stop_price:
            self._close_position(pos.stop_price)
        elif pos.direction == -1 and candle.high >= pos.stop_price:
            self._close_position(pos.stop_price)

    def _close_position(self, exit_price: float) -> None:
        pos = self._open_position
        if pos is None:
            return
        realized = pos.direction * (exit_price - pos.entry_price) * pos.quantity
        self._cash += realized
        self._cash -= self._commission_pct * pos.quantity * exit_price
        self._open_position = None
        self._trades_closed += 1

    def _precompute_atr(self) -> np.ndarray:
        highs = pd.Series([c.high for c in self._candles], dtype=float)
        lows = pd.Series([c.low for c in self._candles], dtype=float)
        closes = pd.Series([c.close for c in self._candles], dtype=float)
        atr = AverageTrueRange(
            high=highs, low=lows, close=closes, window=_ATR_PERIOD, fillna=False
        ).average_true_range()
        return atr.to_numpy(dtype=float)
