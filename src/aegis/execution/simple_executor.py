"""Aegis 2.0 SimpleExecutor — Phase 2.3 baseline.

Floor executor that consumes :class:`ThesisSignal` and decides entries,
exits, and sizing. Deliberately simple — this is the performance floor
the Phase 3 RL executor must beat.

Rules
-----
- Sizing: ``1/N`` of equity per position, where ``N = max_positions``.
- Stop: entry_price ± ``2 * ATR``.
- Target: entry_price ± ``4 * ATR``.
- Exit triggers (in priority order):
    1. Intra-bar stop (``low <= stop`` for longs, ``high >= stop`` shorts).
    2. Intra-bar target (``high >= target`` longs, ``low <= target`` shorts).
    3. Opposing directional signal.
    4. Flat signal (if ``close_on_flat=True``).
- Entry: only when no position is open for the symbol AND open position
  count < ``max_positions`` AND signal direction is ``long`` (or ``short``
  if ``allow_short=True``).
- Same-direction signal on an open position: no-op (don't pyramid).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Mapping

from aegis.common.types import MarketDataPoint, ThesisSignal

ActionKind = Literal["open", "close", "noop"]
ExitReason = Literal["stop", "target", "opposing_signal", "flat_signal"]


def _signal_to_direction(signal: str) -> int:
    if signal == "long":
        return +1
    if signal == "short":
        return -1
    return 0


@dataclass(frozen=True)
class ExecutorPosition:
    """Open position tracked by :class:`SimpleExecutor`.

    Immutable by design — the executor emits new actions rather than
    mutating existing positions.
    """

    symbol: str
    direction: int  # +1 long, -1 short
    quantity: float
    entry_price: float
    stop_price: float
    target_price: float
    notional: float
    entry_timestamp: datetime


@dataclass(frozen=True)
class ExecutionAction:
    """Instruction emitted by :meth:`SimpleExecutor.step`.

    The backtest / live engine materializes these into orders and state
    transitions. ``kind`` determines which fields are meaningful.
    """

    kind: ActionKind
    symbol: str
    timestamp: datetime
    # open-only fields
    direction: int = 0
    notional: float = 0.0
    quantity: float = 0.0
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    # close-only fields
    exit_reason: ExitReason | None = None
    exit_price: float = 0.0
    # provenance
    metadata: dict = field(default_factory=dict)


class SimpleExecutor:
    """Baseline executor: 1/N sizing, ATR stop/target, no pyramiding."""

    def __init__(
        self,
        max_positions: int,
        *,
        allow_short: bool = False,
        close_on_flat: bool = False,
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 4.0,
    ) -> None:
        if max_positions < 1:
            raise ValueError("max_positions must be >= 1")
        if atr_stop_mult <= 0 or atr_target_mult <= 0:
            raise ValueError("ATR multipliers must be positive")
        self._max_positions = max_positions
        self._allow_short = allow_short
        self._close_on_flat = close_on_flat
        self._atr_stop_mult = atr_stop_mult
        self._atr_target_mult = atr_target_mult

    # ---------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------

    def step(
        self,
        *,
        thesis: ThesisSignal,
        candle: MarketDataPoint,
        atr: float,
        equity: float,
        open_positions: Mapping[str, ExecutorPosition],
    ) -> list[ExecutionAction]:
        """Decide actions for ``thesis.symbol`` at ``candle.timestamp``.

        Priority: stop/target exits first (intra-bar), then signal-driven
        exits, then new-entry logic.
        """
        symbol = thesis.symbol
        pos = open_positions.get(symbol)

        # 1) check stop / target exits first — these dominate any signal.
        if pos is not None:
            barrier_close = self._check_barriers(pos, candle)
            if barrier_close is not None:
                return [barrier_close]

        # 2) translate the signal into a raw direction (-1/0/+1) and an
        #    "openable" direction (same but masked by allow_short).
        raw = _signal_to_direction(thesis.direction)
        openable = raw if (raw != -1 or self._allow_short) else 0

        # 3) signal-driven exits on an existing position.
        if pos is not None:
            if raw != 0 and raw == -pos.direction:
                return [self._close(pos, candle, "opposing_signal", candle.close)]
            if thesis.direction == "flat" and self._close_on_flat:
                return [self._close(pos, candle, "flat_signal", candle.close)]
            # same direction or flat-but-don't-close → noop
            return [ExecutionAction(kind="noop", symbol=symbol, timestamp=candle.timestamp)]

        # 4) no position — maybe open one.
        if openable == 0:
            return [ExecutionAction(kind="noop", symbol=symbol, timestamp=candle.timestamp)]

        if len(open_positions) >= self._max_positions:
            return [ExecutionAction(kind="noop", symbol=symbol, timestamp=candle.timestamp)]

        return [self._open(thesis, candle, atr, equity, openable)]

    # ---------------------------------------------------------------
    # internals
    # ---------------------------------------------------------------

    def _check_barriers(
        self,
        pos: ExecutorPosition,
        candle: MarketDataPoint,
    ) -> ExecutionAction | None:
        if pos.direction == +1:
            if candle.low <= pos.stop_price:
                return self._close(pos, candle, "stop", pos.stop_price)
            if candle.high >= pos.target_price:
                return self._close(pos, candle, "target", pos.target_price)
        else:  # short
            if candle.high >= pos.stop_price:
                return self._close(pos, candle, "stop", pos.stop_price)
            if candle.low <= pos.target_price:
                return self._close(pos, candle, "target", pos.target_price)
        return None

    def _open(
        self,
        thesis: ThesisSignal,
        candle: MarketDataPoint,
        atr: float,
        equity: float,
        direction: int,
    ) -> ExecutionAction:
        entry_price = candle.close
        notional = equity / self._max_positions
        quantity = notional / entry_price if entry_price > 0 else 0.0
        stop_offset = self._atr_stop_mult * atr
        target_offset = self._atr_target_mult * atr
        if direction == +1:
            stop_price = entry_price - stop_offset
            target_price = entry_price + target_offset
        else:
            stop_price = entry_price + stop_offset
            target_price = entry_price - target_offset
        return ExecutionAction(
            kind="open",
            symbol=thesis.symbol,
            timestamp=candle.timestamp,
            direction=direction,
            notional=notional,
            quantity=quantity,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            metadata={"conviction": thesis.conviction},
        )

    def _close(
        self,
        pos: ExecutorPosition,
        candle: MarketDataPoint,
        reason: ExitReason,
        exit_price: float,
    ) -> ExecutionAction:
        return ExecutionAction(
            kind="close",
            symbol=pos.symbol,
            timestamp=candle.timestamp,
            direction=pos.direction,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_reason=reason,
            exit_price=exit_price,
        )
