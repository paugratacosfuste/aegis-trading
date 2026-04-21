"""SoloBacktestEngine — single-agent edge validation harness.

Phase 1.2 of the Aegis 2.0 plan. Runs one agent in isolation (no ensemble,
no Kelly, no regime gating) with fixed 10% per signal, 2x ATR stop,
4x ATR target. The goal is to isolate each agent's edge so we can keep
only those with individually positive OOS Sharpe.

Not a production engine — intentionally minimal:
  - One open position per symbol at a time.
  - Entry on any bar where direction != 0 and confidence >= threshold.
  - Exit on opposing signal OR stop OR target OR end of data.
  - Commission per side (entry + exit).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from aegis.common.types import AgentSignal, MarketDataPoint


class _AgentLike(Protocol):
    agent_id: str
    agent_type: str

    def generate_signal(
        self, symbol: str, candles: list[MarketDataPoint]
    ) -> AgentSignal: ...


@dataclass
class SoloResult:
    agent_id: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    trade_count: int
    per_year_sharpe: dict[int, float]
    per_year_return: dict[int, float]
    max_drawdown_pct: float
    trades: list[dict[str, Any]] = field(default_factory=list)


def _atr(candles: list[MarketDataPoint], period: int = 14) -> float | None:
    """Classic ATR on the last ``period`` bars."""
    if len(candles) < period + 1:
        return None
    trs: list[float] = []
    for i in range(len(candles) - period, len(candles)):
        c = candles[i]
        prev_close = candles[i - 1].close
        tr = max(
            c.high - c.low,
            abs(c.high - prev_close),
            abs(c.low - prev_close),
        )
        trs.append(tr)
    return sum(trs) / period


def _max_drawdown_pct(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for val in equity_curve:
        peak = max(peak, val)
        dd = (val - peak) / peak if peak > 0 else 0.0
        worst = min(worst, dd)
    return worst * 100.0


def _annualized_sharpe(returns: list[float], periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    mean = statistics.fmean(returns)
    std = statistics.pstdev(returns)
    if std < 1e-12:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


class SoloBacktestEngine:
    """Run a single agent in isolation across one or more symbols."""

    def __init__(
        self,
        agent: _AgentLike,
        initial_capital: float = 10_000.0,
        fixed_risk_pct: float = 0.10,
        confidence_threshold: float = 0.5,
        commission_pct: float = 0.001,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 4.0,
        min_warmup_bars: int = 20,
    ):
        if not 0.0 < fixed_risk_pct <= 1.0:
            raise ValueError("fixed_risk_pct must be in (0, 1]")
        self._agent = agent
        self._initial_capital = initial_capital
        self._risk_pct = fixed_risk_pct
        self._conf_thr = confidence_threshold
        self._commission_pct = commission_pct
        self._atr_period = atr_period
        self._atr_stop_mult = atr_stop_mult
        self._atr_target_mult = atr_target_mult
        self._min_warmup = min_warmup_bars

    def run(self, candles_by_symbol: dict[str, list[MarketDataPoint]]) -> SoloResult:
        """Run the backtest. Returns SoloResult."""
        equity = self._initial_capital
        equity_curve: list[float] = [equity]
        open_positions: dict[str, dict[str, Any]] = {}
        trades: list[dict[str, Any]] = []
        # Daily equity marker for Sharpe (use timestamp date as key)
        daily_equity: dict[datetime, float] = {}

        # Merge candles across symbols into a single time-ordered stream.
        events: list[tuple[datetime, str, int]] = []
        for symbol, cs in candles_by_symbol.items():
            for i, c in enumerate(cs):
                events.append((c.timestamp, symbol, i))
        events.sort(key=lambda e: e[0])

        per_symbol = {s: list(cs) for s, cs in candles_by_symbol.items()}

        for ts, symbol, idx in events:
            candle = per_symbol[symbol][idx]
            window = per_symbol[symbol][: idx + 1]
            if len(window) < self._min_warmup:
                daily_equity[ts] = equity
                continue

            # 1) Check exits on any open position for this symbol.
            pos = open_positions.get(symbol)
            if pos is not None:
                exit_info = self._maybe_exit(pos, candle)
                if exit_info is not None:
                    equity += exit_info["realized_pnl"]
                    trades.append({**pos, **exit_info})
                    del open_positions[symbol]

            # 2) Ask the agent for a signal.
            try:
                sig = self._agent.generate_signal(symbol, window)
            except Exception:
                sig = None

            # 3) Handle entries.
            if sig is not None and symbol not in open_positions:
                if abs(sig.direction) > 1e-9 and sig.confidence >= self._conf_thr:
                    atr_val = _atr(window, self._atr_period)
                    if atr_val is not None and atr_val > 0:
                        pos = self._open_position(
                            symbol=symbol,
                            candle=candle,
                            equity=equity,
                            direction=1 if sig.direction > 0 else -1,
                            atr=atr_val,
                        )
                        # Pay entry commission.
                        equity -= pos["notional"] * self._commission_pct
                        open_positions[symbol] = pos

            equity_curve.append(equity)
            daily_equity[ts] = equity

        # 4) Close any remaining open positions at their last available close.
        for symbol, pos in list(open_positions.items()):
            last_candle = per_symbol[symbol][-1]
            exit_info = self._close_position(pos, last_candle, reason="end_of_data")
            equity += exit_info["realized_pnl"]
            trades.append({**pos, **exit_info})
            del open_positions[symbol]

        equity_curve.append(equity)

        # 5) Per-year Sharpe and return.
        per_year_sharpe, per_year_return = self._yearly_stats(daily_equity)

        total_return_pct = (equity - self._initial_capital) / self._initial_capital * 100.0
        max_dd = _max_drawdown_pct(equity_curve)

        return SoloResult(
            agent_id=self._agent.agent_id,
            initial_capital=self._initial_capital,
            final_equity=equity,
            total_return_pct=total_return_pct,
            trade_count=len(trades),
            per_year_sharpe=per_year_sharpe,
            per_year_return=per_year_return,
            max_drawdown_pct=max_dd,
            trades=trades,
        )

    def _open_position(
        self,
        *,
        symbol: str,
        candle: MarketDataPoint,
        equity: float,
        direction: int,
        atr: float,
    ) -> dict[str, Any]:
        notional = equity * self._risk_pct
        entry_price = candle.close
        quantity = notional / entry_price if entry_price > 0 else 0.0
        stop = (
            entry_price - self._atr_stop_mult * atr
            if direction > 0
            else entry_price + self._atr_stop_mult * atr
        )
        target = (
            entry_price + self._atr_target_mult * atr
            if direction > 0
            else entry_price - self._atr_target_mult * atr
        )
        return {
            "symbol": symbol,
            "entry_timestamp": candle.timestamp,
            "entry_price": entry_price,
            "direction": direction,
            "notional": notional,
            "quantity": quantity,
            "stop_price": stop,
            "target_price": target,
            "atr_at_entry": atr,
        }

    def _maybe_exit(
        self, pos: dict[str, Any], candle: MarketDataPoint
    ) -> dict[str, Any] | None:
        direction = pos["direction"]
        stop = pos["stop_price"]
        target = pos["target_price"]
        # For long: stop if low <= stop, target if high >= target.
        # For short: stop if high >= stop, target if low <= target.
        if direction > 0:
            if candle.low <= stop:
                return self._close_position(pos, candle, reason="stop", override_price=stop)
            if candle.high >= target:
                return self._close_position(pos, candle, reason="target", override_price=target)
        else:
            if candle.high >= stop:
                return self._close_position(pos, candle, reason="stop", override_price=stop)
            if candle.low <= target:
                return self._close_position(pos, candle, reason="target", override_price=target)
        return None

    def _close_position(
        self,
        pos: dict[str, Any],
        candle: MarketDataPoint,
        *,
        reason: str,
        override_price: float | None = None,
    ) -> dict[str, Any]:
        exit_price = override_price if override_price is not None else candle.close
        direction = pos["direction"]
        qty = pos["quantity"]
        gross = (exit_price - pos["entry_price"]) * qty * direction
        exit_notional = exit_price * qty
        commission = exit_notional * self._commission_pct
        return {
            "exit_timestamp": candle.timestamp,
            "exit_price": exit_price,
            "exit_reason": reason,
            "realized_pnl": gross - commission,
        }

    def _yearly_stats(
        self, daily_equity: dict[datetime, float]
    ) -> tuple[dict[int, float], dict[int, float]]:
        if len(daily_equity) < 2:
            return {}, {}

        # Sort by timestamp.
        sorted_points = sorted(daily_equity.items(), key=lambda kv: kv[0])
        # Group per calendar year by taking last value per day.
        by_day: dict[tuple[int, int, int], float] = {}
        for ts, eq in sorted_points:
            by_day[(ts.year, ts.month, ts.day)] = eq
        ordered_days = sorted(by_day.keys())
        by_year_days: dict[int, list[tuple[tuple[int, int, int], float]]] = {}
        for day in ordered_days:
            by_year_days.setdefault(day[0], []).append((day, by_day[day]))

        per_year_sharpe: dict[int, float] = {}
        per_year_return: dict[int, float] = {}
        for year, points in by_year_days.items():
            if len(points) < 2:
                continue
            equities = [p[1] for p in points]
            daily_returns = []
            for i in range(1, len(equities)):
                if equities[i - 1] > 0:
                    daily_returns.append((equities[i] - equities[i - 1]) / equities[i - 1])
            per_year_sharpe[year] = _annualized_sharpe(daily_returns, periods_per_year=252)
            per_year_return[year] = (equities[-1] - equities[0]) / equities[0] * 100.0 if equities[0] > 0 else 0.0

        return per_year_sharpe, per_year_return
