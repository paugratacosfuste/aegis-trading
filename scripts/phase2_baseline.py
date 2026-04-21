"""Phase 2.4 driver — thesis voter + SimpleExecutor on weekly crypto.

Runs the floor RL must beat: the validated direction-voting agents
aggregated by :class:`ThesisVoter`, executed by :class:`SimpleExecutor`
(1/N sizing, 2×ATR stop, 4×ATR target, close on stop/target/opposing).

Window: 2021-01-01 → 2025-12-31, 1w timeframe, 5 crypto symbols.

Applies plan §2.5 gate — if overall Sharpe < 0, the thesis has no edge
and the project should stop here. The gate is printed as a warning;
deciding what to do with that information is Pau's call.

Usage:
    python scripts/phase2_baseline.py
    python scripts/phase2_baseline.py --out reports/phase2_baseline.json
    python scripts/phase2_baseline.py --interval 1w --start 2021-01-01
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from aegis.agents.factory import create_agents_from_config
from aegis.backtest.data_loader import download_from_binance
from aegis.common.types import (
    AgentSignal,
    MarketDataPoint,
    ThesisSignal,
)
from aegis.ensemble.thesis_voter import ThesisVoter
from aegis.execution.simple_executor import (
    ExecutionAction,
    ExecutorPosition,
    SimpleExecutor,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("phase2_baseline")


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
DEFAULT_START = "2021-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_INTERVAL = "1w"
DEFAULT_CONFIG = "configs/agents_validated.yaml"

# Plan §3: 3 concurrent positions is the risk-layer cap, matches SimpleExecutor default.
MAX_POSITIONS = 3
# Plan §1.2: 10 bps per side (exchange fee floor).
COMMISSION_PCT = 0.001
ATR_PERIOD = 14
MIN_WARMUP_BARS = 20
THESIS_VOTER_THRESHOLD = 0.2  # matches ThesisVoter default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atr(candles: list[MarketDataPoint], period: int = ATR_PERIOD) -> float | None:
    """Classic ATR using the last ``period`` true ranges."""
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
    for v in equity_curve:
        peak = max(peak, v)
        dd = (v - peak) / peak if peak > 0 else 0.0
        worst = min(worst, dd)
    return worst * 100.0


def _annualized_sharpe(returns: list[float], periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    mean = statistics.fmean(returns)
    std = statistics.pstdev(returns)
    if std < 1e-12:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class HodlBenchmark:
    """Equal-weight buy-and-hold reference — the real bar to clear."""

    total_return_pct: float
    overall_sharpe: float
    max_drawdown_pct: float
    per_symbol_return_pct: dict[str, float]


@dataclass
class BaselineResult:
    window: dict[str, str]
    symbols: list[str]
    initial_capital: float
    final_equity: float
    total_return_pct: float
    overall_sharpe: float
    max_drawdown_pct: float
    trade_count: int
    per_year_sharpe: dict[int, float]
    per_year_return: dict[int, float]
    exit_reason_counts: dict[str, int]
    gate_pass: bool
    gate_rule: str
    hodl: HodlBenchmark | None = None
    trades: list[dict[str, Any]] = field(default_factory=list)


def _download_all(
    start: str, end: str, interval: str
) -> dict[str, list[MarketDataPoint]]:
    out: dict[str, list[MarketDataPoint]] = {}
    for sym in SYMBOLS:
        t0 = time.time()
        candles = download_from_binance(
            symbol=sym, interval=interval, start_str=start, end_str=end
        )
        logger.info(
            "  %s: %d candles (%.1fs)", sym, len(candles), time.time() - t0
        )
        out[sym.replace("USDT", "/USDT")] = candles
    return out


def _load_voting_agents(config_path: Path) -> tuple[list[Any], dict[str, str]]:
    """Build agent instances for direction-voting types only.

    The validated config also lists macro/geopolitical agents — those emit
    regime features (via metadata) and are filtered out by ThesisVoter,
    but we skip them at construction time too so we don't pay for their
    provider wiring in Phase 2 (that's Phase 3's concern).
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    agents_cfg = raw["agents"]
    voting_types = {"technical", "statistical", "momentum"}
    filtered = {k: v for k, v in agents_cfg.items() if k in voting_types}
    agents = create_agents_from_config(filtered)
    id_to_type: dict[str, str] = {}
    for t, defs in filtered.items():
        for d in defs:
            id_to_type[d["id"]] = t
    return agents, id_to_type


def _collect_signals(
    agents: list[Any],
    symbol: str,
    window: list[MarketDataPoint],
) -> list[AgentSignal]:
    signals: list[AgentSignal] = []
    for agent in agents:
        try:
            sig = agent.generate_signal(symbol, window)
        except Exception:  # noqa: BLE001
            continue
        if sig is not None:
            signals.append(sig)
    return signals


def run_baseline(
    agents: list[Any],
    candles_by_symbol: dict[str, list[MarketDataPoint]],
    *,
    initial_capital: float,
) -> BaselineResult:
    """Run thesis voter + SimpleExecutor across all symbols/bars."""
    voter = ThesisVoter(threshold=THESIS_VOTER_THRESHOLD)
    executor = SimpleExecutor(
        max_positions=MAX_POSITIONS,
        allow_short=False,
        close_on_flat=False,
    )

    equity = initial_capital
    equity_curve: list[tuple[datetime, float]] = []
    open_positions: dict[str, ExecutorPosition] = {}
    trades: list[dict[str, Any]] = []
    exit_reasons: dict[str, int] = {}

    events: list[tuple[datetime, str, int]] = []
    for symbol, cs in candles_by_symbol.items():
        for i, c in enumerate(cs):
            events.append((c.timestamp, symbol, i))
    events.sort(key=lambda e: (e[0], e[1]))

    per_symbol = {s: list(cs) for s, cs in candles_by_symbol.items()}

    last_ts_logged: datetime | None = None

    for ts, symbol, idx in events:
        candle = per_symbol[symbol][idx]
        window = per_symbol[symbol][: idx + 1]
        if len(window) < MIN_WARMUP_BARS:
            continue

        atr_val = _atr(window, ATR_PERIOD)
        if atr_val is None or atr_val <= 0:
            continue

        signals = _collect_signals(agents, symbol, window)
        thesis = voter.vote(symbol, candle.timestamp, signals)

        actions = executor.step(
            thesis=thesis,
            candle=candle,
            atr=atr_val,
            equity=equity,
            open_positions=open_positions,
        )

        for action in actions:
            equity, open_positions = _apply_action(
                action, candle, equity, open_positions, trades, exit_reasons
            )

        equity_curve.append((ts, equity))

        # periodic progress log (weekly data, so not too chatty)
        if last_ts_logged is None or (ts.year != last_ts_logged.year):
            logger.info(
                "  %s equity=%.2f open=%d trades=%d",
                ts.date(), equity, len(open_positions), len(trades)
            )
            last_ts_logged = ts

    # Close any stragglers at last close per symbol.
    for symbol, pos in list(open_positions.items()):
        last_candle = per_symbol[symbol][-1]
        realized = _realize_close(pos, last_candle.close)
        equity += realized
        trades.append(
            _trade_record(pos, last_candle.timestamp, last_candle.close,
                          "end_of_data", realized)
        )
        exit_reasons["end_of_data"] = exit_reasons.get("end_of_data", 0) + 1
        del open_positions[symbol]

    equity_curve.append((events[-1][0] if events else datetime.now(), equity))

    per_year_sharpe, per_year_return = _yearly_stats(equity_curve)
    overall_sharpe = statistics.fmean(per_year_sharpe.values()) if per_year_sharpe else 0.0
    max_dd = _max_drawdown_pct([eq for _, eq in equity_curve])
    total_return_pct = (equity - initial_capital) / initial_capital * 100.0

    gate_rule = "plan §2.5: overall Sharpe > 0 (else thesis has no edge)"
    gate_pass = overall_sharpe > 0.0

    return BaselineResult(
        window={},
        symbols=list(candles_by_symbol.keys()),
        initial_capital=initial_capital,
        final_equity=equity,
        total_return_pct=total_return_pct,
        overall_sharpe=overall_sharpe,
        max_drawdown_pct=max_dd,
        trade_count=len(trades),
        per_year_sharpe=per_year_sharpe,
        per_year_return=per_year_return,
        exit_reason_counts=exit_reasons,
        gate_pass=gate_pass,
        gate_rule=gate_rule,
        trades=trades,
    )


def _apply_action(
    action: ExecutionAction,
    candle: MarketDataPoint,
    equity: float,
    open_positions: dict[str, ExecutorPosition],
    trades: list[dict[str, Any]],
    exit_reasons: dict[str, int],
) -> tuple[float, dict[str, ExecutorPosition]]:
    if action.kind == "noop":
        return equity, open_positions

    if action.kind == "open":
        entry_commission = action.notional * COMMISSION_PCT
        equity_after = equity - entry_commission
        pos = ExecutorPosition(
            symbol=action.symbol,
            direction=action.direction,
            quantity=action.quantity,
            entry_price=action.entry_price,
            stop_price=action.stop_price,
            target_price=action.target_price,
            notional=action.notional,
            entry_timestamp=action.timestamp,
        )
        new_positions = dict(open_positions)
        new_positions[action.symbol] = pos
        return equity_after, new_positions

    if action.kind == "close":
        pos = open_positions.get(action.symbol)
        if pos is None:
            return equity, open_positions
        realized = _realize_close(pos, action.exit_price)
        equity_after = equity + realized
        reason = action.exit_reason or "unknown"
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        trades.append(
            _trade_record(pos, action.timestamp, action.exit_price, reason, realized)
        )
        new_positions = dict(open_positions)
        del new_positions[action.symbol]
        return equity_after, new_positions

    return equity, open_positions


def _realize_close(pos: ExecutorPosition, exit_price: float) -> float:
    """Gross PnL minus exit commission."""
    gross = (exit_price - pos.entry_price) * pos.quantity * pos.direction
    exit_commission = exit_price * pos.quantity * COMMISSION_PCT
    return gross - exit_commission


def _trade_record(
    pos: ExecutorPosition,
    exit_timestamp: datetime,
    exit_price: float,
    reason: str,
    realized: float,
) -> dict[str, Any]:
    return {
        "symbol": pos.symbol,
        "direction": pos.direction,
        "entry_timestamp": pos.entry_timestamp.isoformat(),
        "entry_price": pos.entry_price,
        "exit_timestamp": exit_timestamp.isoformat(),
        "exit_price": exit_price,
        "exit_reason": reason,
        "notional": pos.notional,
        "realized_pnl": realized,
    }


def compute_hodl_benchmark(
    candles_by_symbol: dict[str, list[MarketDataPoint]],
) -> HodlBenchmark:
    """Equal-weight buy-and-hold over the same window (no rebalancing, no fees).

    Normalizes each symbol to 1.0 at its first close, averages across
    symbols per bar, then computes total return, max DD, and an
    annualized Sharpe on weekly returns (52 periods/year).
    """
    if not candles_by_symbol:
        return HodlBenchmark(0.0, 0.0, 0.0, {})

    first_closes = {s: c[0].close for s, c in candles_by_symbol.items()}
    n_symbols = len(candles_by_symbol)
    # Assume all symbols share the same bar timeline (they do for Binance 1w).
    length = min(len(c) for c in candles_by_symbol.values())
    curve: list[float] = []
    for i in range(length):
        val = sum(
            candles_by_symbol[s][i].close / first_closes[s]
            for s in candles_by_symbol
        ) / n_symbols
        curve.append(val)

    total_return_pct = (curve[-1] / curve[0] - 1.0) * 100.0
    max_dd = _max_drawdown_pct(curve)

    weekly_returns: list[float] = []
    for i in range(1, len(curve)):
        if curve[i - 1] > 0:
            weekly_returns.append((curve[i] - curve[i - 1]) / curve[i - 1])
    sharpe = _annualized_sharpe(weekly_returns, periods_per_year=52)

    per_symbol_return = {
        s: (c[-1].close / c[0].close - 1.0) * 100.0
        for s, c in candles_by_symbol.items()
    }
    return HodlBenchmark(
        total_return_pct=total_return_pct,
        overall_sharpe=sharpe,
        max_drawdown_pct=max_dd,
        per_symbol_return_pct=per_symbol_return,
    )


def _yearly_stats(
    equity_curve: list[tuple[datetime, float]],
) -> tuple[dict[int, float], dict[int, float]]:
    by_year: dict[int, list[float]] = {}
    for ts, eq in equity_curve:
        by_year.setdefault(ts.year, []).append(eq)
    per_year_sharpe: dict[int, float] = {}
    per_year_return: dict[int, float] = {}
    for year, equities in by_year.items():
        if len(equities) < 2:
            continue
        returns: list[float] = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                returns.append((equities[i] - equities[i - 1]) / equities[i - 1])
        # Weekly bars → 52 periods per year.
        per_year_sharpe[year] = round(_annualized_sharpe(returns, periods_per_year=52), 3)
        per_year_return[year] = round(
            (equities[-1] - equities[0]) / equities[0] * 100.0 if equities[0] > 0 else 0.0,
            2,
        )
    return per_year_sharpe, per_year_return


def _result_to_dict(res: BaselineResult) -> dict[str, Any]:
    return {
        "window": res.window,
        "symbols": res.symbols,
        "initial_capital": res.initial_capital,
        "final_equity": round(res.final_equity, 2),
        "total_return_pct": round(res.total_return_pct, 2),
        "overall_sharpe": round(res.overall_sharpe, 3),
        "max_drawdown_pct": round(res.max_drawdown_pct, 2),
        "trade_count": res.trade_count,
        "per_year_sharpe": {str(k): v for k, v in res.per_year_sharpe.items()},
        "per_year_return": {str(k): v for k, v in res.per_year_return.items()},
        "exit_reason_counts": res.exit_reason_counts,
        "gate": {
            "rule": res.gate_rule,
            "pass": res.gate_pass,
        },
        "hodl_benchmark": (
            {
                "total_return_pct": round(res.hodl.total_return_pct, 2),
                "overall_sharpe": round(res.hodl.overall_sharpe, 3),
                "max_drawdown_pct": round(res.hodl.max_drawdown_pct, 2),
                "per_symbol_return_pct": {
                    s: round(v, 2) for s, v in res.hodl.per_symbol_return_pct.items()
                },
                "thesis_vs_hodl_return_ratio": round(
                    res.total_return_pct / res.hodl.total_return_pct, 3
                ) if res.hodl.total_return_pct != 0 else None,
                "thesis_vs_hodl_sharpe_ratio": round(
                    res.overall_sharpe / res.hodl.overall_sharpe, 3
                ) if res.hodl.overall_sharpe != 0 else None,
            }
            if res.hodl is not None
            else None
        ),
        "config": {
            "max_positions": MAX_POSITIONS,
            "commission_pct": COMMISSION_PCT,
            "atr_period": ATR_PERIOD,
            "atr_stop_mult": 2.0,
            "atr_target_mult": 4.0,
            "thesis_voter_threshold": THESIS_VOTER_THRESHOLD,
            "min_warmup_bars": MIN_WARMUP_BARS,
        },
        "trades": res.trades,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 thesis-baseline driver")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--out", default="reports/phase2_baseline.json")
    args = parser.parse_args()

    logger.info("Loading voting agents from %s..", args.config)
    agents, id_to_type = _load_voting_agents(Path(args.config))
    logger.info("  %d voting agents: %s", len(agents),
                [a.agent_id for a in agents])

    logger.info("Downloading %s OHLCV for %s..", args.interval, SYMBOLS)
    candles_by_symbol = _download_all(args.start, args.end, args.interval)

    logger.info("Running thesis-baseline backtest..")
    t0 = time.time()
    result = run_baseline(
        agents=agents,
        candles_by_symbol=candles_by_symbol,
        initial_capital=args.initial_capital,
    )
    result.window = {"start": args.start, "end": args.end, "interval": args.interval}
    result.hodl = compute_hodl_benchmark(candles_by_symbol)
    logger.info("  done (%.1fs)", time.time() - t0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_result_to_dict(result), f, indent=2, default=str)
    logger.info("Wrote report to %s", out_path)

    _print_summary(result, id_to_type, len(agents))
    return 0


def _print_summary(res: BaselineResult, id_to_type: dict[str, str], n_agents: int) -> None:
    print()
    print("=" * 72)
    print("PHASE 2 THESIS-BASELINE RESULTS")
    print("=" * 72)
    print(f"Window:         {res.window['start']} → {res.window['end']} ({res.window['interval']})")
    print(f"Symbols:        {', '.join(res.symbols)}")
    print(f"Voting agents:  {n_agents}  ({', '.join(sorted(id_to_type))})")
    print("-" * 72)
    print(f"Initial capital:  ${res.initial_capital:>12,.2f}")
    print(f"Final equity:     ${res.final_equity:>12,.2f}")
    print(f"Total return:     {res.total_return_pct:>+11.2f}%")
    print(f"Overall Sharpe:   {res.overall_sharpe:>+12.3f}  (mean of yearly)")
    print(f"Max drawdown:     {res.max_drawdown_pct:>+11.2f}%")
    print(f"Trades:           {res.trade_count:>12d}")
    print("-" * 72)
    print("Per-year:")
    for year in sorted(res.per_year_sharpe):
        sh = res.per_year_sharpe[year]
        ret = res.per_year_return.get(year, 0.0)
        print(f"  {year}:  Sharpe {sh:+.2f}   return {ret:+.1f}%")
    print("-" * 72)
    print("Exit reasons:")
    for reason, count in sorted(res.exit_reason_counts.items(),
                                 key=lambda kv: -kv[1]):
        print(f"  {reason:<18s} {count:>4d}")
    print("-" * 72)
    if res.hodl is not None:
        h = res.hodl
        print("HODL benchmark (equal-weight, no rebalancing, no fees):")
        print(f"  Return:   {h.total_return_pct:>+11.1f}%   (thesis {res.total_return_pct:+.1f}%)")
        print(f"  Sharpe:   {h.overall_sharpe:>+11.3f}   (thesis {res.overall_sharpe:+.3f})")
        print(f"  Max DD:   {h.max_drawdown_pct:>+11.1f}%   (thesis {res.max_drawdown_pct:+.1f}%)")
        if h.total_return_pct > 0:
            ratio = res.total_return_pct / h.total_return_pct
            print(f"  Thesis captured {ratio*100:.1f}% of HODL return "
                  f"at {abs(res.max_drawdown_pct / h.max_drawdown_pct)*100:.0f}% of HODL drawdown.")
        print("-" * 72)
    if res.gate_pass:
        print(f"GATE PASS ✓  ({res.gate_rule})")
        print("→ Thesis layer has positive edge. Proceed to Phase 3 (RL executor).")
    else:
        print(f"GATE FAIL ✗  ({res.gate_rule})")
        print("→ Thesis has no edge. Plan §2.5 says stop and ship buy-and-hold.")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
