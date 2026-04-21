"""Phase 1.2/1.3 driver: solo-backtest every direction-voting agent.

Runs each agent in isolation across the crypto universe for 2021–2025 daily,
applies the edge filter from plan §1.3, and emits a JSON report.

Usage:
    python scripts/solo_backtest.py
    python scripts/solo_backtest.py --out reports/solo_backtest.json
    python scripts/solo_backtest.py --start 2021-01-01 --end 2025-12-31

Edge filter (plan §1.3):
    OOS Sharpe > 0.3 in >= 2 of 5 years AND overall OOS Sharpe > 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from aegis.agents.factory import create_agents_from_config
from aegis.backtest.data_loader import download_from_binance
from aegis.backtest.solo_engine import SoloBacktestEngine, SoloResult
from aegis.common.types import MarketDataPoint

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("solo_backtest")


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
DEFAULT_START = "2021-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_INTERVAL = "1d"

# Types that emit direction votes (solo-backtest targets).
# macro/geopolitical emit regime via metadata, not direction — skip.
# sentiment/world_leader hit null providers in backtest — skip.
# fundamental is equity-only — skip in crypto solo-backtest.
VOTING_TYPES = {"technical", "statistical", "momentum", "crypto"}

# Within crypto, only crypto_technical uses OHLCV. The others
# (funding_reversal, dominance, fear_greed_crypto, liquidations, defi_tvl)
# rely on data sources we haven't wired for backtest.
CRYPTO_VOTING_STRATEGIES = {"crypto_technical"}


def _load_agent_config(config_path: Path) -> dict[str, list[dict[str, Any]]]:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return raw["agents"]


def _filter_agents(
    agents_cfg: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep only direction-voting agents whose data sources are wired."""
    filtered: dict[str, list[dict[str, Any]]] = {}
    for agent_type, defs in agents_cfg.items():
        if agent_type not in VOTING_TYPES:
            continue
        kept = []
        for d in defs:
            if agent_type == "crypto" and d["strategy"] not in CRYPTO_VOTING_STRATEGIES:
                continue
            kept.append(d)
        if kept:
            filtered[agent_type] = kept
    return filtered


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
        # Data loader returns symbol as e.g. BTC/USDT but we keyed SYMBOLS as
        # BTCUSDT. Normalize to the slash form for the engine output.
        out[sym.replace("USDT", "/USDT")] = candles
    return out


def _agent_passes_edge_filter(
    per_year_sharpe: dict[int, float], overall_sharpe: float
) -> bool:
    years_above_threshold = sum(
        1 for s in per_year_sharpe.values() if s > 0.3
    )
    return years_above_threshold >= 2 and overall_sharpe > 0


def _overall_sharpe_from_yearly(per_year_sharpe: dict[int, float]) -> float:
    """Naive aggregate: mean of yearly Sharpes. Good enough to rank agents."""
    if not per_year_sharpe:
        return 0.0
    return statistics.fmean(per_year_sharpe.values())


def _result_to_dict(res: SoloResult, agent_type: str, strategy: str,
                    params: dict) -> dict[str, Any]:
    return {
        "agent_id": res.agent_id,
        "agent_type": agent_type,
        "strategy": strategy,
        "params": params,
        "initial_capital": res.initial_capital,
        "final_equity": round(res.final_equity, 2),
        "total_return_pct": round(res.total_return_pct, 2),
        "trade_count": res.trade_count,
        "per_year_sharpe": {str(k): round(v, 3) for k, v in res.per_year_sharpe.items()},
        "per_year_return": {str(k): round(v, 2) for k, v in res.per_year_return.items()},
        "max_drawdown_pct": round(res.max_drawdown_pct, 2),
        "overall_sharpe": round(
            _overall_sharpe_from_yearly(res.per_year_sharpe), 3
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 solo backtest harness")
    parser.add_argument(
        "--config",
        default="configs/backtest.yaml",
        help="Source config with agent list",
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument(
        "--out",
        default="reports/solo_backtest.json",
        help="Output JSON report",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
    )
    parser.add_argument(
        "--fixed-risk-pct",
        type=float,
        default=0.10,
        help="Fraction of equity per signal (plan §1.2 = 0.10)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()

    agents_cfg = _load_agent_config(Path(args.config))
    filtered = _filter_agents(agents_cfg)
    total = sum(len(v) for v in filtered.values())
    logger.info(
        "Solo-backtesting %d agents across %d families",
        total,
        len(filtered),
    )

    logger.info("[1/3] Downloading OHLCV for %s..", SYMBOLS)
    candles_by_symbol = _download_all(args.start, args.end, args.interval)
    total_bars = sum(len(v) for v in candles_by_symbol.values())
    logger.info("  %d total bars across %d symbols", total_bars, len(candles_by_symbol))

    agents = create_agents_from_config(filtered)
    # Map agent_id -> (type, strategy, params) for the report.
    lookup: dict[str, tuple[str, str, dict]] = {}
    for agent_type, defs in filtered.items():
        for d in defs:
            lookup[d["id"]] = (agent_type, d["strategy"], d.get("params", {}))

    logger.info("[2/3] Running solo backtest on %d agents..", len(agents))
    per_agent_reports: list[dict[str, Any]] = []
    for agent in agents:
        t0 = time.time()
        engine = SoloBacktestEngine(
            agent=agent,
            initial_capital=args.initial_capital,
            fixed_risk_pct=args.fixed_risk_pct,
            confidence_threshold=args.confidence_threshold,
        )
        try:
            res = engine.run(candles_by_symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  %s FAILED: %s", agent.agent_id, exc)
            continue
        agent_type, strategy, params = lookup[agent.agent_id]
        rep = _result_to_dict(res, agent_type, strategy, params)
        per_agent_reports.append(rep)
        logger.info(
            "  %-12s  Sharpe=%+.2f  ret=%+.1f%%  trades=%d  dd=%.1f%%  (%.1fs)",
            agent.agent_id,
            rep["overall_sharpe"],
            rep["total_return_pct"],
            rep["trade_count"],
            rep["max_drawdown_pct"],
            time.time() - t0,
        )

    # Rank and apply edge filter.
    per_agent_reports.sort(key=lambda r: r["overall_sharpe"], reverse=True)
    survivors = [
        r
        for r in per_agent_reports
        if _agent_passes_edge_filter(
            {int(k): v for k, v in r["per_year_sharpe"].items()},
            r["overall_sharpe"],
        )
    ]
    deadlist = [r for r in per_agent_reports if r not in survivors]

    logger.info(
        "[3/3] Edge filter: %d survivors / %d total (deadlist: %d)",
        len(survivors),
        len(per_agent_reports),
        len(deadlist),
    )

    report = {
        "window": {"start": args.start, "end": args.end, "interval": args.interval},
        "symbols": SYMBOLS,
        "initial_capital": args.initial_capital,
        "fixed_risk_pct": args.fixed_risk_pct,
        "confidence_threshold": args.confidence_threshold,
        "edge_filter": {
            "rule": "OOS Sharpe > 0.3 in >=2 years AND overall mean Sharpe > 0",
            "survivor_count": len(survivors),
            "deadlist_count": len(deadlist),
        },
        "survivors": survivors,
        "deadlist": deadlist,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote report to %s", out_path)

    print()
    print("=" * 70)
    print(f"{'agent':<14} {'type':<12} {'Sharpe':>7} {'ret %':>8} {'trades':>7} {'DD %':>7}")
    print("-" * 70)
    for r in per_agent_reports:
        mark = "OK" if r in survivors else "x"
        print(
            f"{r['agent_id']:<14} {r['agent_type']:<12} "
            f"{r['overall_sharpe']:>+7.2f} "
            f"{r['total_return_pct']:>+8.1f} "
            f"{r['trade_count']:>7d} "
            f"{r['max_drawdown_pct']:>+7.1f}  {mark}"
        )
    print("=" * 70)
    print(
        f"Survivors: {len(survivors)}/{len(per_agent_reports)}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
