"""Backtest report: prints summary and saves JSON."""

import json
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def _compute_r_multiples(trades: list[dict]) -> dict:
    """Compute R-multiple statistics from trade list.

    R = risk per trade (distance from entry to stop_loss).
    R-multiple = actual PnL / R.
    """
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]

    avg_win = sum(t["net_pnl"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(t["net_pnl"] for t in losses) / len(losses)) if losses else 0.0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

    # Exit reason distribution
    reasons = Counter(t.get("exit_reason", "unknown") for t in trades)

    # Per-symbol breakdown
    symbols: dict[str, dict] = {}
    for t in trades:
        s = t["symbol"]
        if s not in symbols:
            symbols[s] = {"trades": 0, "wins": 0, "pnl": 0.0}
        symbols[s]["trades"] += 1
        symbols[s]["pnl"] += t["net_pnl"]
        if t["net_pnl"] > 0:
            symbols[s]["wins"] += 1

    return {
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "num_wins": len(wins),
        "num_losses": len(losses),
        "exit_reasons": dict(reasons),
        "per_symbol": symbols,
    }


def print_report(results: dict) -> None:
    """Print backtest results to console."""
    m = results["metrics"]
    trades = results.get("trades", [])

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"  Final equity:    ${m['final_equity']:,.2f}")
    print(f"  Total return:    {m['total_return_pct']:+.2f}%")
    print(f"  Sharpe ratio:    {m['sharpe']:.3f}")
    print(f"  Max drawdown:    {m['max_drawdown']:.2%}")
    print(f"  Win rate:        {m['win_rate']:.2%}")
    print(f"  Profit factor:   {m['profit_factor']:.2f}")
    print(f"  Total trades:    {m['total_trades']}")
    print("=" * 50)

    if trades:
        r = _compute_r_multiples(trades)
        print("\nR-MULTIPLE ANALYSIS")
        print("-" * 50)
        print(f"  Avg win:         ${r['avg_win']:.2f}  ({r['num_wins']} trades)")
        print(f"  Avg loss:        ${r['avg_loss']:.2f}  ({r['num_losses']} trades)")
        print(f"  Avg win/loss:    {r['win_loss_ratio']:.2f}x")
        edge = m["win_rate"] * r["avg_win"] - (1 - m["win_rate"]) * r["avg_loss"]
        print(f"  Expected value:  ${edge:.2f} per trade")

        print("\nEXIT REASONS")
        print("-" * 50)
        for reason, count in sorted(r["exit_reasons"].items(), key=lambda x: -x[1]):
            pct = count / len(trades) * 100
            print(f"  {reason:25s} {count:>4} ({pct:.1f}%)")

        print("\nPER-SYMBOL BREAKDOWN")
        print("-" * 50)
        for sym, stats in sorted(r["per_symbol"].items()):
            wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] else 0
            print(
                f"  {sym:12s} {stats['trades']:>3} trades  "
                f"{stats['wins']}W/{stats['trades'] - stats['wins']}L  "
                f"({wr:.0f}%)  PnL: ${stats['pnl']:+.2f}"
            )

    print()


def save_report(results: dict, path: str) -> None:
    """Save results to JSON file."""
    trades = results.get("trades", [])
    serializable = {
        "metrics": results["metrics"],
        "r_analysis": _compute_r_multiples(trades) if trades else {},
        "equity_curve": results["equity_curve"],
        "trades": trades,
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info("Report saved to %s", path)
