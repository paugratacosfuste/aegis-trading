"""Backtest report: prints summary and saves JSON."""

import json
import logging

logger = logging.getLogger(__name__)


def print_report(results: dict) -> None:
    """Print backtest results to console."""
    m = results["metrics"]
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
    print("=" * 50 + "\n")


def save_report(results: dict, path: str) -> None:
    """Save results to JSON file."""
    serializable = {
        "metrics": results["metrics"],
        "equity_curve": results["equity_curve"],
        "trades": results["trades"],
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info("Report saved to %s", path)
