"""Tests for backtest report."""

import json
import os
import tempfile

from aegis.backtest.report import print_report, save_report


def _sample_results():
    return {
        "equity_curve": [5000, 5100, 5050, 5200],
        "trades": [
            {"symbol": "BTC/USDT", "net_pnl": 100},
            {"symbol": "BTC/USDT", "net_pnl": -50},
        ],
        "metrics": {
            "sharpe": 1.5,
            "max_drawdown": -0.05,
            "win_rate": 0.5,
            "profit_factor": 2.0,
            "total_trades": 2,
            "final_equity": 5200,
            "total_return_pct": 4.0,
        },
    }


class TestReport:
    def test_print_report(self, capsys):
        print_report(_sample_results())
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out
        assert "$5,200.00" in captured.out
        assert "1.500" in captured.out

    def test_save_report(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_report(_sample_results(), path)
            with open(path) as f:
                data = json.load(f)
            assert data["metrics"]["sharpe"] == 1.5
            assert len(data["equity_curve"]) == 4
        finally:
            os.unlink(path)
