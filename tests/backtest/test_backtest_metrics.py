"""Tests for backtest metrics. Written FIRST per TDD."""

import pytest

from aegis.backtest.metrics import (
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe,
    calculate_win_rate,
)


class TestSharpeRatio:
    def test_positive_returns(self):
        # Steady 1% daily returns for 252 days -> high Sharpe
        returns = [0.01] * 252
        sharpe = calculate_sharpe(returns)
        assert sharpe > 5.0  # Very high for consistent returns

    def test_zero_returns(self):
        returns = [0.0] * 100
        assert calculate_sharpe(returns) == 0.0

    def test_negative_returns(self):
        returns = [-0.01] * 252
        sharpe = calculate_sharpe(returns)
        assert sharpe < -5.0

    def test_mixed_returns(self):
        returns = [0.02, -0.01, 0.015, -0.005, 0.01]
        sharpe = calculate_sharpe(returns)
        assert isinstance(sharpe, float)

    def test_empty_returns(self):
        assert calculate_sharpe([]) == 0.0

    def test_single_return(self):
        assert calculate_sharpe([0.01]) == 0.0  # Can't compute std with 1 point


class TestMaxDrawdown:
    def test_no_drawdown(self):
        equity = [100, 110, 120, 130, 140]
        assert calculate_max_drawdown(equity) == 0.0

    def test_simple_drawdown(self):
        equity = [100, 110, 90, 95, 100]
        dd = calculate_max_drawdown(equity)
        assert dd == pytest.approx(90 / 110 - 1, abs=0.001)  # -18.18%

    def test_multiple_drawdowns(self):
        equity = [100, 120, 100, 130, 90]
        dd = calculate_max_drawdown(equity)
        assert dd == pytest.approx(90 / 130 - 1, abs=0.001)  # -30.77%

    def test_empty_equity(self):
        assert calculate_max_drawdown([]) == 0.0

    def test_single_point(self):
        assert calculate_max_drawdown([100]) == 0.0


class TestWinRate:
    def test_all_wins(self):
        pnls = [10.0, 20.0, 5.0]
        assert calculate_win_rate(pnls) == pytest.approx(1.0)

    def test_all_losses(self):
        pnls = [-10.0, -20.0, -5.0]
        assert calculate_win_rate(pnls) == pytest.approx(0.0)

    def test_mixed(self):
        pnls = [10.0, -5.0, 20.0, -15.0]
        assert calculate_win_rate(pnls) == pytest.approx(0.5)

    def test_empty(self):
        assert calculate_win_rate([]) == 0.0

    def test_breakeven_counted_as_win(self):
        pnls = [0.0, 10.0, -5.0]
        assert calculate_win_rate(pnls) == pytest.approx(2 / 3, abs=0.01)


class TestProfitFactor:
    def test_positive_factor(self):
        pnls = [20.0, -10.0, 15.0, -5.0]
        pf = calculate_profit_factor(pnls)
        assert pf == pytest.approx(35.0 / 15.0, abs=0.01)

    def test_no_losses(self):
        pnls = [10.0, 20.0]
        assert calculate_profit_factor(pnls) == float("inf")

    def test_no_wins(self):
        pnls = [-10.0, -20.0]
        assert calculate_profit_factor(pnls) == 0.0

    def test_empty(self):
        assert calculate_profit_factor([]) == 0.0
