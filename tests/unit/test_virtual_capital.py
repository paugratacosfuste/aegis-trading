"""Tests for virtual capital tracker."""

import pytest

from aegis.lab.virtual_capital import VirtualCapitalTracker


class TestVirtualCapitalTracker:
    def test_initial_state(self):
        vc = VirtualCapitalTracker(100_000.0)
        assert vc.cash == 100_000.0
        assert vc.initial_capital == 100_000.0
        assert vc.get_open_positions() == []
        assert vc.get_closed_pnls() == []

    def test_open_position(self):
        vc = VirtualCapitalTracker(100_000.0)
        pid = vc.open_position("BTC/USDT", "LONG", 1.0, 50000.0)
        assert pid != ""
        assert len(vc.get_open_positions()) == 1
        # Cash reduced by position value + commission
        assert vc.cash < 100_000.0 - 50000.0

    def test_insufficient_capital(self):
        vc = VirtualCapitalTracker(1_000.0)
        pid = vc.open_position("BTC/USDT", "LONG", 1.0, 50000.0)
        assert pid == ""
        assert len(vc.get_open_positions()) == 0

    def test_close_long_profit(self):
        vc = VirtualCapitalTracker(100_000.0)
        pid = vc.open_position("BTC/USDT", "LONG", 1.0, 50000.0)
        pnl = vc.close_position(pid, 55000.0)
        assert pnl > 0  # Profit minus commissions
        assert len(vc.get_open_positions()) == 0
        assert len(vc.get_closed_pnls()) == 1

    def test_close_long_loss(self):
        vc = VirtualCapitalTracker(100_000.0)
        pid = vc.open_position("BTC/USDT", "LONG", 1.0, 50000.0)
        pnl = vc.close_position(pid, 45000.0)
        assert pnl < 0

    def test_close_short_profit(self):
        vc = VirtualCapitalTracker(100_000.0)
        pid = vc.open_position("BTC/USDT", "SHORT", 1.0, 50000.0)
        pnl = vc.close_position(pid, 45000.0)
        assert pnl > 0  # Short profits when price drops

    def test_close_short_loss(self):
        vc = VirtualCapitalTracker(100_000.0)
        pid = vc.open_position("BTC/USDT", "SHORT", 1.0, 50000.0)
        pnl = vc.close_position(pid, 55000.0)
        assert pnl < 0

    def test_close_nonexistent(self):
        vc = VirtualCapitalTracker(100_000.0)
        assert vc.close_position("nonexistent", 50000.0) == 0.0

    def test_equity_with_open_position(self):
        vc = VirtualCapitalTracker(100_000.0)
        vc.open_position("BTC/USDT", "LONG", 1.0, 50000.0)
        # Price went up -> equity > initial
        equity = vc.get_equity({"BTC/USDT": 55000.0})
        assert equity > 100_000.0

    def test_equity_no_positions(self):
        vc = VirtualCapitalTracker(100_000.0)
        assert vc.get_equity({}) == 100_000.0

    def test_equity_curve(self):
        vc = VirtualCapitalTracker(100_000.0)
        assert vc.get_equity_curve() == [100_000.0]
        vc.record_equity_snapshot({"BTC/USDT": 50000.0})
        assert len(vc.get_equity_curve()) == 2

    def test_commission_deducted(self):
        vc = VirtualCapitalTracker(100_000.0, commission_rate=0.01)
        pid = vc.open_position("BTC/USDT", "LONG", 1.0, 10000.0)
        # Entry commission = 10000 * 0.01 = 100
        assert vc.cash == pytest.approx(100_000.0 - 10000.0 - 100.0)

    def test_total_pnl(self):
        vc = VirtualCapitalTracker(100_000.0, commission_rate=0.0)
        pid1 = vc.open_position("BTC/USDT", "LONG", 1.0, 100.0)
        vc.close_position(pid1, 110.0)
        pid2 = vc.open_position("BTC/USDT", "LONG", 1.0, 100.0)
        vc.close_position(pid2, 95.0)
        assert vc.get_total_pnl() == pytest.approx(10.0 - 5.0)
