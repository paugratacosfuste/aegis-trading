"""Tests for position manager. Written FIRST per TDD."""

from datetime import datetime, timezone

import pytest

from aegis.common.types import Position


def _make_position(symbol="BTC/USDT", direction="LONG", **overrides):
    defaults = dict(
        position_id="pos-1",
        symbol=symbol,
        direction=direction,
        quantity=0.01,
        entry_price=42000.0,
        entry_time=datetime(2025, 6, 1, tzinfo=timezone.utc),
        stop_loss=41000.0,
        take_profit=44000.0,
        unrealized_pnl=0.0,
        risk_amount=10.0,
    )
    defaults.update(overrides)
    return Position(**defaults)


class TestPositionManager:
    @pytest.fixture
    def pm(self):
        from aegis.execution.position_manager import PositionManager

        return PositionManager()

    def test_add_position(self, pm):
        pos = _make_position()
        pm.add(pos)
        assert len(pm.open_positions) == 1
        assert pm.open_positions[0].symbol == "BTC/USDT"

    def test_remove_position(self, pm):
        pos = _make_position()
        pm.add(pos)
        pm.remove("pos-1")
        assert len(pm.open_positions) == 0

    def test_remove_nonexistent_is_noop(self, pm):
        pm.remove("nonexistent")  # Should not raise
        assert len(pm.open_positions) == 0

    def test_get_by_symbol(self, pm):
        pm.add(_make_position(symbol="BTC/USDT", position_id="pos-1"))
        pm.add(_make_position(symbol="ETH/USDT", position_id="pos-2"))
        result = pm.get_by_symbol("BTC/USDT")
        assert result is not None
        assert result.position_id == "pos-1"

    def test_get_by_symbol_returns_none(self, pm):
        assert pm.get_by_symbol("BTC/USDT") is None

    def test_has_position(self, pm):
        pm.add(_make_position())
        assert pm.has_position("BTC/USDT") is True
        assert pm.has_position("ETH/USDT") is False

    def test_update_pnl(self, pm):
        pm.add(_make_position())
        pm.update_pnl("pos-1", current_price=42500.0)
        pos = pm.get_by_id("pos-1")
        assert pos.unrealized_pnl == pytest.approx(5.0, abs=0.01)  # 0.01 * (42500 - 42000)

    def test_update_pnl_short(self, pm):
        pm.add(_make_position(direction="SHORT"))
        pm.update_pnl("pos-1", current_price=41500.0)
        pos = pm.get_by_id("pos-1")
        assert pos.unrealized_pnl == pytest.approx(5.0, abs=0.01)  # 0.01 * (42000 - 41500)

    def test_check_stop_loss_triggered_long(self, pm):
        pm.add(_make_position(stop_loss=41000.0))
        triggered = pm.check_stop_losses({"BTC/USDT": 40999.0})
        assert len(triggered) == 1
        assert triggered[0].position_id == "pos-1"

    def test_check_stop_loss_not_triggered(self, pm):
        pm.add(_make_position(stop_loss=41000.0))
        triggered = pm.check_stop_losses({"BTC/USDT": 41500.0})
        assert len(triggered) == 0

    def test_check_stop_loss_short(self, pm):
        pm.add(_make_position(direction="SHORT", stop_loss=43000.0))
        triggered = pm.check_stop_losses({"BTC/USDT": 43001.0})
        assert len(triggered) == 1
