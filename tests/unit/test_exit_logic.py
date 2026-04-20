"""Tests for exit logic: trailing stop, take-profit, time-based exit.

Covers stop_loss.update_trailing_stop() and backtest engine exit checks.
"""

import pytest

from aegis.risk.stop_loss import update_trailing_stop


class TestUpdateTrailingStop:
    """Test trailing stop tightening logic."""

    def test_long_tightens_when_profitable(self):
        """Trailing stop moves up when LONG is >1% profitable."""
        new_stop = update_trailing_stop(
            current_stop=39000.0,
            current_price=42000.0,
            direction="LONG",
            atr_14=500.0,
            unrealized_pnl_pct=0.05,  # 5% profit → ratchet to 1.0x ATR
        )
        # Trail = 42000 - 500*1.0 = 41500 (ratcheted at 5% profit)
        assert new_stop == 41500.0

    def test_long_never_loosens(self):
        """Trailing stop cannot move below current stop for LONG."""
        new_stop = update_trailing_stop(
            current_stop=41000.0,
            current_price=41500.0,
            direction="LONG",
            atr_14=500.0,
            unrealized_pnl_pct=0.02,
        )
        # Trail = 41500 - 750 = 40750, but current_stop=41000 is higher
        assert new_stop == 41000.0

    def test_long_no_trail_when_not_profitable(self):
        """Trailing stop unchanged when <1% profit."""
        new_stop = update_trailing_stop(
            current_stop=39000.0,
            current_price=39500.0,
            direction="LONG",
            atr_14=500.0,
            unrealized_pnl_pct=0.005,  # 0.5% profit
        )
        assert new_stop == 39000.0

    def test_short_tightens_when_profitable(self):
        """Trailing stop moves down when SHORT is >1% profitable."""
        new_stop = update_trailing_stop(
            current_stop=45000.0,
            current_price=42000.0,
            direction="SHORT",
            atr_14=500.0,
            unrealized_pnl_pct=0.05,  # 5% profit → ratchet to 1.0x ATR
        )
        # Trail = 42000 + 500*1.0 = 42500 (ratcheted at 5% profit)
        assert new_stop == 42500.0

    def test_short_never_loosens(self):
        """Trailing stop cannot move above current stop for SHORT."""
        new_stop = update_trailing_stop(
            current_stop=42500.0,
            current_price=42400.0,
            direction="SHORT",
            atr_14=500.0,
            unrealized_pnl_pct=0.02,
        )
        # Trail = 42400 + 750 = 43150, but current_stop=42500 is lower
        assert new_stop == 42500.0


class TestBacktestExitIntegration:
    """Integration tests for engine exit checks."""

    def _make_position(
        self,
        direction="LONG",
        entry_price=40000.0,
        stop_loss=39000.0,
        quantity=0.1,
        entry_index=0,
        symbol="BTC/USDT",
        partial_taken=False,
    ):
        risk_amount = abs(entry_price - stop_loss)
        if direction == "LONG":
            take_profit = entry_price + 3.0 * risk_amount
        else:
            take_profit = entry_price - 3.0 * risk_amount
        return {
            "id": f"bt-{symbol}-{entry_index}",
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "entry_price": entry_price,
            "entry_index": entry_index,
            "stop_loss": stop_loss,
            "original_stop": stop_loss,
            "take_profit": take_profit,
            "risk_amount": risk_amount,
            "partial_taken": partial_taken,
            "position_value": entry_price * quantity,
            "commission_entry": entry_price * quantity * 0.001,
        }

    def test_stop_loss_still_triggers(self):
        """Stop loss should still close positions."""
        from aegis.backtest.engine import BacktestEngine

        engine = BacktestEngine(initial_capital=10000.0)
        pos = self._make_position(direction="LONG", entry_price=40000.0, stop_loss=39000.0)
        engine._positions = [pos]
        engine._check_exits(38500.0, 10, atr_14=500.0)
        assert len(engine._positions) == 0
        assert len(engine._closed_trades) == 1
        assert engine._closed_trades[0]["exit_reason"] == "stop_loss"

    def test_take_profit_full_at_3r(self):
        """Full exit when price reaches 3R."""
        from aegis.backtest.engine import BacktestEngine

        engine = BacktestEngine(initial_capital=10000.0)
        # LONG entry=40000, stop=39000, risk=1000, 3R target=43000
        pos = self._make_position(direction="LONG", entry_price=40000.0, stop_loss=39000.0)
        engine._positions = [pos]
        engine._check_exits(43100.0, 10, atr_14=500.0)  # Beyond 3R
        assert len(engine._positions) == 0
        assert len(engine._closed_trades) == 1
        assert engine._closed_trades[0]["exit_reason"] == "take_profit"

    def test_partial_exit_at_1_5r(self):
        """50% partial exit at 1.5R, position remains with reduced quantity."""
        from aegis.backtest.engine import BacktestEngine

        engine = BacktestEngine(initial_capital=10000.0)
        # LONG entry=40000, stop=39000, risk=1000, 1.5R target=41500
        pos = self._make_position(
            direction="LONG", entry_price=40000.0, stop_loss=39000.0, quantity=0.1
        )
        engine._positions = [pos]
        engine._check_exits(41600.0, 10, atr_14=500.0)  # Beyond 1.5R
        # Should have partial trade closed + remaining position
        assert len(engine._closed_trades) == 1
        assert engine._closed_trades[0]["exit_reason"] == "take_profit_partial"
        assert len(engine._positions) == 1
        assert engine._positions[0]["partial_taken"] is True
        assert engine._positions[0]["quantity"] == pytest.approx(0.05, rel=1e-6)

    def test_no_double_partial(self):
        """Once partial_taken=True, don't partial exit again at 1.5R."""
        from aegis.backtest.engine import BacktestEngine

        engine = BacktestEngine(initial_capital=10000.0)
        pos = self._make_position(
            direction="LONG", entry_price=40000.0, stop_loss=39000.0,
            quantity=0.05, partial_taken=True,
        )
        engine._positions = [pos]
        engine._check_exits(41600.0, 10, atr_14=500.0)
        # Position still open (between 1.5R and 3R, partial already taken)
        assert len(engine._positions) == 1
        assert len(engine._closed_trades) == 0

    def test_trailing_stop_tightens_in_exit_check(self):
        """Trailing stop updates during exit checks when profitable."""
        from aegis.backtest.engine import BacktestEngine

        engine = BacktestEngine(initial_capital=10000.0)
        # LONG entry=40000, stop=39000, price moved to 41500 (3.75% profit)
        pos = self._make_position(direction="LONG", entry_price=40000.0, stop_loss=39000.0)
        engine._positions = [pos]
        # Price is profitable but not at 1.5R take-profit (41500 = 1.5R)
        engine._check_exits(41400.0, 10, atr_14=500.0)
        # Stop should tighten: 41400 - 750 = 40650 > 39000
        assert engine._positions[0]["stop_loss"] > 39000.0

    def test_trailing_stop_then_reversal_closes(self):
        """Position opens, trailing stop tightens, then reversal triggers it."""
        from aegis.backtest.engine import BacktestEngine

        engine = BacktestEngine(initial_capital=10000.0)
        pos = self._make_position(direction="LONG", entry_price=40000.0, stop_loss=39000.0)
        engine._positions = [pos]

        # Step 1: price moves up, trailing stop tightens
        engine._check_exits(41400.0, 10, atr_14=500.0)
        tightened_stop = engine._positions[0]["stop_loss"]
        assert tightened_stop > 39000.0

        # Step 2: price reverses below tightened stop
        engine._check_exits(tightened_stop - 10, 11, atr_14=500.0)
        assert len(engine._positions) == 0
        assert engine._closed_trades[0]["exit_reason"] == "trailing_stop"

    def test_time_based_exit_when_losing(self):
        """Position exits when held >72 bars and at a loss (R < 0)."""
        from aegis.backtest.engine import BacktestEngine

        engine = BacktestEngine(initial_capital=10000.0)
        # LONG entry=40000, stop=39000, risk=1000
        pos = self._make_position(
            direction="LONG", entry_price=40000.0, stop_loss=39000.0, entry_index=0,
        )
        engine._positions = [pos]
        # At index 80 (>72 bars), price below entry → R = -0.3 < 0
        engine._check_exits(39700.0, 80, atr_14=500.0)
        assert len(engine._positions) == 0
        assert engine._closed_trades[0]["exit_reason"] == "time_exit"

    def test_no_time_exit_if_winning(self):
        """Don't time-exit if position is profitable — let trailing stop manage."""
        from aegis.backtest.engine import BacktestEngine

        engine = BacktestEngine(initial_capital=10000.0)
        pos = self._make_position(
            direction="LONG", entry_price=40000.0, stop_loss=39000.0, entry_index=0,
        )
        engine._positions = [pos]
        # At index 80 (>72 bars), price at 40300 → R = +0.3 (small winner)
        engine._check_exits(40300.0, 80, atr_14=500.0)
        assert len(engine._positions) == 1  # Still open, rides to trailing stop

    def test_short_take_profit(self):
        """SHORT position closes at 3R below entry."""
        from aegis.backtest.engine import BacktestEngine

        engine = BacktestEngine(initial_capital=10000.0)
        # SHORT entry=40000, stop=41000, risk=1000, 3R target=37000
        pos = self._make_position(
            direction="SHORT", entry_price=40000.0, stop_loss=41000.0
        )
        engine._positions = [pos]
        engine._check_exits(36900.0, 10, atr_14=500.0)
        assert len(engine._positions) == 0
        assert engine._closed_trades[0]["exit_reason"] == "take_profit"

    def test_multi_symbol_exit_checks(self):
        """Multi-symbol exit check uses per-symbol prices."""
        from aegis.backtest.engine import BacktestEngine

        engine = BacktestEngine(initial_capital=10000.0)
        pos_btc = self._make_position(
            direction="LONG", entry_price=40000.0, stop_loss=39000.0, symbol="BTC/USDT"
        )
        pos_eth = self._make_position(
            direction="LONG", entry_price=3000.0, stop_loss=2800.0, symbol="ETH/USDT"
        )
        engine._positions = [pos_btc, pos_eth]

        prices = {"BTC/USDT": 40500.0, "ETH/USDT": 2700.0}  # ETH below stop
        atrs = {"BTC/USDT": 500.0, "ETH/USDT": 50.0}
        engine._check_exits_multi(prices, 10, atrs)

        assert len(engine._positions) == 1
        assert engine._positions[0]["symbol"] == "BTC/USDT"
        assert len(engine._closed_trades) == 1
        assert engine._closed_trades[0]["symbol"] == "ETH/USDT"
        assert engine._closed_trades[0]["exit_reason"] == "stop_loss"
