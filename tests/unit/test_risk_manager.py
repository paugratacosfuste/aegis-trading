"""Tests for risk manager orchestration. Written FIRST per TDD."""

from datetime import datetime, timezone

import pytest

from aegis.common.types import Position, TradeDecision


def _make_decision(action="LONG", symbol="BTC/USDT", direction=0.7, confidence=0.8):
    return TradeDecision(
        action=action,
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        quantity=0.0,
        entry_price=42000.0,
        stop_loss=None,
        take_profit=None,
        contributing_signals={},
        reason="test",
    )


def _make_position(symbol="BTC/USDT"):
    return Position(
        position_id="pos-1",
        symbol=symbol,
        direction="LONG",
        quantity=0.01,
        entry_price=42000.0,
        entry_time=datetime(2025, 6, 1, tzinfo=timezone.utc),
        stop_loss=41000.0,
        take_profit=44000.0,
        unrealized_pnl=0.0,
        risk_amount=10.0,
    )


class TestRiskManager:
    def test_approves_valid_trade(self):
        from aegis.risk.risk_manager import RiskManager

        rm = RiskManager(
            max_open_positions=5,
            max_risk_pct=0.02,
            portfolio_value=5000.0,
            daily_halt=-0.15,
            weekly_halt=-0.25,
        )
        decision = _make_decision()
        verdict = rm.evaluate(decision, open_positions=[], atr_14=500.0)
        assert verdict.approved
        assert verdict.position_size > 0
        assert verdict.stop_loss > 0

    def test_rejects_when_max_positions_reached(self):
        from aegis.risk.risk_manager import RiskManager

        rm = RiskManager(
            max_open_positions=2,
            max_risk_pct=0.02,
            portfolio_value=5000.0,
            daily_halt=-0.15,
            weekly_halt=-0.25,
        )
        positions = [_make_position("ETH/USDT"), _make_position("SOL/USDT")]
        decision = _make_decision()
        verdict = rm.evaluate(decision, open_positions=positions, atr_14=500.0)
        assert not verdict.approved
        assert "Max positions" in verdict.reason

    def test_rejects_duplicate_symbol(self):
        from aegis.risk.risk_manager import RiskManager

        rm = RiskManager(
            max_open_positions=5,
            max_risk_pct=0.02,
            portfolio_value=5000.0,
            daily_halt=-0.15,
            weekly_halt=-0.25,
        )
        positions = [_make_position("BTC/USDT")]
        decision = _make_decision(symbol="BTC/USDT")
        verdict = rm.evaluate(decision, open_positions=positions, atr_14=500.0)
        assert not verdict.approved
        assert "Already positioned" in verdict.reason

    def test_rejects_no_trade(self):
        from aegis.risk.risk_manager import RiskManager

        rm = RiskManager(
            max_open_positions=5,
            max_risk_pct=0.02,
            portfolio_value=5000.0,
            daily_halt=-0.15,
            weekly_halt=-0.25,
        )
        decision = _make_decision(action="NO_TRADE")
        verdict = rm.evaluate(decision, open_positions=[], atr_14=500.0)
        assert not verdict.approved

    def test_stop_loss_set_on_approval(self):
        from aegis.risk.risk_manager import RiskManager

        rm = RiskManager(
            max_open_positions=5,
            max_risk_pct=0.02,
            portfolio_value=5000.0,
            daily_halt=-0.15,
            weekly_halt=-0.25,
        )
        decision = _make_decision(action="LONG")
        verdict = rm.evaluate(decision, open_positions=[], atr_14=500.0)
        assert verdict.approved
        assert verdict.stop_loss < 42000.0  # LONG stop below entry

    def test_daily_timeframe_wider_stop(self):
        """Daily equities should get wider stops than hourly."""
        from aegis.risk.risk_manager import RiskManager

        rm = RiskManager(
            max_open_positions=5,
            max_risk_pct=0.02,
            portfolio_value=5000.0,
        )
        decision = _make_decision(action="LONG", symbol="AAPL")

        v_1h = rm.evaluate(decision, [], atr_14=2.0, timeframe="1h", entry_price=200.0)
        v_1d = rm.evaluate(decision, [], atr_14=2.0, timeframe="1d", entry_price=200.0)
        # Daily should have wider stop (lower for LONG)
        assert v_1d.stop_loss < v_1h.stop_loss

    def test_position_size_scales_with_stop_distance(self):
        """Wider stops should result in smaller position sizes (same dollar risk)."""
        from aegis.risk.risk_manager import RiskManager

        rm = RiskManager(
            max_open_positions=5,
            max_risk_pct=0.02,
            portfolio_value=5000.0,
        )
        decision = _make_decision(action="LONG", symbol="AAPL")

        v_1h = rm.evaluate(decision, [], atr_14=2.0, timeframe="1h", entry_price=200.0)
        v_1d = rm.evaluate(decision, [], atr_14=2.0, timeframe="1d", entry_price=200.0)
        # Daily should have smaller or equal position (wider stop = same risk budget)
        assert v_1d.position_size <= v_1h.position_size

    def test_explicit_entry_price_overrides_decision(self):
        """When entry_price kwarg is passed, stop is computed from it, not decision.entry_price."""
        from aegis.risk.risk_manager import RiskManager

        rm = RiskManager(
            max_open_positions=5,
            max_risk_pct=0.02,
            portfolio_value=5000.0,
        )
        # Decision has entry_price=42000, but we pass entry_price=200 (equity)
        decision = _make_decision(action="LONG", symbol="AAPL")
        verdict = rm.evaluate(
            decision, [], atr_14=2.0, timeframe="1d", entry_price=200.0,
        )
        assert verdict.approved
        # Stop should be near 200, not near 42000
        assert verdict.stop_loss < 200.0
        assert verdict.stop_loss > 190.0  # 200 - 2*2*1.4 = 194.4

    def test_rejects_when_no_entry_price(self):
        """Without any entry price, trade is rejected (no more $42K placeholder)."""
        from aegis.risk.risk_manager import RiskManager

        rm = RiskManager(
            max_open_positions=5,
            max_risk_pct=0.02,
            portfolio_value=5000.0,
        )
        decision = TradeDecision(
            action="LONG", symbol="TEST", direction=0.7, confidence=0.8,
            quantity=0.0, entry_price=0.0, stop_loss=None, take_profit=None,
            contributing_signals={}, reason="test",
        )
        verdict = rm.evaluate(decision, [], atr_14=2.0)
        assert not verdict.approved

    def test_atr_risk_cap_limits_position(self):
        """Position size should never risk more than max_risk_pct of portfolio."""
        from aegis.risk.risk_manager import RiskManager

        rm = RiskManager(
            max_open_positions=5,
            max_risk_pct=0.02,
            portfolio_value=5000.0,
        )
        decision = _make_decision(action="LONG", symbol="AAPL")
        entry = 200.0
        verdict = rm.evaluate(
            decision, [], atr_14=2.0, timeframe="1d", entry_price=entry,
        )

        assert verdict.approved
        stop_dist = abs(entry - verdict.stop_loss)
        shares = verdict.position_size / entry
        dollar_risk = shares * stop_dist
        # Dollar risk should not exceed 2% of portfolio ($100)
        assert dollar_risk <= 5000.0 * 0.02 * 1.01  # 1% tolerance
