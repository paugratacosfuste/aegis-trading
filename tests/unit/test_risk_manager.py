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
