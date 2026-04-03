"""Tests for trade logger. Written FIRST per TDD."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from aegis.common.types import TradeLog


def _make_trade_log(**overrides) -> TradeLog:
    defaults = dict(
        trade_id="t-001",
        account_type="paper",
        symbol="BTC/USDT",
        asset_class="crypto",
        direction="LONG",
        entry_price=42000.0,
        entry_time=datetime(2025, 6, 1, tzinfo=timezone.utc),
        exit_price=None,
        exit_time=None,
        quantity=0.01,
        position_value=420.0,
        commission_entry=0.42,
        commission_exit=0.0,
        estimated_slippage=0.1,
        total_costs=0.52,
        gross_pnl=None,
        net_pnl=None,
        return_pct=None,
        r_multiple=None,
        holding_period_hours=None,
        ensemble_confidence=0.75,
        ensemble_direction=0.6,
        agent_signals_json='{"rsi_ema": 0.7}',
        regime_at_entry="normal",
        initial_stop_loss=41000.0,
        risk_amount=10.0,
        risk_pct_of_portfolio=0.02,
        exit_reason=None,
        feature_snapshot_json="{}",
    )
    defaults.update(overrides)
    return TradeLog(**defaults)


class TestTradeLogger:
    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.execute = MagicMock()
        db.fetch_one = MagicMock()
        db.fetch_all = MagicMock()
        return db

    @pytest.fixture
    def logger(self, mock_db):
        from aegis.execution.trade_logger import TradeLogger

        return TradeLogger(mock_db)

    def test_log_entry_inserts_trade(self, logger, mock_db):
        trade = _make_trade_log()
        logger.log_entry(trade)
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO trades" in sql

    def test_log_exit_updates_trade(self, logger, mock_db):
        logger.log_exit(
            trade_id="t-001",
            exit_price=43000.0,
            exit_time=datetime(2025, 6, 2, tzinfo=timezone.utc),
            exit_reason="take_profit",
            gross_pnl=10.0,
            net_pnl=9.48,
            return_pct=0.0238,
            r_multiple=1.0,
            holding_period_hours=24.0,
            commission_exit=0.43,
            total_costs=0.95,
        )
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "UPDATE trades" in sql
        assert "exit_price" in sql

    def test_get_trade_returns_row(self, logger, mock_db):
        mock_db.fetch_one.return_value = {"trade_id": "t-001", "symbol": "BTC/USDT"}
        result = logger.get_trade("t-001")
        assert result["trade_id"] == "t-001"

    def test_get_open_trades(self, logger, mock_db):
        mock_db.fetch_all.return_value = [
            {"trade_id": "t-001", "exit_price": None},
            {"trade_id": "t-002", "exit_price": None},
        ]
        result = logger.get_open_trades()
        assert len(result) == 2

    def test_get_closed_trades(self, logger, mock_db):
        mock_db.fetch_all.return_value = [
            {"trade_id": "t-003", "exit_price": 43000.0},
        ]
        result = logger.get_closed_trades(limit=50)
        assert len(result) == 1
        sql = mock_db.fetch_all.call_args[0][0]
        assert "exit_price IS NOT NULL" in sql
