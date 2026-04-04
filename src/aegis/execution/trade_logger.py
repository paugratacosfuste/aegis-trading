"""Trade logger: writes TradeLog records to the trades table.

From 05-EXECUTION.md.
"""

from datetime import datetime

from aegis.common.db import DatabasePool
from aegis.common.types import TradeLog


class TradeLogger:
    def __init__(self, db: DatabasePool):
        self._db = db

    def log_entry(self, trade: TradeLog) -> None:
        sql = """
            INSERT INTO trades (
                trade_id, account_type, symbol, asset_class, direction,
                entry_price, entry_time, quantity, position_value,
                commission_entry, estimated_slippage, total_costs,
                ensemble_confidence, ensemble_direction, agent_signals_json,
                regime_at_entry, initial_stop_loss, risk_amount,
                risk_pct_of_portfolio, feature_snapshot_json, cohort_id
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s
            )
        """
        self._db.execute(sql, (
            trade.trade_id,
            trade.account_type,
            trade.symbol,
            trade.asset_class,
            trade.direction,
            trade.entry_price,
            trade.entry_time,
            trade.quantity,
            trade.position_value,
            trade.commission_entry,
            trade.estimated_slippage,
            trade.total_costs,
            trade.ensemble_confidence,
            trade.ensemble_direction,
            trade.agent_signals_json,
            trade.regime_at_entry,
            trade.initial_stop_loss,
            trade.risk_amount,
            trade.risk_pct_of_portfolio,
            trade.feature_snapshot_json,
            trade.cohort_id,
        ))

    def log_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        gross_pnl: float,
        net_pnl: float,
        return_pct: float,
        r_multiple: float,
        holding_period_hours: float,
        commission_exit: float,
        total_costs: float,
    ) -> None:
        sql = """
            UPDATE trades SET
                exit_price = %s,
                exit_time = %s,
                exit_reason = %s,
                gross_pnl = %s,
                net_pnl = %s,
                return_pct = %s,
                r_multiple = %s,
                holding_period_hours = %s,
                commission_exit = %s,
                total_costs = %s
            WHERE trade_id = %s
        """
        self._db.execute(sql, (
            exit_price,
            exit_time,
            exit_reason,
            gross_pnl,
            net_pnl,
            return_pct,
            r_multiple,
            holding_period_hours,
            commission_exit,
            total_costs,
            trade_id,
        ))

    def get_trade(self, trade_id: str) -> dict | None:
        return self._db.fetch_one(
            "SELECT * FROM trades WHERE trade_id = %s",
            (trade_id,),
        )

    def get_open_trades(self, cohort_id: str | None = None) -> list[dict]:
        if cohort_id is not None:
            return self._db.fetch_all(
                "SELECT * FROM trades WHERE exit_price IS NULL AND cohort_id = %s ORDER BY entry_time DESC",
                (cohort_id,),
            )
        return self._db.fetch_all(
            "SELECT * FROM trades WHERE exit_price IS NULL ORDER BY entry_time DESC"
        )

    def get_closed_trades(self, limit: int = 100, cohort_id: str | None = None) -> list[dict]:
        if cohort_id is not None:
            return self._db.fetch_all(
                "SELECT * FROM trades WHERE exit_price IS NOT NULL AND cohort_id = %s ORDER BY exit_time DESC LIMIT %s",
                (cohort_id, limit),
            )
        return self._db.fetch_all(
            "SELECT * FROM trades WHERE exit_price IS NOT NULL ORDER BY exit_time DESC LIMIT %s",
            (limit,),
        )
