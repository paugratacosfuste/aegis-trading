"""Backtest replay engine.

Feeds historical candles through the same agent/ensemble/risk pipeline.
No look-ahead bias: each step only sees candles up to current index.
"""

import logging

from aegis.agents.base import BaseAgent
from aegis.agents.factory import create_default_agents
from aegis.backtest.metrics import (
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe,
    calculate_win_rate,
)
from aegis.common.types import MarketDataPoint, Position
from aegis.ensemble.voter import vote
from aegis.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)

MIN_LOOKBACK = 21  # Agents need at least 21 candles


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 5000.0,
        commission_pct: float = 0.001,
        confidence_threshold: float = 0.45,
        max_open_positions: int = 5,
        max_risk_pct: float = 0.05,
        agents: list[BaseAgent] | None = None,
    ):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self._commission_pct = commission_pct
        self._confidence_threshold = confidence_threshold

        self._agents = agents if agents is not None else create_default_agents()
        self._risk_manager = RiskManager(
            max_open_positions=max_open_positions,
            max_risk_pct=max_risk_pct,
            portfolio_value=initial_capital,
        )

        self._positions: list[dict] = []
        self._closed_trades: list[dict] = []
        self._equity_curve: list[float] = []

    def run(self, candles: list[MarketDataPoint]) -> dict:
        """Run backtest over candle history. Returns results dict."""
        self.equity = self.initial_capital
        self._positions = []
        self._closed_trades = []
        self._equity_curve = [self.initial_capital]

        if len(candles) < MIN_LOOKBACK:
            return self._build_results()

        for i in range(MIN_LOOKBACK, len(candles)):
            window = candles[: i + 1]
            current_price = window[-1].close
            symbol = window[-1].symbol

            # Check stop-losses on open positions
            self._check_exits(current_price, i)

            # Generate signals from all agents
            signals = []
            for agent in self._agents:
                sig = agent.generate_signal(symbol, window)
                if abs(sig.direction) > 0.01:
                    signals.append(sig)

            if not signals:
                self._update_equity(current_price)
                continue

            # Ensemble vote
            decision = vote(signals, self._confidence_threshold)

            if decision.action == "NO_TRADE":
                self._update_equity(current_price)
                continue

            # Risk check
            open_pos_objects = [
                Position(
                    position_id=p["id"],
                    symbol=p["symbol"],
                    direction=p["direction"],
                    quantity=p["quantity"],
                    entry_price=p["entry_price"],
                    entry_time=window[-1].timestamp,
                    stop_loss=p["stop_loss"],
                    take_profit=None,
                    unrealized_pnl=0.0,
                    risk_amount=0.0,
                )
                for p in self._positions
            ]

            # Compute ATR from last 14 candles
            atr = self._compute_atr(window[-14:])

            self._risk_manager.update_portfolio_value(self.equity)
            verdict = self._risk_manager.evaluate(
                decision, open_pos_objects, atr_14=atr
            )

            if not verdict.approved:
                self._update_equity(current_price)
                continue

            # Execute trade
            quantity = verdict.position_size / current_price if current_price > 0 else 0
            commission = verdict.position_size * self._commission_pct

            self._positions.append(
                {
                    "id": f"bt-{i}",
                    "symbol": symbol,
                    "direction": decision.action,
                    "quantity": quantity,
                    "entry_price": current_price,
                    "entry_index": i,
                    "stop_loss": verdict.stop_loss,
                    "position_value": verdict.position_size,
                    "commission_entry": commission,
                }
            )
            self.equity -= commission

            self._update_equity(current_price)

        # Close all remaining positions at last price
        if candles:
            self._close_all(candles[-1].close, len(candles) - 1)

        return self._build_results()

    def _check_exits(self, current_price: float, index: int) -> None:
        remaining = []
        for pos in self._positions:
            triggered = False
            if pos["direction"] == "LONG" and current_price <= pos["stop_loss"]:
                triggered = True
            elif pos["direction"] == "SHORT" and current_price >= pos["stop_loss"]:
                triggered = True

            if triggered:
                self._close_position(pos, current_price, index, "stop_loss")
            else:
                remaining.append(pos)
        self._positions = remaining

    def _close_position(
        self, pos: dict, exit_price: float, index: int, reason: str
    ) -> None:
        if pos["direction"] == "LONG":
            pnl = pos["quantity"] * (exit_price - pos["entry_price"])
        else:
            pnl = pos["quantity"] * (pos["entry_price"] - exit_price)

        commission_exit = abs(pos["quantity"] * exit_price) * self._commission_pct
        net_pnl = pnl - pos["commission_entry"] - commission_exit

        self.equity += net_pnl

        self._closed_trades.append(
            {
                "symbol": pos["symbol"],
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "quantity": pos["quantity"],
                "gross_pnl": pnl,
                "net_pnl": net_pnl,
                "commission": pos["commission_entry"] + commission_exit,
                "exit_reason": reason,
            }
        )

    def _close_all(self, price: float, index: int) -> None:
        for pos in self._positions:
            self._close_position(pos, price, index, "end_of_backtest")
        self._positions = []

    def _update_equity(self, current_price: float) -> None:
        unrealized = 0.0
        for pos in self._positions:
            if pos["direction"] == "LONG":
                unrealized += pos["quantity"] * (current_price - pos["entry_price"])
            else:
                unrealized += pos["quantity"] * (pos["entry_price"] - current_price)
        self._equity_curve.append(self.equity + unrealized)

    def _compute_atr(self, candles: list[MarketDataPoint]) -> float:
        if len(candles) < 2:
            return 500.0
        trs = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        return sum(trs) / len(trs) if trs else 500.0

    def _build_results(self) -> dict:
        pnls = [t["net_pnl"] for t in self._closed_trades]

        # Daily returns from equity curve
        returns = []
        for i in range(1, len(self._equity_curve)):
            prev = self._equity_curve[i - 1]
            if prev > 0:
                returns.append((self._equity_curve[i] - prev) / prev)

        return {
            "equity_curve": self._equity_curve,
            "trades": self._closed_trades,
            "metrics": {
                "sharpe": calculate_sharpe(returns, periods_per_year=365 * 24),
                "max_drawdown": calculate_max_drawdown(self._equity_curve),
                "win_rate": calculate_win_rate(pnls),
                "profit_factor": calculate_profit_factor(pnls),
                "total_trades": len(self._closed_trades),
                "final_equity": self._equity_curve[-1] if self._equity_curve else self.initial_capital,
                "total_return_pct": (
                    (self._equity_curve[-1] / self.initial_capital - 1) * 100
                    if self._equity_curve
                    else 0.0
                ),
            },
        }
