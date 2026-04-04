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
from aegis.backtest.regime_detector import PriceRegimeDetector
from aegis.common.types import MarketDataPoint, Position
from aegis.ensemble.voter import vote
from aegis.risk.risk_manager import RiskManager
from aegis.risk.stop_loss import update_trailing_stop

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
        shadow_hook=None,
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
        self._regime_detector = PriceRegimeDetector()
        self._shadow_hook = shadow_hook

        self._positions: list[dict] = []
        self._closed_trades: list[dict] = []
        self._equity_curve: list[float] = []

        # Debug counters for ensemble diagnostics
        self._debug_total_cycles = 0
        self._debug_no_signals = 0
        self._debug_no_trade = 0
        self._debug_near_threshold = 0  # within 0.1 of threshold
        self._debug_rejection_reasons: dict[str, int] = {}
        self._debug_max_conf_seen = 0.0
        self._debug_passed_ensemble = 0
        self._debug_risk_rejected = 0

    def run(self, candles: list[MarketDataPoint]) -> dict:
        """Run backtest over candle history. Returns results dict."""
        self.equity = self.initial_capital
        self._positions = []
        self._closed_trades = []
        self._equity_curve = [self.initial_capital]

        if len(candles) < MIN_LOOKBACK:
            return self._build_results()

        # Reset debug counters
        self._debug_total_cycles = 0
        self._debug_no_signals = 0
        self._debug_no_trade = 0
        self._debug_near_threshold = 0
        self._debug_rejection_reasons = {}
        self._debug_max_conf_seen = 0.0

        # Train HMM regime detector on available data
        self._regime_detector = PriceRegimeDetector()
        warmup_end = min(self._regime_detector.warmup_bars, len(candles))
        if self._regime_detector.train(candles[:warmup_end]):
            logger.info("HMM regime detector trained on %d bars", warmup_end)
        else:
            logger.warning("HMM regime detector training failed — using regime='normal'")

        # Track regime distribution
        self._debug_regime_counts: dict[str, int] = {}

        regime = "normal"
        _REGIME_INTERVAL = 24  # Re-detect every 24 bars
        _REGIME_CAP = 2000
        _AGENT_CAP = 200  # Agents only need last ~120 candles max

        for i in range(MIN_LOOKBACK, len(candles)):
            window = candles[max(0, i + 1 - _AGENT_CAP) : i + 1]
            current_price = window[-1].close
            symbol = window[-1].symbol

            # Check exits on open positions (trailing stop, take-profit, stop-loss, time)
            atr = self._compute_atr(window[-14:]) if len(window) >= 14 else self._compute_atr(window)
            self._check_exits(current_price, i, atr_14=atr)

            # Detect regime periodically
            if (i - MIN_LOOKBACK) % _REGIME_INTERVAL == 0:
                regime_window = candles[max(0, i + 1 - _REGIME_CAP) : i + 1]
                regime = self._regime_detector.predict(regime_window)
            self._debug_regime_counts[regime] = self._debug_regime_counts.get(regime, 0) + 1

            # Generate signals from all agents
            self._debug_total_cycles += 1
            signals = []
            for agent in self._agents:
                sig = agent.generate_signal(symbol, window)
                if abs(sig.direction) > 0.01:
                    signals.append(sig)

            if not signals:
                self._debug_no_signals += 1
                self._update_equity(current_price)
                continue

            # Ensemble vote with regime-adjusted weights
            decision = vote(signals, self._confidence_threshold, regime=regime)

            if decision.action == "NO_TRADE":
                self._debug_no_trade += 1
                # Track rejection reason
                reason_key = decision.reason.split(":")[0] if ":" in decision.reason else decision.reason
                if reason_key.startswith("Confidence"):
                    reason_key = "below_threshold"
                elif reason_key.startswith("Direction"):
                    reason_key = "weak_direction"
                self._debug_rejection_reasons[reason_key] = (
                    self._debug_rejection_reasons.get(reason_key, 0) + 1
                )
                # Track near-threshold
                if decision.confidence > 0 and (
                    self._confidence_threshold - decision.confidence
                ) < 0.1:
                    self._debug_near_threshold += 1
                if decision.confidence > self._debug_max_conf_seen:
                    self._debug_max_conf_seen = decision.confidence
                self._update_equity(current_price)
                continue

            # Track max confidence from accepted trades too
            self._debug_passed_ensemble += 1
            if decision.confidence > self._debug_max_conf_seen:
                self._debug_max_conf_seen = decision.confidence

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
                self._debug_risk_rejected += 1
                self._update_equity(current_price)
                continue

            # Execute trade
            quantity = verdict.position_size / current_price if current_price > 0 else 0
            commission = verdict.position_size * self._commission_pct

            risk_amount = abs(current_price - verdict.stop_loss)
            if decision.action == "LONG":
                take_profit = current_price + 3.0 * risk_amount
            else:
                take_profit = current_price - 3.0 * risk_amount

            self._positions.append(
                {
                    "id": f"bt-{i}",
                    "symbol": symbol,
                    "direction": decision.action,
                    "quantity": quantity,
                    "entry_price": current_price,
                    "entry_index": i,
                    "stop_loss": verdict.stop_loss,
                    "original_stop": verdict.stop_loss,
                    "take_profit": take_profit,
                    "risk_amount": risk_amount,
                    "partial_taken": False,
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

    def run_multi(self, candles_by_symbol: dict[str, list[MarketDataPoint]]) -> dict:
        """Run backtest over multiple symbols simultaneously.

        Each symbol's candles are processed independently for signal generation,
        but share the same portfolio, equity, and risk manager.
        """
        if not candles_by_symbol:
            return self._build_results()

        # Single-symbol shortcut
        if len(candles_by_symbol) == 1:
            return self.run(next(iter(candles_by_symbol.values())))

        self.equity = self.initial_capital
        self._positions = []
        self._closed_trades = []
        self._equity_curve = [self.initial_capital]

        # Reset debug counters
        self._debug_total_cycles = 0
        self._debug_no_signals = 0
        self._debug_no_trade = 0
        self._debug_near_threshold = 0
        self._debug_rejection_reasons = {}
        self._debug_max_conf_seen = 0.0
        self._debug_passed_ensemble = 0
        self._debug_risk_rejected = 0
        self._debug_regime_counts = {}

        # Find common length (all symbols should have same timestamp grid)
        min_len = min(len(c) for c in candles_by_symbol.values())
        if min_len < MIN_LOOKBACK:
            return self._build_results()

        # Train HMM on the first (primary) symbol
        primary_symbol = next(iter(candles_by_symbol))
        primary_candles = candles_by_symbol[primary_symbol]
        self._regime_detector = PriceRegimeDetector()
        warmup_end = min(self._regime_detector.warmup_bars, min_len)
        if self._regime_detector.train(primary_candles[:warmup_end]):
            logger.info("HMM regime detector trained on %s (%d bars)", primary_symbol, warmup_end)
        else:
            logger.warning("HMM regime detector training failed — using regime='normal'")

        logger.info("Multi-symbol backtest: %d symbols, %d bars each", len(candles_by_symbol), min_len)

        regime = "normal"
        _REGIME_UPDATE_INTERVAL = 24  # Re-detect regime every 24 bars (daily for 1h)
        _REGIME_WINDOW_CAP = 2000
        _AGENT_WINDOW_CAP = 200  # Agents only need last ~120 candles max

        for i in range(MIN_LOOKBACK, min_len):
            # Detect regime periodically from primary symbol
            if (i - MIN_LOOKBACK) % _REGIME_UPDATE_INTERVAL == 0:
                regime_window = primary_candles[max(0, i + 1 - _REGIME_WINDOW_CAP) : i + 1]
                regime = self._regime_detector.predict(regime_window)
            self._debug_regime_counts[regime] = self._debug_regime_counts.get(regime, 0) + 1

            # Check exits on all open positions (need per-symbol prices and ATRs)
            current_prices = {
                sym: candles[i].close
                for sym, candles in candles_by_symbol.items()
            }
            current_atrs = {
                sym: self._compute_atr(candles[max(0, i + 1 - 14) : i + 1])
                for sym, candles in candles_by_symbol.items()
            }
            self._check_exits_multi(current_prices, i, current_atrs)

            # Process each symbol
            for symbol, candles in candles_by_symbol.items():
                window = candles[max(0, i + 1 - _AGENT_WINDOW_CAP) : i + 1]
                current_price = window[-1].close

                self._debug_total_cycles += 1
                signals = []
                for agent in self._agents:
                    sig = agent.generate_signal(symbol, window)
                    if abs(sig.direction) > 0.01:
                        signals.append(sig)

                if not signals:
                    self._debug_no_signals += 1
                    continue

                decision = vote(signals, self._confidence_threshold, regime=regime)

                # Shadow hook: record weight allocation prediction
                if self._shadow_hook and hasattr(self._shadow_hook, 'tracker') and self._shadow_hook.tracker:
                    try:
                        self._shadow_hook.tracker.on_ensemble_vote(
                            signals, regime, self.equity, self._equity_curve,
                        )
                    except Exception:
                        pass

                if decision.action == "NO_TRADE":
                    self._debug_no_trade += 1
                    reason_key = decision.reason.split(":")[0] if ":" in decision.reason else decision.reason
                    if reason_key.startswith("Confidence"):
                        reason_key = "below_threshold"
                    elif reason_key.startswith("Direction"):
                        reason_key = "weak_direction"
                    self._debug_rejection_reasons[reason_key] = (
                        self._debug_rejection_reasons.get(reason_key, 0) + 1
                    )
                    if decision.confidence > 0 and (
                        self._confidence_threshold - decision.confidence
                    ) < 0.1:
                        self._debug_near_threshold += 1
                    if decision.confidence > self._debug_max_conf_seen:
                        self._debug_max_conf_seen = decision.confidence
                    continue

                self._debug_passed_ensemble += 1
                if decision.confidence > self._debug_max_conf_seen:
                    self._debug_max_conf_seen = decision.confidence

                # Risk check (shared across all symbols)
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

                atr = self._compute_atr(window[-14:])
                self._risk_manager.update_portfolio_value(self.equity)
                verdict = self._risk_manager.evaluate(
                    decision, open_pos_objects, atr_14=atr
                )

                if not verdict.approved:
                    self._debug_risk_rejected += 1
                    continue

                quantity = verdict.position_size / current_price if current_price > 0 else 0
                commission = verdict.position_size * self._commission_pct

                risk_amount = abs(current_price - verdict.stop_loss)
                if decision.action == "LONG":
                    take_profit = current_price + 3.0 * risk_amount
                else:
                    take_profit = current_price - 3.0 * risk_amount

                self._positions.append(
                    {
                        "id": f"bt-{symbol}-{i}",
                        "symbol": symbol,
                        "direction": decision.action,
                        "quantity": quantity,
                        "entry_price": current_price,
                        "entry_index": i,
                        "stop_loss": verdict.stop_loss,
                        "original_stop": verdict.stop_loss,
                        "take_profit": take_profit,
                        "risk_amount": risk_amount,
                        "partial_taken": False,
                        "position_value": verdict.position_size,
                        "commission_entry": commission,
                    }
                )
                self.equity -= commission

            # Update equity using per-symbol prices for unrealized PnL
            self._update_equity_multi(current_prices)

        # Close all remaining positions at last price per symbol
        final_prices = {
            sym: candles[-1].close
            for sym, candles in candles_by_symbol.items()
        }
        for pos in list(self._positions):
            exit_price = final_prices.get(pos["symbol"], pos["entry_price"])
            self._close_position(pos, exit_price, min_len - 1, "end_of_backtest")
        self._positions = []

        return self._build_results()

    def _check_exits_multi(self, prices: dict[str, float], index: int, atrs: dict[str, float] | None = None) -> None:
        """Check exits using per-symbol prices and ATRs."""
        remaining = []
        for pos in self._positions:
            price = prices.get(pos["symbol"], pos["entry_price"])
            atr = (atrs or {}).get(pos["symbol"], 500.0)
            exit_reason = self._evaluate_exit(pos, price, index, atr)

            if exit_reason == "take_profit_partial":
                self._partial_close(pos, price, index)
                remaining.append(pos)
            elif exit_reason is not None:
                self._close_position(pos, price, index, exit_reason)
            else:
                remaining.append(pos)
        self._positions = remaining

    def _update_equity_multi(self, prices: dict[str, float]) -> None:
        """Update equity curve using per-symbol prices."""
        unrealized = 0.0
        for pos in self._positions:
            price = prices.get(pos["symbol"], pos["entry_price"])
            if pos["direction"] == "LONG":
                unrealized += pos["quantity"] * (price - pos["entry_price"])
            else:
                unrealized += pos["quantity"] * (pos["entry_price"] - price)
        self._equity_curve.append(self.equity + unrealized)

    def _check_exits(self, current_price: float, index: int, atr_14: float = 500.0) -> None:
        remaining = []
        for pos in self._positions:
            exit_reason = self._evaluate_exit(pos, current_price, index, atr_14)

            if exit_reason == "take_profit_partial":
                self._partial_close(pos, current_price, index)
                remaining.append(pos)
            elif exit_reason is not None:
                self._close_position(pos, current_price, index, exit_reason)
            else:
                remaining.append(pos)
        self._positions = remaining

    # -- Exit evaluation constants --
    _TIME_EXIT_BARS = 72  # 1.5x expected holding (~48h for 1h bars)
    _TIME_EXIT_R_THRESHOLD = 0.5

    def _evaluate_exit(self, pos: dict, current_price: float, index: int, atr_14: float) -> str | None:
        """Evaluate all exit conditions for a position. Returns exit reason or None."""
        direction = pos["direction"]
        entry = pos["entry_price"]
        risk_amount = pos.get("risk_amount", abs(entry - pos["stop_loss"]))

        # Compute R-multiple
        if direction == "LONG":
            pnl_per_unit = current_price - entry
        else:
            pnl_per_unit = entry - current_price

        r_multiple = pnl_per_unit / risk_amount if risk_amount > 0 else 0.0

        # 1. Full take-profit at 3R
        if r_multiple >= 3.0:
            return "take_profit"

        # 2. Partial take-profit at 1.5R (if not already taken)
        if r_multiple >= 1.5 and not pos.get("partial_taken", False):
            return "take_profit_partial"

        # 3. Trailing stop update + check
        unrealized_pnl_pct = pnl_per_unit / entry if entry > 0 else 0.0
        old_stop = pos["stop_loss"]
        new_stop = update_trailing_stop(
            current_stop=old_stop,
            current_price=current_price,
            direction=direction,
            atr_14=atr_14,
            unrealized_pnl_pct=unrealized_pnl_pct,
        )
        pos["stop_loss"] = new_stop  # Tighten in-place (mutable dict)

        # Check if trailing stop was hit
        if new_stop != old_stop:
            # Stop was tightened — check if price already below the new stop
            if direction == "LONG" and current_price <= new_stop:
                return "trailing_stop"
            if direction == "SHORT" and current_price >= new_stop:
                return "trailing_stop"

        # Check original/tightened stop
        original_stop = pos.get("original_stop", pos.get("stop_loss"))
        if direction == "LONG" and current_price <= pos["stop_loss"]:
            if pos["stop_loss"] != original_stop:
                return "trailing_stop"
            return "stop_loss"
        if direction == "SHORT" and current_price >= pos["stop_loss"]:
            if pos["stop_loss"] != original_stop:
                return "trailing_stop"
            return "stop_loss"

        # 4. Time-based exit
        bars_held = index - pos["entry_index"]
        if bars_held > self._TIME_EXIT_BARS and abs(r_multiple) < self._TIME_EXIT_R_THRESHOLD:
            return "time_exit"

        return None

    def _partial_close(self, pos: dict, exit_price: float, index: int) -> None:
        """Close 50% of position as partial take-profit."""
        close_qty = pos["quantity"] * 0.5

        if pos["direction"] == "LONG":
            pnl = close_qty * (exit_price - pos["entry_price"])
        else:
            pnl = close_qty * (pos["entry_price"] - exit_price)

        commission_exit = abs(close_qty * exit_price) * self._commission_pct
        # Attribute half the entry commission to this partial close
        commission_entry_partial = pos["commission_entry"] * 0.5
        net_pnl = pnl - commission_entry_partial - commission_exit

        self.equity += net_pnl

        self._closed_trades.append(
            {
                "symbol": pos["symbol"],
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "quantity": close_qty,
                "gross_pnl": pnl,
                "net_pnl": net_pnl,
                "commission": commission_entry_partial + commission_exit,
                "exit_reason": "take_profit_partial",
            }
        )

        # Update position: reduce quantity, mark partial taken, halve remaining commission
        pos["quantity"] = pos["quantity"] - close_qty
        pos["partial_taken"] = True
        pos["commission_entry"] = pos["commission_entry"] * 0.5

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

        trade = {
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
        self._closed_trades.append(trade)

        # Shadow hook: track trade outcome
        if self._shadow_hook and hasattr(self._shadow_hook, 'tracker') and self._shadow_hook.tracker:
            try:
                self._shadow_hook.tracker.on_trade_closed(trade)
            except Exception:
                pass

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

        # Print diagnostic summaries
        self._print_debug_summary()
        self._risk_manager.print_debug_summary()

        results = {
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

        # Add shadow summary if available
        if self._shadow_hook:
            try:
                results["shadow_summary"] = self._shadow_hook.get_summary()
            except Exception:
                pass

        return results

    def _print_debug_summary(self) -> None:
        """Print ensemble diagnostic summary at end of backtest."""
        total = self._debug_total_cycles
        if total == 0:
            return
        traded = len(self._closed_trades) + len(self._positions)
        print("\n" + "=" * 60)
        print("ENSEMBLE DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print(f"  Total evaluation cycles:     {total}")
        print(f"  Cycles with no signals:      {self._debug_no_signals} ({self._debug_no_signals/total*100:.1f}%)")
        print(f"  Ensemble rejected (NO_TRADE):{self._debug_no_trade} ({self._debug_no_trade/total*100:.1f}%)")
        print(f"  Passed ensemble -> risk:     {self._debug_passed_ensemble} ({self._debug_passed_ensemble/total*100:.1f}%)")
        print(f"  Risk rejected:               {self._debug_risk_rejected}")
        print(f"  Actually traded:             {traded}")
        print(f"  Near-threshold (within 0.1): {self._debug_near_threshold}")
        print(f"  Max confidence seen:         {self._debug_max_conf_seen:.4f}")
        print(f"  Confidence threshold:        {self._confidence_threshold}")
        if self._debug_rejection_reasons:
            print(f"  Rejection breakdown:")
            for reason, count in sorted(
                self._debug_rejection_reasons.items(), key=lambda x: -x[1]
            ):
                print(f"    {reason:30s} {count:>6} ({count/total*100:.1f}%)")
        if hasattr(self, "_debug_regime_counts") and self._debug_regime_counts:
            print(f"  Regime distribution:")
            for regime, count in sorted(
                self._debug_regime_counts.items(), key=lambda x: -x[1]
            ):
                print(f"    {regime:30s} {count:>6} ({count/total*100:.1f}%)")
        print("=" * 60)
