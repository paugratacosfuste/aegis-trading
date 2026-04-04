"""Cohort runner: independent decision pipeline for one strategy cohort."""

import logging

from aegis.common.types import AgentSignal, TradeDecision
from aegis.ensemble.voter import vote
from aegis.lab.types import CohortPerformance, StrategyCohort
from aegis.lab.virtual_capital import VirtualCapitalTracker
from aegis.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)

# Sentiment agent types whose direction gets inverted for contrarian cohorts
_SENTIMENT_TYPES = frozenset({"sentiment", "world_leader"})

# Macro regime -> Kelly fraction multiplier for macro-first cohorts
_MACRO_KELLY_MULTIPLIERS: dict[str, float] = {
    "risk_on": 1.0,
    "expansion": 1.0,
    "early_cycle": 1.0,
    "bull": 1.0,
    "risk_off": 0.25,
    "contraction": 0.25,
    "bear": 0.25,
    "crisis": 0.25,
    "recession_risk": 0.25,
}
_DEFAULT_KELLY_MULT = 0.5  # neutral/unknown regimes


class CohortRunner:
    """Runs one cohort's decision pipeline on shared signals."""

    def __init__(self, cohort: StrategyCohort, risk_config: dict | None = None):
        self._cohort = cohort
        self._capital = VirtualCapitalTracker(
            initial_capital=cohort.virtual_capital,
        )
        risk_cfg = {**(risk_config or {}), **cohort.config.risk_params}
        self._risk = RiskManager(risk_cfg)

    @property
    def cohort(self) -> StrategyCohort:
        return self._cohort

    @property
    def capital(self) -> VirtualCapitalTracker:
        return self._capital

    def process_signals(
        self,
        signals: list[AgentSignal],
        current_prices: dict[str, float],
        regime: str = "normal",
    ) -> TradeDecision | None:
        """Run ensemble vote + risk check for this cohort's config.

        Returns a TradeDecision if a trade is generated, else None.
        """
        if not signals:
            return None

        # Apply sentiment inversion for contrarian cohorts (cohort E)
        if self._cohort.config.invert_sentiment:
            signals = self._invert_sentiment(signals)

        # Vote with cohort-specific weights and threshold
        decision = vote(
            signals=signals,
            confidence_threshold=self._cohort.config.confidence_threshold,
            agent_weights=self._cohort.config.agent_weights,
            regime=regime,
        )

        if decision.action == "NO_TRADE":
            return None

        # Apply macro position sizing for cohort F
        if self._cohort.config.macro_position_sizing:
            kelly_mult = _MACRO_KELLY_MULTIPLIERS.get(regime, _DEFAULT_KELLY_MULT)
            # Scale quantity by Kelly multiplier (applied later in sizing)
            decision = TradeDecision(
                action=decision.action,
                symbol=decision.symbol,
                direction=decision.direction,
                confidence=decision.confidence * kelly_mult,
                quantity=decision.quantity,
                entry_price=decision.entry_price,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                contributing_signals=decision.contributing_signals,
                reason=f"{decision.reason} [macro_kelly={kelly_mult:.2f}]",
            )

        # Execute on virtual capital if we have a price
        symbol = decision.symbol
        price = current_prices.get(symbol)
        if price is None or price <= 0:
            return None

        # Simple position sizing: risk_params max_risk * capital / price
        max_risk = self._cohort.config.risk_params.get("max_risk_per_trade", 0.05)
        position_value = self._capital.cash * max_risk * decision.confidence
        quantity = position_value / price
        if quantity <= 0:
            return None

        pid = self._capital.open_position(
            symbol=symbol,
            direction=decision.action,
            quantity=quantity,
            entry_price=price,
        )
        if not pid:
            return None  # Insufficient capital

        return decision

    def check_exits(self, current_prices: dict[str, float]) -> list[dict]:
        """Check stop-losses on open positions. Returns list of closed trades."""
        closed = []
        for pos in list(self._capital.get_open_positions()):
            price = current_prices.get(pos.symbol)
            if price is None:
                continue
            # Simple stop-loss: close if > 5% loss
            stop_pct = self._cohort.config.risk_params.get("stop_loss_pct", 0.05)
            if pos.direction == "LONG" and price < pos.entry_price * (1 - stop_pct):
                pnl = self._capital.close_position(pos.position_id, price)
                closed.append({"position_id": pos.position_id, "pnl": pnl, "reason": "stop_loss"})
            elif pos.direction == "SHORT" and price > pos.entry_price * (1 + stop_pct):
                pnl = self._capital.close_position(pos.position_id, price)
                closed.append({"position_id": pos.position_id, "pnl": pnl, "reason": "stop_loss"})
        return closed

    def get_performance(self) -> CohortPerformance:
        """Snapshot of current performance metrics."""
        from aegis.backtest.metrics import (
            calculate_max_drawdown,
            calculate_profit_factor,
            calculate_sharpe,
            calculate_win_rate,
        )

        pnls = self._capital.get_closed_pnls()
        equity = self._capital.get_equity_curve()

        # Convert PnLs to returns for Sharpe
        returns = []
        if equity and len(equity) > 1:
            for i in range(1, len(equity)):
                if equity[i - 1] > 0:
                    returns.append((equity[i] - equity[i - 1]) / equity[i - 1])

        return CohortPerformance(
            cohort_id=self._cohort.cohort_id,
            sharpe=calculate_sharpe(returns),
            win_rate=calculate_win_rate(pnls),
            max_drawdown=calculate_max_drawdown(equity),
            profit_factor=calculate_profit_factor(pnls),
            total_trades=len(pnls),
            net_pnl=self._capital.get_total_pnl(),
            equity_curve=tuple(equity),
        )

    def record_equity(self, current_prices: dict[str, float]) -> None:
        self._capital.record_equity_snapshot(current_prices)

    def _invert_sentiment(self, signals: list[AgentSignal]) -> list[AgentSignal]:
        """Create new signals with inverted sentiment direction. Never mutates input."""
        result = []
        for s in signals:
            if s.agent_type in _SENTIMENT_TYPES:
                result.append(AgentSignal(
                    agent_id=s.agent_id,
                    agent_type=s.agent_type,
                    symbol=s.symbol,
                    timestamp=s.timestamp,
                    direction=-s.direction,
                    confidence=s.confidence,
                    timeframe=s.timeframe,
                    expected_holding_period=s.expected_holding_period,
                    entry_price=s.entry_price,
                    stop_loss=s.stop_loss,
                    take_profit=s.take_profit,
                    reasoning=s.reasoning,
                    features_used=s.features_used,
                    metadata=s.metadata,
                ))
            else:
                result.append(s)
        return result
