"""Risk manager: orchestrates all pre-trade checks.

Gate between ensemble decisions and execution.
"""

from aegis.common.types import Position, RiskVerdict, TradeDecision
from aegis.risk.circuit_breaker import CircuitBreaker
from aegis.risk.position_sizer import calculate_position_size
from aegis.risk.stop_loss import calculate_stop_loss

# Cold-start defaults when no trade history exists
DEFAULT_WIN_RATE = 0.45
DEFAULT_AVG_WIN = 0.02
DEFAULT_AVG_LOSS = 0.015


class RiskManager:
    def __init__(
        self,
        max_open_positions: int = 5,
        max_risk_pct: float = 0.02,
        portfolio_value: float = 5000.0,
        daily_halt: float = -0.15,
        weekly_halt: float = -0.25,
        win_rate: float = DEFAULT_WIN_RATE,
        avg_win: float = DEFAULT_AVG_WIN,
        avg_loss: float = DEFAULT_AVG_LOSS,
    ):
        self._max_positions = max_open_positions
        self._max_risk_pct = max_risk_pct
        self._portfolio_value = portfolio_value
        self._circuit_breaker = CircuitBreaker(daily_halt, weekly_halt)
        self._win_rate = win_rate
        self._avg_win = avg_win
        self._avg_loss = avg_loss

    def update_portfolio_value(self, value: float) -> None:
        self._portfolio_value = value

    def evaluate(
        self,
        decision: TradeDecision,
        open_positions: list[Position],
        atr_14: float = 500.0,
        start_of_day_value: float | None = None,
        start_of_week_value: float | None = None,
    ) -> RiskVerdict:
        """Run all pre-trade checks. Returns APPROVE or REJECT."""

        if decision.action == "NO_TRADE":
            return RiskVerdict.reject("No trade signal")

        # Check circuit breakers
        sod = start_of_day_value or self._portfolio_value
        sow = start_of_week_value or self._portfolio_value
        cb_status = self._circuit_breaker.check(self._portfolio_value, sod, sow)
        if cb_status == "HALT":
            return RiskVerdict.reject("Circuit breaker HALT active")

        # Check position limit
        if len(open_positions) >= self._max_positions:
            return RiskVerdict.reject(f"Max positions reached ({self._max_positions})")

        # Check duplicate symbol
        open_symbols = {p.symbol for p in open_positions}
        if decision.symbol in open_symbols:
            return RiskVerdict.reject(f"Already positioned in {decision.symbol}")

        # Calculate position size
        size = calculate_position_size(
            portfolio_value=self._portfolio_value,
            win_rate=self._win_rate,
            avg_win=self._avg_win,
            avg_loss=self._avg_loss,
            signal_confidence=decision.confidence,
            max_risk_pct=self._max_risk_pct,
        )

        if size <= 0:
            return RiskVerdict.reject("Negative Kelly, no edge")

        # Reduce size on WARNING
        if cb_status == "WARNING":
            size *= 0.5

        # Calculate stop-loss
        entry = decision.entry_price or 0.0
        if entry <= 0:
            # Estimate entry from latest data (caller should provide)
            entry = 42000.0  # Placeholder, will be set by caller

        stop = calculate_stop_loss(
            entry_price=entry,
            direction=decision.action,
            atr_14=atr_14,
            volatility_regime="normal",
            timeframe="1h",
        )

        return RiskVerdict.approve(position_size=size, stop_loss=stop)
