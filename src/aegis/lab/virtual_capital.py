"""Virtual capital tracker for paper-trading cohorts."""

from dataclasses import dataclass, field
import uuid


@dataclass(frozen=True)
class VirtualPosition:
    position_id: str
    symbol: str
    direction: str  # "LONG" | "SHORT"
    quantity: float
    entry_price: float
    commission: float


class VirtualCapitalTracker:
    """Tracks paper capital, open positions, and equity for one cohort."""

    def __init__(self, initial_capital: float = 100_000.0, commission_rate: float = 0.001):
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._commission_rate = commission_rate
        self._positions: dict[str, VirtualPosition] = {}
        self._closed_pnls: list[float] = []
        self._equity_curve: list[float] = [initial_capital]

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    def open_position(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        entry_price: float,
    ) -> str:
        """Open a paper position. Returns position_id."""
        position_value = quantity * entry_price
        commission = position_value * self._commission_rate
        if position_value + commission > self._cash:
            return ""  # Insufficient capital

        self._cash -= position_value + commission
        pid = f"vp_{uuid.uuid4().hex[:8]}"
        self._positions[pid] = VirtualPosition(
            position_id=pid,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            commission=commission,
        )
        return pid

    def close_position(self, position_id: str, exit_price: float) -> float:
        """Close a position and return net PnL."""
        pos = self._positions.pop(position_id, None)
        if pos is None:
            return 0.0

        exit_value = pos.quantity * exit_price
        exit_commission = exit_value * self._commission_rate

        if pos.direction == "LONG":
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        net_pnl = gross_pnl - pos.commission - exit_commission
        self._cash += exit_value - exit_commission
        self._closed_pnls.append(net_pnl)
        return net_pnl

    def get_equity(self, current_prices: dict[str, float]) -> float:
        """Total equity = cash + mark-to-market of open positions."""
        equity = self._cash
        for pos in self._positions.values():
            price = current_prices.get(pos.symbol, pos.entry_price)
            if pos.direction == "LONG":
                equity += pos.quantity * price
            else:
                # Short: profit when price drops
                equity += pos.quantity * (2 * pos.entry_price - price)
        return equity

    def record_equity_snapshot(self, current_prices: dict[str, float]) -> None:
        """Append current equity to the equity curve."""
        self._equity_curve.append(self.get_equity(current_prices))

    def get_equity_curve(self) -> list[float]:
        return list(self._equity_curve)

    def get_closed_pnls(self) -> list[float]:
        return list(self._closed_pnls)

    def get_open_positions(self) -> list[VirtualPosition]:
        return list(self._positions.values())

    def get_total_pnl(self) -> float:
        return sum(self._closed_pnls)
