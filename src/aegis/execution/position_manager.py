"""In-memory position tracking with stop-loss monitoring.

From 05-EXECUTION.md.
"""

from dataclasses import replace

from aegis.common.types import Position


class PositionManager:
    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}  # position_id -> Position

    @property
    def open_positions(self) -> list[Position]:
        return list(self._positions.values())

    def add(self, position: Position) -> None:
        self._positions[position.position_id] = position

    def remove(self, position_id: str) -> None:
        self._positions.pop(position_id, None)

    def get_by_id(self, position_id: str) -> Position | None:
        return self._positions.get(position_id)

    def get_by_symbol(self, symbol: str) -> Position | None:
        for pos in self._positions.values():
            if pos.symbol == symbol:
                return pos
        return None

    def has_position(self, symbol: str) -> bool:
        return any(p.symbol == symbol for p in self._positions.values())

    def update_pnl(self, position_id: str, current_price: float) -> None:
        pos = self._positions.get(position_id)
        if pos is None:
            return
        if pos.direction == "LONG":
            pnl = pos.quantity * (current_price - pos.entry_price)
        else:
            pnl = pos.quantity * (pos.entry_price - current_price)
        self._positions[position_id] = replace(pos, unrealized_pnl=pnl)

    def check_stop_losses(self, current_prices: dict[str, float]) -> list[Position]:
        """Return positions whose stop-loss has been triggered."""
        triggered = []
        for pos in self._positions.values():
            price = current_prices.get(pos.symbol)
            if price is None:
                continue
            if pos.direction == "LONG" and price <= pos.stop_loss:
                triggered.append(pos)
            elif pos.direction == "SHORT" and price >= pos.stop_loss:
                triggered.append(pos)
        return triggered
