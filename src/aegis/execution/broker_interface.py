"""Abstract broker interface.

From 05-EXECUTION.md.
"""

from abc import ABC, abstractmethod

from aegis.common.types import Order


class BrokerInterface(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def get_account_balance(self) -> dict: ...

    @abstractmethod
    async def place_order(self, order: Order) -> dict: ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: int) -> dict: ...

    @abstractmethod
    async def get_order_status(self, symbol: str, order_id: int) -> dict: ...
