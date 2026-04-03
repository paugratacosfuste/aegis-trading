"""Binance testnet broker implementation.

From 05-EXECUTION.md. Phase 1: market orders only.
"""

import logging

from binance import AsyncClient

from aegis.common.exceptions import BrokerError, OrderError
from aegis.common.types import Order
from aegis.execution.broker_interface import BrokerInterface

logger = logging.getLogger(__name__)


class BinanceBroker(BrokerInterface):
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._client: AsyncClient | None = None

    async def connect(self) -> None:
        self._client = await AsyncClient.create(
            api_key=self._api_key,
            api_secret=self._api_secret,
            testnet=self._testnet,
        )
        logger.info("Connected to Binance %s", "testnet" if self._testnet else "mainnet")

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close_connection()
            self._client = None

    async def get_account_balance(self) -> dict:
        if not self._client:
            raise BrokerError("Not connected")
        account = await self._client.get_account()
        return {
            b["asset"]: {"free": float(b["free"]), "locked": float(b["locked"])}
            for b in account["balances"]
            if float(b["free"]) > 0 or float(b["locked"]) > 0
        }

    async def place_order(self, order: Order) -> dict:
        if not self._client:
            raise BrokerError("Not connected")

        symbol = self._to_binance_symbol(order.symbol)
        params: dict = {
            "symbol": symbol,
            "side": order.side,
            "quantity": order.quantity,
        }

        if order.order_type == "MARKET":
            params["type"] = "MARKET"
        elif order.order_type == "LIMIT":
            params["type"] = "LIMIT"
            params["timeInForce"] = order.time_in_force
            params["price"] = order.limit_price
        elif order.order_type == "STOP":
            params["type"] = "STOP_LOSS"
            params["stopPrice"] = order.stop_price
        elif order.order_type == "STOP_LIMIT":
            params["type"] = "STOP_LOSS_LIMIT"
            params["stopPrice"] = order.stop_price
            params["price"] = order.limit_price
            params["timeInForce"] = order.time_in_force
        else:
            raise OrderError(f"Unsupported order type: {order.order_type}")

        try:
            result = await self._client.create_order(**params)
            logger.info("Order placed: %s %s %s qty=%s", order.side, symbol, order.order_type, order.quantity)
            return result
        except Exception as exc:
            raise OrderError(f"Order failed: {exc}") from exc

    async def cancel_order(self, symbol: str, order_id: int) -> dict:
        if not self._client:
            raise BrokerError("Not connected")
        binance_symbol = self._to_binance_symbol(symbol)
        return await self._client.cancel_order(symbol=binance_symbol, orderId=order_id)

    async def get_order_status(self, symbol: str, order_id: int) -> dict:
        if not self._client:
            raise BrokerError("Not connected")
        binance_symbol = self._to_binance_symbol(symbol)
        return await self._client.get_order(symbol=binance_symbol, orderId=order_id)

    @staticmethod
    def _to_binance_symbol(aegis_symbol: str) -> str:
        """Convert 'BTC/USDT' -> 'BTCUSDT'."""
        return aegis_symbol.replace("/", "")
