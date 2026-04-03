"""Tests for Binance broker interface. Written FIRST per TDD."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegis.common.types import Order, Position


class TestBinanceBroker:
    @pytest.fixture
    def broker(self):
        from aegis.execution.binance_broker import BinanceBroker

        return BinanceBroker(
            api_key="test-key",
            api_secret="test-secret",
            testnet=True,
        )

    @pytest.mark.asyncio
    async def test_connect_creates_client(self, broker):
        with patch(
            "aegis.execution.binance_broker.AsyncClient.create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = AsyncMock()
            await broker.connect()
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_closes_client(self, broker):
        mock_client = AsyncMock()
        broker._client = mock_client
        await broker.disconnect()
        mock_client.close_connection.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_account_balance(self, broker):
        mock_client = AsyncMock()
        mock_client.get_account.return_value = {
            "balances": [
                {"asset": "USDT", "free": "5000.0", "locked": "100.0"},
                {"asset": "BTC", "free": "0.01", "locked": "0.0"},
            ]
        }
        broker._client = mock_client
        balance = await broker.get_account_balance()
        assert balance["USDT"] == {"free": 5000.0, "locked": 100.0}
        assert balance["BTC"] == {"free": 0.01, "locked": 0.0}

    @pytest.mark.asyncio
    async def test_place_market_order(self, broker):
        mock_client = AsyncMock()
        mock_client.create_order.return_value = {
            "orderId": 12345,
            "status": "FILLED",
            "executedQty": "0.01",
            "cummulativeQuoteQty": "420.0",
        }
        broker._client = mock_client

        order = Order(
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.01,
            limit_price=None,
            stop_price=None,
            time_in_force="GTC",
            broker="binance",
            account_type="paper",
            trade_id="t-001",
            signal_id="s-001",
        )
        result = await broker.place_order(order)
        assert result["orderId"] == 12345
        assert result["status"] == "FILLED"
        mock_client.create_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_stop_loss_order(self, broker):
        mock_client = AsyncMock()
        mock_client.create_order.return_value = {
            "orderId": 12346,
            "status": "NEW",
        }
        broker._client = mock_client

        order = Order(
            symbol="BTC/USDT",
            side="SELL",
            order_type="STOP",
            quantity=0.01,
            limit_price=None,
            stop_price=41000.0,
            time_in_force="GTC",
            broker="binance",
            account_type="paper",
            trade_id="t-001",
            signal_id="s-001",
        )
        result = await broker.place_order(order)
        assert result["orderId"] == 12346
        call_kwargs = mock_client.create_order.call_args[1]
        assert call_kwargs["stopPrice"] == 41000.0

    @pytest.mark.asyncio
    async def test_symbol_conversion(self, broker):
        assert broker._to_binance_symbol("BTC/USDT") == "BTCUSDT"
        assert broker._to_binance_symbol("ETH/USDT") == "ETHUSDT"

    @pytest.mark.asyncio
    async def test_get_order_status(self, broker):
        mock_client = AsyncMock()
        mock_client.get_order.return_value = {
            "orderId": 12345,
            "status": "FILLED",
            "executedQty": "0.01",
        }
        broker._client = mock_client
        status = await broker.get_order_status("BTC/USDT", 12345)
        assert status["status"] == "FILLED"
