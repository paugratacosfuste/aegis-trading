"""Binance WebSocket collector for real-time crypto OHLCV data."""

import asyncio
import json
import logging
from datetime import datetime, timezone

import websockets

from aegis.common.types import MarketDataPoint
from aegis.data.repository import MarketDataRepository

logger = logging.getLogger(__name__)

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
BINANCE_TESTNET_WS_URL = "wss://testnet.binance.vision/ws"

DEFAULT_SYMBOLS = ["btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt"]


def _symbol_to_aegis(binance_symbol: str) -> str:
    """Convert binance symbol like 'btcusdt' to 'BTC/USDT'."""
    s = binance_symbol.upper()
    if s.endswith("USDT"):
        return s[:-4] + "/USDT"
    return s


def _parse_kline(data: dict) -> MarketDataPoint | None:
    """Parse a Binance kline WebSocket message into a MarketDataPoint.

    Only returns a point when the candle is closed (is_closed == True).
    """
    kline = data.get("k", {})
    if not kline.get("x", False):  # x = is this kline closed?
        return None

    timestamp = datetime.fromtimestamp(kline["t"] / 1000, tz=timezone.utc)
    symbol = _symbol_to_aegis(kline["s"])

    interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
    timeframe = interval_map.get(kline["i"], kline["i"])

    return MarketDataPoint(
        symbol=symbol,
        asset_class="crypto",
        timestamp=timestamp,
        timeframe=timeframe,
        open=float(kline["o"]),
        high=float(kline["h"]),
        low=float(kline["l"]),
        close=float(kline["c"]),
        volume=float(kline["v"]),
        source="binance",
    )


class BinanceWebSocketCollector:
    """Collects real-time kline data from Binance WebSocket."""

    def __init__(
        self,
        repository: MarketDataRepository,
        symbols: list[str] | None = None,
        testnet: bool = True,
        interval: str = "1m",
    ):
        self._repo = repository
        self._symbols = symbols or DEFAULT_SYMBOLS
        self._testnet = testnet
        self._interval = interval
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

    def _build_url(self) -> str:
        base = BINANCE_TESTNET_WS_URL if self._testnet else BINANCE_WS_URL
        streams = "/".join(f"{s}@kline_{self._interval}" for s in self._symbols)
        return f"{base}/{streams}"

    async def start(self) -> None:
        """Start collecting data. Reconnects on failure with exponential backoff."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                if not self._running:
                    break
                logger.warning(
                    "WebSocket disconnected: %s. Reconnecting in %.0fs...",
                    e, self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )

    async def _connect_and_listen(self) -> None:
        url = self._build_url()
        logger.info("Connecting to Binance WS: %s", url)

        async with websockets.connect(url) as ws:
            self._reconnect_delay = 1.0  # Reset on successful connect
            logger.info("Connected to Binance WebSocket")

            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    # Combined stream wraps in {"stream": ..., "data": ...}
                    if "data" in data:
                        data = data["data"]
                    point = _parse_kline(data)
                    if point is not None:
                        self._repo.insert_candle(point)
                        logger.debug("Stored candle: %s %s %s", point.symbol, point.timeframe, point.timestamp)
                except Exception as e:
                    logger.error("Error processing WS message: %s", e)

    def stop(self) -> None:
        """Signal the collector to stop."""
        self._running = False
