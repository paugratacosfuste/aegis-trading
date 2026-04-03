"""yfinance daily equity data collector."""

import logging
from datetime import datetime, timezone

import yfinance as yf

from aegis.common.types import MarketDataPoint
from aegis.data.repository import MarketDataRepository

logger = logging.getLogger(__name__)

DEFAULT_EQUITIES = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]


class YFinanceCollector:
    """Downloads daily OHLCV data for equities via yfinance."""

    def __init__(
        self,
        repository: MarketDataRepository,
        symbols: list[str] | None = None,
    ):
        self._repo = repository
        self._symbols = symbols or DEFAULT_EQUITIES

    def collect_daily(self, period: str = "5d") -> int:
        """Download recent daily data for all configured symbols.

        Returns the number of candles stored.
        """
        total = 0
        for symbol in self._symbols:
            try:
                count = self._collect_symbol(symbol, period)
                total += count
                logger.info("Collected %d candles for %s", count, symbol)
            except Exception as e:
                logger.error("Failed to collect %s: %s", symbol, e)
        return total

    def _collect_symbol(self, symbol: str, period: str) -> int:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d")

        if df.empty:
            return 0

        points = []
        for idx, row in df.iterrows():
            ts = idx.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            points.append(
                MarketDataPoint(
                    symbol=symbol,
                    asset_class="equity",
                    timestamp=ts,
                    timeframe="1d",
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    source="yfinance",
                )
            )

        if points:
            self._repo.insert_candles_batch(points)

        return len(points)

    def collect_historical(self, symbol: str, start: str, end: str) -> int:
        """Download historical daily data for a single symbol.

        start/end format: 'YYYY-MM-DD'
        """
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval="1d")

        if df.empty:
            return 0

        points = []
        for idx, row in df.iterrows():
            ts = idx.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            points.append(
                MarketDataPoint(
                    symbol=symbol,
                    asset_class="equity",
                    timestamp=ts,
                    timeframe="1d",
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    source="yfinance",
                )
            )

        if points:
            self._repo.insert_candles_batch(points)

        return len(points)
