"""Backtest data loader: loads historical candles from DB, Binance, or yfinance."""

import logging
from datetime import datetime, timezone

import yfinance as yf

from aegis.common.types import MarketDataPoint

logger = logging.getLogger(__name__)

# Crypto base symbols for symbol routing
_CRYPTO_BASES = {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC"}


def is_crypto_symbol(symbol: str) -> bool:
    """Detect if a symbol is crypto (contains USDT or known crypto base)."""
    upper = symbol.upper()
    if "USDT" in upper:
        return True
    base = upper.split("/")[0]
    return base in _CRYPTO_BASES


def load_from_db(db, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[MarketDataPoint]:
    """Load candles from the market_data table."""
    rows = db.fetch_all(
        """
        SELECT symbol, asset_class, timestamp, timeframe,
               open, high, low, close, volume, source
        FROM market_data
        WHERE symbol = %s AND timeframe = %s
          AND timestamp >= %s AND timestamp <= %s
        ORDER BY timestamp ASC
        """,
        (symbol, timeframe, start, end),
    )
    return [
        MarketDataPoint(
            symbol=r["symbol"],
            asset_class=r["asset_class"],
            timestamp=r["timestamp"],
            timeframe=r["timeframe"],
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=float(r["volume"]),
            source=r["source"],
        )
        for r in rows
    ]


def download_from_binance(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start_str: str = "1 Jan, 2025",
    end_str: str = "1 Apr, 2026",
) -> list[MarketDataPoint]:
    """Download historical klines from Binance REST API.

    Returns list of MarketDataPoint (not saved to DB).
    """
    from binance.client import Client

    client = Client("", "")  # Public endpoint, no keys needed
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)

    aegis_symbol = _to_aegis_symbol(symbol)
    candles = []
    for k in klines:
        ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
        candles.append(
            MarketDataPoint(
                symbol=aegis_symbol,
                asset_class="crypto",
                timestamp=ts,
                timeframe=interval,
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                source="binance",
            )
        )
    logger.info("Downloaded %d candles for %s", len(candles), aegis_symbol)
    return candles


def download_from_yfinance(
    symbol: str = "AAPL",
    start: str = "2025-04-01",
    end: str = "2026-04-01",
    interval: str = "1d",
) -> list[MarketDataPoint]:
    """Download historical OHLCV from yfinance for equity symbols.

    Returns list of MarketDataPoint with asset_class='equity'.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval)

    if df.empty:
        logger.warning("No data from yfinance for %s", symbol)
        return []

    candles = []
    for idx, row in df.iterrows():
        ts = idx.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        candles.append(
            MarketDataPoint(
                symbol=symbol,
                asset_class="equity",
                timestamp=ts,
                timeframe=interval,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
                source="yfinance",
            )
        )
    logger.info("Downloaded %d candles for %s from yfinance", len(candles), symbol)
    return candles


def _to_aegis_symbol(binance_symbol: str) -> str:
    """Convert 'BTCUSDT' -> 'BTC/USDT'."""
    if binance_symbol.endswith("USDT"):
        base = binance_symbol[: -4]
        return f"{base}/USDT"
    return binance_symbol
