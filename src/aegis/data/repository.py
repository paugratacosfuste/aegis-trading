"""Market data repository. Raw SQL reads/writes to the market_data table."""

from datetime import datetime

from aegis.common.db import DatabasePool
from aegis.common.types import MarketDataPoint


class MarketDataRepository:
    def __init__(self, db: DatabasePool):
        self._db = db

    def insert_candle(self, point: MarketDataPoint) -> None:
        """Insert a single candle. Ignores duplicates via UNIQUE constraint."""
        self._db.execute(
            """
            INSERT INTO market_data
                (symbol, asset_class, timestamp, timeframe, open, high, low, close, volume, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, timeframe, timestamp, source) DO NOTHING
            """,
            (
                point.symbol,
                point.asset_class,
                point.timestamp,
                point.timeframe,
                point.open,
                point.high,
                point.low,
                point.close,
                point.volume,
                point.source,
            ),
        )

    def insert_candles_batch(self, points: list[MarketDataPoint]) -> None:
        """Insert multiple candles in one transaction."""
        if not points:
            return
        self._db.execute_many(
            """
            INSERT INTO market_data
                (symbol, asset_class, timestamp, timeframe, open, high, low, close, volume, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, timeframe, timestamp, source) DO NOTHING
            """,
            [
                (
                    p.symbol, p.asset_class, p.timestamp, p.timeframe,
                    p.open, p.high, p.low, p.close, p.volume, p.source,
                )
                for p in points
            ],
        )

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[MarketDataPoint]:
        """Get candles for a symbol within a time range, ordered by timestamp."""
        rows = self._db.fetch_all(
            """
            SELECT symbol, asset_class, timestamp, timeframe,
                   open, high, low, close, volume, source
            FROM market_data
            WHERE symbol = %s AND timeframe = %s AND timestamp >= %s AND timestamp <= %s
            ORDER BY timestamp ASC
            """,
            (symbol, timeframe, start, end),
        )
        return [self._row_to_point(row) for row in rows]

    def get_latest_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> list[MarketDataPoint]:
        """Get the most recent N candles for a symbol, ordered oldest first."""
        rows = self._db.fetch_all(
            """
            SELECT symbol, asset_class, timestamp, timeframe,
                   open, high, low, close, volume, source
            FROM (
                SELECT * FROM market_data
                WHERE symbol = %s AND timeframe = %s
                ORDER BY timestamp DESC
                LIMIT %s
            ) sub
            ORDER BY timestamp ASC
            """,
            (symbol, timeframe, limit),
        )
        return [self._row_to_point(row) for row in rows]

    def get_latest_price(self, symbol: str) -> float | None:
        """Get the most recent close price for a symbol."""
        row = self._db.fetch_one(
            """
            SELECT close FROM market_data
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (symbol,),
        )
        return row[0] if row else None

    @staticmethod
    def _row_to_point(row: tuple) -> MarketDataPoint:
        return MarketDataPoint(
            symbol=row[0],
            asset_class=row[1],
            timestamp=row[2],
            timeframe=row[3],
            open=row[4],
            high=row[5],
            low=row[6],
            close=row[7],
            volume=row[8],
            source=row[9],
        )
