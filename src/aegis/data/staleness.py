"""Data staleness checker. Determines if market data is too old to trust."""

from datetime import datetime, timezone

from aegis.common.db import DatabasePool


class StalenessChecker:
    def __init__(
        self,
        db: DatabasePool,
        crypto_max_sec: float = 30,
        equity_market_max_sec: float = 300,
        equity_after_hours_max_sec: float = 86400,
    ):
        self._db = db
        self._crypto_max = crypto_max_sec
        self._equity_market_max = equity_market_max_sec
        self._equity_after_hours_max = equity_after_hours_max_sec

    def check(
        self,
        symbol: str,
        asset_class: str,
        market_open: bool = True,
    ) -> tuple[bool, float]:
        """Check if data for a symbol is stale.

        Returns (is_stale, age_in_seconds). If no data exists, age is inf.
        """
        row = self._db.fetch_one(
            "SELECT MAX(timestamp) FROM market_data WHERE symbol = %s",
            (symbol,),
        )

        if row is None or row["max"] is None:
            return True, float("inf")

        latest_ts = row["max"]
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.replace(tzinfo=timezone.utc)

        age_sec = (datetime.now(timezone.utc) - latest_ts).total_seconds()

        if asset_class == "crypto":
            max_age = self._crypto_max
        elif market_open:
            max_age = self._equity_market_max
        else:
            max_age = self._equity_after_hours_max

        return age_sec > max_age, age_sec
