"""Download macro indicators from yfinance for backtest.

No API key required. Uses public yfinance tickers:
- VIX: ^VIX (volatility index)
- DXY: DX-Y.NYB (US dollar index)
- 10Y yield: ^TNX (CBOE 10-year Treasury note)
- 13-week yield: ^IRX (CBOE 13-week Treasury bill, proxy for short end)

CPI and Fed rate use recent known values (change infrequently).
"""

import logging
from datetime import datetime, timezone

import yfinance as yf

from aegis.common.types import MacroDataPoint

logger = logging.getLogger(__name__)

# Monthly CPI year-over-year % (Bureau of Labor Statistics)
# Updated monthly. Use latest known value for forward dates.
_CPI_MONTHLY: dict[str, float] = {
    "2024-10": 2.6, "2024-11": 2.7, "2024-12": 2.9,
    "2025-01": 3.0, "2025-02": 2.8, "2025-03": 2.4,
}

# Fed funds target rate upper bound (FOMC decisions)
_FED_RATE_CHANGES: list[tuple[str, float]] = [
    ("2024-09-18", 5.00), ("2024-11-07", 4.75), ("2024-12-18", 4.50),
    ("2025-01-29", 4.50), ("2025-03-19", 4.50),
]

# Fallback defaults when data is unavailable
_DEFAULT_CPI = 2.8
_DEFAULT_FED_RATE = 4.50
_DEFAULT_VIX = 18.0
_DEFAULT_DXY = 103.0
_DEFAULT_10Y = 4.25
_DEFAULT_2Y = 4.00


def _classify_vix(vix: float) -> str:
    if vix < 15:
        return "low"
    if vix < 25:
        return "normal"
    if vix < 35:
        return "high"
    return "extreme"


def _get_cpi_for_date(dt: datetime) -> float:
    """Return CPI value for a given date (uses latest known month)."""
    key = dt.strftime("%Y-%m")
    if key in _CPI_MONTHLY:
        return _CPI_MONTHLY[key]
    # Find the latest available month before this date
    available = sorted(k for k in _CPI_MONTHLY if k <= key)
    return _CPI_MONTHLY[available[-1]] if available else _DEFAULT_CPI


def _get_fed_rate_for_date(dt: datetime) -> float:
    """Return Fed funds rate for a given date."""
    date_str = dt.strftime("%Y-%m-%d")
    rate = _DEFAULT_FED_RATE
    for change_date, change_rate in _FED_RATE_CHANGES:
        if change_date <= date_str:
            rate = change_rate
    return rate


def download_macro_data(
    start: str = "2025-04-01",
    end: str = "2026-04-01",
) -> list[MacroDataPoint]:
    """Download macro indicators for backtest period.

    Returns one MacroDataPoint per trading day, sorted by timestamp.
    Falls back to defaults for any missing data.
    """
    tickers = {
        "vix": "^VIX",
        "dxy": "DX-Y.NYB",
        "tnx": "^TNX",
        "irx": "^IRX",
    }

    dfs: dict[str, dict[str, float]] = {}
    for name, ticker in tickers.items():
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start, end=end, interval="1d")
            if not df.empty:
                dfs[name] = {
                    idx.strftime("%Y-%m-%d"): float(row["Close"])
                    for idx, row in df.iterrows()
                }
                logger.info("Downloaded %d rows for %s (%s)", len(dfs[name]), name, ticker)
            else:
                logger.warning("No data for %s (%s)", name, ticker)
                dfs[name] = {}
        except Exception as e:
            logger.warning("Failed to download %s: %s", ticker, e)
            dfs[name] = {}

    # Build date union from all available data
    all_dates: set[str] = set()
    for name_data in dfs.values():
        all_dates.update(name_data.keys())

    if not all_dates:
        logger.warning("No macro data downloaded for %s to %s", start, end)
        return []

    snapshots: list[MacroDataPoint] = []
    for date_str in sorted(all_dates):
        vix = dfs.get("vix", {}).get(date_str, _DEFAULT_VIX)
        dxy = dfs.get("dxy", {}).get(date_str, _DEFAULT_DXY)
        yield_10y = dfs.get("tnx", {}).get(date_str, _DEFAULT_10Y)
        yield_2y = dfs.get("irx", {}).get(date_str, _DEFAULT_2Y)

        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        cpi = _get_cpi_for_date(dt)
        fed_rate = _get_fed_rate_for_date(dt)

        snapshots.append(
            MacroDataPoint(
                timestamp=dt,
                yield_10y=yield_10y,
                yield_2y=yield_2y,
                yield_spread=yield_10y - yield_2y,
                vix=vix,
                vix_regime=_classify_vix(vix),
                dxy=dxy,
                fed_rate=fed_rate,
                cpi_latest=cpi,
            )
        )

    logger.info("Built %d macro snapshots for %s to %s", len(snapshots), start, end)
    return snapshots
