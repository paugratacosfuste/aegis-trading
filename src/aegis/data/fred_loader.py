"""FRED macro data loader — public CSV endpoint, no API key required.

FRED exposes every public series at

    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>

returning CSV of the form:

    DATE,<SERIES_ID>
    YYYY-MM-DD,<value or ".">
    ...

A literal "." marks missing observations (e.g. market-closed days) and must
be skipped — it is NOT a number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import httpx

from aegis.common.types import MacroDataPoint

logger = logging.getLogger(__name__)

# --- FRED series used for the Aegis 2.0 thesis layer ---
# These are the *core* macro series. Additional series (EPU, UNRATE, M2, WTI
# etc.) can be added to the observation builder without touching this loader.
FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

FRED_SERIES: dict[str, str] = {
    "yield_10y": "DGS10",
    "yield_2y": "DGS2",
    "vix": "VIXCLS",
    "dxy": "DTWEXBGS",
    "fed_rate": "DFF",
    "cpi": "CPIAUCSL",
}

# Fallback defaults when a series is unavailable. Chosen to be recent
# plausible values so macro agents degrade gracefully rather than emitting
# zero-valued signals.
_DEFAULT_VIX = 18.0
_DEFAULT_DXY = 103.0
_DEFAULT_10Y = 4.25
_DEFAULT_2Y = 4.00
_DEFAULT_FED_RATE = 4.50
_DEFAULT_CPI = 310.0

_HTTP_TIMEOUT = 20.0


class _ClientLike(Protocol):
    def get(self, url: str, timeout: float | None = None): ...


@dataclass(frozen=True)
class _FredRow:
    date_str: str
    value: float


def _classify_vix(vix: float) -> str:
    if vix < 15:
        return "low"
    if vix < 25:
        return "normal"
    if vix < 35:
        return "high"
    return "extreme"


def _parse_csv(body: str) -> list[_FredRow]:
    rows: list[_FredRow] = []
    for idx, raw in enumerate(body.splitlines()):
        if idx == 0:
            continue  # header
        parts = raw.strip().split(",")
        if len(parts) < 2:
            continue
        date_str, value_str = parts[0], parts[1]
        if value_str == "." or not date_str:
            continue
        try:
            value = float(value_str)
        except ValueError:
            continue
        rows.append(_FredRow(date_str=date_str, value=value))
    return rows


def fetch_fred_series(
    series_id: str,
    start: str,
    end: str,
    *,
    client: _ClientLike | None = None,
) -> dict[str, float]:
    """Fetch one FRED series and return a date_str -> float mapping.

    Dates outside ``[start, end)`` are filtered out. Missing rows ("."),
    malformed rows, and network failures all yield no entry for that date —
    never an exception — so a single flaky series cannot break the whole
    macro snapshot.
    """
    url = f"{FRED_BASE_URL}?id={series_id}"
    owns_client = client is None
    if client is None:
        client = httpx.Client()
    try:
        try:
            response = client.get(url, timeout=_HTTP_TIMEOUT)
            response.raise_for_status()
            body = response.text
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            logger.warning("FRED fetch failed for %s: %s", series_id, exc)
            return {}
    finally:
        if owns_client:
            try:
                client.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    rows = _parse_csv(body)
    filtered: dict[str, float] = {}
    for row in rows:
        if start <= row.date_str < end:
            filtered[row.date_str] = row.value
    logger.info("FRED %s: %d rows in [%s, %s)", series_id, len(filtered), start, end)
    return filtered


def _monthly_lookup(monthly: dict[str, float], day_str: str, fallback: float) -> float:
    """Return the most recent monthly value at or before ``day_str``."""
    if not monthly:
        return fallback
    target_month = day_str[:7]
    eligible = sorted(k for k in monthly if k[:7] <= target_month)
    if not eligible:
        return fallback
    return monthly[eligible[-1]]


def download_fred_macro_data(
    start: str,
    end: str,
    *,
    client: _ClientLike | None = None,
) -> list[MacroDataPoint]:
    """Build daily MacroDataPoint snapshots from FRED.

    Produces one snapshot per distinct date across the daily series. If every
    series is empty, returns an empty list (callers decide whether to fall
    back to yfinance). Monthly series (CPI) carry forward the latest prior
    monthly reading.
    """
    owns_client = client is None
    if client is None:
        client = httpx.Client()

    try:
        daily_series = {
            key: fetch_fred_series(FRED_SERIES[key], start, end, client=client)
            for key in ("yield_10y", "yield_2y", "vix", "dxy", "fed_rate")
        }
        # CPI is monthly — widen the window so we can always find a prior month.
        cpi_start = f"{int(start[:4]) - 1:04d}{start[4:]}"
        cpi_monthly = fetch_fred_series(
            FRED_SERIES["cpi"], cpi_start, end, client=client
        )
    finally:
        if owns_client:
            try:
                client.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    all_dates: set[str] = set()
    for s in daily_series.values():
        all_dates.update(s.keys())

    if not all_dates:
        logger.warning("FRED: no data returned for %s .. %s", start, end)
        return []

    snapshots: list[MacroDataPoint] = []
    for date_str in sorted(all_dates):
        yield_10y = daily_series["yield_10y"].get(date_str, _DEFAULT_10Y)
        yield_2y = daily_series["yield_2y"].get(date_str, _DEFAULT_2Y)
        vix = daily_series["vix"].get(date_str, _DEFAULT_VIX)
        dxy = daily_series["dxy"].get(date_str, _DEFAULT_DXY)
        fed_rate = daily_series["fed_rate"].get(date_str, _DEFAULT_FED_RATE)
        cpi = _monthly_lookup(cpi_monthly, date_str, _DEFAULT_CPI)

        ts = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        snapshots.append(
            MacroDataPoint(
                timestamp=ts,
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

    logger.info("FRED: built %d macro snapshots for %s .. %s", len(snapshots), start, end)
    return snapshots
