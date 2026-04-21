"""GDELT 2.0 Events loader — public zip/CSV endpoint, no API key required.

GDELT publishes an events snapshot every 15 minutes at
    http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.export.CSV.zip

Each zip contains one tab-separated CSV (no header, 61 columns). For the
thesis layer we fetch one representative file per day (noon UTC), filter to
high-severity events via the Goldstein scale, and convert each row into a
``GeopoliticalEvent``.

Column indices used (GDELT 2.0 schema, see GDELT-Event_Codebook-V2.0.pdf):
    0   GLOBALEVENTID
    1   SQLDATE             (YYYYMMDD)
    7   Actor1CountryCode
    17  Actor2CountryCode
    26  EventCode           (CAMEO)
    29  QuadClass           (1/2 cooperation, 3/4 conflict)
    30  GoldsteinScale      (-10..+10, net impact)
    31  NumMentions
    34  AvgTone             (-100..+100)
    53  ActionGeo_CountryCode
    59  DATEADDED           (YYYYMMDDHHMMSS)
    60  SOURCEURL
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Protocol

import httpx

from aegis.common.types import GeopoliticalEvent

logger = logging.getLogger(__name__)

GDELT_BASE_URL = "http://data.gdeltproject.org/gdeltv2"
_HTTP_TIMEOUT = 30.0

# Drop everything with |GoldsteinScale| below this. GDELT publishes ~20k
# events per 15-min file; the vast majority are low-impact chatter. A cutoff
# of 5 keeps diplomatic sanctions, conflict, and acute policy events while
# discarding the noise floor.
_GOLDSTEIN_SEVERITY_MIN = 5.0

# Minimum row length to parse safely — indexes up to 60 are read.
_MIN_ROW_COLS = 61

# Representative slot fetched per day. The daily file set is dense — choosing
# one 15-min slot keeps the request volume cheap for backtests while
# preserving the events that were active at midday UTC.
_DAILY_SLOT = "120000"


class _ClientLike(Protocol):
    def get(self, url: str, timeout: float | None = None): ...


def _quad_to_category(quad: str) -> str:
    if quad in ("3", "4"):
        return "conflict"
    if quad in ("1", "2"):
        return "cooperation"
    return "other"


def _half_life_for(severity: float, category: str) -> int:
    # Acute conflict stays in the tape longer than routine diplomatic moves.
    if category == "conflict" and severity >= 0.8:
        return 72
    if category == "conflict":
        return 48
    if category == "cooperation":
        return 12
    return 24


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_gdelt_timestamp(date_added: str, sqldate: str) -> datetime | None:
    """Parse DATEADDED (YYYYMMDDHHMMSS). Fall back to SQLDATE (YYYYMMDD)."""
    if date_added and len(date_added) == 14:
        try:
            return datetime.strptime(date_added, "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
    if sqldate and len(sqldate) == 8:
        try:
            return datetime.strptime(sqldate, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _parse_row(row: str) -> GeopoliticalEvent | None:
    cols = row.split("\t")
    if len(cols) < _MIN_ROW_COLS:
        return None

    goldstein = _safe_float(cols[30])
    if abs(goldstein) < _GOLDSTEIN_SEVERITY_MIN:
        return None

    # Prefer DATEADDED (when GDELT catalogued the record) — that's when a
    # live-running system would first observe the event. Fall back to SQLDATE
    # only if DATEADDED is missing/malformed.
    ts = _parse_gdelt_timestamp(cols[59], cols[1])
    if ts is None:
        return None

    avg_tone = _safe_float(cols[34])
    sentiment = max(-1.0, min(1.0, avg_tone / 100.0))
    severity = min(1.0, abs(goldstein) / 10.0)
    category = _quad_to_category(cols[29])

    actor1 = cols[7].strip()
    actor2 = cols[17].strip()
    regions: tuple[str, ...] = tuple(c for c in (actor1, actor2) if c)

    return GeopoliticalEvent(
        event_id=cols[0].strip() or f"gdelt:{sqldate}",
        timestamp=ts,
        source="gdelt",
        category=category,
        severity=severity,
        affected_sectors=(),
        affected_regions=regions,
        raw_text=cols[60].strip(),
        sentiment_score=sentiment,
        half_life_hours=_half_life_for(severity, category),
    )


def parse_gdelt_zip(body: bytes) -> list[GeopoliticalEvent]:
    """Parse the zip-of-TSV payload GDELT returns. Returns filtered events.

    Swallows corrupt/empty zips — a single bad day should not break a
    multi-day download.
    """
    if not body:
        return []
    try:
        buf = io.BytesIO(body)
        with zipfile.ZipFile(buf) as zf:
            names = zf.namelist()
            if not names:
                return []
            csv_bytes = zf.read(names[0])
    except zipfile.BadZipFile:
        logger.warning("GDELT parse: not a valid zip (%d bytes)", len(body))
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("GDELT parse failed: %s", exc)
        return []

    text = csv_bytes.decode("utf-8", errors="replace")
    events: list[GeopoliticalEvent] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        ev = _parse_row(raw)
        if ev is not None:
            events.append(ev)
    return events


def _fetch_day(
    date_str_compact: str, client: _ClientLike
) -> list[GeopoliticalEvent]:
    """Fetch and parse a single GDELT daily slot. Empty list on any failure."""
    url = f"{GDELT_BASE_URL}/{date_str_compact}{_DAILY_SLOT}.export.CSV.zip"
    try:
        response = client.get(url, timeout=_HTTP_TIMEOUT)
        response.raise_for_status()
        body = response.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("GDELT fetch failed for %s: %s", date_str_compact, exc)
        return []
    return parse_gdelt_zip(body)


def _daterange(start: str, end: str) -> list[datetime]:
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days: list[datetime] = []
    cur = start_dt
    while cur < end_dt:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def download_gdelt_events(
    start: str,
    end: str,
    *,
    client: _ClientLike | None = None,
) -> list[GeopoliticalEvent]:
    """Download geopolitical events in ``[start, end)`` (YYYY-MM-DD).

    One file per day (noon UTC slot). Events returned sorted by timestamp.
    Per-day failures are logged and skipped so a flaky day cannot abort the
    whole backtest range.
    """
    owns_client = client is None
    if client is None:
        client = httpx.Client()

    try:
        all_events: list[GeopoliticalEvent] = []
        for day in _daterange(start, end):
            date_compact = day.strftime("%Y%m%d")
            all_events.extend(_fetch_day(date_compact, client))
    finally:
        if owns_client:
            try:
                client.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    all_events.sort(key=lambda e: e.timestamp)
    logger.info(
        "GDELT: %d events across %s .. %s", len(all_events), start, end
    )
    return all_events
