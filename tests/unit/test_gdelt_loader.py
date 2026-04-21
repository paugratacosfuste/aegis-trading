"""Tests for GDELT 2.0 events loader.

GDELT exposes event snapshots every 15 minutes at:
    http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.export.CSV.zip

Each zip contains one tab-separated CSV (no header) with 61 columns. These
tests construct fake zip bytes in-memory so they run fully offline.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone

import pytest

from aegis.common.types import GeopoliticalEvent


# --- helpers ---------------------------------------------------------------


def _make_row(
    *,
    event_id: str = "1",
    sqldate: str = "20250102",
    actor1_country: str = "USA",
    actor2_country: str = "CHN",
    event_code: str = "190",  # CAMEO: use conventional military force
    quad_class: str = "4",  # material conflict
    goldstein: str = "-8.0",
    num_mentions: str = "25",
    avg_tone: str = "-6.5",
    action_country: str = "USA",
    source_url: str = "https://example.com/story",
) -> str:
    """Build one tab-separated GDELT row. 61 columns, indices matter."""
    cols = [""] * 61
    cols[0] = event_id
    cols[1] = sqldate
    cols[7] = actor1_country
    cols[17] = actor2_country
    cols[26] = event_code
    cols[29] = quad_class
    cols[30] = goldstein
    cols[31] = num_mentions
    cols[34] = avg_tone
    cols[53] = action_country
    cols[59] = sqldate + "120000"
    cols[60] = source_url
    return "\t".join(cols)


def _zip_bytes(rows: list[str]) -> bytes:
    """Wrap GDELT rows in the ZIP-of-one-TSV format the real endpoint serves."""
    csv_body = "\n".join(rows) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("events.export.CSV", csv_body)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    """Fake httpx-style client keyed by URL substring -> zip bytes / Exception."""

    def __init__(self, responses: dict[str, bytes | Exception]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def get(self, url: str, timeout: float | None = None) -> _FakeResponse:
        self.calls.append(url)
        for key, body in self._responses.items():
            if key in url:
                if isinstance(body, Exception):
                    raise body
                return _FakeResponse(body)
        return _FakeResponse(b"", status_code=404)

    def close(self) -> None:
        pass


# --- parser ----------------------------------------------------------------


class TestParseGdeltZip:
    def test_parses_high_severity_event(self):
        from aegis.data.gdelt_loader import parse_gdelt_zip

        row = _make_row(event_id="42", goldstein="-9.0", avg_tone="-7.0")
        events = parse_gdelt_zip(_zip_bytes([row]))

        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, GeopoliticalEvent)
        assert ev.event_id == "42"
        assert ev.source == "gdelt"
        assert ev.severity == pytest.approx(0.9)  # |-9|/10
        assert ev.sentiment_score == pytest.approx(-0.07)  # -7.0 / 100
        assert ev.timestamp.tzinfo is timezone.utc

    def test_filters_out_low_severity_events(self):
        """High-volume noise (|goldstein| < threshold) must be dropped.
        Otherwise the downstream agents drown in market-irrelevant chatter."""
        from aegis.data.gdelt_loader import parse_gdelt_zip

        rows = [
            _make_row(event_id="1", goldstein="-1.0"),  # below threshold → drop
            _make_row(event_id="2", goldstein="-7.5"),  # keep
            _make_row(event_id="3", goldstein="0.5"),   # drop
        ]
        events = parse_gdelt_zip(_zip_bytes(rows))

        assert {e.event_id for e in events} == {"2"}

    def test_assigns_category_from_quad_class(self):
        from aegis.data.gdelt_loader import parse_gdelt_zip

        rows = [
            _make_row(event_id="c1", quad_class="4", goldstein="-8.0"),
            _make_row(event_id="c2", quad_class="3", goldstein="-8.0"),
            _make_row(event_id="c3", quad_class="2", goldstein="-8.0"),
            _make_row(event_id="c4", quad_class="1", goldstein="-8.0"),
        ]
        events = parse_gdelt_zip(_zip_bytes(rows))
        by_id = {e.event_id: e for e in events}

        assert by_id["c1"].category == "conflict"
        assert by_id["c2"].category == "conflict"
        assert by_id["c3"].category in ("policy", "cooperation")
        assert by_id["c4"].category in ("policy", "cooperation")

    def test_affected_regions_includes_both_actors(self):
        from aegis.data.gdelt_loader import parse_gdelt_zip

        row = _make_row(
            event_id="r1",
            actor1_country="RUS",
            actor2_country="UKR",
            goldstein="-9.5",
        )
        events = parse_gdelt_zip(_zip_bytes([row]))

        assert len(events) == 1
        regions = set(events[0].affected_regions)
        assert "RUS" in regions
        assert "UKR" in regions

    def test_malformed_row_is_skipped(self):
        from aegis.data.gdelt_loader import parse_gdelt_zip

        rows = [
            "short\trow",  # too few columns
            _make_row(event_id="ok", goldstein="-9.0"),
        ]
        events = parse_gdelt_zip(_zip_bytes(rows))
        assert len(events) == 1
        assert events[0].event_id == "ok"

    def test_empty_zip_returns_empty(self):
        from aegis.data.gdelt_loader import parse_gdelt_zip
        assert parse_gdelt_zip(_zip_bytes([])) == []

    def test_invalid_zip_returns_empty(self):
        """Corrupted/non-zip bytes must not crash — we want one bad day to
        degrade gracefully, not blow up the whole backtest loader."""
        from aegis.data.gdelt_loader import parse_gdelt_zip
        assert parse_gdelt_zip(b"not a zip file") == []

    def test_prefers_dateadded_over_sqldate(self):
        """GDELT files often re-surface older events. DATEADDED (when the
        record was catalogued) is the correct 'when could we have seen this'
        timestamp for a live-running system, not the original SQLDATE."""
        from aegis.data.gdelt_loader import parse_gdelt_zip

        # Old story re-mentioned: SQLDATE is 2022, but catalogued on 2025-01-02.
        row = _make_row(event_id="rev", sqldate="20220315", goldstein="-8.0")
        cols = row.split("\t")
        cols[59] = "20250102120000"
        row = "\t".join(cols)

        events = parse_gdelt_zip(_zip_bytes([row]))

        assert len(events) == 1
        assert events[0].timestamp.date() == datetime(2025, 1, 2).date()


# --- downloader ------------------------------------------------------------


class TestDownloadGdeltEvents:
    def test_fetches_one_file_per_day_in_range(self):
        from aegis.data.gdelt_loader import download_gdelt_events

        # Each 8-char date prefix keys its own day.
        responses = {
            "20250102": _zip_bytes([_make_row(event_id="a", goldstein="-8.0")]),
            "20250103": _zip_bytes([_make_row(event_id="b", goldstein="-9.0")]),
        }
        client = _FakeClient(responses)

        events = download_gdelt_events("2025-01-02", "2025-01-04", client=client)

        assert {e.event_id for e in events} == {"a", "b"}
        # One fetch per day in the range
        assert len(client.calls) == 2

    def test_skips_days_with_network_error(self):
        from aegis.data.gdelt_loader import download_gdelt_events

        responses = {
            "20250102": RuntimeError("boom"),  # day 1 fails
            "20250103": _zip_bytes([_make_row(event_id="b", goldstein="-9.0")]),
        }
        client = _FakeClient(responses)

        events = download_gdelt_events("2025-01-02", "2025-01-04", client=client)

        assert {e.event_id for e in events} == {"b"}

    def test_empty_range_returns_empty(self):
        from aegis.data.gdelt_loader import download_gdelt_events

        client = _FakeClient({})
        events = download_gdelt_events("2025-01-02", "2025-01-02", client=client)
        assert events == []

    def test_events_sorted_by_timestamp(self):
        from aegis.data.gdelt_loader import download_gdelt_events

        responses = {
            "20250104": _zip_bytes(
                [_make_row(event_id="late", sqldate="20250104", goldstein="-8.0")]
            ),
            "20250102": _zip_bytes(
                [_make_row(event_id="early", sqldate="20250102", goldstein="-8.0")]
            ),
        }
        client = _FakeClient(responses)

        events = download_gdelt_events("2025-01-02", "2025-01-05", client=client)

        ids = [e.event_id for e in events]
        assert ids == ["early", "late"]

    def test_timestamps_are_utc(self):
        from aegis.data.gdelt_loader import download_gdelt_events

        responses = {
            "20250102": _zip_bytes([_make_row(event_id="x", goldstein="-8.0")]),
        }
        client = _FakeClient(responses)

        events = download_gdelt_events("2025-01-02", "2025-01-03", client=client)
        assert events[0].timestamp.tzinfo is timezone.utc
        assert events[0].timestamp.date() == datetime(2025, 1, 2).date()
