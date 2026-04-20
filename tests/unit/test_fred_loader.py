"""Tests for FRED macro loader.

FRED exposes a public CSV endpoint at
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>
which returns:
    DATE,<SERIES_ID>
    YYYY-MM-DD,<value or ".">
    ...
No API key required. These tests use an in-process fake HTTP client so they
run fully offline.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aegis.common.types import MacroDataPoint


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    """Fake httpx-style client. Maps series_id -> CSV body (or Exception)."""

    def __init__(self, responses: dict[str, str | Exception]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def get(self, url: str, timeout: float | None = None) -> _FakeResponse:
        self.calls.append(url)
        for series_id, body in self._responses.items():
            if f"id={series_id}" in url:
                if isinstance(body, Exception):
                    raise body
                return _FakeResponse(body)
        return _FakeResponse("", status_code=404)

    def close(self) -> None:
        pass

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _csv(series_id: str, rows: list[tuple[str, str]]) -> str:
    header = f"DATE,{series_id}"
    lines = [header] + [f"{d},{v}" for d, v in rows]
    return "\n".join(lines) + "\n"


class TestFetchFredSeries:
    def test_parses_csv_rows(self):
        from aegis.data.fred_loader import fetch_fred_series

        client = _FakeClient(
            {
                "DGS10": _csv(
                    "DGS10",
                    [("2025-01-02", "4.12"), ("2025-01-03", "4.15")],
                )
            }
        )

        values = fetch_fred_series("DGS10", "2025-01-01", "2025-02-01", client=client)

        assert values == {"2025-01-02": 4.12, "2025-01-03": 4.15}
        assert any("id=DGS10" in c for c in client.calls)

    def test_skips_missing_dot_values(self):
        """FRED marks missing data as '.' — those rows must be skipped, not parsed."""
        from aegis.data.fred_loader import fetch_fred_series

        client = _FakeClient(
            {
                "VIXCLS": _csv(
                    "VIXCLS",
                    [
                        ("2025-01-02", "18.5"),
                        ("2025-01-03", "."),  # market closed / missing
                        ("2025-01-06", "19.0"),
                    ],
                )
            }
        )

        values = fetch_fred_series("VIXCLS", "2025-01-01", "2025-02-01", client=client)

        assert values == {"2025-01-02": 18.5, "2025-01-06": 19.0}

    def test_filters_out_of_range_dates(self):
        from aegis.data.fred_loader import fetch_fred_series

        client = _FakeClient(
            {
                "DGS2": _csv(
                    "DGS2",
                    [
                        ("2024-12-20", "4.00"),  # before start
                        ("2025-01-05", "4.10"),  # in range
                        ("2025-02-10", "4.20"),  # after end
                    ],
                )
            }
        )

        values = fetch_fred_series("DGS2", "2025-01-01", "2025-02-01", client=client)

        assert values == {"2025-01-05": 4.10}

    def test_network_failure_returns_empty(self):
        """A failing fetch should yield an empty dict, not raise — so other
        series can still populate the macro snapshot."""
        from aegis.data.fred_loader import fetch_fred_series

        client = _FakeClient({"DGS10": RuntimeError("boom")})

        values = fetch_fred_series("DGS10", "2025-01-01", "2025-02-01", client=client)

        assert values == {}

    def test_malformed_row_is_skipped(self):
        from aegis.data.fred_loader import fetch_fred_series

        # bad number "abc" and short row must not crash
        body = "DATE,DGS10\n2025-01-02,4.10\n2025-01-03,abc\nshortrow\n"
        client = _FakeClient({"DGS10": body})

        values = fetch_fred_series("DGS10", "2025-01-01", "2025-02-01", client=client)

        assert values == {"2025-01-02": 4.10}


class TestDownloadFredMacroData:
    def _client_with_all_series(self) -> _FakeClient:
        return _FakeClient(
            {
                "DGS10": _csv("DGS10", [("2025-01-02", "4.20"), ("2025-01-03", "4.22")]),
                "DGS2": _csv("DGS2", [("2025-01-02", "4.00"), ("2025-01-03", "4.05")]),
                "VIXCLS": _csv("VIXCLS", [("2025-01-02", "18.0"), ("2025-01-03", "28.0")]),
                "DTWEXBGS": _csv("DTWEXBGS", [("2025-01-02", "103.5"), ("2025-01-03", "104.0")]),
                "DFF": _csv("DFF", [("2025-01-02", "4.33"), ("2025-01-03", "4.33")]),
                "CPIAUCSL": _csv("CPIAUCSL", [("2024-12-01", "310.0")]),  # monthly
            }
        )

    def test_builds_macro_snapshots_per_day(self):
        from aegis.data.fred_loader import download_fred_macro_data

        client = self._client_with_all_series()
        snaps = download_fred_macro_data("2025-01-01", "2025-02-01", client=client)

        assert len(snaps) == 2
        assert all(isinstance(s, MacroDataPoint) for s in snaps)
        # Sorted by timestamp
        assert snaps[0].timestamp < snaps[1].timestamp

    def test_yield_spread_is_computed(self):
        from aegis.data.fred_loader import download_fred_macro_data

        client = self._client_with_all_series()
        snaps = download_fred_macro_data("2025-01-01", "2025-02-01", client=client)

        # 4.20 - 4.00 = 0.20 on 2025-01-02
        day1 = snaps[0]
        assert day1.yield_10y == pytest.approx(4.20)
        assert day1.yield_2y == pytest.approx(4.00)
        assert day1.yield_spread == pytest.approx(0.20)

    def test_vix_regime_is_classified(self):
        from aegis.data.fred_loader import download_fred_macro_data

        client = self._client_with_all_series()
        snaps = download_fred_macro_data("2025-01-01", "2025-02-01", client=client)

        assert snaps[0].vix_regime == "normal"  # 18.0
        assert snaps[1].vix_regime == "high"  # 28.0

    def test_timestamps_are_utc(self):
        from aegis.data.fred_loader import download_fred_macro_data

        client = self._client_with_all_series()
        snaps = download_fred_macro_data("2025-01-01", "2025-02-01", client=client)

        for s in snaps:
            assert s.timestamp.tzinfo is timezone.utc

    def test_falls_back_when_series_missing(self):
        """If a series fails entirely, the loader must still produce snapshots
        using default values — partial data is better than no data."""
        from aegis.data.fred_loader import download_fred_macro_data

        # DGS2 missing → yield_2y falls back, yield_spread still computable
        client = _FakeClient(
            {
                "DGS10": _csv("DGS10", [("2025-01-02", "4.20")]),
                "DGS2": RuntimeError("fail"),
                "VIXCLS": _csv("VIXCLS", [("2025-01-02", "18.0")]),
                "DTWEXBGS": _csv("DTWEXBGS", [("2025-01-02", "103.5")]),
                "DFF": _csv("DFF", [("2025-01-02", "4.33")]),
                "CPIAUCSL": _csv("CPIAUCSL", [("2024-12-01", "310.0")]),
            }
        )

        snaps = download_fred_macro_data("2025-01-01", "2025-02-01", client=client)

        assert len(snaps) == 1
        assert snaps[0].yield_10y == pytest.approx(4.20)
        # yield_2y uses the default fallback (non-zero, sensible)
        assert snaps[0].yield_2y > 0

    def test_empty_when_no_series_returns_data(self):
        from aegis.data.fred_loader import download_fred_macro_data

        client = _FakeClient({})  # every call 404s
        snaps = download_fred_macro_data("2025-01-01", "2025-02-01", client=client)

        assert snaps == []

    def test_cpi_uses_latest_available_month(self):
        """CPI is monthly. Daily snapshots should carry the latest prior month's value."""
        from aegis.data.fred_loader import download_fred_macro_data

        client = _FakeClient(
            {
                "DGS10": _csv("DGS10", [("2025-03-03", "4.30"), ("2025-03-04", "4.31")]),
                "DGS2": _csv("DGS2", [("2025-03-03", "4.10"), ("2025-03-04", "4.11")]),
                "VIXCLS": _csv("VIXCLS", [("2025-03-03", "17.0"), ("2025-03-04", "17.5")]),
                "DTWEXBGS": _csv(
                    "DTWEXBGS", [("2025-03-03", "104.0"), ("2025-03-04", "104.1")]
                ),
                "DFF": _csv("DFF", [("2025-03-03", "4.33"), ("2025-03-04", "4.33")]),
                "CPIAUCSL": _csv(
                    "CPIAUCSL",
                    [("2025-01-01", "308.0"), ("2025-02-01", "310.0")],
                ),
            }
        )

        snaps = download_fred_macro_data("2025-03-01", "2025-04-01", client=client)

        # Both March snapshots should use the most recent available CPI (Feb).
        assert all(s.cpi_latest == pytest.approx(310.0) for s in snaps)
