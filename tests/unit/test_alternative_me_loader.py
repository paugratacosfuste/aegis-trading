"""Tests for alternative.me Fear & Greed Index loader.

alternative.me publishes a daily Crypto Fear & Greed reading at
    https://api.alternative.me/fng/?limit=<N>

Response shape:
    {
      "name": "Fear and Greed Index",
      "data": [
        {"value": "45", "value_classification": "Fear",
         "timestamp": "1704067200", "time_until_update": "..."},
        ...
      ]
    }

0 = extreme fear, 100 = extreme greed, 50 = neutral. We normalise to the
standard [-1, +1] sentiment scale used across the codebase.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from aegis.common.types import SentimentDataPoint


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    def __init__(self, payload: dict | Exception) -> None:
        self._payload = payload
        self.calls: list[str] = []

    def get(self, url: str, timeout: float | None = None) -> _FakeResponse:
        self.calls.append(url)
        if isinstance(self._payload, Exception):
            raise self._payload
        return _FakeResponse(json.dumps(self._payload))

    def close(self) -> None:
        pass


def _api_payload(entries: list[tuple[int, int, str]]) -> dict:
    """entries = list of (timestamp_seconds, value_int, classification)."""
    return {
        "name": "Fear and Greed Index",
        "data": [
            {
                "value": str(v),
                "value_classification": cls,
                "timestamp": str(ts),
                "time_until_update": "12345",
            }
            for (ts, v, cls) in entries
        ],
    }


class TestFearGreedLoader:
    def test_parses_entries_into_sentiment_points(self):
        from aegis.data.alternative_me_loader import download_fear_greed

        payload = _api_payload(
            [
                (1704067200, 45, "Fear"),      # 2024-01-01
                (1703980800, 72, "Greed"),     # 2023-12-31
            ]
        )
        client = _FakeClient(payload)

        points = download_fear_greed(limit=2, client=client)

        assert len(points) == 2
        assert all(isinstance(p, SentimentDataPoint) for p in points)

    def test_normalizes_value_to_sentiment_scale(self):
        """0/50/100 → -1/0/+1 so downstream agents can consume it like any
        other sentiment source."""
        from aegis.data.alternative_me_loader import download_fear_greed

        payload = _api_payload(
            [
                (1704067200, 0, "Extreme Fear"),
                (1703980800, 50, "Neutral"),
                (1703894400, 100, "Extreme Greed"),
            ]
        )
        client = _FakeClient(payload)

        points = download_fear_greed(limit=3, client=client)
        by_score = {round(p.sentiment_score, 3): p for p in points}

        assert -1.0 in by_score
        assert 0.0 in by_score
        assert 1.0 in by_score

    def test_timestamps_are_utc(self):
        from aegis.data.alternative_me_loader import download_fear_greed

        # 1704067200 = 2024-01-01 00:00 UTC
        payload = _api_payload([(1704067200, 45, "Fear")])
        client = _FakeClient(payload)

        points = download_fear_greed(limit=1, client=client)

        assert points[0].timestamp.tzinfo is timezone.utc
        assert points[0].timestamp == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_sorted_by_timestamp_ascending(self):
        """The API returns newest-first. The RL observation builder walks
        forward, so we flip it at the loader boundary."""
        from aegis.data.alternative_me_loader import download_fear_greed

        payload = _api_payload(
            [
                (1704067200, 45, "Fear"),    # newer
                (1703980800, 72, "Greed"),   # older
                (1703894400, 30, "Fear"),    # oldest
            ]
        )
        client = _FakeClient(payload)

        points = download_fear_greed(limit=3, client=client)
        ts = [p.timestamp for p in points]

        assert ts == sorted(ts)

    def test_limit_parameter_forwarded_in_url(self):
        from aegis.data.alternative_me_loader import download_fear_greed

        client = _FakeClient(_api_payload([(1704067200, 45, "Fear")]))
        download_fear_greed(limit=365, client=client)

        assert any("limit=365" in c for c in client.calls)

    def test_network_failure_returns_empty(self):
        """Missing sentiment should degrade to 'no feature' — not break the
        macro/geo pipeline."""
        from aegis.data.alternative_me_loader import download_fear_greed

        client = _FakeClient(RuntimeError("boom"))
        points = download_fear_greed(limit=30, client=client)

        assert points == []

    def test_malformed_entry_is_skipped(self):
        from aegis.data.alternative_me_loader import download_fear_greed

        payload = {
            "name": "Fear and Greed Index",
            "data": [
                {"value": "abc", "timestamp": "1704067200", "value_classification": "x"},
                {"value": "45", "timestamp": "notnumeric", "value_classification": "x"},
                {"value": "50", "timestamp": "1704067200", "value_classification": "n"},
            ],
        }
        client = _FakeClient(payload)
        points = download_fear_greed(limit=3, client=client)

        assert len(points) == 1
        assert points[0].sentiment_score == pytest.approx(0.0)

    def test_symbol_and_source_are_set(self):
        from aegis.data.alternative_me_loader import download_fear_greed

        client = _FakeClient(_api_payload([(1704067200, 45, "Fear")]))
        points = download_fear_greed(limit=1, client=client)

        assert points[0].source == "fear_greed"
        # Symbol is crypto-market-wide; choose a convention and lock it in.
        assert points[0].symbol in ("BTC", "CRYPTO")
