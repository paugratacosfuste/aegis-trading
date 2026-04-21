"""Tests for CoinGecko market-chart loader.

Endpoint (free, no API key):
    https://api.coingecko.com/api/v3/coins/{id}/market_chart
      ?vs_currency=usd&days=<N>

Response:
    {
      "prices":        [[ts_ms, price_usd], ...],
      "market_caps":   [[ts_ms, mcap_usd], ...],
      "total_volumes": [[ts_ms, vol_24h_usd], ...]
    }

We surface each row as a MarketDataPoint (source="coingecko") — CoinGecko
free tier only exposes a single price-per-timestamp, so OHLC all equal price.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from aegis.common.types import MarketDataPoint


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


def _payload(rows: list[tuple[int, float, float, float]]) -> dict:
    """rows = list of (ts_ms, price, market_cap, volume)."""
    return {
        "prices": [[ts, p] for ts, p, _, _ in rows],
        "market_caps": [[ts, mc] for ts, _, mc, _ in rows],
        "total_volumes": [[ts, v] for ts, _, _, v in rows],
    }


class TestCoinGeckoLoader:
    def test_builds_market_data_points(self):
        from aegis.data.coingecko_loader import download_coin_history

        # 1704067200000 ms = 2024-01-01 UTC
        client = _FakeClient(
            _payload(
                [
                    (1704067200000, 42000.0, 800_000_000_000.0, 20_000_000_000.0),
                    (1704153600000, 43000.0, 820_000_000_000.0, 21_000_000_000.0),
                ]
            )
        )

        points = download_coin_history("bitcoin", days=7, client=client)

        assert len(points) == 2
        assert all(isinstance(p, MarketDataPoint) for p in points)
        assert points[0].symbol == "BTC"
        assert points[0].source == "coingecko"
        assert points[0].asset_class == "crypto"

    def test_ohlc_all_equal_when_only_close_available(self):
        """Free tier market_chart gives one price per timestamp — O=H=L=C is
        the honest representation, not a fabricated candle."""
        from aegis.data.coingecko_loader import download_coin_history

        client = _FakeClient(_payload([(1704067200000, 42000.0, 800e9, 20e9)]))
        points = download_coin_history("bitcoin", days=1, client=client)

        p = points[0]
        assert p.open == p.high == p.low == p.close == 42000.0

    def test_timestamps_are_utc(self):
        from aegis.data.coingecko_loader import download_coin_history

        client = _FakeClient(_payload([(1704067200000, 42000.0, 0.0, 0.0)]))
        points = download_coin_history("bitcoin", days=1, client=client)

        assert points[0].timestamp == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_sorted_ascending(self):
        from aegis.data.coingecko_loader import download_coin_history

        client = _FakeClient(
            _payload(
                [
                    (1704153600000, 43000.0, 0.0, 0.0),  # later
                    (1704067200000, 42000.0, 0.0, 0.0),  # earlier
                ]
            )
        )
        points = download_coin_history("bitcoin", days=2, client=client)

        assert points[0].timestamp < points[1].timestamp

    def test_volume_comes_from_total_volumes(self):
        from aegis.data.coingecko_loader import download_coin_history

        client = _FakeClient(_payload([(1704067200000, 42000.0, 0.0, 17_500_000_000.0)]))
        points = download_coin_history("bitcoin", days=1, client=client)

        assert points[0].volume == pytest.approx(17_500_000_000.0)

    def test_coin_id_and_days_forwarded(self):
        from aegis.data.coingecko_loader import download_coin_history

        client = _FakeClient(_payload([(1704067200000, 1.0, 0.0, 0.0)]))
        download_coin_history("ethereum", days=90, client=client)

        url = client.calls[0]
        assert "/coins/ethereum/market_chart" in url
        assert "days=90" in url
        assert "vs_currency=usd" in url

    def test_network_failure_returns_empty(self):
        from aegis.data.coingecko_loader import download_coin_history

        client = _FakeClient(RuntimeError("boom"))
        assert download_coin_history("bitcoin", days=1, client=client) == []

    def test_empty_response_returns_empty(self):
        from aegis.data.coingecko_loader import download_coin_history

        client = _FakeClient({"prices": [], "market_caps": [], "total_volumes": []})
        assert download_coin_history("bitcoin", days=1, client=client) == []

    def test_symbol_mapping(self):
        from aegis.data.coingecko_loader import download_coin_history

        client = _FakeClient(_payload([(1704067200000, 2500.0, 0.0, 0.0)]))
        eth_points = download_coin_history("ethereum", days=1, client=client)

        assert eth_points[0].symbol == "ETH"
