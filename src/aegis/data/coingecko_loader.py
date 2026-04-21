"""CoinGecko historical market-chart loader — free tier, no API key.

Endpoint:
    https://api.coingecko.com/api/v3/coins/{id}/market_chart
      ?vs_currency=usd&days=<N>

The free tier returns one price point per timestamp (no OHLC), plus market
cap and 24h volume series at the same timestamps. We surface each row as a
``MarketDataPoint`` with ``source="coingecko"``. Open/High/Low all equal
close — this is the honest shape of the underlying data, not a fabricated
candle.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol

import httpx

from aegis.common.types import MarketDataPoint

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.coingecko.com/api/v3"
_HTTP_TIMEOUT = 20.0

# Known symbol mapping for the coins we care about. CoinGecko uses slugs
# ("bitcoin", "ethereum") that don't match exchange tickers. If we ever need
# an obscure coin it will fall through to an upper-cased slug which is
# good enough — the RL observation keys off the slug anyway.
_COIN_ID_TO_SYMBOL: dict[str, str] = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "ripple": "XRP",
}


class _ClientLike(Protocol):
    def get(self, url: str, timeout: float | None = None): ...


def _symbol_for(coin_id: str) -> str:
    return _COIN_ID_TO_SYMBOL.get(coin_id.lower(), coin_id.upper())


def download_coin_history(
    coin_id: str,
    days: int = 365,
    *,
    client: _ClientLike | None = None,
) -> list[MarketDataPoint]:
    """Download the last ``days`` of daily price/volume for ``coin_id``.

    Returns ``MarketDataPoint`` rows sorted by timestamp. Any network or
    parse failure yields ``[]`` — the RL observation treats missing crypto
    metrics as "no feature" rather than aborting.
    """
    url = (
        f"{_BASE_URL}/coins/{coin_id}/market_chart"
        f"?vs_currency=usd&days={int(days)}"
    )
    owns_client = client is None
    if client is None:
        client = httpx.Client()

    try:
        try:
            response = client.get(url, timeout=_HTTP_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("CoinGecko fetch failed for %s: %s", coin_id, exc)
            return []
    finally:
        if owns_client:
            try:
                client.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    if not isinstance(payload, dict):
        return []

    prices = payload.get("prices") or []
    volumes = payload.get("total_volumes") or []
    volume_by_ts: dict[int, float] = {}
    for entry in volumes:
        if isinstance(entry, list) and len(entry) >= 2:
            try:
                volume_by_ts[int(entry[0])] = float(entry[1])
            except (ValueError, TypeError):
                continue

    symbol = _symbol_for(coin_id)
    points: list[MarketDataPoint] = []
    for entry in prices:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        try:
            ts_ms = int(entry[0])
            price = float(entry[1])
        except (ValueError, TypeError):
            continue

        ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        volume = volume_by_ts.get(ts_ms, 0.0)

        points.append(
            MarketDataPoint(
                symbol=symbol,
                asset_class="crypto",
                timestamp=ts,
                timeframe="1d",
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                source="coingecko",
            )
        )

    points.sort(key=lambda p: p.timestamp)
    logger.info("CoinGecko: %d rows for %s (days=%d)", len(points), coin_id, days)
    return points
