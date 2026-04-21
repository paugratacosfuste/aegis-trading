"""alternative.me Crypto Fear & Greed Index loader — public JSON, no key.

Endpoint:
    https://api.alternative.me/fng/?limit=<N>

Returns the last N daily readings (newest first). Values are integers in
[0, 100]; we normalise to the codebase-standard [-1, +1] sentiment scale.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx

from aegis.common.types import SentimentDataPoint

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.alternative.me/fng/"
_HTTP_TIMEOUT = 15.0
_CRYPTO_SYMBOL = "BTC"  # the index is market-wide but anchored to BTC


class _ClientLike(Protocol):
    def get(self, url: str, timeout: float | None = None): ...


def _normalize(value: int) -> float:
    """Map 0..100 → -1..+1. Clamped to handle any API drift."""
    sentiment = (value - 50.0) / 50.0
    return max(-1.0, min(1.0, sentiment))


def _parse_entry(entry: dict[str, Any]) -> SentimentDataPoint | None:
    try:
        value = int(entry["value"])
        ts_seconds = int(entry["timestamp"])
    except (KeyError, ValueError, TypeError):
        return None

    return SentimentDataPoint(
        symbol=_CRYPTO_SYMBOL,
        timestamp=datetime.fromtimestamp(ts_seconds, tz=timezone.utc),
        source="fear_greed",
        sentiment_score=_normalize(value),
        mention_count=0,  # the index does not expose article counts
        sentiment_velocity=0.0,
    )


def download_fear_greed(
    limit: int = 30,
    *,
    client: _ClientLike | None = None,
) -> list[SentimentDataPoint]:
    """Fetch the last ``limit`` daily Fear & Greed readings.

    Returns points sorted by timestamp ascending (older first), so the
    result is ready to replay in a walk-forward backtest. Any HTTP or
    parse failure yields ``[]`` rather than raising — downstream agents
    already treat empty sentiment as "no signal".
    """
    url = f"{_BASE_URL}?limit={int(limit)}&format=json"
    owns_client = client is None
    if client is None:
        client = httpx.Client()

    try:
        try:
            response = client.get(url, timeout=_HTTP_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("alternative.me fetch failed: %s", exc)
            return []
    finally:
        if owns_client:
            try:
                client.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        logger.warning("alternative.me: unexpected payload shape")
        return []

    points: list[SentimentDataPoint] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        parsed = _parse_entry(entry)
        if parsed is not None:
            points.append(parsed)

    points.sort(key=lambda p: p.timestamp)
    logger.info("alternative.me: %d fear/greed readings", len(points))
    return points
