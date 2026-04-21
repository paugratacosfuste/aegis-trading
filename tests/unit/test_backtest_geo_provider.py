"""Tests for BacktestGeopoliticalProvider.

Analogous to BacktestMacroProvider: advances with the bar clock, exposes
the events visible at the current time, and computes a rolling risk score
weighted by event severity / recency. No look-ahead.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aegis.common.types import GeopoliticalEvent


def _ev(day: int, severity: float = 0.8, category: str = "conflict") -> GeopoliticalEvent:
    return GeopoliticalEvent(
        event_id=f"ev-{day}",
        timestamp=datetime(2025, 1, day, 12, 0, tzinfo=timezone.utc),
        source="gdelt",
        category=category,
        severity=severity,
        affected_sectors=(),
        affected_regions=("USA",),
        raw_text="",
        sentiment_score=-0.3,
        half_life_hours=48,
    )


class TestBacktestGeopoliticalProvider:
    def test_no_events_before_advancing(self):
        from aegis.agents.geopolitical.providers import BacktestGeopoliticalProvider

        provider = BacktestGeopoliticalProvider([_ev(5), _ev(10)])
        # Clock is at its default start → nothing is visible yet
        provider.advance_to(datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert provider.get_recent_events() == []

    def test_advance_reveals_past_events_only(self):
        """Provider must never leak events from the future."""
        from aegis.agents.geopolitical.providers import BacktestGeopoliticalProvider

        provider = BacktestGeopoliticalProvider([_ev(2), _ev(5), _ev(9)])

        provider.advance_to(datetime(2025, 1, 6, tzinfo=timezone.utc))
        ids = {e.event_id for e in provider.get_recent_events(hours=24 * 30)}
        assert ids == {"ev-2", "ev-5"}  # ev-9 is in the future

    def test_recent_events_honours_lookback_hours(self):
        from aegis.agents.geopolitical.providers import BacktestGeopoliticalProvider

        provider = BacktestGeopoliticalProvider([_ev(1), _ev(5), _ev(7)])
        provider.advance_to(datetime(2025, 1, 8, tzinfo=timezone.utc))

        # 48h window @ day 8 → only day 7 qualifies (day 5 is 72h back)
        recent = provider.get_recent_events(hours=48)
        assert {e.event_id for e in recent} == {"ev-7"}

    def test_risk_score_positive_when_events_present(self):
        from aegis.agents.geopolitical.providers import BacktestGeopoliticalProvider

        provider = BacktestGeopoliticalProvider(
            [_ev(3, severity=0.9), _ev(5, severity=0.8)]
        )
        provider.advance_to(datetime(2025, 1, 6, tzinfo=timezone.utc))

        score = provider.get_risk_score()
        assert 0.0 < score <= 1.0

    def test_risk_score_zero_when_no_visible_events(self):
        from aegis.agents.geopolitical.providers import BacktestGeopoliticalProvider

        provider = BacktestGeopoliticalProvider([_ev(10)])
        provider.advance_to(datetime(2025, 1, 1, tzinfo=timezone.utc))

        assert provider.get_risk_score() == 0.0

    def test_risk_score_decays_with_age(self):
        """A fresh event should move the needle more than a week-old one."""
        from aegis.agents.geopolitical.providers import BacktestGeopoliticalProvider

        fresh_provider = BacktestGeopoliticalProvider([_ev(7, severity=0.9)])
        fresh_provider.advance_to(datetime(2025, 1, 8, tzinfo=timezone.utc))
        fresh_score = fresh_provider.get_risk_score()

        stale_provider = BacktestGeopoliticalProvider([_ev(1, severity=0.9)])
        stale_provider.advance_to(datetime(2025, 1, 8, tzinfo=timezone.utc))
        stale_score = stale_provider.get_risk_score()

        assert fresh_score > stale_score

    def test_empty_provider_is_safe(self):
        from aegis.agents.geopolitical.providers import BacktestGeopoliticalProvider

        provider = BacktestGeopoliticalProvider([])
        provider.advance_to(datetime(2025, 1, 1, tzinfo=timezone.utc))

        assert provider.get_recent_events() == []
        assert provider.get_risk_score() == 0.0

    def test_implements_geopolitical_provider_protocol(self):
        """Structural: callers should be able to type-hint the protocol."""
        from aegis.agents.geopolitical.providers import (
            BacktestGeopoliticalProvider,
            GeopoliticalProvider,
        )

        provider: GeopoliticalProvider = BacktestGeopoliticalProvider([_ev(1)])
        # If attribute access works and types line up, the protocol is satisfied.
        assert callable(provider.get_recent_events)
        assert callable(provider.get_risk_score)
