"""Unit tests for ThesisVoter — Aegis 2.0 Phase 2.2.

Voter rules:
- Input: list[AgentSignal] from direction-voting agents for one (symbol, bar).
- Ignore signals with confidence <= 0 or direction == 0 — they abstain.
- Ignore signals from thesis-feature types (macro, geopolitical) — those
  emit regime features, not direction votes.
- Weighted average of `direction * confidence` across voters.
- Emit ``long`` when score > +threshold, ``short`` when score < -threshold,
  otherwise ``flat``. Conviction = |score|, clipped to [0, 1].
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aegis.common.types import AgentSignal, ThesisSignal
from aegis.ensemble.thesis_voter import ThesisVoter


def _sig(
    agent_id: str,
    agent_type: str,
    direction: float,
    confidence: float,
    *,
    ts: datetime | None = None,
    symbol: str = "BTC/USDT",
) -> AgentSignal:
    return AgentSignal(
        agent_id=agent_id,
        agent_type=agent_type,
        symbol=symbol,
        timestamp=ts or datetime(2024, 1, 1, tzinfo=timezone.utc),
        direction=direction,
        confidence=confidence,
        timeframe="1w",
        expected_holding_period="weeks",
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        reasoning={},
        features_used={},
        metadata={},
    )


def test_empty_input_emits_flat():
    voter = ThesisVoter()
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    out = voter.vote("BTC/USDT", ts, [])
    assert out.direction == "flat"
    assert out.conviction == 0.0
    assert out.contributing_agents == ()


def test_all_long_emits_long():
    voter = ThesisVoter()
    signals = [
        _sig("tech_01", "technical", +1.0, 0.8),
        _sig("mom_01", "momentum", +1.0, 0.6),
        _sig("stat_08", "statistical", +1.0, 0.9),
    ]
    out = voter.vote("BTC/USDT", datetime(2024, 1, 1, tzinfo=timezone.utc), signals)
    assert out.direction == "long"
    assert out.conviction > 0.5
    assert set(out.contributing_agents) == {"tech_01", "mom_01", "stat_08"}


def test_all_short_emits_short():
    voter = ThesisVoter()
    signals = [
        _sig("tech_01", "technical", -1.0, 0.8),
        _sig("mom_01", "momentum", -1.0, 0.6),
    ]
    out = voter.vote("BTC/USDT", datetime(2024, 1, 1, tzinfo=timezone.utc), signals)
    assert out.direction == "short"
    assert out.conviction > 0.5


def test_mixed_opinions_net_flat():
    voter = ThesisVoter()
    signals = [
        _sig("a", "technical", +1.0, 0.5),
        _sig("b", "momentum", -1.0, 0.5),
    ]
    out = voter.vote("BTC/USDT", datetime(2024, 1, 1, tzinfo=timezone.utc), signals)
    assert out.direction == "flat"


def test_weighted_by_confidence():
    # Long voter with high conviction outweighs two weak shorts.
    voter = ThesisVoter()
    signals = [
        _sig("big_long", "technical", +1.0, 0.9),
        _sig("tiny_short_1", "momentum", -1.0, 0.1),
        _sig("tiny_short_2", "statistical", -1.0, 0.1),
    ]
    out = voter.vote("BTC/USDT", datetime(2024, 1, 1, tzinfo=timezone.utc), signals)
    assert out.direction == "long"


def test_ignores_macro_thesis_features():
    # macro agents expose regime via metadata, not direction. Their
    # direction/confidence are 0 by design and must not tilt the vote.
    voter = ThesisVoter()
    signals = [
        _sig("macro_01", "macro", 0.0, 0.0),
        _sig("tech_01", "technical", +1.0, 0.8),
    ]
    out = voter.vote("BTC/USDT", datetime(2024, 1, 1, tzinfo=timezone.utc), signals)
    assert out.direction == "long"
    assert "macro_01" not in out.contributing_agents


def test_ignores_geopolitical_thesis_features():
    voter = ThesisVoter()
    signals = [
        _sig("geo_01", "geopolitical", +1.0, 0.9),
        _sig("tech_01", "technical", +1.0, 0.5),
    ]
    out = voter.vote("BTC/USDT", datetime(2024, 1, 1, tzinfo=timezone.utc), signals)
    # Only tech_01 contributes.
    assert "geo_01" not in out.contributing_agents
    assert "tech_01" in out.contributing_agents


def test_abstentions_ignored():
    # Zero-confidence voters abstain — they don't pull conviction toward flat.
    voter = ThesisVoter()
    signals = [
        _sig("a", "technical", +1.0, 0.8),
        _sig("b", "momentum", 0.0, 0.0),  # abstain
        _sig("c", "statistical", 0.0, 0.0),  # abstain
    ]
    out = voter.vote("BTC/USDT", datetime(2024, 1, 1, tzinfo=timezone.utc), signals)
    assert out.direction == "long"
    # conviction is the weighted score of only the active voter.
    assert out.conviction == pytest.approx(0.8, rel=0.01)
    assert out.contributing_agents == ("a",)


def test_threshold_creates_flat_zone():
    voter = ThesisVoter(threshold=0.4)
    # weak consensus below threshold → flat
    signals = [
        _sig("a", "technical", +1.0, 0.2),
        _sig("b", "momentum", +1.0, 0.3),
    ]
    out = voter.vote("BTC/USDT", datetime(2024, 1, 1, tzinfo=timezone.utc), signals)
    assert out.direction == "flat"


def test_result_is_thesis_signal():
    voter = ThesisVoter()
    signals = [_sig("a", "technical", +1.0, 0.8)]
    out = voter.vote("BTC/USDT", datetime(2024, 1, 1, tzinfo=timezone.utc), signals)
    assert isinstance(out, ThesisSignal)
    assert out.symbol == "BTC/USDT"
    assert out.timestamp == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert 0.0 <= out.conviction <= 1.0
