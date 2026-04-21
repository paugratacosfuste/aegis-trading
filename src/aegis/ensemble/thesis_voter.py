"""Aegis 2.0 thesis voter.

Aggregates direction-voting agents into a single ThesisSignal per (symbol,
bar). This is the floor RL must beat — simple weighted vote, no sizing,
no stop, no timing logic. The executor (SimpleExecutor in Phase 2,
RLExecutor in Phase 3) decides sizing and risk.

Ignored by design:
  - Signals with confidence == 0 or direction == 0 (abstentions).
  - Signals from ``macro`` / ``geopolitical`` agent types — in Aegis 2.0
    those expose regime/risk features via metadata and do not vote
    directionally (see CLAUDE.md, plan §3).
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from aegis.common.types import AgentSignal, ThesisSignal

# Types that contribute features, not directional votes.
_THESIS_FEATURE_TYPES = frozenset({"macro", "geopolitical"})


class ThesisVoter:
    """Simple weighted voter over direction-emitting agents.

    score = sum(direction_i * confidence_i) / count_of_voters
      → long if score > +threshold
      → short if score < -threshold
      → flat otherwise
    conviction = min(1.0, |score|)
    """

    def __init__(self, threshold: float = 0.2):
        if not 0.0 <= threshold < 1.0:
            raise ValueError("threshold must be in [0, 1)")
        self._threshold = threshold

    def vote(
        self,
        symbol: str,
        timestamp: datetime,
        signals: Iterable[AgentSignal],
    ) -> ThesisSignal:
        contributing: list[AgentSignal] = []
        for s in signals:
            if s.agent_type in _THESIS_FEATURE_TYPES:
                continue
            if abs(s.direction) < 1e-9:
                continue
            if s.confidence <= 0.0:
                continue
            contributing.append(s)

        if not contributing:
            return ThesisSignal(
                symbol=symbol,
                timestamp=timestamp,
                direction="flat",
                conviction=0.0,
                contributing_agents=(),
                metadata={"reason": "no active voters"},
            )

        weighted_sum = sum(s.direction * s.confidence for s in contributing)
        score = weighted_sum / len(contributing)

        if score > self._threshold:
            direction = "long"
        elif score < -self._threshold:
            direction = "short"
        else:
            direction = "flat"

        conviction = min(1.0, abs(score))

        return ThesisSignal(
            symbol=symbol,
            timestamp=timestamp,
            direction=direction,
            conviction=conviction,
            contributing_agents=tuple(s.agent_id for s in contributing),
            metadata={
                "score": score,
                "voter_count": len(contributing),
                "threshold": self._threshold,
            },
        )
