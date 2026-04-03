"""Geopolitical agent 01: Conflict and sanctions monitoring."""

from aegis.agents.geopolitical.base_geo import BaseGeopoliticalAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("geopolitical", "conflict_sanctions")
class ConflictSanctionsAgent(BaseGeopoliticalAgent):
    """Monitors GDELT conflict events and sanctions.

    High conflict risk -> bearish direction + potential VETO.
    """

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        events = self._provider.get_recent_events(hours=24)
        risk_score = self._provider.get_risk_score()

        if not events and risk_score == 0.0:
            return self._neutral_signal(symbol)

        # Filter for conflict/sanctions events
        conflict_events = [e for e in events if e.category in ("conflict", "policy")]
        if not conflict_events and risk_score < 0.1:
            return self._neutral_signal(symbol)

        # Average severity of conflict events
        avg_severity = (
            sum(e.severity for e in conflict_events) / len(conflict_events)
            if conflict_events else 0.0
        )

        # Combine event severity with overall risk score
        combined_risk = max(risk_score, avg_severity)

        # Direction: negative (bearish) proportional to risk
        direction = -combined_risk
        confidence = min(combined_risk + 0.1, 1.0)

        return self._build_geo_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            risk_score=combined_risk,
            reasoning={
                "n_conflict_events": len(conflict_events),
                "avg_severity": round(avg_severity, 3),
                "risk_score": round(combined_risk, 3),
            },
        )
