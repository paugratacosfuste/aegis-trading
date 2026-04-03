"""Geopolitical agent 02: Trade policy and tariffs monitoring."""

from aegis.agents.geopolitical.base_geo import BaseGeopoliticalAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint

_PROTECTIONIST_KEYWORDS = ("tariff", "import duty", "export ban", "trade war", "sanctions")
_FREE_TRADE_KEYWORDS = ("trade deal", "trade agreement", "tariff reduction", "free trade")


@register_agent("geopolitical", "trade_policy")
class TradePolicyAgent(BaseGeopoliticalAgent):
    """Tracks tariff announcements, trade agreements, and policy shifts.

    Protectionist moves -> bearish. Free trade -> bullish.
    """

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        events = self._provider.get_recent_events(hours=48)
        risk_score = self._provider.get_risk_score()

        if not events and risk_score == 0.0:
            return self._neutral_signal(symbol)

        trade_events = [e for e in events if e.category == "trade"]
        if not trade_events and risk_score < 0.1:
            return self._neutral_signal(symbol)

        # Score trade policy direction from event text
        protectionist_count = 0
        free_trade_count = 0
        max_severity = 0.0

        for event in trade_events:
            text_lower = event.raw_text.lower()
            if any(kw in text_lower for kw in _PROTECTIONIST_KEYWORDS):
                protectionist_count += 1
            if any(kw in text_lower for kw in _FREE_TRADE_KEYWORDS):
                free_trade_count += 1
            max_severity = max(max_severity, event.severity)

        total = protectionist_count + free_trade_count
        if total == 0:
            # Use overall risk score as fallback
            direction = -risk_score * 0.5
        else:
            # Net direction: protectionist = negative, free trade = positive
            direction = (free_trade_count - protectionist_count) / total

        combined_risk = max(risk_score, max_severity)
        confidence = min(combined_risk + 0.1, 1.0)

        return self._build_geo_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            risk_score=combined_risk,
            reasoning={
                "n_trade_events": len(trade_events),
                "protectionist": protectionist_count,
                "free_trade": free_trade_count,
                "max_severity": round(max_severity, 3),
            },
        )
