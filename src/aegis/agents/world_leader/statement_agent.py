"""World leader agent: statement classification and sentiment scoring."""

import logging

from aegis.agents.registry import register_agent
from aegis.agents.world_leader.base_leader import BaseLeaderAgent, HALF_LIVES
from aegis.common.types import AgentSignal, MarketDataPoint

logger = logging.getLogger(__name__)

# Keywords for direction inference
_HAWKISH = ("raise rates", "tighten", "combat inflation", "restrictive", "hawkish")
_DOVISH = ("rate cut", "ease", "accommodate", "dovish", "stimul")
_TRADE_NEGATIVE = ("tariff", "ban", "sanctions", "restrict", "penalty")
_TRADE_POSITIVE = ("deal", "agreement", "reduction", "cooperat", "partner")

_MAX_TEXT_LENGTH = 2000
_VALID_STMT_TYPES = frozenset(HALF_LIVES.keys())


@register_agent("world_leader", "statement")
class StatementAgent(BaseLeaderAgent):
    """Classifies leader statements and scores market direction.

    Combines keyword sentiment with provided sentiment_score.
    """

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        statements = self._provider.get_recent_statements(hours=48)
        if not statements:
            return self._neutral_signal(symbol)

        # Aggregate direction from all recent statements
        total_weight = 0.0
        weighted_direction = 0.0
        last_leader = ""
        last_type = "official_statement"

        for stmt in statements:
            raw_sentiment = stmt.get("sentiment_score", 0.0)
            try:
                sentiment = max(-1.0, min(1.0, float(raw_sentiment)))
            except (TypeError, ValueError):
                logger.warning("Invalid sentiment_score %r, skipping", raw_sentiment)
                continue
            text = str(stmt.get("text", ""))[:_MAX_TEXT_LENGTH].lower()
            raw_type = stmt.get("statement_type", "official_statement")
            stmt_type = raw_type if raw_type in _VALID_STMT_TYPES else "official_statement"

            # Keyword-based direction enhancement
            keyword_dir = self._keyword_direction(text)

            # Combine provided sentiment with keyword analysis
            direction = (sentiment * 0.6) + (keyword_dir * 0.4)

            # Weight by recency — more recent = higher weight
            weight = 1.0
            total_weight += weight
            weighted_direction += direction * weight

            last_leader = stmt.get("leader", "unknown")
            last_type = stmt_type

        if total_weight == 0:
            return self._neutral_signal(symbol)

        avg_direction = weighted_direction / total_weight
        confidence = min(abs(avg_direction) + 0.1, 1.0)

        return self._build_leader_signal(
            symbol=symbol,
            direction=max(-1.0, min(1.0, avg_direction)),
            confidence=confidence,
            leader=last_leader,
            statement_type=last_type,
            reasoning={
                "n_statements": len(statements),
                "avg_direction": round(avg_direction, 3),
                "last_leader": last_leader,
            },
        )

    def _keyword_direction(self, text: str) -> float:
        score = 0.0
        for kw in _HAWKISH:
            if kw in text:
                score -= 0.3
        for kw in _DOVISH:
            if kw in text:
                score += 0.3
        for kw in _TRADE_NEGATIVE:
            if kw in text:
                score -= 0.2
        for kw in _TRADE_POSITIVE:
            if kw in text:
                score += 0.2
        return max(-1.0, min(1.0, score))
