"""Fundamental agent: Market-cap tier scorer.

3 instances: large ($10B+), mid ($2B-$10B), small ($300M-$2B).
"""

from aegis.agents.fundamental.base_fundamental import BaseFundamentalAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint

# Different scoring emphasis per tier
_TIER_WEIGHTS: dict[str, dict[str, float]] = {
    "large": {"quality": 0.5, "growth": 0.2, "value": 0.3},   # Quality + momentum
    "mid": {"quality": 0.3, "growth": 0.4, "value": 0.3},     # Growth at reasonable price
    "small": {"quality": 0.2, "growth": 0.3, "value": 0.5},   # Deep value
}


@register_agent("fundamental", "market_cap")
class MarketCapAgent(BaseFundamentalAgent):
    """Market-cap tier quality scorer. Config: {tier: "large"}.

    Only activates for symbols matching the configured tier.
    """

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        score = self._provider.get_fundamentals(symbol)
        if score is None:
            return self._neutral_signal(symbol)

        tier = self.config.get("tier", "large")
        if score.market_cap_tier != tier:
            return self._neutral_signal(symbol)

        weights = _TIER_WEIGHTS.get(tier, _TIER_WEIGHTS["large"])

        composite = (
            score.quality_score * weights["quality"]
            + score.growth_score * weights["growth"]
            + score.value_score * weights["value"]
        )

        return self._build_fundamental_signal(
            symbol=symbol,
            quality_score=composite,
            reasoning={
                "tier": tier,
                "quality": round(score.quality_score, 3),
                "growth": round(score.growth_score, 3),
                "value": round(score.value_score, 3),
                "composite": round(composite, 3),
            },
        )
