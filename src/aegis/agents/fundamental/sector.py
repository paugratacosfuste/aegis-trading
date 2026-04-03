"""Fundamental agent: Sector-specific quality scorer.

11 instances, one per GICS sector, configured via params.
"""

from aegis.agents.fundamental.base_fundamental import BaseFundamentalAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint

# Sector-specific metric weights
_SECTOR_WEIGHTS: dict[str, dict[str, float]] = {
    "tech": {"quality": 0.3, "growth": 0.5, "value": 0.2},
    "healthcare": {"quality": 0.4, "growth": 0.4, "value": 0.2},
    "financials": {"quality": 0.5, "growth": 0.2, "value": 0.3},
    "consumer_disc": {"quality": 0.3, "growth": 0.4, "value": 0.3},
    "consumer_staples": {"quality": 0.4, "growth": 0.2, "value": 0.4},
    "energy": {"quality": 0.3, "growth": 0.3, "value": 0.4},
    "industrials": {"quality": 0.4, "growth": 0.3, "value": 0.3},
    "materials": {"quality": 0.3, "growth": 0.3, "value": 0.4},
    "utilities": {"quality": 0.5, "growth": 0.1, "value": 0.4},
    "real_estate": {"quality": 0.4, "growth": 0.2, "value": 0.4},
    "communications": {"quality": 0.3, "growth": 0.4, "value": 0.3},
}

_DEFAULT_WEIGHTS = {"quality": 0.34, "growth": 0.33, "value": 0.33}


@register_agent("fundamental", "sector")
class SectorFundamentalAgent(BaseFundamentalAgent):
    """Sector-specific quality scorer. Config: {sector: "tech"}.

    Computes a composite quality score using sector-appropriate metric weights.
    """

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        score = self._provider.get_fundamentals(symbol)
        if score is None:
            return self._neutral_signal(symbol)

        sector = self.config.get("sector", "")
        weights = _SECTOR_WEIGHTS.get(sector, _DEFAULT_WEIGHTS)

        # Composite quality score from provider data
        composite = (
            score.quality_score * weights["quality"]
            + score.growth_score * weights["growth"]
            + score.value_score * weights["value"]
        )

        return self._build_fundamental_signal(
            symbol=symbol,
            quality_score=composite,
            reasoning={
                "sector": sector,
                "quality": round(score.quality_score, 3),
                "growth": round(score.growth_score, 3),
                "value": round(score.value_score, 3),
                "composite": round(composite, 3),
            },
        )
