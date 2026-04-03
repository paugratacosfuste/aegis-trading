"""Fundamental agent: Earnings surprise / post-earnings drift detection."""

from aegis.agents.fundamental.base_fundamental import BaseFundamentalAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("fundamental", "earnings_surprise")
class EarningsSurpriseAgent(BaseFundamentalAgent):
    """Post-earnings drift detection.

    High growth + high quality = positive surprise proxy.
    Uses revenue_growth and quality_score as stand-in for actual EPS surprise
    (actual EPS surprise data requires paid data feeds).
    """

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        score = self._provider.get_fundamentals(symbol)
        if score is None:
            return self._neutral_signal(symbol)

        # Surprise proxy: exceptional growth + quality = likely beat
        surprise_score = (score.growth_score * 0.6) + (score.quality_score * 0.4)

        # Revenue growth as a secondary signal
        if score.revenue_growth > 0.15:
            surprise_score = min(surprise_score + 0.1, 1.0)
        elif score.revenue_growth < -0.05:
            surprise_score = max(surprise_score - 0.15, 0.0)

        return self._build_fundamental_signal(
            symbol=symbol,
            quality_score=surprise_score,
            reasoning={
                "growth_score": round(score.growth_score, 3),
                "quality_score": round(score.quality_score, 3),
                "revenue_growth": round(score.revenue_growth, 3),
                "surprise_score": round(surprise_score, 3),
            },
        )
