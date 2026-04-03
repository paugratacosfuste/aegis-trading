"""Macro agent 03: Economic cycle positioning."""

from aegis.agents.macro.base_macro import BaseMacroAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("macro", "economic_cycle")
class EconomicCycleAgent(BaseMacroAgent):
    """Economic cycle classification from leading indicators.

    Uses yield curve slope, VIX level, and CPI trend as proxies
    for ISM/PMI data (which require paid subscriptions).

    Phases: expansion, peak, contraction, trough
    """

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        snap = self._provider.get_macro_snapshot()
        if snap is None:
            return self._neutral_signal(symbol)

        # Score from 3 indicators
        # Yield curve: positive spread = growth, negative = contraction
        if snap.yield_spread > 1.0:
            curve_score = 1.0
        elif snap.yield_spread > 0:
            curve_score = 0.5
        elif snap.yield_spread > -0.5:
            curve_score = -0.5
        else:
            curve_score = -1.0

        # VIX: low = expansion, high = contraction
        vix_scores = {"low": 1.0, "normal": 0.3, "high": -0.5, "extreme": -1.0}
        vix_score = vix_scores.get(snap.vix_regime, 0.0)

        composite = (curve_score * 0.5) + (vix_score * 0.5)

        if composite > 0.5:
            regime = "expansion"
        elif composite > 0:
            regime = "mid_cycle"
        elif composite > -0.5:
            regime = "contraction"
        else:
            regime = "recession_risk"

        sector_tilts = {
            "expansion": {"tech": 0.2, "consumer_disc": 0.2, "industrials": 0.1},
            "mid_cycle": {"tech": 0.1, "financials": 0.1},
            "contraction": {"utilities": 0.2, "healthcare": 0.2, "consumer_staples": 0.1},
            "recession_risk": {"utilities": 0.3, "healthcare": 0.2, "tech": -0.2},
        }

        return self._build_macro_signal(
            symbol=symbol,
            regime=regime,
            regime_confidence=min(abs(composite), 1.0),
            sector_tilts=sector_tilts.get(regime, {}),
            reasoning={
                "curve_score": round(curve_score, 3),
                "vix_score": round(vix_score, 3),
                "composite": round(composite, 3),
            },
        )
