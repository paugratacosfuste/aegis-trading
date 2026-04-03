"""Macro agent 01: Yield curve + Fed stance regime classifier."""

from aegis.agents.macro.base_macro import BaseMacroAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("macro", "yield_curve_fed")
class YieldCurveFedAgent(BaseMacroAgent):
    """Rule-based yield curve and Fed stance regime classification.

    Regimes:
    - recession_risk: inverted curve (10Y < 2Y)
    - late_cycle: flat curve (spread < 0.5)
    - early_cycle: steep curve (spread > 2.0)
    - mid_cycle: normal curve (0.5 <= spread <= 2.0)
    """

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        snap = self._provider.get_macro_snapshot()
        if snap is None:
            return self._neutral_signal(symbol)

        spread = snap.yield_spread
        regime, confidence = self._classify_regime(spread, snap.fed_rate)

        sector_tilts = self._sector_tilts_for_regime(regime)

        return self._build_macro_signal(
            symbol=symbol,
            regime=regime,
            regime_confidence=confidence,
            sector_tilts=sector_tilts,
            reasoning={
                "yield_spread": round(spread, 3),
                "fed_rate": snap.fed_rate,
                "classification": regime,
            },
        )

    def _classify_regime(self, spread: float, fed_rate: float) -> tuple[str, float]:
        if spread < 0:
            confidence = min(abs(spread) / 1.0, 1.0)
            return "recession_risk", max(0.6, confidence)
        if spread < 0.5:
            return "late_cycle", 0.6
        if spread > 2.0:
            return "early_cycle", 0.7
        return "mid_cycle", 0.5

    def _sector_tilts_for_regime(self, regime: str) -> dict[str, float]:
        tilts = {
            "recession_risk": {"utilities": 0.2, "healthcare": 0.1, "tech": -0.2, "consumer_disc": -0.2},
            "late_cycle": {"healthcare": 0.1, "consumer_staples": 0.1, "tech": -0.1},
            "early_cycle": {"financials": 0.2, "industrials": 0.2, "tech": 0.1, "utilities": -0.1},
            "mid_cycle": {"tech": 0.1, "consumer_disc": 0.1},
        }
        return tilts.get(regime, {})
