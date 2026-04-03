"""Macro agent 02: Risk-on / risk-off classification."""

from aegis.agents.macro.base_macro import BaseMacroAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("macro", "risk_regime")
class RiskRegimeAgent(BaseMacroAgent):
    """Composite risk regime from VIX, DXY, and credit spreads.

    VIX low + DXY weak = risk_on
    VIX extreme + DXY strong = risk_off
    Mixed signals = neutral
    """

    # Scoring: negative = risk-off, positive = risk-on
    _VIX_SCORES = {"low": 1.0, "normal": 0.0, "high": -0.5, "extreme": -1.0}
    _DXY_STRONG_THRESHOLD = 105.0
    _DXY_WEAK_THRESHOLD = 100.0

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        snap = self._provider.get_macro_snapshot()
        if snap is None:
            return self._neutral_signal(symbol)

        vix_score = self._VIX_SCORES.get(snap.vix_regime, 0.0)

        # DXY: strong dollar = risk-off, weak dollar = risk-on
        if snap.dxy > self._DXY_STRONG_THRESHOLD:
            dxy_score = -0.5
        elif snap.dxy < self._DXY_WEAK_THRESHOLD:
            dxy_score = 0.5
        else:
            dxy_score = 0.0

        composite = (vix_score * 0.6) + (dxy_score * 0.4)

        if composite > 0.3:
            regime = "risk_on"
        elif composite < -0.3:
            regime = "risk_off"
        else:
            regime = "neutral"

        confidence = min(abs(composite), 1.0)

        asset_class_tilts = {
            "risk_on": {"equity": 0.2, "crypto": 0.3},
            "risk_off": {"equity": -0.2, "crypto": -0.3},
            "neutral": {},
        }

        return self._build_macro_signal(
            symbol=symbol,
            regime=regime,
            regime_confidence=confidence,
            asset_class_tilts=asset_class_tilts.get(regime, {}),
            reasoning={
                "vix": snap.vix,
                "vix_regime": snap.vix_regime,
                "dxy": snap.dxy,
                "composite": round(composite, 3),
            },
        )
