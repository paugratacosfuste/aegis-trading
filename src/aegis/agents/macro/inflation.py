"""Macro agent 04: Inflation regime classification."""

from aegis.agents.macro.base_macro import BaseMacroAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint


@register_agent("macro", "inflation_regime")
class InflationRegimeAgent(BaseMacroAgent):
    """Inflation regime from CPI, real rates, and Fed rate.

    Regimes: deflationary, low, moderate, high, very_high
    """

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        snap = self._provider.get_macro_snapshot()
        if snap is None:
            return self._neutral_signal(symbol)

        cpi = snap.cpi_latest
        real_rate = snap.yield_10y - cpi

        regime, confidence = self._classify(cpi, real_rate)

        sector_tilts = self._sector_tilts(regime)
        asset_tilts = self._asset_tilts(regime)

        return self._build_macro_signal(
            symbol=symbol,
            regime=regime,
            regime_confidence=confidence,
            sector_tilts=sector_tilts,
            asset_class_tilts=asset_tilts,
            reasoning={
                "cpi": cpi,
                "real_rate": round(real_rate, 3),
                "regime": regime,
            },
        )

    def _classify(self, cpi: float, real_rate: float) -> tuple[str, float]:
        if cpi < 0:
            return "deflationary", 0.8
        if cpi < 2.0:
            return "low", 0.6
        if cpi < 3.5:
            return "moderate", 0.5
        if cpi < 5.0:
            return "high", 0.7
        return "very_high", 0.8

    def _sector_tilts(self, regime: str) -> dict[str, float]:
        tilts = {
            "deflationary": {"utilities": 0.2, "tech": 0.1, "energy": -0.3},
            "low": {"tech": 0.1, "consumer_disc": 0.1},
            "moderate": {},
            "high": {"energy": 0.2, "materials": 0.2, "tech": -0.1, "utilities": -0.1},
            "very_high": {"energy": 0.3, "materials": 0.2, "real_estate": -0.2, "tech": -0.2},
        }
        return tilts.get(regime, {})

    def _asset_tilts(self, regime: str) -> dict[str, float]:
        tilts = {
            "deflationary": {"equity": -0.1, "crypto": -0.2},
            "low": {"equity": 0.1},
            "moderate": {},
            "high": {"equity": -0.1, "crypto": 0.1},
            "very_high": {"equity": -0.2, "crypto": 0.2},
        }
        return tilts.get(regime, {})
