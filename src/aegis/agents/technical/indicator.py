"""Parameterized technical indicator agent (tech_01 through tech_10).

Uses presets to select indicator subsets and period styles.
One class covers all standard indicator-based technical agents.
"""

import pandas as pd

from aegis.agents.base import BaseAgent
from aegis.agents.registry import register_agent
from aegis.agents.technical.indicators import (
    compute_adx_confidence,
    compute_adx_signal,
    compute_aroon_signal,
    compute_bollinger_signal,
    compute_cci_signal,
    compute_ema_trend_signal,
    compute_keltner_signal,
    compute_macd_signal,
    compute_obv_signal,
    compute_rsi_signal,
    compute_sma_cross_signal,
    compute_stochastic_signal,
    compute_volume_sma_ratio,
    compute_williams_r_signal,
)
from aegis.agents.technical.presets import INDICATOR_PRESETS, PERIOD_STYLES
from aegis.common.types import AgentSignal, MarketDataPoint

# Minimum candles for any indicator to work
MIN_CANDLES = 30

# Maps indicator names to their compute functions
# Functions that need OHLCV are handled specially in _compute_all
_CLOSE_ONLY = {"rsi", "macd", "ema_trend", "sma_cross", "bollinger"}
_OHLC = {"adx", "stochastic", "cci", "williams_r", "aroon", "keltner"}
_VOLUME = {"obv", "volume_sma_ratio"}


@register_agent("technical", "indicator")
class TechnicalIndicatorAgent(BaseAgent):
    """Generates signals from a configurable set of technical indicators."""

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        preset_name = config.get("preset", "full_suite")
        period_style = config.get("period_style", "standard")
        self._weights = dict(INDICATOR_PRESETS.get(preset_name, INDICATOR_PRESETS["full_suite"]))
        self._periods = dict(PERIOD_STYLES.get(period_style, PERIOD_STYLES["standard"]))
        self._timeframe = config.get("timeframe", "1h")

    @property
    def agent_type(self) -> str:
        return "technical"

    def generate_signal(
        self, symbol: str, candles: list[MarketDataPoint]
    ) -> AgentSignal:
        if len(candles) < MIN_CANDLES:
            return self._neutral_signal(symbol, self._timeframe)

        closes = pd.Series([c.close for c in candles])
        highs = pd.Series([c.high for c in candles])
        lows = pd.Series([c.low for c in candles])
        volumes = pd.Series([c.volume for c in candles])

        signals = self._compute_all(closes, highs, lows, volumes)

        if not signals:
            return self._neutral_signal(symbol, self._timeframe)

        # Weighted sum of indicator signals
        direction = sum(
            signals[name] * self._weights[name]
            for name in signals
            if name in self._weights
        )

        # Confidence from ADX + indicator agreement
        adx_conf = compute_adx_confidence(
            highs, lows, closes, self._periods.get("adx_period", 14)
        )
        indicator_vals = list(signals.values())
        if indicator_vals:
            agreement = 1.0 if len(indicator_vals) == 1 else (
                sum(1 for v in indicator_vals if (v > 0) == (direction > 0)) / len(indicator_vals)
            )
        else:
            agreement = 0.5
        confidence = adx_conf * 0.5 + agreement * 0.5

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=self._timeframe,
            reasoning={name: round(val, 3) for name, val in signals.items()},
            features=signals,
        )

    def _compute_all(
        self,
        closes: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        volumes: pd.Series,
    ) -> dict[str, float]:
        """Compute all indicators in the preset, return {name: signal}."""
        p = self._periods
        results: dict[str, float] = {}

        for name in self._weights:
            if name == "rsi":
                results[name] = compute_rsi_signal(closes, p.get("rsi_period", 14))
            elif name == "macd":
                results[name] = compute_macd_signal(closes)
            elif name == "ema_trend":
                results[name] = compute_ema_trend_signal(
                    closes, p.get("ema_short", 9), p.get("ema_long", 21)
                )
            elif name == "sma_cross":
                results[name] = compute_sma_cross_signal(
                    closes, p.get("sma_short", 50), p.get("sma_long", 200)
                )
            elif name == "adx":
                results[name] = compute_adx_signal(
                    highs, lows, closes, p.get("adx_period", 14)
                )
            elif name == "bollinger":
                results[name] = compute_bollinger_signal(
                    closes, p.get("bb_period", 20)
                )
            elif name == "stochastic":
                results[name] = compute_stochastic_signal(highs, lows, closes)
            elif name == "cci":
                results[name] = compute_cci_signal(highs, lows, closes)
            elif name == "williams_r":
                results[name] = compute_williams_r_signal(highs, lows, closes)
            elif name == "aroon":
                results[name] = compute_aroon_signal(highs, lows, closes)
            elif name == "keltner":
                results[name] = compute_keltner_signal(highs, lows, closes)
            elif name == "obv":
                results[name] = compute_obv_signal(closes, volumes)
            elif name == "volume_sma_ratio":
                results[name] = compute_volume_sma_ratio(volumes)

        return results
