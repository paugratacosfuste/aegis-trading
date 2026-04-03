"""Crypto agent: Technical indicators with crypto-calibrated thresholds.

Reuses the TechnicalIndicatorAgent logic but with wider thresholds.
5 instances via config presets.
"""

from aegis.agents.crypto.base_crypto import BaseCryptoAgent
from aegis.agents.registry import register_agent
from aegis.agents.technical.indicators import (
    compute_adx_confidence,
    compute_aroon_signal,
    compute_bollinger_signal,
    compute_cci_signal,
    compute_ema_trend_signal,
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

import pandas as pd

# Crypto-specific threshold overrides (wider than equity)
_CRYPTO_OVERRIDES = {
    "rsi_overbought": 75,   # vs 70 for equity
    "rsi_oversold": 25,     # vs 30 for equity
    "bb_width_multiplier": 1.25,  # Wider bands expected
}

_INDICATOR_FUNCS = {
    "rsi": compute_rsi_signal,
    "macd": compute_macd_signal,
    "ema_trend": compute_ema_trend_signal,
    "bollinger": compute_bollinger_signal,
    "stochastic": compute_stochastic_signal,
    "cci": compute_cci_signal,
    "williams_r": compute_williams_r_signal,
    "aroon": compute_aroon_signal,
    "obv": compute_obv_signal,
    "volume_sma_ratio": compute_volume_sma_ratio,
    "sma_cross": compute_sma_cross_signal,
    "adx": compute_adx_confidence,
}


@register_agent("crypto", "crypto_technical")
class CryptoTechnicalAgent(BaseCryptoAgent):
    """Technical indicator agent with crypto-calibrated thresholds.

    Uses same indicator logic as TechnicalIndicatorAgent but:
    - RSI overbought: 75 (not 70)
    - RSI oversold: 25 (not 30)
    - Z-score thresholds: wider
    - 5 instances with different presets
    """

    def __init__(self, agent_id, config, provider=None):
        super().__init__(agent_id, config, provider)
        preset_name = config.get("preset", "trend_following")
        self._weights = dict(INDICATOR_PRESETS.get(preset_name, INDICATOR_PRESETS["trend_following"]))
        period_style = config.get("period_style", "fast")
        self._periods = PERIOD_STYLES.get(period_style, PERIOD_STYLES["fast"])

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        if not self._is_crypto(symbol):
            return self._neutral_signal(symbol)

        if len(candles) < 21:
            return self._neutral_signal(symbol)

        close = pd.Series([c.close for c in candles])
        high = pd.Series([c.high for c in candles])
        low = pd.Series([c.low for c in candles])
        volume = pd.Series([c.volume for c in candles])

        weighted_sum = 0.0
        total_weight = 0.0

        for name, weight in self._weights.items():
            func = _INDICATOR_FUNCS.get(name)
            if func is None:
                continue

            try:
                if name == "rsi":
                    val = func(
                        close, self._periods,
                        oversold=_CRYPTO_OVERRIDES["rsi_oversold"],
                        overbought=_CRYPTO_OVERRIDES["rsi_overbought"],
                    )
                elif name in ("obv", "volume_sma_ratio"):
                    val = func(close, volume)
                elif name in ("stochastic", "williams_r", "aroon", "cci"):
                    val = func(high, low, close, self._periods)
                elif name == "adx":
                    val = func(high, low, close, self._periods)
                elif name in ("bollinger", "keltner"):
                    val = func(close, self._periods)
                elif name == "sma_cross":
                    val = func(close, self._periods)
                else:
                    val = func(close, self._periods)

                weighted_sum += val * weight
                total_weight += weight
            except Exception:
                continue

        if total_weight == 0:
            return self._neutral_signal(symbol)

        direction = weighted_sum / total_weight
        confidence = min(abs(direction) + 0.1, 1.0)

        return self._build_signal(
            symbol=symbol,
            direction=max(-1.0, min(1.0, direction)),
            confidence=confidence,
            timeframe="1h",
            reasoning={"preset": self.config.get("preset", "trend_following")},
            features={"direction": direction},
        )
