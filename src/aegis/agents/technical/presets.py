"""Technical indicator presets and period styles.

Each preset maps indicator names to weights (must sum to ~1.0).
Period styles define fast/standard/slow parameters for each indicator.
"""

from types import MappingProxyType

INDICATOR_PRESETS: dict[str, MappingProxyType] = {
    "momentum_fast": MappingProxyType({
        "rsi": 0.35, "macd": 0.30, "stochastic": 0.20, "cci": 0.15,
    }),
    "volume_confirmed": MappingProxyType({
        "rsi": 0.20, "macd": 0.20, "obv": 0.30, "volume_sma_ratio": 0.30,
    }),
    "trend_following": MappingProxyType({
        "ema_trend": 0.30, "adx": 0.25, "aroon": 0.25, "sma_cross": 0.20,
    }),
    "mean_reversion": MappingProxyType({
        "bollinger": 0.35, "rsi": 0.30, "keltner": 0.20, "williams_r": 0.15,
    }),
    "full_suite": MappingProxyType({
        "rsi": 0.15, "macd": 0.15, "ema_trend": 0.15, "adx": 0.10,
        "bollinger": 0.10, "obv": 0.10, "stochastic": 0.10,
        "cci": 0.05, "williams_r": 0.05, "aroon": 0.05,
    }),
    "multi_confirmation": MappingProxyType({
        "rsi": 0.15, "macd": 0.15, "ema_trend": 0.20, "adx": 0.15,
        "bollinger": 0.10, "stochastic": 0.10, "aroon": 0.15,
    }),
}

PERIOD_STYLES: dict[str, dict[str, int]] = {
    "fast": {
        "rsi_period": 7, "ema_short": 5, "ema_long": 13,
        "bb_period": 10, "adx_period": 10, "sma_short": 20, "sma_long": 50,
    },
    "standard": {
        "rsi_period": 14, "ema_short": 9, "ema_long": 21,
        "bb_period": 20, "adx_period": 14, "sma_short": 50, "sma_long": 200,
    },
    "slow": {
        "rsi_period": 21, "ema_short": 21, "ema_long": 50,
        "bb_period": 30, "adx_period": 21, "sma_short": 50, "sma_long": 200,
    },
    "mixed": {
        "rsi_period": 14, "ema_short": 9, "ema_long": 50,
        "bb_period": 20, "adx_period": 14, "sma_short": 50, "sma_long": 200,
    },
}
