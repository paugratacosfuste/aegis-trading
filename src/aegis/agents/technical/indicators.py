"""Centralized indicator computation.

Each function returns a signal float in [-1.0, 1.0].
Returns 0.0 on insufficient data or NaN.
"""

import numpy as np
import pandas as pd
from ta.momentum import (
    RSIIndicator,
    StochasticOscillator,
    WilliamsRIndicator,
)
from ta.trend import ADXIndicator, AroonIndicator, CCIIndicator, EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


def _safe_last(series: pd.Series) -> float:
    val = series.iloc[-1]
    return 0.0 if pd.isna(val) else float(val)


def compute_rsi_signal(
    closes: pd.Series,
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
) -> float:
    if len(closes) < period + 1:
        return 0.0
    rsi = _safe_last(RSIIndicator(close=closes, window=period).rsi())
    if rsi == 0.0:
        return 0.0
    if rsi < oversold:
        return 1.0
    if rsi > overbought:
        return -1.0
    mid = (oversold + overbought) / 2.0
    half_range = (overbought - oversold) / 2.0
    return (mid - rsi) / half_range if half_range > 0 else 0.0


def compute_macd_signal(closes: pd.Series) -> float:
    if len(closes) < 26:
        return 0.0
    macd = MACD(close=closes)
    hist = _safe_last(macd.macd_diff())
    signal_line = _safe_last(macd.macd_signal())
    macd_line = _safe_last(macd.macd())
    if hist == 0.0 and signal_line == 0.0:
        return 0.0
    # Crossover: MACD above signal = bullish
    cross = macd_line - signal_line
    price_scale = closes.iloc[-1] if closes.iloc[-1] != 0 else 1.0
    normalized = cross / (price_scale * 0.01)  # Normalize to ~[-1, 1]
    return max(-1.0, min(1.0, normalized))


def compute_ema_trend_signal(closes: pd.Series, short: int = 9, long: int = 21) -> float:
    if len(closes) < long + 1:
        return 0.0
    ema_short = _safe_last(EMAIndicator(close=closes, window=short).ema_indicator())
    ema_long = _safe_last(EMAIndicator(close=closes, window=long).ema_indicator())
    if ema_long == 0:
        return 0.0
    spread = (ema_short - ema_long) / ema_long
    return max(-1.0, min(1.0, spread * 100))


def compute_sma_cross_signal(closes: pd.Series, short: int = 50, long: int = 200) -> float:
    if len(closes) < long + 1:
        return 0.0
    sma_short = _safe_last(SMAIndicator(close=closes, window=short).sma_indicator())
    sma_long = _safe_last(SMAIndicator(close=closes, window=long).sma_indicator())
    if sma_long == 0:
        return 0.0
    spread = (sma_short - sma_long) / sma_long
    return max(-1.0, min(1.0, spread * 50))


def compute_adx_signal(
    highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14
) -> float:
    if len(closes) < period * 2:
        return 0.0
    adx = ADXIndicator(high=highs, low=lows, close=closes, window=period)
    adx_val = _safe_last(adx.adx())
    plus_di = _safe_last(adx.adx_pos())
    minus_di = _safe_last(adx.adx_neg())
    if adx_val < 20:
        return 0.0  # No trend
    # Direction from DI crossover, strength from ADX
    di_diff = plus_di - minus_di
    strength = min(adx_val / 50.0, 1.0)
    direction = 1.0 if di_diff > 0 else -1.0
    return direction * strength


def compute_bollinger_signal(closes: pd.Series, period: int = 20, std_dev: float = 2.0) -> float:
    if len(closes) < period + 1:
        return 0.0
    bb = BollingerBands(close=closes, window=period, window_dev=std_dev)
    pband = _safe_last(bb.bollinger_pband())  # %B: 0 = lower band, 1 = upper band
    # Below lower band (pband < 0) = oversold = buy
    # Above upper band (pband > 1) = overbought = sell
    signal = -(pband - 0.5) * 2.0  # 0.5 center -> 0; 0 -> +1; 1 -> -1
    return max(-1.0, min(1.0, signal))


def compute_stochastic_signal(
    highs: pd.Series, lows: pd.Series, closes: pd.Series
) -> float:
    if len(closes) < 14:
        return 0.0
    stoch = StochasticOscillator(high=highs, low=lows, close=closes)
    k = _safe_last(stoch.stoch())
    if k < 20:
        return 1.0  # Oversold
    if k > 80:
        return -1.0  # Overbought
    return (50 - k) / 30.0


def compute_cci_signal(
    highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 20
) -> float:
    if len(closes) < period + 1:
        return 0.0
    cci = _safe_last(CCIIndicator(high=highs, low=lows, close=closes, window=period).cci())
    # CCI: > +100 overbought, < -100 oversold
    return max(-1.0, min(1.0, -cci / 200.0))


def compute_williams_r_signal(
    highs: pd.Series, lows: pd.Series, closes: pd.Series
) -> float:
    if len(closes) < 14:
        return 0.0
    wr = _safe_last(WilliamsRIndicator(high=highs, low=lows, close=closes).williams_r())
    # Williams %R: -80 to -100 = oversold (buy), 0 to -20 = overbought (sell)
    # Normalize: -100 -> +1, 0 -> -1
    return max(-1.0, min(1.0, -(wr + 50) / 50.0))


def compute_aroon_signal(
    highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 25
) -> float:
    if len(closes) < period + 1:
        return 0.0
    aroon = AroonIndicator(high=highs, low=lows, window=period)
    up = _safe_last(aroon.aroon_up())
    down = _safe_last(aroon.aroon_down())
    diff = (up - down) / 100.0  # [-1, +1]
    return max(-1.0, min(1.0, diff))


def compute_keltner_signal(
    highs: pd.Series, lows: pd.Series, closes: pd.Series
) -> float:
    if len(closes) < 20:
        return 0.0
    kc = KeltnerChannel(high=highs, low=lows, close=closes)
    high_band = _safe_last(kc.keltner_channel_hband())
    low_band = _safe_last(kc.keltner_channel_lband())
    mid = _safe_last(kc.keltner_channel_mband())
    if high_band == low_band:
        return 0.0
    price = closes.iloc[-1]
    # Position within channel: -1 at lower, +1 at upper
    position = (price - mid) / (high_band - mid) if high_band != mid else 0.0
    return max(-1.0, min(1.0, -position))  # Contrarian


def compute_obv_signal(closes: pd.Series, volumes: pd.Series) -> float:
    if len(closes) < 20:
        return 0.0
    obv = OnBalanceVolumeIndicator(close=closes, volume=volumes).on_balance_volume()
    obv_vals = obv.dropna()
    if len(obv_vals) < 10:
        return 0.0
    # OBV trend: compare recent OBV to its SMA
    obv_sma = obv_vals.rolling(10).mean().iloc[-1]
    if pd.isna(obv_sma) or obv_sma == 0:
        return 0.0
    diff = (obv_vals.iloc[-1] - obv_sma) / abs(obv_sma)
    return max(-1.0, min(1.0, diff * 5))


def compute_volume_sma_ratio(volumes: pd.Series, period: int = 20) -> float:
    if len(volumes) < period:
        return 0.0
    avg_vol = volumes.rolling(period).mean().iloc[-1]
    if pd.isna(avg_vol) or avg_vol == 0:
        return 0.0
    ratio = volumes.iloc[-1] / avg_vol
    # High volume confirms moves: > 1.5x = strong, < 0.5x = weak
    return max(-1.0, min(1.0, (ratio - 1.0) * 2))


def compute_adx_confidence(
    highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14
) -> float:
    """ADX-based confidence: higher ADX = more confident in trend signals."""
    if len(closes) < period * 2:
        return 0.5
    adx_val = _safe_last(
        ADXIndicator(high=highs, low=lows, close=closes, window=period).adx()
    )
    return min(adx_val / 50.0, 1.0)
