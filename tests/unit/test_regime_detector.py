"""Tests for the price-based HMM regime detector."""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from aegis.backtest.regime_detector import PriceRegimeDetector
from aegis.common.types import MarketDataPoint

_BASE = datetime(2025, 1, 1, tzinfo=timezone.utc)

# Valid regime labels that map to ensemble weight keys
_VALID_REGIMES = {"bull", "bear", "transition", "recovery", "normal"}


def _make_candles(
    n: int,
    base_price: float = 40000.0,
    daily_return: float = 0.001,
    volatility: float = 0.01,
    symbol: str = "BTC/USDT",
) -> list[MarketDataPoint]:
    """Generate synthetic candles with controlled drift and volatility."""
    rng = np.random.default_rng(42)
    candles = []
    price = base_price
    for i in range(n):
        ret = daily_return + volatility * rng.standard_normal()
        new_price = price * (1 + ret)
        high = max(price, new_price) * (1 + abs(volatility) * 0.5)
        low = min(price, new_price) * (1 - abs(volatility) * 0.5)
        candles.append(
            MarketDataPoint(
                symbol=symbol,
                asset_class="crypto",
                timestamp=_BASE + timedelta(hours=i),
                timeframe="1h",
                open=price,
                high=high,
                low=low,
                close=new_price,
                volume=1000.0 + rng.uniform(-200, 200),
                source="binance",
            )
        )
        price = new_price
    return candles


class TestPriceRegimeDetector:
    def test_init_defaults(self):
        det = PriceRegimeDetector()
        assert det.n_states == 4
        assert det.warmup_bars == 60 * 24  # 60 days * 24 hours
        assert not det.is_trained

    def test_train_returns_true_with_sufficient_data(self):
        candles = _make_candles(2000)
        det = PriceRegimeDetector()
        result = det.train(candles)
        assert result is True
        assert det.is_trained

    def test_train_returns_false_with_insufficient_data(self):
        candles = _make_candles(10)
        det = PriceRegimeDetector()
        result = det.train(candles)
        assert result is False
        assert not det.is_trained

    def test_predict_returns_valid_regime(self):
        candles = _make_candles(2000)
        det = PriceRegimeDetector()
        det.train(candles)
        regime = det.predict(candles[-100:])
        assert regime in _VALID_REGIMES

    def test_predict_without_training_returns_normal(self):
        det = PriceRegimeDetector()
        candles = _make_candles(100)
        regime = det.predict(candles)
        assert regime == "normal"

    def test_predict_with_too_few_candles_returns_normal(self):
        candles = _make_candles(2000)
        det = PriceRegimeDetector()
        det.train(candles)
        regime = det.predict(candles[:5])
        assert regime == "normal"

    def test_state_labels_map_to_weight_keys(self):
        """All possible regime outputs must exist in ensemble weights."""
        from aegis.ensemble.weights import REGIME_WEIGHT_ADJUSTMENTS

        det = PriceRegimeDetector()
        for label in det.state_labels.values():
            assert label in REGIME_WEIGHT_ADJUSTMENTS or label == "normal", (
                f"Regime label '{label}' not in REGIME_WEIGHT_ADJUSTMENTS"
            )

    def test_all_four_states_assigned(self):
        """Training should assign all 4 distinct regime labels."""
        candles = _make_candles(2000)
        det = PriceRegimeDetector()
        det.train(candles)
        labels = set(det.state_labels.values())
        assert labels == {"bull", "bear", "transition", "recovery"}

    def test_regime_changes_over_time(self):
        """Predicting over a long series should produce at least 2 regimes."""
        candles = _make_candles(3000, volatility=0.015)
        det = PriceRegimeDetector()
        det.train(candles)

        regimes = set()
        # Sample every 500 bars
        for start in range(500, len(candles) - 500, 500):
            r = det.predict(candles[: start + 1])
            regimes.add(r)

        assert len(regimes) >= 2, f"Expected regime variety, got {regimes}"

    def test_predict_confidence(self):
        candles = _make_candles(2000)
        det = PriceRegimeDetector()
        det.train(candles)
        regime, confidence = det.predict_with_confidence(candles[-100:])
        assert regime in _VALID_REGIMES
        assert 0.0 <= confidence <= 1.0

    def test_compute_features_shape(self):
        candles = _make_candles(100)
        det = PriceRegimeDetector()
        features = det._compute_features(candles)
        # Should have 3 features: daily_return, rolling_14d_vol, rolling_14d_return
        assert features.shape[1] == 3
        # Rows = len(candles) - lookback period
        assert features.shape[0] > 0
        assert features.shape[0] < len(candles)

    def test_compute_features_no_nan(self):
        candles = _make_candles(200)
        det = PriceRegimeDetector()
        features = det._compute_features(candles)
        assert not np.any(np.isnan(features))

    def test_different_symbols_independent(self):
        """Detector should work regardless of symbol."""
        btc = _make_candles(2000, symbol="BTC/USDT")
        eth = _make_candles(2000, base_price=3000.0, symbol="ETH/USDT")
        det = PriceRegimeDetector()
        det.train(btc)
        regime_btc = det.predict(btc[-100:])
        regime_eth = det.predict(eth[-100:])
        # Both should be valid regimes
        assert regime_btc in _VALID_REGIMES
        assert regime_eth in _VALID_REGIMES
