"""Tests for ATR-based stop-loss. Written FIRST per TDD."""

import pytest


class TestStopLoss:
    def test_long_stop_below_entry(self):
        """Long stop should be below entry price."""
        from aegis.risk.stop_loss import calculate_stop_loss

        stop = calculate_stop_loss(
            entry_price=42000.0,
            direction="LONG",
            atr_14=500.0,
            volatility_regime="normal",
            timeframe="1h",
        )
        assert stop < 42000.0

    def test_short_stop_above_entry(self):
        """Short stop should be above entry price."""
        from aegis.risk.stop_loss import calculate_stop_loss

        stop = calculate_stop_loss(
            entry_price=42000.0,
            direction="SHORT",
            atr_14=500.0,
            volatility_regime="normal",
            timeframe="1h",
        )
        assert stop > 42000.0

    def test_high_vol_wider_stop(self):
        """Higher volatility regime should give wider stop distance."""
        from aegis.risk.stop_loss import calculate_stop_loss

        stop_normal = calculate_stop_loss(
            entry_price=42000.0,
            direction="LONG",
            atr_14=500.0,
            volatility_regime="normal",
            timeframe="1h",
        )
        stop_high = calculate_stop_loss(
            entry_price=42000.0,
            direction="LONG",
            atr_14=500.0,
            volatility_regime="high",
            timeframe="1h",
        )
        # High vol = wider stop = lower price for LONG
        assert stop_high < stop_normal

    def test_longer_timeframe_wider_stop(self):
        """Longer timeframe should give wider stop distance."""
        from aegis.risk.stop_loss import calculate_stop_loss

        stop_1h = calculate_stop_loss(
            entry_price=42000.0,
            direction="LONG",
            atr_14=500.0,
            volatility_regime="normal",
            timeframe="1h",
        )
        stop_4h = calculate_stop_loss(
            entry_price=42000.0,
            direction="LONG",
            atr_14=500.0,
            volatility_regime="normal",
            timeframe="4h",
        )
        assert stop_4h < stop_1h

    def test_extreme_vol_widest(self):
        from aegis.risk.stop_loss import calculate_stop_loss

        stop = calculate_stop_loss(
            entry_price=42000.0,
            direction="LONG",
            atr_14=500.0,
            volatility_regime="extreme",
            timeframe="1h",
        )
        distance = 42000.0 - stop
        assert distance == 500.0 * 3.0 * 1.0  # extreme=3.0, 1h=1.0
