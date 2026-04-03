"""Tests for position sizer. Written FIRST per TDD."""

import pytest


class TestPositionSizer:
    def test_basic_kelly(self):
        """Standard Kelly with good edge should return positive size."""
        from aegis.risk.position_sizer import calculate_position_size

        size = calculate_position_size(
            portfolio_value=5000.0,
            win_rate=0.55,
            avg_win=0.03,
            avg_loss=0.02,
            signal_confidence=0.8,
            max_risk_pct=0.02,
        )
        assert size > 0
        assert size <= 5000 * 0.10  # Never more than 10%

    def test_two_pct_cap(self):
        """Position size must not risk more than 2% of portfolio."""
        from aegis.risk.position_sizer import calculate_position_size

        size = calculate_position_size(
            portfolio_value=5000.0,
            win_rate=0.70,
            avg_win=0.10,
            avg_loss=0.02,
            signal_confidence=1.0,
            max_risk_pct=0.02,
        )
        # Max risk amount = 5000 * 0.02 = 100
        # Max position = 100 / 0.02 = 5000
        # But capped at 10% = 500
        assert size <= 500.0

    def test_zero_avg_loss_returns_zero(self):
        """No loss history should return 0 (don't trade)."""
        from aegis.risk.position_sizer import calculate_position_size

        size = calculate_position_size(
            portfolio_value=5000.0,
            win_rate=0.55,
            avg_win=0.03,
            avg_loss=0.0,
            signal_confidence=0.8,
        )
        assert size == 0.0

    def test_negative_kelly_returns_zero(self):
        """Negative expected value should return 0."""
        from aegis.risk.position_sizer import calculate_position_size

        size = calculate_position_size(
            portfolio_value=5000.0,
            win_rate=0.30,  # Lose most trades
            avg_win=0.01,   # Small wins
            avg_loss=0.03,  # Big losses
            signal_confidence=0.8,
        )
        assert size == 0.0

    def test_low_confidence_reduces_size(self):
        """Lower confidence should produce smaller position."""
        from aegis.risk.position_sizer import calculate_position_size

        high_conf = calculate_position_size(
            portfolio_value=5000.0,
            win_rate=0.55,
            avg_win=0.03,
            avg_loss=0.02,
            signal_confidence=0.9,
        )
        low_conf = calculate_position_size(
            portfolio_value=5000.0,
            win_rate=0.55,
            avg_win=0.03,
            avg_loss=0.02,
            signal_confidence=0.3,
        )
        assert high_conf > low_conf

    def test_ten_pct_max_position(self):
        """Never more than 10% of portfolio in one position."""
        from aegis.risk.position_sizer import calculate_position_size

        size = calculate_position_size(
            portfolio_value=10000.0,
            win_rate=0.80,
            avg_win=0.20,
            avg_loss=0.01,
            signal_confidence=1.0,
            max_risk_pct=0.10,  # Very high risk tolerance
        )
        assert size <= 10000 * 0.10
