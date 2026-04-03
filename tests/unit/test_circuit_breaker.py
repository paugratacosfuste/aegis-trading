"""Tests for circuit breaker. Written FIRST per TDD."""

import pytest


class TestCircuitBreaker:
    def test_normal_when_no_loss(self):
        from aegis.risk.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(daily_halt=-0.15, weekly_halt=-0.25)
        status = cb.check(
            portfolio_value=5000.0,
            start_of_day_value=5000.0,
            start_of_week_value=5000.0,
        )
        assert status == "NORMAL"

    def test_warning_on_moderate_daily_loss(self):
        from aegis.risk.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(daily_halt=-0.15, weekly_halt=-0.25)
        # 8% daily loss (above half of halt threshold)
        status = cb.check(
            portfolio_value=4600.0,
            start_of_day_value=5000.0,
            start_of_week_value=5000.0,
        )
        assert status == "WARNING"

    def test_halt_on_daily_threshold(self):
        from aegis.risk.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(daily_halt=-0.15, weekly_halt=-0.25)
        # 16% daily loss (beyond halt)
        status = cb.check(
            portfolio_value=4200.0,
            start_of_day_value=5000.0,
            start_of_week_value=5000.0,
        )
        assert status == "HALT"

    def test_halt_on_weekly_threshold(self):
        from aegis.risk.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(daily_halt=-0.15, weekly_halt=-0.25)
        # Small daily loss but big weekly loss
        status = cb.check(
            portfolio_value=3700.0,
            start_of_day_value=3800.0,  # Only 2.6% daily
            start_of_week_value=5000.0,  # 26% weekly
        )
        assert status == "HALT"

    def test_production_thresholds_stricter(self):
        from aegis.risk.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(daily_halt=-0.05, weekly_halt=-0.10)
        # 6% daily loss: fine in lab, halt in production
        status = cb.check(
            portfolio_value=4700.0,
            start_of_day_value=5000.0,
            start_of_week_value=5000.0,
        )
        assert status == "HALT"
