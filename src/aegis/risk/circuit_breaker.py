"""Circuit breaker: halts trading on excessive drawdown.

From 04-RISK-MANAGEMENT.md.
"""


class CircuitBreaker:
    def __init__(
        self,
        daily_halt: float = -0.15,
        weekly_halt: float = -0.25,
    ):
        self._daily_halt = daily_halt
        self._weekly_halt = weekly_halt

    def check(
        self,
        portfolio_value: float,
        start_of_day_value: float,
        start_of_week_value: float,
    ) -> str:
        """Check circuit breaker status.

        Returns: 'NORMAL', 'WARNING', or 'HALT'.
        """
        daily_return = (
            (portfolio_value - start_of_day_value) / start_of_day_value
            if start_of_day_value > 0
            else 0.0
        )
        weekly_return = (
            (portfolio_value - start_of_week_value) / start_of_week_value
            if start_of_week_value > 0
            else 0.0
        )

        # HALT: beyond threshold
        if daily_return <= self._daily_halt or weekly_return <= self._weekly_halt:
            return "HALT"

        # WARNING: beyond half of threshold
        daily_warning = self._daily_halt / 2
        weekly_warning = self._weekly_halt / 2
        if daily_return <= daily_warning or weekly_return <= weekly_warning:
            return "WARNING"

        return "NORMAL"
