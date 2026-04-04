"""Tests for lab types: StrategyCohort, CohortConfig, TournamentResult."""

from datetime import date, datetime, timezone

import pytest

from aegis.lab.types import (
    CohortConfig,
    CohortPerformance,
    CohortStatus,
    StrategyCohort,
    TournamentResult,
)


class TestCohortStatus:
    def test_all_statuses_defined(self):
        expected = {"created", "burn_in", "evaluating", "promoted", "relegated", "retired"}
        assert CohortStatus.ALL == expected

    def test_status_values(self):
        assert CohortStatus.CREATED == "created"
        assert CohortStatus.BURN_IN == "burn_in"
        assert CohortStatus.EVALUATING == "evaluating"
        assert CohortStatus.PROMOTED == "promoted"
        assert CohortStatus.RELEGATED == "relegated"
        assert CohortStatus.RETIRED == "retired"


class TestCohortConfig:
    def test_frozen(self):
        config = CohortConfig(
            agent_weights={"technical": 0.5, "statistical": 0.5},
            confidence_threshold=0.45,
            risk_params={"max_risk_per_trade": 0.05},
            universe=("BTC/USDT", "ETH/USDT"),
        )
        with pytest.raises(AttributeError):
            config.confidence_threshold = 0.50  # type: ignore

    def test_to_dict_roundtrip(self):
        config = CohortConfig(
            agent_weights={"technical": 0.5, "statistical": 0.5},
            confidence_threshold=0.45,
            risk_params={"max_risk_per_trade": 0.05},
            universe=("BTC/USDT",),
            invert_sentiment=True,
        )
        d = config.to_dict()
        restored = CohortConfig.from_dict(d)
        assert restored.agent_weights == config.agent_weights
        assert restored.confidence_threshold == config.confidence_threshold
        assert restored.invert_sentiment is True
        assert restored.universe == ("BTC/USDT",)

    def test_defaults(self):
        config = CohortConfig(
            agent_weights={}, confidence_threshold=0.5,
            risk_params={}, universe=(),
        )
        assert config.invert_sentiment is False
        assert config.macro_position_sizing is False


class TestStrategyCohort:
    def test_frozen(self):
        cohort = StrategyCohort(
            cohort_id="test_A",
            name="Test Baseline",
            status=CohortStatus.CREATED,
            config=CohortConfig(
                agent_weights={"technical": 1.0},
                confidence_threshold=0.45,
                risk_params={},
                universe=(),
            ),
        )
        with pytest.raises(AttributeError):
            cohort.status = CohortStatus.BURN_IN  # type: ignore

    def test_defaults(self):
        cohort = StrategyCohort(
            cohort_id="test_A", name="Test",
            status=CohortStatus.CREATED,
            config=CohortConfig(
                agent_weights={}, confidence_threshold=0.5,
                risk_params={}, universe=(),
            ),
        )
        assert cohort.generation == 0
        assert cohort.parent_cohort_id is None
        assert cohort.relegation_count == 0
        assert cohort.virtual_capital == 100_000.0
        assert cohort.burn_in_start is None
        assert cohort.evaluation_start is None


class TestTournamentResult:
    def test_frozen(self):
        result = TournamentResult(
            cohort_id="test_A", week_start=date(2025, 6, 1),
            sharpe=1.5, win_rate=0.55, max_drawdown=-0.10,
            profit_factor=2.0, total_trades=50, net_pnl=5000.0,
            composite_score=0.75, rank=1,
        )
        with pytest.raises(AttributeError):
            result.rank = 2  # type: ignore

    def test_all_fields(self):
        result = TournamentResult(
            cohort_id="test_B", week_start=date(2025, 6, 8),
            sharpe=0.8, win_rate=0.45, max_drawdown=-0.15,
            profit_factor=1.3, total_trades=30, net_pnl=2000.0,
            composite_score=0.55, rank=3,
        )
        assert result.cohort_id == "test_B"
        assert result.composite_score == 0.55


class TestCohortPerformance:
    def test_frozen(self):
        perf = CohortPerformance(
            cohort_id="test_A", sharpe=1.0, win_rate=0.5,
            max_drawdown=-0.10, profit_factor=1.5,
            total_trades=20, net_pnl=1000.0,
        )
        with pytest.raises(AttributeError):
            perf.sharpe = 2.0  # type: ignore

    def test_default_equity_curve(self):
        perf = CohortPerformance(
            cohort_id="test_A", sharpe=1.0, win_rate=0.5,
            max_drawdown=-0.10, profit_factor=1.5,
            total_trades=20, net_pnl=1000.0,
        )
        assert perf.equity_curve == ()
