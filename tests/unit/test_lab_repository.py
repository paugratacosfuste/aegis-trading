"""Tests for lab repository with mocked DatabasePool."""

import json
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

from aegis.lab.repository import CohortRepository
from aegis.lab.types import CohortConfig, CohortStatus, StrategyCohort, TournamentResult


def _make_config() -> CohortConfig:
    return CohortConfig(
        agent_weights={"technical": 0.5, "statistical": 0.5},
        confidence_threshold=0.45,
        risk_params={"max_risk_per_trade": 0.05},
        universe=("BTC/USDT",),
    )


def _make_cohort(cohort_id: str = "cohort_A") -> StrategyCohort:
    return StrategyCohort(
        cohort_id=cohort_id,
        name="Baseline",
        status=CohortStatus.CREATED,
        config=_make_config(),
        created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )


class TestCohortRepository:
    def test_insert_cohort(self):
        db = MagicMock()
        repo = CohortRepository(db)
        cohort = _make_cohort()
        repo.insert_cohort(cohort)
        db.execute.assert_called_once()
        args = db.execute.call_args[0]
        assert "INSERT INTO cohorts" in args[0]
        assert args[1][0] == "cohort_A"

    def test_update_status(self):
        db = MagicMock()
        repo = CohortRepository(db)
        ts = datetime(2025, 6, 15, tzinfo=timezone.utc)
        repo.update_status("cohort_A", CohortStatus.BURN_IN, burn_in_start=ts)
        db.execute.assert_called_once()
        sql = db.execute.call_args[0][0]
        assert "status = %s" in sql
        assert "burn_in_start = %s" in sql

    def test_get_cohort_found(self):
        db = MagicMock()
        db.fetch_one.return_value = {
            "cohort_id": "cohort_A",
            "name": "Baseline",
            "status": "created",
            "config": json.dumps(_make_config().to_dict()),
            "generation": 0,
            "parent_cohort_id": None,
            "relegation_count": 0,
            "created_at": datetime(2025, 6, 1, tzinfo=timezone.utc),
            "burn_in_start": None,
            "evaluation_start": None,
            "virtual_capital": 100000.0,
        }
        repo = CohortRepository(db)
        result = repo.get_cohort("cohort_A")
        assert result is not None
        assert result.cohort_id == "cohort_A"
        assert result.config.confidence_threshold == 0.45

    def test_get_cohort_not_found(self):
        db = MagicMock()
        db.fetch_one.return_value = None
        repo = CohortRepository(db)
        assert repo.get_cohort("nonexistent") is None

    def test_get_active_cohorts(self):
        db = MagicMock()
        db.fetch_all.return_value = [
            {
                "cohort_id": "cohort_A",
                "name": "Baseline",
                "status": "created",
                "config": _make_config().to_dict(),
                "generation": 0,
                "parent_cohort_id": None,
                "relegation_count": 0,
                "created_at": datetime(2025, 6, 1, tzinfo=timezone.utc),
                "burn_in_start": None,
                "evaluation_start": None,
                "virtual_capital": 100000.0,
            }
        ]
        repo = CohortRepository(db)
        cohorts = repo.get_active_cohorts()
        assert len(cohorts) == 1
        assert cohorts[0].cohort_id == "cohort_A"

    def test_insert_tournament_result(self):
        db = MagicMock()
        repo = CohortRepository(db)
        result = TournamentResult(
            cohort_id="cohort_A", week_start=date(2025, 6, 1),
            sharpe=1.5, win_rate=0.55, max_drawdown=-0.10,
            profit_factor=2.0, total_trades=50, net_pnl=5000.0,
            composite_score=0.75, rank=1,
        )
        repo.insert_tournament_result(result)
        db.execute.assert_called_once()
        assert "INSERT INTO tournaments" in db.execute.call_args[0][0]

    def test_save_tournament_results_batch(self):
        db = MagicMock()
        repo = CohortRepository(db)
        results = [
            TournamentResult(
                cohort_id=f"cohort_{c}", week_start=date(2025, 6, 1),
                sharpe=1.0, win_rate=0.5, max_drawdown=-0.10,
                profit_factor=1.5, total_trades=20, net_pnl=1000.0,
                composite_score=0.60, rank=i + 1,
            )
            for i, c in enumerate(["A", "B", "C"])
        ]
        repo.save_tournament_results(results)
        db.execute_many.assert_called_once()
        assert len(db.execute_many.call_args[0][1]) == 3

    def test_get_tournament_history(self):
        db = MagicMock()
        db.fetch_all.return_value = [
            {
                "cohort_id": "cohort_A", "week_start": date(2025, 6, 1),
                "sharpe": 1.5, "win_rate": 0.55, "max_drawdown": -0.10,
                "profit_factor": 2.0, "total_trades": 50, "net_pnl": 5000.0,
                "composite_score": 0.75, "rank": 1,
            }
        ]
        repo = CohortRepository(db)
        history = repo.get_tournament_history("cohort_A")
        assert len(history) == 1
        assert history[0].sharpe == 1.5

    def test_increment_relegation_count(self):
        db = MagicMock()
        repo = CohortRepository(db)
        repo.increment_relegation_count("cohort_B")
        db.execute.assert_called_once()
        assert "relegation_count = relegation_count + 1" in db.execute.call_args[0][0]

    def test_config_from_json_string(self):
        """Config stored as JSON string in DB is correctly deserialized."""
        db = MagicMock()
        db.fetch_one.return_value = {
            "cohort_id": "cohort_A",
            "name": "Baseline",
            "status": "evaluating",
            "config": json.dumps({"agent_weights": {"technical": 1.0},
                                   "confidence_threshold": 0.5}),
            "generation": 1,
        }
        repo = CohortRepository(db)
        cohort = repo.get_cohort("cohort_A")
        assert cohort is not None
        assert cohort.config.agent_weights == {"technical": 1.0}
