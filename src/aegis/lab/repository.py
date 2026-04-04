"""Lab repository: raw SQL CRUD for cohorts and tournaments tables."""

import json
from datetime import date, datetime

from aegis.common.db import DatabasePool
from aegis.lab.types import CohortConfig, CohortStatus, StrategyCohort, TournamentResult


_ALLOWED_STATUS_COLS = frozenset({
    "promoted_at", "relegated_at", "retired_at",
    "burn_in_start", "evaluation_start", "virtual_capital",
    "relegation_count",
})

_COHORT_COLUMNS = (
    "cohort_id, name, config, status, generation, parent_cohort_id, "
    "relegation_count, burn_in_start, evaluation_start, virtual_capital, created_at"
)

_TOURNAMENT_COLUMNS = (
    "cohort_id, week_start, sharpe, win_rate, max_drawdown, "
    "profit_factor, total_trades, net_pnl, composite_score, rank"
)


class CohortRepository:
    """Data access for cohorts and tournaments tables."""

    def __init__(self, db: DatabasePool):
        self._db = db

    def insert_cohort(self, cohort: StrategyCohort) -> None:
        sql = """
            INSERT INTO cohorts (
                cohort_id, name, config, status, generation,
                parent_cohort_id, relegation_count, burn_in_start,
                evaluation_start, virtual_capital, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """
        self._db.execute(sql, (
            cohort.cohort_id,
            cohort.name,
            json.dumps(cohort.config.to_dict()),
            cohort.status,
            cohort.generation,
            cohort.parent_cohort_id,
            cohort.relegation_count,
            cohort.burn_in_start,
            cohort.evaluation_start,
            cohort.virtual_capital,
            cohort.created_at,
        ))

    def update_status(
        self,
        cohort_id: str,
        status: str,
        **kwargs: datetime | float | int | None,
    ) -> None:
        sets = ["status = %s", "updated_at = NOW()"]
        params: list = [status]
        for col, val in kwargs.items():
            if col not in _ALLOWED_STATUS_COLS:
                raise ValueError(f"Column '{col}' is not allowed in update_status")
            sets.append(f"{col} = %s")
            params.append(val)
        params.append(cohort_id)
        sql = f"UPDATE cohorts SET {', '.join(sets)} WHERE cohort_id = %s"
        self._db.execute(sql, tuple(params))

    def get_cohort(self, cohort_id: str) -> StrategyCohort | None:
        row = self._db.fetch_one(
            f"SELECT {_COHORT_COLUMNS} FROM cohorts WHERE cohort_id = %s", (cohort_id,)
        )
        if row is None:
            return None
        return self._row_to_cohort(row)

    def get_active_cohorts(self) -> list[StrategyCohort]:
        active = (CohortStatus.CREATED, CohortStatus.BURN_IN, CohortStatus.EVALUATING)
        rows = self._db.fetch_all(
            f"SELECT {_COHORT_COLUMNS} FROM cohorts WHERE status IN (%s, %s, %s) ORDER BY created_at",
            active,
        )
        return [self._row_to_cohort(r) for r in rows]

    def get_by_status(self, status: str) -> list[StrategyCohort]:
        rows = self._db.fetch_all(
            f"SELECT {_COHORT_COLUMNS} FROM cohorts WHERE status = %s ORDER BY created_at", (status,)
        )
        return [self._row_to_cohort(r) for r in rows]

    def increment_relegation_count(self, cohort_id: str) -> None:
        self._db.execute(
            "UPDATE cohorts SET relegation_count = relegation_count + 1, updated_at = NOW() WHERE cohort_id = %s",
            (cohort_id,),
        )

    def insert_tournament_result(self, result: TournamentResult) -> None:
        sql = """
            INSERT INTO tournaments (
                week_start, cohort_id, sharpe, win_rate, max_drawdown,
                total_trades, net_pnl, rank, composite_score, profit_factor
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        self._db.execute(sql, (
            result.week_start,
            result.cohort_id,
            result.sharpe,
            result.win_rate,
            result.max_drawdown,
            result.total_trades,
            result.net_pnl,
            result.rank,
            result.composite_score,
            result.profit_factor,
        ))

    def save_tournament_results(self, results: list[TournamentResult]) -> None:
        sql = """
            INSERT INTO tournaments (
                week_start, cohort_id, sharpe, win_rate, max_drawdown,
                total_trades, net_pnl, rank, composite_score, profit_factor
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params_list = [
            (r.week_start, r.cohort_id, r.sharpe, r.win_rate,
             r.max_drawdown, r.total_trades, r.net_pnl,
             r.rank, r.composite_score, r.profit_factor)
            for r in results
        ]
        self._db.execute_many(sql, params_list)

    def get_tournament_history(
        self, cohort_id: str, limit: int = 12
    ) -> list[TournamentResult]:
        limit = max(1, limit)
        rows = self._db.fetch_all(
            f"SELECT {_TOURNAMENT_COLUMNS} FROM tournaments WHERE cohort_id = %s ORDER BY week_start DESC LIMIT %s",
            (cohort_id, limit),
        )
        return [self._row_to_tournament(r) for r in rows]

    def get_tournament_results(self, week_start: date) -> list[TournamentResult]:
        rows = self._db.fetch_all(
            f"SELECT {_TOURNAMENT_COLUMNS} FROM tournaments WHERE week_start = %s ORDER BY rank",
            (week_start,),
        )
        return [self._row_to_tournament(r) for r in rows]

    def _row_to_cohort(self, row: dict) -> StrategyCohort:
        config_data = row["config"]
        if isinstance(config_data, str):
            config_data = json.loads(config_data)
        return StrategyCohort(
            cohort_id=row["cohort_id"],
            name=row["name"],
            status=row["status"],
            config=CohortConfig.from_dict(config_data),
            generation=row.get("generation", 0),
            parent_cohort_id=row.get("parent_cohort_id"),
            relegation_count=row.get("relegation_count", 0),
            created_at=row.get("created_at"),
            burn_in_start=row.get("burn_in_start"),
            evaluation_start=row.get("evaluation_start"),
            virtual_capital=row.get("virtual_capital", 100_000.0),
        )

    def _row_to_tournament(self, row: dict) -> TournamentResult:
        return TournamentResult(
            cohort_id=row["cohort_id"],
            week_start=row["week_start"],
            sharpe=row.get("sharpe", 0.0) or 0.0,
            win_rate=row.get("win_rate", 0.0) or 0.0,
            max_drawdown=row.get("max_drawdown", 0.0) or 0.0,
            profit_factor=row.get("profit_factor", 0.0) or 0.0,
            total_trades=row.get("total_trades", 0) or 0,
            net_pnl=row.get("net_pnl", 0.0) or 0.0,
            composite_score=row.get("composite_score", 0.0) or 0.0,
            rank=row.get("rank", 0) or 0,
        )
