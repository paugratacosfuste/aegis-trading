"""Feedback repository: raw SQL CRUD for all feedback tables."""

import json
from datetime import date, datetime

from aegis.common.db import DatabasePool
from aegis.feedback.types import (
    FeatureImportance,
    ModelVersion,
    MonthlyRetrospective,
    RegimePlaybookEntry,
    WeightUpdateLog,
)


class FeedbackRepository:
    """Data access for feedback loop tables."""

    def __init__(self, db: DatabasePool):
        self._db = db

    # ── Weight update log ──────────────────────────────────────────

    def insert_weight_update_log(self, log: WeightUpdateLog) -> None:
        sql = """
            INSERT INTO weight_update_log (
                agent_id, agent_type, old_weight, new_weight,
                hit_rate, information_coefficient, composite_score,
                n_signals, update_date
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        self._db.execute(sql, (
            log.agent_id, log.agent_type, log.old_weight, log.new_weight,
            log.hit_rate, log.ic, log.composite_score,
            log.n_signals, log.update_date,
        ))

    def insert_weight_update_logs(self, logs: list[WeightUpdateLog]) -> None:
        if not logs:
            return
        sql = """
            INSERT INTO weight_update_log (
                agent_id, agent_type, old_weight, new_weight,
                hit_rate, information_coefficient, composite_score,
                n_signals, update_date
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = [
            (l.agent_id, l.agent_type, l.old_weight, l.new_weight,
             l.hit_rate, l.ic, l.composite_score, l.n_signals, l.update_date)
            for l in logs
        ]
        self._db.execute_many(sql, params)

    def get_weight_history(
        self, agent_id: str, limit: int = 30,
    ) -> list[dict]:
        return self._db.fetch_all(
            "SELECT * FROM weight_update_log WHERE agent_id = %s "
            "ORDER BY update_date DESC LIMIT %s",
            (agent_id, max(1, limit)),
        )

    def get_weight_updates_for_date(self, update_date: date) -> list[dict]:
        return self._db.fetch_all(
            "SELECT * FROM weight_update_log WHERE update_date = %s "
            "ORDER BY agent_type, agent_id",
            (update_date,),
        )

    # ── Agent weights (existing table) ─────────────────────────────

    def get_current_weights(self) -> list[dict]:
        return self._db.fetch_all(
            "SELECT agent_id, agent_type, weight, hit_rate, "
            "information_coefficient, rolling_sharpe, updated_at "
            "FROM agent_weights ORDER BY agent_type, agent_id"
        )

    def upsert_agent_weight(
        self,
        agent_id: str,
        agent_type: str,
        weight: float,
        hit_rate: float,
        ic: float,
        rolling_sharpe: float | None = None,
    ) -> None:
        sql = """
            INSERT INTO agent_weights (agent_id, agent_type, weight, hit_rate,
                                       information_coefficient, rolling_sharpe, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (agent_id) DO UPDATE SET
                weight = EXCLUDED.weight,
                hit_rate = EXCLUDED.hit_rate,
                information_coefficient = EXCLUDED.information_coefficient,
                rolling_sharpe = EXCLUDED.rolling_sharpe,
                updated_at = NOW()
        """
        self._db.execute(sql, (agent_id, agent_type, weight, hit_rate, ic, rolling_sharpe))

    # ── Signal-outcome matching ────────────────────────────────────

    def get_signal_outcomes(
        self, since: datetime,
    ) -> list[dict]:
        """Join agent_signals with agent_performance for matched outcomes."""
        sql = """
            SELECT s.agent_id, s.agent_type, s.direction AS predicted_direction,
                   p.actual_return, p.is_correct
            FROM agent_signals s
            JOIN agent_performance p
                ON s.agent_id = p.agent_id AND s.symbol = p.symbol
                AND s.timestamp = p.signal_timestamp
            WHERE s.timestamp >= %s
            ORDER BY s.agent_id, s.timestamp
        """
        return self._db.fetch_all(sql, (since,))

    # ── Model versions ─────────────────────────────────────────────

    def insert_model_version(self, mv: ModelVersion) -> None:
        sql = """
            INSERT INTO model_versions (
                model_id, model_type, version, train_start, train_end,
                val_start, val_end, train_samples, val_samples, val_auc,
                previous_auc, accepted, feature_names, model_path
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        self._db.execute(sql, (
            mv.model_id, mv.model_type, mv.version,
            mv.train_start, mv.train_end, mv.val_start, mv.val_end,
            mv.train_samples, mv.val_samples, mv.val_auc,
            mv.previous_auc, mv.accepted,
            json.dumps(list(mv.feature_names)), mv.model_path,
        ))

    def get_latest_model(self, model_type: str) -> dict | None:
        return self._db.fetch_one(
            "SELECT * FROM model_versions "
            "WHERE model_type = %s AND accepted = TRUE "
            "ORDER BY created_at DESC LIMIT 1",
            (model_type,),
        )

    def get_model_history(
        self, model_type: str, limit: int = 10,
    ) -> list[dict]:
        return self._db.fetch_all(
            "SELECT * FROM model_versions WHERE model_type = %s "
            "ORDER BY created_at DESC LIMIT %s",
            (model_type, max(1, limit)),
        )

    # ── Feature importance ─────────────────────────────────────────

    def insert_feature_importances(
        self, importances: list[FeatureImportance],
    ) -> None:
        if not importances:
            return
        sql = """
            INSERT INTO feature_importance (
                model_id, retrain_date, feature_name,
                shap_importance, rank, status
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """
        params = [
            (fi.model_id, fi.retrain_date, fi.feature_name,
             fi.shap_importance, fi.rank, fi.status)
            for fi in importances
        ]
        self._db.execute_many(sql, params)

    def get_feature_importances(self, model_id: str) -> list[dict]:
        return self._db.fetch_all(
            "SELECT * FROM feature_importance WHERE model_id = %s ORDER BY rank",
            (model_id,),
        )

    def get_latest_feature_importances(self, model_type: str) -> list[dict]:
        """Get feature importances for the latest accepted model of this type."""
        latest = self.get_latest_model(model_type)
        if latest is None:
            return []
        return self.get_feature_importances(latest["model_id"])

    # ── Regime playbook ────────────────────────────────────────────

    def upsert_regime_playbook(self, entry: RegimePlaybookEntry) -> None:
        sql = """
            INSERT INTO regime_playbook (
                regime, last_updated, total_observations,
                best_agent_weights, best_cohort_ids,
                avg_sharpe, avg_win_rate, worst_agent_types,
                recommended_position_size_mult,
                recommended_max_positions,
                recommended_confidence_threshold
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (regime, last_updated) DO UPDATE SET
                total_observations = EXCLUDED.total_observations,
                best_agent_weights = EXCLUDED.best_agent_weights,
                best_cohort_ids = EXCLUDED.best_cohort_ids,
                avg_sharpe = EXCLUDED.avg_sharpe,
                avg_win_rate = EXCLUDED.avg_win_rate,
                worst_agent_types = EXCLUDED.worst_agent_types,
                recommended_position_size_mult = EXCLUDED.recommended_position_size_mult,
                recommended_max_positions = EXCLUDED.recommended_max_positions,
                recommended_confidence_threshold = EXCLUDED.recommended_confidence_threshold
        """
        self._db.execute(sql, (
            entry.regime, entry.last_updated, entry.total_observations,
            json.dumps(dict(entry.best_agent_weights)),
            json.dumps(list(entry.best_cohort_ids)),
            entry.avg_sharpe, entry.avg_win_rate,
            json.dumps(list(entry.worst_agent_types)),
            entry.recommended_position_size_mult,
            entry.recommended_max_positions,
            entry.recommended_confidence_threshold,
        ))

    def get_regime_playbook(self, regime: str) -> dict | None:
        return self._db.fetch_one(
            "SELECT * FROM regime_playbook WHERE regime = %s "
            "ORDER BY last_updated DESC LIMIT 1",
            (regime,),
        )

    def get_all_playbook_entries(self) -> list[dict]:
        return self._db.fetch_all(
            "SELECT DISTINCT ON (regime) * FROM regime_playbook "
            "ORDER BY regime, last_updated DESC"
        )

    # ── Retrospective reports ──────────────────────────────────────

    def insert_retrospective(self, retro: MonthlyRetrospective) -> None:
        report = {
            "production_return": retro.production_return,
            "production_sharpe": retro.production_sharpe,
            "benchmark_return": retro.benchmark_return,
            "alpha": retro.alpha,
            "best_lab_cohort": retro.best_lab_cohort,
            "worst_lab_cohort": retro.worst_lab_cohort,
            "cohorts_promoted": list(retro.cohorts_promoted),
            "cohorts_relegated": list(retro.cohorts_relegated),
            "most_improved_agent": retro.most_improved_agent,
            "most_degraded_agent": retro.most_degraded_agent,
            "agents_evolved": list(retro.agents_evolved),
            "feature_importance_shifts": dict(retro.feature_importance_shifts),
            "regimes_encountered": dict(retro.regimes_encountered),
            "regime_performance": dict(retro.regime_performance),
            "rl_shadow_performance": dict(retro.rl_shadow_performance),
            "recommendations": list(retro.recommendations),
        }
        sql = """
            INSERT INTO retrospective_reports (month, report)
            VALUES (%s, %s)
            ON CONFLICT (month) DO UPDATE SET report = EXCLUDED.report
        """
        self._db.execute(sql, (retro.month, json.dumps(report)))

    def get_retrospective(self, month: str) -> dict | None:
        row = self._db.fetch_one(
            "SELECT * FROM retrospective_reports WHERE month = %s", (month,),
        )
        if row is None:
            return None
        report = row["report"]
        if isinstance(report, str):
            report = json.loads(report)
        return report

    # ── Trade queries for feedback ─────────────────────────────────

    def get_trades_in_range(
        self, start: datetime, end: datetime,
    ) -> list[dict]:
        return self._db.fetch_all(
            "SELECT * FROM trades WHERE exit_time >= %s AND exit_time < %s "
            "ORDER BY exit_time",
            (start, end),
        )

    def get_trades_by_regime(self, since: datetime) -> list[dict]:
        return self._db.fetch_all(
            "SELECT * FROM trades WHERE exit_time >= %s "
            "ORDER BY regime_at_entry, exit_time",
            (since,),
        )
