"""Feedback loop scheduler: registers daily/weekly/monthly jobs with APScheduler.

All jobs are exception-safe — errors are logged, never crash the trading system.
"""

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from aegis.common.db import DatabasePool

logger = logging.getLogger(__name__)


def get_job_configs(feedback_config: dict[str, Any]) -> list[dict]:
    """Build job configuration dicts from the feedback config section.

    Returns list of dicts with id, trigger, hour, minute, etc.
    Does NOT require APScheduler — pure config -> job-spec mapping.
    """
    jobs: list[dict] = []

    # Daily weight update
    daily = feedback_config.get("daily_weight_update", {})
    if daily.get("enabled", False):
        jobs.append({
            "id": "feedback_daily_weights",
            "trigger": "cron",
            "hour": daily.get("cron_hour_utc", 22),
            "minute": daily.get("cron_minute", 0),
        })

    # Weekly retrain
    weekly = feedback_config.get("weekly_retrain", {})
    if weekly.get("enabled", False):
        jobs.append({
            "id": "feedback_weekly_retrain",
            "trigger": "cron",
            "day_of_week": weekly.get("cron_day_of_week", "sun"),
            "hour": weekly.get("cron_hour_utc", 3),
            "minute": 0,
        })

    # Monthly evolution (includes regime playbook)
    monthly = feedback_config.get("monthly_evolution", {})
    if monthly.get("enabled", False):
        jobs.append({
            "id": "feedback_monthly_evolution",
            "trigger": "cron",
            "day": monthly.get("cron_day", 1),
            "hour": monthly.get("cron_hour_utc", 4),
            "minute": 0,
        })

    # Monthly retrospective
    retro = feedback_config.get("retrospective", {})
    if retro.get("enabled", False):
        jobs.append({
            "id": "feedback_retrospective",
            "trigger": "cron",
            "day": retro.get("cron_day", 2),
            "hour": retro.get("cron_hour_utc", 5),
            "minute": 0,
        })

    return jobs


def register_feedback_jobs(
    scheduler: Any,
    db: DatabasePool,
    feedback_config: dict[str, Any],
) -> None:
    """Register all enabled feedback jobs with an APScheduler instance.

    Args:
        scheduler: AsyncIOScheduler instance.
        db: DatabasePool for DB operations.
        feedback_config: The 'feedback' section from Settings.
    """
    job_configs = get_job_configs(feedback_config)
    if not job_configs:
        logger.info("No feedback jobs enabled")
        return

    for jc in job_configs:
        job_id = jc.pop("id")
        trigger = jc.pop("trigger")

        job_func = _make_job_func(job_id, db, feedback_config)

        scheduler.add_job(
            job_func,
            trigger,
            id=job_id,
            **jc,
        )
        logger.info("Registered feedback job: %s", job_id)


def _make_job_func(
    job_id: str,
    db: DatabasePool,
    feedback_config: dict[str, Any],
) -> Callable[[], Coroutine[Any, Any, None]]:
    """Create an exception-safe async job function."""

    async def _daily_weights_job() -> None:
        try:
            from datetime import datetime, timedelta, timezone

            from aegis.feedback.daily_weights import run_daily_update
            from aegis.feedback.repository import FeedbackRepository

            repo = FeedbackRepository(db)
            cfg = feedback_config.get("daily_weight_update", {})

            lookback_days = cfg.get("lookback_days", 30)
            since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            signal_outcomes = repo.get_signal_outcomes(since)
            current_weights_rows = repo.get_current_weights()

            # Build weights dict, preserving rolling_sharpe for re-persistence
            current_weights = {}
            rolling_sharpes: dict[str, float | None] = {}
            for r in current_weights_rows:
                aid = r["agent_id"]
                current_weights[aid] = {
                    "agent_type": r["agent_type"],
                    "weight": r.get("weight", 0.15),
                }
                rolling_sharpes[aid] = r.get("rolling_sharpe")

            logs = run_daily_update(
                signal_outcomes=signal_outcomes,
                current_weights=current_weights,
                learning_rate=cfg.get("learning_rate", 0.10),
                min_signals=cfg.get("min_signals", 5),
                weight_min=cfg.get("weight_min", 0.05),
                weight_max=cfg.get("weight_max", 0.30),
            )

            # Persist — preserve existing rolling_sharpe values
            for log in logs:
                repo.upsert_agent_weight(
                    log.agent_id, log.agent_type, log.new_weight,
                    log.hit_rate, log.ic,
                    rolling_sharpe=rolling_sharpes.get(log.agent_id),
                )
            repo.insert_weight_update_logs(logs)
            logger.info("Daily weight update: %d agents updated", len(logs))

        except Exception:
            logger.exception("Daily weight update failed")

    async def _weekly_retrain_job() -> None:
        try:
            logger.warning(
                "Weekly retrain job is a stub — no model retraining will occur. "
                "Implement full pipeline or set weekly_retrain.enabled: false."
            )
        except Exception:
            logger.exception("Weekly retrain failed")

    async def _monthly_evolution_job() -> None:
        try:
            logger.warning(
                "Monthly evolution job is a stub — no agent evolution will occur. "
                "Implement full pipeline or set monthly_evolution.enabled: false."
            )
        except Exception:
            logger.exception("Monthly evolution failed")

    async def _retrospective_job() -> None:
        try:
            logger.warning(
                "Monthly retrospective job is a stub — no report will be generated. "
                "Implement full pipeline or set retrospective.enabled: false."
            )
        except Exception:
            logger.exception("Monthly retrospective failed")

    dispatch: dict[str, Callable[[], Coroutine[Any, Any, None]]] = {
        "feedback_daily_weights": _daily_weights_job,
        "feedback_weekly_retrain": _weekly_retrain_job,
        "feedback_monthly_evolution": _monthly_evolution_job,
        "feedback_retrospective": _retrospective_job,
    }
    func = dispatch.get(job_id)
    if func is None:
        raise ValueError(f"No job function registered for job_id={job_id!r}")
    return func
