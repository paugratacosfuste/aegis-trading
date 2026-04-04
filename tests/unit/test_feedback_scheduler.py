"""Tests for feedback scheduler wiring."""

import pytest

from aegis.feedback.scheduler import get_job_configs


class TestGetJobConfigs:
    def test_all_enabled(self):
        config = {
            "daily_weight_update": {"enabled": True, "cron_hour_utc": 22, "cron_minute": 0},
            "weekly_retrain": {"enabled": True, "cron_day_of_week": "sun", "cron_hour_utc": 3},
            "monthly_evolution": {"enabled": True, "cron_day": 1, "cron_hour_utc": 4},
            "regime_playbook": {"enabled": True},
            "retrospective": {"enabled": True, "cron_day": 2, "cron_hour_utc": 5},
        }
        jobs = get_job_configs(config)
        assert len(jobs) == 4  # 4 scheduled jobs (playbook runs with evolution)
        ids = {j["id"] for j in jobs}
        assert "feedback_daily_weights" in ids
        assert "feedback_weekly_retrain" in ids
        assert "feedback_monthly_evolution" in ids
        assert "feedback_retrospective" in ids

    def test_all_disabled(self):
        config = {
            "daily_weight_update": {"enabled": False},
            "weekly_retrain": {"enabled": False},
            "monthly_evolution": {"enabled": False},
            "regime_playbook": {"enabled": False},
            "retrospective": {"enabled": False},
        }
        jobs = get_job_configs(config)
        assert len(jobs) == 0

    def test_partial_enabled(self):
        config = {
            "daily_weight_update": {"enabled": True, "cron_hour_utc": 22, "cron_minute": 0},
            "weekly_retrain": {"enabled": False},
            "monthly_evolution": {"enabled": False},
            "regime_playbook": {"enabled": False},
            "retrospective": {"enabled": False},
        }
        jobs = get_job_configs(config)
        assert len(jobs) == 1
        assert jobs[0]["id"] == "feedback_daily_weights"

    def test_empty_config(self):
        jobs = get_job_configs({})
        assert len(jobs) == 0

    def test_correct_cron_params(self):
        config = {
            "daily_weight_update": {"enabled": True, "cron_hour_utc": 22, "cron_minute": 15},
            "weekly_retrain": {"enabled": False},
            "monthly_evolution": {"enabled": False},
            "regime_playbook": {"enabled": False},
            "retrospective": {"enabled": False},
        }
        jobs = get_job_configs(config)
        daily = jobs[0]
        assert daily["trigger"] == "cron"
        assert daily["hour"] == 22
        assert daily["minute"] == 15
