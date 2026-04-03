"""Tests for scheduler creation."""

from aegis.scheduler import create_scheduler


class TestScheduler:
    def test_create_scheduler(self):
        scheduler = create_scheduler({})
        assert scheduler is not None

    def test_scheduler_not_running(self):
        scheduler = create_scheduler({})
        assert not scheduler.running
