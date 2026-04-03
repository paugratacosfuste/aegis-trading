"""APScheduler-based job scheduler for Aegis trading system.

Jobs:
- collect_crypto_prices: 60s interval (WS reconnect check)
- run_signal_pipeline: 300s interval (agents -> ensemble -> risk -> execute)
- check_exits: 60s interval (stop-loss monitoring)
- collect_equities: daily cron after market close
"""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger(__name__)


def create_scheduler(config: dict) -> AsyncIOScheduler:
    """Create and configure the scheduler with all jobs."""
    scheduler = AsyncIOScheduler()

    # Jobs will be registered by main.py with the actual pipeline functions
    logger.info("Scheduler created")
    return scheduler
