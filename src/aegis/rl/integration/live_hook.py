"""Live shadow hook: wires ShadowTracker into LabOrchestrator with DB persistence."""

import logging

from aegis.rl.shadow.tracker import ShadowTracker

logger = logging.getLogger(__name__)


class LiveShadowHook:
    """Hook for live/lab mode with optional DB persistence."""

    def __init__(self, rl_config: dict | None = None, db=None):
        self._config = rl_config or {}
        self._db = db
        self._tracker: ShadowTracker | None = None
        self._repository = None

    def setup(self) -> None:
        """Initialize RL components and DB repository."""
        if not self._config.get("enabled", False):
            return

        # Reuse backtest hook logic for component creation
        from aegis.rl.integration.backtest_hook import BacktestShadowHook

        bt_hook = BacktestShadowHook(self._config)
        bt_hook.setup()
        self._tracker = bt_hook.tracker

        # Set up DB persistence if available
        if self._db is not None:
            try:
                from aegis.rl.shadow.repository import ShadowRepository

                self._repository = ShadowRepository(self._db)
                logger.info("Shadow repository initialized with DB")
            except Exception:
                logger.warning("Failed to initialize shadow repository", exc_info=True)

    @property
    def tracker(self) -> ShadowTracker | None:
        return self._tracker

    def get_summary(self) -> dict:
        if self._tracker is None:
            return {}
        try:
            return self._tracker.get_summary()
        except Exception:
            return {}
