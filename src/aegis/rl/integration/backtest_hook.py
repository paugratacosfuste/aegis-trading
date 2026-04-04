"""Backtest shadow hook: wires ShadowTracker into BacktestEngine."""

import logging

from aegis.rl.shadow.tracker import ShadowTracker

logger = logging.getLogger(__name__)


class BacktestShadowHook:
    """Hook that creates and manages a ShadowTracker for backtesting.

    All methods are exception-safe. RL failures never affect backtest results.
    """

    def __init__(self, rl_config: dict | None = None):
        self._config = rl_config or {}
        self._tracker: ShadowTracker | None = None

    def setup(self) -> None:
        """Initialize RL components based on config."""
        if not self._config.get("enabled", False):
            logger.info("RL shadow mode disabled")
            return

        components = self._config.get("components", {})

        weight_bandit = None
        position_sizer = None
        exit_manager = None

        try:
            if components.get("weight_allocator", {}).get("enabled", False):
                from aegis.rl.weight_allocator.bandit import WeightAllocatorBandit

                eps = components["weight_allocator"].get("exploration_rate", 0.1)
                weight_bandit = WeightAllocatorBandit(exploration_rate=eps)
                logger.info("Shadow weight allocator enabled (eps=%.2f)", eps)
        except Exception:
            logger.warning("Failed to initialize shadow weight allocator", exc_info=True)

        try:
            if components.get("position_sizer", {}).get("enabled", False):
                from aegis.rl.position_sizer.agent import PositionSizerAgent

                position_sizer = PositionSizerAgent()
                logger.info("Shadow position sizer enabled")
        except Exception:
            logger.warning("Failed to initialize shadow position sizer", exc_info=True)

        try:
            if components.get("exit_manager", {}).get("enabled", False):
                from aegis.rl.exit_manager.agent import ExitManagerAgent

                exit_manager = ExitManagerAgent()
                logger.info("Shadow exit manager enabled")
        except Exception:
            logger.warning("Failed to initialize shadow exit manager", exc_info=True)

        self._tracker = ShadowTracker(
            weight_bandit=weight_bandit,
            position_sizer=position_sizer,
            exit_manager=exit_manager,
        )
        logger.info("Shadow tracker initialized")

    @property
    def tracker(self) -> ShadowTracker | None:
        return self._tracker

    def get_summary(self) -> dict:
        """Get shadow tracking summary for results."""
        if self._tracker is None:
            return {}
        try:
            return self._tracker.get_summary()
        except Exception:
            logger.debug("Failed to get shadow summary", exc_info=True)
            return {}
