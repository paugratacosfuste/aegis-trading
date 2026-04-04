"""Shadow tracker: observes decisions and records counterfactual predictions.

All components are optional. All calls are exception-safe — RL errors
never crash the trading system.
"""

import logging
from datetime import datetime, timezone

from aegis.common.types import AgentSignal
from aegis.rl.types import ExitAction

logger = logging.getLogger(__name__)


class ShadowTracker:
    """Observe rule-based decisions, record what RL would have done.

    All methods are no-op safe: if a component is None, the call is skipped.
    All methods catch exceptions to never affect the trading pipeline.
    """

    def __init__(
        self,
        weight_bandit=None,
        position_sizer=None,
        exit_manager=None,
    ):
        self._weight_bandit = weight_bandit
        self._position_sizer = position_sizer
        self._exit_manager = exit_manager

        # In-memory tracking for backtest mode
        self._predictions: list[dict] = []

    def on_ensemble_vote(
        self,
        signals: list[AgentSignal],
        regime: str,
        portfolio_value: float,
        equity_curve: list[float] | None = None,
        baseline_weights: dict[str, float] | None = None,
    ) -> dict[str, float] | None:
        """Called after ensemble vote. Records weight allocation prediction.

        Returns the RL-suggested weights (for tracking only, not applied).
        """
        if self._weight_bandit is None:
            return None

        try:
            from aegis.rl.weight_allocator.feature_extractor import extract_context

            context = extract_context(signals, regime, portfolio_value, equity_curve)
            config = self._weight_bandit.predict(context)

            self._predictions.append({
                "component": "weight_allocator",
                "timestamp": datetime.now(timezone.utc),
                "prediction": {"config_id": config.config_id, "weights": config.weights},
                "baseline": baseline_weights,
                "context": context.tolist(),
            })

            return config.weights
        except Exception:
            logger.debug("Shadow weight allocation failed", exc_info=True)
            return None

    def on_position_sized(
        self,
        obs,
        kelly_size: float,
        actual_size: float,
    ) -> float | None:
        """Called after position sizing. Records RL size prediction."""
        if self._position_sizer is None:
            return None

        try:
            rl_size = self._position_sizer.predict(obs, kelly_size=kelly_size)

            self._predictions.append({
                "component": "position_sizer",
                "timestamp": datetime.now(timezone.utc),
                "prediction": {"rl_size": rl_size},
                "baseline": {"kelly_size": kelly_size, "actual_size": actual_size},
            })

            return rl_size
        except Exception:
            logger.debug("Shadow position sizing failed", exc_info=True)
            return None

    def on_exit_check(
        self,
        obs,
        actual_action: str,
    ) -> ExitAction | None:
        """Called at each exit check. Records RL exit prediction."""
        if self._exit_manager is None:
            return None

        try:
            rl_action = self._exit_manager.predict(obs)

            self._predictions.append({
                "component": "exit_manager",
                "timestamp": datetime.now(timezone.utc),
                "prediction": {"rl_action": rl_action.name},
                "baseline": {"actual_action": actual_action},
            })

            return rl_action
        except Exception:
            logger.debug("Shadow exit check failed", exc_info=True)
            return None

    def on_trade_closed(
        self,
        trade: dict,
    ) -> None:
        """Called when a trade closes. Updates counterfactual tracking."""
        try:
            self._predictions.append({
                "component": "trade_outcome",
                "timestamp": datetime.now(timezone.utc),
                "trade": {
                    "symbol": trade.get("symbol"),
                    "net_pnl": trade.get("net_pnl"),
                    "exit_reason": trade.get("exit_reason"),
                },
            })
        except Exception:
            logger.debug("Shadow trade tracking failed", exc_info=True)

    def get_summary(self) -> dict:
        """Return summary of shadow predictions for reporting."""
        from collections import Counter

        by_component = Counter(p["component"] for p in self._predictions)
        return {
            "total_predictions": len(self._predictions),
            "by_component": dict(by_component),
            "predictions": self._predictions,
        }
