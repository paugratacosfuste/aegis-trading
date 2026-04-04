"""Shadow mode reporter: compare RL vs baseline performance."""

import logging

from aegis.rl.constants import PROMOTION_MIN_DAYS, PROMOTION_SHARPE_THRESHOLD
from aegis.rl.types import PromotionStage, ShadowSummary

logger = logging.getLogger(__name__)


def generate_report(
    shadow_predictions: list[dict],
) -> dict:
    """Generate shadow performance report from in-memory predictions.

    Returns comparison metrics for each RL component.
    """
    by_component: dict[str, list[dict]] = {}
    for p in shadow_predictions:
        comp = p["component"]
        if comp != "trade_outcome":
            by_component.setdefault(comp, []).append(p)

    report = {}
    for component, preds in by_component.items():
        report[component] = {
            "total_predictions": len(preds),
            "sample_prediction": preds[0]["prediction"] if preds else {},
        }

    return report


def check_promotion_criteria(
    summary: ShadowSummary,
) -> bool:
    """Check if an RL component is ready for promotion.

    Criteria:
      1. At least PROMOTION_MIN_DAYS in shadow mode
      2. RL Sharpe exceeds baseline Sharpe by PROMOTION_SHARPE_THRESHOLD
      3. Currently in SHADOW stage
    """
    if summary.stage != PromotionStage.SHADOW:
        return False

    if summary.total_days < PROMOTION_MIN_DAYS:
        logger.info(
            "%s: only %d/%d days — not ready",
            summary.component, summary.total_days, PROMOTION_MIN_DAYS,
        )
        return False

    sharpe_margin = summary.rl_sharpe - summary.baseline_sharpe
    if sharpe_margin < PROMOTION_SHARPE_THRESHOLD:
        logger.info(
            "%s: Sharpe margin %.3f < %.3f threshold — not ready",
            summary.component, sharpe_margin, PROMOTION_SHARPE_THRESHOLD,
        )
        return False

    logger.info(
        "%s: PROMOTION READY — %d days, Sharpe margin %.3f",
        summary.component, summary.total_days, sharpe_margin,
    )
    return True
