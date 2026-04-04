"""Shadow mode DB repository. Raw SQL, no ORM."""

import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ShadowRepository:
    """Persistence for RL shadow predictions and outcomes."""

    def __init__(self, db):
        self._db = db

    def insert_prediction(
        self,
        model_type: str,
        timestamp: datetime,
        symbol: str,
        prediction: dict,
        context_features: dict,
        model_version: str,
        mode: str = "shadow",
    ) -> int:
        """Insert a shadow prediction. Returns prediction id."""
        row = self._db.fetch_one(
            """
            INSERT INTO rl_predictions
                (model_type, timestamp, symbol, prediction, context_features,
                 model_version, mode)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                model_type, timestamp, symbol,
                json.dumps(prediction),
                json.dumps(context_features),
                model_version, mode,
            ),
        )
        return row["id"]

    def update_outcome(
        self,
        prediction_id: int,
        actual_outcome: dict,
        counterfactual_pnl: float,
    ) -> None:
        """Update a prediction with actual outcome."""
        self._db.execute(
            """
            UPDATE rl_predictions
            SET actual_outcome = %s, counterfactual_pnl = %s
            WHERE id = %s
            """,
            (json.dumps(actual_outcome), counterfactual_pnl, prediction_id),
        )

    def get_counterfactual_summary(
        self,
        model_type: str,
        since: datetime | None = None,
    ) -> dict:
        """Get aggregate counterfactual performance for a model type."""
        where = "WHERE model_type = %s AND counterfactual_pnl IS NOT NULL"
        params: list = [model_type]

        if since:
            where += " AND timestamp >= %s"
            params.append(since)

        row = self._db.fetch_one(
            f"""
            SELECT
                COUNT(*) as total_predictions,
                SUM(counterfactual_pnl) as total_counterfactual_pnl,
                AVG(counterfactual_pnl) as avg_counterfactual_pnl,
                COUNT(CASE WHEN counterfactual_pnl > 0 THEN 1 END) as positive_count
            FROM rl_predictions
            {where}
            """,
            tuple(params),
        )

        return {
            "model_type": model_type,
            "total_predictions": row["total_predictions"] or 0,
            "total_counterfactual_pnl": float(row["total_counterfactual_pnl"] or 0),
            "avg_counterfactual_pnl": float(row["avg_counterfactual_pnl"] or 0),
            "positive_count": row["positive_count"] or 0,
        }
