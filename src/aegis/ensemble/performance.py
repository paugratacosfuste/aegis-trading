"""Agent performance tracking with Bayesian weight updates.

In-memory tracker for backtest mode. DB persistence for live mode
uses the agent_performance table (migration 014).
"""

from dataclasses import dataclass, field

import numpy as np

_MIN_RETURN = 0.0005  # Ignore returns below this (noise)
_PRIOR_ALPHA = 5.0    # Beta distribution prior successes
_PRIOR_BETA = 5.0     # Beta distribution prior failures
_WEIGHT_MIN = 0.1
_WEIGHT_MAX = 3.0


@dataclass
class _AgentRecord:
    predictions: list[float] = field(default_factory=list)
    actuals: list[float] = field(default_factory=list)
    n_correct: int = 0
    n_predictions: int = 0


class AgentPerformanceTracker:
    """Track agent prediction accuracy and compute Bayesian weights."""

    def __init__(self) -> None:
        self._records: dict[str, _AgentRecord] = {}

    def _get_record(self, agent_id: str) -> _AgentRecord:
        if agent_id not in self._records:
            self._records[agent_id] = _AgentRecord()
        return self._records[agent_id]

    def record_outcome(
        self,
        agent_id: str,
        predicted_direction: float,
        actual_return: float,
    ) -> None:
        """Record a prediction outcome for an agent.

        Args:
            agent_id: The agent identifier.
            predicted_direction: Agent's predicted direction [-1, 1].
            actual_return: Actual market return for the holding period.
        """
        if abs(actual_return) < _MIN_RETURN:
            return  # Noise, skip

        rec = self._get_record(agent_id)
        rec.predictions.append(predicted_direction)
        rec.actuals.append(actual_return)
        rec.n_predictions += 1

        # Correct if signs match
        if (predicted_direction > 0 and actual_return > 0) or \
           (predicted_direction < 0 and actual_return < 0):
            rec.n_correct += 1

    def get_weight(self, agent_id: str) -> float:
        """Get Bayesian-updated weight for an agent.

        Uses Beta(alpha, beta) posterior where:
            alpha = prior_alpha + n_correct
            beta  = prior_beta + n_wrong
            weight = alpha / (alpha + beta) * 2  (scaled so 0.5 hit_rate = 1.0)
        """
        if agent_id not in self._records:
            return 1.0

        rec = self._records[agent_id]
        if rec.n_predictions == 0:
            return 1.0

        alpha = _PRIOR_ALPHA + rec.n_correct
        beta = _PRIOR_BETA + (rec.n_predictions - rec.n_correct)
        posterior_mean = alpha / (alpha + beta)

        # Scale: 0.5 -> 1.0, 1.0 -> 2.0, 0.0 -> 0.0
        weight = posterior_mean * 2.0
        return max(_WEIGHT_MIN, min(_WEIGHT_MAX, weight))

    def get_stats(self, agent_id: str) -> dict:
        """Get performance statistics for an agent."""
        if agent_id not in self._records or self._records[agent_id].n_predictions == 0:
            return {
                "n_predictions": 0,
                "n_correct": 0,
                "hit_rate": 0.0,
                "ic": 0.0,
                "weight": 1.0,
            }

        rec = self._records[agent_id]
        hit_rate = rec.n_correct / rec.n_predictions if rec.n_predictions > 0 else 0.0

        # Information coefficient: correlation between predictions and actuals
        ic = 0.0
        if len(rec.predictions) >= 3:
            preds = np.array(rec.predictions)
            acts = np.array(rec.actuals)
            if np.std(preds) > 0 and np.std(acts) > 0:
                ic = float(np.corrcoef(preds, acts)[0, 1])

        return {
            "n_predictions": rec.n_predictions,
            "n_correct": rec.n_correct,
            "hit_rate": hit_rate,
            "ic": ic,
            "weight": self.get_weight(agent_id),
        }

    def get_all_weights(self) -> dict[str, float]:
        """Get weights for all tracked agents."""
        return {aid: self.get_weight(aid) for aid in self._records}

    def reset(self, agent_id: str) -> None:
        """Reset an agent's performance history."""
        if agent_id in self._records:
            del self._records[agent_id]
