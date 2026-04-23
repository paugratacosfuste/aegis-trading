"""Differential Sharpe reward for the Aegis 2.0 RL executor.

Implements Moody & Saffell (1998), "Reinforcement Learning for Trading".
Provides a dense, per-bar reward that is the marginal contribution of the
latest return to an exponentially-weighted Sharpe ratio estimate.

Intuition
---------
Instead of getting reward only at episode end (terminal Sharpe), the agent
sees how the current bar moved the sliding Sharpe estimate. Upside with
low volatility ⇒ positive reward; upside amid high volatility is
discounted; drawdowns are penalized proportionally to how unexpected they
are.

State
-----
Two EMA scalars:
  A = E[r_t]      (mean return)
  B = E[r_t^2]    (mean squared return)
Variance is ``B - A^2``; volatility is ``sqrt(B - A^2)``.

Update (per bar with return ``r_t`` and memory rate ``eta``):
    delta_A = r_t - A_prev
    delta_B = r_t^2 - B_prev
    A_new   = A_prev + eta * delta_A
    B_new   = B_prev + eta * delta_B
    D_t     = (B_prev * delta_A - 0.5 * A_prev * delta_B) / (B_prev - A_prev^2)^1.5

When the variance term is ~0 (e.g., very first bar, or a pathological
constant-return run), ``D_t`` is clipped to 0 to avoid division by zero.
"""

from __future__ import annotations

from typing import Any

_VAR_EPSILON = 1e-8


class DifferentialSharpeReward:
    """Per-bar differential Sharpe reward with online EMA state."""

    def __init__(self, eta: float = 0.01) -> None:
        if not 0.0 < eta <= 1.0:
            raise ValueError("eta must be in (0, 1]")
        self._eta = eta
        self._A = 0.0
        self._B = 0.0

    def step(self, r_t: float) -> float:
        """Consume the bar return and return the differential Sharpe reward.

        Also updates the internal EMA state in-place (learning signal is
        online, so the state must advance with every bar).
        """
        A_prev = self._A
        B_prev = self._B

        delta_A = r_t - A_prev
        delta_B = r_t * r_t - B_prev

        variance = B_prev - A_prev * A_prev
        if variance < _VAR_EPSILON:
            reward = 0.0
        else:
            denom = variance ** 1.5
            reward = (B_prev * delta_A - 0.5 * A_prev * delta_B) / denom

        self._A = A_prev + self._eta * delta_A
        self._B = B_prev + self._eta * delta_B
        return reward

    def reset(self) -> None:
        self._A = 0.0
        self._B = 0.0

    @property
    def ema_return(self) -> float:
        return self._A

    @property
    def ema_squared_return(self) -> float:
        return self._B

    @property
    def variance(self) -> float:
        return max(0.0, self._B - self._A * self._A)


# ---------------------------------------------------------------------------
# shape_reward — Pau's 10-line contribution slot
# ---------------------------------------------------------------------------


def shape_reward(
    base_reward: float,
    *,
    state: dict[str, Any],
    action: dict[str, Any],
    market: dict[str, Any],
) -> float:
    """Domain-specific reward shaping on top of differential Sharpe.

    TODO(Pau): encode your trading values here as additive penalties /
    bonuses. Examples the plan suggests (§3.4):

        if market.get("is_fomc_day"):                base_reward -= 0.05
        if action.get("opened_trade"):               base_reward -= 0.001
        if state.get("exposure", 0.0) > 0.5:
            base_reward -= 0.01 * (state["exposure"] - 0.5)
        if market.get("vix", 0.0) > 40 and action.get("opened_trade"):
            base_reward -= 0.02

    Keep the total magnitude of shaping below ~20% of |base_reward| so
    the differential Sharpe signal stays dominant.
    """
    return base_reward
