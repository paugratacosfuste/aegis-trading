"""Aegis 2.0 RL executor — Gym environment package.

See ``12-AEGIS-2.0-PLAN.md`` §3 for the locked schema:
- Observation: versioned (``obs_v1``). Schema change invalidates policies.
- Action: hybrid ``Tuple(Discrete(5), Box([1.0], [4.0]))``.
- Reward: differential Sharpe (Moody & Saffell 1998) + ``shape_reward()``
  domain-specific penalties (Pau's 10-line contribution).
"""

from __future__ import annotations

OBS_VERSION = "obs_v1"

__all__ = ["OBS_VERSION"]
