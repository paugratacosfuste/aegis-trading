"""Exit management reward wrapper."""

from aegis.rl.common.reward import exit_management_reward


def compute_reward(
    step_pnl_change: float,
    action_taken: int,
    bars_held: int,
    r_multiple: float,
) -> float:
    """Compute exit management reward. Thin wrapper."""
    return exit_management_reward(step_pnl_change, action_taken, bars_held, r_multiple)
