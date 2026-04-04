"""Tests for RL reward functions: incentive alignment."""

import pytest

from aegis.rl.common.reward import (
    exit_management_reward,
    position_sizing_reward,
    weight_allocation_reward,
)


class TestWeightAllocationReward:
    def test_positive_sharpe(self):
        returns = [0.01, 0.02, 0.005, 0.015, 0.01]  # All positive
        reward = weight_allocation_reward(returns)
        assert reward > 0

    def test_negative_returns(self):
        returns = [-0.01, -0.02, -0.005, -0.015, -0.01]
        reward = weight_allocation_reward(returns)
        assert reward < 0

    def test_empty_returns(self):
        assert weight_allocation_reward([]) == 0.0
        assert weight_allocation_reward([0.01]) == 0.0

    def test_zero_volatility(self):
        returns = [0.01, 0.01, 0.01]
        reward = weight_allocation_reward(returns)
        assert reward > 0  # Consistent positive returns


class TestPositionSizingReward:
    def test_profitable_trade(self):
        reward = position_sizing_reward(trade_pnl=100.0, position_size_pct=0.03)
        assert reward > 0

    def test_losing_trade(self):
        reward = position_sizing_reward(trade_pnl=-50.0, position_size_pct=0.03)
        assert reward < 0

    def test_oversized_loss_penalized_more(self):
        small_loss = position_sizing_reward(trade_pnl=-50.0, position_size_pct=0.03)
        big_loss = position_sizing_reward(trade_pnl=-50.0, position_size_pct=0.08)
        assert big_loss < small_loss  # Bigger position with same loss = worse

    def test_drawdown_penalty(self):
        no_dd = position_sizing_reward(trade_pnl=50.0, position_size_pct=0.03, max_drawdown_during=0.0)
        with_dd = position_sizing_reward(trade_pnl=50.0, position_size_pct=0.03, max_drawdown_during=-0.05)
        assert with_dd < no_dd


class TestExitManagementReward:
    def test_hold_small_cost(self):
        reward = exit_management_reward(step_pnl_change=0.0, action_taken=0, bars_held=1, r_multiple=0.0)
        assert reward < 0  # Hold cost

    def test_good_exit_bonus(self):
        reward = exit_management_reward(step_pnl_change=0.1, action_taken=4, bars_held=20, r_multiple=2.5)
        assert reward > 0.5  # PnL + bonus

    def test_late_stop_penalty(self):
        reward = exit_management_reward(step_pnl_change=-0.1, action_taken=4, bars_held=100, r_multiple=-1.5)
        assert reward < -0.1  # PnL + penalty

    def test_good_partial_timing(self):
        reward = exit_management_reward(step_pnl_change=0.05, action_taken=2, bars_held=20, r_multiple=1.8)
        assert reward > 0.3  # PnL + partial bonus

    def test_holding_loser_too_long_penalized(self):
        early = exit_management_reward(step_pnl_change=0.0, action_taken=0, bars_held=10, r_multiple=-0.6)
        late = exit_management_reward(step_pnl_change=0.0, action_taken=0, bars_held=50, r_multiple=-0.6)
        assert late < early  # Holding losers longer is worse

    def test_tighten_when_profitable(self):
        reward = exit_management_reward(step_pnl_change=0.0, action_taken=1, bars_held=20, r_multiple=1.0)
        assert reward > 0  # Small bonus for good tightening
