"""Tests for regime playbook builder."""

from datetime import date

import pytest

from aegis.feedback.regime_playbook import build_playbook_entries


class TestBuildPlaybookEntries:
    def test_empty_trades(self):
        result = build_playbook_entries(trades=[], agent_performances=[])
        assert result == []

    def test_single_regime_enough_observations(self):
        trades = [
            {"regime_at_entry": "bull", "net_pnl": 100.0, "return_pct": 0.02}
            for _ in range(35)
        ]
        result = build_playbook_entries(
            trades=trades, agent_performances=[], min_observations=30,
        )
        assert len(result) == 1
        assert result[0].regime == "bull"
        assert result[0].total_observations == 35
        assert result[0].avg_win_rate == 1.0

    def test_below_min_observations_excluded(self):
        trades = [
            {"regime_at_entry": "crisis", "net_pnl": -50.0, "return_pct": -0.01}
            for _ in range(20)
        ]
        result = build_playbook_entries(
            trades=trades, agent_performances=[], min_observations=30,
        )
        assert len(result) == 0

    def test_multiple_regimes(self):
        trades = (
            [{"regime_at_entry": "bull", "net_pnl": 80.0, "return_pct": 0.02}] * 40
            + [{"regime_at_entry": "bear", "net_pnl": -30.0, "return_pct": -0.01}] * 35
        )
        result = build_playbook_entries(
            trades=trades, agent_performances=[], min_observations=30,
        )
        assert len(result) == 2
        regimes = {e.regime for e in result}
        assert regimes == {"bull", "bear"}

    def test_best_worst_agent_types(self):
        agent_perfs = [
            {"agent_id": "tech_01", "agent_type": "technical",
             "regime_at_entry": "bull", "is_correct": True},
            {"agent_id": "tech_01", "agent_type": "technical",
             "regime_at_entry": "bull", "is_correct": True},
            {"agent_id": "stat_01", "agent_type": "statistical",
             "regime_at_entry": "bull", "is_correct": False},
            {"agent_id": "stat_01", "agent_type": "statistical",
             "regime_at_entry": "bull", "is_correct": False},
        ]
        trades = [
            {"regime_at_entry": "bull", "net_pnl": 100.0, "return_pct": 0.02}
        ] * 30
        result = build_playbook_entries(
            trades=trades, agent_performances=agent_perfs, min_observations=30,
        )
        assert len(result) == 1
        entry = result[0]
        # Technical should be best, statistical worst
        assert entry.best_agent_weights.get("technical", 0) > 0
        assert "statistical" in entry.worst_agent_types

    def test_recommended_position_size_adjusts(self):
        # High-vol regime with losses -> lower position size
        trades = [
            {"regime_at_entry": "high_volatility", "net_pnl": -50.0, "return_pct": -0.03}
        ] * 30
        result = build_playbook_entries(
            trades=trades, agent_performances=[], min_observations=30,
        )
        assert len(result) == 1
        assert result[0].recommended_position_size_mult <= 1.0

    def test_playbook_entry_has_correct_date(self):
        trades = [
            {"regime_at_entry": "bull", "net_pnl": 50.0, "return_pct": 0.01}
        ] * 30
        result = build_playbook_entries(
            trades=trades, agent_performances=[],
            min_observations=30, as_of_date=date(2026, 4, 1),
        )
        assert result[0].last_updated == date(2026, 4, 1)
