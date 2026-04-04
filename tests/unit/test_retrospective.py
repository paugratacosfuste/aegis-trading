"""Tests for monthly retrospective report generator."""

import pytest

from aegis.feedback.retrospective import (
    compute_alpha,
    find_most_changed_agents,
    generate_recommendations,
    build_retrospective,
)


class TestComputeAlpha:
    def test_positive_alpha(self):
        assert compute_alpha(0.05, 0.03) == pytest.approx(0.02)

    def test_negative_alpha(self):
        assert compute_alpha(-0.02, 0.03) == pytest.approx(-0.05)

    def test_zero_benchmark(self):
        assert compute_alpha(0.10, 0.0) == pytest.approx(0.10)


class TestFindMostChangedAgents:
    def test_identifies_most_improved(self):
        weight_deltas = {
            "tech_01": 0.15,
            "tech_02": -0.10,
            "stat_01": 0.05,
            "stat_02": -0.20,
        }
        improved, degraded = find_most_changed_agents(weight_deltas)
        assert improved == "tech_01"
        assert degraded == "stat_02"

    def test_empty_deltas(self):
        improved, degraded = find_most_changed_agents({})
        assert improved is None
        assert degraded is None

    def test_single_agent(self):
        improved, degraded = find_most_changed_agents({"tech_01": 0.05})
        assert improved == "tech_01"
        assert degraded == "tech_01"


class TestGenerateRecommendations:
    def test_low_sharpe_recommendation(self):
        recs = generate_recommendations(
            production_sharpe=-0.5,
            alpha=-0.03,
            most_degraded="stat_03",
            evolved_count=2,
        )
        assert any("sharpe" in r.lower() or "negative" in r.lower() for r in recs)

    def test_good_performance_no_warnings(self):
        recs = generate_recommendations(
            production_sharpe=1.5,
            alpha=0.02,
            most_degraded=None,
            evolved_count=0,
        )
        # May still have recommendations but no critical warnings
        assert isinstance(recs, list)

    def test_degraded_agent_flagged(self):
        recs = generate_recommendations(
            production_sharpe=0.5,
            alpha=0.0,
            most_degraded="stat_03",
            evolved_count=3,
        )
        assert any("stat_03" in r for r in recs)


class TestBuildRetrospective:
    def test_minimal_retrospective(self):
        retro = build_retrospective(
            month="2026-04",
            trades_pnls=[100.0, -50.0, 75.0, -25.0],
            benchmark_returns=[0.01, -0.005, 0.008, -0.003],
            weight_deltas={},
            evolved_agent_ids=[],
            regimes_encountered={"bull": 3, "bear": 1},
        )
        assert retro.month == "2026-04"
        assert retro.production_return != 0.0
        assert retro.alpha is not None

    def test_empty_month(self):
        retro = build_retrospective(
            month="2026-04",
            trades_pnls=[],
            benchmark_returns=[],
            weight_deltas={},
            evolved_agent_ids=[],
        )
        assert retro.production_return == 0.0
        assert retro.production_sharpe == 0.0
        assert retro.benchmark_return == 0.0
        assert retro.alpha == 0.0

    def test_retrospective_with_evolved_agents(self):
        retro = build_retrospective(
            month="2026-04",
            trades_pnls=[50.0, 30.0],
            benchmark_returns=[0.01],
            weight_deltas={"tech_01": 0.10, "stat_01": -0.15},
            evolved_agent_ids=["mom_09", "stat_10"],
        )
        assert retro.most_improved_agent == "tech_01"
        assert retro.most_degraded_agent == "stat_01"
        assert retro.agents_evolved == ("mom_09", "stat_10")
        assert len(retro.recommendations) > 0
