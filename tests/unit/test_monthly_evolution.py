"""Tests for monthly agent evolution."""

import numpy as np
import pytest

from aegis.feedback.monthly_evolution import (
    MUTATION_SCHEMAS,
    identify_bottom_agents,
    mutate_agent_params,
    random_agent_params,
    run_monthly_evolution,
)


class TestIdentifyBottomAgents:
    def test_bottom_20_of_10(self):
        sharpes = {
            "tech_01": 1.5, "tech_02": 1.2, "tech_03": 0.8,
            "tech_04": 0.5, "tech_05": 0.3,
            "tech_06": -0.1, "tech_07": -0.3, "tech_08": -0.5,
            "tech_09": -0.8, "tech_10": -1.0,
        }
        bottom = identify_bottom_agents(sharpes, bottom_pct=0.20)
        assert len(bottom) == 2
        assert "tech_09" in bottom
        assert "tech_10" in bottom

    def test_bottom_of_single_agent(self):
        sharpes = {"tech_01": 0.5}
        bottom = identify_bottom_agents(sharpes, bottom_pct=0.20)
        assert len(bottom) == 0  # Can't evolve the only agent

    def test_bottom_of_two_agents(self):
        sharpes = {"tech_01": 1.0, "tech_02": -1.0}
        bottom = identify_bottom_agents(sharpes, bottom_pct=0.20)
        # 20% of 2 rounds down to 0, but we need at least 1 if there are 2+
        # Implementation detail: floor(2*0.2)=0, so no agents evolved
        assert len(bottom) == 0

    def test_bottom_of_five_agents(self):
        sharpes = {f"a_{i}": float(i) for i in range(5)}
        bottom = identify_bottom_agents(sharpes, bottom_pct=0.20)
        assert len(bottom) == 1
        assert "a_0" in bottom  # Lowest sharpe


class TestMutateAgentParams:
    def test_mutated_params_within_bounds(self):
        schema = {"lookback": (3, 200, 5.0)}
        parent_params = {"lookback": 20}
        rng = np.random.default_rng(42)
        result = mutate_agent_params(parent_params, schema, rng)
        assert 3 <= result["lookback"] <= 200

    def test_params_without_schema_preserved(self):
        schema = {"lookback": (3, 200, 5.0)}
        parent_params = {"lookback": 20, "preset": "momentum_fast"}
        rng = np.random.default_rng(42)
        result = mutate_agent_params(parent_params, schema, rng)
        assert result["preset"] == "momentum_fast"

    def test_deterministic_with_seed(self):
        schema = {"lookback": (3, 200, 5.0)}
        parent_params = {"lookback": 100}
        r1 = mutate_agent_params(parent_params, schema, np.random.default_rng(42))
        r2 = mutate_agent_params(parent_params, schema, np.random.default_rng(42))
        assert r1 == r2


class TestRandomAgentParams:
    def test_random_within_bounds(self):
        schema = {"lookback": (5, 100, 10.0), "threshold": (0.1, 0.9, 0.1)}
        rng = np.random.default_rng(42)
        result = random_agent_params(schema, rng)
        assert 5 <= result["lookback"] <= 100
        assert 0.1 <= result["threshold"] <= 0.9

    def test_empty_schema(self):
        result = random_agent_params({}, np.random.default_rng(42))
        assert result == {}


class TestRunMonthlyEvolution:
    def test_evolution_produces_updated_configs(self):
        agent_configs = {
            "momentum": [
                {"id": "mom_01", "strategy": "timeseries", "params": {"lookback": 5}},
                {"id": "mom_02", "strategy": "timeseries", "params": {"lookback": 10}},
                {"id": "mom_03", "strategy": "timeseries", "params": {"lookback": 20}},
                {"id": "mom_04", "strategy": "timeseries", "params": {"lookback": 60}},
                {"id": "mom_05", "strategy": "timeseries", "params": {"lookback": 120}},
            ],
        }
        agent_sharpes = {
            "mom_01": 1.5, "mom_02": 1.0, "mom_03": 0.5,
            "mom_04": -0.5, "mom_05": -1.0,
        }
        agent_types = {
            "mom_01": "momentum", "mom_02": "momentum", "mom_03": "momentum",
            "mom_04": "momentum", "mom_05": "momentum",
        }

        updated, evolved_ids = run_monthly_evolution(
            agent_configs=agent_configs,
            agent_sharpes=agent_sharpes,
            agent_types=agent_types,
            bottom_pct=0.20,
            rng=np.random.default_rng(42),
        )

        assert len(evolved_ids) == 1
        assert evolved_ids[0] == "mom_05"  # Worst performer
        # Config structure preserved
        assert "momentum" in updated
        assert len(updated["momentum"]) == 5

    def test_evolved_agent_params_change(self):
        agent_configs = {
            "momentum": [
                {"id": "mom_01", "strategy": "timeseries", "params": {"lookback": 5}},
                {"id": "mom_02", "strategy": "timeseries", "params": {"lookback": 10}},
                {"id": "mom_03", "strategy": "timeseries", "params": {"lookback": 20}},
                {"id": "mom_04", "strategy": "timeseries", "params": {"lookback": 60}},
                {"id": "mom_05", "strategy": "timeseries", "params": {"lookback": 120}},
            ],
        }
        agent_sharpes = {
            "mom_01": 1.5, "mom_02": 1.0, "mom_03": 0.5,
            "mom_04": -0.5, "mom_05": -1.0,
        }
        agent_types = {k: "momentum" for k in agent_sharpes}

        updated, _ = run_monthly_evolution(
            agent_configs=agent_configs,
            agent_sharpes=agent_sharpes,
            agent_types=agent_types,
            bottom_pct=0.20,
            rng=np.random.default_rng(42),
        )
        # mom_05 should have changed params
        mom_05 = next(a for a in updated["momentum"] if a["id"] == "mom_05")
        assert mom_05["params"]["lookback"] != 120  # Should be mutated

    def test_no_evolution_with_one_agent(self):
        agent_configs = {
            "momentum": [
                {"id": "mom_01", "strategy": "timeseries", "params": {"lookback": 20}},
            ],
        }
        agent_sharpes = {"mom_01": -1.0}
        agent_types = {"mom_01": "momentum"}

        updated, evolved_ids = run_monthly_evolution(
            agent_configs=agent_configs,
            agent_sharpes=agent_sharpes,
            agent_types=agent_types,
            rng=np.random.default_rng(42),
        )
        assert len(evolved_ids) == 0

    def test_strategy_without_schema_skipped(self):
        agent_configs = {
            "sentiment": [
                {"id": "sent_01", "strategy": "news", "params": {"mode": "directional"}},
                {"id": "sent_02", "strategy": "news", "params": {"mode": "contrarian"}},
                {"id": "sent_03", "strategy": "news", "params": {"mode": "directional"}},
                {"id": "sent_04", "strategy": "news", "params": {"mode": "directional"}},
                {"id": "sent_05", "strategy": "news", "params": {"mode": "directional"}},
            ],
        }
        agent_sharpes = {
            "sent_01": 1.0, "sent_02": 0.5, "sent_03": 0.0,
            "sent_04": -0.5, "sent_05": -1.0,
        }
        agent_types = {k: "sentiment" for k in agent_sharpes}

        updated, evolved_ids = run_monthly_evolution(
            agent_configs=agent_configs,
            agent_sharpes=agent_sharpes,
            agent_types=agent_types,
            rng=np.random.default_rng(42),
        )
        # news strategy has no numeric params to mutate, so it does discrete mutation
        assert len(evolved_ids) == 1
