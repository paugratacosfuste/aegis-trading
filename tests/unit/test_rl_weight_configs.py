"""Tests for weight allocator configs: all 50 valid."""

import pytest

from aegis.rl.common.safety import validate_weight_config
from aegis.rl.constants import AGENT_TYPES, NUM_WEIGHT_CONFIGS
from aegis.rl.weight_allocator.configs import WEIGHT_CONFIG_MAP, WEIGHT_CONFIGS


def test_exactly_50_configs():
    assert len(WEIGHT_CONFIGS) == NUM_WEIGHT_CONFIGS


def test_all_configs_sum_to_one():
    for wc in WEIGHT_CONFIGS:
        total = sum(wc.weights.values())
        assert abs(total - 1.0) < 0.001, f"Config {wc.config_id} ({wc.name}): sum={total}"


def test_all_weights_non_negative():
    for wc in WEIGHT_CONFIGS:
        for agent_type, weight in wc.weights.items():
            assert weight >= 0, f"Config {wc.config_id}: {agent_type}={weight}"


def test_all_7_types_present():
    for wc in WEIGHT_CONFIGS:
        for t in AGENT_TYPES:
            assert t in wc.weights, f"Config {wc.config_id} missing {t}"


def test_all_pass_safety_validation():
    for wc in WEIGHT_CONFIGS:
        assert validate_weight_config(wc), f"Config {wc.config_id} ({wc.name}) failed validation"


def test_unique_ids():
    ids = [wc.config_id for wc in WEIGHT_CONFIGS]
    assert len(ids) == len(set(ids))


def test_unique_names():
    names = [wc.name for wc in WEIGHT_CONFIGS]
    assert len(names) == len(set(names))


def test_config_map_complete():
    assert len(WEIGHT_CONFIG_MAP) == NUM_WEIGHT_CONFIGS
    for i in range(NUM_WEIGHT_CONFIGS):
        assert i in WEIGHT_CONFIG_MAP


def test_baseline_matches_ensemble_weights():
    from aegis.ensemble.weights import BASE_TYPE_WEIGHTS

    baseline = WEIGHT_CONFIGS[0]
    assert baseline.name == "baseline"
    for t in AGENT_TYPES:
        assert abs(baseline.weights[t] - BASE_TYPE_WEIGHTS[t]) < 0.01
