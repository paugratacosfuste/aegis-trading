"""Tests for RL constants."""

from aegis.rl.constants import (
    AGENT_TYPES,
    EXIT_OBS_DIM,
    MAX_KELLY_DIVERGENCE,
    MAX_POSITION_SIZE,
    MIN_POSITION_SIZE,
    NUM_EXIT_ACTIONS,
    NUM_WEIGHT_CONFIGS,
    POSITION_OBS_DIM,
    PROMOTION_MIN_DAYS,
    WEIGHT_CONTEXT_DIM,
)


def test_position_size_bounds():
    assert 0 < MIN_POSITION_SIZE < MAX_POSITION_SIZE <= 1.0
    assert MAX_POSITION_SIZE == 0.10


def test_kelly_divergence_positive():
    assert 0 < MAX_KELLY_DIVERGENCE < 1.0


def test_feature_dims_positive():
    assert WEIGHT_CONTEXT_DIM > 0
    assert POSITION_OBS_DIM > 0
    assert EXIT_OBS_DIM > 0


def test_action_spaces():
    assert NUM_WEIGHT_CONFIGS == 50
    assert NUM_EXIT_ACTIONS == 5


def test_promotion_criteria():
    assert PROMOTION_MIN_DAYS >= 90


def test_agent_types_match_weights():
    from aegis.ensemble.weights import BASE_TYPE_WEIGHTS

    for t in AGENT_TYPES:
        assert t in BASE_TYPE_WEIGHTS, f"Agent type '{t}' not in BASE_TYPE_WEIGHTS"
