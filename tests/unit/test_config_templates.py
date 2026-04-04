"""Tests for cohort configuration templates."""

import pytest

from aegis.lab.config_templates import (
    TEMPLATES,
    create_cohort_from_template,
    get_default_templates,
)
from aegis.lab.types import CohortStatus


class TestConfigTemplates:
    def test_ten_templates_defined(self):
        assert len(TEMPLATES) == 10
        expected_ids = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}
        assert set(TEMPLATES.keys()) == expected_ids

    def test_all_weights_sum_to_one(self):
        for tid, tmpl in TEMPLATES.items():
            total = sum(tmpl["config"].agent_weights.values())
            assert total == pytest.approx(1.0, abs=0.01), (
                f"Template {tid} weights sum to {total}"
            )

    def test_unique_names(self):
        names = [t["name"] for t in TEMPLATES.values()]
        assert len(names) == len(set(names))

    def test_create_from_template(self):
        cohort = create_cohort_from_template("A")
        assert cohort.cohort_id == "cohort_A"
        assert cohort.name == "Baseline Production"
        assert cohort.status == CohortStatus.CREATED
        assert cohort.generation == 0
        assert cohort.config.confidence_threshold == 0.40

    def test_get_default_templates(self):
        cohorts = get_default_templates()
        assert len(cohorts) == 10
        ids = {c.cohort_id for c in cohorts}
        assert "cohort_A" in ids
        assert "cohort_J" in ids

    def test_contrarian_has_invert(self):
        cohort = create_cohort_from_template("E")
        assert cohort.config.invert_sentiment is True

    def test_macro_first_has_macro_sizing(self):
        cohort = create_cohort_from_template("F")
        assert cohort.config.macro_position_sizing is True

    def test_conservative_high_threshold(self):
        cohort = create_cohort_from_template("I")
        assert cohort.config.confidence_threshold == 0.60
        assert cohort.config.risk_params["max_risk_per_trade"] == 0.01

    def test_aggressive_low_threshold(self):
        cohort = create_cohort_from_template("J")
        assert cohort.config.confidence_threshold == 0.35
        assert cohort.config.risk_params["max_risk_per_trade"] == 0.08

    def test_trend_following_weights(self):
        cohort = create_cohort_from_template("B")
        assert cohort.config.agent_weights["technical"] == 0.35
        assert cohort.config.agent_weights["momentum"] == 0.25

    def test_mean_reversion_weights(self):
        cohort = create_cohort_from_template("C")
        assert cohort.config.agent_weights["statistical"] == 0.40

    def test_all_templates_have_universe(self):
        for tid in TEMPLATES:
            cohort = create_cohort_from_template(tid)
            assert len(cohort.config.universe) > 0

    def test_generation_param(self):
        cohort = create_cohort_from_template("A", generation=3)
        assert cohort.generation == 3
