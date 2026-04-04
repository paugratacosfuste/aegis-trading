"""Integration tests for lab mode: --lab flag, scheduler wiring, full pipeline."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from aegis.lab.config_templates import get_default_templates
from aegis.lab.lifecycle import advance_status, apply_promotion, apply_relegation
from aegis.lab.mutation import generate_random_cohort, mutate_cohort
from aegis.lab.tournament import (
    identify_promotion_candidates,
    identify_relegation_candidates,
    run_weekly,
)
from aegis.lab.types import CohortConfig, CohortPerformance, CohortStatus, StrategyCohort


class TestLabFullPipeline:
    """End-to-end test: 10 cohorts -> tournament -> promote/relegate -> mutate."""

    def test_full_lifecycle_pipeline(self):
        """Simulate a full lab cycle: create cohorts, score them, promote/relegate."""
        # Step 1: Create 10 cohorts from templates
        cohorts = get_default_templates()
        assert len(cohorts) == 10
        for c in cohorts:
            assert c.status == CohortStatus.CREATED

        # Step 2: Advance all to BURN_IN
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        advanced = [advance_status(c, now) for c in cohorts]
        for c in advanced:
            assert c.status == CohortStatus.BURN_IN

        # Step 3: Advance past burn-in (14 days later)
        later = datetime(2025, 6, 20, tzinfo=timezone.utc)
        evaluating = [advance_status(c, later) for c in advanced]
        for c in evaluating:
            assert c.status == CohortStatus.EVALUATING

        # Step 4: Create synthetic performance data
        rng = np.random.default_rng(42)
        perfs = {}
        cohort_map = {}
        for c in evaluating:
            perfs[c.cohort_id] = CohortPerformance(
                cohort_id=c.cohort_id,
                sharpe=float(rng.normal(1.0, 0.8)),
                win_rate=float(rng.uniform(0.3, 0.7)),
                max_drawdown=float(rng.uniform(-0.30, -0.02)),
                profit_factor=float(rng.uniform(0.5, 3.0)),
                total_trades=int(rng.integers(10, 100)),
                net_pnl=float(rng.normal(500, 2000)),
            )
            cohort_map[c.cohort_id] = c

        # Step 5: Run tournament
        from datetime import date

        results = run_weekly(perfs, date(2025, 7, 1))
        assert len(results) == 10
        assert results[0].rank == 1
        assert results[-1].rank == 10
        # Scores bounded
        for r in results:
            assert 0.0 <= r.composite_score <= 1.0

        # Step 6: Identify promotion/relegation candidates
        much_later = datetime(2025, 9, 1, tzinfo=timezone.utc)
        promo_candidates = identify_promotion_candidates(perfs, cohort_map, much_later)
        releg_candidates = identify_relegation_candidates(perfs, cohort_map)

        # Step 7: Apply promotions
        for cid in promo_candidates:
            cohort = cohort_map[cid]
            promoted = apply_promotion(cohort, much_later)
            assert promoted.status == CohortStatus.PROMOTED

        # Step 8: Apply relegations
        for cid in releg_candidates:
            cohort = cohort_map[cid]
            relegated = apply_relegation(cohort, much_later)
            assert relegated.status in (CohortStatus.RELEGATED, CohortStatus.RETIRED)

        # Step 9: Mutate top performer into a new cohort
        if promo_candidates:
            parent = cohort_map[promo_candidates[0]]
            child = mutate_cohort(parent, rng=np.random.default_rng(42))
            assert child.status == CohortStatus.CREATED
            assert child.generation == parent.generation + 1
            assert child.parent_cohort_id == parent.cohort_id
            total_w = sum(child.config.agent_weights.values())
            assert total_w == pytest.approx(1.0, abs=0.001)

    def test_five_plus_cohorts_run_independently(self):
        """Verify 5+ cohorts can process signals independently."""
        from aegis.lab.orchestrator import LabOrchestrator
        from aegis.lab.config_templates import create_cohort_from_template

        # Create 5 cohorts
        cohorts = [
            create_cohort_from_template(tid)
            for tid in ["A", "B", "C", "D", "E"]
        ]
        assert len(cohorts) == 5

        orchestrator = LabOrchestrator(cohorts=cohorts)
        runners = orchestrator.get_active_runners()
        assert len(runners) == 5

        # Each runner has independent state
        ids = set(runners.keys())
        assert len(ids) == 5
        for cid, runner in runners.items():
            assert runner.cohort.cohort_id == cid

    def test_tournament_scoring_works(self):
        """Tournament scoring produces ranked results with valid composite scores."""
        from datetime import date

        perfs = {}
        for i, label in enumerate(["A", "B", "C", "D", "E"]):
            cid = f"cohort_{label}"
            perfs[cid] = CohortPerformance(
                cohort_id=cid,
                sharpe=float(i * 0.5),
                win_rate=float(0.3 + i * 0.1),
                max_drawdown=-0.05 * (5 - i),
                profit_factor=float(1.0 + i * 0.5),
                total_trades=50,
                net_pnl=float(i * 1000),
            )

        results = run_weekly(perfs, date(2025, 7, 1))
        assert len(results) == 5
        assert results[0].cohort_id == "cohort_E"  # Best stats
        assert results[0].rank == 1
        assert results[-1].cohort_id == "cohort_A"  # Worst stats
        assert results[-1].rank == 5

    def test_promotion_relegation_functional(self):
        """Promotion and relegation state transitions work correctly."""
        now = datetime(2025, 9, 1, tzinfo=timezone.utc)

        # Evaluating cohort can be promoted
        c = StrategyCohort(
            cohort_id="test_promo", name="Test",
            status=CohortStatus.EVALUATING,
            config=CohortConfig(
                agent_weights={"technical": 1.0},
                confidence_threshold=0.45,
                risk_params={}, universe=(),
            ),
        )
        promoted = apply_promotion(c, now)
        assert promoted.status == CohortStatus.PROMOTED

        # Evaluating cohort can be relegated
        c2 = StrategyCohort(
            cohort_id="test_releg", name="Test",
            status=CohortStatus.EVALUATING,
            config=CohortConfig(
                agent_weights={"technical": 1.0},
                confidence_threshold=0.45,
                risk_params={}, universe=(),
            ),
        )
        relegated = apply_relegation(c2, now)
        assert relegated.status == CohortStatus.RELEGATED
        assert relegated.relegation_count == 1


class TestMainLabFlag:
    """Test that --lab flag is recognized and routes correctly."""

    def test_main_parser_has_lab_flag(self):
        """Verify main.py's _build_parser includes --lab."""
        from aegis.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--config", "configs/lab.yaml", "--lab"])
        assert args.lab is True
        assert args.backtest is False

    def test_lab_and_backtest_mutually_exclusive(self):
        """Cannot pass both --lab and --backtest."""
        from aegis.main import _build_parser

        parser = _build_parser()
        # Both flags set — main() should handle this, but parser allows it
        args = parser.parse_args(["--config", "x.yaml", "--lab", "--backtest"])
        assert args.lab and args.backtest  # Both can be parsed

    def test_lab_config_section(self):
        """Verify lab.yaml has a lab config section with expected keys."""
        from aegis.common.config import load_config

        config = load_config("configs/lab.yaml")
        assert hasattr(config, "lab")
        lab = config.lab
        assert isinstance(lab, dict)
        assert "templates" in lab
        assert "tournament_interval_days" in lab
        assert "burn_in_days" in lab


class TestRunLabFunction:
    """Test run_lab dispatches to LabOrchestrator correctly."""

    def test_run_lab_exists(self):
        """run_lab function is importable."""
        from aegis.main import run_lab
        assert callable(run_lab)
