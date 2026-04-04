"""Tests for lab orchestrator."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from aegis.common.types import AgentSignal, MarketDataPoint
from aegis.lab.orchestrator import LabOrchestrator
from aegis.lab.types import CohortConfig, CohortStatus, StrategyCohort


def _cohort(cohort_id: str, threshold: float = 0.30) -> StrategyCohort:
    return StrategyCohort(
        cohort_id=cohort_id,
        name=f"Test {cohort_id}",
        status=CohortStatus.EVALUATING,
        config=CohortConfig(
            agent_weights={"technical": 0.30, "statistical": 0.25,
                           "momentum": 0.25, "sentiment": 0.20},
            confidence_threshold=threshold,
            risk_params={"max_risk_per_trade": 0.05},
            universe=(),
        ),
    )


def _make_candles(n=22, base_price=50000.0) -> list[MarketDataPoint]:
    candles = []
    for i in range(n):
        price = base_price + i * 100  # Uptrend
        candles.append(MarketDataPoint(
            symbol="BTC/USDT",
            timestamp=datetime(2025, 6, 1, i, 0, tzinfo=timezone.utc),
            open=price,
            high=price + 50,
            low=price - 50,
            close=price + 25,
            volume=1000.0,
            asset_class="crypto",
            timeframe="1h",
            source="test",
        ))
    return candles


class TestLabOrchestrator:
    def test_init_with_cohorts(self):
        cohorts = [_cohort("A"), _cohort("B")]
        orch = LabOrchestrator(cohorts=cohorts)
        assert len(orch.get_active_runners()) == 2

    def test_add_cohort(self):
        orch = LabOrchestrator(cohorts=[])
        assert len(orch.get_active_runners()) == 0
        orch.add_cohort(_cohort("A"))
        assert len(orch.get_active_runners()) == 1
        assert orch.get_runner("A") is not None

    def test_remove_cohort(self):
        orch = LabOrchestrator(cohorts=[_cohort("A"), _cohort("B")])
        orch.remove_cohort("A")
        assert len(orch.get_active_runners()) == 1
        assert orch.get_runner("A") is None

    def test_tick_dispatches_to_all_cohorts(self):
        cohorts = [_cohort("A", threshold=0.20), _cohort("B", threshold=0.20)]
        orch = LabOrchestrator(cohorts=cohorts)

        candles = _make_candles()
        candles_by_symbol = {"BTC/USDT": candles}
        prices = {"BTC/USDT": 53000.0}

        results = orch.tick(candles_by_symbol, prices)
        # Both cohorts processed — results may or may not have trades
        # depending on agent signals, but no crash
        assert isinstance(results, dict)

    def test_tick_empty_candles(self):
        orch = LabOrchestrator(cohorts=[_cohort("A")])
        results = orch.tick({}, {})
        assert results == {}

    def test_agents_created_from_defaults(self):
        orch = LabOrchestrator(cohorts=[_cohort("A")])
        assert len(orch.agents) > 0  # Default agents created

    def test_get_runner(self):
        orch = LabOrchestrator(cohorts=[_cohort("A")])
        runner = orch.get_runner("A")
        assert runner is not None
        assert runner.cohort.cohort_id == "A"

    def test_multiple_symbols(self):
        orch = LabOrchestrator(cohorts=[_cohort("A")])
        candles_btc = _make_candles(base_price=50000.0)
        candles_eth = []
        for c in _make_candles(base_price=3000.0):
            candles_eth.append(MarketDataPoint(
                symbol="ETH/USDT", timestamp=c.timestamp,
                open=c.open / 15, high=c.high / 15, low=c.low / 15,
                close=c.close / 15, volume=c.volume,
                asset_class="crypto", timeframe="1h", source="test",
            ))
        results = orch.tick(
            {"BTC/USDT": candles_btc, "ETH/USDT": candles_eth},
            {"BTC/USDT": 53000.0, "ETH/USDT": 3500.0},
        )
        assert isinstance(results, dict)
