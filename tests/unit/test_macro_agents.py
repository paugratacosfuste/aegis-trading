"""Tests for macro agents (M3)."""

from datetime import datetime, timezone

import pytest

from aegis.agents.macro.providers import HistoricalMacroProvider, NullMacroProvider
from aegis.common.types import MacroDataPoint


def _macro_snap(**overrides) -> MacroDataPoint:
    defaults = dict(
        timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        yield_10y=4.25, yield_2y=4.05, yield_spread=0.20,
        vix=18.5, vix_regime="normal", dxy=104.5,
        fed_rate=5.25, cpi_latest=3.2,
    )
    defaults.update(overrides)
    return MacroDataPoint(**defaults)


def _provider(snap=None, yield_curve=None, vix_history=None):
    snaps = [snap] if snap else [_macro_snap()]
    return HistoricalMacroProvider(
        snapshots=snaps,
        yield_curve=yield_curve or {"2Y": 4.05, "10Y": 4.25, "30Y": 4.50},
        vix_history=vix_history or [18.0, 19.0, 18.5],
    )


class TestBaseMacroAgent:
    def test_agent_type(self):
        from aegis.agents.macro.yield_curve import YieldCurveFedAgent
        agent = YieldCurveFedAgent("macro_01", {})
        assert agent.agent_type == "macro"


class TestYieldCurveFedAgent:
    def test_no_data_neutral(self, sample_candles_uptrend):
        from aegis.agents.macro.yield_curve import YieldCurveFedAgent
        agent = YieldCurveFedAgent("macro_01", {})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_inverted_curve_recession(self, sample_candles_uptrend):
        from aegis.agents.macro.yield_curve import YieldCurveFedAgent
        snap = _macro_snap(yield_10y=3.5, yield_2y=4.5, yield_spread=-1.0)
        agent = YieldCurveFedAgent("macro_01", {}, _provider(snap))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata["regime"] == "recession_risk"

    def test_steep_curve_early_cycle(self, sample_candles_uptrend):
        from aegis.agents.macro.yield_curve import YieldCurveFedAgent
        snap = _macro_snap(yield_10y=5.0, yield_2y=2.5, yield_spread=2.5)
        agent = YieldCurveFedAgent("macro_01", {}, _provider(snap))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata["regime"] == "early_cycle"

    def test_flat_curve_late_cycle(self, sample_candles_uptrend):
        from aegis.agents.macro.yield_curve import YieldCurveFedAgent
        snap = _macro_snap(yield_10y=4.3, yield_2y=4.0, yield_spread=0.3)
        agent = YieldCurveFedAgent("macro_01", {}, _provider(snap))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata["regime"] == "late_cycle"

    def test_normal_curve_mid_cycle(self, sample_candles_uptrend):
        from aegis.agents.macro.yield_curve import YieldCurveFedAgent
        snap = _macro_snap(yield_10y=4.5, yield_2y=3.5, yield_spread=1.0)
        agent = YieldCurveFedAgent("macro_01", {}, _provider(snap))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata["regime"] == "mid_cycle"


class TestRiskRegimeAgent:
    def test_no_data_neutral(self, sample_candles_uptrend):
        from aegis.agents.macro.risk_regime import RiskRegimeAgent
        agent = RiskRegimeAgent("macro_02", {})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_low_vix_risk_on(self, sample_candles_uptrend):
        from aegis.agents.macro.risk_regime import RiskRegimeAgent
        snap = _macro_snap(vix=12.0, vix_regime="low", dxy=98.0)
        agent = RiskRegimeAgent("macro_02", {}, _provider(snap))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata["regime"] == "risk_on"

    def test_extreme_vix_risk_off(self, sample_candles_uptrend):
        from aegis.agents.macro.risk_regime import RiskRegimeAgent
        snap = _macro_snap(vix=40.0, vix_regime="extreme", dxy=110.0)
        agent = RiskRegimeAgent("macro_02", {}, _provider(snap))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata["regime"] == "risk_off"


class TestEconomicCycleAgent:
    def test_no_data_neutral(self, sample_candles_uptrend):
        from aegis.agents.macro.economic_cycle import EconomicCycleAgent
        agent = EconomicCycleAgent("macro_03", {})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_expansion_positive_indicators(self, sample_candles_uptrend):
        from aegis.agents.macro.economic_cycle import EconomicCycleAgent
        snap = _macro_snap(yield_spread=1.5, vix=15.0, vix_regime="low")
        agent = EconomicCycleAgent("macro_03", {}, _provider(snap))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata["regime"] in ("expansion", "mid_cycle")

    def test_contraction_negative_indicators(self, sample_candles_uptrend):
        from aegis.agents.macro.economic_cycle import EconomicCycleAgent
        snap = _macro_snap(yield_spread=-0.5, vix=35.0, vix_regime="high")
        agent = EconomicCycleAgent("macro_03", {}, _provider(snap))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata["regime"] in ("contraction", "recession_risk")


class TestInflationRegimeAgent:
    def test_no_data_neutral(self, sample_candles_uptrend):
        from aegis.agents.macro.inflation import InflationRegimeAgent
        agent = InflationRegimeAgent("macro_04", {})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_high_cpi_high_inflation(self, sample_candles_uptrend):
        from aegis.agents.macro.inflation import InflationRegimeAgent
        snap = _macro_snap(cpi_latest=6.0, fed_rate=5.0, yield_10y=5.5)
        agent = InflationRegimeAgent("macro_04", {}, _provider(snap))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata["regime"] in ("high", "very_high")

    def test_low_cpi_low_inflation(self, sample_candles_uptrend):
        from aegis.agents.macro.inflation import InflationRegimeAgent
        snap = _macro_snap(cpi_latest=1.0, fed_rate=2.0, yield_10y=3.0)
        agent = InflationRegimeAgent("macro_04", {}, _provider(snap))
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.metadata["regime"] in ("low", "deflationary")


class TestHmmRegimeAgent:
    def test_no_model_neutral(self, sample_candles_uptrend):
        from aegis.agents.macro.hmm_regime import HmmRegimeAgent
        agent = HmmRegimeAgent("macro_05", {"model_path": "nonexistent.pkl"})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0
        assert signal.metadata.get("regime", "unknown") == "unknown"

    def test_agent_type(self):
        from aegis.agents.macro.hmm_regime import HmmRegimeAgent
        agent = HmmRegimeAgent("macro_05", {})
        assert agent.agent_type == "macro"


class TestRegistration:
    def test_all_macro_registered(self):
        from aegis.agents.registry import get_agent_class
        from aegis.agents.macro.yield_curve import YieldCurveFedAgent
        from aegis.agents.macro.risk_regime import RiskRegimeAgent
        from aegis.agents.macro.economic_cycle import EconomicCycleAgent
        from aegis.agents.macro.inflation import InflationRegimeAgent
        from aegis.agents.macro.hmm_regime import HmmRegimeAgent
        assert get_agent_class("macro", "yield_curve_fed") is YieldCurveFedAgent
        assert get_agent_class("macro", "risk_regime") is RiskRegimeAgent
        assert get_agent_class("macro", "economic_cycle") is EconomicCycleAgent
        assert get_agent_class("macro", "inflation_regime") is InflationRegimeAgent
        assert get_agent_class("macro", "hmm_regime") is HmmRegimeAgent
