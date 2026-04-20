"""Tests for macro data loader and BacktestMacroProvider."""

from datetime import datetime, timezone

import pytest

from aegis.common.types import MacroDataPoint


class TestClassifyVix:
    def test_low_vix(self):
        from aegis.data.macro_data_loader import _classify_vix
        assert _classify_vix(12.0) == "low"

    def test_normal_vix(self):
        from aegis.data.macro_data_loader import _classify_vix
        assert _classify_vix(20.0) == "normal"

    def test_high_vix(self):
        from aegis.data.macro_data_loader import _classify_vix
        assert _classify_vix(30.0) == "high"

    def test_extreme_vix(self):
        from aegis.data.macro_data_loader import _classify_vix
        assert _classify_vix(40.0) == "extreme"


class TestCpiLookup:
    def test_known_month(self):
        from aegis.data.macro_data_loader import _get_cpi_for_date
        dt = datetime(2025, 3, 15, tzinfo=timezone.utc)
        assert _get_cpi_for_date(dt) == 2.4

    def test_fallback_to_latest(self):
        from aegis.data.macro_data_loader import _get_cpi_for_date
        dt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        # Should return the latest available (2025-03)
        assert _get_cpi_for_date(dt) == 2.4


class TestFedRateLookup:
    def test_after_cut(self):
        from aegis.data.macro_data_loader import _get_fed_rate_for_date
        dt = datetime(2025, 2, 1, tzinfo=timezone.utc)
        assert _get_fed_rate_for_date(dt) == 4.50

    def test_before_any_change(self):
        from aegis.data.macro_data_loader import _get_fed_rate_for_date
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Default fallback
        assert _get_fed_rate_for_date(dt) == 4.50


class TestBacktestMacroProvider:
    def _make_snapshot(self, day: int, vix: float = 18.0) -> MacroDataPoint:
        return MacroDataPoint(
            timestamp=datetime(2025, 4, day, tzinfo=timezone.utc),
            yield_10y=4.25,
            yield_2y=4.00,
            yield_spread=0.25,
            vix=vix,
            vix_regime="normal",
            dxy=103.0,
            fed_rate=4.50,
            cpi_latest=2.8,
        )

    def test_advance_to_returns_correct_snapshot(self):
        from aegis.agents.macro.providers import BacktestMacroProvider

        snaps = [self._make_snapshot(1, 15.0), self._make_snapshot(2, 20.0), self._make_snapshot(3, 30.0)]
        provider = BacktestMacroProvider(snaps)

        # At start
        snap = provider.get_macro_snapshot()
        assert snap.vix == 15.0

        # Advance to day 2
        provider.advance_to(datetime(2025, 4, 2, tzinfo=timezone.utc))
        snap = provider.get_macro_snapshot()
        assert snap.vix == 20.0

        # Advance to day 3
        provider.advance_to(datetime(2025, 4, 3, tzinfo=timezone.utc))
        snap = provider.get_macro_snapshot()
        assert snap.vix == 30.0

    def test_advance_no_lookahead(self):
        """Provider should never return data from the future."""
        from aegis.agents.macro.providers import BacktestMacroProvider

        snaps = [self._make_snapshot(5, 15.0), self._make_snapshot(10, 25.0)]
        provider = BacktestMacroProvider(snaps)

        # Advance to day 7 (between snapshots)
        provider.advance_to(datetime(2025, 4, 7, tzinfo=timezone.utc))
        snap = provider.get_macro_snapshot()
        assert snap.vix == 15.0  # day 5, not day 10

    def test_empty_provider(self):
        from aegis.agents.macro.providers import BacktestMacroProvider

        provider = BacktestMacroProvider([])
        assert provider.get_macro_snapshot() is None

    def test_get_yield_curve(self):
        from aegis.agents.macro.providers import BacktestMacroProvider

        provider = BacktestMacroProvider([self._make_snapshot(1)])
        curve = provider.get_yield_curve()
        assert curve == {"10Y": 4.25, "2Y": 4.00}

    def test_get_vix_history(self):
        from aegis.agents.macro.providers import BacktestMacroProvider

        snaps = [self._make_snapshot(d, 10.0 + d) for d in range(1, 6)]
        provider = BacktestMacroProvider(snaps)
        provider.advance_to(datetime(2025, 4, 5, tzinfo=timezone.utc))
        history = provider.get_vix_history(lookback_days=3)
        assert len(history) == 3
        assert history[-1] == 15.0  # day 5


class TestMacroAgentsWithProvider:
    """Test that macro agents produce real signals when given data."""

    def _make_provider(self):
        from aegis.agents.macro.providers import BacktestMacroProvider
        snap = MacroDataPoint(
            timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
            yield_10y=4.50,
            yield_2y=4.00,
            yield_spread=0.50,
            vix=28.0,
            vix_regime="high",
            dxy=106.0,
            fed_rate=4.50,
            cpi_latest=3.5,
        )
        return BacktestMacroProvider([snap])

    def _make_candle(self):
        from aegis.common.types import MarketDataPoint
        return MarketDataPoint(
            symbol="AAPL", asset_class="equity",
            timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
            timeframe="1d", open=200.0, high=202.0, low=199.0,
            close=201.0, volume=1000000.0, source="test",
        )

    def test_yield_curve_agent_produces_regime(self):
        from aegis.agents.macro.yield_curve import YieldCurveFedAgent
        provider = self._make_provider()
        agent = YieldCurveFedAgent("test_yc", {}, provider=provider)
        sig = agent.generate_signal("AAPL", [self._make_candle()])
        assert sig.metadata.get("regime") is not None
        assert sig.metadata["regime"] != "unknown"

    def test_risk_regime_agent_produces_regime(self):
        from aegis.agents.macro.risk_regime import RiskRegimeAgent
        provider = self._make_provider()
        agent = RiskRegimeAgent("test_rr", {}, provider=provider)
        sig = agent.generate_signal("AAPL", [self._make_candle()])
        assert sig.metadata.get("regime") in ("risk_on", "risk_off", "neutral")

    def test_inflation_agent_produces_regime(self):
        from aegis.agents.macro.inflation import InflationRegimeAgent
        provider = self._make_provider()
        agent = InflationRegimeAgent("test_inf", {}, provider=provider)
        sig = agent.generate_signal("AAPL", [self._make_candle()])
        assert sig.metadata["regime"] in ("deflationary", "low", "moderate", "high", "very_high")

    def test_economic_cycle_agent_produces_regime(self):
        from aegis.agents.macro.economic_cycle import EconomicCycleAgent
        provider = self._make_provider()
        agent = EconomicCycleAgent("test_ec", {}, provider=provider)
        sig = agent.generate_signal("AAPL", [self._make_candle()])
        assert sig.metadata["regime"] in ("expansion", "mid_cycle", "contraction", "recession_risk")

    def test_null_provider_returns_neutral(self):
        """Without real data, macro agents return neutral."""
        from aegis.agents.macro.yield_curve import YieldCurveFedAgent
        agent = YieldCurveFedAgent("test_yc", {})
        sig = agent.generate_signal("AAPL", [self._make_candle()])
        assert sig.direction == 0.0
        assert sig.confidence == 0.0
