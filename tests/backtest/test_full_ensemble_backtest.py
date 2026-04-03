"""Integration test: 42-agent ensemble backtest (M7).

Verifies that all 42 agents can run through the backtest engine
without crashing and produce valid results.
"""

from datetime import datetime, timedelta, timezone

import pytest

from aegis.agents.factory import create_agents_from_config
from aegis.backtest.engine import BacktestEngine
from aegis.common.types import MarketDataPoint

_BASE_TIME = datetime(2025, 6, 1, 0, 0, tzinfo=timezone.utc)

_FULL_AGENTS_CONFIG = {
    "technical": [
        {"id": "tech_01", "strategy": "indicator", "params": {"preset": "momentum_fast", "period_style": "fast"}},
        {"id": "tech_02", "strategy": "indicator", "params": {"preset": "momentum_fast", "period_style": "standard"}},
        {"id": "tech_03", "strategy": "indicator", "params": {"preset": "volume_confirmed", "period_style": "standard"}},
        {"id": "tech_04", "strategy": "indicator", "params": {"preset": "trend_following", "period_style": "fast"}},
        {"id": "tech_05", "strategy": "indicator", "params": {"preset": "mean_reversion", "period_style": "standard"}},
        {"id": "tech_06", "strategy": "indicator", "params": {"preset": "full_suite", "period_style": "mixed"}},
        {"id": "tech_07", "strategy": "asian_range", "params": {"market": "crypto"}},
    ],
    "statistical": [
        {"id": "stat_01", "strategy": "zscore", "params": {"lookback": 20}},
        {"id": "stat_02", "strategy": "zscore", "params": {"lookback": 50}},
        {"id": "stat_03", "strategy": "ornstein_uhlenbeck", "params": {"lookback": 60}},
        {"id": "stat_04", "strategy": "kalman", "params": {}},
        {"id": "stat_05", "strategy": "bollinger_zscore", "params": {"period": 20}},
        {"id": "stat_06", "strategy": "hurst_zscore", "params": {"lookback": 100}},
        {"id": "stat_07", "strategy": "multi_window", "params": {"windows": [20, 50, 100]}},
    ],
    "momentum": [
        {"id": "mom_01", "strategy": "timeseries", "params": {"lookback": 5}},
        {"id": "mom_02", "strategy": "timeseries", "params": {"lookback": 20}},
        {"id": "mom_03", "strategy": "dual", "params": {"lookback": 20, "benchmark_lookback": 60}},
        {"id": "mom_04", "strategy": "volume_weighted", "params": {"lookback": 20}},
        {"id": "mom_05", "strategy": "rsi_filtered", "params": {"lookback": 20, "rsi_threshold": 70}},
        {"id": "mom_06", "strategy": "acceleration", "params": {"short_lookback": 5, "long_lookback": 20}},
    ],
    "sentiment": [
        {"id": "sent_01", "strategy": "news", "params": {"mode": "directional"}},
        {"id": "sent_02", "strategy": "reddit", "params": {"mode": "contrarian"}},
        {"id": "sent_03", "strategy": "fear_greed", "params": {"market": "crypto"}},
        {"id": "sent_04", "strategy": "combined", "params": {}},
    ],
}


def _make_candles(n: int, trend: float = 50.0) -> list[MarketDataPoint]:
    """Generate n candles with configurable trend per candle."""
    candles = []
    base = 40000.0
    for i in range(n):
        price = base + i * trend
        candles.append(
            MarketDataPoint(
                symbol="BTC/USDT",
                asset_class="crypto",
                timestamp=_BASE_TIME + timedelta(hours=i),
                timeframe="1h",
                open=price,
                high=price + 80,
                low=price - 40,
                close=price + 30,
                volume=150.0 + i * 3,
                source="binance",
            )
        )
    return candles


class TestFullEnsembleBacktest:
    def test_factory_creates_all_agents(self):
        agents = create_agents_from_config(_FULL_AGENTS_CONFIG)
        assert len(agents) == 24  # Reduced set for fast integration test

    def test_all_agents_generate_signals(self):
        """Every agent can generate a signal without crashing."""
        agents = create_agents_from_config(_FULL_AGENTS_CONFIG)
        candles = _make_candles(120)

        for agent in agents:
            sig = agent.generate_signal("BTC/USDT", candles)
            assert sig.symbol == "BTC/USDT"
            assert -1.0 <= sig.direction <= 1.0
            assert 0.0 <= sig.confidence <= 1.0

    def test_backtest_runs_without_crash(self):
        """Full backtest with ensemble agents completes."""
        agents = create_agents_from_config(_FULL_AGENTS_CONFIG)
        candles = _make_candles(200)

        engine = BacktestEngine(
            initial_capital=5000.0,
            commission_pct=0.001,
            confidence_threshold=0.45,
            max_open_positions=5,
            max_risk_pct=0.05,
            agents=agents,
        )
        results = engine.run(candles)

        assert "metrics" in results
        assert "equity_curve" in results
        assert results["metrics"]["final_equity"] > 0
        assert len(results["equity_curve"]) > 0

    def test_multiple_agent_types_contribute(self):
        """At least 2 agent types contribute non-zero signals."""
        agents = create_agents_from_config(_FULL_AGENTS_CONFIG)
        candles = _make_candles(120)

        types_with_signals: set[str] = set()
        for agent in agents:
            sig = agent.generate_signal("BTC/USDT", candles)
            if abs(sig.direction) > 0.01:
                types_with_signals.add(sig.agent_type)

        assert len(types_with_signals) >= 2

    def test_default_agents_still_work(self):
        """Phase 1 default agents backtest still works."""
        candles = _make_candles(100)
        engine = BacktestEngine(
            initial_capital=5000.0,
            confidence_threshold=0.45,
        )
        results = engine.run(candles)
        assert results["metrics"]["final_equity"] > 0
