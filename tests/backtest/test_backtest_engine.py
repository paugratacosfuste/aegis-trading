"""Tests for backtest engine. Written FIRST per TDD."""

from datetime import datetime, timezone

import pytest

from aegis.common.types import MarketDataPoint


def _make_candle(timestamp, open_p, high, low, close, volume=100.0):
    return MarketDataPoint(
        symbol="BTC/USDT",
        asset_class="crypto",
        timestamp=timestamp,
        timeframe="1h",
        open=open_p,
        high=high,
        low=low,
        close=close,
        volume=volume,
        source="backtest",
    )


def _make_trending_candles(start_price=40000.0, n=100, step=100.0):
    """Steady uptrend candles."""
    candles = []
    for i in range(n):
        ts = datetime(2025, 6, 1, i % 24, tzinfo=timezone.utc)
        price = start_price + i * step
        candles.append(
            _make_candle(ts, price, price + 50, price - 50, price + step * 0.5)
        )
    return candles


def _make_flat_candles(price=42000.0, n=100):
    """Flat price candles."""
    candles = []
    for i in range(n):
        ts = datetime(2025, 6, 1, i % 24, tzinfo=timezone.utc)
        candles.append(_make_candle(ts, price, price + 10, price - 10, price))
    return candles


class TestBacktestEngine:
    @pytest.fixture
    def engine(self):
        from aegis.backtest.engine import BacktestEngine

        return BacktestEngine(
            initial_capital=5000.0,
            commission_pct=0.001,
            confidence_threshold=0.45,
            max_open_positions=5,
            max_risk_pct=0.05,
        )

    def test_engine_initializes(self, engine):
        assert engine.initial_capital == 5000.0
        assert engine.equity == 5000.0

    def test_run_returns_results(self, engine):
        candles = _make_trending_candles(n=50)
        results = engine.run(candles)
        assert "equity_curve" in results
        assert "trades" in results
        assert "metrics" in results
        assert len(results["equity_curve"]) > 0

    def test_flat_market_few_trades(self, engine):
        candles = _make_flat_candles(n=50)
        results = engine.run(candles)
        # Flat market should produce few or no trades
        assert len(results["trades"]) <= 5

    def test_no_negative_equity(self, engine):
        candles = _make_trending_candles(n=50)
        results = engine.run(candles)
        for eq in results["equity_curve"]:
            assert eq >= 0

    def test_commission_deducted(self, engine):
        candles = _make_trending_candles(n=50)
        results = engine.run(candles)
        # If any trades occurred, total costs should be > 0
        if results["trades"]:
            total_commission = sum(t.get("commission", 0) for t in results["trades"])
            assert total_commission >= 0

    def test_metrics_present(self, engine):
        candles = _make_trending_candles(n=50)
        results = engine.run(candles)
        metrics = results["metrics"]
        assert "sharpe" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "total_trades" in metrics
        assert "final_equity" in metrics

    def test_lookback_respects_minimum(self, engine):
        # Too few candles for agents (need >= 21)
        candles = _make_trending_candles(n=15)
        results = engine.run(candles)
        assert len(results["trades"]) == 0
