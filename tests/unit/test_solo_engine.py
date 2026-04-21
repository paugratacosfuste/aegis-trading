"""Unit tests for SoloBacktestEngine — a single-agent backtest harness.

Phase 1.2 of the Aegis 2.0 plan: validate edge of each agent in isolation,
no ensemble voting. Fixed 10% per signal, 2x ATR stop, 4x ATR target.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from aegis.backtest.solo_engine import SoloBacktestEngine, SoloResult
from aegis.common.types import AgentSignal, MarketDataPoint


def _candle(ts: datetime, close: float, *, high: float | None = None,
            low: float | None = None, volume: float = 1000.0) -> MarketDataPoint:
    return MarketDataPoint(
        symbol="BTC/USDT",
        asset_class="crypto",
        timestamp=ts,
        timeframe="1d",
        open=close,
        high=high if high is not None else close * 1.01,
        low=low if low is not None else close * 0.99,
        close=close,
        volume=volume,
        source="binance",
    )


def _make_candles(n: int, start_price: float = 100.0,
                  drift: float = 0.0) -> list[MarketDataPoint]:
    candles = []
    ts = datetime(2022, 1, 1, tzinfo=timezone.utc)
    price = start_price
    for i in range(n):
        price *= 1.0 + drift
        candles.append(_candle(ts + timedelta(days=i), price))
    return candles


class _AlwaysLongAgent:
    """Test double: emits direction=+1.0, confidence=0.9 forever."""

    agent_id = "always_long"
    agent_type = "technical"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        last = candles[-1]
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=symbol,
            timestamp=last.timestamp,
            direction=1.0,
            confidence=0.9,
            timeframe="1d",
            expected_holding_period="days",
            entry_price=last.close,
            stop_loss=None,
            take_profit=None,
            reasoning={},
            features_used={},
            metadata={},
        )


class _NeutralAgent:
    agent_id = "neutral"
    agent_type = "technical"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        last = candles[-1]
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=symbol,
            timestamp=last.timestamp,
            direction=0.0,
            confidence=0.0,
            timeframe="1d",
            expected_holding_period="days",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasoning={},
            features_used={},
            metadata={},
        )


class _LowConfidenceLongAgent:
    agent_id = "low_conf"
    agent_type = "technical"

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        last = candles[-1]
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=symbol,
            timestamp=last.timestamp,
            direction=1.0,
            confidence=0.2,
            timeframe="1d",
            expected_holding_period="days",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasoning={},
            features_used={},
            metadata={},
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_neutral_agent_makes_no_trades():
    candles = _make_candles(100, drift=0.01)
    engine = SoloBacktestEngine(agent=_NeutralAgent(), initial_capital=10_000.0)
    result = engine.run({"BTC/USDT": candles})
    assert result.trade_count == 0
    assert result.final_equity == pytest.approx(10_000.0)


def test_low_confidence_signals_are_skipped():
    candles = _make_candles(100, drift=0.01)
    engine = SoloBacktestEngine(
        agent=_LowConfidenceLongAgent(),
        initial_capital=10_000.0,
        confidence_threshold=0.6,
    )
    result = engine.run({"BTC/USDT": candles})
    assert result.trade_count == 0


def test_always_long_enters_on_first_qualifying_bar():
    candles = _make_candles(50, drift=0.0)  # flat market
    engine = SoloBacktestEngine(agent=_AlwaysLongAgent(), initial_capital=10_000.0)
    result = engine.run({"BTC/USDT": candles})
    assert result.trade_count >= 1


def test_rising_market_always_long_makes_money():
    candles = _make_candles(200, drift=0.01)  # +1% per day, very bullish
    engine = SoloBacktestEngine(
        agent=_AlwaysLongAgent(),
        initial_capital=10_000.0,
        commission_pct=0.001,
    )
    result = engine.run({"BTC/USDT": candles})
    assert result.final_equity > 10_000.0
    assert result.total_return_pct > 0.0


def test_falling_market_always_long_loses_money():
    candles = _make_candles(200, drift=-0.005)  # -0.5% per day
    engine = SoloBacktestEngine(
        agent=_AlwaysLongAgent(),
        initial_capital=10_000.0,
        commission_pct=0.001,
    )
    result = engine.run({"BTC/USDT": candles})
    assert result.final_equity < 10_000.0
    assert result.total_return_pct < 0.0


def test_per_year_sharpe_populated():
    # Two calendar years
    ts = datetime(2022, 1, 1, tzinfo=timezone.utc)
    candles_22 = [_candle(ts + timedelta(days=i), 100.0 * (1 + 0.001 * i)) for i in range(200)]
    ts23 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    candles_23 = [_candle(ts23 + timedelta(days=i), 120.0 * (1 + 0.001 * i)) for i in range(200)]
    all_candles = candles_22 + candles_23

    engine = SoloBacktestEngine(agent=_AlwaysLongAgent(), initial_capital=10_000.0)
    result = engine.run({"BTC/USDT": all_candles})
    assert 2022 in result.per_year_sharpe
    assert 2023 in result.per_year_sharpe


def test_result_shape():
    candles = _make_candles(60, drift=0.002)
    engine = SoloBacktestEngine(agent=_AlwaysLongAgent(), initial_capital=10_000.0)
    result = engine.run({"BTC/USDT": candles})
    assert isinstance(result, SoloResult)
    assert result.agent_id == "always_long"
    assert result.initial_capital == 10_000.0
    assert isinstance(result.per_year_sharpe, dict)
    assert isinstance(result.trade_count, int)
    assert result.trade_count >= 0


def test_position_size_matches_risk_fraction():
    # 30 flat candles so ATR is defined and warm-up passes.
    candles = _make_candles(30, start_price=100.0, drift=0.0)
    engine = SoloBacktestEngine(
        agent=_AlwaysLongAgent(),
        initial_capital=10_000.0,
        fixed_risk_pct=0.10,
        commission_pct=0.0,
    )
    result = engine.run({"BTC/USDT": candles})
    assert result.trade_count >= 1
    first_trade = result.trades[0]
    # First trade notional should be ~10% of initial equity.
    assert first_trade["notional"] == pytest.approx(1_000.0, rel=0.05)


def test_stop_loss_triggers_exit():
    # Bar 0: price 100 → agent enters long at 100. Stop at 100 - 2*ATR.
    # Bar 1: price crashes below stop → exit triggered.
    ts = datetime(2022, 1, 1, tzinfo=timezone.utc)
    candles = [
        _candle(ts, 100.0, high=101.0, low=99.0),
        _candle(ts + timedelta(days=1), 100.0, high=101.0, low=99.0),
        _candle(ts + timedelta(days=2), 100.0, high=101.0, low=99.0),
    ]
    # Warm up ATR with stable bars, then crash.
    for i in range(3, 20):
        candles.append(_candle(ts + timedelta(days=i), 100.0, high=101.0, low=99.0))
    # Crash bar
    candles.append(_candle(ts + timedelta(days=20), 80.0, high=100.0, low=75.0))

    engine = SoloBacktestEngine(
        agent=_AlwaysLongAgent(),
        initial_capital=10_000.0,
        atr_stop_mult=2.0,
    )
    result = engine.run({"BTC/USDT": candles})
    # Should have at least one completed trade (stopped out or closed at end).
    assert result.trade_count >= 1
    # At least one trade should be a "stop" exit.
    exit_reasons = {t["exit_reason"] for t in result.trades}
    assert "stop" in exit_reasons
