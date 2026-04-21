"""Unit tests for SimpleExecutor — Aegis 2.0 Phase 2.3.

Floor executor: 1/N capital on ``long``, 2x ATR stop, 4x ATR target,
close on stop / target / opposing signal / flat signal. N = max_positions.
Long-only initially (crypto spot). Shorts ignored or treated as ``flat``
depending on allow_short flag.

SimpleExecutor.step(thesis_signal, market_state, portfolio_state) →
  list[ExecutionAction]
where ExecutionAction is a frozen dataclass with action kind (``open`` |
``close`` | ``noop``) plus the operational fields the engine needs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aegis.common.types import MarketDataPoint, ThesisSignal
from aegis.execution.simple_executor import (
    ExecutionAction,
    ExecutorPosition,
    SimpleExecutor,
)


def _candle(ts: datetime, close: float,
            *, high: float | None = None,
            low: float | None = None) -> MarketDataPoint:
    return MarketDataPoint(
        symbol="BTC/USDT",
        asset_class="crypto",
        timestamp=ts,
        timeframe="1w",
        open=close,
        high=high if high is not None else close * 1.02,
        low=low if low is not None else close * 0.98,
        close=close,
        volume=1000.0,
        source="binance",
    )


def _thesis(direction: str, *, conviction: float = 0.5,
            symbol: str = "BTC/USDT",
            ts: datetime | None = None) -> ThesisSignal:
    return ThesisSignal(
        symbol=symbol,
        timestamp=ts or datetime(2024, 1, 1, tzinfo=timezone.utc),
        direction=direction,
        conviction=conviction,
        contributing_agents=("tech_01",),
        metadata={},
    )


# ---------------------------------------------------------------------------


def test_flat_signal_with_no_position_is_noop():
    ex = SimpleExecutor(max_positions=3)
    actions = ex.step(
        thesis=_thesis("flat"),
        candle=_candle(datetime(2024, 1, 1, tzinfo=timezone.utc), 100.0),
        atr=2.0,
        equity=10_000.0,
        open_positions={},
    )
    assert len(actions) == 1
    assert actions[0].kind == "noop"


def test_long_signal_opens_long_position_sized_to_one_over_n():
    ex = SimpleExecutor(max_positions=3)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    actions = ex.step(
        thesis=_thesis("long"),
        candle=_candle(ts, 100.0),
        atr=2.0,
        equity=9_000.0,
        open_positions={},
    )
    opens = [a for a in actions if a.kind == "open"]
    assert len(opens) == 1
    op = opens[0]
    # 1/N of equity at entry_price
    assert op.notional == pytest.approx(3_000.0)
    assert op.direction == +1
    assert op.stop_price == pytest.approx(100.0 - 2.0 * 2.0)  # 2x ATR = 4
    assert op.target_price == pytest.approx(100.0 + 4.0 * 2.0)  # 4x ATR = 8


def test_short_signal_opens_short_when_allowed():
    ex = SimpleExecutor(max_positions=3, allow_short=True)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    actions = ex.step(
        thesis=_thesis("short"),
        candle=_candle(ts, 100.0),
        atr=2.0,
        equity=6_000.0,
        open_positions={},
    )
    opens = [a for a in actions if a.kind == "open"]
    assert len(opens) == 1
    op = opens[0]
    assert op.direction == -1
    assert op.stop_price == pytest.approx(100.0 + 2.0 * 2.0)  # 2x ATR = 4
    assert op.target_price == pytest.approx(100.0 - 4.0 * 2.0)  # 4x ATR = 8


def test_short_signal_ignored_when_not_allowed():
    ex = SimpleExecutor(max_positions=3, allow_short=False)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    actions = ex.step(
        thesis=_thesis("short"),
        candle=_candle(ts, 100.0),
        atr=2.0,
        equity=6_000.0,
        open_positions={},
    )
    opens = [a for a in actions if a.kind == "open"]
    assert len(opens) == 0


def test_max_positions_cap_prevents_new_entry():
    ex = SimpleExecutor(max_positions=2)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    existing = {
        "ETH/USDT": ExecutorPosition(
            symbol="ETH/USDT", direction=+1, quantity=1.0,
            entry_price=2000.0, stop_price=1900.0, target_price=2200.0,
            notional=2000.0, entry_timestamp=ts,
        ),
        "SOL/USDT": ExecutorPosition(
            symbol="SOL/USDT", direction=+1, quantity=5.0,
            entry_price=100.0, stop_price=95.0, target_price=110.0,
            notional=500.0, entry_timestamp=ts,
        ),
    }
    actions = ex.step(
        thesis=_thesis("long"),
        candle=_candle(ts, 50.0),
        atr=1.0,
        equity=5_000.0,
        open_positions=existing,
    )
    assert all(a.kind != "open" for a in actions)


def test_stop_exit_triggered_when_low_pierces_stop():
    ex = SimpleExecutor(max_positions=3)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    pos = ExecutorPosition(
        symbol="BTC/USDT", direction=+1, quantity=10.0,
        entry_price=100.0, stop_price=95.0, target_price=120.0,
        notional=1000.0, entry_timestamp=ts,
    )
    bad = _candle(ts + timedelta(days=7), 92.0, high=98.0, low=90.0)
    actions = ex.step(
        thesis=_thesis("long"),
        candle=bad,
        atr=2.0,
        equity=10_000.0,
        open_positions={"BTC/USDT": pos},
    )
    closes = [a for a in actions if a.kind == "close"]
    assert len(closes) == 1
    assert closes[0].exit_reason == "stop"
    assert closes[0].exit_price == pytest.approx(95.0)


def test_target_exit_triggered_when_high_reaches_target():
    ex = SimpleExecutor(max_positions=3)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    pos = ExecutorPosition(
        symbol="BTC/USDT", direction=+1, quantity=10.0,
        entry_price=100.0, stop_price=95.0, target_price=120.0,
        notional=1000.0, entry_timestamp=ts,
    )
    good = _candle(ts + timedelta(days=7), 118.0, high=122.0, low=105.0)
    actions = ex.step(
        thesis=_thesis("long"),
        candle=good,
        atr=2.0,
        equity=10_000.0,
        open_positions={"BTC/USDT": pos},
    )
    closes = [a for a in actions if a.kind == "close"]
    assert len(closes) == 1
    assert closes[0].exit_reason == "target"
    assert closes[0].exit_price == pytest.approx(120.0)


def test_opposing_signal_closes_position():
    ex = SimpleExecutor(max_positions=3)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    pos = ExecutorPosition(
        symbol="BTC/USDT", direction=+1, quantity=10.0,
        entry_price=100.0, stop_price=95.0, target_price=120.0,
        notional=1000.0, entry_timestamp=ts,
    )
    actions = ex.step(
        thesis=_thesis("short"),
        candle=_candle(ts + timedelta(days=7), 105.0),
        atr=2.0,
        equity=10_000.0,
        open_positions={"BTC/USDT": pos},
    )
    closes = [a for a in actions if a.kind == "close"]
    assert len(closes) == 1
    assert closes[0].exit_reason == "opposing_signal"


def test_flat_signal_closes_open_position():
    ex = SimpleExecutor(max_positions=3, close_on_flat=True)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    pos = ExecutorPosition(
        symbol="BTC/USDT", direction=+1, quantity=10.0,
        entry_price=100.0, stop_price=95.0, target_price=120.0,
        notional=1000.0, entry_timestamp=ts,
    )
    actions = ex.step(
        thesis=_thesis("flat"),
        candle=_candle(ts + timedelta(days=7), 105.0),
        atr=2.0,
        equity=10_000.0,
        open_positions={"BTC/USDT": pos},
    )
    closes = [a for a in actions if a.kind == "close"]
    assert len(closes) == 1
    assert closes[0].exit_reason == "flat_signal"


def test_same_direction_signal_with_open_position_is_noop():
    ex = SimpleExecutor(max_positions=3)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    pos = ExecutorPosition(
        symbol="BTC/USDT", direction=+1, quantity=10.0,
        entry_price=100.0, stop_price=95.0, target_price=120.0,
        notional=1000.0, entry_timestamp=ts,
    )
    actions = ex.step(
        thesis=_thesis("long"),
        candle=_candle(ts + timedelta(days=7), 105.0),
        atr=2.0,
        equity=10_000.0,
        open_positions={"BTC/USDT": pos},
    )
    opens = [a for a in actions if a.kind == "open"]
    closes = [a for a in actions if a.kind == "close"]
    assert len(opens) == 0
    assert len(closes) == 0
