"""Tests for shared dataclasses. Written FIRST per TDD."""

from datetime import datetime, timezone

import pytest


class TestMarketDataPoint:
    def test_create_valid(self):
        from aegis.common.types import MarketDataPoint

        p = MarketDataPoint(
            symbol="BTC/USDT",
            asset_class="crypto",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            timeframe="1h",
            open=42000.0,
            high=42500.0,
            low=41800.0,
            close=42200.0,
            volume=1500.5,
            source="binance",
        )
        assert p.symbol == "BTC/USDT"
        assert p.close == 42200.0

    def test_frozen(self):
        from aegis.common.types import MarketDataPoint

        p = MarketDataPoint(
            symbol="BTC/USDT",
            asset_class="crypto",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            timeframe="1h",
            open=42000.0,
            high=42500.0,
            low=41800.0,
            close=42200.0,
            volume=1500.5,
            source="binance",
        )
        with pytest.raises(AttributeError):
            p.close = 99999.0  # type: ignore


class TestAgentSignal:
    def test_create_valid(self):
        from aegis.common.types import AgentSignal

        s = AgentSignal(
            agent_id="tech_03",
            agent_type="technical",
            symbol="BTC/USDT",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            direction=0.7,
            confidence=0.85,
            timeframe="1h",
            expected_holding_period="hours",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasoning={"rsi": 28, "ema_cross": "bullish"},
            features_used={"rsi_14": 28.0},
            metadata={},
        )
        assert s.direction == 0.7
        assert s.confidence == 0.85

    def test_direction_clamped_to_bounds(self):
        """Direction outside [-1, 1] should be clamped."""
        from aegis.common.types import AgentSignal

        s = AgentSignal(
            agent_id="test",
            agent_type="technical",
            symbol="BTC/USDT",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            direction=1.5,
            confidence=0.5,
            timeframe="1h",
            expected_holding_period="hours",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasoning={},
            features_used={},
            metadata={},
        )
        assert s.direction == 1.0

    def test_direction_negative_clamped(self):
        from aegis.common.types import AgentSignal

        s = AgentSignal(
            agent_id="test",
            agent_type="technical",
            symbol="BTC/USDT",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            direction=-2.0,
            confidence=0.5,
            timeframe="1h",
            expected_holding_period="hours",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasoning={},
            features_used={},
            metadata={},
        )
        assert s.direction == -1.0

    def test_confidence_clamped_to_bounds(self):
        """Confidence outside [0, 1] should be clamped."""
        from aegis.common.types import AgentSignal

        s = AgentSignal(
            agent_id="test",
            agent_type="technical",
            symbol="BTC/USDT",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            direction=0.5,
            confidence=1.5,
            timeframe="1h",
            expected_holding_period="hours",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasoning={},
            features_used={},
            metadata={},
        )
        assert s.confidence == 1.0

    def test_confidence_negative_clamped(self):
        from aegis.common.types import AgentSignal

        s = AgentSignal(
            agent_id="test",
            agent_type="technical",
            symbol="BTC/USDT",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            direction=0.5,
            confidence=-0.3,
            timeframe="1h",
            expected_holding_period="hours",
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasoning={},
            features_used={},
            metadata={},
        )
        assert s.confidence == 0.0


class TestTradeDecision:
    def test_no_trade(self):
        from aegis.common.types import TradeDecision

        d = TradeDecision(
            action="NO_TRADE",
            symbol="BTC/USDT",
            direction=0.0,
            confidence=0.2,
            quantity=0.0,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            contributing_signals={},
            reason="Below confidence threshold",
        )
        assert d.action == "NO_TRADE"
        assert d.quantity == 0.0

    def test_long_decision(self):
        from aegis.common.types import TradeDecision

        d = TradeDecision(
            action="LONG",
            symbol="ETH/USDT",
            direction=0.8,
            confidence=0.65,
            quantity=0.5,
            entry_price=3200.0,
            stop_loss=3100.0,
            take_profit=3400.0,
            contributing_signals={},
            reason="Strong consensus",
        )
        assert d.action == "LONG"
        assert d.entry_price == 3200.0


class TestTradeLog:
    def test_create_entry(self):
        from aegis.common.types import TradeLog

        t = TradeLog(
            trade_id="trade-001",
            account_type="paper",
            symbol="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=42000.0,
            entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            exit_price=None,
            exit_time=None,
            quantity=0.01,
            position_value=420.0,
            commission_entry=0.42,
            commission_exit=0.0,
            estimated_slippage=0.1,
            total_costs=0.52,
            gross_pnl=None,
            net_pnl=None,
            return_pct=None,
            r_multiple=None,
            holding_period_hours=None,
            ensemble_confidence=0.65,
            ensemble_direction=0.8,
            agent_signals_json="{}",
            regime_at_entry="normal",
            initial_stop_loss=41000.0,
            risk_amount=10.0,
            risk_pct_of_portfolio=0.02,
            exit_reason=None,
            feature_snapshot_json="{}",
        )
        assert t.trade_id == "trade-001"
        assert t.exit_price is None


class TestPosition:
    def test_create_position(self):
        from aegis.common.types import Position

        pos = Position(
            position_id="pos-001",
            symbol="BTC/USDT",
            direction="LONG",
            quantity=0.01,
            entry_price=42000.0,
            entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            stop_loss=41000.0,
            take_profit=44000.0,
            unrealized_pnl=0.0,
            risk_amount=10.0,
        )
        assert pos.symbol == "BTC/USDT"
        assert pos.direction == "LONG"


class TestOrder:
    def test_create_market_order(self):
        from aegis.common.types import Order

        o = Order(
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.01,
            limit_price=None,
            stop_price=None,
            time_in_force="GTC",
            broker="binance",
            account_type="paper",
            trade_id="trade-001",
            signal_id="sig-001",
        )
        assert o.order_type == "MARKET"
        assert o.limit_price is None


class TestSentimentDataPoint:
    def test_create(self):
        from aegis.common.types import SentimentDataPoint

        s = SentimentDataPoint(
            symbol="BTC/USDT",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            source="finnhub",
            sentiment_score=0.3,
            mention_count=42,
            sentiment_velocity=0.05,
        )
        assert s.sentiment_score == 0.3


class TestMacroDataPoint:
    def test_create(self):
        from aegis.common.types import MacroDataPoint

        m = MacroDataPoint(
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            yield_10y=4.5,
            yield_2y=4.2,
            yield_spread=0.3,
            vix=18.5,
            vix_regime="normal",
            dxy=104.5,
            fed_rate=5.25,
            cpi_latest=3.2,
        )
        assert m.yield_spread == 0.3
        assert m.vix_regime == "normal"
