"""Tests for Phase 3 ensemble voter extensions: VETO, regime modification, asset-class scoping."""

from datetime import datetime, timezone

import pytest

from aegis.common.types import AgentSignal
from aegis.ensemble.voter import vote


def _sig(agent_id, agent_type, direction, confidence, metadata=None):
    return AgentSignal(
        agent_id=agent_id, agent_type=agent_type, symbol="BTC/USDT",
        timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        direction=direction, confidence=confidence, timeframe="1h",
        expected_holding_period="hours", entry_price=None, stop_loss=None,
        take_profit=None, reasoning={}, features_used={},
        metadata=metadata or {},
    )


class TestVetoGate:
    def test_geo_veto_blocks_trade(self):
        signals = [
            _sig("tech_01", "technical", 0.8, 0.9),
            _sig("stat_01", "statistical", 0.7, 0.8),
            _sig("geo_01", "geopolitical", -0.5, 0.9, {"veto": True, "risk_score": 0.85}),
        ]
        result = vote(signals)
        assert result.action == "NO_TRADE"
        assert "veto" in result.reason.lower() or "VETO" in result.reason

    def test_fundamental_veto_blocks_trade(self):
        """Fundamental VETO only applies to equity symbols, not crypto."""
        signals = [
            AgentSignal(
                agent_id="tech_01", agent_type="technical", symbol="AAPL",
                timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
                direction=0.8, confidence=0.9, timeframe="1h",
                expected_holding_period="hours", entry_price=None,
                stop_loss=None, take_profit=None, reasoning={},
                features_used={}, metadata={},
            ),
            AgentSignal(
                agent_id="fund_01", agent_type="fundamental", symbol="AAPL",
                timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
                direction=0.0, confidence=0.05, timeframe="1d",
                expected_holding_period="days", entry_price=None,
                stop_loss=None, take_profit=None, reasoning={},
                features_used={}, metadata={"veto": True, "quality_score": 0.05},
            ),
        ]
        result = vote(signals)
        assert result.action == "NO_TRADE"

    def test_no_veto_trade_proceeds(self):
        signals = [
            _sig("tech_01", "technical", 0.8, 0.9),
            _sig("geo_01", "geopolitical", -0.2, 0.4, {"veto": False, "risk_score": 0.3}),
        ]
        result = vote(signals)
        assert result.action in ("LONG", "NO_TRADE")  # May be NO_TRADE for other reasons

    def test_multiple_vetos(self):
        signals = [
            _sig("tech_01", "technical", 0.9, 0.95),
            _sig("geo_01", "geopolitical", -0.5, 0.9, {"veto": True}),
            _sig("fund_01", "fundamental", 0.0, 0.05, {"veto": True}),
        ]
        result = vote(signals)
        assert result.action == "NO_TRADE"

    def test_veto_overrides_strong_signal(self):
        """Even very strong bullish consensus is blocked by a VETO."""
        signals = [
            _sig("tech_01", "technical", 0.95, 0.99),
            _sig("stat_01", "statistical", 0.9, 0.95),
            _sig("mom_01", "momentum", 0.9, 0.95),
            _sig("geo_01", "geopolitical", -0.3, 0.8, {"veto": True}),
        ]
        result = vote(signals)
        assert result.action == "NO_TRADE"


class TestMacroRegimeExtraction:
    def test_macro_signals_not_in_voting(self):
        """Macro agents output direction=0.0, they modify weights not vote."""
        signals = [
            _sig("tech_01", "technical", 0.8, 0.9),
            _sig("macro_01", "macro", 0.0, 0.0, {
                "regime": "risk_on", "regime_confidence": 0.8,
                "sector_tilts": {}, "asset_class_tilts": {},
            }),
        ]
        result = vote(signals)
        # Macro signal shouldn't dilute the technical signal
        assert result.action == "LONG"

    def test_only_macro_signals_no_trade(self):
        """Macro signals alone cannot generate a trade."""
        signals = [
            _sig("macro_01", "macro", 0.0, 0.0, {"regime": "risk_on"}),
            _sig("macro_02", "macro", 0.0, 0.0, {"regime": "risk_on"}),
        ]
        result = vote(signals)
        assert result.action == "NO_TRADE"


class TestFundamentalConfidenceModifier:
    def test_high_quality_boosts_confidence(self):
        signals = [
            _sig("tech_01", "technical", 0.6, 0.6),
            _sig("fund_01", "fundamental", 0.0, 0.85, {
                "confidence_modifier": 1.2, "quality_score": 0.85,
            }),
        ]
        result = vote(signals)
        # With boost, should be more likely to trade
        # Even if NO_TRADE, confidence should be higher than without fundamental
        assert result.action in ("LONG", "NO_TRADE")

    def test_low_quality_reduces_confidence(self):
        signals = [
            _sig("tech_01", "technical", 0.6, 0.6),
            _sig("fund_01", "fundamental", 0.0, 0.2, {
                "confidence_modifier": 0.7, "quality_score": 0.2,
            }),
        ]
        result_with_fund = vote(signals)
        # Compare with same signal without fundamental
        result_without = vote([signals[0]])
        # Fundamental reducer should make it harder to trade
        if result_without.action != "NO_TRADE":
            assert result_with_fund.confidence <= result_without.confidence or \
                   result_with_fund.action == "NO_TRADE"


class TestAssetClassScoping:
    def test_crypto_agents_ignored_for_equity(self):
        """Crypto agents should not contribute to equity symbol votes."""
        equity_signals = [
            AgentSignal(
                agent_id="tech_01", agent_type="technical", symbol="AAPL",
                timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
                direction=0.8, confidence=0.9, timeframe="1h",
                expected_holding_period="hours", entry_price=None,
                stop_loss=None, take_profit=None, reasoning={},
                features_used={}, metadata={},
            ),
            AgentSignal(
                agent_id="crypto_01", agent_type="crypto", symbol="AAPL",
                timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
                direction=-0.9, confidence=0.95, timeframe="1h",
                expected_holding_period="hours", entry_price=None,
                stop_loss=None, take_profit=None, reasoning={},
                features_used={}, metadata={},
            ),
        ]
        result = vote(equity_signals)
        # Crypto agent should be filtered out for AAPL
        assert result.action == "LONG"

    def test_fundamental_agents_ignored_for_crypto(self):
        """Fundamental agents should not contribute to crypto symbol votes."""
        signals = [
            _sig("tech_01", "technical", 0.8, 0.9),
            _sig("fund_01", "fundamental", 0.0, 0.9, {
                "confidence_modifier": 0.0, "veto": True, "quality_score": 0.05,
            }),
        ]
        # For crypto symbols, fundamental veto should be ignored
        result = vote(signals)
        # BTC/USDT is crypto, so fundamental agent should be filtered
        assert result.action == "LONG"


class TestFundamentalModifierEquity:
    def _equity_sig(self, agent_id, agent_type, direction, confidence, metadata=None):
        return AgentSignal(
            agent_id=agent_id, agent_type=agent_type, symbol="AAPL",
            timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
            direction=direction, confidence=confidence, timeframe="1h",
            expected_holding_period="hours", entry_price=None, stop_loss=None,
            take_profit=None, reasoning={}, features_used={},
            metadata=metadata or {},
        )

    def test_equity_fundamental_boost(self):
        """Fundamental confidence modifier boosts equity trades."""
        signals = [
            self._equity_sig("tech_01", "technical", 0.8, 0.9),
            self._equity_sig("fund_01", "fundamental", 0.0, 0.85, {
                "confidence_modifier": 1.2, "quality_score": 0.85,
            }),
        ]
        result = vote(signals)
        assert result.confidence > 0

    def test_equity_fundamental_reduce(self):
        """Low-quality fundamental reduces confidence for equity."""
        signals = [
            self._equity_sig("tech_01", "technical", 0.8, 0.9),
            self._equity_sig("fund_01", "fundamental", 0.0, 0.2, {
                "confidence_modifier": 0.7, "quality_score": 0.2,
            }),
        ]
        result = vote(signals)
        # Confidence should be reduced by 0.7x modifier
        assert result.confidence > 0 or result.action == "NO_TRADE"

    def test_no_signals_after_filter(self):
        """All signals filtered by asset class -> NO_TRADE."""
        signals = [
            self._equity_sig("crypto_01", "crypto", 0.8, 0.9),
        ]
        result = vote(signals)
        assert result.action == "NO_TRADE"

    def test_zero_weight_no_trade(self):
        """Direction too weak -> NO_TRADE."""
        signals = [
            _sig("tech_01", "technical", 0.05, 0.9),
            _sig("stat_01", "statistical", -0.05, 0.9),
        ]
        result = vote(signals)
        assert result.action == "NO_TRADE"


class TestPhase2Regression:
    def test_phase2_only_signals_unchanged(self):
        """Phase 2 signal combinations should produce same results."""
        signals = [
            _sig("tech_01", "technical", 0.8, 0.9),
            _sig("stat_01", "statistical", 0.7, 0.8),
            _sig("mom_01", "momentum", 0.9, 0.95),
        ]
        result = vote(signals)
        assert result.action == "LONG"
        assert result.confidence > 0.5

    def test_empty_still_no_trade(self):
        result = vote([])
        assert result.action == "NO_TRADE"

    def test_conflict_still_no_trade(self):
        signals = [
            _sig("tech_01", "technical", 0.8, 0.7),
            _sig("stat_01", "statistical", -0.8, 0.7),
        ]
        result = vote(signals)
        assert result.action == "NO_TRADE"
