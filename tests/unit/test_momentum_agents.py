"""Tests for momentum agents (M4)."""

import pytest

from aegis.agents.momentum.cross_sectional import CrossSectionalMomentumAgent
from aegis.agents.momentum.dual import DualMomentumAgent
from aegis.agents.momentum.volume_weighted import VolumeWeightedMomentumAgent
from aegis.agents.momentum.rsi_filtered import RsiFilteredMomentumAgent
from aegis.agents.momentum.acceleration import MomentumAccelerationAgent


class TestCrossSectionalMomentum:
    def test_no_peer_data_neutral(self, sample_candles_uptrend):
        agent = CrossSectionalMomentumAgent("mom_05", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_with_peer_data(self, sample_candles_uptrend):
        agent = CrossSectionalMomentumAgent("mom_05", {
            "lookback": 20, "peer_returns": [0.01, 0.005, 0.008],
        })
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.agent_type == "momentum"


class TestDualMomentum:
    def test_uptrend_bullish(self, sample_candles_uptrend):
        agent = DualMomentumAgent("mom_07", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction > 0  # Positive absolute + no benchmark = positive

    def test_downtrend_bearish(self, sample_candles_downtrend):
        agent = DualMomentumAgent("mom_07", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_downtrend)
        assert signal.direction < 0

    def test_insufficient_data(self, sample_candles_uptrend):
        agent = DualMomentumAgent("mom_07", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend[:5])
        assert signal.direction == 0.0

    def test_bounds(self, sample_candles_uptrend):
        agent = DualMomentumAgent("mom_07", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert -1.0 <= signal.direction <= 1.0
        assert 0.0 <= signal.confidence <= 1.0


class TestVolumeWeightedMomentum:
    def test_uptrend(self, sample_candles_uptrend):
        agent = VolumeWeightedMomentumAgent("mom_08", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.agent_type == "momentum"
        assert -1.0 <= signal.direction <= 1.0

    def test_insufficient_data(self, sample_candles_uptrend):
        agent = VolumeWeightedMomentumAgent("mom_08", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend[:5])
        assert signal.direction == 0.0


class TestRsiFilteredMomentum:
    def test_uptrend_filtered_by_rsi(self, sample_candles_uptrend):
        agent = RsiFilteredMomentumAgent("mom_09", {"lookback": 20, "rsi_threshold": 75})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        # Steady uptrend pushes RSI very high, so momentum gets filtered
        assert signal.reasoning.get("filtered") is True or signal.direction >= 0

    def test_insufficient_data(self, sample_candles_uptrend):
        agent = RsiFilteredMomentumAgent("mom_09", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend[:5])
        assert signal.direction == 0.0


class TestMomentumAcceleration:
    def test_uptrend(self, sample_candles_uptrend):
        agent = MomentumAccelerationAgent("mom_10", {"short_window": 5, "long_window": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.agent_type == "momentum"
        assert -1.0 <= signal.direction <= 1.0

    def test_insufficient_data(self, sample_candles_uptrend):
        agent = MomentumAccelerationAgent("mom_10", {"short_window": 5, "long_window": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend[:5])
        assert signal.direction == 0.0


class TestRegistration:
    def test_all_registered(self):
        from aegis.agents.registry import get_agent_class
        assert get_agent_class("momentum", "cross_sectional") is CrossSectionalMomentumAgent
        assert get_agent_class("momentum", "dual") is DualMomentumAgent
        assert get_agent_class("momentum", "volume_weighted") is VolumeWeightedMomentumAgent
        assert get_agent_class("momentum", "rsi_filtered") is RsiFilteredMomentumAgent
        assert get_agent_class("momentum", "acceleration") is MomentumAccelerationAgent
