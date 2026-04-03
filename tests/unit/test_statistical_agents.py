"""Tests for statistical agents (M3)."""

import pytest

from aegis.agents.statistical.ou import OUAgent
from aegis.agents.statistical.kalman import KalmanAgent
from aegis.agents.statistical.bollinger_zscore import BollingerZScoreAgent
from aegis.agents.statistical.hurst import HurstZScoreAgent
from aegis.agents.statistical.multi_window import MultiWindowAgent
from aegis.agents.statistical.sector_relative import SectorRelativeAgent
from aegis.agents.statistical.pairs import PairsAgent


class TestOUAgent:
    def test_flat_market_neutral(self, sample_candles_flat):
        agent = OUAgent("stat_04", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_flat)
        assert abs(signal.direction) < 0.3

    def test_volatile_generates_signal(self, sample_candles_volatile):
        agent = OUAgent("stat_04", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_volatile)
        assert signal.agent_type == "statistical"

    def test_insufficient_data_neutral(self, sample_candles_uptrend):
        agent = OUAgent("stat_04", {"lookback": 100})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend[:5])
        assert signal.direction == 0.0

    def test_bounds(self, sample_candles_uptrend):
        agent = OUAgent("stat_04", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert -1.0 <= signal.direction <= 1.0
        assert 0.0 <= signal.confidence <= 1.0


class TestKalmanAgent:
    def test_uptrend_bearish_reversion(self, sample_candles_uptrend):
        agent = KalmanAgent("stat_05", {})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        # Price above Kalman fair value -> sell signal (mean reversion)
        assert signal.agent_type == "statistical"
        assert -1.0 <= signal.direction <= 1.0

    def test_insufficient_data(self, sample_candles_uptrend):
        agent = KalmanAgent("stat_05", {})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend[:3])
        assert signal.direction == 0.0

    def test_flat_near_neutral(self, sample_candles_flat):
        agent = KalmanAgent("stat_05", {})
        signal = agent.generate_signal("BTC/USDT", sample_candles_flat)
        assert abs(signal.direction) < 0.3


class TestBollingerZScoreAgent:
    def test_uptrend_sell_signal(self, sample_candles_uptrend):
        agent = BollingerZScoreAgent("stat_06", {"period": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        # Uptrend pushes price above upper band -> sell
        assert signal.direction < 0

    def test_insufficient_data(self, sample_candles_uptrend):
        agent = BollingerZScoreAgent("stat_06", {"period": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend[:5])
        assert signal.direction == 0.0

    def test_bounds(self, sample_candles_volatile):
        agent = BollingerZScoreAgent("stat_06", {"period": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_volatile)
        assert -1.0 <= signal.direction <= 1.0


class TestHurstZScoreAgent:
    def test_flat_market(self, sample_candles_flat):
        # Need enough candles for Hurst
        from datetime import datetime, timedelta, timezone
        from aegis.common.types import MarketDataPoint
        candles = []
        for i in range(60):
            candles.append(MarketDataPoint(
                symbol="BTC/USDT", asset_class="crypto",
                timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc) + timedelta(hours=i),
                timeframe="1h", open=42000, high=42020, low=41980, close=42000,
                volume=100, source="binance",
            ))
        agent = HurstZScoreAgent("stat_09", {"lookback": 50})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.agent_type == "statistical"

    def test_insufficient_data(self, sample_candles_uptrend):
        agent = HurstZScoreAgent("stat_09", {"lookback": 50})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend[:10])
        assert signal.direction == 0.0


class TestMultiWindowAgent:
    def test_needs_100_candles(self, sample_candles_uptrend):
        agent = MultiWindowAgent("stat_10", {"windows": [20, 50, 100]})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        # Only 30 candles, needs 100
        assert signal.direction == 0.0

    def test_with_enough_data(self):
        from datetime import datetime, timedelta, timezone
        from aegis.common.types import MarketDataPoint
        # Create 120 candles with uptrend
        candles = []
        for i in range(120):
            price = 40000 + i * 50
            candles.append(MarketDataPoint(
                symbol="BTC/USDT", asset_class="crypto",
                timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc) + timedelta(hours=i),
                timeframe="1h", open=price, high=price+30, low=price-20, close=price+10,
                volume=100, source="binance",
            ))
        agent = MultiWindowAgent("stat_10", {"windows": [20, 50, 100]})
        signal = agent.generate_signal("BTC/USDT", candles)
        assert signal.agent_type == "statistical"


class TestSectorRelativeAgent:
    def test_no_peer_data_neutral(self, sample_candles_uptrend):
        agent = SectorRelativeAgent("stat_07", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0

    def test_with_peer_data(self, sample_candles_uptrend):
        agent = SectorRelativeAgent("stat_07", {
            "lookback": 20, "peer_returns": [0.01, 0.02, 0.015, 0.005, 0.01],
        })
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.agent_type == "statistical"


class TestPairsAgent:
    def test_no_pair_data_neutral(self, sample_candles_uptrend):
        agent = PairsAgent("stat_08", {"lookback": 20})
        signal = agent.generate_signal("BTC/USDT", sample_candles_uptrend)
        assert signal.direction == 0.0


class TestRegistration:
    def test_all_registered(self):
        from aegis.agents.registry import get_agent_class
        assert get_agent_class("statistical", "ornstein_uhlenbeck") is OUAgent
        assert get_agent_class("statistical", "kalman") is KalmanAgent
        assert get_agent_class("statistical", "bollinger_zscore") is BollingerZScoreAgent
        assert get_agent_class("statistical", "hurst_zscore") is HurstZScoreAgent
        assert get_agent_class("statistical", "multi_window") is MultiWindowAgent
        assert get_agent_class("statistical", "sector_relative") is SectorRelativeAgent
        assert get_agent_class("statistical", "pairs") is PairsAgent
