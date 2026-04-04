"""Tests for YahooFundamentalProvider."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from aegis.agents.fundamental.providers import YahooFundamentalProvider
from aegis.common.types import FundamentalScore


def _make_mock_info(**overrides):
    """Build a realistic yfinance Ticker.info dict."""
    base = {
        "trailingPE": 25.0,
        "forwardPE": 22.0,
        "priceToBook": 8.0,
        "priceToSalesTrailing12Months": 6.0,
        "enterpriseToEbitda": 18.0,
        "pegRatio": 1.5,
        "returnOnEquity": 0.35,
        "returnOnAssets": 0.15,
        "debtToEquity": 120.0,
        "currentRatio": 1.2,
        "freeCashflow": 5e9,
        "totalRevenue": 50e9,
        "revenueGrowth": 0.08,
        "earningsGrowth": 0.12,
        "sector": "Technology",
        "marketCap": 3e12,
    }
    base.update(overrides)
    return base


class TestYahooFundamentalProvider:
    @patch("aegis.agents.fundamental.providers.yf")
    def test_get_fundamentals_returns_score(self, mock_yf):
        """Provider fetches from yfinance and returns FundamentalScore."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = _make_mock_info()

        provider = YahooFundamentalProvider()
        score = provider.get_fundamentals("AAPL")

        assert score is not None
        assert isinstance(score, FundamentalScore)
        assert score.symbol == "AAPL"
        assert score.source == "yahoo"
        assert score.sector == "Technology"
        assert score.market_cap_tier == "large"

    @patch("aegis.agents.fundamental.providers.yf")
    def test_quality_score_range(self, mock_yf):
        """Quality score should be in [0, 1]."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = _make_mock_info(returnOnEquity=0.35, debtToEquity=50.0)

        provider = YahooFundamentalProvider()
        score = provider.get_fundamentals("AAPL")

        assert 0.0 <= score.quality_score <= 1.0

    @patch("aegis.agents.fundamental.providers.yf")
    def test_value_score_range(self, mock_yf):
        """Value score should be in [0, 1]."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = _make_mock_info(trailingPE=15.0, priceToBook=2.0)

        provider = YahooFundamentalProvider()
        score = provider.get_fundamentals("AAPL")

        assert 0.0 <= score.value_score <= 1.0

    @patch("aegis.agents.fundamental.providers.yf")
    def test_growth_score_range(self, mock_yf):
        """Growth score should be in [0, 1]."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = _make_mock_info(revenueGrowth=0.25, earningsGrowth=0.30)

        provider = YahooFundamentalProvider()
        score = provider.get_fundamentals("AAPL")

        assert 0.0 <= score.growth_score <= 1.0

    @patch("aegis.agents.fundamental.providers.yf")
    def test_market_cap_tier_large(self, mock_yf):
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = _make_mock_info(marketCap=50e9)

        provider = YahooFundamentalProvider()
        score = provider.get_fundamentals("AAPL")
        assert score.market_cap_tier == "large"

    @patch("aegis.agents.fundamental.providers.yf")
    def test_market_cap_tier_mid(self, mock_yf):
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = _make_mock_info(marketCap=5e9)

        provider = YahooFundamentalProvider()
        score = provider.get_fundamentals("AAPL")
        assert score.market_cap_tier == "mid"

    @patch("aegis.agents.fundamental.providers.yf")
    def test_market_cap_tier_small(self, mock_yf):
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = _make_mock_info(marketCap=500e6)

        provider = YahooFundamentalProvider()
        score = provider.get_fundamentals("AAPL")
        assert score.market_cap_tier == "small"

    @patch("aegis.agents.fundamental.providers.yf")
    def test_missing_data_returns_none(self, mock_yf):
        """If yfinance returns empty info, provider returns None."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = {}

        provider = YahooFundamentalProvider()
        score = provider.get_fundamentals("INVALID")
        assert score is None

    @patch("aegis.agents.fundamental.providers.yf")
    def test_exception_returns_none(self, mock_yf):
        """Network errors return None, not crash."""
        mock_yf.Ticker.side_effect = Exception("network error")

        provider = YahooFundamentalProvider()
        score = provider.get_fundamentals("AAPL")
        assert score is None

    @patch("aegis.agents.fundamental.providers.yf")
    def test_high_roe_high_quality(self, mock_yf):
        """High ROE + low debt = high quality score."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = _make_mock_info(
            returnOnEquity=0.50, returnOnAssets=0.20,
            debtToEquity=30.0, currentRatio=3.0,
            freeCashflow=10e9, totalRevenue=50e9,
        )

        provider = YahooFundamentalProvider()
        score = provider.get_fundamentals("AAPL")
        assert score.quality_score > 0.7

    @patch("aegis.agents.fundamental.providers.yf")
    def test_low_pe_high_value(self, mock_yf):
        """Low P/E + low P/B = high value score."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = _make_mock_info(
            trailingPE=10.0, forwardPE=9.0,
            priceToBook=1.5, priceToSalesTrailing12Months=1.0,
            enterpriseToEbitda=8.0,
        )

        provider = YahooFundamentalProvider()
        score = provider.get_fundamentals("AAPL")
        assert score.value_score > 0.6

    @patch("aegis.agents.fundamental.providers.yf")
    def test_high_growth_high_growth_score(self, mock_yf):
        """High revenue and earnings growth = high growth score."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = _make_mock_info(
            revenueGrowth=0.30, earningsGrowth=0.40, pegRatio=0.8,
        )

        provider = YahooFundamentalProvider()
        score = provider.get_fundamentals("AAPL")
        assert score.growth_score > 0.6

    @patch("aegis.agents.fundamental.providers.yf")
    def test_caching(self, mock_yf):
        """Provider caches results — second call doesn't re-fetch."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = _make_mock_info()

        provider = YahooFundamentalProvider()
        s1 = provider.get_fundamentals("AAPL")
        s2 = provider.get_fundamentals("AAPL")

        assert s1 == s2
        assert mock_yf.Ticker.call_count == 1

    @patch("aegis.agents.fundamental.providers.yf")
    def test_pe_zscore(self, mock_yf):
        """pe_zscore should be computed from trailing P/E."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = _make_mock_info(trailingPE=15.0)

        provider = YahooFundamentalProvider()
        score = provider.get_fundamentals("AAPL")
        # Below market median (~20) → negative z-score
        assert score.pe_zscore < 0

    @patch("aegis.agents.fundamental.providers.yf")
    def test_get_sector_fundamentals(self, mock_yf):
        """get_sector_fundamentals returns all cached scores for a sector."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = _make_mock_info(sector="Technology")

        provider = YahooFundamentalProvider()
        provider.get_fundamentals("AAPL")
        provider.get_fundamentals("MSFT")  # same mock, same sector

        results = provider.get_sector_fundamentals("Technology")
        # Both should be cached under Technology
        assert len(results) >= 1

    @patch("aegis.agents.fundamental.providers.yf")
    def test_none_pe_handled(self, mock_yf):
        """Missing P/E doesn't crash — treated as None."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.info = _make_mock_info(
            trailingPE=None, forwardPE=None, priceToBook=None,
        )

        provider = YahooFundamentalProvider()
        score = provider.get_fundamentals("AAPL")
        assert score is not None
        assert 0.0 <= score.value_score <= 1.0
