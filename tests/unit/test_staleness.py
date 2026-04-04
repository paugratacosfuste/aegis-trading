"""Tests for data staleness checker. Written FIRST per TDD."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest


class TestStalenessChecker:
    def test_crypto_fresh(self):
        """Crypto data < 30s old is not stale."""
        from aegis.data.staleness import StalenessChecker

        db = MagicMock()
        now = datetime.now(timezone.utc)
        db.fetch_one.return_value = {"max": now - timedelta(seconds=10)}

        checker = StalenessChecker(db, crypto_max_sec=30, equity_market_max_sec=300)
        is_stale, age = checker.check("BTC/USDT", "crypto")
        assert not is_stale
        assert age < 30

    def test_crypto_stale(self):
        """Crypto data > 30s old is stale."""
        from aegis.data.staleness import StalenessChecker

        db = MagicMock()
        now = datetime.now(timezone.utc)
        db.fetch_one.return_value = {"max": now - timedelta(seconds=60)}

        checker = StalenessChecker(db, crypto_max_sec=30, equity_market_max_sec=300)
        is_stale, age = checker.check("BTC/USDT", "crypto")
        assert is_stale
        assert age >= 60

    def test_equity_fresh_during_market_hours(self):
        """Equity data < 5min old during market hours is not stale."""
        from aegis.data.staleness import StalenessChecker

        db = MagicMock()
        now = datetime.now(timezone.utc)
        db.fetch_one.return_value = {"max": now - timedelta(seconds=120)}

        checker = StalenessChecker(db, crypto_max_sec=30, equity_market_max_sec=300)
        is_stale, age = checker.check("AAPL", "equity", market_open=True)
        assert not is_stale

    def test_equity_stale_during_market_hours(self):
        """Equity data > 5min old during market hours is stale."""
        from aegis.data.staleness import StalenessChecker

        db = MagicMock()
        now = datetime.now(timezone.utc)
        db.fetch_one.return_value = {"max": now - timedelta(seconds=600)}

        checker = StalenessChecker(db, crypto_max_sec=30, equity_market_max_sec=300)
        is_stale, age = checker.check("AAPL", "equity", market_open=True)
        assert is_stale

    def test_equity_after_hours_lenient(self):
        """Equity data < 24h during after hours is not stale."""
        from aegis.data.staleness import StalenessChecker

        db = MagicMock()
        now = datetime.now(timezone.utc)
        db.fetch_one.return_value = {"max": now - timedelta(hours=12)}

        checker = StalenessChecker(
            db, crypto_max_sec=30, equity_market_max_sec=300,
            equity_after_hours_max_sec=86400,
        )
        is_stale, age = checker.check("AAPL", "equity", market_open=False)
        assert not is_stale

    def test_no_data_is_stale(self):
        """No data at all for a symbol is stale."""
        from aegis.data.staleness import StalenessChecker

        db = MagicMock()
        db.fetch_one.return_value = None

        checker = StalenessChecker(db, crypto_max_sec=30, equity_market_max_sec=300)
        is_stale, age = checker.check("BTC/USDT", "crypto")
        assert is_stale
        assert age == float("inf")
