"""Fundamental data providers: protocol and concrete implementations.

Decouples fundamental agents from Yahoo Finance / Financial Modeling Prep.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol

import yfinance as yf

from aegis.common.types import FundamentalScore

logger = logging.getLogger(__name__)


class FundamentalProvider(Protocol):
    def get_fundamentals(self, symbol: str) -> FundamentalScore | None:
        """Return latest fundamental score for a symbol."""
        ...

    def get_sector_fundamentals(self, sector: str) -> list[FundamentalScore]:
        """Return all fundamental scores for a given sector."""
        ...


class NullFundamentalProvider:
    """Always returns None. Used in backtest without fundamental data."""

    def get_fundamentals(self, symbol: str) -> None:
        return None

    def get_sector_fundamentals(self, sector: str) -> list[FundamentalScore]:
        return []


class HistoricalFundamentalProvider:
    """Reads from preloaded FundamentalScores."""

    def __init__(self, scores: list[FundamentalScore] | None = None):
        self._by_symbol: dict[str, FundamentalScore] = {}
        self._by_sector: dict[str, list[FundamentalScore]] = {}
        for s in (scores or []):
            self._by_symbol[s.symbol] = s
            self._by_sector.setdefault(s.sector, []).append(s)

    def get_fundamentals(self, symbol: str) -> FundamentalScore | None:
        return self._by_symbol.get(symbol)

    def get_sector_fundamentals(self, sector: str) -> list[FundamentalScore]:
        return self._by_sector.get(sector, [])


# Market median P/E for z-score computation
_MARKET_MEDIAN_PE = 20.0
_MARKET_PE_STD = 10.0


def _safe_float(info: dict, key: str, default: float = 0.0) -> float:
    """Extract a float from yfinance info, handling None/missing."""
    val = info.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


class YahooFundamentalProvider:
    """Fetches real fundamental data from Yahoo Finance via yfinance.

    Computes quality_score, value_score, growth_score from:
    - Valuation: P/E, P/B, P/S, EV/EBITDA, PEG
    - Quality: ROE, ROA, debt-to-equity, current ratio, FCF margin
    - Growth: revenue growth YoY, earnings growth

    Results are cached per symbol for the lifetime of the provider.
    """

    def __init__(self) -> None:
        self._cache: dict[str, FundamentalScore | None] = {}
        self._by_sector: dict[str, list[FundamentalScore]] = {}

    def get_fundamentals(self, symbol: str) -> FundamentalScore | None:
        if symbol in self._cache:
            return self._cache[symbol]

        try:
            score = self._fetch(symbol)
        except Exception:
            logger.exception("Failed to fetch fundamentals for %s", symbol)
            score = None

        self._cache[symbol] = score
        if score is not None:
            self._by_sector.setdefault(score.sector, []).append(score)
        return score

    def get_sector_fundamentals(self, sector: str) -> list[FundamentalScore]:
        return self._by_sector.get(sector, [])

    def _fetch(self, symbol: str) -> FundamentalScore | None:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Minimum data check — need at least marketCap to proceed
        if not info or not info.get("marketCap"):
            return None

        sector = info.get("sector", "Unknown")
        market_cap = _safe_float(info, "marketCap")

        # Market cap tier
        if market_cap >= 10e9:
            tier = "large"
        elif market_cap >= 2e9:
            tier = "mid"
        else:
            tier = "small"

        quality = self._compute_quality(info)
        value = self._compute_value(info)
        growth = self._compute_growth(info)

        # P/E z-score vs market median
        pe = _safe_float(info, "trailingPE")
        pe_zscore = (pe - _MARKET_MEDIAN_PE) / _MARKET_PE_STD if pe > 0 else 0.0

        revenue_growth = _safe_float(info, "revenueGrowth")

        return FundamentalScore(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            sector=sector,
            market_cap_tier=tier,
            quality_score=quality,
            value_score=value,
            growth_score=growth,
            pe_zscore=pe_zscore,
            revenue_growth=revenue_growth,
            source="yahoo",
        )

    def _compute_quality(self, info: dict) -> float:
        """Quality = ROE + ROA + low debt + liquidity + FCF margin."""
        roe = _safe_float(info, "returnOnEquity")
        roa = _safe_float(info, "returnOnAssets")
        dte = _safe_float(info, "debtToEquity", 100.0)
        current = _safe_float(info, "currentRatio", 1.0)
        fcf = _safe_float(info, "freeCashflow")
        revenue = _safe_float(info, "totalRevenue", 1.0)

        # ROE score: 0-1 mapped from 0% to 40%+
        roe_score = _clamp01(roe / 0.40) if roe > 0 else 0.0

        # ROA score: 0-1 mapped from 0% to 20%+
        roa_score = _clamp01(roa / 0.20) if roa > 0 else 0.0

        # Debt score: lower is better. D/E < 50 = great, > 200 = poor
        debt_score = _clamp01(1.0 - dte / 300.0) if dte >= 0 else 0.5

        # Current ratio score: >2 is great, <1 is poor
        cr_score = _clamp01((current - 0.5) / 2.0)

        # FCF margin score
        fcf_margin = fcf / revenue if revenue > 0 else 0.0
        fcf_score = _clamp01(fcf_margin / 0.25)  # 25%+ = perfect

        # Weighted composite
        quality = (
            roe_score * 0.30
            + roa_score * 0.20
            + debt_score * 0.20
            + cr_score * 0.15
            + fcf_score * 0.15
        )
        return _clamp01(quality)

    def _compute_value(self, info: dict) -> float:
        """Value = inverse of valuation multiples (lower P/E = higher value)."""
        pe = _safe_float(info, "trailingPE")
        fwd_pe = _safe_float(info, "forwardPE")
        pb = _safe_float(info, "priceToBook")
        ps = _safe_float(info, "priceToSalesTrailing12Months")
        ev_ebitda = _safe_float(info, "enterpriseToEbitda")

        # P/E score: <10 = great (1.0), >40 = poor (0.0)
        pe_score = _clamp01((40.0 - pe) / 30.0) if pe > 0 else 0.5

        # Forward P/E score
        fwd_pe_score = _clamp01((35.0 - fwd_pe) / 25.0) if fwd_pe > 0 else 0.5

        # P/B score: <2 = great, >10 = poor
        pb_score = _clamp01((10.0 - pb) / 8.0) if pb > 0 else 0.5

        # P/S score: <2 = great, >10 = poor
        ps_score = _clamp01((10.0 - ps) / 8.0) if ps > 0 else 0.5

        # EV/EBITDA score: <10 = great, >25 = poor
        ev_score = _clamp01((25.0 - ev_ebitda) / 15.0) if ev_ebitda > 0 else 0.5

        value = (
            pe_score * 0.25
            + fwd_pe_score * 0.20
            + pb_score * 0.20
            + ps_score * 0.15
            + ev_score * 0.20
        )
        return _clamp01(value)

    def _compute_growth(self, info: dict) -> float:
        """Growth = revenue growth + earnings growth + PEG ratio."""
        rev_growth = _safe_float(info, "revenueGrowth")
        eps_growth = _safe_float(info, "earningsGrowth")
        peg = _safe_float(info, "pegRatio")

        # Revenue growth: 0%=0, 30%+=1.0
        rev_score = _clamp01(rev_growth / 0.30) if rev_growth > 0 else 0.0

        # Earnings growth: 0%=0, 40%+=1.0
        eps_score = _clamp01(eps_growth / 0.40) if eps_growth > 0 else 0.0

        # PEG ratio: <1 = great, >3 = poor
        peg_score = _clamp01((3.0 - peg) / 2.0) if peg > 0 else 0.5

        growth = rev_score * 0.40 + eps_score * 0.35 + peg_score * 0.25
        return _clamp01(growth)
