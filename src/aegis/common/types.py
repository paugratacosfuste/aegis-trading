"""Shared dataclasses for the Aegis Trading System.

All types used across modules are defined here. Frozen where possible
to enforce immutability.
"""

from dataclasses import dataclass, field
from datetime import datetime


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class MarketDataPoint:
    symbol: str
    asset_class: str  # "equity" | "crypto"
    timestamp: datetime  # UTC
    timeframe: str  # "1m" | "5m" | "15m" | "1h" | "4h" | "1d"
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str  # "binance" | "yfinance" | "ibkr"


@dataclass(frozen=True)
class SentimentDataPoint:
    symbol: str
    timestamp: datetime
    source: str  # "finnhub" | "reddit" | "fear_greed"
    sentiment_score: float  # [-1.0, +1.0]
    mention_count: int
    sentiment_velocity: float


@dataclass(frozen=True)
class MacroDataPoint:
    timestamp: datetime
    yield_10y: float
    yield_2y: float
    yield_spread: float
    vix: float
    vix_regime: str  # "low" | "normal" | "high" | "extreme"
    dxy: float
    fed_rate: float
    cpi_latest: float


@dataclass(frozen=True)
class GeopoliticalEvent:
    event_id: str
    timestamp: datetime
    source: str
    category: str  # "conflict" | "trade" | "policy" | "leader"
    severity: float  # [0, 1]
    affected_sectors: tuple[str, ...]
    affected_regions: tuple[str, ...]
    raw_text: str
    sentiment_score: float
    half_life_hours: int


@dataclass(frozen=True)
class FundamentalScore:
    symbol: str
    timestamp: datetime
    sector: str
    market_cap_tier: str  # "large" | "mid" | "small"
    quality_score: float  # [0, 1] composite
    value_score: float  # [0, 1]
    growth_score: float  # [0, 1]
    pe_zscore: float  # vs sector median
    revenue_growth: float
    source: str


@dataclass(frozen=True)
class CryptoMetrics:
    symbol: str
    timestamp: datetime
    funding_rate: float
    open_interest: float
    btc_dominance: float
    fear_greed_index: int  # 0-100
    tvl: float  # Total Value Locked (USD)
    tvl_change_24h: float  # percentage
    liquidations_24h: float  # USD
    source: str


class AgentSignal:
    """Signal output from an agent. Direction and confidence are clamped to valid ranges."""

    __slots__ = (
        "agent_id",
        "agent_type",
        "symbol",
        "timestamp",
        "direction",
        "confidence",
        "timeframe",
        "expected_holding_period",
        "entry_price",
        "stop_loss",
        "take_profit",
        "reasoning",
        "features_used",
        "metadata",
    )

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        symbol: str,
        timestamp: datetime,
        direction: float,
        confidence: float,
        timeframe: str,
        expected_holding_period: str,
        entry_price: float | None,
        stop_loss: float | None,
        take_profit: float | None,
        reasoning: dict,
        features_used: dict,
        metadata: dict,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.symbol = symbol
        self.timestamp = timestamp
        self.direction = _clamp(direction, -1.0, 1.0)
        self.confidence = _clamp(confidence, 0.0, 1.0)
        self.timeframe = timeframe
        self.expected_holding_period = expected_holding_period
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.reasoning = reasoning
        self.features_used = features_used
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"AgentSignal(agent_id={self.agent_id!r}, symbol={self.symbol!r}, "
            f"direction={self.direction:.3f}, confidence={self.confidence:.3f})"
        )


@dataclass
class TradeDecision:
    action: str  # "LONG" | "SHORT" | "NO_TRADE"
    symbol: str
    direction: float
    confidence: float
    quantity: float
    entry_price: float | None
    stop_loss: float | None
    take_profit: float | None
    contributing_signals: dict  # agent_id -> AgentSignal
    reason: str


@dataclass
class Order:
    symbol: str
    side: str  # "BUY" | "SELL"
    order_type: str  # "MARKET" | "LIMIT" | "STOP" | "STOP_LIMIT"
    quantity: float
    limit_price: float | None
    stop_price: float | None
    time_in_force: str  # "GTC" | "DAY" | "IOC"
    broker: str  # "binance" | "ibkr"
    account_type: str  # "live" | "paper"
    trade_id: str
    signal_id: str


@dataclass
class Position:
    position_id: str
    symbol: str
    direction: str  # "LONG" | "SHORT"
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float | None
    unrealized_pnl: float
    risk_amount: float


@dataclass
class TradeLog:
    trade_id: str
    account_type: str
    symbol: str
    asset_class: str
    direction: str  # "LONG" | "SHORT"
    entry_price: float
    entry_time: datetime
    exit_price: float | None
    exit_time: datetime | None
    quantity: float
    position_value: float
    commission_entry: float
    commission_exit: float
    estimated_slippage: float
    total_costs: float
    gross_pnl: float | None
    net_pnl: float | None
    return_pct: float | None
    r_multiple: float | None
    holding_period_hours: float | None
    ensemble_confidence: float
    ensemble_direction: float
    agent_signals_json: str  # JSON serialized
    regime_at_entry: str
    initial_stop_loss: float
    risk_amount: float
    risk_pct_of_portfolio: float
    exit_reason: str | None
    feature_snapshot_json: str  # JSON serialized


@dataclass
class RiskVerdict:
    approved: bool
    reason: str
    position_size: float = 0.0
    stop_loss: float = 0.0

    @classmethod
    def approve(cls, position_size: float, stop_loss: float) -> "RiskVerdict":
        return cls(approved=True, reason="approved", position_size=position_size, stop_loss=stop_loss)

    @classmethod
    def reject(cls, reason: str) -> "RiskVerdict":
        return cls(approved=False, reason=reason)
