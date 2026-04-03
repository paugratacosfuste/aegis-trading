"""Custom exception hierarchy for Aegis Trading System."""


class AegisError(Exception):
    """Base exception for all Aegis errors."""


class ConfigError(AegisError):
    """Configuration loading or validation error."""


class DatabaseError(AegisError):
    """Database connection or query error."""


class DataStaleError(AegisError):
    """Data is too old to be trusted."""


class InsufficientDataError(AegisError):
    """Not enough data points to compute a signal."""


class RiskVetoError(AegisError):
    """Risk management rejected the trade."""


class BrokerError(AegisError):
    """Broker API communication error."""


class OrderError(BrokerError):
    """Order placement or management error."""
