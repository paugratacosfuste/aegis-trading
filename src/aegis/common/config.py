"""Configuration loader with environment variable substitution."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aegis.common.exceptions import ConfigError

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} with environment variable values."""
    if isinstance(value, str):
        def _replacer(match: re.Match) -> str:
            return os.environ.get(match.group(1), "")
        return _ENV_VAR_PATTERN.sub(_replacer, value)
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


_REQUIRED_FIELDS = [
    "mode",
    "confidence_threshold",
    "max_risk_per_trade",
    "max_open_positions",
    "kelly_fraction",
    "daily_drawdown_halt",
    "weekly_drawdown_halt",
    "initial_capital",
    "symbols",
    "database",
]


@dataclass
class Settings:
    """Parsed configuration settings."""

    mode: str
    confidence_threshold: float
    max_risk_per_trade: float
    max_open_positions: int
    kelly_fraction: float
    daily_drawdown_halt: float
    weekly_drawdown_halt: float
    initial_capital: float
    symbols: dict[str, list[str]]
    database: dict[str, Any]
    binance: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    staleness: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    backtest: dict[str, Any] = field(default_factory=dict)
    agents: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    ensemble: dict[str, Any] = field(default_factory=dict)
    lab: dict[str, Any] = field(default_factory=dict)
    rl: dict[str, Any] = field(default_factory=dict)


def load_config(path: str) -> Settings:
    """Load YAML config file with env var substitution.

    Raises ConfigError on missing file, invalid YAML, or missing required fields.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw_text = config_path.read_text()
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a YAML mapping, got {type(data).__name__}")

    for field_name in _REQUIRED_FIELDS:
        if field_name not in data:
            raise ConfigError(f"Missing required config field: {field_name}")

    data = _substitute_env_vars(data)

    return Settings(
        mode=data["mode"],
        confidence_threshold=float(data["confidence_threshold"]),
        max_risk_per_trade=float(data["max_risk_per_trade"]),
        max_open_positions=int(data["max_open_positions"]),
        kelly_fraction=float(data["kelly_fraction"]),
        daily_drawdown_halt=float(data["daily_drawdown_halt"]),
        weekly_drawdown_halt=float(data["weekly_drawdown_halt"]),
        initial_capital=float(data["initial_capital"]),
        symbols=data["symbols"],
        database=data["database"],
        binance=data.get("binance", {}),
        scheduler=data.get("scheduler", {}),
        staleness=data.get("staleness", {}),
        risk=data.get("risk", {}),
        backtest=data.get("backtest", {}),
        agents=data.get("agents", {}),
        ensemble=data.get("ensemble", {}),
        lab=data.get("lab", {}),
        rl=data.get("rl", {}),
    )
