"""Tests for config loading. Written FIRST per TDD."""

import os
import tempfile
from pathlib import Path

import pytest


class TestConfigLoading:
    def test_load_lab_config(self, tmp_path: Path):
        """Load a valid lab config YAML."""
        from aegis.common.config import load_config

        yaml_content = """
mode: lab
confidence_threshold: 0.45
max_risk_per_trade: 0.05
max_open_positions: 5
kelly_fraction: 0.5
daily_drawdown_halt: -0.15
weekly_drawdown_halt: -0.25
initial_capital: 5000.0
symbols:
  crypto:
    - BTC/USDT
database:
  host: localhost
  port: 5432
  dbname: aegis
  user: aegis
  password: aegis
  min_connections: 2
  max_connections: 10
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)

        config = load_config(str(config_file))
        assert config.mode == "lab"
        assert config.confidence_threshold == 0.45
        assert config.max_risk_per_trade == 0.05
        assert config.max_open_positions == 5
        assert config.initial_capital == 5000.0
        assert "BTC/USDT" in config.symbols["crypto"]

    def test_load_production_config(self, tmp_path: Path):
        """Production config has stricter thresholds."""
        from aegis.common.config import load_config

        yaml_content = """
mode: production
confidence_threshold: 0.70
max_risk_per_trade: 0.02
max_open_positions: 15
kelly_fraction: 0.5
daily_drawdown_halt: -0.05
weekly_drawdown_halt: -0.10
initial_capital: 5000.0
symbols:
  crypto: []
database:
  host: localhost
  port: 5432
  dbname: aegis
  user: aegis
  password: aegis
  min_connections: 2
  max_connections: 10
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)

        config = load_config(str(config_file))
        assert config.mode == "production"
        assert config.confidence_threshold == 0.70
        assert config.max_risk_per_trade == 0.02

    def test_env_var_substitution(self, tmp_path: Path, monkeypatch):
        """${VAR} in YAML should be replaced with env var values."""
        from aegis.common.config import load_config

        monkeypatch.setenv("TEST_API_KEY", "my-secret-key")
        monkeypatch.setenv("TEST_API_SECRET", "my-secret")

        yaml_content = """
mode: lab
confidence_threshold: 0.45
max_risk_per_trade: 0.05
max_open_positions: 5
kelly_fraction: 0.5
daily_drawdown_halt: -0.15
weekly_drawdown_halt: -0.25
initial_capital: 5000.0
symbols:
  crypto: []
database:
  host: localhost
  port: 5432
  dbname: aegis
  user: aegis
  password: aegis
  min_connections: 2
  max_connections: 10
binance:
  testnet: true
  api_key: ${TEST_API_KEY}
  api_secret: ${TEST_API_SECRET}
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)

        config = load_config(str(config_file))
        assert config.binance["api_key"] == "my-secret-key"
        assert config.binance["api_secret"] == "my-secret"

    def test_missing_file_raises(self):
        """Loading a nonexistent file should raise ConfigError."""
        from aegis.common.config import load_config
        from aegis.common.exceptions import ConfigError

        with pytest.raises(ConfigError):
            load_config("/nonexistent/path.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path):
        """Malformed YAML should raise ConfigError."""
        from aegis.common.config import load_config
        from aegis.common.exceptions import ConfigError

        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{invalid yaml: [")

        with pytest.raises(ConfigError):
            load_config(str(config_file))

    def test_missing_required_field_raises(self, tmp_path: Path):
        """Config missing required fields should raise ConfigError."""
        from aegis.common.config import load_config
        from aegis.common.exceptions import ConfigError

        yaml_content = """
mode: lab
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)

        with pytest.raises(ConfigError):
            load_config(str(config_file))

    def test_env_var_missing_stays_as_placeholder(self, tmp_path: Path):
        """Undefined env vars stay as empty string."""
        from aegis.common.config import load_config

        yaml_content = """
mode: lab
confidence_threshold: 0.45
max_risk_per_trade: 0.05
max_open_positions: 5
kelly_fraction: 0.5
daily_drawdown_halt: -0.15
weekly_drawdown_halt: -0.25
initial_capital: 5000.0
symbols:
  crypto: []
database:
  host: localhost
  port: 5432
  dbname: aegis
  user: aegis
  password: aegis
  min_connections: 2
  max_connections: 10
binance:
  api_key: ${UNDEFINED_VAR_12345}
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)

        config = load_config(str(config_file))
        assert config.binance["api_key"] == ""

    def test_database_config_accessible(self, tmp_path: Path):
        """Database config section loads correctly."""
        from aegis.common.config import load_config

        yaml_content = """
mode: lab
confidence_threshold: 0.45
max_risk_per_trade: 0.05
max_open_positions: 5
kelly_fraction: 0.5
daily_drawdown_halt: -0.15
weekly_drawdown_halt: -0.25
initial_capital: 5000.0
symbols:
  crypto: []
database:
  host: myhost
  port: 5433
  dbname: mydb
  user: myuser
  password: mypass
  min_connections: 1
  max_connections: 5
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)

        config = load_config(str(config_file))
        assert config.database["host"] == "myhost"
        assert config.database["port"] == 5433
        assert config.database["dbname"] == "mydb"
