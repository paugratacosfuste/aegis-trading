"""Tests for main entry point argument parsing."""

from unittest.mock import patch

import pytest


class TestMain:
    def test_backtest_mode_calls_run_backtest(self):
        from aegis.main import main

        with patch("aegis.main.run_backtest") as mock_bt, \
             patch("sys.argv", ["main", "--config", "configs/backtest.yaml", "--backtest"]):
            main()
            mock_bt.assert_called_once_with("configs/backtest.yaml")

    def test_live_mode_calls_run_live(self):
        from aegis.main import main

        with patch("aegis.main.asyncio") as mock_asyncio, \
             patch("sys.argv", ["main", "--config", "configs/lab.yaml"]):
            main()
            mock_asyncio.run.assert_called_once()

    def test_missing_config_raises(self):
        from aegis.main import main

        with patch("sys.argv", ["main"]), pytest.raises(SystemExit):
            main()
