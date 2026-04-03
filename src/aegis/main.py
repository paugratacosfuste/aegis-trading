"""Aegis Trading System entry point.

Usage:
    python -m aegis.main --config configs/lab.yaml
    python -m aegis.main --config configs/backtest.yaml --backtest
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from aegis.common.config import load_config

# Load .env before anything reads env vars
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_backtest(config_path: str) -> None:
    """Run backtest from config."""
    from aegis.agents.factory import create_agents_from_config, create_default_agents
    from aegis.backtest.data_loader import download_from_binance
    from aegis.backtest.engine import BacktestEngine
    from aegis.backtest.report import print_report, save_report

    config = load_config(config_path)
    bt_cfg = config.backtest

    symbol = bt_cfg.get("symbol", "BTC/USDT")
    binance_symbol = symbol.replace("/", "")
    start = bt_cfg.get("start_date", "1 Apr, 2025")
    end = bt_cfg.get("end_date", "1 Apr, 2026")

    logger.info("Downloading historical data for %s...", symbol)
    candles = download_from_binance(
        symbol=binance_symbol,
        interval=bt_cfg.get("timeframe", "1h"),
        start_str=start,
        end_str=end,
    )
    logger.info("Downloaded %d candles", len(candles))

    # Use config-driven agents if defined, else Phase 1 defaults
    agents = (
        create_agents_from_config(config.agents)
        if config.agents
        else create_default_agents()
    )
    logger.info("Created %d agents", len(agents))

    engine = BacktestEngine(
        initial_capital=config.initial_capital,
        commission_pct=bt_cfg.get("commission_pct", 0.001),
        confidence_threshold=config.confidence_threshold,
        max_open_positions=config.max_open_positions,
        max_risk_pct=config.max_risk_per_trade,
        agents=agents,
    )

    results = engine.run(candles)
    print_report(results)
    save_report(results, "backtest_results.json")


async def run_live(config_path: str) -> None:
    """Run live/paper trading loop."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    from aegis.agents.factory import create_agents_from_config, create_default_agents
    from aegis.common.db import DatabasePool
    from aegis.data.binance_ws import BinanceWebSocketCollector
    from aegis.data.repository import MarketDataRepository

    config = load_config(config_path)
    logger.info("Starting Aegis in %s mode", config.mode)

    # Create agents from config or defaults
    agents = (
        create_agents_from_config(config.agents)
        if config.agents
        else create_default_agents()
    )
    logger.info("Created %d agents for live trading", len(agents))

    # Database
    db = DatabasePool.from_config(config.database)
    repo = MarketDataRepository(db)
    logger.info("Database pool created")

    # Binance WebSocket collector (uses production WS for market data - read-only, no auth)
    crypto_symbols = [s.replace("/", "").lower() for s in config.symbols.get("crypto", [])]
    collector = BinanceWebSocketCollector(
        repository=repo,
        symbols=crypto_symbols,
        interval="1m",
    )

    # Scheduler for periodic jobs
    scheduler = AsyncIOScheduler()

    async def signal_pipeline_job():
        logger.info("Signal pipeline tick")

    async def check_exits_job():
        logger.debug("Checking exits")

    scheduler.add_job(signal_pipeline_job, "interval", seconds=300, id="signal_pipeline")
    scheduler.add_job(check_exits_job, "interval", seconds=60, id="check_exits")
    scheduler.start()
    logger.info("Scheduler started")

    # Run WS collector as the main async task
    logger.info("Starting Binance WS collector for %s", crypto_symbols)
    try:
        await collector.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        collector.stop()
        scheduler.shutdown()
        db.close()
        logger.info("Aegis shut down.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aegis Trading System")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--backtest", action="store_true", help="Run backtest mode")
    args = parser.parse_args()

    if args.backtest:
        run_backtest(args.config)
    else:
        asyncio.run(run_live(args.config))


if __name__ == "__main__":
    main()
