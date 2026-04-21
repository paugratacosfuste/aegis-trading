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
    from aegis.backtest.data_loader import (
        download_from_binance,
        download_from_yfinance,
        is_crypto_symbol,
    )
    from aegis.backtest.engine import BacktestEngine
    from aegis.backtest.report import print_report, save_report

    config = load_config(config_path)
    bt_cfg = config.backtest

    # Support single symbol (backward compat) or multiple symbols
    symbols = bt_cfg.get("symbols", [bt_cfg.get("symbol", "BTC/USDT")])
    if isinstance(symbols, str):
        symbols = [symbols]

    start = bt_cfg.get("start_date", "2025-04-01")
    end = bt_cfg.get("end_date", "2026-04-01")
    interval = bt_cfg.get("timeframe", "1h")

    candles_by_symbol: dict[str, list] = {}
    for sym in symbols:
        logger.info("Downloading historical data for %s...", sym)
        if is_crypto_symbol(sym):
            binance_sym = sym.replace("/", "")
            candles = download_from_binance(
                symbol=binance_sym,
                interval=interval,
                start_str=start,
                end_str=end,
            )
        else:
            candles = download_from_yfinance(
                symbol=sym,
                start=start,
                end=end,
                interval=interval,
            )
        logger.info("Downloaded %d candles for %s", len(candles), sym)
        candles_by_symbol[sym] = candles

    # Download macro data if macro agents are enabled.
    # FRED is primary (richer, more reliable); yfinance is the fallback for
    # offline/network-gated environments.
    enabled_types = config.ensemble.get("enabled_types")
    macro_provider = None
    if enabled_types is None or "macro" in enabled_types:
        from aegis.agents.macro.providers import BacktestMacroProvider
        from aegis.data.fred_loader import download_fred_macro_data
        from aegis.data.macro_data_loader import download_macro_data

        logger.info("Downloading FRED macro data for %s to %s...", start, end)
        macro_snapshots = download_fred_macro_data(start=start, end=end)
        if not macro_snapshots:
            logger.warning("FRED returned nothing — falling back to yfinance macro loader")
            macro_snapshots = download_macro_data(start=start, end=end)
        if macro_snapshots:
            macro_provider = BacktestMacroProvider(macro_snapshots)
            logger.info("Macro provider loaded with %d snapshots", len(macro_snapshots))

    # Download geopolitical events if geopolitical agents are enabled.
    geo_provider = None
    if enabled_types is None or "geopolitical" in enabled_types:
        from aegis.agents.geopolitical.providers import BacktestGeopoliticalProvider
        from aegis.data.gdelt_loader import download_gdelt_events

        logger.info("Downloading GDELT events for %s to %s...", start, end)
        events = download_gdelt_events(start=start, end=end)
        if events:
            geo_provider = BacktestGeopoliticalProvider(events)
            logger.info("Geo provider loaded with %d events", len(events))
        else:
            logger.warning("GDELT returned no events — geopolitical agents will be neutral")

    # Use config-driven agents if defined, else Phase 1 defaults
    agents = (
        create_agents_from_config(
            config.agents, enabled_types=enabled_types,
            macro_provider=macro_provider,
            geo_provider=geo_provider,
        )
        if config.agents
        else create_default_agents()
    )
    logger.info("Created %d agents (enabled_types=%s)", len(agents), enabled_types or "all")

    engine = BacktestEngine(
        initial_capital=config.initial_capital,
        commission_pct=bt_cfg.get("commission_pct", 0.001),
        confidence_threshold=config.confidence_threshold,
        max_open_positions=config.max_open_positions,
        max_risk_pct=config.max_risk_per_trade,
        agents=agents,
        macro_provider=macro_provider,
        geo_provider=geo_provider,
    )

    if len(candles_by_symbol) == 1:
        results = engine.run(next(iter(candles_by_symbol.values())))
    else:
        results = engine.run_multi(candles_by_symbol)
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

    # Register feedback loop jobs
    from aegis.feedback.scheduler import register_feedback_jobs
    register_feedback_jobs(scheduler, db, config.feedback)

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


async def run_lab(config_path: str) -> None:
    """Run lab mode: parallel paper trading with cohort system."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    from aegis.agents.factory import create_agents_from_config, create_default_agents
    from aegis.common.db import DatabasePool
    from aegis.data.binance_ws import BinanceWebSocketCollector
    from aegis.data.repository import MarketDataRepository
    from aegis.lab.config_templates import get_default_templates
    from aegis.lab.orchestrator import LabOrchestrator
    from aegis.lab.repository import CohortRepository

    config = load_config(config_path)
    lab_cfg = config.lab
    logger.info("Starting Aegis in LAB mode")

    # Create shared agents
    agents = (
        create_agents_from_config(config.agents)
        if config.agents
        else create_default_agents()
    )
    logger.info("Created %d shared agents for lab", len(agents))

    # Database
    db = DatabasePool.from_config(config.database)
    repo = CohortRepository(db)
    logger.info("Database pool created")

    # Initialize orchestrator
    templates_to_use = lab_cfg.get("templates", list("ABCDEFGHIJ"))
    cohorts = get_default_templates()
    cohorts = [c for c in cohorts if c.cohort_id.replace("cohort_", "") in templates_to_use]

    orchestrator = LabOrchestrator(
        repository=repo,
        agents=agents,
        cohorts=cohorts,
        agents_config=config.agents,
    )
    logger.info("Lab orchestrator initialized with %d cohorts", len(cohorts))

    # Binance WS for market data
    crypto_symbols = [s.replace("/", "").lower() for s in config.symbols.get("crypto", [])]
    market_repo = MarketDataRepository(db)
    collector = BinanceWebSocketCollector(
        repository=market_repo,
        symbols=crypto_symbols,
        interval="1m",
    )

    # Scheduler
    scheduler = AsyncIOScheduler()
    signal_interval = lab_cfg.get("signal_interval_sec", 300)
    exit_interval = lab_cfg.get("exit_check_interval_sec", 60)

    async def lab_signal_job():
        logger.info("Lab signal pipeline tick (%d cohorts)", len(orchestrator.get_active_runners()))

    async def lab_exit_job():
        logger.debug("Lab exit check")

    scheduler.add_job(lab_signal_job, "interval", seconds=signal_interval, id="lab_signal")
    scheduler.add_job(lab_exit_job, "interval", seconds=exit_interval, id="lab_exits")

    # Register feedback loop jobs
    from aegis.feedback.scheduler import register_feedback_jobs
    register_feedback_jobs(scheduler, db, config.feedback)

    scheduler.start()
    logger.info("Lab scheduler started (signal=%ds, exits=%ds)", signal_interval, exit_interval)

    try:
        await collector.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        collector.stop()
        scheduler.shutdown()
        db.close()
        logger.info("Lab mode shut down.")


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Aegis Trading System")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--backtest", action="store_true", help="Run backtest mode")
    parser.add_argument("--lab", action="store_true", help="Run lab mode (parallel paper trading)")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.backtest and args.lab:
        parser.error("Cannot use --backtest and --lab together")
    elif args.backtest:
        run_backtest(args.config)
    elif args.lab:
        asyncio.run(run_lab(args.config))
    else:
        asyncio.run(run_live(args.config))


if __name__ == "__main__":
    main()
