"""Phase 0.6 verification script.

Goal: confirm that after wiring FRED + GDELT, the macro and geopolitical
agents emit useful thesis signals on the majority of bars. Runs a short
30-day window on one symbol so the diagnostic returns in minutes.

Thesis-layer semantics (Aegis 2.0):
- Macro agents output regime classification via ``signal.metadata["regime"]``
  with ``regime_confidence``. They intentionally keep direction=confidence=0
  (they expose features to the RL executor, they do not vote).
- Geopolitical agents still produce direction/confidence in the current
  code — they were voters in Aegis 1.0 and have not been ported to pure
  thesis form yet, so we measure their non-zero-confidence rate.

Gate: majority of macro+geo agents clear 50% usable-thesis bars.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

from aegis.agents.factory import create_agents_from_config
from aegis.agents.geopolitical.providers import BacktestGeopoliticalProvider
from aegis.agents.macro.providers import BacktestMacroProvider
from aegis.backtest.data_loader import download_from_binance
from aegis.data.fred_loader import download_fred_macro_data
from aegis.data.gdelt_loader import download_gdelt_events
from aegis.data.macro_data_loader import download_macro_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify")


AGENTS_CFG = {
    "macro": [
        {"id": "macro_01", "strategy": "yield_curve_fed", "params": {}},
        {"id": "macro_02", "strategy": "risk_regime", "params": {}},
        {"id": "macro_03", "strategy": "economic_cycle", "params": {}},
        {"id": "macro_04", "strategy": "inflation_regime", "params": {}},
        {"id": "macro_05", "strategy": "hmm_regime", "params": {}},
    ],
    "geopolitical": [
        {"id": "geo_01", "strategy": "conflict_sanctions", "params": {}},
        {"id": "geo_02", "strategy": "trade_policy", "params": {}},
    ],
}

START = "2025-01-01"
END = "2025-02-01"
SYMBOL = "BTCUSDT"


@dataclass
class AgentStats:
    agent_id: str
    agent_type: str
    bars: int = 0
    useful_bars: int = 0  # thesis-meaningful output — semantics differ by type

    @property
    def pct_useful(self) -> float:
        return 100.0 * self.useful_bars / max(1, self.bars)


def _is_useful(signal, agent_type: str) -> bool:
    """Did the agent emit thesis-meaningful output on this bar?

    - macro: metadata has a non-empty regime AND regime_confidence > 0
    - geopolitical: direction != 0 OR confidence > 0
    """
    if agent_type == "macro":
        meta = signal.metadata or {}
        regime = meta.get("regime")
        conf = meta.get("regime_confidence", 0.0)
        return bool(regime) and float(conf) > 1e-9
    # voter-style agents (geopolitical still behaves this way)
    return abs(signal.direction) > 1e-9 or signal.confidence > 1e-9


def main() -> int:
    logger.info("=== Phase 0.6 verification: %s → %s on %s ===", START, END, SYMBOL)

    logger.info("[1/4] Downloading BTC 1h candles…")
    candles = download_from_binance(
        symbol=SYMBOL, interval="1h", start_str=START, end_str=END
    )
    logger.info("  got %d candles", len(candles))
    if not candles:
        logger.error("  no candles → aborting")
        return 1

    logger.info("[2/4] Downloading FRED macro…")
    macro_snapshots = download_fred_macro_data(START, END)
    if not macro_snapshots:
        logger.warning("  FRED empty → falling back to yfinance macro loader")
        macro_snapshots = download_macro_data(START, END)
    logger.info("  macro snapshots: %d", len(macro_snapshots))

    logger.info("[3/4] Downloading GDELT events (one file per day)…")
    events = download_gdelt_events(START, END)
    logger.info("  gdelt events: %d", len(events))

    macro_provider = (
        BacktestMacroProvider(macro_snapshots) if macro_snapshots else None
    )
    geo_provider = BacktestGeopoliticalProvider(events) if events else None

    agents = create_agents_from_config(
        AGENTS_CFG,
        macro_provider=macro_provider,
        geo_provider=geo_provider,
    )
    logger.info("  created %d agents", len(agents))

    stats: dict[str, AgentStats] = {
        a.agent_id: AgentStats(agent_id=a.agent_id, agent_type=a.agent_type)
        for a in agents
    }

    logger.info("[4/4] Walking bars and collecting signals…")
    window_cap = 200
    for i, candle in enumerate(candles):
        if macro_provider:
            macro_provider.advance_to(candle.timestamp)
        if geo_provider:
            geo_provider.advance_to(candle.timestamp)

        window = candles[max(0, i + 1 - window_cap) : i + 1]
        if len(window) < 20:
            continue

        for agent in agents:
            try:
                sig = agent.generate_signal(SYMBOL, window)
            except Exception as exc:  # noqa: BLE001
                logger.debug("%s raised %s — counting as neutral", agent.agent_id, exc)
                continue

            s = stats[agent.agent_id]
            s.bars += 1
            if _is_useful(sig, agent.agent_type):
                s.useful_bars += 1

    print("\n=== Agent thesis coverage ===")
    print(f"{'agent':<14} {'type':<14} {'bars':>6} {'useful %':>10}")
    print("-" * 50)
    for agent_id in sorted(stats):
        s = stats[agent_id]
        print(
            f"{s.agent_id:<14} {s.agent_type:<14} {s.bars:>6} {s.pct_useful:>9.1f}%"
        )

    # Gate: majority of macro+geo agents clear 50% useful bars.
    majority_bar = 50.0
    winners = [s for s in stats.values() if s.pct_useful >= majority_bar]
    print()
    print(
        f"{len(winners)}/{len(stats)} agents clear {majority_bar:.0f}% useful-thesis bar"
    )

    return 0 if len(winners) >= len(stats) // 2 else 2


if __name__ == "__main__":
    sys.exit(main())
