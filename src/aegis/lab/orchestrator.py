"""Lab orchestrator: runs multiple cohorts on shared signal data."""

import logging
from datetime import datetime

from aegis.agents.base import BaseAgent
from aegis.agents.factory import create_agents_from_config, create_default_agents
from aegis.common.types import AgentSignal, MarketDataPoint
from aegis.lab.cohort_runner import CohortRunner
from aegis.lab.repository import CohortRepository
from aegis.lab.types import CohortStatus, StrategyCohort

logger = logging.getLogger(__name__)


class LabOrchestrator:
    """Manages multiple cohort runners, dispatches shared signals."""

    def __init__(
        self,
        repository: CohortRepository | None = None,
        agents: list[BaseAgent] | None = None,
        cohorts: list[StrategyCohort] | None = None,
        agents_config: dict | None = None,
    ):
        self._repository = repository
        self._runners: dict[str, CohortRunner] = {}
        self._agents = agents or []

        # Create agents from config if provided and no agents passed
        if not self._agents and agents_config:
            self._agents = create_agents_from_config(agents_config)
        elif not self._agents:
            self._agents = create_default_agents()

        # Initialize runners from provided cohorts
        if cohorts:
            for cohort in cohorts:
                self._runners[cohort.cohort_id] = CohortRunner(cohort)

    def initialize_from_db(self) -> None:
        """Load active cohorts from database and create runners."""
        if self._repository is None:
            return
        cohorts = self._repository.get_active_cohorts()
        for cohort in cohorts:
            if cohort.cohort_id not in self._runners:
                self._runners[cohort.cohort_id] = CohortRunner(cohort)
        logger.info("Lab initialized with %d active cohorts", len(self._runners))

    def tick(
        self,
        candles_by_symbol: dict[str, list[MarketDataPoint]],
        current_prices: dict[str, float],
        current_time: datetime | None = None,
    ) -> dict[str, list]:
        """Main loop: generate signals once, dispatch to all cohort runners.

        Returns {cohort_id: [trade_decisions]} for any trades generated.
        """
        # Step 1: Generate signals from all agents (shared across cohorts)
        all_signals: list[AgentSignal] = []
        for agent in self._agents:
            for symbol, candles in candles_by_symbol.items():
                try:
                    signal = agent.generate_signal(symbol, candles)
                    if signal.confidence > 0.001:
                        all_signals.append(signal)
                except Exception:
                    logger.debug("Agent %s failed on %s", agent.agent_id, symbol)

        # Step 2: Group signals by symbol
        signals_by_symbol: dict[str, list[AgentSignal]] = {}
        for sig in all_signals:
            signals_by_symbol.setdefault(sig.symbol, []).append(sig)

        # Step 3: Dispatch to each cohort runner
        results: dict[str, list] = {}
        for cohort_id, runner in self._runners.items():
            decisions = []
            for symbol, sym_signals in signals_by_symbol.items():
                # Filter to cohort's universe if specified
                universe = runner.cohort.config.universe
                if universe and symbol not in universe:
                    continue
                decision = runner.process_signals(sym_signals, current_prices)
                if decision is not None:
                    decisions.append(decision)
            if decisions:
                results[cohort_id] = decisions
            # Record equity snapshot
            runner.record_equity(current_prices)

        # Step 4: Check exits for all cohorts
        for cohort_id, runner in self._runners.items():
            closed = runner.check_exits(current_prices)
            if closed:
                results.setdefault(cohort_id, []).extend(closed)

        return results

    def add_cohort(self, cohort: StrategyCohort) -> None:
        """Add a new cohort runner."""
        self._runners[cohort.cohort_id] = CohortRunner(cohort)
        if self._repository:
            self._repository.insert_cohort(cohort)
        logger.info("Added cohort %s (%s)", cohort.cohort_id, cohort.name)

    def remove_cohort(self, cohort_id: str) -> None:
        """Remove a cohort runner."""
        self._runners.pop(cohort_id, None)
        logger.info("Removed cohort %s", cohort_id)

    def get_runner(self, cohort_id: str) -> CohortRunner | None:
        return self._runners.get(cohort_id)

    def get_active_runners(self) -> dict[str, CohortRunner]:
        return dict(self._runners)

    @property
    def agents(self) -> list[BaseAgent]:
        return self._agents
