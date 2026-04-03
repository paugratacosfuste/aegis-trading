"""Macro agent 05: Cross-asset HMM regime detector.

Uses hmmlearn GaussianHMM with 4 hidden states.
Optional: falls back to neutral if hmmlearn is not installed or model not trained.
"""

import logging
from pathlib import Path

from aegis.agents.macro.base_macro import BaseMacroAgent
from aegis.agents.registry import register_agent
from aegis.common.types import AgentSignal, MarketDataPoint

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM  # noqa: F401
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    logger.warning("hmmlearn not available — macro_05 HMM agent will return neutral")

# State labels for a 4-state HMM
_STATE_LABELS = {
    0: "bull",        # Low vol, positive drift
    1: "transition",  # Medium vol, mixed drift
    2: "bear",        # High vol, negative drift
    3: "recovery",    # High vol, positive drift
}


@register_agent("macro", "hmm_regime")
class HmmRegimeAgent(BaseMacroAgent):
    """HMM-based regime detector. Requires a pre-trained model pickle.

    Config params:
        model_path: Path to trained HMM model pickle (default: models/hmm_regime.pkl)
    """

    def __init__(self, agent_id, config, provider=None):
        super().__init__(agent_id, config, provider)
        self._model = None
        self._load_model()

    # Sandbox model loading to the models/ directory within the project
    _ALLOWED_MODEL_DIR = Path(__file__).resolve().parents[4] / "models"

    def _load_model(self) -> None:
        if not _HMM_AVAILABLE:
            return

        raw_path = self.config.get("model_path", "models/hmm_regime.pkl")
        path = (self._ALLOWED_MODEL_DIR.parent / raw_path).resolve()

        # Prevent path traversal outside the sanctioned models/ directory
        if not str(path).startswith(str(self._ALLOWED_MODEL_DIR.resolve())):
            logger.error("Rejected model path outside models/ directory: %s", path)
            return

        if path.suffix != ".pkl":
            logger.error("Rejected model path with unexpected extension: %s", path)
            return

        if not path.exists():
            logger.warning("HMM model not found at %s — agent will return neutral", path)
            return

        try:
            import joblib
            self._model = joblib.load(path)
            logger.info("Loaded HMM model from %s", path)
        except Exception:
            logger.exception("Failed to load HMM model from %s", path)

    def generate_signal(self, symbol: str, candles: list[MarketDataPoint]) -> AgentSignal:
        if self._model is None:
            return self._build_macro_signal(
                symbol=symbol,
                regime="unknown",
                regime_confidence=0.0,
                reasoning={"note": "HMM model not available"},
            )

        snap = self._provider.get_macro_snapshot()
        if snap is None:
            return self._build_macro_signal(
                symbol=symbol,
                regime="unknown",
                regime_confidence=0.0,
                reasoning={"note": "No macro data available"},
            )

        # Build feature vector from macro snapshot
        # In production, this would use rolling window of daily features
        try:
            import numpy as np
            features = np.array([[
                snap.yield_spread,
                snap.vix / 100.0,  # Normalize
                snap.dxy / 100.0 - 1.0,  # Center around 0
                snap.cpi_latest / 10.0,
            ]])
            state = int(self._model.predict(features)[-1])
            regime = _STATE_LABELS.get(state, "unknown")

            # Get state probabilities for confidence
            probs = self._model.predict_proba(features)[-1]
            confidence = float(probs[state])
        except Exception:
            logger.exception("HMM prediction failed")
            regime = "unknown"
            confidence = 0.0

        return self._build_macro_signal(
            symbol=symbol,
            regime=regime,
            regime_confidence=confidence,
            reasoning={"hmm_state": regime},
        )
