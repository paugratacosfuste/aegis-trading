"""Price-based HMM regime detector for backtesting.

Trains a 4-state GaussianHMM on price features (returns, volatility)
so regime detection works without external macro data.

State labeling maps to ensemble weight keys:
    bull       = positive drift, low volatility
    recovery   = positive drift, high volatility
    transition = negative drift, low volatility
    bear       = negative drift, high volatility
"""

import logging

import numpy as np

from aegis.common.types import MarketDataPoint

logger = logging.getLogger(__name__)

_MIN_TRAINING_BARS = 336  # 14 days of hourly data minimum


class PriceRegimeDetector:
    """HMM regime detector trained on price data alone."""

    def __init__(
        self,
        n_states: int = 4,
        warmup_bars: int = 60 * 24,
        lookback_window: int = 14 * 24,
    ):
        self.n_states = n_states
        self.warmup_bars = warmup_bars
        self._lookback = lookback_window
        self._model = None
        self._state_map: dict[int, str] = {}
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def state_labels(self) -> dict[int, str]:
        return dict(self._state_map)

    def train(self, candles: list[MarketDataPoint]) -> bool:
        """Fit HMM on candle data. Returns True if training succeeded."""
        features = self._compute_features(candles)
        if features.shape[0] < _MIN_TRAINING_BARS:
            logger.warning(
                "Not enough data for HMM training: %d rows (need %d)",
                features.shape[0],
                _MIN_TRAINING_BARS,
            )
            return False

        try:
            from hmmlearn.hmm import GaussianHMM

            # Standardize features for numerical stability
            self._feature_mean = np.mean(features, axis=0)
            self._feature_std = np.std(features, axis=0)
            self._feature_std[self._feature_std < 1e-10] = 1.0
            scaled = (features - self._feature_mean) / self._feature_std

            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=200,
                random_state=42,
            )
            model.fit(scaled)
            self._model = model
            self._label_states(features, scaled)
            logger.info(
                "HMM trained on %d samples. State labels: %s",
                features.shape[0],
                self._state_map,
            )
            return True
        except Exception:
            logger.exception("HMM training failed")
            return False

    def predict(self, candles: list[MarketDataPoint]) -> str:
        """Predict current regime from recent candles."""
        regime, _ = self.predict_with_confidence(candles)
        return regime

    def predict_with_confidence(
        self, candles: list[MarketDataPoint]
    ) -> tuple[str, float]:
        """Predict regime with confidence score."""
        if self._model is None or self._feature_mean is None:
            return "normal", 0.0

        features = self._compute_features(candles)
        if features.shape[0] < 2:
            return "normal", 0.0

        try:
            scaled = (features - self._feature_mean) / self._feature_std
            state = int(self._model.predict(scaled)[-1])
            probs = self._model.predict_proba(scaled)[-1]
            confidence = float(probs[state])
            regime = self._state_map.get(state, "normal")
            return regime, confidence
        except Exception:
            logger.exception("HMM prediction failed")
            return "normal", 0.0

    def _compute_features(self, candles: list[MarketDataPoint]) -> np.ndarray:
        """Extract features: daily_return, rolling_14d_vol, rolling_14d_return."""
        closes = np.array([c.close for c in candles], dtype=np.float64)
        if len(closes) < 2:
            return np.empty((0, 3))

        # Log returns
        returns = np.diff(np.log(closes))

        # Rolling window for volatility and cumulative return
        window = min(14 * 24, len(returns) // 2)
        if window < 10:
            window = 10

        rows = []
        for i in range(window, len(returns)):
            window_rets = returns[i - window : i]
            daily_ret = returns[i]
            rolling_vol = float(np.std(window_rets))
            rolling_ret = float(np.sum(window_rets))
            rows.append([daily_ret, rolling_vol, rolling_ret])

        return np.array(rows, dtype=np.float64) if rows else np.empty((0, 3))

    def _label_states(self, features: np.ndarray, scaled: np.ndarray) -> None:
        """Post-hoc label HMM states by mean return and volatility.

        Sorts states into quadrants:
            high return + low vol  -> bull
            high return + high vol -> recovery
            low return + low vol   -> transition
            low return + high vol  -> bear
        """
        states = self._model.predict(scaled)
        state_stats: dict[int, tuple[float, float]] = {}

        for s in range(self.n_states):
            mask = states == s
            if not np.any(mask):
                state_stats[s] = (0.0, 0.0)
                continue
            mean_ret = float(np.mean(features[mask, 2]))  # rolling_14d_return
            mean_vol = float(np.mean(features[mask, 1]))  # rolling_14d_vol
            state_stats[s] = (mean_ret, mean_vol)

        # Find median return and volatility across states
        rets = [v[0] for v in state_stats.values()]
        vols = [v[1] for v in state_stats.values()]
        med_ret = float(np.median(rets))
        med_vol = float(np.median(vols))

        for s, (ret, vol) in state_stats.items():
            if ret >= med_ret and vol < med_vol:
                self._state_map[s] = "bull"
            elif ret >= med_ret and vol >= med_vol:
                self._state_map[s] = "recovery"
            elif ret < med_ret and vol < med_vol:
                self._state_map[s] = "transition"
            else:
                self._state_map[s] = "bear"

        logger.info(
            "HMM state stats: %s",
            {self._state_map[s]: f"ret={r:.4f} vol={v:.4f}" for s, (r, v) in state_stats.items()},
        )
