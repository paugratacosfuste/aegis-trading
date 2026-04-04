"""Tests for feature selection module."""

import numpy as np
import pandas as pd
import pytest

from aegis.feedback.feature_selection import select_features


class TestSelectFeatures:
    def _make_data(self, n=200, n_features=10, seed=42):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(
            rng.standard_normal((n, n_features)),
            columns=[f"feat_{i}" for i in range(n_features)],
        )
        # Feature 0 is predictive
        y = pd.Series((X["feat_0"] > 0).astype(int), name="target")
        return X, y

    def test_returns_top_k(self):
        X, y = self._make_data(n_features=10)
        selected = select_features(X, y, top_k=5)
        assert len(selected) == 5
        assert all(isinstance(f, str) for f in selected)

    def test_predictive_feature_selected(self):
        X, y = self._make_data(n_features=10)
        selected = select_features(X, y, top_k=3)
        assert "feat_0" in selected

    def test_high_missing_dropped(self):
        X, y = self._make_data(n_features=5)
        # Make feat_1 have 40% missing
        X.loc[X.index[:80], "feat_1"] = np.nan
        selected = select_features(X, y, top_k=5, max_missing_pct=0.30)
        assert "feat_1" not in selected

    def test_low_variance_dropped(self):
        X, y = self._make_data(n_features=5)
        # Make feat_2 constant
        X["feat_2"] = 1.0
        selected = select_features(X, y, top_k=5)
        assert "feat_2" not in selected

    def test_high_correlation_dropped(self):
        X, y = self._make_data(n_features=5)
        # Make feat_3 a near-copy of feat_0
        X["feat_3"] = X["feat_0"] + np.random.default_rng(99).normal(0, 0.001, len(X))
        selected = select_features(X, y, top_k=5, max_correlation=0.95)
        # At most one of feat_0, feat_3 survives
        both = {"feat_0", "feat_3"}
        assert len(both & set(selected)) <= 1

    def test_fewer_features_than_k(self):
        X, y = self._make_data(n_features=3)
        selected = select_features(X, y, top_k=10)
        assert len(selected) <= 3

    def test_empty_dataframe(self):
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        selected = select_features(X, y, top_k=5)
        assert selected == []

    def test_all_missing_columns_dropped(self):
        X = pd.DataFrame({"a": [np.nan] * 100, "b": np.random.randn(100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        selected = select_features(X, y, top_k=5, max_missing_pct=0.30)
        assert "a" not in selected
