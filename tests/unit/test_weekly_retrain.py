"""Tests for weekly model retraining pipeline."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aegis.feedback.weekly_retrain import (
    calculate_walk_forward_dates,
    check_acceptance,
    train_and_validate,
)


class TestWalkForwardDates:
    def test_standard_dates(self):
        dates = calculate_walk_forward_dates(
            train_end=date(2026, 4, 1),
            train_years=3,
            gap_days=30,
            validation_days=90,
        )
        assert dates["train_start"] == date(2023, 4, 1)
        assert dates["train_end"] == date(2026, 4, 1)
        assert dates["val_start"] == date(2026, 5, 1)
        assert dates["val_end"] == date(2026, 7, 30)

    def test_custom_window(self):
        dates = calculate_walk_forward_dates(
            train_end=date(2026, 1, 1),
            train_years=1,
            gap_days=14,
            validation_days=60,
        )
        assert dates["train_start"] == date(2025, 1, 1)
        assert dates["val_start"] == date(2026, 1, 15)

    def test_no_look_ahead(self):
        """Validation start must be strictly after train end + gap."""
        dates = calculate_walk_forward_dates(
            train_end=date(2026, 4, 1),
            train_years=3,
            gap_days=30,
            validation_days=90,
        )
        assert dates["val_start"] > dates["train_end"]
        gap = (dates["val_start"] - dates["train_end"]).days
        assert gap >= 30


class TestCheckAcceptance:
    def test_accepted(self):
        # 0.62 > 0.60 * 1.02 = 0.612 -> True
        assert check_acceptance(0.62, 0.60, threshold=1.02) is True

    def test_rejected(self):
        # 0.61 > 0.60 * 1.02 = 0.612 -> False
        assert check_acceptance(0.61, 0.60, threshold=1.02) is False

    def test_first_model_always_accepted(self):
        assert check_acceptance(0.51, None, threshold=1.02) is True

    def test_marginal_improvement(self):
        # Exactly at threshold -> not accepted (strict >)
        assert check_acceptance(0.612, 0.60, threshold=1.02) is False

    def test_large_improvement(self):
        assert check_acceptance(0.80, 0.60, threshold=1.02) is True


class TestTrainAndValidate:
    def _make_dataset(self, n=600, seed=42):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame({
            f"feat_{i}": rng.standard_normal(n) for i in range(5)
        })
        # feat_0 is predictive
        y = pd.Series((X["feat_0"] + rng.normal(0, 0.5, n) > 0).astype(int))
        return X, y

    def test_train_produces_valid_result(self, tmp_path):
        X_train, y_train = self._make_dataset(n=600, seed=42)
        X_val, y_val = self._make_dataset(n=200, seed=99)
        features = list(X_train.columns)

        result = train_and_validate(
            X_train=X_train[features],
            y_train=y_train,
            X_val=X_val[features],
            y_val=y_val,
            feature_names=features,
            model_dir=tmp_path,
            model_type="test_model",
        )
        assert result["val_auc"] > 0.5  # Better than random
        assert result["model_path"].endswith(".json")
        assert Path(tmp_path / result["model_path"]).exists()

    def test_shap_importances_returned(self, tmp_path):
        X_train, y_train = self._make_dataset(n=600, seed=42)
        X_val, y_val = self._make_dataset(n=200, seed=99)
        features = list(X_train.columns)

        result = train_and_validate(
            X_train=X_train[features],
            y_train=y_train,
            X_val=X_val[features],
            y_val=y_val,
            feature_names=features,
            model_dir=tmp_path,
            model_type="test_model",
        )
        assert "shap_importances" in result
        assert len(result["shap_importances"]) == len(features)
        # SHAP importances are non-negative (absolute mean)
        assert all(v >= 0 for v in result["shap_importances"].values())

    def test_model_save_load_roundtrip(self, tmp_path):
        import xgboost as xgb

        X_train, y_train = self._make_dataset(n=600, seed=42)
        features = list(X_train.columns)

        result = train_and_validate(
            X_train=X_train[features],
            y_train=y_train,
            X_val=X_train[features],  # Just for roundtrip test
            y_val=y_train,
            feature_names=features,
            model_dir=tmp_path,
            model_type="roundtrip",
        )
        # Load model back via Booster
        model_path = tmp_path / result["model_path"]
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        dmat = xgb.DMatrix(X_train[features])
        preds = booster.predict(dmat)
        assert preds.shape[0] == len(X_train)

    def test_insufficient_samples_raises(self, tmp_path):
        X = pd.DataFrame({"f": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        with pytest.raises(ValueError, match="Insufficient"):
            train_and_validate(
                X_train=X, y_train=y, X_val=X, y_val=y,
                feature_names=["f"], model_dir=tmp_path,
                model_type="test", min_train_samples=500,
            )
