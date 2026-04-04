"""Weekly walk-forward XGBoost retraining pipeline: Speed 2 of the feedback loop.

Trains a model on a rolling window, validates on held-out data with a gap,
and accepts only if it outperforms the current model by a margin.
"""

import calendar
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

_DEFAULT_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "random_state": 42,
}


def calculate_walk_forward_dates(
    train_end: date,
    train_years: int = 3,
    gap_days: int = 30,
    validation_days: int = 90,
) -> dict[str, date]:
    """Calculate walk-forward date boundaries.

    Returns dict with train_start, train_end, val_start, val_end.
    Ensures no look-ahead leakage: val_start > train_end + gap.
    """
    target_year = train_end.year - train_years
    try:
        train_start = date(target_year, train_end.month, train_end.day)
    except ValueError:
        # Handle leap day: clamp to last day of target month
        last_day = calendar.monthrange(target_year, train_end.month)[1]
        train_start = date(target_year, train_end.month, last_day)
    val_start = train_end + timedelta(days=gap_days)
    val_end = val_start + timedelta(days=validation_days)
    return {
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
    }


def check_acceptance(
    new_auc: float,
    previous_auc: float | None,
    threshold: float = 1.02,
) -> bool:
    """Check if new model should be accepted.

    First model (previous_auc is None) is always accepted.
    Otherwise, new_auc must strictly exceed previous_auc * threshold.
    """
    if previous_auc is None:
        return True
    return new_auc > previous_auc * threshold


def train_and_validate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    model_dir: Path,
    model_type: str,
    xgb_params: dict | None = None,
    min_train_samples: int = 500,
) -> dict:
    """Train XGBoost model and validate.

    Returns dict with val_auc, model_path, shap_importances, predictions.
    Raises ValueError if insufficient training samples.
    """
    if len(X_train) < min_train_samples:
        raise ValueError(
            f"Insufficient training data: {len(X_train)} < {min_train_samples}"
        )

    params = dict(_DEFAULT_XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)

    # Train
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)

    # Validate
    val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)

    # Ensure model directory exists and save model as JSON via booster
    model_dir.mkdir(parents=True, exist_ok=True)
    version = date.today().isoformat()
    filename = f"{model_type}_{version}.json"
    model_path = model_dir / filename
    model.get_booster().save_model(str(model_path))

    # SHAP feature importance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    # Mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importances = {
        name: float(importance)
        for name, importance in zip(feature_names, mean_abs_shap)
    }

    return {
        "val_auc": val_auc,
        "model_path": filename,
        "shap_importances": shap_importances,
        "predictions": val_proba,
    }
