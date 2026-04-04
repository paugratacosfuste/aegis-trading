"""Feature selection for model retraining pipeline.

Uses mutual information scoring after filtering out high-missing,
low-variance, and highly-correlated features.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "mutual_information",
    top_k: int = 30,
    max_missing_pct: float = 0.30,
    min_variance: float = 0.001,
    max_correlation: float = 0.95,
) -> list[str]:
    """Select top-k features after filtering.

    Steps:
        1. Drop columns with > max_missing_pct fraction missing.
        2. Impute remaining NaN with column median.
        3. Drop columns with variance < min_variance.
        4. Drop one of each pair with correlation > max_correlation.
        5. Score remaining features by mutual information with target.
        6. Return top-k feature names.
    """
    if X.empty or len(X.columns) == 0:
        return []

    candidates = X.copy()

    # Step 1: Drop high-missing columns
    missing_pct = candidates.isnull().mean()
    keep_cols = missing_pct[missing_pct <= max_missing_pct].index.tolist()
    candidates = candidates[keep_cols]

    if candidates.empty:
        return []

    # Step 2: Impute remaining NaN with column median
    candidates = candidates.fillna(candidates.median())

    # Step 3: Drop low-variance columns
    variances = candidates.var()
    keep_cols = variances[variances >= min_variance].index.tolist()
    candidates = candidates[keep_cols]

    if candidates.empty:
        return []

    # Step 4: Drop highly correlated features
    corr_matrix = candidates.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )
    to_drop = set()
    for col in upper.columns:
        if any(upper[col] > max_correlation):
            to_drop.add(col)
    candidates = candidates.drop(columns=list(to_drop))

    if candidates.empty:
        return []

    # Step 5: Score by mutual information
    # Align y with candidates index
    y_aligned = y.loc[candidates.index]
    mi_scores = mutual_info_classif(candidates, y_aligned, random_state=42)
    scored = sorted(
        zip(candidates.columns, mi_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # Step 6: Return top-k
    k = min(top_k, len(scored))
    return [name for name, _ in scored[:k]]
