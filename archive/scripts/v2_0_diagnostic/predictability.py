"""v2.0 archetype diagnostic — Step 4 (b): per-archetype predictability AUC.

For each (dataset, K, archetype), train a logistic regression on the
cross-dataset entry feature set to predict "is this trade in this
archetype" (binary target). 5-fold CV ROC-AUC.

LogisticRegression(random_state=42, max_iter=1000) on standardised features
(StandardScaler within a Pipeline to avoid leakage between folds).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = (
    "body_to_range_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "range_to_atr_14",
    "ret_5bar_atr",
    "ret_20bar_atr",
    "pos_in_20bar_range",
    "rsi_14",
)


def _make_estimator() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(random_state=42, max_iter=1000)),
    ])


def per_archetype_auc(
    entry_features: pd.DataFrame,
    assignments: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """Returns DataFrame[archetype_id, auc_mean, auc_std, auc_fold_1..5]."""
    df = entry_features.merge(assignments, on="trade_id", how="inner", validate="one_to_one")
    df = df.dropna(subset=list(FEATURES))
    X = df[list(FEATURES)].astype("float64").to_numpy()

    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for arch in range(k):
        y = (df["archetype_id"] == arch).astype("int32").to_numpy()
        n_pos = int(y.sum())
        # cross_val_score requires both classes per fold for ROC-AUC.
        if n_pos < 5 or n_pos > len(y) - 5:
            rows.append({
                "archetype_id": arch,
                "auc_mean": float("nan"),
                "auc_std":  float("nan"),
                "auc_fold_1": float("nan"),
                "auc_fold_2": float("nan"),
                "auc_fold_3": float("nan"),
                "auc_fold_4": float("nan"),
                "auc_fold_5": float("nan"),
                "n_pos": n_pos,
                "n_total": int(len(y)),
            })
            continue
        scores = cross_val_score(
            _make_estimator(), X, y, cv=skf, scoring="roc_auc", n_jobs=1,
        )
        rows.append({
            "archetype_id": arch,
            "auc_mean": float(np.mean(scores)),
            "auc_std":  float(np.std(scores)),
            "auc_fold_1": float(scores[0]),
            "auc_fold_2": float(scores[1]),
            "auc_fold_3": float(scores[2]),
            "auc_fold_4": float(scores[3]),
            "auc_fold_5": float(scores[4]),
            "n_pos": n_pos,
            "n_total": int(len(y)),
        })
    return pd.DataFrame(rows)
