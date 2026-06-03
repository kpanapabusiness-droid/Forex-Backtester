"""Shared model + CV utilities for the predictability investigation.

Deterministic random_state=42 throughout. 5-fold stratified ROC-AUC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def cv_splitter(seed: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def fit_rf(X: np.ndarray, y: np.ndarray) -> tuple[float, float, list[float]]:
    """5-fold CV ROC-AUC for Random Forest with project defaults.

    Returns (mean, std, [fold scores]). Returns NaNs if either class is
    too rare to stratify across 5 folds.
    """
    if not _class_balance_ok(y):
        return float("nan"), float("nan"), [float("nan")] * 5
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=1,  # single-thread for determinism
    )
    scores = cross_val_score(rf, X, y, cv=cv_splitter(), scoring="roc_auc", n_jobs=1)
    return float(np.mean(scores)), float(np.std(scores)), [float(s) for s in scores]


def fit_logistic(X: np.ndarray, y: np.ndarray, max_iter: int = 2000) -> tuple[float, float, list[float]]:
    if not _class_balance_ok(y):
        return float("nan"), float("nan"), [float("nan")] * 5
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(random_state=42, max_iter=max_iter)),
    ])
    scores = cross_val_score(pipe, X, y, cv=cv_splitter(), scoring="roc_auc", n_jobs=1)
    return float(np.mean(scores)), float(np.std(scores)), [float(s) for s in scores]


def _class_balance_ok(y: np.ndarray) -> bool:
    n_pos = int(y.sum())
    n_neg = int(len(y) - y.sum())
    return n_pos >= 5 and n_neg >= 5


def rf_feature_importance(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Fit RF on the full data and return per-feature importance (Gini)."""
    if not _class_balance_ok(y):
        return pd.DataFrame({"feature": feature_names, "importance": [float("nan")] * len(feature_names)})
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=1,
    )
    rf.fit(X, y)
    return pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
