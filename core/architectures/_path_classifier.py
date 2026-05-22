"""Per-fold classifier helper shared by A3 (Pipeline DE) and A4 (Pipeline D
exits).

Both A3 and A4 retrain a fresh classifier per WFO fold using path-so-far
features. A3 trains to predict cluster membership at bar N post-signal;
A4 trains to predict "final R > 0" at every bar post-entry. The training
data structure is the same; the target differs.

L_PROTOCOL Amendment 2: random_state=42, n_jobs=1; train on IS only;
evaluate at each new signal in OOS.

Feature schema: per :mod:`core.features_path_so_far`, the locked
ENTRY_FEATURE_KEYS + PATH_FEATURE_KEYS (8 + 7 = 15 features).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from core.features_path_so_far import ALL_FEATURE_KEYS
from core.steps._classifier_defaults import build_rf


@dataclass(frozen=True)
class PathClassifierFit:
    """One fit of the path-so-far classifier.

    ``threshold`` is the AUC-best threshold from the IS evaluation
    (Youden's J on the IS data itself for in-fold estimation; A3/A4
    apply this to OOS at runtime).
    """

    model: Any  # trained classifier (predict_proba interface)
    threshold: float
    fit_auc: float  # in-sample AUC for diagnostic logging
    feature_order: tuple[str, ...]


def fit_path_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    feature_order: tuple[str, ...] = ALL_FEATURE_KEYS,
) -> PathClassifierFit:
    """Train a single RandomForest on path-so-far features.

    Determinism: random_state=42, n_jobs=1 (via build_rf defaults).
    """
    if X.shape[0] < 50 or len(set(y.tolist())) < 2:
        # Insufficient data — return a degenerate fit that admits nothing.
        # Caller can detect via threshold > 1.0 sentinel.
        from sklearn.dummy import DummyClassifier
        d = DummyClassifier(strategy="constant", constant=0)
        d.fit(np.zeros((2, len(feature_order))), [0, 1])
        return PathClassifierFit(
            model=d,
            threshold=1.5,  # impossible to exceed; effectively zero admit
            fit_auc=float("nan"),
            feature_order=feature_order,
        )
    X_ordered = X[list(feature_order)]
    finite = X_ordered.notna().all(axis=1).values
    X_clean = X_ordered.loc[finite]
    y_clean = y[finite]
    if X_clean.shape[0] < 50 or len(set(y_clean.tolist())) < 2:
        from sklearn.dummy import DummyClassifier
        d = DummyClassifier(strategy="constant", constant=0)
        d.fit(np.zeros((2, len(feature_order))), [0, 1])
        return PathClassifierFit(
            model=d,
            threshold=1.5,
            fit_auc=float("nan"),
            feature_order=feature_order,
        )
    model = build_rf()
    model.fit(X_clean.values, y_clean)
    proba = model.predict_proba(X_clean.values)[:, 1]
    auc = float(roc_auc_score(y_clean, proba))
    fpr, tpr, thresh = roc_curve(y_clean, proba)
    j = tpr - fpr
    threshold = float(thresh[int(np.argmax(j))]) if len(j) else 0.5
    return PathClassifierFit(
        model=model,
        threshold=threshold,
        fit_auc=auc,
        feature_order=tuple(feature_order),
    )


def predict_admit(
    fit: PathClassifierFit,
    feature_row: Mapping[str, float],
    *,
    threshold_override: float | None = None,
) -> tuple[bool, float]:
    """Predict admit/reject + probability for one feature row.

    Returns (admit, probability). Missing or non-finite features ->
    (False, 0.0).
    """
    try:
        row = np.array(
            [float(feature_row[k]) for k in fit.feature_order],
            dtype=np.float64,
        ).reshape(1, -1)
    except (KeyError, TypeError, ValueError):
        return False, 0.0
    if not np.all(np.isfinite(row)):
        return False, 0.0
    try:
        proba = float(fit.model.predict_proba(row)[0, 1])
    except Exception:
        return False, 0.0
    threshold = threshold_override if threshold_override is not None else fit.threshold
    return proba >= threshold, proba


__all__ = (
    "PathClassifierFit",
    "fit_path_classifier",
    "predict_admit",
)
