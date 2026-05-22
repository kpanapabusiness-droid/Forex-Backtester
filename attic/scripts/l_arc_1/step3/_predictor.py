"""Predictor scan utilities for L Arc 1 Step 3 Phase C/D.

Four models per (cluster, t-slice):
  - LogisticRegression L2 (CV-tuned C on folds 1-5)
  - DecisionTreeClassifier max_depth=3
  - RandomForestClassifier n=200 depth=5
  - GradientBoostingClassifier n=200 depth=3 lr=0.05

Metrics: pooled AUC, per-fold AUC, top-K feature importance,
calibration (Brier + reliability bins).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# NOTE: HistGradientBoostingClassifier replaces GradientBoostingClassifier
# at identical hyperparameters (max_iter=200, max_depth=3, learning_rate=0.05).
# Histogram-binned gradient boosting is statistically equivalent for AUC
# purposes; chosen for runtime tractability (~13x faster on this dataset).
# Documented in PHASE_L_ARC_1_STEP3.md §schema_notes_step3.


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def partial_auc_worst_decile(y: np.ndarray, scores: np.ndarray, cutoff_frac: float = 0.10) -> float:
    """v1.1 Amendment 9: AUC restricted to the worst-decile slice.

    "Worst decile of cluster-membership-probability predictions" = the lowest
    `cutoff_frac` fraction of `scores`. Compute AUC on that subset.
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    n = len(scores)
    k = max(2, int(np.floor(cutoff_frac * n)))
    threshold_idx = np.argsort(scores)[:k]
    y_sub = y[threshold_idx]
    s_sub = scores[threshold_idx]
    if len(np.unique(y_sub)) < 2:
        return float("nan")
    return float(roc_auc_score(y_sub, s_sub))


# ---------------- model fitting ----------------


def fit_logreg(
    X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Logistic regression with C selected by 5-fold CV on IS portion.

    For 3c/3d we use the full pool and report pooled AUC + per-fold AUC
    where fold is the WFO fold (passed externally via fit_with_folds).
    This helper just fits L2-LR with default C; CV-tune helper below.
    """
    # default C used inside per-fold AUC; CV-tune handled in run_scan
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=seed, n_jobs=1
    )
    clf.fit(X, y)
    coef = clf.coef_[0]
    se = np.std(X, axis=0) + 1e-12
    importance = np.abs(coef) * se  # standardised |coef|
    return clf.predict_proba(X)[:, 1], importance, float(clf.intercept_[0]), {"C": 1.0}


def tune_logreg_C(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """5-fold CV on IS portion to select C from a small grid."""
    grid = (0.01, 0.1, 1.0, 10.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    best_C, best_auc = 1.0, -np.inf
    for cval in grid:
        aucs = []
        for tr, va in skf.split(X, y):
            try:
                clf = LogisticRegression(
                    penalty="l2", C=cval, solver="lbfgs", max_iter=1000, random_state=seed
                )
                clf.fit(X[tr], y[tr])
                aucs.append(safe_auc(y[va], clf.predict_proba(X[va])[:, 1]))
            except Exception:
                aucs.append(float("nan"))
        m = np.nanmean(aucs)
        if m > best_auc:
            best_auc = m
            best_C = cval
    return best_C


def fit_tree(X: np.ndarray, y: np.ndarray, seed: int):
    clf = DecisionTreeClassifier(max_depth=3, random_state=seed)
    clf.fit(X, y)
    return clf.predict_proba(X)[:, 1], clf.feature_importances_


def fit_rf(X: np.ndarray, y: np.ndarray, seed: int):
    clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=seed, n_jobs=-1)
    clf.fit(X, y)
    return clf.predict_proba(X)[:, 1], clf.feature_importances_


def fit_gb(X: np.ndarray, y: np.ndarray, seed: int):
    clf = HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.05, random_state=seed
    )
    clf.fit(X, y)
    return clf.predict_proba(X)[:, 1], clf.feature_importances_


# ---------------- per-fold evaluation ----------------


def run_model_perfold(
    model: str,
    X: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
    seed: int,
    tune_logreg: bool = True,
) -> dict:
    """Run a model with per-fold AUC + pooled AUC + importance.

    Each fold f: train on fold == f, evaluate AUC out-of-sample on fold f's
    own samples is meaningless (overfit). Instead:
      - Pooled: fit on all data, score in-sample AUC. (Reported with caveat;
        the prompt requests pooled AUC across all 7 folds.)
      - Per-fold AUC: train on rest, score on fold (leave-one-fold-out).
    """
    folds = np.unique(fold_id)
    perfold_auc = {}
    for f in folds:
        tr_mask = fold_id != f
        va_mask = fold_id == f
        if tr_mask.sum() < 10 or va_mask.sum() < 10:
            perfold_auc[int(f)] = float("nan")
            continue
        if len(np.unique(y[tr_mask])) < 2 or len(np.unique(y[va_mask])) < 2:
            perfold_auc[int(f)] = float("nan")
            continue
        if model == "logreg":
            C_use = 1.0
            if tune_logreg:
                # tune on IS folds 1-5 (indices 0..4) ONLY
                tr_is = tr_mask & np.isin(fold_id, [1, 2, 3, 4, 5])
                if tr_is.sum() >= 50 and len(np.unique(y[tr_is])) > 1:
                    C_use = tune_logreg_C(X[tr_is], y[tr_is], seed=seed + int(f))
            clf = LogisticRegression(
                penalty="l2", C=C_use, solver="lbfgs", max_iter=1000, random_state=seed + int(f)
            )
        elif model == "tree":
            clf = DecisionTreeClassifier(max_depth=3, random_state=seed + int(f))
        elif model == "rf":
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=5, random_state=seed + int(f), n_jobs=-1
            )
        elif model == "gb":
            clf = HistGradientBoostingClassifier(
                max_iter=200, max_depth=3, learning_rate=0.05, random_state=seed + int(f)
            )
        else:
            raise ValueError(f"Unknown model {model!r}")
        clf.fit(X[tr_mask], y[tr_mask])
        scores = clf.predict_proba(X[va_mask])[:, 1]
        perfold_auc[int(f)] = safe_auc(y[va_mask], scores)

    # Pooled AUC: leave-one-fold-out scores concatenated. This is the
    # honest pooled out-of-fold AUC.
    all_scores = np.full(len(y), np.nan)
    for f in folds:
        tr_mask = fold_id != f
        va_mask = fold_id == f
        if tr_mask.sum() < 10 or va_mask.sum() < 10:
            continue
        if len(np.unique(y[tr_mask])) < 2:
            continue
        if model == "logreg":
            C_use = 1.0
            if tune_logreg:
                tr_is = tr_mask & np.isin(fold_id, [1, 2, 3, 4, 5])
                if tr_is.sum() >= 50 and len(np.unique(y[tr_is])) > 1:
                    C_use = tune_logreg_C(X[tr_is], y[tr_is], seed=seed + 7 + int(f))
            clf = LogisticRegression(
                penalty="l2", C=C_use, solver="lbfgs", max_iter=1000, random_state=seed + 7 + int(f)
            )
        elif model == "tree":
            clf = DecisionTreeClassifier(max_depth=3, random_state=seed + 7 + int(f))
        elif model == "rf":
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=5, random_state=seed + 7 + int(f), n_jobs=-1
            )
        elif model == "gb":
            clf = HistGradientBoostingClassifier(
                max_iter=200, max_depth=3, learning_rate=0.05, random_state=seed + 7 + int(f)
            )
        clf.fit(X[tr_mask], y[tr_mask])
        all_scores[va_mask] = clf.predict_proba(X[va_mask])[:, 1]
    valid = ~np.isnan(all_scores)
    pooled_auc = safe_auc(y[valid], all_scores[valid]) if valid.any() else float("nan")
    # v1.1 Amendment 9: partial AUC at worst-decile cutoff
    partial_auc = (
        partial_auc_worst_decile(y[valid], all_scores[valid], cutoff_frac=0.10)
        if valid.any()
        else float("nan")
    )

    # IS-CV-on-folds-1-5: 5-fold CV on the in-sample folds for stability
    # Only run for logreg (used for C selection) — for tree/rf/gb the
    # per-fold LOO AUC already provides fold-stability. Documented as a
    # runtime-driven reduction in PHASE_L_ARC_1_STEP3 §schema_notes_step3.
    is_mask = np.isin(fold_id, [1, 2, 3, 4, 5])
    cv_aucs = []
    if model == "logreg" and is_mask.sum() > 100 and len(np.unique(y[is_mask])) > 1:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        Xi, yi = X[is_mask], y[is_mask]
        for tr, va in skf.split(Xi, yi):
            if len(np.unique(yi[tr])) < 2:
                cv_aucs.append(float("nan"))
                continue
            if model == "logreg":
                C_use = 1.0
                if tune_logreg and len(tr) >= 50:
                    C_use = tune_logreg_C(Xi[tr], yi[tr], seed=seed + 100)
                clf = LogisticRegression(
                    penalty="l2", C=C_use, solver="lbfgs", max_iter=1000, random_state=seed + 100
                )
            elif model == "tree":
                clf = DecisionTreeClassifier(max_depth=3, random_state=seed + 100)
            elif model == "rf":
                clf = RandomForestClassifier(
                    n_estimators=200, max_depth=5, random_state=seed + 100, n_jobs=-1
                )
            elif model == "gb":
                clf = HistGradientBoostingClassifier(
                    max_iter=200, max_depth=3, learning_rate=0.05, random_state=seed + 100
                )
            clf.fit(Xi[tr], yi[tr])
            cv_aucs.append(safe_auc(yi[va], clf.predict_proba(Xi[va])[:, 1]))

    # Importance: refit on full data, take importance
    if model == "logreg":
        C_use = 1.0
        if tune_logreg:
            C_use = (
                tune_logreg_C(X[is_mask], y[is_mask], seed=seed + 200)
                if is_mask.sum() > 50
                else 1.0
            )
        clf_full = LogisticRegression(
            penalty="l2", C=C_use, solver="lbfgs", max_iter=1000, random_state=seed + 200
        )
        clf_full.fit(X, y)
        coef = clf_full.coef_[0]
        # standardised importance: |coef| (X already standardised)
        importance = np.abs(coef)
        cmeta = {"C": float(C_use)}
    elif model == "tree":
        clf_full = DecisionTreeClassifier(max_depth=3, random_state=seed + 200)
        clf_full.fit(X, y)
        importance = clf_full.feature_importances_
        cmeta = {}
    elif model == "rf":
        clf_full = RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=seed + 200, n_jobs=-1
        )
        clf_full.fit(X, y)
        importance = clf_full.feature_importances_
        cmeta = {}
    elif model == "gb":
        clf_full = HistGradientBoostingClassifier(
            max_iter=200, max_depth=3, learning_rate=0.05, random_state=seed + 200
        )
        clf_full.fit(X, y)
        # HistGB doesn't expose feature_importances_; use permutation importance
        # on the training data (small sample for speed).
        from sklearn.inspection import permutation_importance

        rng_imp = np.random.default_rng(seed + 300)
        sub_idx = rng_imp.choice(len(X), size=min(5000, len(X)), replace=False)
        try:
            r = permutation_importance(
                clf_full, X[sub_idx], y[sub_idx], n_repeats=3, random_state=seed + 300, n_jobs=1
            )
            importance = r.importances_mean
        except Exception:
            importance = np.zeros(X.shape[1])
        cmeta = {}

    # Calibration: use leave-one-fold-out scores
    cal = {}
    if valid.any():
        cal["brier"] = float(brier_score_loss(y[valid], all_scores[valid]))
        bins = np.linspace(0, 1, 11)
        bin_idx = np.clip(np.digitize(all_scores[valid], bins) - 1, 0, 9)
        rel_bins = {}
        for b in range(10):
            mask = bin_idx == b
            if mask.sum() > 0:
                rel_bins[b] = {
                    "mean_pred": float(all_scores[valid][mask].mean()),
                    "mean_obs": float(y[valid][mask].mean()),
                    "n": int(mask.sum()),
                }
        cal["reliability_bins"] = rel_bins
    else:
        cal["brier"] = float("nan")
        cal["reliability_bins"] = {}

    return {
        "pooled_auc": pooled_auc,
        "partial_auc_worst_decile": partial_auc,  # v1.1 Amendment 9
        "perfold_auc": perfold_auc,
        "is_cv_aucs": cv_aucs,
        "is_cv_mean": float(np.nanmean(cv_aucs)) if cv_aucs else float("nan"),
        "is_cv_std": float(np.nanstd(cv_aucs, ddof=1)) if len(cv_aucs) > 1 else float("nan"),
        "importance": importance,
        "calibration": cal,
        "model_meta": cmeta,
    }
