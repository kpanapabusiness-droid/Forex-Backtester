"""Predictor scan utilities for L Arc 2 Step 3 Phase C/D (mirrors arc 1).

Four models per (cluster, t-slice):
  - LogisticRegression L2 (CV-tuned C on folds 1-5)
  - DecisionTreeClassifier max_depth=3
  - RandomForestClassifier n=200 depth=5
  - HistGradientBoostingClassifier max_iter=200 depth=3 lr=0.05
    (sklearn HistGB substitutes for GradientBoostingClassifier for runtime;
    AUC equivalent.)

Metrics: pooled AUC (leave-one-fold-out concatenated), per-fold AUC, partial AUC
at worst-decile cutoff (Amendment 9), top-K feature importance, calibration
(Brier + reliability bins).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from . import _common as C


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def partial_auc_worst_decile(y: np.ndarray, scores: np.ndarray,
                             cutoff_frac: float = 0.10) -> float:
    """Amendment 9: AUC restricted to the worst-decile slice."""
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


def tune_logreg_C(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    grid = (0.01, 0.1, 1.0, 10.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    best_C, best_auc = 1.0, -np.inf
    for cval in grid:
        aucs = []
        for tr, va in skf.split(X, y):
            try:
                clf = LogisticRegression(penalty="l2", C=cval, solver="lbfgs",
                                         max_iter=1000, random_state=seed)
                clf.fit(X[tr], y[tr])
                aucs.append(safe_auc(y[va], clf.predict_proba(X[va])[:, 1]))
            except Exception:
                aucs.append(float("nan"))
        m = np.nanmean(aucs)
        if m > best_auc:
            best_auc = m
            best_C = cval
    return best_C


def _make_clf(model: str, seed: int):
    if model == "logreg":
        return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                  max_iter=1000, random_state=seed)
    if model == "tree":
        return DecisionTreeClassifier(max_depth=3, random_state=seed)
    if model == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=5,
                                      random_state=seed, n_jobs=-1)
    if model == "gb":
        return HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                              learning_rate=0.05,
                                              random_state=seed)
    raise ValueError(f"Unknown model {model!r}")


def run_model_perfold(model: str, X: np.ndarray, y: np.ndarray,
                      fold_id: np.ndarray, seed: int,
                      tune_logreg: bool = True) -> dict:
    """Run a model with per-fold leave-one-fold-out AUC + pooled AUC + importance."""
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
                tr_is = tr_mask & np.isin(fold_id, [1, 2, 3, 4, 5])
                if tr_is.sum() >= 50 and len(np.unique(y[tr_is])) > 1:
                    C_use = tune_logreg_C(X[tr_is], y[tr_is], seed=seed + int(f))
            clf = LogisticRegression(penalty="l2", C=C_use, solver="lbfgs",
                                     max_iter=1000, random_state=seed + int(f))
        else:
            clf = _make_clf(model, seed + int(f))
        clf.fit(X[tr_mask], y[tr_mask])
        scores = clf.predict_proba(X[va_mask])[:, 1]
        perfold_auc[int(f)] = safe_auc(y[va_mask], scores)

    # Pooled out-of-fold scores
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
            clf = LogisticRegression(penalty="l2", C=C_use, solver="lbfgs",
                                     max_iter=1000, random_state=seed + 7 + int(f))
        else:
            clf = _make_clf(model, seed + 7 + int(f))
        clf.fit(X[tr_mask], y[tr_mask])
        all_scores[va_mask] = clf.predict_proba(X[va_mask])[:, 1]
    valid = ~np.isnan(all_scores)
    pooled_auc = safe_auc(y[valid], all_scores[valid]) if valid.any() else float("nan")
    partial_auc = (
        partial_auc_worst_decile(y[valid], all_scores[valid], cutoff_frac=C.PARTIAL_AUC_DECILE_CUTOFF)
        if valid.any() else float("nan")
    )

    is_mask = np.isin(fold_id, [1, 2, 3, 4, 5])
    cv_aucs = []
    if model == "logreg" and is_mask.sum() > 100 and len(np.unique(y[is_mask])) > 1:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        Xi, yi = X[is_mask], y[is_mask]
        for tr, va in skf.split(Xi, yi):
            if len(np.unique(yi[tr])) < 2:
                cv_aucs.append(float("nan"))
                continue
            C_use = 1.0
            if tune_logreg and len(tr) >= 50:
                C_use = tune_logreg_C(Xi[tr], yi[tr], seed=seed + 100)
            clf = LogisticRegression(penalty="l2", C=C_use, solver="lbfgs",
                                     max_iter=1000, random_state=seed + 100)
            clf.fit(Xi[tr], yi[tr])
            cv_aucs.append(safe_auc(yi[va], clf.predict_proba(Xi[va])[:, 1]))

    # Importance (refit on full data)
    if model == "logreg":
        C_use = 1.0
        if tune_logreg:
            C_use = tune_logreg_C(X[is_mask], y[is_mask], seed=seed + 200) if is_mask.sum() > 50 else 1.0
        clf_full = LogisticRegression(penalty="l2", C=C_use, solver="lbfgs",
                                      max_iter=1000, random_state=seed + 200)
        clf_full.fit(X, y)
        coef = clf_full.coef_[0]
        importance = np.abs(coef)
        cmeta = {"C": float(C_use)}
    elif model == "tree":
        clf_full = DecisionTreeClassifier(max_depth=3, random_state=seed + 200)
        clf_full.fit(X, y)
        importance = clf_full.feature_importances_
        cmeta = {}
    elif model == "rf":
        clf_full = RandomForestClassifier(n_estimators=200, max_depth=5,
                                          random_state=seed + 200, n_jobs=-1)
        clf_full.fit(X, y)
        importance = clf_full.feature_importances_
        cmeta = {}
    elif model == "gb":
        clf_full = HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                                  learning_rate=0.05,
                                                  random_state=seed + 200)
        clf_full.fit(X, y)
        from sklearn.inspection import permutation_importance
        rng_imp = np.random.default_rng(seed + 300)
        sub_idx = rng_imp.choice(len(X), size=min(2000, len(X)), replace=False)
        try:
            r = permutation_importance(clf_full, X[sub_idx], y[sub_idx],
                                       n_repeats=3, random_state=seed + 300, n_jobs=1)
            importance = r.importances_mean
        except Exception:
            importance = np.zeros(X.shape[1])
        cmeta = {}

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
        "partial_auc_worst_decile": partial_auc,
        "perfold_auc": perfold_auc,
        "is_cv_aucs": cv_aucs,
        "is_cv_mean": float(np.nanmean(cv_aucs)) if cv_aucs else float("nan"),
        "is_cv_std": float(np.nanstd(cv_aucs, ddof=1)) if len(cv_aucs) > 1 else float("nan"),
        "importance": importance,
        "calibration": cal,
        "model_meta": cmeta,
    }
