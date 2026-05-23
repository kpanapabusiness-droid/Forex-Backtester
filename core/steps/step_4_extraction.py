"""Step 4 — entry-time extractability per L_PROTOCOL §2 Step 4.

For each candidate cluster from Step 3, this trains RandomForest +
LightGBM + LogisticRegression on entry-time features predicting cluster
membership. 5-fold TimeSeriesSplit per classifier; per-fold AUC;
threshold sweep at AUC-best and F1-best; permutation importance ranks
top contributors.

Lineage enforcement (per L_PROTOCOL §2 Step 4 + Amendment 2): features
tagged 'suspect' or 'unverified' are excluded from training. The
exclusion list is recorded in the extraction summary.

Output identifies, per cluster:
  - best classifier name (highest mean AUC across folds)
  - per-classifier AUC mean + per-fold AUC
  - AUC-best threshold and the precision / recall / n_trades sweep
  - top-10 features by mean permutation importance across folds

Step 5 consumes these to wire A2 (uses best classifier + AUC-best
threshold) and A6 (same classifier, sized by confidence).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit

from core.steps._classifier_defaults import (
    build_lgbm,
    build_lr,
    build_rf,
    is_lgbm_available,
)

try:
    import lightgbm as _lightgbm_mod  # type: ignore
    _LGBM_VERSION = str(_lightgbm_mod.__version__)
except ImportError:
    _LGBM_VERSION = ""

N_TS_FOLDS = 5


@dataclass(frozen=True)
class ClassifierFoldResult:
    """One classifier's per-fold output for one cluster."""

    classifier: str  # "rf" | "lgbm" | "lr"
    fold: int
    auc: float
    threshold_auc_best: float
    precision_at_threshold: float
    recall_at_threshold: float
    n_admit_at_threshold: int


@dataclass(frozen=True)
class ClusterExtraction:
    """Aggregated extraction result for one candidate cluster.

    The ``fitted_classifier_*`` fields are populated when
    :func:`run_step_4` is called with ``persistence_dir`` set. The
    fitted estimator itself is NOT held in memory — only the path to
    the joblib-pickled artefact, the sklearn / LGBM class name, and the
    feature column order at fit time. Use
    :func:`core.steps.classifier_persistence.load_classifier` to read
    back the estimator. All three fields are ``None`` when persistence
    was not requested or when the cluster did not produce a viable
    best-AUC classifier.
    """

    cluster_id: int
    n_trades: int
    excluded_features: tuple[str, ...]
    used_features: tuple[str, ...]
    classifier_fold_results: tuple[ClassifierFoldResult, ...]
    best_classifier: str
    best_classifier_mean_auc: float
    best_threshold: float
    feature_importance: pd.DataFrame  # feature, mean_importance, std_importance
    fitted_classifier_path: Path | None = None
    fitted_classifier_type: str | None = None
    fitted_classifier_feature_order: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Step4Result:
    """Output of :func:`run_step_4`."""

    per_cluster: tuple[ClusterExtraction, ...]
    extraction_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    summary_md: str


def _filter_lineage(
    feature_matrix: pd.DataFrame,
    lineage: pd.DataFrame | None,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Apply lineage-tag enforcement.

    Returns (filtered_matrix, used_features, excluded_features). When
    lineage is None, all features are used.
    """
    if lineage is None or "name" not in lineage.columns or "causal_lineage" not in lineage.columns:
        cols = tuple(feature_matrix.columns)
        return feature_matrix, cols, ()
    accepted = set(lineage[lineage["causal_lineage"] == "clean"]["name"].astype(str))
    used = tuple(c for c in feature_matrix.columns if c in accepted)
    excluded = tuple(c for c in feature_matrix.columns if c not in accepted)
    return feature_matrix[list(used)], used, excluded


def _auc_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold maximising Youden's J (TPR - FPR) on the ROC curve.

    Falls back to 0.5 if the ROC curve is degenerate.
    """
    if len(set(y_true)) < 2:
        return 0.5
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    j = tpr - fpr
    if len(j) == 0:
        return 0.5
    return float(thresh[int(np.argmax(j))])


def _prec_recall_n_at(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> tuple[float, float, int]:
    """Precision / recall / admit count at ``threshold``."""
    admit = y_score >= threshold
    if admit.sum() == 0:
        return 0.0, 0.0, 0
    tp = int(((admit == 1) & (y_true == 1)).sum())
    n_admit = int(admit.sum())
    n_pos = int((y_true == 1).sum())
    precision = tp / max(1, n_admit)
    recall = tp / max(1, n_pos)
    return float(precision), float(recall), n_admit


def _train_one_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    cv: TimeSeriesSplit,
    builder,
    name: str,
) -> tuple[list[ClassifierFoldResult], list[pd.DataFrame]]:
    """Fit one classifier across CV folds; return per-fold + importance frames."""
    fold_results: list[ClassifierFoldResult] = []
    importance_frames: list[pd.DataFrame] = []
    for k, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
            fold_results.append(ClassifierFoldResult(
                classifier=name,
                fold=k,
                auc=float("nan"),
                threshold_auc_best=0.5,
                precision_at_threshold=0.0,
                recall_at_threshold=0.0,
                n_admit_at_threshold=0,
            ))
            continue
        model = builder()
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
        thr = _auc_best_threshold(y_test, proba)
        prec, rec, n_admit = _prec_recall_n_at(y_test, proba, thr)
        fold_results.append(ClassifierFoldResult(
            classifier=name,
            fold=k,
            auc=auc,
            threshold_auc_best=thr,
            precision_at_threshold=prec,
            recall_at_threshold=rec,
            n_admit_at_threshold=n_admit,
        ))
        # Permutation importance on the test fold
        try:
            pi = permutation_importance(
                model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
            )
            importance_frames.append(pd.DataFrame({
                "feature": X.columns,
                "importance": pi.importances_mean,
                "importance_std": pi.importances_std,
                "fold": k,
                "classifier": name,
            }))
        except Exception:
            # Permutation importance can fail on some sklearn pipelines for
            # small folds — skip this fold's importance contribution.
            pass
    return fold_results, importance_frames


_BUILDERS = {"rf": build_rf, "lr": build_lr, "lgbm": build_lgbm}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _persist_best_classifier(
    *,
    persistence_dir: Path,
    cluster_id: int,
    best_clf_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    best_threshold: float,
    best_mean_auc: float,
) -> tuple[Path, str, tuple[str, ...], float, int]:
    """Refit the best-AUC algorithm on the full lineage-filtered pool,
    pickle it to disk, return (path, type_name, feature_order, in_sample_auc, n_train).

    Per chat resolution Q1 + Q4: the persisted classifier trains on
    the same data Step 4's CV iterated over (the full pool passed to
    ``run_step_4``). When the upstream Step 4 pool inherits Step 1's
    holdout window, the persisted classifier inherits the same data
    scope. Document explicitly in the manifest + caller.
    """
    builder = _BUILDERS[best_clf_name]
    model = builder()
    model.fit(X, y)
    in_sample_proba = model.predict_proba(X)[:, 1]
    if len(set(y)) >= 2:
        in_sample_auc = float(roc_auc_score(y, in_sample_proba))
    else:
        in_sample_auc = float("nan")

    persistence_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = persistence_dir / f"{int(cluster_id)}.pkl"
    joblib.dump(model, pkl_path, compress=3)

    class_name = type(model).__name__
    feat_order = tuple(X.columns)
    return pkl_path, class_name, feat_order, in_sample_auc, int(len(X))


def _write_manifest(
    *,
    persistence_dir: Path,
    arc_name: str,
    entries: dict,
    train_end: pd.Timestamp | None,
) -> None:
    """Write classifiers/manifest.json with SHA256 + provenance.

    Versions of joblib / sklearn / lightgbm captured for cross-env
    troubleshooting (dispatch Risk #1). The loader emits a UserWarning
    on mismatch but does not error — pinning is a separate concern.

    ``train_end`` is recorded as an ISO timestamp (UTC) when the
    holdout-exclusion contract was active, ``null`` otherwise. The
    field is declarative — closures reading the manifest can audit
    the data scope without re-running Step 4. The loader does not
    enforce this field; it is informational.
    """
    train_end_str: str | None = None
    if train_end is not None:
        ts = pd.Timestamp(train_end)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        train_end_str = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "arc_name": arc_name,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_end": train_end_str,
        "joblib_version": str(joblib.__version__),
        "sklearn_version": str(sklearn.__version__),
        "lightgbm_version": _LGBM_VERSION,
        "classifiers": entries,
    }
    out = persistence_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")


def run_step_4(
    trades: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    *,
    feature_lineage: pd.DataFrame | None = None,
    candidate_cluster_ids: tuple[int, ...] | None = None,
    n_ts_folds: int = N_TS_FOLDS,
    persistence_dir: Path | None = None,
    arc_name: str = "",
    train_end: pd.Timestamp | None = None,
) -> Step4Result:
    """Run Step 4 across candidate clusters.

    Parameters
    ----------
    trades
        Step 1 trades table with trade_id + entry_time (used for
        time-ordering before TimeSeriesSplit).
    feature_matrix
        Per-trade entry-time feature matrix, keyed on trade_id (or
        sharing trade_id column).
    cluster_assignments
        Step 2 output; trade_id + cluster_id.
    feature_lineage
        Optional lineage DataFrame (feature name + causal_lineage tag).
        Features tagged anything other than "clean" are excluded.
    candidate_cluster_ids
        If supplied, restrict Step 4 to these clusters. If None, run
        across every cluster that has at least 50 trades AND at least
        25 trades in each of two classes (membership / non-membership).
    n_ts_folds
        5 by default per L_PROTOCOL Step 4.
    persistence_dir
        When supplied, after CV the best-AUC algorithm is refit on the
        in-sample pool for that cluster (see ``train_end``), pickled via
        joblib to ``persistence_dir / "{cluster_id}.pkl"``, and a
        ``manifest.json`` is written alongside with SHA256 + provenance
        per :mod:`core.steps.classifier_persistence`. When ``None``,
        no classifier objects are written and ``ClusterExtraction``'s
        ``fitted_classifier_*`` fields stay ``None``.
    arc_name
        Recorded in the manifest's ``arc_name`` field for traceability.
        Defaults to empty string when persistence is not requested.
    train_end
        When supplied, restrict BOTH the 5-fold TimeSeriesSplit CV
        evaluation AND the persisted classifier refit to trades with
        ``entry_time < train_end``. This is the holdout-exclusion
        contract — Step 4's algorithm-selection and the persisted
        artefact must not see the WFO holdout window so A2 / A6 can
        be evaluated cleanly on it. ``None`` preserves backwards-compat
        behavior (Step 4 sees every trade in the pool). The orchestrator
        threads this from ``WfoStructure.holdout.oos_start`` when a
        holdout window is configured.
    """
    # Apply holdout exclusion FIRST so both CV and persistence see the
    # same restricted pool. Filtering on trades.entry_time means cluster
    # labels and features are subset together — no risk of class-balance
    # drift from a later filter.
    if train_end is not None:
        cutoff = pd.Timestamp(train_end)
        trades_in = trades.copy()
        trades_in["entry_time"] = pd.to_datetime(trades_in["entry_time"], utc=True)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        is_mask = trades_in["entry_time"] < cutoff
        trades_in = trades_in.loc[is_mask].reset_index(drop=True)
        is_trade_ids = set(trades_in["trade_id"].astype(int))
        # Filter cluster_assignments and feature_matrix to the IS subset
        cluster_assignments = cluster_assignments[
            cluster_assignments["trade_id"].astype(int).isin(is_trade_ids)
        ].reset_index(drop=True)
        if "trade_id" in feature_matrix.columns:
            feature_matrix = feature_matrix[
                feature_matrix["trade_id"].astype(int).isin(is_trade_ids)
            ].reset_index(drop=True)
        else:
            feature_matrix = feature_matrix.loc[
                feature_matrix.index.isin(is_trade_ids)
            ].copy()
        trades = trades_in
    # Time-sort trades for TimeSeriesSplit
    trades_sorted = trades.sort_values(["entry_time", "trade_id"]).reset_index(drop=True)

    if "trade_id" in feature_matrix.columns:
        fm = feature_matrix.set_index("trade_id")
    else:
        fm = feature_matrix.copy()
        fm.index.name = "trade_id"

    fm_filtered, used, excluded = _filter_lineage(fm, feature_lineage)

    # Build target per cluster
    cluster_map = cluster_assignments.set_index("trade_id")["cluster_id"].to_dict()

    cluster_ids_to_run = candidate_cluster_ids
    if cluster_ids_to_run is None:
        cluster_ids_to_run = tuple(sorted(set(cluster_map.values())))

    per_cluster: list[ClusterExtraction] = []
    all_metric_rows: list[dict] = []
    all_importance_rows: list[pd.DataFrame] = []
    manifest_entries: dict = {}

    for cid in cluster_ids_to_run:
        # Build (X, y) in time order
        ordered_tids = trades_sorted["trade_id"].astype(int).values
        # Restrict to trades present in feature matrix AND cluster assignments
        common = [int(t) for t in ordered_tids if t in fm_filtered.index and t in cluster_map]
        if len(common) < n_ts_folds * 10:
            continue
        X = fm_filtered.loc[common].copy()
        y = np.array([1 if cluster_map[t] == cid else 0 for t in common], dtype=int)
        if (y == 1).sum() < 25 or (y == 0).sum() < 25:
            continue
        # Drop any rows with NaN features (no imputation at v3.0 — bias risk)
        finite_mask = X.notna().all(axis=1).values
        X = X.iloc[finite_mask]
        y = y[finite_mask]
        if len(X) < n_ts_folds * 10:
            continue

        cv = TimeSeriesSplit(n_splits=n_ts_folds)
        fold_results: list[ClassifierFoldResult] = []
        importance_frames: list[pd.DataFrame] = []

        rf_fr, rf_imp = _train_one_classifier(X, y, cv, build_rf, "rf")
        fold_results.extend(rf_fr)
        importance_frames.extend(rf_imp)
        lr_fr, lr_imp = _train_one_classifier(X, y, cv, build_lr, "lr")
        fold_results.extend(lr_fr)
        importance_frames.extend(lr_imp)
        if is_lgbm_available():
            lgbm_fr, lgbm_imp = _train_one_classifier(X, y, cv, build_lgbm, "lgbm")
            fold_results.extend(lgbm_fr)
            importance_frames.extend(lgbm_imp)

        # Aggregate mean AUC per classifier; pick best
        by_clf: dict[str, list[float]] = {}
        for fr in fold_results:
            if np.isfinite(fr.auc):
                by_clf.setdefault(fr.classifier, []).append(fr.auc)
        mean_aucs = {c: float(np.mean(aucs)) for c, aucs in by_clf.items() if aucs}
        if not mean_aucs:
            continue
        best_clf = max(mean_aucs, key=lambda c: mean_aucs[c])
        best_mean_auc = mean_aucs[best_clf]
        # Aggregate best threshold as mean AUC-best threshold across folds
        thr_values = [
            fr.threshold_auc_best for fr in fold_results
            if fr.classifier == best_clf and np.isfinite(fr.auc)
        ]
        best_threshold = float(np.mean(thr_values)) if thr_values else 0.5

        # Aggregate importance: mean over folds + classifiers
        if importance_frames:
            all_imp = pd.concat(importance_frames, ignore_index=True)
            agg = (
                all_imp.groupby("feature")["importance"]
                .agg(["mean", "std"])
                .reset_index()
                .rename(columns={"mean": "mean_importance", "std": "std_importance"})
            )
            agg = agg.sort_values("mean_importance", ascending=False).reset_index(drop=True)
        else:
            agg = pd.DataFrame(columns=["feature", "mean_importance", "std_importance"])

        fitted_path: Path | None = None
        fitted_type: str | None = None
        fitted_feat_order: tuple[str, ...] | None = None
        if persistence_dir is not None:
            fitted_path, fitted_type, fitted_feat_order, in_sample_auc, n_train = (
                _persist_best_classifier(
                    persistence_dir=persistence_dir,
                    cluster_id=int(cid),
                    best_clf_name=best_clf,
                    X=X,
                    y=y,
                    best_threshold=best_threshold,
                    best_mean_auc=best_mean_auc,
                )
            )
            manifest_entries[str(int(cid))] = {
                "path": fitted_path.name,
                "sha256": _sha256_file(fitted_path),
                "classifier_type": fitted_type,
                "classifier_name": best_clf,
                "feature_order": list(fitted_feat_order),
                "best_threshold": best_threshold,
                "auc_in_sample": in_sample_auc,
                "auc_oos_cv5": best_mean_auc,
                "trained_on_pool_size": n_train,
            }

        per_cluster.append(ClusterExtraction(
            cluster_id=int(cid),
            n_trades=int(len(X)),
            excluded_features=excluded,
            used_features=tuple(used),
            classifier_fold_results=tuple(fold_results),
            best_classifier=best_clf,
            best_classifier_mean_auc=best_mean_auc,
            best_threshold=best_threshold,
            feature_importance=agg,
            fitted_classifier_path=fitted_path,
            fitted_classifier_type=fitted_type,
            fitted_classifier_feature_order=fitted_feat_order,
        ))
        for fr in fold_results:
            all_metric_rows.append({
                "cluster_id": int(cid),
                "classifier": fr.classifier,
                "fold": fr.fold,
                "auc": fr.auc,
                "threshold_auc_best": fr.threshold_auc_best,
                "precision_at_threshold": fr.precision_at_threshold,
                "recall_at_threshold": fr.recall_at_threshold,
                "n_admit_at_threshold": fr.n_admit_at_threshold,
            })
        for imp in importance_frames:
            imp_with_cid = imp.assign(cluster_id=int(cid))
            all_importance_rows.append(imp_with_cid)

    extraction_metrics = pd.DataFrame(all_metric_rows)
    if len(extraction_metrics) > 0:
        extraction_metrics = extraction_metrics.sort_values(
            ["cluster_id", "classifier", "fold"]
        ).reset_index(drop=True)
    feature_importance = (
        pd.concat(all_importance_rows, ignore_index=True)
        if all_importance_rows
        else pd.DataFrame(columns=["feature", "importance", "importance_std", "fold", "classifier", "cluster_id"])
    )
    if len(feature_importance) > 0:
        feature_importance = feature_importance.sort_values(
            ["cluster_id", "classifier", "fold", "feature"]
        ).reset_index(drop=True)
    if persistence_dir is not None and manifest_entries:
        _write_manifest(
            persistence_dir=persistence_dir,
            arc_name=arc_name,
            entries=manifest_entries,
            train_end=train_end,
        )
    summary_md = _render_summary_md(per_cluster, excluded)
    return Step4Result(
        per_cluster=tuple(per_cluster),
        extraction_metrics=extraction_metrics,
        feature_importance=feature_importance,
        summary_md=summary_md,
    )


def _render_summary_md(
    per_cluster: list[ClusterExtraction], excluded: tuple[str, ...]
) -> str:
    lines = ["# Step 4 — Extraction Summary", ""]
    if excluded:
        lines.append(
            f"**Lineage-excluded features** ({len(excluded)}): "
            + ", ".join(sorted(excluded))
        )
        lines.append("")
    if not per_cluster:
        lines.append("(no clusters met the minimum trade / class-balance threshold)")
        return "\n".join(lines) + "\n"
    lines.append(
        "| cluster | n | best clf | mean AUC | threshold | top-3 features |"
    )
    lines.append("|---:|---:|---|---:|---:|---|")
    for ce in per_cluster:
        top3 = (
            ", ".join(ce.feature_importance.head(3)["feature"].astype(str).tolist())
            if len(ce.feature_importance) > 0
            else "(none)"
        )
        lines.append(
            f"| {ce.cluster_id} | {ce.n_trades} | {ce.best_classifier} | "
            f"{ce.best_classifier_mean_auc:.4f} | {ce.best_threshold:.4f} | {top3} |"
        )
    return "\n".join(lines) + "\n"


def step_4_sha256(result: Step4Result) -> str:
    """Two-run determinism hash on extraction_metrics CSV."""
    payload = result.extraction_metrics.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = (
    "N_TS_FOLDS",
    "ClassifierFoldResult",
    "ClusterExtraction",
    "Step4Result",
    "run_step_4",
    "step_4_sha256",
)
