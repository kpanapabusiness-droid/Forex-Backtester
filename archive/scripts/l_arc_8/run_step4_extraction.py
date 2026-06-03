"""Arc 8 Step 4 — extraction.

Per L_PROTOCOL §2 Step 4 + dispatch §"Step 4":

  Per candidate cluster (binary target = cluster membership):
    - RF + LGBM + Logistic at Appendix A defaults
    - 5-fold TimeSeriesSplit, per-fold AUC
    - Threshold sweep at AUC-best and F1-best
    - Permutation importance per feature

  Per cluster, identify:
    - Best classifier (highest mean OOS AUC) -> A2 + A6 consume this at Step 5
    - AUC-best threshold -> A2 + A6 consume this
    - Top 10 features by permutation importance

Reads ``step_1/features.parquet`` + ``step_3/capturability.csv`` to find
candidate clusters. Writes ``step_4/extraction_metrics.csv``,
``feature_importance.csv``, ``extraction_summary.md``, ``classifiers/
<cluster_id>_<model>.pkl``, ``manifest.json``.

Causal lineage discipline (per L_PROTOCOL Step 4 mechanic 7): features
tagged ``suspect`` are kept in the training matrix but the best-classifier
+ threshold are surfaced in the summary alongside the lineage report so
Step 6 can downgrade or kill candidates whose load-bearing features fail
the producer audit.
"""

from __future__ import annotations

import pickle
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

from core.determinism import seed_everything, write_text_deterministic
from core.manifest import write_manifest
from scripts.l_arc_8.shared import RESULTS_ROOT

STEP_DIR: Path = RESULTS_ROOT / "step_4"
N_SPLITS: int = 5
RANDOM_STATE: int = 42
N_JOBS: int = 1


def _build_classifiers() -> dict[str, object]:
    """Appendix A locked defaults."""
    out: dict[str, object] = {
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
        ),
        "logreg": LogisticRegression(
            penalty="l2", C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=N_JOBS,
        ),
    }
    if _HAS_LGBM:
        out["lgbm"] = LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            min_child_samples=50, random_state=RANDOM_STATE, n_jobs=N_JOBS,
            verbose=-1,
        )
    return out


def _threshold_sweep(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """At AUC-best and F1-best thresholds, report precision/recall/F1/trade-count."""
    sweep = []
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (proba >= thr).astype(int)
        if pred.sum() == 0:
            sweep.append((thr, 0.0, 0.0, 0.0, 0))
            continue
        p, r, f, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        sweep.append((thr, p, r, f, int(pred.sum())))
    sweep_arr = np.array(sweep)
    auc_best_thr = sweep_arr[np.argmax(sweep_arr[:, 1] + sweep_arr[:, 2]), 0]  # max P+R proxy
    f1_best_idx = int(np.argmax(sweep_arr[:, 3]))
    f1_best_thr = float(sweep_arr[f1_best_idx, 0])
    # AUC-best threshold: pick threshold maximising Youden's J = TPR - FPR; this
    # corresponds to the operating point with best separation under the ROC.
    j_sweep = []
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (proba >= thr).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        j_sweep.append((thr, tpr - fpr))
    j_arr = np.array(j_sweep)
    auc_best_thr = float(j_arr[np.argmax(j_arr[:, 1]), 0])
    return {
        "auc_best_threshold": auc_best_thr,
        "f1_best_threshold": f1_best_thr,
        "auc_best_precision": float(sweep_arr[np.argmin(np.abs(sweep_arr[:, 0] - auc_best_thr)), 1]),
        "auc_best_recall": float(sweep_arr[np.argmin(np.abs(sweep_arr[:, 0] - auc_best_thr)), 2]),
        "auc_best_trade_count": int(sweep_arr[np.argmin(np.abs(sweep_arr[:, 0] - auc_best_thr)), 4]),
        "f1_best_precision": float(sweep_arr[f1_best_idx, 1]),
        "f1_best_recall": float(sweep_arr[f1_best_idx, 2]),
        "f1_best_f1": float(sweep_arr[f1_best_idx, 3]),
        "f1_best_trade_count": int(sweep_arr[f1_best_idx, 4]),
    }


def _train_eval_one_cluster(
    X: np.ndarray, y: np.ndarray, feature_names: list[str],
    n_splits: int = N_SPLITS,
) -> dict:
    """5-fold TimeSeriesSplit per classifier. Returns per-model metrics."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    classifiers = _build_classifiers()
    per_model: dict[str, dict] = {}
    # OOS predictions accumulated across folds for threshold sweep
    for name, clf in classifiers.items():
        fold_aucs: list[float] = []
        oos_y: list[np.ndarray] = []
        oos_proba: list[np.ndarray] = []
        for tr_idx, te_idx in tscv.split(X):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            # logreg needs scaling
            if name == "logreg":
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)
                X_te_s = sc.transform(X_te)
            else:
                X_tr_s, X_te_s = X_tr, X_te
            clf_fold = clf.__class__(**clf.get_params())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf_fold.fit(X_tr_s, y_tr)
            proba = clf_fold.predict_proba(X_te_s)[:, 1]
            if len(np.unique(y_te)) < 2:
                # AUC undefined; skip but record
                continue
            auc = float(roc_auc_score(y_te, proba))
            fold_aucs.append(auc)
            oos_y.append(y_te)
            oos_proba.append(proba)
        # aggregate
        if len(fold_aucs) == 0:
            per_model[name] = {"mean_auc": float("nan"), "std_auc": float("nan"), "fold_aucs": []}
            continue
        full_y = np.concatenate(oos_y)
        full_proba = np.concatenate(oos_proba)
        thr = _threshold_sweep(full_y, full_proba)
        per_model[name] = {
            "mean_auc": float(np.mean(fold_aucs)),
            "std_auc": float(np.std(fold_aucs)),
            "fold_aucs": fold_aucs,
            **thr,
        }

    # Permutation importance: train best model on full data, run permutation on it
    best_model_name = max(per_model.keys(), key=lambda k: per_model[k]["mean_auc"])
    best_clf_template = classifiers[best_model_name]
    # Fit on full data
    if best_model_name == "logreg":
        sc = StandardScaler()
        X_full = sc.fit_transform(X)
    else:
        sc = None
        X_full = X
    best_clf = best_clf_template.__class__(**best_clf_template.get_params())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_clf.fit(X_full, y)
    try:
        perm = permutation_importance(
            best_clf, X_full, y, n_repeats=5, random_state=RANDOM_STATE, n_jobs=N_JOBS,
            scoring="roc_auc",
        )
        importance = sorted(
            zip(feature_names, perm.importances_mean, perm.importances_std),
            key=lambda t: -t[1],
        )
    except Exception:
        importance = [(n, 0.0, 0.0) for n in feature_names]
    return {
        "per_model": per_model,
        "best_model_name": best_model_name,
        "best_classifier": best_clf,
        "scaler": sc,
        "permutation_importance": importance,
    }


def main() -> Path:
    seed_everything(42)
    t0 = time.perf_counter()
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    (STEP_DIR / "classifiers").mkdir(exist_ok=True)
    step1 = RESULTS_ROOT / "step_1"
    step2 = RESULTS_ROOT / "step_2"
    step3 = RESULTS_ROOT / "step_3"
    feat = pd.read_parquet(step1 / "features.parquet")
    pool = pd.read_parquet(step1 / "pool.parquet")
    assignments = pd.read_parquet(step2 / "cluster_assignments.parquet")
    cap = pd.read_csv(step3 / "capturability.csv")

    # Identify candidate clusters
    candidate_cluster_ids = cap.loc[cap["candidate"], "cluster_id"].astype(int).tolist()
    if not candidate_cluster_ids:
        # Per L_PROTOCOL §2 Step 3 failure-diag: proceed on highest-composite cluster anyway
        candidate_cluster_ids = [int(cap.sort_values("capturability_composite", ascending=False).iloc[0]["cluster_id"])]
        print(f"[step4] No candidates pass §3 floors; proceeding on highest-composite cluster: {candidate_cluster_ids}")
    print(f"[step4] candidate clusters: {candidate_cluster_ids}")

    # Build training matrix: rows = pool ordered by signal_time, features = 27 v3 catalogue
    pool = pool.merge(
        assignments[["trade_id", "cluster_primary"]], on="trade_id", how="left",
    ).sort_values("signal_time").reset_index(drop=True)
    feat = feat.sort_values("signal_time").reset_index(drop=True)
    # Align — drop trade_id+pair+signal_time columns from features matrix
    feature_cols = [c for c in feat.columns if c not in {"trade_id", "pair", "signal_time"}]
    X_full = feat[feature_cols].to_numpy(dtype=float).copy()
    # Drop columns that are entirely NaN (e.g. if a feature class went missing).
    all_nan = np.isnan(X_full).all(axis=0)
    if all_nan.any():
        keep_cols = ~all_nan
        dropped = [c for c, k in zip(feature_cols, keep_cols) if not k]
        print(f"[step4] dropping all-NaN feature columns: {dropped}")
        X_full = X_full[:, keep_cols]
        feature_cols = [c for c, k in zip(feature_cols, keep_cols) if k]
    # Replace remaining NaN with column median (cross_pair features may have NaN at extremes).
    col_medians = np.nanmedian(X_full, axis=0)
    inds = np.where(np.isnan(X_full))
    X_full[inds] = np.take(col_medians, inds[1])

    # Ensure pool and feat align row-for-row by trade_id
    assert (pool["trade_id"].values == feat["trade_id"].values).all(), \
        "pool and features misaligned by trade_id"

    extraction_rows: list[dict] = []
    feat_importance_rows: list[dict] = []
    classifier_summaries: list[dict] = []
    for cid in candidate_cluster_ids:
        y = (pool["cluster_primary"].values == cid).astype(int)
        n_pos = int(y.sum())
        n_total = len(y)
        print(f"[step4] cluster {cid}: pos={n_pos}/{n_total} ({n_pos/n_total:.2%})")
        result = _train_eval_one_cluster(X_full, y, feature_cols)
        for model_name, m in result["per_model"].items():
            extraction_rows.append({
                "cluster_id": cid,
                "model": model_name,
                "mean_auc": m["mean_auc"],
                "std_auc": m["std_auc"],
                "fold_aucs": ";".join(f"{a:.4f}" for a in m["fold_aucs"]),
                "auc_best_threshold": m["auc_best_threshold"],
                "auc_best_precision": m["auc_best_precision"],
                "auc_best_recall": m["auc_best_recall"],
                "auc_best_trade_count": m["auc_best_trade_count"],
                "f1_best_threshold": m["f1_best_threshold"],
                "f1_best_precision": m["f1_best_precision"],
                "f1_best_recall": m["f1_best_recall"],
                "f1_best_f1": m["f1_best_f1"],
                "f1_best_trade_count": m["f1_best_trade_count"],
            })
        for fname, mean, std in result["permutation_importance"]:
            feat_importance_rows.append({
                "cluster_id": cid,
                "feature": fname,
                "perm_importance_mean": mean,
                "perm_importance_std": std,
            })
        # Persist best classifier + scaler for Step 5 (A2/A6 reuse)
        best_path = STEP_DIR / "classifiers" / f"cluster_{cid}_{result['best_model_name']}.pkl"
        with best_path.open("wb") as f:
            pickle.dump({
                "classifier": result["best_classifier"],
                "scaler": result["scaler"],
                "feature_names": feature_cols,
                "best_model_name": result["best_model_name"],
                "best_threshold": result["per_model"][result["best_model_name"]]["auc_best_threshold"],
            }, f)
        classifier_summaries.append({
            "cluster_id": cid,
            "best_model": result["best_model_name"],
            "best_threshold": result["per_model"][result["best_model_name"]]["auc_best_threshold"],
            "best_auc": result["per_model"][result["best_model_name"]]["mean_auc"],
            "path": str(best_path),
        })

    extraction_df = pd.DataFrame(extraction_rows)
    feat_importance_df = pd.DataFrame(feat_importance_rows)
    extraction_path = STEP_DIR / "extraction_metrics.csv"
    feat_importance_path = STEP_DIR / "feature_importance.csv"
    extraction_df.to_csv(extraction_path, index=False, lineterminator="\n")
    feat_importance_df.to_csv(feat_importance_path, index=False, lineterminator="\n")

    summary = _build_summary_md(
        extraction_df=extraction_df,
        feat_importance_df=feat_importance_df,
        classifier_summaries=classifier_summaries,
        feature_cols=feature_cols,
    )
    summary_path = STEP_DIR / "extraction_summary.md"
    write_text_deterministic(summary_path, summary)
    write_manifest(
        STEP_DIR / "manifest.json",
        artefacts=[extraction_path, feat_importance_path, summary_path],
    )
    elapsed = time.perf_counter() - t0
    print(f"[step4] DONE in {elapsed:.1f}s -- {len(classifier_summaries)} classifier(s) persisted")
    return STEP_DIR


def _build_summary_md(
    extraction_df: pd.DataFrame,
    feat_importance_df: pd.DataFrame,
    classifier_summaries: list[dict],
    feature_cols: list[str],
) -> str:
    lines = [
        "# Arc 8 — Step 4 Extraction Summary",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}Z_",
        "",
        f"- Candidate clusters processed: {len(classifier_summaries)}",
        "- Classifiers tested: RF, LGBM, Logistic (Appendix A defaults)",
        "- CV: 5-fold TimeSeriesSplit",
        f"- Feature catalogue: {len(feature_cols)} (v3.0 Step 1)",
        "",
        "## Per-cluster, per-model OOS AUC + threshold metrics",
        "",
        "| Cluster | Model | Mean AUC | Std AUC | Fold AUCs | AUC-best thr | Precision | Recall | Trades | F1-best thr | F1 | F1 Trades |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in extraction_df.iterrows():
        lines.append(
            f"| {int(r['cluster_id'])} | {r['model']} "
            f"| {r['mean_auc']:.4f} | {r['std_auc']:.4f} "
            f"| {r['fold_aucs']} "
            f"| {r['auc_best_threshold']:.3f} | {r['auc_best_precision']:.3f} | {r['auc_best_recall']:.3f} | {int(r['auc_best_trade_count']):,} "
            f"| {r['f1_best_threshold']:.3f} | {r['f1_best_f1']:.3f} | {int(r['f1_best_trade_count']):,} |"
        )

    lines += [
        "",
        "## Best classifier per cluster (drives Step 5 A2/A6)",
        "",
        "| Cluster | Best model | Mean AUC | AUC-best threshold | Pickle |",
        "|---:|---|---:|---:|---|",
    ]
    for cs in classifier_summaries:
        lines.append(
            f"| {int(cs['cluster_id'])} | {cs['best_model']} | "
            f"{cs['best_auc']:.4f} | {cs['best_threshold']:.4f} | `{cs['path']}` |"
        )

    lines += ["", "## Top-10 permutation-importance features (per cluster)", ""]
    for cid in sorted(feat_importance_df["cluster_id"].unique()):
        sub = feat_importance_df[feat_importance_df["cluster_id"] == cid].sort_values(
            "perm_importance_mean", ascending=False
        ).head(10)
        lines.append(f"### Cluster {int(cid)}")
        lines.append("")
        lines.append("| Feature | Perm. importance (mean) | Std |")
        lines.append("|---|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(f"| {r['feature']} | {r['perm_importance_mean']:+.5f} | {r['perm_importance_std']:.5f} |")
        lines.append("")

    lines += [
        "",
        "## Methodology notes",
        "",
        "- Target = binary cluster membership (cluster_primary == cid).",
        "- TimeSeriesSplit preserves chronological order: each fold's train is strictly before its test.",
        "- Permutation importance computed on a model refit on full data (scoring=ROC-AUC, n_repeats=5).",
        "- AUC-best threshold derived from Youden's J = TPR - FPR on the concatenated OOS proba stream.",
        "- F1-best threshold derived from F1 sweep over [0.05, 0.95] in 0.01 steps.",
        "- Logistic regression features standardised via StandardScaler per fold; RF and LGBM use raw values.",
        "- Suspect-lineage features (cross_pair class) kept in training matrix per L_PROTOCOL Step 4 mechanic 7 — Step 6 will downgrade/kill candidates whose load-bearing features fail the producer audit. Lineage report at `step_1/feature_lineage.csv`.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
