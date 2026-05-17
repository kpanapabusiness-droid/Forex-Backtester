"""KH-24 v2.0 Step 4 extractability (amendment-conditional).

Pipeline E + Pipeline D1 classifier production for c1 and c4 — the two clusters
surviving Step 3 §2 under proposed AMENDMENT_PROPOSAL_S2_SHAPE_TAG_S11_EXPANSION.md.

Classifier target: binary cluster membership (not profitability). See dispatch
critical methodology note. Step 6 (post D1 PR 2) couples each classifier with its
§11 / amendment exit policy.

Inputs:
- trades_all.csv (arc-specific features + bars_held)
- trades_features_base8.csv (§8 base 8, sidecar from commit 1a7c9f8)
- trades_paths.csv (per-bar path; source of D1 path-so-far features)
- clusters_K5.csv (cluster assignments; target labels)

Outputs:
- predictability_angle_E.csv (per cluster: AUCs, gap, importances, threshold)
- predictability_angle_D1.csv (per cluster per t: AUCs, exclusion, chosen-t)
- extractability_pass_list.csv (per cluster: E pass, D1 pass, pipeline)
- <cluster>_E_classifier.joblib + <cluster>_E_filter.yaml (per Pipeline E pass)
- <cluster>_D1_classifier.joblib + <cluster>_D1_policy.yaml (per Pipeline D1 pass)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Path-so-far feature computation (per bar t)
# ============================================================================

PATH_SO_FAR_COLS = [
    "close_r_at_t",
    "mfe_so_far_r_at_t",
    "mae_so_far_r_at_t",
    "bars_in_profit_at_t",
    "local_peaks_so_far_at_t",
    "monotonicity_so_far_at_t",
    "velocity_first_t",
]


def compute_path_so_far_features(
    trades_paths: pd.DataFrame, t: int
) -> pd.DataFrame:
    """For each trade, compute path-so-far features at bar offset t.

    Trades whose path doesn't extend to bar t (bars_held + forward window < t)
    are dropped — the caller should also apply the bars_held < t exclusion via
    trades_all['bars_held'].
    """
    sub = trades_paths[trades_paths["bar_offset"] <= t].copy()
    sub = sub.sort_values(["trade_id", "bar_offset"])

    rows = []
    for trade_id, grp in sub.groupby("trade_id", sort=False):
        close_r = grp["close_r"].to_numpy()
        mfe = grp["mfe_so_far_r"].to_numpy()
        mae = grp["mae_so_far_r"].to_numpy()
        n = len(close_r)
        if n == 0:
            continue

        close_r_at_t = float(close_r[-1])
        mfe_at_t = float(mfe[-1])
        mae_at_t = float(mae[-1])
        bars_in_profit = int(np.sum(close_r > 0))
        local_peaks = int(np.sum(np.diff(mfe) > 0)) if n >= 2 else 0

        in_profit = close_r[close_r > 0]
        if len(in_profit) == 0:
            mono = 0.0
        elif len(in_profit) == 1:
            mono = 1.0
        else:
            diffs = np.diff(in_profit)
            mono = float(np.sum(diffs >= 0)) / (len(in_profit) - 1)

        velocity = close_r_at_t / max(t, 1)

        rows.append({
            "trade_id": trade_id,
            "close_r_at_t": close_r_at_t,
            "mfe_so_far_r_at_t": mfe_at_t,
            "mae_so_far_r_at_t": mae_at_t,
            "bars_in_profit_at_t": bars_in_profit,
            "local_peaks_so_far_at_t": local_peaks,
            "monotonicity_so_far_at_t": mono,
            "velocity_first_t": velocity,
        })
    return pd.DataFrame(rows)


# ============================================================================
# Data assembly
# ============================================================================

@dataclass
class AssembledData:
    full: pd.DataFrame                # one row per trade with all features + targets
    base8_cols: list[str]
    arc_specific_cols: list[str]
    target_columns: dict[str, str]    # cluster_label -> target column name


def assemble_inputs(cfg: dict, repo_root: Path) -> AssembledData:
    paths_cfg = cfg["inputs"]
    trades_all = pd.read_csv(repo_root / paths_cfg["trades_all"])
    sidecar = pd.read_csv(repo_root / paths_cfg["trades_features_base8"])
    clusters = pd.read_csv(repo_root / paths_cfg["clusters"])

    n_trades_all = len(trades_all)
    if len(sidecar) != n_trades_all:
        raise RuntimeError(
            f"Row count mismatch: trades_all={n_trades_all}, sidecar={len(sidecar)}"
        )
    if len(clusters) != n_trades_all:
        raise RuntimeError(
            f"Row count mismatch: trades_all={n_trades_all}, clusters={len(clusters)}"
        )

    cat = yaml.safe_load((repo_root / paths_cfg["feature_catalogue"]).read_text())
    base8_cols = cat["base8"]
    arc_specific_cols = cat.get("selected_arc_specific", []) or []

    # Three-way inner join on trade_id
    merged = trades_all.merge(
        sidecar.drop(columns=["pair"]),  # avoid pair collision
        on="trade_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        clusters, on="trade_id", how="inner", validate="one_to_one",
    )
    if len(merged) != n_trades_all:
        raise RuntimeError(
            f"Inner join dropped rows: result={len(merged)}, expected={n_trades_all} — "
            f"trade_id mismatch across inputs"
        )

    # Build target columns
    target_columns: dict[str, str] = {}
    for cohort in cfg["target_clusters"]:
        cid = cohort["id"]
        label = cohort["label"]
        col = f"target_{label}"
        merged[col] = (merged["cluster_id"] == cid).astype(int)
        target_columns[label] = col

    # Verify base8 NaN count is zero (Step 1 patch guarantees this)
    base8_nan = int(merged[base8_cols].isna().any(axis=1).sum())
    if base8_nan != 0:
        raise RuntimeError(
            f"Base8 features have {base8_nan} NaN rows post-join — sidecar regression"
        )

    return AssembledData(
        full=merged,
        base8_cols=base8_cols,
        arc_specific_cols=arc_specific_cols,
        target_columns=target_columns,
    )


# ============================================================================
# Classifier training + CV evaluation
# ============================================================================

@dataclass
class CVMetrics:
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    per_fold_roc_auc: list[float] = field(default_factory=list)


def stratified_cv_metrics(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_cfg: dict,
) -> CVMetrics:
    """5-fold stratified CV: returns ROC-AUC + PR-AUC mean/std + per-fold ROC-AUC."""
    skf = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg["shuffle"],
        random_state=cv_cfg["random_state"],
    )
    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        est = _clone_with_state(estimator)
        est.fit(X[train_idx], y[train_idx])
        proba = est.predict_proba(X[test_idx])[:, 1]
        roc_aucs.append(roc_auc_score(y[test_idx], proba))
        pr_aucs.append(average_precision_score(y[test_idx], proba))
    return CVMetrics(
        roc_auc_mean=float(np.mean(roc_aucs)),
        roc_auc_std=float(np.std(roc_aucs, ddof=1)),
        pr_auc_mean=float(np.mean(pr_aucs)),
        pr_auc_std=float(np.std(pr_aucs, ddof=1)),
        per_fold_roc_auc=[float(x) for x in roc_aucs],
    )


def _clone_with_state(estimator):
    """sklearn.base.clone but for pipelines containing fitted scalers — easier to
    re-instantiate from the same params."""
    from sklearn.base import clone
    return clone(estimator)


def build_rf(rf_cfg: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        random_state=rf_cfg["random_state"],
        n_jobs=rf_cfg.get("n_jobs", 1),
    )


def build_logistic(log_cfg: dict) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            max_iter=log_cfg["max_iter"],
            random_state=log_cfg["random_state"],
        )),
    ])


# ============================================================================
# Threshold sweep + selection
# ============================================================================

@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    tp: int
    fp: int
    tn: int
    fn: int


def select_threshold(
    y_true: np.ndarray, proba: np.ndarray, candidates: list[float], min_recall: float
) -> ThresholdResult | None:
    """Return the threshold maximising precision subject to recall ≥ min_recall.
    Returns None if no candidate meets the recall floor.
    """
    feasible: list[ThresholdResult] = []
    for thr in candidates:
        pred = (proba >= thr).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        tn = int(np.sum((pred == 0) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if recall >= min_recall:
            feasible.append(ThresholdResult(
                threshold=thr, precision=precision, recall=recall,
                tp=tp, fp=fp, tn=tn, fn=fn,
            ))
    if not feasible:
        return None
    # Max precision; tiebreak on higher recall, then smaller threshold (looser)
    feasible.sort(key=lambda r: (-r.precision, -r.recall, r.threshold))
    return feasible[0]


# ============================================================================
# Angle E
# ============================================================================

@dataclass
class AngleEResult:
    cluster_label: str
    n_trades: int
    n_positive: int
    positive_rate: float
    feature_set: list[str]
    rf_metrics: CVMetrics
    logistic_metrics: CVMetrics
    rf_logistic_gap: float
    rf_feature_importances: dict[str, float]
    selected_threshold: ThresholdResult | None
    filter_selection_path: str   # "step_a" | "step_b_topN" | "step_c_stacking"
    fitted_rf: RandomForestClassifier | None = None
    fitted_features: list[str] | None = None


def run_angle_e(
    data: AssembledData, cohort: dict, cfg: dict
) -> AngleEResult:
    label = cohort["label"]
    target_col = data.target_columns[label]
    full_features = data.base8_cols + data.arc_specific_cols

    y = data.full[target_col].to_numpy()
    X = data.full[full_features].to_numpy()

    rf = build_rf(cfg["random_forest"])
    logit = build_logistic(cfg["logistic_regression"])

    rf_metrics = stratified_cv_metrics(rf, X, y, cfg["cv"])
    log_metrics = stratified_cv_metrics(logit, X, y, cfg["cv"])
    gap = rf_metrics.roc_auc_mean - log_metrics.roc_auc_mean

    # Importances on full-feature RF (trained on all data — separate from CV; only
    # used for ranking, not for gate decision).
    rf_full = build_rf(cfg["random_forest"])
    rf_full.fit(X, y)
    importances = {
        f: float(imp) for f, imp in zip(full_features, rf_full.feature_importances_)
    }

    # Filter selection — Step A baseline; Step B subsets if A fails; Step C stacking
    # not implemented here (feature set has only 9 columns; Step C's stacking budget is
    # best deployed when richer feature catalogues are available — flagged in result doc).
    gate = cfg["gates"]["pipeline_e_rf_auc_min"]
    selected_features = full_features
    final_rf = rf_full
    final_metrics = rf_metrics

    if rf_metrics.roc_auc_mean >= gate:
        selection_path = "step_a_pass"
    else:
        # Step B: top-N subsets by importance
        selection_path = "step_a_fail"
        for n in cfg["filter_selection"]["step_b_subset_sizes"]:
            if n >= len(full_features):
                continue  # subset same as full — skip
            top_n = sorted(importances.items(), key=lambda kv: -kv[1])[:n]
            top_features = [f for f, _ in top_n]
            X_sub = data.full[top_features].to_numpy()
            sub_rf = build_rf(cfg["random_forest"])
            sub_metrics = stratified_cv_metrics(sub_rf, X_sub, y, cfg["cv"])
            label_tag = f"step_b_top{n}"
            if sub_metrics.roc_auc_mean >= gate:
                selected_features = top_features
                selection_path = f"{label_tag}_pass"
                final_rf = build_rf(cfg["random_forest"])
                final_rf.fit(X_sub, y)
                final_metrics = sub_metrics
                break
            else:
                selection_path = f"{selection_path}|{label_tag}_fail({sub_metrics.roc_auc_mean:.3f})"
        # No Step C attempted (see note above). Mark explicitly.
        if "pass" not in selection_path:
            selection_path = f"{selection_path}|step_c_not_attempted"

    # Threshold sweep (only meaningful if Pipeline E passes; otherwise we still
    # report the best-effort selection for diagnostic context).
    X_final = data.full[selected_features].to_numpy()
    proba = final_rf.predict_proba(X_final)[:, 1]
    threshold = select_threshold(
        y, proba,
        cfg["threshold_sweep"],
        cfg["gates"].get("threshold_recall_min", cfg.get("threshold_recall_min", 0.60)),
    )

    return AngleEResult(
        cluster_label=label,
        n_trades=int(len(y)),
        n_positive=int(np.sum(y == 1)),
        positive_rate=float(np.mean(y == 1)),
        feature_set=selected_features,
        rf_metrics=final_metrics,
        logistic_metrics=log_metrics,
        rf_logistic_gap=gap,
        rf_feature_importances=importances,
        selected_threshold=threshold,
        filter_selection_path=selection_path,
        fitted_rf=final_rf,
        fitted_features=selected_features,
    )


# ============================================================================
# Angle D1
# ============================================================================

@dataclass
class AngleD1tResult:
    t: int
    n_trades_alive_at_t: int
    n_positive_at_t: int
    n_positive_excluded: int
    exclusion_rate: float
    rf_metrics: CVMetrics | None
    selected_threshold: ThresholdResult | None
    fitted_rf: RandomForestClassifier | None = None
    fitted_features: list[str] | None = None


@dataclass
class AngleD1Result:
    cluster_label: str
    per_t: dict[int, AngleD1tResult]
    chosen_t: int | None
    chosen_pass: bool


def run_angle_d1(
    data: AssembledData, trades_paths: pd.DataFrame, cohort: dict, cfg: dict
) -> AngleD1Result:
    label = cohort["label"]
    target_col = data.target_columns[label]
    base_entry_features = data.base8_cols + data.arc_specific_cols

    per_t: dict[int, AngleD1tResult] = {}
    chosen_t: int | None = None
    auc_gate = cfg["gates"]["pipeline_d1_rf_auc_min"]
    excl_max = cfg["gates"]["pipeline_d1_exclusion_max"]

    for t in cfg["d1_t_values"]:
        # Exclude trades with bars_held < t
        alive_mask = data.full["bars_held"] >= t
        alive_df = data.full[alive_mask].copy()
        n_alive = int(len(alive_df))

        # Path-so-far features at t for alive trades
        psf = compute_path_so_far_features(
            trades_paths[trades_paths["trade_id"].isin(alive_df["trade_id"])], t
        )
        joined = alive_df.merge(psf, on="trade_id", how="inner")
        if len(joined) == 0:
            per_t[t] = AngleD1tResult(
                t=t, n_trades_alive_at_t=0, n_positive_at_t=0,
                n_positive_excluded=0, exclusion_rate=1.0,
                rf_metrics=None, selected_threshold=None,
            )
            continue

        # Exclusion stats: per §8, exclusion = positives_excluded / positives_total
        positives_total = int((data.full[target_col] == 1).sum())
        positives_alive = int((joined[target_col] == 1).sum())
        positives_excluded = positives_total - positives_alive
        exclusion_rate = positives_excluded / max(positives_total, 1)

        y = joined[target_col].to_numpy()
        feat_cols = base_entry_features + PATH_SO_FAR_COLS
        X = joined[feat_cols].to_numpy()

        # Skip CV if only one class present at this t
        if len(np.unique(y)) < 2:
            per_t[t] = AngleD1tResult(
                t=t, n_trades_alive_at_t=n_alive, n_positive_at_t=positives_alive,
                n_positive_excluded=positives_excluded, exclusion_rate=exclusion_rate,
                rf_metrics=None, selected_threshold=None,
            )
            continue

        rf = build_rf(cfg["random_forest"])
        rf_metrics = stratified_cv_metrics(rf, X, y, cfg["cv"])

        # Fit on all alive-at-t data; sweep threshold
        rf_fit = build_rf(cfg["random_forest"])
        rf_fit.fit(X, y)
        proba = rf_fit.predict_proba(X)[:, 1]
        threshold = select_threshold(
            y, proba, cfg["threshold_sweep"],
            cfg["gates"].get("threshold_recall_min", cfg.get("threshold_recall_min", 0.60)),
        )

        per_t[t] = AngleD1tResult(
            t=t,
            n_trades_alive_at_t=n_alive,
            n_positive_at_t=positives_alive,
            n_positive_excluded=positives_excluded,
            exclusion_rate=exclusion_rate,
            rf_metrics=rf_metrics,
            selected_threshold=threshold,
            fitted_rf=rf_fit,
            fitted_features=feat_cols,
        )

        # Smallest-t selection
        if chosen_t is None:
            passes = (
                rf_metrics.roc_auc_mean >= auc_gate
                and exclusion_rate <= excl_max
            )
            if passes:
                chosen_t = t

    return AngleD1Result(
        cluster_label=label,
        per_t=per_t,
        chosen_t=chosen_t,
        chosen_pass=chosen_t is not None,
    )


# ============================================================================
# Output writers
# ============================================================================

def write_predictability_angle_e(
    results: list[AngleEResult], output_dir: Path, gate: float
) -> None:
    rows = []
    for r in results:
        importances_top10 = sorted(r.rf_feature_importances.items(), key=lambda kv: -kv[1])[:10]
        importances_str = ";".join(f"{f}={imp:.4f}" for f, imp in importances_top10)
        rows.append({
            "cluster_label": r.cluster_label,
            "n_trades": r.n_trades,
            "n_positive": r.n_positive,
            "positive_rate": r.positive_rate,
            "feature_set_size": len(r.feature_set),
            "feature_set": ";".join(r.feature_set),
            "filter_selection_path": r.filter_selection_path,
            "rf_roc_auc_mean": r.rf_metrics.roc_auc_mean,
            "rf_roc_auc_std": r.rf_metrics.roc_auc_std,
            "rf_pr_auc_mean": r.rf_metrics.pr_auc_mean,
            "rf_pr_auc_std": r.rf_metrics.pr_auc_std,
            "logistic_roc_auc_mean": r.logistic_metrics.roc_auc_mean,
            "logistic_roc_auc_std": r.logistic_metrics.roc_auc_std,
            "logistic_pr_auc_mean": r.logistic_metrics.pr_auc_mean,
            "rf_logistic_gap": r.rf_logistic_gap,
            "rf_per_fold_roc_auc": ";".join(f"{x:.4f}" for x in r.rf_metrics.per_fold_roc_auc),
            "rf_top10_importance": importances_str,
            "selected_threshold": r.selected_threshold.threshold if r.selected_threshold else None,
            "threshold_precision": r.selected_threshold.precision if r.selected_threshold else None,
            "threshold_recall": r.selected_threshold.recall if r.selected_threshold else None,
            "threshold_tp": r.selected_threshold.tp if r.selected_threshold else None,
            "threshold_fp": r.selected_threshold.fp if r.selected_threshold else None,
            "threshold_tn": r.selected_threshold.tn if r.selected_threshold else None,
            "threshold_fn": r.selected_threshold.fn if r.selected_threshold else None,
            "passes_gate": r.rf_metrics.roc_auc_mean >= gate,
        })
    pd.DataFrame(rows).to_csv(output_dir / "predictability_angle_E.csv", index=False, lineterminator="\n")


def write_predictability_angle_d1(
    results: list[AngleD1Result], output_dir: Path, auc_gate: float, excl_max: float
) -> None:
    rows = []
    for r in results:
        for t, t_res in r.per_t.items():
            rows.append({
                "cluster_label": r.cluster_label,
                "t": t,
                "n_trades_alive_at_t": t_res.n_trades_alive_at_t,
                "n_positive_at_t": t_res.n_positive_at_t,
                "n_positive_excluded": t_res.n_positive_excluded,
                "exclusion_rate": t_res.exclusion_rate,
                "rf_roc_auc_mean": t_res.rf_metrics.roc_auc_mean if t_res.rf_metrics else None,
                "rf_roc_auc_std": t_res.rf_metrics.roc_auc_std if t_res.rf_metrics else None,
                "rf_pr_auc_mean": t_res.rf_metrics.pr_auc_mean if t_res.rf_metrics else None,
                "rf_pr_auc_std": t_res.rf_metrics.pr_auc_std if t_res.rf_metrics else None,
                "rf_per_fold_roc_auc": (
                    ";".join(f"{x:.4f}" for x in t_res.rf_metrics.per_fold_roc_auc)
                    if t_res.rf_metrics else ""
                ),
                "passes_auc_gate": (
                    t_res.rf_metrics.roc_auc_mean >= auc_gate if t_res.rf_metrics else False
                ),
                "passes_exclusion_gate": t_res.exclusion_rate <= excl_max,
                "chosen_t_for_cluster": (r.chosen_t == t),
                "selected_threshold": t_res.selected_threshold.threshold if t_res.selected_threshold else None,
                "threshold_precision": t_res.selected_threshold.precision if t_res.selected_threshold else None,
                "threshold_recall": t_res.selected_threshold.recall if t_res.selected_threshold else None,
            })
    pd.DataFrame(rows).to_csv(output_dir / "predictability_angle_D1.csv", index=False, lineterminator="\n")


def write_pass_list(
    e_results: list[AngleEResult],
    d1_results: list[AngleD1Result],
    output_dir: Path,
    gate: float,
) -> None:
    e_by_label = {r.cluster_label: r for r in e_results}
    d1_by_label = {r.cluster_label: r for r in d1_results}
    rows = []
    for label in e_by_label:
        e = e_by_label[label]
        d1 = d1_by_label.get(label)
        e_pass = e.rf_metrics.roc_auc_mean >= gate
        d1_pass = d1.chosen_pass if d1 is not None else False
        if e_pass and d1_pass:
            pipeline = "both"
        elif e_pass:
            pipeline = "E"
        elif d1_pass:
            pipeline = "D1"
        else:
            pipeline = "neither"
        rows.append({
            "cluster_label": label,
            "pipeline_e_pass": e_pass,
            "pipeline_d1_pass": d1_pass,
            "pipeline_d1_chosen_t": d1.chosen_t if d1 and d1.chosen_pass else None,
            "assigned_pipeline": pipeline,
            "e_rf_roc_auc": e.rf_metrics.roc_auc_mean,
            "e_rf_pr_auc": e.rf_metrics.pr_auc_mean,
            "d1_chosen_t_rf_roc_auc": (
                d1.per_t[d1.chosen_t].rf_metrics.roc_auc_mean
                if d1 and d1.chosen_pass and d1.per_t[d1.chosen_t].rf_metrics
                else None
            ),
        })
    pd.DataFrame(rows).to_csv(output_dir / "extractability_pass_list.csv", index=False, lineterminator="\n")


def write_classifier_artefacts(
    e_result: AngleEResult,
    d1_result: AngleD1Result,
    cohort: dict,
    output_dir: Path,
    cfg: dict,
    e_gate: float,
) -> None:
    label = cohort["label"]
    if e_result.rf_metrics.roc_auc_mean >= e_gate:
        joblib_path = output_dir / f"{label}_E_classifier.joblib"
        joblib.dump(e_result.fitted_rf, joblib_path)
        filter_yaml = {
            "cluster_id": label,
            "archetype": cohort["archetype"],
            "amendment_dependency": cfg["amendment_dependency"],
            "pipeline": "E",
            "target": "cluster_membership_binary",
            "positives_count": e_result.n_positive,
            "negatives_count": e_result.n_trades - e_result.n_positive,
            "classifier_path": str(joblib_path.name),
            "admit_threshold": (
                e_result.selected_threshold.threshold if e_result.selected_threshold else None
            ),
            "admit_logic": f"P({label}) >= admit_threshold",
            "feature_set": e_result.feature_set,
            "feature_sources": {
                "base8": cfg["inputs"]["trades_features_base8"],
                "arc_specific": cfg["inputs"]["trades_all"],
            },
            "rf_roc_auc_cv_mean": e_result.rf_metrics.roc_auc_mean,
            "rf_pr_auc_cv_mean": e_result.rf_metrics.pr_auc_mean,
            "filter_selection_path": e_result.filter_selection_path,
            "exit_policy_ref": {
                **{
                    "source": cohort["exit_policy"].get("source_ref", "TBD"),
                    "initial_sl": f"{cohort['selected_sl_atr']}xATR",
                },
                "on_admission_then": {
                    k: v for k, v in cohort["exit_policy"].items() if k != "source_ref"
                },
                "executor": "D1 PR 2 backtester (deferred)",
            },
        }
        (output_dir / f"{label}_E_filter.yaml").write_text(yaml.safe_dump(
            filter_yaml, sort_keys=False, default_flow_style=False, allow_unicode=True
        ), encoding="utf-8")

    if d1_result is not None and d1_result.chosen_pass:
        t = d1_result.chosen_t
        t_res = d1_result.per_t[t]
        joblib_path = output_dir / f"{label}_D1_classifier.joblib"
        joblib.dump(t_res.fitted_rf, joblib_path)
        policy_yaml = {
            "cluster_id": label,
            "archetype": cohort["archetype"],
            "amendment_dependency": cfg["amendment_dependency"],
            "pipeline": "D1",
            "target": "cluster_membership_binary",
            "positives_count": int(d1_result.per_t[t].n_positive_at_t),
            "negatives_count": int(
                d1_result.per_t[t].n_trades_alive_at_t - d1_result.per_t[t].n_positive_at_t
            ),
            "chosen_t": t,
            "classifier_path": str(joblib_path.name),
            "admit_threshold": (
                t_res.selected_threshold.threshold if t_res.selected_threshold else None
            ),
            "admit_logic": f"P({label} | features_at_t={t}) >= admit_threshold",
            "feature_set": t_res.fitted_features,
            "feature_sources": {
                "base8": cfg["inputs"]["trades_features_base8"],
                "arc_specific": cfg["inputs"]["trades_all"],
                "path_so_far": cfg["inputs"]["trades_paths"],
            },
            "rf_roc_auc_cv_mean": t_res.rf_metrics.roc_auc_mean,
            "rf_pr_auc_cv_mean": t_res.rf_metrics.pr_auc_mean,
            "exclusion_rate": t_res.exclusion_rate,
            "exit_policy_ref": {
                **{
                    "source": cohort["exit_policy"].get("source_ref", "TBD"),
                    "initial_sl": f"{cohort['selected_sl_atr']}xATR (entry-to-bar-N pre-classification)",
                },
                "on_bar_N_if_admitted": {
                    k: v for k, v in cohort["exit_policy"].items() if k != "source_ref"
                },
                "on_bar_N_if_rejected": {
                    "close_at_market": "bar N+1 open",
                },
                "executor": "D1 PR 2 backtester (deferred; PR #131 close-at-market path operational)",
            },
        }
        (output_dir / f"{label}_D1_policy.yaml").write_text(yaml.safe_dump(
            policy_yaml, sort_keys=False, default_flow_style=False, allow_unicode=True
        ), encoding="utf-8")


# ============================================================================
# Driver
# ============================================================================

def run(config_path: Path, override_output_dir: Path | None) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load(config_path.read_text())

    data = assemble_inputs(cfg, repo_root)
    trades_paths = pd.read_csv(repo_root / cfg["inputs"]["trades_paths"])

    e_results: list[AngleEResult] = []
    d1_results: list[AngleD1Result] = []
    for cohort in cfg["target_clusters"]:
        print(f"=== Cluster {cohort['label']} (archetype: {cohort['archetype']}) ===")
        e = run_angle_e(data, cohort, cfg)
        print(f"  Angle E: RF ROC-AUC {e.rf_metrics.roc_auc_mean:.4f} "
              f"PR-AUC {e.rf_metrics.pr_auc_mean:.4f} (gap {e.rf_logistic_gap:+.4f}) "
              f"path={e.filter_selection_path}")
        e_results.append(e)

        d1 = run_angle_d1(data, trades_paths, cohort, cfg)
        if d1.chosen_t is not None:
            tr = d1.per_t[d1.chosen_t]
            print(f"  Angle D1: chosen t={d1.chosen_t} "
                  f"RF ROC-AUC {tr.rf_metrics.roc_auc_mean:.4f} "
                  f"PR-AUC {tr.rf_metrics.pr_auc_mean:.4f} "
                  f"excl {tr.exclusion_rate:.3f}")
        else:
            print("  Angle D1: no t passes both AUC and exclusion gates")
        d1_results.append(d1)

    output_dir = override_output_dir or (repo_root / cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    write_predictability_angle_e(
        e_results, output_dir, cfg["gates"]["pipeline_e_rf_auc_min"],
    )
    write_predictability_angle_d1(
        d1_results, output_dir,
        cfg["gates"]["pipeline_d1_rf_auc_min"],
        cfg["gates"]["pipeline_d1_exclusion_max"],
    )
    write_pass_list(
        e_results, d1_results, output_dir, cfg["gates"]["pipeline_e_rf_auc_min"],
    )

    for cohort, e, d1 in zip(cfg["target_clusters"], e_results, d1_results):
        write_classifier_artefacts(
            e, d1, cohort, output_dir, cfg, cfg["gates"]["pipeline_e_rf_auc_min"],
        )

    print(f"Wrote outputs to: {output_dir}")
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Override output_dir from config (used for determinism re-runs)")
    args = parser.parse_args()
    return run(args.config, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
