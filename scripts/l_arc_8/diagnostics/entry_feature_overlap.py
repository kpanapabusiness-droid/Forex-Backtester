"""Arc 8 — Entry-Feature Overlap Diagnostic (post-Step-5 follow-up).

Determines whether c1 is distinguishable from c0/c2/c3 at the entry bar
using only entry-time features. This is the diagnostic that decides
whether the Step 5 full-pool failure is fixable via a multiclass reframe
(c1 vs others at entry) or whether c1 is structurally unforecastable
at entry time.

NOT a protocol step. Pure analysis on locked Step 1/2 outputs.

Three parts:
  A. Univariate per-feature overlap of c1 vs each other cluster
     (KS statistic + p-value, Wasserstein distance, overlap coefficient)
  B. Multivariate separability — multiclass RF on 4 cluster labels,
     5-fold TimeSeriesSplit, per-class AUC + confusion matrix; key metric
     is c1's precision@recall=0.60 in one-vs-rest framing
  C. 6×3 KDE grid for all 18 features (visualisation)
  D. Top-5 separating features deep dive

Feature set (18 entry-time):
  - 8 base entry (computed via scripts.l_arc_8.step4_extractability
    compute_base_e_features from raw 4H bars):
    body_to_range_ratio, upper_wick_ratio, lower_wick_ratio,
    range_to_atr_14, ret_5bar_atr, ret_20bar_atr, pos_in_20bar_range,
    rsi_14
  - 10 PR-HHHL-specific (already in results/l_arc_8/step1_verbatim/
    trades_all.csv):
    num_higher_highs, num_higher_lows, most_recent_sh_age,
    most_recent_sl_age, hh_range_atr, hl_range_atr, pullback_depth_atr,
    trigger_body_atr, trigger_close_pos, trigger_break_size_atr

random_state=42, n_jobs=1 throughout. Deterministic.

Usage:
    py scripts/l_arc_8/diagnostics/entry_feature_overlap.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the locked feature builder from Step 4 (function-level inspection only
# per dispatch — not retraining or modifying Step 4 outputs).
from scripts.l_arc_8.step4_extractability import (  # noqa: E402
    PIPELINE_E_ARC8_FEATURES,
    PIPELINE_E_BASE_FEATURES,
    _build_pair_cache,
    _impute_nans,
    compute_base_e_features,
)

OUT_DIR = _REPO_ROOT / "results" / "l_arc_8" / "diagnostics" / "entry_feature_overlap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_COLORS = {0: "#888888", 1: "#2ca02c", 2: "#d62728", 3: "#1f77b4"}
CLUSTER_LABELS = {0: "c0 (unassigned)", 1: "c1 (V-shape FG-weak)",
                  2: "c2 (Early-peak hold)", 3: "c3 (V-shape canonical)"}


# ============================================================
# Part A — univariate overlap
# ============================================================


def overlap_coefficient(x: np.ndarray, y: np.ndarray, bins: int = 30) -> float:
    """1 - total variation distance from histograms on union range.

    Returns value in [0, 1] where 1 = identical distributions and
    0 = disjoint supports. Conceptually: the area shared between the
    two normalised histograms.
    """
    if x.size == 0 or y.size == 0:
        return 0.0
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if lo == hi:
        return 1.0  # both constant on same value
    edges = np.linspace(lo, hi, bins + 1)
    hx, _ = np.histogram(x, bins=edges, density=False)
    hy, _ = np.histogram(y, bins=edges, density=False)
    px = hx.astype(float) / hx.sum() if hx.sum() > 0 else hx.astype(float)
    py = hy.astype(float) / hy.sum() if hy.sum() > 0 else hy.astype(float)
    return float(np.minimum(px, py).sum())


def part_a_univariate(
    trades_with_features: pd.DataFrame, feature_cols: List[str],
    cluster_col: str = "cluster_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-cluster stats + pairwise c1-vs-other comparisons.

    Returns (long_overlap_df, summary_ranked_df).
    """
    from scipy import stats as sps

    clusters = sorted(trades_with_features[cluster_col].unique().tolist())
    print(f"  clusters present: {clusters}", file=sys.stderr)

    # Per-cluster stats (separate output for inspection).
    per_cluster_stats: List[Dict] = []
    for f in feature_cols:
        for cid in clusters:
            v = trades_with_features.loc[trades_with_features[cluster_col] == cid, f].dropna().to_numpy(dtype=float)
            if v.size == 0:
                row = {"feature": f, "cluster_id": cid, "n": 0,
                       "mean": float("nan"), "std": float("nan"),
                       "p5": float("nan"), "p25": float("nan"), "p50": float("nan"),
                       "p75": float("nan"), "p95": float("nan")}
            else:
                row = {
                    "feature": f, "cluster_id": cid, "n": int(v.size),
                    "mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                    "p5": float(np.percentile(v, 5)),
                    "p25": float(np.percentile(v, 25)),
                    "p50": float(np.percentile(v, 50)),
                    "p75": float(np.percentile(v, 75)),
                    "p95": float(np.percentile(v, 95)),
                }
            per_cluster_stats.append(row)

    # Pairwise c1 vs each other cluster.
    comparisons = [(1, other) for other in clusters if other != 1]
    long_rows: List[Dict] = []
    for f in feature_cols:
        c1 = trades_with_features.loc[trades_with_features[cluster_col] == 1, f].dropna().to_numpy(dtype=float)
        c1_std = float(c1.std(ddof=1)) if c1.size > 1 else 1.0
        if c1_std == 0:
            c1_std = 1.0  # protect against divide-by-zero in wasserstein normalisation
        for c1_cid, other_cid in comparisons:
            other = trades_with_features.loc[trades_with_features[cluster_col] == other_cid, f].dropna().to_numpy(dtype=float)
            if c1.size < 2 or other.size < 2:
                ks_stat = float("nan")
                ks_p = float("nan")
                wass = float("nan")
                wass_norm = float("nan")
                oc = float("nan")
            else:
                ks_stat_, ks_p_ = sps.ks_2samp(c1, other)
                ks_stat = float(ks_stat_)
                ks_p = float(ks_p_)
                wass = float(sps.wasserstein_distance(c1, other))
                wass_norm = wass / c1_std
                oc = overlap_coefficient(c1, other, bins=30)
            long_rows.append({
                "feature": f,
                "comparison": f"c{c1_cid}_vs_c{other_cid}",
                "n_c1": int(c1.size),
                "n_other": int(other.size),
                "ks_stat": ks_stat,
                "ks_pvalue": ks_p,
                "wasserstein": wass,
                "wasserstein_norm_by_c1_std": wass_norm,
                "overlap_coef": oc,
            })

    long_df = pd.DataFrame(long_rows)

    # Summary ranked per feature (mean overlap across 3 comparisons).
    summary_rows: List[Dict] = []
    for f in feature_cols:
        sub = long_df[long_df["feature"] == f]
        oc_vals = sub["overlap_coef"].to_numpy(dtype=float)
        oc_vals = oc_vals[~np.isnan(oc_vals)]
        mean_oc = float(oc_vals.mean()) if oc_vals.size else float("nan")
        min_oc = float(oc_vals.min()) if oc_vals.size else float("nan")
        max_oc = float(oc_vals.max()) if oc_vals.size else float("nan")
        # KS stat mean (higher = more separation).
        ks_vals = sub["ks_stat"].to_numpy(dtype=float)
        ks_vals = ks_vals[~np.isnan(ks_vals)]
        mean_ks = float(ks_vals.mean()) if ks_vals.size else float("nan")
        # Verdict from mean overlap.
        if math.isnan(mean_oc):
            verdict = "no_data"
        elif mean_oc <= 0.70:
            verdict = "separating"
        elif mean_oc <= 0.85:
            verdict = "weak"
        else:
            verdict = "indistinguishable"
        summary_rows.append({
            "feature": f, "mean_overlap_coef": mean_oc,
            "min_overlap_coef": min_oc, "max_overlap_coef": max_oc,
            "mean_ks_stat": mean_ks, "verdict": verdict,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_overlap_coef", kind="mergesort").reset_index(drop=True)
    return long_df, summary_df


# ============================================================
# Part B — multiclass separability
# ============================================================


def part_b_multiclass(
    trades_with_features: pd.DataFrame, feature_cols: List[str],
    cluster_col: str = "cluster_id",
    n_splits: int = 5,
) -> Dict:
    """Multiclass RF predicting cluster_id ∈ {0, 1, 2, 3} from entry features.

    5-fold TimeSeriesSplit on entry-time order. Aggregate out-of-fold
    predictions for per-class metrics + confusion matrix.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_curve,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.model_selection import TimeSeriesSplit

    # Sort by entry_time for time-ordered CV.
    df = trades_with_features.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df = df.sort_values("entry_time", kind="mergesort").reset_index(drop=True)

    X = _impute_nans(df[feature_cols])
    y = df[cluster_col].to_numpy(dtype=int)
    classes = sorted(np.unique(y).tolist())
    print(f"  multiclass: n={len(y)}, classes={classes}, "
          f"class counts={dict(zip(*np.unique(y, return_counts=True)))}", file=sys.stderr)

    tss = TimeSeriesSplit(n_splits=n_splits)
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros((len(y), len(classes)), dtype=float)
    fold_idx = np.full(len(y), -1, dtype=int)
    per_fold_acc: List[float] = []
    for fold_n, (tr_idx, te_idx) in enumerate(tss.split(X)):
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=1,
            class_weight="balanced",
        )
        clf.fit(X[tr_idx], y[tr_idx])
        pred = clf.predict(X[te_idx])
        prob = clf.predict_proba(X[te_idx])
        # Align prob columns to global classes order — class_ may be subset.
        prob_aligned = np.zeros((prob.shape[0], len(classes)), dtype=float)
        for i, c in enumerate(clf.classes_):
            if c in classes:
                prob_aligned[:, classes.index(c)] = prob[:, i]
        oof_pred[te_idx] = pred
        oof_proba[te_idx, :] = prob_aligned
        fold_idx[te_idx] = fold_n
        per_fold_acc.append(float(accuracy_score(y[te_idx], pred)))

    # Aggregate metrics on out-of-fold predictions (some samples in fold 0
    # train have no OOF — TimeSeriesSplit excludes them).
    mask = oof_pred >= 0
    y_oof = y[mask]
    pred_oof = oof_pred[mask]
    proba_oof = oof_proba[mask]

    # Per-class precision/recall/f1.
    p, r, f1, support = precision_recall_fscore_support(y_oof, pred_oof, labels=classes, zero_division=0)
    accuracy = float(accuracy_score(y_oof, pred_oof))
    conf_mat = confusion_matrix(y_oof, pred_oof, labels=classes)

    # Per-class one-vs-rest ROC AUC.
    per_class_auc: Dict[int, float] = {}
    for i, c in enumerate(classes):
        y_bin = (y_oof == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            per_class_auc[c] = float("nan")
            continue
        try:
            auc = float(roc_auc_score(y_bin, proba_oof[:, i]))
        except Exception:
            auc = float("nan")
        per_class_auc[c] = auc

    # c1 precision@recall=0.60 in one-vs-rest framing.
    c1_idx = classes.index(1) if 1 in classes else None
    c1_p_at_r60 = float("nan")
    c1_threshold_at_r60 = float("nan")
    if c1_idx is not None and len(np.unique((y_oof == 1).astype(int))) == 2:
        y_bin = (y_oof == 1).astype(int)
        prec, rec, thr = precision_recall_curve(y_bin, proba_oof[:, c1_idx])
        # Want max precision among points where recall >= 0.60.
        # prec/rec arrays have len = thr+1. Last point is recall=0, ignore.
        valid_idxs = np.where(rec >= 0.60)[0]
        if valid_idxs.size > 0:
            best = valid_idxs[np.argmax(prec[valid_idxs])]
            c1_p_at_r60 = float(prec[best])
            # threshold[i] corresponds to (prec[i], rec[i]) for i in 0..len(thr)-1
            if best < len(thr):
                c1_threshold_at_r60 = float(thr[best])

    return {
        "n_samples": int(len(y)),
        "n_samples_oof": int(mask.sum()),
        "n_classes": len(classes),
        "classes": classes,
        "class_counts": {int(c): int((y == c).sum()) for c in classes},
        "class_base_rates": {int(c): float((y == c).mean()) for c in classes},
        "n_splits": n_splits,
        "model": "RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=1, class_weight='balanced')",
        "per_fold_accuracy": per_fold_acc,
        "oof_accuracy": accuracy,
        "per_class": {
            int(c): {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "auc_one_vs_rest": per_class_auc[c],
                "base_rate": float((y == c).mean()),
            } for i, c in enumerate(classes)
        },
        "c1_precision_at_recall_060": c1_p_at_r60,
        "c1_threshold_at_recall_060": c1_threshold_at_r60,
        "confusion_matrix": {
            "labels": classes,
            "rows_true_cols_pred": conf_mat.tolist(),
        },
    }


# ============================================================
# Part C — feature distribution visuals (6x3 KDE grid)
# ============================================================


def part_c_kde_grid(
    trades_with_features: pd.DataFrame, feature_cols: List[str],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps

    n_features = len(feature_cols)
    n_rows = (n_features + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows))
    axes = np.array(axes).reshape(n_rows, 3)

    for ax_idx, f in enumerate(feature_cols):
        r = ax_idx // 3
        c = ax_idx % 3
        ax = axes[r, c]
        clusters = sorted(trades_with_features["cluster_id"].unique().tolist())
        all_vals = trades_with_features[f].dropna().to_numpy(dtype=float)
        if all_vals.size < 2 or all_vals.max() == all_vals.min():
            ax.text(0.5, 0.5, "constant or empty", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10)
            ax.set_title(f, fontsize=10)
            continue
        x_grid = np.linspace(float(all_vals.min()), float(all_vals.max()), 200)
        c1_median = None
        for cid in clusters:
            v = trades_with_features.loc[trades_with_features["cluster_id"] == cid, f].dropna().to_numpy(dtype=float)
            if v.size < 2:
                continue
            if v.std(ddof=1) == 0:
                # Spike at single value — plot as vertical line.
                ax.axvline(float(v.mean()), color=CLUSTER_COLORS[cid],
                           linewidth=1.5, alpha=0.6, label=CLUSTER_LABELS[cid])
                continue
            try:
                kde = sps.gaussian_kde(v)
                ax.plot(x_grid, kde(x_grid), color=CLUSTER_COLORS[cid],
                        linewidth=1.8 if cid == 1 else 1.2,
                        alpha=0.95 if cid == 1 else 0.75,
                        label=CLUSTER_LABELS[cid])
            except Exception:
                continue
            if cid == 1:
                c1_median = float(np.median(v))
        if c1_median is not None:
            ax.axvline(c1_median, color=CLUSTER_COLORS[1], linewidth=1.0,
                       linestyle="--", alpha=0.7)
        ax.set_title(f, fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, linewidth=0.3, alpha=0.4)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused axes.
    for ax_idx in range(n_features, n_rows * 3):
        r = ax_idx // 3
        c = ax_idx % 3
        axes[r, c].axis("off")

    fig.suptitle("Arc 8 — entry-time feature distributions by cluster "
                 "(c1 = V-shape recovery FG-weak, highlighted; c1 median dashed)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110, metadata={"Software": ""})
    plt.close(fig)


# ============================================================
# Part D — top-5 deep dive
# ============================================================


def part_d_top5_detail(
    trades_with_features: pd.DataFrame, top5_features: List[str],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps

    rng = np.random.default_rng(42)
    n_rows = len(top5_features)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = np.array(axes).reshape(n_rows, 3)

    clusters = sorted(trades_with_features["cluster_id"].unique().tolist())

    for row_idx, f in enumerate(top5_features):
        # Column 0: jittered scatter by cluster.
        ax = axes[row_idx, 0]
        for cid in clusters:
            v = trades_with_features.loc[trades_with_features["cluster_id"] == cid, f].dropna().to_numpy(dtype=float)
            jitter = rng.normal(cid, 0.10, size=v.size)
            ax.scatter(jitter, v, s=8, color=CLUSTER_COLORS[cid], alpha=0.45,
                       label=f"c{cid}", edgecolors="none")
        ax.set_xticks(clusters)
        ax.set_xticklabels([f"c{c}" for c in clusters], fontsize=9)
        ax.set_ylabel(f, fontsize=9)
        ax.set_title(f"{f} — jittered scatter", fontsize=10)
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.tick_params(axis="both", labelsize=8)

        # Column 1: boxplot.
        ax = axes[row_idx, 1]
        data = [trades_with_features.loc[trades_with_features["cluster_id"] == cid, f].dropna().to_numpy(dtype=float)
                for cid in clusters]
        bp = ax.boxplot(data, labels=[f"c{c}" for c in clusters], patch_artist=True, widths=0.55)
        for patch, cid in zip(bp["boxes"], clusters):
            patch.set_facecolor(CLUSTER_COLORS[cid])
            patch.set_alpha(0.65)
        ax.set_title(f"{f} — boxplot", fontsize=10)
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.tick_params(axis="both", labelsize=8)

        # Column 2: 1D KDE c1 vs not-c1.
        ax = axes[row_idx, 2]
        all_vals = trades_with_features[f].dropna().to_numpy(dtype=float)
        if all_vals.size >= 2 and all_vals.max() > all_vals.min():
            x_grid = np.linspace(float(all_vals.min()), float(all_vals.max()), 200)
            c1 = trades_with_features.loc[trades_with_features["cluster_id"] == 1, f].dropna().to_numpy(dtype=float)
            notc1 = trades_with_features.loc[trades_with_features["cluster_id"] != 1, f].dropna().to_numpy(dtype=float)
            if c1.size >= 2 and c1.std(ddof=1) > 0:
                ax.plot(x_grid, sps.gaussian_kde(c1)(x_grid),
                        color=CLUSTER_COLORS[1], linewidth=2.0, label="c1")
            if notc1.size >= 2 and notc1.std(ddof=1) > 0:
                ax.plot(x_grid, sps.gaussian_kde(notc1)(x_grid),
                        color="#999999", linewidth=1.5, linestyle="--", label="not-c1")
            ax.legend(fontsize=8)
        ax.set_title(f"{f} — c1 vs not-c1 KDE", fontsize=10)
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.tick_params(axis="both", labelsize=8)

    fig.suptitle("Arc 8 — Top-5 most-separating entry features (deep dive)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110, metadata={"Software": ""})
    plt.close(fig)


# ============================================================
# Summary writer
# ============================================================


def write_diagnostic_summary(
    out_path: Path,
    summary_ranked: pd.DataFrame,
    multiclass: Dict,
    long_overlap: pd.DataFrame,
    feature_cols: List[str],
) -> None:
    # Verdict from c1 precision@recall=0.60.
    p_at_r60 = multiclass.get("c1_precision_at_recall_060", float("nan"))
    if math.isnan(p_at_r60):
        verdict = "c1_NOT_SEPARABLE_AT_ENTRY"
        verdict_note = "c1 precision@recall=0.60 could not be computed (single-class fold)"
    elif p_at_r60 >= 0.40:
        verdict = "c1_SEPARABLE_AT_ENTRY"
        verdict_note = f"c1 precision@recall=0.60 = {p_at_r60:.4f} ≥ 0.40"
    elif p_at_r60 >= 0.20:
        verdict = "c1_MARGINAL"
        verdict_note = f"c1 precision@recall=0.60 = {p_at_r60:.4f} in [0.20, 0.40)"
    else:
        verdict = "c1_NOT_SEPARABLE_AT_ENTRY"
        verdict_note = f"c1 precision@recall=0.60 = {p_at_r60:.4f} < 0.20"

    # Hardest comparison from long overlap data.
    pairwise_means = long_overlap.groupby("comparison")["overlap_coef"].mean().sort_values(ascending=False)
    hardest = pairwise_means.index[0] if len(pairwise_means) else "n/a"
    hardest_mean = float(pairwise_means.iloc[0]) if len(pairwise_means) else float("nan")

    # Recommended next action.
    if verdict == "c1_SEPARABLE_AT_ENTRY":
        recommend = (
            "Draft v2.4 protocol amendment: replace Step 4 within-cluster success "
            "classifier with multiclass cluster-ID classifier. Use predicted "
            "cluster ∈ {c1} as the entry filter. Full-pool deployment economics "
            "evaluated at Step 4 (not Step 5) per Open-22/23/24 cross-arc finding. "
            "Tier 2 lift candidates remain optional."
        )
    elif verdict == "c1_MARGINAL":
        recommend = (
            "Multiclass classifier alone is insufficient. Consider EITHER: "
            "(a) multiclass + post-entry confirmation at D1 t≥3, OR "
            "(b) tighten PR-HHHL signal trigger to mechanically reduce c2-like dead "
            "trades at signal time (e.g. require trigger_close_pos ≥ 0.7 instead of 0.5, "
            "or require pullback_depth_atr ≥ 1.0 to exclude shallow pullbacks). "
            "Run a follow-up diagnostic on tightened signal pool to measure cluster "
            "rebalance before committing to (b)."
        )
    else:
        recommend = (
            "Entry-time prediction is structurally hard for c1 vs c0/c2/c3. "
            "Pursue post-entry confirmation: run D1 t=3 + t=5 with multiclass framing "
            "(predict cluster ID at bar t, not just within-c1 success). The path-shape "
            "features that DEFINE the clusters appear at t≥10 (peak_mfe location), so "
            "later t may discriminate where entry features cannot. Alternatively, "
            "tighten the PR-HHHL signal trigger to bypass cluster prediction entirely."
        )

    lines: List[str] = []
    lines.append("# Arc 8 — Entry-Feature Overlap Diagnostic — Summary")
    lines.append("")
    lines.append("> Post-Step-5 follow-up. Determines whether c1 is separable from "
                 "c0/c2/c3 at the entry bar using only entry-time features.")
    lines.append("")
    lines.append("## Headline verdict")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append(f"_{verdict_note}_")
    lines.append("")
    lines.append("## Key numbers")
    lines.append("")
    c1 = multiclass["per_class"].get(1, {})
    lines.append("- Multiclass RF on full 1,327 pool with 4 cluster labels")
    lines.append(f"- 5-fold TimeSeriesSplit out-of-fold ({multiclass['n_samples_oof']} test predictions)")
    lines.append(f"- OOF accuracy (4-class): **{multiclass['oof_accuracy']:.4f}**")
    lines.append(f"- c1 one-vs-rest AUC: **{c1.get('auc_one_vs_rest', float('nan')):.4f}** "
                 f"(base rate {c1.get('base_rate', float('nan')):.4f})")
    lines.append(f"- c1 precision: {c1.get('precision', float('nan')):.4f}, "
                 f"recall: {c1.get('recall', float('nan')):.4f}, "
                 f"f1: {c1.get('f1', float('nan')):.4f} "
                 f"(at default RF prediction threshold)")
    lines.append(f"- **c1 precision@recall=0.60 (one-vs-rest, swept threshold): "
                 f"{p_at_r60:.4f}** (decision metric)")
    lines.append(f"  - threshold at this op-point: "
                 f"{multiclass.get('c1_threshold_at_recall_060', float('nan')):.4f}")
    lines.append("")
    lines.append("## Cross-cluster comparison summary")
    lines.append("")
    lines.append("Mean overlap coefficient across all 18 features, by pairwise comparison "
                 "(higher = more overlap = harder to separate at entry):")
    lines.append("")
    lines.append("| Comparison | Mean overlap (across 18 features) |")
    lines.append("|---|---:|")
    for comp, val in pairwise_means.items():
        lines.append(f"| {comp} | {val:.4f} |")
    lines.append("")
    lines.append(f"**Hardest pair:** `{hardest}` (mean overlap {hardest_mean:.4f})")
    lines.append("")
    lines.append("## Univariate top-5 separating features (lowest mean overlap coefficient)")
    lines.append("")
    lines.append("| Rank | Feature | Mean overlap | Min overlap | Mean KS stat | Verdict |")
    lines.append("|---:|---|---:|---:|---:|:---:|")
    for i, row in summary_ranked.head(5).iterrows():
        lines.append(f"| {i+1} | `{row['feature']}` | {row['mean_overlap_coef']:.4f} "
                     f"| {row['min_overlap_coef']:.4f} | {row['mean_ks_stat']:.4f} "
                     f"| {row['verdict']} |")
    lines.append("")
    lines.append("## Univariate bottom-5 (most-overlapping = least separating)")
    lines.append("")
    lines.append("| Rank | Feature | Mean overlap | Verdict |")
    lines.append("|---:|---|---:|:---:|")
    for i, row in summary_ranked.tail(5).iterrows():
        rank = len(summary_ranked) - (len(summary_ranked) - 1 - i)
        lines.append(f"| {rank} | `{row['feature']}` | {row['mean_overlap_coef']:.4f} | {row['verdict']} |")
    lines.append("")
    lines.append("## Verdict distribution (per-feature)")
    lines.append("")
    counts = summary_ranked["verdict"].value_counts()
    for v in ["separating", "weak", "indistinguishable"]:
        n = int(counts.get(v, 0))
        lines.append(f"- **{v}**: {n} / {len(summary_ranked)} features")
    lines.append("")
    lines.append("## Confusion matrix (out-of-fold, 4 classes)")
    lines.append("")
    cm = multiclass["confusion_matrix"]
    labels = cm["labels"]
    lines.append("Rows = true cluster, columns = predicted cluster.")
    lines.append("")
    header = "| true \\ pred | " + " | ".join(f"c{c}" for c in labels) + " | row sum |"
    sep = "|---|" + "|".join(["---:"] * (len(labels) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for i, c in enumerate(labels):
        row = cm["rows_true_cols_pred"][i]
        rs = sum(row)
        lines.append("| c" + str(c) + " | " + " | ".join(str(x) for x in row) + f" | {rs} |")
    lines.append("")
    lines.append("Per-class metrics (out-of-fold):")
    lines.append("")
    lines.append("| Class | Base rate | OOF Precision | OOF Recall | F1 | AUC (1-vs-rest) |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for c in labels:
        pc = multiclass["per_class"][int(c)]
        lines.append(f"| c{c} | {pc['base_rate']:.4f} | {pc['precision']:.4f} "
                     f"| {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['auc_one_vs_rest']:.4f} |")
    lines.append("")
    lines.append("## Recommended next action")
    lines.append("")
    lines.append(recommend)
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `entry_feature_univariate_overlap.csv` — long-format per-feature × comparison")
    lines.append("- `entry_feature_separation_ranked.csv` — one row per feature, ranked by mean overlap")
    lines.append("- `multiclass_diagnostic_results.json` — full multiclass output")
    lines.append("- `multiclass_confusion_matrix.csv` — 4×4 confusion matrix")
    lines.append("- `entry_feature_distributions_by_cluster.png` — 6×3 KDE grid all 18 features")
    lines.append("- `top5_separating_features_detail.png` — deep dive on top-5 most-separating features")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Driver
# ============================================================


def main() -> int:
    print("[diagnostic] Loading inputs...", file=sys.stderr)
    trades = pd.read_csv(_REPO_ROOT / "results/l_arc_8/step1_verbatim/trades_all.csv")
    clusters = pd.read_csv(_REPO_ROOT / "results/l_arc_8/step2/clusters_K4.csv")

    # Row count consistency check.
    if len(trades) != len(clusters):
        print(f"[HALT] trades_all.csv n_rows={len(trades)} != clusters_K4.csv n_rows={len(clusters)}",
              file=sys.stderr)
        return 1

    # Merge cluster ID into trades.
    trades_with = trades.merge(clusters, on="trade_id", how="left")
    if trades_with["cluster_id"].isna().any():
        missing = int(trades_with["cluster_id"].isna().sum())
        print(f"[HALT] {missing} trades missing cluster_id after merge", file=sys.stderr)
        return 1

    # Verify all 10 PR-HHHL features are present.
    arc8_required = list(PIPELINE_E_ARC8_FEATURES)
    missing_arc8 = [c for c in arc8_required if c not in trades_with.columns]
    if missing_arc8:
        print(f"[HALT] missing PR-HHHL features in trades_all.csv: {missing_arc8}", file=sys.stderr)
        return 1
    print(f"[diagnostic] PR-HHHL features present: {len(arc8_required)}", file=sys.stderr)

    # Compute 8 base entry features via pair caches.
    print("[diagnostic] Computing 8 base entry features from 4H bars...", file=sys.stderr)
    pairs = sorted(trades_with["pair"].unique())
    pair_caches = {p: _build_pair_cache(p, "data/4hr") for p in pairs}
    base_e_df = compute_base_e_features(trades_with, pair_caches)
    trades_with = trades_with.merge(base_e_df, on="trade_id", how="left")

    feature_cols = list(PIPELINE_E_BASE_FEATURES) + list(PIPELINE_E_ARC8_FEATURES)
    print(f"[diagnostic] Feature set: {len(feature_cols)} features "
          f"(8 base + {len(PIPELINE_E_ARC8_FEATURES)} PR-HHHL)", file=sys.stderr)

    # Variance check.
    for f in feature_cols:
        v = trades_with[f].dropna().to_numpy(dtype=float)
        if v.size > 0 and v.std(ddof=1 if v.size > 1 else 0) == 0:
            print(f"[HALT] feature {f!r} is constant in the full pool (zero variance)",
                  file=sys.stderr)
            return 1

    # Part A.
    print("[diagnostic] Part A: univariate overlap...", file=sys.stderr)
    long_overlap, summary_ranked = part_a_univariate(trades_with, feature_cols)
    long_overlap.to_csv(OUT_DIR / "entry_feature_univariate_overlap.csv", index=False, lineterminator="\n")
    summary_ranked.to_csv(OUT_DIR / "entry_feature_separation_ranked.csv", index=False, lineterminator="\n")
    print("[diagnostic] Part A: top 5 most-separating features:", file=sys.stderr)
    for i, row in summary_ranked.head(5).iterrows():
        print(f"  {i+1}. {row['feature']}: mean_overlap={row['mean_overlap_coef']:.4f} "
              f"verdict={row['verdict']}", file=sys.stderr)

    # Part B.
    print("[diagnostic] Part B: multiclass RF (4 clusters)...", file=sys.stderr)
    multiclass = part_b_multiclass(trades_with, feature_cols)
    (OUT_DIR / "multiclass_diagnostic_results.json").write_text(
        json.dumps(multiclass, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    # Confusion matrix CSV.
    cm = multiclass["confusion_matrix"]
    with (OUT_DIR / "multiclass_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["true_cluster"] + [f"pred_c{c}" for c in cm["labels"]] + ["row_sum"])
        for i, c in enumerate(cm["labels"]):
            row = cm["rows_true_cols_pred"][i]
            w.writerow([f"c{c}"] + [str(x) for x in row] + [str(sum(row))])
    c1 = multiclass["per_class"].get(1, {})
    print(f"  c1 one-vs-rest AUC: {c1.get('auc_one_vs_rest', float('nan')):.4f}", file=sys.stderr)
    print(f"  c1 precision@recall=0.60: {multiclass.get('c1_precision_at_recall_060', float('nan')):.4f}",
          file=sys.stderr)
    print(f"  4-class OOF accuracy: {multiclass['oof_accuracy']:.4f}", file=sys.stderr)

    # Part C.
    print("[diagnostic] Part C: 6x3 KDE grid...", file=sys.stderr)
    part_c_kde_grid(trades_with, feature_cols,
                    OUT_DIR / "entry_feature_distributions_by_cluster.png")

    # Part D.
    top5 = summary_ranked.head(5)["feature"].tolist()
    print(f"[diagnostic] Part D: top-5 deep dive ({top5})...", file=sys.stderr)
    part_d_top5_detail(trades_with, top5, OUT_DIR / "top5_separating_features_detail.png")

    # Summary.
    print("[diagnostic] Writing DIAGNOSTIC_SUMMARY.md...", file=sys.stderr)
    write_diagnostic_summary(OUT_DIR / "DIAGNOSTIC_SUMMARY.md",
                              summary_ranked, multiclass, long_overlap, feature_cols)
    print("[diagnostic] DONE.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
