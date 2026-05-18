"""Arc 8 — Path 1 Diagnostic: Post-entry confirmation t-sweep.

Tests whether cluster membership becomes predictable as path-so-far
information accumulates at bar offsets t ∈ {3, 5, 8, 12}.

For each t:
  - Build 25-feature set (8 base entry + 10 PR-HHHL + 7 path-so-far at t)
  - Multiclass RF predicting cluster ∈ {0, 1, 2, 3} with 5-fold TimeSeriesSplit
  - Report c1 one-vs-rest AUC, c1 precision@recall=0.60, per-class metrics,
    confusion matrix, top-5 feature importances
  - Survivorship per cluster (eligibility under SL=4.0×ATR — c1's deploy SL)

Plus an economic overlay on c1 trades:
  - Fraction of c1 trades with peak MFE already reached by bar t
  - mfe_so_far_r at t (mean, p50) — how much of eventual MFE consumed
  - mae_so_far_r at t (p50, p95) — drawdown depth by t
  - Implied slippage = c1 eventual mfe_p50 − c1 mfe_so_far_p50 at t
    (in unit R = SL=4.0×ATR frame)

random_state=42, n_jobs=1 throughout. Deterministic.

Usage:
    py scripts/l_arc_8/diagnostics/post_entry_confirmation.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.l_arc_8.step3_capturability import _eval_trade_at_sl  # noqa: E402
from scripts.l_arc_8.step4_extractability import (  # noqa: E402
    PIPELINE_E_BASE_FEATURES,
    PIPELINE_E_ARC8_FEATURES,
    _build_pair_cache,
    _impute_nans,
    compute_base_e_features,
    compute_d1_features_at_t,
)

OUT_DIR = _REPO_ROOT / "results" / "l_arc_8" / "diagnostics" / "post_entry_confirmation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR = OUT_DIR / "post_entry_confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)

T_SWEEP: List[int] = [3, 5, 8, 12]
UNIT_SL: float = 4.0
ORIGINAL_SL: float = 2.0
SCALE_TO_UNIT: float = ORIGINAL_SL / UNIT_SL  # 0.5

PATH_SO_FAR_FEATURES: List[str] = [
    "close_r_at_t", "mfe_so_far_r_at_t", "mae_so_far_r_at_t",
    "bars_in_profit_at_t", "local_peaks_so_far_at_t",
    "monotonicity_so_far_at_t", "velocity_first_t",
]


def _build_paths_index(paths_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    paths_sorted = paths_df.sort_values(["trade_id", "bar_offset"], kind="mergesort")
    for tid, g in paths_sorted.groupby("trade_id", sort=True):
        out[int(tid)] = g.reset_index(drop=True)
    return out


def run_multiclass_at_t(
    trades_with: pd.DataFrame, paths_index: Dict[int, pd.DataFrame],
    base_e_with: pd.DataFrame, all_feature_cols: List[str], t: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Multiclass RF at bar t. Returns (metrics_dict, eligibility_mask)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, precision_recall_curve,
        precision_recall_fscore_support, roc_auc_score,
    )
    from sklearn.model_selection import TimeSeriesSplit

    # Sort by entry_time for time-ordered CV.
    df = trades_with.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df = df.sort_values("entry_time", kind="mergesort").reset_index(drop=True)

    # Compute path-so-far features at t and eligibility mask.
    d1_feat, eligible = compute_d1_features_at_t(
        df, paths_index, t, UNIT_SL, ORIGINAL_SL
    )

    # Merge base+arc8 entry features (we already have these in base_e_with).
    base_for_df = base_e_with.set_index("trade_id").loc[df["trade_id"].to_numpy()].reset_index()

    # Combine: base + d1_feat (aligned by trade_id).
    merged = base_for_df.merge(d1_feat, on="trade_id")
    merged = merged.set_index("trade_id").loc[df["trade_id"].to_numpy()].reset_index()

    X_all = merged[all_feature_cols]
    y_all = df["cluster_id"].to_numpy(dtype=int)

    # Apply eligibility filter.
    X = _impute_nans(X_all.iloc[eligible])
    y = y_all[eligible]
    classes = sorted(np.unique(y).tolist())

    survivorship = {
        int(c): {
            "total": int((y_all == c).sum()),
            "eligible_at_t": int(((y_all == c) & eligible).sum()),
        }
        for c in [0, 1, 2, 3]
    }

    if len(y) < 50 or len(classes) < 2:
        return ({
            "t": t,
            "n_total": int(len(y_all)),
            "n_eligible": int(eligible.sum()),
            "classes_present": classes,
            "survivorship": survivorship,
            "error": "Insufficient eligible data for multiclass training",
        }, eligible)

    tss = TimeSeriesSplit(n_splits=5)
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros((len(y), len(classes)), dtype=float)
    per_fold_acc: List[float] = []
    feat_importances_sum = np.zeros(len(all_feature_cols), dtype=float)
    feat_importance_folds = 0
    for fold_n, (tr_idx, te_idx) in enumerate(tss.split(X)):
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=1,
            class_weight="balanced",
        )
        if len(np.unique(y[tr_idx])) < 2:
            continue
        clf.fit(X[tr_idx], y[tr_idx])
        pred = clf.predict(X[te_idx])
        prob = clf.predict_proba(X[te_idx])
        # Align prob columns to global classes order.
        prob_aligned = np.zeros((prob.shape[0], len(classes)), dtype=float)
        for i, c in enumerate(clf.classes_):
            if c in classes:
                prob_aligned[:, classes.index(c)] = prob[:, i]
        oof_pred[te_idx] = pred
        oof_proba[te_idx, :] = prob_aligned
        per_fold_acc.append(float(accuracy_score(y[te_idx], pred)))
        feat_importances_sum += clf.feature_importances_
        feat_importance_folds += 1

    mask = oof_pred >= 0
    y_oof = y[mask]
    pred_oof = oof_pred[mask]
    proba_oof = oof_proba[mask]

    p, r, f1, support = precision_recall_fscore_support(
        y_oof, pred_oof, labels=classes, zero_division=0
    )
    accuracy = float(accuracy_score(y_oof, pred_oof))
    conf_mat = confusion_matrix(y_oof, pred_oof, labels=classes)

    per_class_auc: Dict[int, float] = {}
    for i, c in enumerate(classes):
        y_bin = (y_oof == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            per_class_auc[c] = float("nan")
            continue
        try:
            per_class_auc[c] = float(roc_auc_score(y_bin, proba_oof[:, i]))
        except Exception:
            per_class_auc[c] = float("nan")

    # c1 precision@recall=0.60.
    c1_p_at_r60 = float("nan")
    c1_thr_at_r60 = float("nan")
    if 1 in classes:
        c1_idx = classes.index(1)
        y_bin = (y_oof == 1).astype(int)
        if len(np.unique(y_bin)) == 2:
            prec, rec, thr = precision_recall_curve(y_bin, proba_oof[:, c1_idx])
            valid_idxs = np.where(rec >= 0.60)[0]
            if valid_idxs.size > 0:
                best = valid_idxs[np.argmax(prec[valid_idxs])]
                c1_p_at_r60 = float(prec[best])
                if best < len(thr):
                    c1_thr_at_r60 = float(thr[best])

    # Top-5 feature importances (mean across folds).
    if feat_importance_folds > 0:
        mean_imp = feat_importances_sum / feat_importance_folds
        top5_idx = np.argsort(mean_imp)[::-1][:5]
        top5 = [(all_feature_cols[i], float(mean_imp[i])) for i in top5_idx]
    else:
        top5 = []

    return ({
        "t": t,
        "n_total": int(len(y_all)),
        "n_eligible": int(eligible.sum()),
        "eligibility_pct": float(eligible.sum() / len(y_all)),
        "classes_present": classes,
        "survivorship": survivorship,
        "per_fold_accuracy": per_fold_acc,
        "oof_accuracy": accuracy,
        "per_class": {
            int(c): {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "auc_one_vs_rest": per_class_auc[c],
            } for i, c in enumerate(classes)
        },
        "c1_precision_at_recall_060": c1_p_at_r60,
        "c1_threshold_at_recall_060": c1_thr_at_r60,
        "confusion_matrix": {
            "labels": classes,
            "rows_true_cols_pred": conf_mat.tolist(),
        },
        "top5_feature_importances": top5,
    }, eligible)


def c1_economic_overlay(
    trades_with: pd.DataFrame, paths_index: Dict[int, pd.DataFrame], t: int,
) -> Dict[str, Any]:
    """Economic cost of waiting to bar t for c1 trades.

    All values in unit R-frame (SL=4.0×ATR). Reports:
      - c1 trades with peak MFE already reached by bar t
      - mfe_so_far_r at t (mean, p50)
      - mae_so_far_r at t (p50, p95)
      - Eventual mfe_r (mean, p50) for c1 under SL=4.0
      - Implied slippage: mfe_p50 - mfe_so_far_p50 at t
    """
    c1 = trades_with[trades_with["cluster_id"] == 1].copy()
    c1["time_to_peak_mfe"] = c1["time_to_peak_mfe"].astype(int)

    # Recompute eventual mfe under unit SL.
    eventual_mfe_unit: List[float] = []
    mfe_at_t_unit: List[float] = []
    mae_at_t_unit: List[float] = []
    peak_reached_by_t: List[bool] = []
    eligible_at_t: List[bool] = []
    actual_peak_bar_under_unit: List[int] = []
    for _, tr in c1.iterrows():
        path = paths_index[int(tr["trade_id"])]
        # Eligibility under SL=4.0×ATR.
        ev = _eval_trade_at_sl(path, UNIT_SL, ORIGINAL_SL)
        eligible = ev.truncated_at_bar >= t
        eligible_at_t.append(eligible)
        eventual_mfe_unit.append(float(ev.fwd_mfe_new_r))
        actual_peak_bar_under_unit.append(int(ev.peak_mfe_bar))
        # mfe_so_far at t — scale original to unit.
        ps = path.sort_values("bar_offset", kind="mergesort").reset_index(drop=True)
        slice_end = t + 1
        if slice_end > len(ps):
            slice_end = len(ps)
        seg = ps.iloc[:slice_end]
        if len(seg) == 0:
            mfe_at_t_unit.append(0.0)
            mae_at_t_unit.append(0.0)
            peak_reached_by_t.append(False)
            continue
        mfe_orig_at_t = float(seg["mfe_so_far_r"].iloc[-1])
        mae_orig_at_t = float(seg["mae_so_far_r"].iloc[-1])
        mfe_at_t_unit.append(mfe_orig_at_t * SCALE_TO_UNIT)
        mae_at_t_unit.append(mae_orig_at_t * SCALE_TO_UNIT)
        peak_reached_by_t.append(int(ev.peak_mfe_bar) <= t)

    eventual_arr = np.array(eventual_mfe_unit, dtype=float)
    mfe_t_arr = np.array(mfe_at_t_unit, dtype=float)
    mae_t_arr = np.array(mae_at_t_unit, dtype=float)
    elig_arr = np.array(eligible_at_t, dtype=bool)
    peak_arr = np.array(peak_reached_by_t, dtype=bool)
    peak_bar_arr = np.array(actual_peak_bar_under_unit, dtype=int)

    eventual_mfe_p50 = float(np.percentile(eventual_arr, 50))
    mfe_so_far_p50_at_t = float(np.percentile(mfe_t_arr[elig_arr], 50)) if elig_arr.sum() > 0 else float("nan")
    implied_slippage_r = eventual_mfe_p50 - mfe_so_far_p50_at_t if not math.isnan(mfe_so_far_p50_at_t) else float("nan")

    return {
        "t": t,
        "c1_n_total": int(len(c1)),
        "c1_n_eligible_at_t": int(elig_arr.sum()),
        "c1_eligibility_pct": float(elig_arr.mean()),
        "c1_pct_peak_reached_by_t": float(peak_arr.mean()),
        "c1_peak_bar_p50": float(np.percentile(peak_bar_arr, 50)),
        "c1_peak_bar_p95": float(np.percentile(peak_bar_arr, 95)),
        "c1_mfe_so_far_at_t_mean": float(mfe_t_arr[elig_arr].mean()) if elig_arr.sum() > 0 else float("nan"),
        "c1_mfe_so_far_at_t_p50": mfe_so_far_p50_at_t,
        "c1_mae_so_far_at_t_p50": float(np.percentile(mae_t_arr[elig_arr], 50)) if elig_arr.sum() > 0 else float("nan"),
        "c1_mae_so_far_at_t_p95": float(np.percentile(mae_t_arr[elig_arr], 95)) if elig_arr.sum() > 0 else float("nan"),
        "c1_eventual_mfe_p50_unit": eventual_mfe_p50,
        "c1_eventual_mfe_mean_unit": float(eventual_arr.mean()),
        "c1_implied_slippage_r": implied_slippage_r,
        "c1_pct_eventual_mfe_consumed_at_t_p50": (
            mfe_so_far_p50_at_t / eventual_mfe_p50 if eventual_mfe_p50 > 0 else float("nan")
        ),
    }


def write_separability_plot(
    t_results: List[Dict[str, Any]], overlay_results: List[Dict[str, Any]], out_path: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [r["t"] for r in t_results]
    aucs = [r["per_class"][1]["auc_one_vs_rest"] if "per_class" in r and 1 in r["per_class"] else float("nan") for r in t_results]
    p_at_r60 = [r.get("c1_precision_at_recall_060", float("nan")) for r in t_results]
    slippages = [o["c1_implied_slippage_r"] for o in overlay_results]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("bar offset t")
    ax1.set_ylabel("classifier metric (c1)", color="#1f77b4")
    l1 = ax1.plot(ts, aucs, "o-", color="#1f77b4", label="c1 1-vs-rest AUC", linewidth=2)
    l2 = ax1.plot(ts, p_at_r60, "s--", color="#2ca02c", label="c1 precision@recall=0.60", linewidth=2)
    ax1.axhline(0.65, color="#1f77b4", linewidth=0.5, linestyle=":", alpha=0.5)
    ax1.axhline(0.40, color="#2ca02c", linewidth=0.5, linestyle=":", alpha=0.5)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, 1)
    ax1.grid(True, linewidth=0.3, alpha=0.4)

    ax2 = ax1.twinx()
    ax2.set_ylabel("c1 implied slippage (R, unit SL=4×ATR)", color="#d62728")
    l3 = ax2.plot(ts, slippages, "^-", color="#d62728", label="c1 implied slippage (R)", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=9)

    ax1.set_title("Arc 8 — post-entry confirmation: c1 separability vs slippage by t\n"
                  "(dotted: AUC 0.65 / precision 0.40 viability thresholds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, metadata={"Software": ""})
    plt.close(fig)


def main() -> int:
    print("[path1] Loading inputs...", file=sys.stderr)
    trades = pd.read_csv(_REPO_ROOT / "results/l_arc_8/step1_verbatim/trades_all.csv")
    paths_df = pd.read_csv(_REPO_ROOT / "results/l_arc_8/step1_verbatim/trades_paths.csv")
    clusters = pd.read_csv(_REPO_ROOT / "results/l_arc_8/step2/clusters_K4.csv")

    if len(trades) != len(clusters):
        print(f"[HALT] row count mismatch: trades {len(trades)} vs clusters {len(clusters)}",
              file=sys.stderr)
        return 1

    trades_with = trades.merge(clusters, on="trade_id", how="left")
    if trades_with["cluster_id"].isna().any():
        print(f"[HALT] {trades_with['cluster_id'].isna().sum()} trades missing cluster_id",
              file=sys.stderr)
        return 1

    paths_index = _build_paths_index(paths_df)

    # Build base entry features (8 base) — re-use Step 4 logic.
    print("[path1] Building base entry features (8 base + 10 PR-HHHL pre-existing)...",
          file=sys.stderr)
    pairs = sorted(trades_with["pair"].unique())
    pair_caches = {p: _build_pair_cache(p, "data/4hr") for p in pairs}
    base_e_df = compute_base_e_features(trades_with, pair_caches)
    # Merge PR-HHHL features.
    base_e_with = base_e_df.merge(
        trades_with[["trade_id"] + list(PIPELINE_E_ARC8_FEATURES)],
        on="trade_id", how="left",
    )

    all_feature_cols = (
        list(PIPELINE_E_BASE_FEATURES)
        + list(PIPELINE_E_ARC8_FEATURES)
        + PATH_SO_FAR_FEATURES
    )
    assert len(all_feature_cols) == 25, f"expected 25 features, got {len(all_feature_cols)}"
    print(f"[path1] Feature set: {len(all_feature_cols)} features (8 base + 10 PR-HHHL + 7 path-so-far)",
          file=sys.stderr)

    # t-sweep.
    t_results: List[Dict[str, Any]] = []
    overlay_results: List[Dict[str, Any]] = []
    per_t_top5: List[Dict[str, Any]] = []

    for t in T_SWEEP:
        print(f"[path1] === t = {t} ===", file=sys.stderr)
        res, _eligible = run_multiclass_at_t(
            trades_with, paths_index, base_e_with, all_feature_cols, t
        )
        t_results.append(res)
        if "error" not in res:
            c1_metrics = res["per_class"].get(1, {})
            print(f"  t={t}: eligible={res['n_eligible']}/{res['n_total']} "
                  f"({res['eligibility_pct']:.3f}); c1 AUC={c1_metrics.get('auc_one_vs_rest', float('nan')):.4f}; "
                  f"c1 prec@rec=0.60={res.get('c1_precision_at_recall_060', float('nan')):.4f}; "
                  f"4-class acc={res['oof_accuracy']:.4f}", file=sys.stderr)
        else:
            print(f"  t={t}: {res['error']}", file=sys.stderr)

        # Economic overlay (c1 only).
        overlay = c1_economic_overlay(trades_with, paths_index, t)
        overlay_results.append(overlay)
        print(f"  t={t}: c1 mfe_so_far p50={overlay['c1_mfe_so_far_at_t_p50']:.3f} (eventual p50 "
              f"{overlay['c1_eventual_mfe_p50_unit']:.3f}); peak_reached_pct={overlay['c1_pct_peak_reached_by_t']:.3f}",
              file=sys.stderr)

        # Confusion matrix CSV per t.
        if "confusion_matrix" in res and "rows_true_cols_pred" in res["confusion_matrix"]:
            cm = res["confusion_matrix"]
            labels = cm["labels"]
            with (CM_DIR / f"confusion_matrix_t{t}.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(["true_cluster"] + [f"pred_c{c}" for c in labels])
                for i, c in enumerate(labels):
                    w.writerow([f"c{c}"] + [str(x) for x in cm["rows_true_cols_pred"][i]])

        # Feature importances row.
        for rank, (feat, imp) in enumerate(res.get("top5_feature_importances", []), start=1):
            per_t_top5.append({"t": t, "rank": rank, "feature": feat, "importance": imp})

    # Write per-t metric CSV.
    sweep_rows: List[Dict[str, Any]] = []
    for r in t_results:
        if "error" in r:
            sweep_rows.append({
                "t": r["t"], "n_eligible": r["n_eligible"], "n_total": r["n_total"],
                "error": r["error"],
            })
            continue
        c0 = r["per_class"].get(0, {})
        c1 = r["per_class"].get(1, {})
        c2 = r["per_class"].get(2, {})
        c3 = r["per_class"].get(3, {})
        s = r["survivorship"]
        sweep_rows.append({
            "t": r["t"], "n_total": r["n_total"], "n_eligible": r["n_eligible"],
            "eligibility_pct": r["eligibility_pct"],
            "oof_accuracy_4class": r["oof_accuracy"],
            "c1_auc_one_vs_rest": c1.get("auc_one_vs_rest", float("nan")),
            "c1_precision_at_recall_060": r.get("c1_precision_at_recall_060", float("nan")),
            "c1_precision_default_thr": c1.get("precision", float("nan")),
            "c1_recall_default_thr": c1.get("recall", float("nan")),
            "c1_f1_default_thr": c1.get("f1", float("nan")),
            "c0_auc": c0.get("auc_one_vs_rest", float("nan")),
            "c2_auc": c2.get("auc_one_vs_rest", float("nan")),
            "c3_auc": c3.get("auc_one_vs_rest", float("nan")),
            "c0_survivorship_pct": s[0]["eligible_at_t"] / s[0]["total"] if s[0]["total"] else 0.0,
            "c1_survivorship_pct": s[1]["eligible_at_t"] / s[1]["total"] if s[1]["total"] else 0.0,
            "c2_survivorship_pct": s[2]["eligible_at_t"] / s[2]["total"] if s[2]["total"] else 0.0,
            "c3_survivorship_pct": s[3]["eligible_at_t"] / s[3]["total"] if s[3]["total"] else 0.0,
            "error": "",
        })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(OUT_DIR / "post_entry_t_sweep_results.csv", index=False, lineterminator="\n")

    overlay_df = pd.DataFrame(overlay_results)
    overlay_df.to_csv(OUT_DIR / "post_entry_c1_economic_overlay.csv", index=False, lineterminator="\n")

    importances_df = pd.DataFrame(per_t_top5)
    importances_df.to_csv(OUT_DIR / "post_entry_feature_importances.csv", index=False, lineterminator="\n")

    # Separability plot.
    write_separability_plot(t_results, overlay_results, OUT_DIR / "post_entry_t_separability_plot.png")

    # Verdict.
    valid_p = [r.get("c1_precision_at_recall_060", float("nan")) for r in t_results if "error" not in r]
    best_p = max([p for p in valid_p if not math.isnan(p)], default=float("nan"))
    best_t_idx = next((i for i, r in enumerate(t_results)
                       if "error" not in r and r.get("c1_precision_at_recall_060", float("nan")) == best_p),
                      None)
    best_t = T_SWEEP[best_t_idx] if best_t_idx is not None else None

    if math.isnan(best_p):
        verdict = "PATH_1_DEAD"
    elif best_p >= 0.40:
        verdict = "PATH_1_VIABLE"
    elif best_p >= 0.20:
        verdict = "PATH_1_MARGINAL"
    else:
        verdict = "PATH_1_DEAD"

    # Slippage downgrade check.
    economic_downgrade = False
    if best_t is not None and best_t_idx is not None:
        overlay = overlay_results[best_t_idx]
        consumed = overlay.get("c1_pct_eventual_mfe_consumed_at_t_p50", float("nan"))
        if (not math.isnan(consumed)) and consumed > 0.50 and verdict == "PATH_1_VIABLE":
            verdict = "PATH_1_MARGINAL"
            economic_downgrade = True

    summary_payload = {
        "verdict": verdict,
        "best_t": best_t,
        "best_c1_precision_at_recall_060": best_p if not math.isnan(best_p) else None,
        "economic_downgrade_applied": economic_downgrade,
        "t_sweep": T_SWEEP,
        "per_t_summary": [{
            "t": r["t"],
            "c1_auc_one_vs_rest": (r["per_class"][1]["auc_one_vs_rest"] if "per_class" in r and 1 in r["per_class"] else None),
            "c1_precision_at_recall_060": r.get("c1_precision_at_recall_060"),
            "c1_eligibility_pct": (
                r["survivorship"][1]["eligible_at_t"] / r["survivorship"][1]["total"] if r.get("survivorship") and r["survivorship"][1]["total"] else None
            ),
            "c1_mfe_consumed_pct_at_t_p50": overlay_results[i].get("c1_pct_eventual_mfe_consumed_at_t_p50"),
            "c1_implied_slippage_r": overlay_results[i]["c1_implied_slippage_r"],
        } for i, r in enumerate(t_results)],
    }
    (OUT_DIR / "PATH1_VERDICT.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"\n[path1] Verdict: {verdict}", file=sys.stderr)
    print(f"  best_t = {best_t}, c1 precision@recall=0.60 at best_t = {best_p:.4f}", file=sys.stderr)
    if economic_downgrade:
        print(f"  (downgraded from PATH_1_VIABLE → PATH_1_MARGINAL due to >50% MFE consumed by t)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
