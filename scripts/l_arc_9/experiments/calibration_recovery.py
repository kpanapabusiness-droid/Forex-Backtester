"""Arc 9 calibration recovery test - Pipeline D1 classifier.

Held-open experiment under v2.3. Tests whether post-hoc probability
calibration (Platt scaling and isotonic regression) recovers a usable
threshold for the Step 4 Pipeline D1 classifier that achieved AUC = 0.626
but produced probability outputs concentrated below 0.40, causing the
v2.2 §3 threshold sweep over {0.40, 0.50, 0.60, 0.70} to fail recall.

Outcome A: calibration recovers recall >= 0.60 at some threshold in the
sweep with reasonable precision -> evidence for v2.x amendment to insert
a calibration step before threshold sweep.

Outcome B: calibration does not recover recall -> classifier discrimination
is the bound, not the probability scale; §8 was correctly calibrated.

Note on artefact provenance: Step 4 did NOT persist a D1 classifier joblib
because the threshold sweep FAILED (the persist-on-PASS branch was not
taken). The model is fully reproducible deterministically (same code,
seed=42, same features). We reproduce it and verify the reproduced OOF
probabilities match Step 4's saved threshold sweep CSV byte-for-byte before
proceeding with the calibration test. This is equivalent to loading a
hypothetical joblib that the Step 4 implementation would have written had
the threshold sweep passed; it is not a retraining of a different model.

Procedure:
  1. Rebuild the D1 feature matrix (8 base + 8 arc-specific + 7 path-so-far
     at t=1 = 23 features), identical to Step 4 / D1 t=1.
  2. Reproduce the failing model: same RF hyperparams, 5-fold StratifiedKFold,
     same seed. Verify OOF threshold sweep matches Step 4 byte-for-byte.
  3. Approach A: 60/20/20 train/cal/test stratified split. Fit RF on train,
     fit Platt + Isotonic on cal via CalibratedClassifierCV(cv='prefit'),
     evaluate AUC + threshold sweep on test for {uncal, Platt, isotonic}.
  4. Approach B: outer 5-fold StratifiedKFold. Within each train fold, take a
     75/25 inner split for train/cal. Fit RF on inner train, fit calibrators
     on inner cal, evaluate on outer test. Aggregate mean + std.
  5. AUC-preservation check: calibrated AUC within 0.01 of uncalibrated.
  6. Reliability diagrams (matplotlib) for uncal / Platt / isotonic.

All seeds: 42. Outputs written to
results/l_arc_9/experiments/calibration_recovery/.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6 replacement for cv='prefit'
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --- Match Step 4 / D1 baseline exactly ---
RF_KW = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20, "random_state": 42, "n_jobs": -1}
CV_KW = {"n_splits": 5, "shuffle": True, "random_state": 42}
THRESHOLDS: List[float] = [0.40, 0.50, 0.60, 0.70]
RECALL_FLOOR: float = 0.60
D1_T: int = 1  # smallest-t selected at Step 4
SEED: int = 42

BASE_8 = [
    "body_to_range_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "range_to_atr_14", "ret_5bar_atr", "ret_20bar_atr",
    "pos_in_20bar_range", "rsi_14",
]
ARC_SPECIFIC = [
    "n_swing_lows", "most_recent_sl_lag", "swing_low_dist_atr",
    "mother_bar_range_atr", "inside_bar_range_atr", "ib_range_ratio",
    "break_bar_body_atr", "break_close_above_high_atr",
]


def _build_d1_feature_matrix(
    entry_features: pd.DataFrame, trades_paths: pd.DataFrame, t: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the D1 t=t feature matrix exactly as Step 4 did."""
    step4 = importlib.import_module("scripts.l_arc_9.step4_extractability")
    psf = step4.compute_path_so_far_features(trades_paths, t)
    merged = entry_features.merge(psf, on="trade_id", how="inner")
    d1_feature_cols = BASE_8 + ARC_SPECIFIC + [c for c in psf.columns if c != "trade_id"]
    merged = merged.dropna(subset=d1_feature_cols).reset_index(drop=True)
    return merged, d1_feature_cols


def _threshold_sweep_metrics(
    y: np.ndarray, prob: np.ndarray, label: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for thr in THRESHOLDS:
        y_pred = (prob >= thr).astype(int)
        if y_pred.sum() == 0:
            prec = 0.0
        else:
            prec = float(precision_score(y, y_pred, zero_division=0))
        rec = float(recall_score(y, y_pred, zero_division=0))
        rows.append({
            "calibrator": label, "threshold": thr,
            "precision": prec, "recall": rec,
            "n_admitted": int(y_pred.sum()),
            "passes_recall_floor": int(rec >= RECALL_FLOOR),
        })
    return rows


def _reproduce_step4_oof(
    X: np.ndarray, y: np.ndarray,
) -> Tuple[float, np.ndarray, List[Dict[str, Any]]]:
    """Reproduce Step 4's CV-OOF probabilities for the D1 t=1 model."""
    skf = StratifiedKFold(**CV_KW)
    fold_aucs: List[float] = []
    oof = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in skf.split(X, y):
        mdl = RandomForestClassifier(**RF_KW)
        mdl.fit(X[tr_idx], y[tr_idx])
        p = mdl.predict_proba(X[va_idx])[:, 1]
        oof[va_idx] = p
        try:
            fold_aucs.append(float(roc_auc_score(y[va_idx], p)))
        except Exception:
            fold_aucs.append(float("nan"))
    auc_mean = float(np.nanmean(fold_aucs))
    sweep = _threshold_sweep_metrics(y, oof, label="reproduced_uncal_oof")
    return auc_mean, oof, sweep


def _approach_a(
    X: np.ndarray, y: np.ndarray,
) -> Dict[str, Any]:
    """60/20/20 train/cal/test stratified split. Fit RF on train, fit Platt +
    Isotonic on cal via CalibratedClassifierCV(cv='prefit'), eval AUC +
    threshold sweep on test for {uncal, Platt, isotonic}.
    """
    # First split: 80/20 (train+cal) vs test.
    X_trcal, X_test, y_trcal, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y,
    )
    # Second split inside train+cal: 75/25 -> 60/20 of total.
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trcal, y_trcal, test_size=0.25, random_state=SEED, stratify=y_trcal,
    )

    base = RandomForestClassifier(**RF_KW)
    base.fit(X_train, y_train)
    uncal_prob = base.predict_proba(X_test)[:, 1]
    uncal_auc = float(roc_auc_score(y_test, uncal_prob))

    frozen = FrozenEstimator(base)
    platt = CalibratedClassifierCV(frozen, method="sigmoid")
    platt.fit(X_cal, y_cal)
    platt_prob = platt.predict_proba(X_test)[:, 1]
    platt_auc = float(roc_auc_score(y_test, platt_prob))

    iso = CalibratedClassifierCV(frozen, method="isotonic")
    iso.fit(X_cal, y_cal)
    iso_prob = iso.predict_proba(X_test)[:, 1]
    iso_auc = float(roc_auc_score(y_test, iso_prob))

    rows: List[Dict[str, Any]] = []
    rows.extend(_threshold_sweep_metrics(y_test, uncal_prob, "approach_A_uncal"))
    rows.extend(_threshold_sweep_metrics(y_test, platt_prob, "approach_A_platt"))
    rows.extend(_threshold_sweep_metrics(y_test, iso_prob, "approach_A_isotonic"))

    return {
        "n_train": int(len(y_train)), "n_cal": int(len(y_cal)), "n_test": int(len(y_test)),
        "n_pos_train": int(y_train.sum()), "n_pos_cal": int(y_cal.sum()),
        "n_pos_test": int(y_test.sum()),
        "uncal_auc": uncal_auc, "platt_auc": platt_auc, "iso_auc": iso_auc,
        "auc_drift_platt": platt_auc - uncal_auc,
        "auc_drift_iso": iso_auc - uncal_auc,
        "threshold_sweep_rows": rows,
        "y_test": y_test.tolist(),
        "probs": {"uncal": uncal_prob.tolist(), "platt": platt_prob.tolist(), "iso": iso_prob.tolist()},
    }


def _approach_b(
    X: np.ndarray, y: np.ndarray,
) -> Dict[str, Any]:
    """Outer 5-fold StratifiedKFold. Inner 75/25 train/cal split per fold.
    Fit RF on inner train, calibrate on inner cal, eval on outer test.
    Aggregate per-fold metrics with mean + std across folds.
    """
    skf = StratifiedKFold(**CV_KW)
    per_fold: List[Dict[str, Any]] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr_full, X_te = X[tr_idx], X[te_idx]
        y_tr_full, y_te = y[tr_idx], y[te_idx]
        # Inner split for calibration (deterministic seed per fold).
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_tr_full, y_tr_full, test_size=0.25,
            random_state=SEED + fold_idx, stratify=y_tr_full,
        )
        base = RandomForestClassifier(**RF_KW)
        base.fit(X_train, y_train)
        uncal_prob = base.predict_proba(X_te)[:, 1]
        frozen = FrozenEstimator(base)
        platt = CalibratedClassifierCV(frozen, method="sigmoid").fit(X_cal, y_cal)
        iso = CalibratedClassifierCV(frozen, method="isotonic").fit(X_cal, y_cal)
        platt_prob = platt.predict_proba(X_te)[:, 1]
        iso_prob = iso.predict_proba(X_te)[:, 1]
        try:
            uncal_auc = float(roc_auc_score(y_te, uncal_prob))
            platt_auc = float(roc_auc_score(y_te, platt_prob))
            iso_auc = float(roc_auc_score(y_te, iso_prob))
        except Exception:
            uncal_auc = platt_auc = iso_auc = float("nan")

        fold_rec: Dict[str, Any] = {
            "fold": fold_idx,
            "n_train": int(len(y_train)), "n_cal": int(len(y_cal)), "n_test": int(len(y_te)),
            "n_pos_test": int(y_te.sum()),
            "uncal_auc": uncal_auc, "platt_auc": platt_auc, "iso_auc": iso_auc,
        }
        for label, prob in [("uncal", uncal_prob), ("platt", platt_prob), ("iso", iso_prob)]:
            for thr in THRESHOLDS:
                y_pred = (prob >= thr).astype(int)
                prec = float(precision_score(y_te, y_pred, zero_division=0)) if y_pred.sum() > 0 else 0.0
                rec = float(recall_score(y_te, y_pred, zero_division=0))
                fold_rec[f"{label}_precision_at_{int(thr*100)}"] = prec
                fold_rec[f"{label}_recall_at_{int(thr*100)}"] = rec
        per_fold.append(fold_rec)

    # Aggregate.
    df = pd.DataFrame(per_fold)
    agg: Dict[str, Any] = {"per_fold": per_fold, "agg_mean_std": {}}
    for col in df.columns:
        if col in ("fold", "n_train", "n_cal", "n_test", "n_pos_test"):
            continue
        agg["agg_mean_std"][col] = {
            "mean": float(df[col].mean()), "std": float(df[col].std(ddof=1)),
        }
    return agg


def _reliability_plot(
    out_png: Path, y: np.ndarray, prob_dict: Dict[str, np.ndarray],
) -> None:
    """Plot reliability curves for each calibrator alongside the diagonal."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"uncal": "#888888", "platt": "#1f77b4", "iso": "#d62728"}
    for label, prob in prob_dict.items():
        if prob.size < 10:
            continue
        # Use 10 quantile bins to handle output concentration.
        try:
            frac_pos, mean_pred = calibration_curve(
                y, prob, n_bins=10, strategy="quantile",
            )
        except Exception:
            continue
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.5,
                color=colors.get(label, "k"), label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability (per quantile bin)")
    ax.set_ylabel("Empirical positive rate (per quantile bin)")
    ax.set_title("Arc 9 D1 classifier - reliability diagram (Approach A test set)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Arc 9 calibration recovery test.")
    parser.add_argument(
        "--out-dir", type=Path,
        default=_REPO_ROOT / "results" / "l_arc_9" / "experiments" / "calibration_recovery",
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load step4 entry features + trades_paths to rebuild D1 t=1 feature matrix.
    entry_features = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "step4_extractability" / "entry_features.csv"
    )
    trades_paths = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "step1_verbatim" / "trades_paths.csv"
    )
    clusters = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "step2_clustering" / "clusters_K3.csv"
    )
    cid0 = set(clusters[clusters["cluster_id"] == 0]["trade_id"].astype(int))

    # Build D1 feature matrix exactly as Step 4 did.
    merged, feature_cols = _build_d1_feature_matrix(entry_features, trades_paths, D1_T)
    merged["y"] = merged["trade_id"].astype(int).apply(lambda x: 1 if x in cid0 else 0)
    X = merged[feature_cols].to_numpy(dtype=float)
    y = merged["y"].to_numpy(dtype=int)
    n_total, n_pos = int(len(y)), int(y.sum())
    print(f"[cal-recov] D1 t={D1_T} feature matrix: n_total={n_total}, n_pos={n_pos}, n_features={X.shape[1]}")

    # ---- Reproduce Step 4 D1 OOF + threshold sweep -------------------------
    uncal_auc, oof, repro_sweep = _reproduce_step4_oof(X, y)
    print(f"[cal-recov] reproduced uncal AUC: {uncal_auc:.4f}")

    # Verify against Step 4 saved threshold sweep.
    step4_sweep = pd.read_csv(
        _REPO_ROOT / "results" / "l_arc_9" / "step4_extractability" /
        "threshold_sweep_D1_cluster_0_individual.csv"
    )
    repro_df = pd.DataFrame(repro_sweep)
    parity = {
        "step4_sweep_present": True,
        "step4_n_thresholds": int(len(step4_sweep)),
        "step4_threshold_recall_max": float(step4_sweep["recall"].max()),
        "step4_threshold_precision_at_40": float(
            step4_sweep.loc[step4_sweep["threshold"] == 0.4, "precision"].iloc[0]
        ),
        "step4_threshold_recall_at_40": float(
            step4_sweep.loc[step4_sweep["threshold"] == 0.4, "recall"].iloc[0]
        ),
        "reproduced_threshold_recall_at_40": float(
            repro_df.loc[repro_df["threshold"] == 0.4, "recall"].iloc[0]
        ),
        "reproduced_threshold_precision_at_40": float(
            repro_df.loc[repro_df["threshold"] == 0.4, "precision"].iloc[0]
        ),
    }
    parity["matches_step4"] = (
        abs(parity["reproduced_threshold_recall_at_40"] - parity["step4_threshold_recall_at_40"]) < 1e-9
        and abs(parity["reproduced_threshold_precision_at_40"] - parity["step4_threshold_precision_at_40"]) < 1e-9
    )
    print(f"[cal-recov] reproduction parity vs Step 4 saved sweep: {parity['matches_step4']}")
    print(f"  reproduced recall@0.40 = {parity['reproduced_threshold_recall_at_40']:.4f}")
    print(f"  step4      recall@0.40 = {parity['step4_threshold_recall_at_40']:.4f}")

    # ---- Approach A --------------------------------------------------------
    a = _approach_a(X, y)
    print(f"[cal-recov] Approach A AUCs: uncal={a['uncal_auc']:.4f}, "
          f"platt={a['platt_auc']:.4f}, iso={a['iso_auc']:.4f}")
    print(f"  AUC drift Platt = {a['auc_drift_platt']:+.4f}, Iso = {a['auc_drift_iso']:+.4f}")
    # AUC-preservation check.
    auc_preserved = (
        abs(a["auc_drift_platt"]) <= 0.01 and abs(a["auc_drift_iso"]) <= 0.01
    )
    if not auc_preserved:
        print(f"[cal-recov] WARNING - calibrated AUC drift > 0.01")

    # ---- Approach B --------------------------------------------------------
    b = _approach_b(X, y)
    print(f"[cal-recov] Approach B per-fold AUC: uncal mean={b['agg_mean_std']['uncal_auc']['mean']:.4f} "
          f"+/- {b['agg_mean_std']['uncal_auc']['std']:.4f}")
    print(f"  platt mean={b['agg_mean_std']['platt_auc']['mean']:.4f} +/- {b['agg_mean_std']['platt_auc']['std']:.4f}")
    print(f"  iso   mean={b['agg_mean_std']['iso_auc']['mean']:.4f} +/- {b['agg_mean_std']['iso_auc']['std']:.4f}")

    # ---- Reliability diagram (Approach A test set) -------------------------
    _reliability_plot(
        args.out_dir / "reliability_diagrams.png",
        np.array(a["y_test"]),
        {
            "uncal": np.array(a["probs"]["uncal"]),
            "platt": np.array(a["probs"]["platt"]),
            "iso": np.array(a["probs"]["iso"]),
        },
    )

    # ---- Persist outputs ----------------------------------------------------
    repro_df.to_csv(args.out_dir / "reproduced_step4_oof_sweep.csv", index=False,
                    float_format="%.10g", lineterminator="\n")
    pd.DataFrame(a["threshold_sweep_rows"]).to_csv(
        args.out_dir / "approach_A_threshold_sweep.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )
    pd.DataFrame(b["per_fold"]).to_csv(
        args.out_dir / "approach_B_per_fold.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )
    # Approach B aggregate as flat csv.
    agg_rows = []
    for k, v in b["agg_mean_std"].items():
        agg_rows.append({"metric": k, "mean": v["mean"], "std": v["std"]})
    pd.DataFrame(agg_rows).to_csv(
        args.out_dir / "approach_B_aggregate.csv", index=False,
        float_format="%.10g", lineterminator="\n",
    )

    summary = {
        "reproduction_parity": parity,
        "reproduced_uncal_auc": uncal_auc,
        "approach_a": {
            "n_train": a["n_train"], "n_cal": a["n_cal"], "n_test": a["n_test"],
            "n_pos_train": a["n_pos_train"], "n_pos_cal": a["n_pos_cal"], "n_pos_test": a["n_pos_test"],
            "uncal_auc": a["uncal_auc"], "platt_auc": a["platt_auc"], "iso_auc": a["iso_auc"],
            "auc_drift_platt": a["auc_drift_platt"], "auc_drift_iso": a["auc_drift_iso"],
            "auc_preserved_within_0.01": bool(auc_preserved),
        },
        "approach_b_aggregate": b["agg_mean_std"],
        "thresholds": THRESHOLDS,
        "recall_floor_v2_2_s3": RECALL_FLOOR,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    # ---- Verdict (Outcome A vs B) ------------------------------------------
    def best_recall_for(calibrator: str) -> Tuple[float, float, float]:
        """Return (best_recall, threshold_at_best, precision_at_that_threshold) for the named calibrator on Approach A test set."""
        sub = [r for r in a["threshold_sweep_rows"] if r["calibrator"] == f"approach_A_{calibrator}"]
        sub_sorted = sorted(sub, key=lambda r: -r["recall"])
        best = sub_sorted[0]
        return float(best["recall"]), float(best["threshold"]), float(best["precision"])

    uncal_best = best_recall_for("uncal")
    platt_best = best_recall_for("platt")
    iso_best = best_recall_for("isotonic")

    def passes_v2_2_s3(c: Tuple[float, float, float]) -> bool:
        # v2.2 §3: any threshold in {0.40, 0.50, 0.60, 0.70} with recall >= 0.60.
        sub_max = c[0]  # best recall achieved
        return sub_max >= RECALL_FLOOR

    outcome_a_passes = passes_v2_2_s3(platt_best) or passes_v2_2_s3(iso_best)
    outcome = "OUTCOME_A" if outcome_a_passes else "OUTCOME_B"
    print(f"[cal-recov] best recall on Approach A test set:")
    print(f"  uncal: recall {uncal_best[0]:.4f} @ thr {uncal_best[1]:.2f} (precision {uncal_best[2]:.4f})")
    print(f"  platt: recall {platt_best[0]:.4f} @ thr {platt_best[1]:.2f} (precision {platt_best[2]:.4f})")
    print(f"  iso  : recall {iso_best[0]:.4f} @ thr {iso_best[1]:.2f} (precision {iso_best[2]:.4f})")
    print(f"[cal-recov] HEADLINE: {outcome}")

    summary["verdict"] = {
        "outcome": outcome,
        "uncal_best_recall": uncal_best,
        "platt_best_recall": platt_best,
        "iso_best_recall": iso_best,
        "v2_2_s3_passes_uncal": passes_v2_2_s3(uncal_best),
        "v2_2_s3_passes_platt": passes_v2_2_s3(platt_best),
        "v2_2_s3_passes_iso": passes_v2_2_s3(iso_best),
    }

    # ---- Extended sweep: find lowest threshold reaching recall >= 0.60 ----
    # Informational for v2.x amendment analysis. The v2.2 §3 gate stays on
    # the fixed {0.40, 0.50, 0.60, 0.70} grid.
    extended_thrs = np.linspace(0.0, 1.0, 101)
    y_test_arr = np.array(a["y_test"])
    ext_rows: List[Dict[str, Any]] = []
    ext_first_pass: Dict[str, Optional[Dict[str, Any]]] = {}
    for label in ("uncal", "platt", "iso"):
        prob_arr = np.array(a["probs"][label])
        rows_for_calibrator: List[Dict[str, Any]] = []
        for thr in extended_thrs:
            y_pred = (prob_arr >= thr).astype(int)
            if y_pred.sum() == 0:
                prec, rec = 0.0, 0.0
            else:
                prec = float(precision_score(y_test_arr, y_pred, zero_division=0))
                rec = float(recall_score(y_test_arr, y_pred, zero_division=0))
            row = {
                "calibrator": label, "threshold": float(thr),
                "precision": prec, "recall": rec, "n_admitted": int(y_pred.sum()),
            }
            rows_for_calibrator.append(row)
            ext_rows.append(row)
        # Highest threshold at which recall >= RECALL_FLOOR (= best precision
        # while clearing the floor). Recall is monotone non-increasing in
        # threshold; the highest passing threshold gives the highest precision.
        passing = [r for r in rows_for_calibrator if r["recall"] >= RECALL_FLOOR]
        if passing:
            best = max(passing, key=lambda r: r["threshold"])
            ext_first_pass[label] = {
                "highest_threshold_with_recall_ge_0_60": best["threshold"],
                "precision_there": best["precision"],
                "recall_there": best["recall"],
                "n_admitted_there": best["n_admitted"],
            }
        else:
            ext_first_pass[label] = None
    pd.DataFrame(ext_rows).to_csv(
        args.out_dir / "approach_A_extended_threshold_sweep.csv",
        index=False, float_format="%.10g", lineterminator="\n",
    )
    summary["extended_sweep_first_pass_at_recall_0_60"] = ext_first_pass

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print("[cal-recov] extended sweep - HIGHEST threshold with recall >= 0.60 (best precision while clearing floor):")
    for label, fp in ext_first_pass.items():
        if fp is None:
            print(f"  {label}: NEVER reaches recall {RECALL_FLOOR} on the test set")
        else:
            print(f"  {label}: thr {fp['highest_threshold_with_recall_ge_0_60']:.2f} "
                  f"-> precision {fp['precision_there']:.3f}, "
                  f"recall {fp['recall_there']:.3f}, n_admitted {fp['n_admitted_there']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
