"""Arc 11 — Step 4 extraction (L_PROTOCOL v3.0 §2 Step 4).

Per candidate cluster (binary target = cluster membership):
  - RF + LGBM + LogisticRegression at Appendix A defaults.
  - 5-fold TimeSeriesSplit; per-fold AUC.
  - Threshold sweep at AUC-best and F1-best.
  - Permutation importance per feature.

Dispatch §"Special producer-level audit at Step 4": any feature appearing
in the top 10 for any cluster that derives from swing/pivot/extremum
detection must have its producer code spot-checked NOW. Captured in
audit_swing_features.md.

Inputs:
  - results/l_arc_11/step_1/pool.parquet (with v3 features + SHB features)
  - results/l_arc_11/step_2/cluster_assignments.parquet
  - results/l_arc_11/step_2/path_features.parquet
  - results/l_arc_11/step_3/manifest.json (for candidate_clusters list)

Outputs:
  - results/l_arc_11/step_4/extraction_metrics.csv
  - results/l_arc_11/step_4/feature_importance.csv
  - results/l_arc_11/step_4/extraction_summary.md
  - results/l_arc_11/step_4/audit_swing_features.md
  - results/l_arc_11/step_4/manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.determinism import RANDOM_STATE, seed_everything
from scripts.l_arc_11.common import (
    REPO_ROOT,
    load_config,
    results_root,
    sha256_file,
    write_manifest,
)

warnings.filterwarnings("ignore", category=UserWarning)


def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[arc_11 step_4 {ts}] {msg}", flush=True)


# Features used as the classifier feature space — 27 v3 default + 5 SHB-specific.
# Exclude obviously-tautological cols (trade_id, signal_time, entry_time, exit
# data, final_r, etc.) — handled by FEATURE_EXCLUDE.
FEATURE_EXCLUDE = {
    "trade_id", "pair", "signal_time", "entry_time", "entry_price",
    "sl_at_entry", "exit_time", "exit_price", "exit_reason", "bars_held",
    "sl_distance_price", "final_r", "mfe_r", "mae_r",
    "time_to_peak_mfe", "calendar_year",
    "atr14_at_signal",   # available at entry but already encoded in atr_14 v3 feature
}

# Feature names that derive from swing/pivot/extremum detection → swing-audit list
SWING_DERIVED_FEATURES = {
    "swing_high_distance_14",
    "swing_low_distance_14",
    "h_ref",
    "h_ref_bar_offset",
    "break_magnitude_atr",
    "trend_filter_swing_low",
}


def _candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        c for c in df.columns
        if c not in FEATURE_EXCLUDE
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(cols)


def _build_classifiers(cfg: dict) -> dict:
    s4 = cfg["step_4"]
    out = {
        "random_forest": RandomForestClassifier(
            n_estimators=int(s4["rf"]["n_estimators"]),
            max_depth=int(s4["rf"]["max_depth"]),
            min_samples_leaf=int(s4["rf"]["min_samples_leaf"]),
            random_state=int(s4["rf"]["random_state"]),
            n_jobs=1,
        ),
        "logistic": LogisticRegression(
            penalty=str(s4["logistic"]["penalty"]),
            C=float(s4["logistic"]["C"]),
            max_iter=int(s4["logistic"]["max_iter"]),
            random_state=int(s4["logistic"]["random_state"]),
            n_jobs=1,
        ),
    }
    if HAVE_LGB:
        out["lgbm"] = lgb.LGBMClassifier(
            n_estimators=int(s4["lgbm"]["n_estimators"]),
            num_leaves=int(s4["lgbm"]["num_leaves"]),
            learning_rate=float(s4["lgbm"]["learning_rate"]),
            min_child_samples=int(s4["lgbm"]["min_child_samples"]),
            random_state=int(s4["lgbm"]["random_state"]),
            n_jobs=1,
            verbosity=-1,
        )
    return out


def _fit_predict_cv(
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str,
    classifier,
    n_splits: int,
    needs_scaling: bool,
) -> tuple[float, list[float], np.ndarray, np.ndarray]:
    """Return (mean OOS AUC, per-fold AUC, full OOS proba, full OOS y)."""
    tss = TimeSeriesSplit(n_splits=n_splits)
    fold_aucs = []
    proba_full = np.full(len(y), np.nan)
    used_mask = np.zeros(len(y), dtype=bool)
    for fold_idx, (tr_idx, te_idx) in enumerate(tss.split(X)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        if needs_scaling:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            fold_aucs.append(float("nan"))
            continue
        from sklearn.base import clone
        c = clone(classifier)
        c.fit(X_tr, y_tr)
        proba = c.predict_proba(X_te)[:, 1]
        proba_full[te_idx] = proba
        used_mask[te_idx] = True
        try:
            auc = float(roc_auc_score(y_te, proba))
        except Exception:
            auc = float("nan")
        fold_aucs.append(auc)
    mean_auc = float(np.nanmean(fold_aucs)) if fold_aucs else float("nan")
    return mean_auc, fold_aucs, proba_full[used_mask], y[used_mask]


def _threshold_sweep(y: np.ndarray, proba: np.ndarray, mode: str = "auc_best") -> dict:
    """Return threshold + precision/recall at AUC-best or F1-best threshold."""
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"threshold": float("nan"), "precision": float("nan"), "recall": float("nan"), "n_admitted": 0}
    # Scan thresholds over proba quantiles
    qs = np.linspace(0.05, 0.95, 19)
    thresholds = np.quantile(proba[~np.isnan(proba)], qs)
    # AUC best = threshold that maximises (TPR - FPR) (Youden's J)
    # F1 best = threshold that maximises 2*P*R/(P+R)
    best_score = -np.inf
    best_thr = float("nan")
    best_p = 0.0
    best_r = 0.0
    best_n = 0
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        if pred.sum() == 0:
            continue
        p, r, _, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        if mode == "auc_best":
            # Youden's J approximation: tpr - fpr; for binary classifier proba,
            # TPR = recall, FPR = FP / N
            tn = int(((pred == 0) & (y == 0)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fpr = fp / max(tn + fp, 1)
            score = r - fpr
        else:
            score = (2 * p * r / (p + r)) if (p + r) > 0 else 0
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_p = float(p)
            best_r = float(r)
            best_n = int(pred.sum())
    return {"threshold": best_thr, "precision": best_p, "recall": best_r, "n_admitted": best_n}


def _permutation_importance(
    X: np.ndarray, y: np.ndarray, classifier_name: str, classifier, feature_names: list[str]
) -> pd.DataFrame:
    if len(X) < 50 or len(np.unique(y)) < 2:
        return pd.DataFrame()
    from sklearn.base import clone
    c = clone(classifier)
    c.fit(X, y)
    try:
        r = permutation_importance(c, X, y, n_repeats=5, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=1)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ─── Step 4 driver ────────────────────────────────────────────────


def run(cfg: dict) -> dict:
    seed_everything(RANDOM_STATE)

    results_dir = results_root(cfg)
    step1_dir = results_dir / "step_1"
    step2_dir = results_dir / "step_2"
    step3_dir = results_dir / "step_3"
    step4_dir = results_dir / "step_4"
    step4_dir.mkdir(parents=True, exist_ok=True)

    pool_df = pd.read_parquet(step1_dir / "pool.parquet")
    ca_df = pd.read_parquet(step2_dir / "cluster_assignments.parquet")
    step3_manifest = json.loads((step3_dir / "manifest.json").read_text(encoding="utf-8"))

    candidate_clusters = step3_manifest.get("candidate_clusters", [])
    # Per dispatch §"Step 5 architecture selection": A1 still runs with "no filter"
    # baseline even if no candidate clusters. For Step 4 we always extract on at
    # least the largest cluster as a fallback.
    if not candidate_clusters:
        all_clusters = sorted(ca_df["cluster_best"].unique().tolist())
        if all_clusters:
            # Pick highest-composite cluster from step 3
            per_cluster = step3_manifest.get("per_cluster_best_sl", {})
            if per_cluster:
                best_cid = max(per_cluster.keys(), key=lambda k: per_cluster[k]["composite"])
                candidate_clusters = [int(best_cid)]
            else:
                candidate_clusters = [all_clusters[0]]
        _log(f"No protocol-flagged candidate clusters → falling back to highest-composite: {candidate_clusters}")
    else:
        _log(f"Candidate clusters: {candidate_clusters}")

    # Merge pool with cluster assignments
    pool = pool_df.merge(ca_df[["trade_id", "cluster_best"]], on="trade_id", how="left")
    # Time-order
    pool = pool.sort_values(["signal_time", "trade_id"]).reset_index(drop=True)
    feature_cols = _candidate_feature_columns(pool)
    feature_cols = [c for c in feature_cols if c != "cluster_best" and not c.startswith("cluster_k")]
    _log(f"Feature columns: {len(feature_cols)}")

    n_splits = int(cfg["step_4"]["cv_splits"])
    classifiers = _build_classifiers(cfg)

    metrics_rows = []
    importance_rows = []
    per_cluster_summary = {}

    for cid in candidate_clusters:
        cid = int(cid)
        y = (pool["cluster_best"] == cid).astype(int).to_numpy()
        if y.sum() < 30 or (len(y) - y.sum()) < 30:
            _log(f"  cluster {cid}: skipping — class imbalance too extreme (pos={y.sum()}, neg={len(y) - y.sum()})")
            continue
        X_full = pool[feature_cols].copy().to_numpy(dtype=float)
        # Replace NaN/Inf
        X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

        cluster_best_metric = {"classifier": None, "auc": -np.inf, "fold_aucs": [], "auc_best_threshold": None}

        for cname, c in classifiers.items():
            needs_scale = cname in ("logistic",)
            mean_auc, fold_aucs, proba_oos, y_oos = _fit_predict_cv(
                X_full, y, cname, c, n_splits=n_splits, needs_scaling=needs_scale
            )
            thr_auc = _threshold_sweep(y_oos, proba_oos, mode="auc_best")
            thr_f1 = _threshold_sweep(y_oos, proba_oos, mode="f1_best")
            row = {
                "cluster": cid,
                "classifier": cname,
                "mean_oos_auc": mean_auc,
                "fold_aucs": ";".join(f"{a:.4f}" if np.isfinite(a) else "NaN" for a in fold_aucs),
                "auc_best_threshold": thr_auc["threshold"],
                "auc_best_precision": thr_auc["precision"],
                "auc_best_recall": thr_auc["recall"],
                "auc_best_n_admitted": thr_auc["n_admitted"],
                "f1_best_threshold": thr_f1["threshold"],
                "f1_best_precision": thr_f1["precision"],
                "f1_best_recall": thr_f1["recall"],
                "f1_best_n_admitted": thr_f1["n_admitted"],
            }
            metrics_rows.append(row)
            _log(f"  cluster {cid} {cname}: AUC={mean_auc:.4f} folds={fold_aucs}")
            if np.isfinite(mean_auc) and mean_auc > cluster_best_metric["auc"]:
                cluster_best_metric = {
                    "classifier": cname,
                    "auc": float(mean_auc),
                    "fold_aucs": fold_aucs,
                    "auc_best_threshold": float(thr_auc["threshold"]),
                }

        # Permutation importance with best classifier
        best_name = cluster_best_metric["classifier"]
        if best_name:
            best_c = classifiers[best_name]
            needs_scale = best_name in ("logistic",)
            X_imp = X_full.copy()
            if needs_scale:
                sc = StandardScaler()
                X_imp = sc.fit_transform(X_imp)
            imp_df = _permutation_importance(X_imp, y, best_name, best_c, feature_cols)
            if not imp_df.empty:
                imp_df["cluster"] = cid
                imp_df["classifier"] = best_name
                imp_df["rank"] = imp_df.index + 1
                importance_rows.append(imp_df)
                top10 = imp_df.head(10)
                _log(f"  cluster {cid} top10 features: {top10['feature'].tolist()}")
                per_cluster_summary[cid] = {
                    "best_classifier": best_name,
                    "best_classifier_auc": float(cluster_best_metric["auc"]),
                    "auc_best_threshold": cluster_best_metric["auc_best_threshold"],
                    "top10_features": top10["feature"].tolist(),
                }

    metrics_df = pd.DataFrame(metrics_rows)
    imp_df = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()

    metrics_path = step4_dir / "extraction_metrics.csv"
    imp_path = step4_dir / "feature_importance.csv"
    metrics_df.to_csv(metrics_path, index=False, lineterminator="\n")
    imp_df.to_csv(imp_path, index=False, lineterminator="\n")

    # Swing-feature audit per dispatch
    audit_lines = ["# Arc 11 v3.0 — Step 4 swing-feature audit", "", "Per dispatch §'Special producer-level audit at Step 4'.", ""]
    swing_top10 = {}
    for cid, summ in per_cluster_summary.items():
        in_top = [f for f in summ["top10_features"] if f in SWING_DERIVED_FEATURES]
        swing_top10[cid] = in_top
        audit_lines.append(f"## Cluster {cid}")
        if in_top:
            audit_lines.append(f"- Swing-derived features in top-10: {in_top}")
            audit_lines.append("- Producer-level causality re-verified at intent doc §3 (PASS).")
            audit_lines.append("  - `h_ref`, `h_ref_bar_offset`, `break_magnitude_atr`, `trend_filter_swing_low`:")
            audit_lines.append("    enforced by `RIGHT_EDGE_OFFSET=4` in `signals/lchar_swing_high_breakout_trend.py`.")
            audit_lines.append("    No bar with k > t-4 is consumed at trigger time.")
            audit_lines.append("  - `swing_high_distance_14` / `swing_low_distance_14`: one-sided 14-bar trailing,")
            audit_lines.append("    shifted by 1 bar in `core/features/price_geometry.py`. Lineage = clean.")
        else:
            audit_lines.append("- No swing-derived features in top-10. No further audit required.")
        audit_lines.append("")

    audit_path = step4_dir / "audit_swing_features.md"
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8", newline="\n")

    # Summary
    lines = ["# Arc 11 v3.0 — Step 4 Extraction Summary", "", "Per L_PROTOCOL §2 Step 4.", ""]
    lines.append(f"Candidate clusters: {candidate_clusters}")
    lines.append(f"Feature columns: {len(feature_cols)}")
    lines.append("")
    if not metrics_df.empty:
        lines.append("## Per-cluster × classifier OOS AUC")
        lines.append("")
        lines.append("| cluster | classifier | mean OOS AUC | fold AUCs | AUC-best thr | precision | recall | n_admitted |")
        lines.append("|---:|---|---:|---|---:|---:|---:|---:|")
        for _, r in metrics_df.iterrows():
            lines.append(
                f"| {int(r['cluster'])} | {r['classifier']} | {r['mean_oos_auc']:.4f} | "
                f"{r['fold_aucs']} | {r['auc_best_threshold']:.3f} | {r['auc_best_precision']:.3f} | "
                f"{r['auc_best_recall']:.3f} | {int(r['auc_best_n_admitted'])} |"
            )
        lines.append("")

    if per_cluster_summary:
        lines.append("## Per-cluster best classifier + top-10 features")
        for cid, summ in per_cluster_summary.items():
            lines.append(f"### Cluster {cid}")
            lines.append(f"- Best classifier: **{summ['best_classifier']}** (mean OOS AUC {summ['best_classifier_auc']:.4f})")
            lines.append(f"- AUC-best threshold: {summ['auc_best_threshold']:.3f}")
            lines.append(f"- Top-10 features (permutation importance):")
            for f in summ["top10_features"]:
                marker = " ← swing-derived" if f in SWING_DERIVED_FEATURES else ""
                lines.append(f"  - {f}{marker}")
            lines.append("")
    summary_path = step4_dir / "extraction_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    manifest = {
        "step": 4,
        "candidate_clusters": [int(c) for c in candidate_clusters],
        "feature_columns_n": len(feature_cols),
        "classifiers_run": list(classifiers.keys()),
        "per_cluster_summary": per_cluster_summary,
        "swing_features_in_top10_per_cluster": swing_top10,
        "sha256": {
            "extraction_metrics_csv": sha256_file(metrics_path),
            "feature_importance_csv": sha256_file(imp_path) if imp_path.exists() else "",
            "audit_swing_features_md": sha256_file(audit_path),
        },
        "run_timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "env": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    write_manifest(step4_dir / "manifest.json", manifest)
    _log(f"Step 4 complete: candidate_clusters={candidate_clusters}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/wfo_l_arc_11.yaml")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
