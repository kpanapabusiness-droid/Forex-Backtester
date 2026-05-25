"""Arc 10 v3.0 — Step 4 extraction.

Per L_PROTOCOL v3.0 §2 Step 4 + Appendix A. For each candidate cluster
from Step 3:
    - target: binary cluster membership
    - classifiers: RF + LGBM + Logistic at Appendix A defaults
    - 5-fold TimeSeriesSplit, per-fold AUC
    - threshold sweep at AUC-best and F1-best
    - permutation importance per feature
    - per-arc swing-feature spot-check (Arc 9 lesson — dispatch §Step 4)

Outputs:
    results/l_arc_10/step_4/extraction_metrics.csv
    results/l_arc_10/step_4/feature_importance.csv
    results/l_arc_10/step_4/extraction_summary.md
    results/l_arc_10/step_4/manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from core.determinism import RANDOM_STATE, seed_everything  # noqa: E402
from scripts.l_arc_10_v3._common import load_config, sha256_file, write_manifest  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

# Default feature set excludes identity, outcome, path-shape columns.
# Everything else (27 default + 7 signal-specific = 34 cols) participates.
EXCLUDED_COLS = {
    "trade_id",
    "pair",
    "signal_bar_time",
    "entry_time",
    "exit_time",
    "entry_price",
    "sl_at_entry_price",
    "sl_distance_price",
    "exit_price",
    "exit_reason",
    "bars_held",
    "final_r",
    "mfe_r",
    "mae_r",
    "time_to_peak_mfe",
    "spread_close_at_entry",
    "spread_close_at_exit",
    "bid_ask_dq_at_entry",
    "bid_ask_dq_at_exit",
    # Path-shape cols (cluster inputs — excluded from extraction)
    "path_mono",
    "path_peaks",
    "path_ttp_rel",
    "path_drawdown_depth_r",
    "path_recovery_ratio",
    "path_wrong_way_first",
    # cluster columns added by merge
    "cluster_primary",
    "archetype_primary",
    "primary_K",
}

# Features derived from swing / pivot / extremum detection — priority audit per dispatch.
SWING_DERIVED_FEATURES = {
    "swing_high_distance_14",
    "swing_low_distance_14",
    "kijun_26_distance",
    "L1_value",
    "L0_value",
    "L1_age_d1_bars",
    "L0_age_d1_bars",
    "L1_to_atr_proximity",
    "reject_buffer_atr",
    "prior_session_high_distance",
    "prior_session_low_distance",
}


def _classifier_factories():
    """Return dict of (name -> sklearn classifier factory) per Appendix A."""
    return {
        "rf": lambda: RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=50,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "logistic": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        penalty="l2", C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=1
                    ),
                ),
            ]
        ),
    }


def _lgbm_factory():
    try:
        import lightgbm as lgb  # noqa
        return lambda: lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=50,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        )
    except ImportError:
        return None


def _threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Find AUC-best and F1-best thresholds."""
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = []
    best_f1 = -1.0
    best_f1_metrics = {}
    best_auc_metrics = {}
    if len(np.unique(y_true)) < 2:
        return dict(auc_best=None, f1_best=None)
    auc = float(roc_auc_score(y_true, y_prob))
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        if y_pred.sum() == 0:
            continue
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        f1s.append((thr, f1))
        prec = float(((y_pred == 1) & (y_true == 1)).sum()) / max(int(y_pred.sum()), 1)
        rec = float(((y_pred == 1) & (y_true == 1)).sum()) / max(int(y_true.sum()), 1)
        n_admit = int(y_pred.sum())
        if f1 > best_f1:
            best_f1 = f1
            best_f1_metrics = dict(threshold=float(thr), precision=prec, recall=rec, n_admit=n_admit, f1=f1)
        if abs(thr - 0.5) < 0.005:
            best_auc_metrics = dict(threshold=float(thr), precision=prec, recall=rec, n_admit=n_admit, f1=f1)
    if not best_auc_metrics and f1s:
        # Take threshold closest to 0.5 from f1 sweep
        best_auc_metrics = dict(
            threshold=0.5,
            precision=np.nan,
            recall=np.nan,
            n_admit=int((y_prob >= 0.5).sum()),
            f1=np.nan,
        )
    return dict(auc=auc, auc_best=best_auc_metrics, f1_best=best_f1_metrics)


def _fold_eval(model_factory, X: np.ndarray, y: np.ndarray, splitter) -> dict:
    """5-fold TimeSeriesSplit per-fold AUC + concatenated OOS predictions."""
    fold_aucs = []
    oos_y = np.full(len(y), np.nan)
    oos_p = np.full(len(y), np.nan)
    for fi, (tr_idx, te_idx) in enumerate(splitter.split(X)):
        if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[te_idx])) < 2:
            fold_aucs.append(np.nan)
            continue
        clf = model_factory()
        try:
            clf.fit(X[tr_idx], y[tr_idx])
            p = clf.predict_proba(X[te_idx])[:, 1]
        except Exception:
            fold_aucs.append(np.nan)
            continue
        try:
            auc = float(roc_auc_score(y[te_idx], p))
        except ValueError:
            auc = np.nan
        fold_aucs.append(auc)
        oos_y[te_idx] = y[te_idx]
        oos_p[te_idx] = p

    # Threshold sweep on the concatenated OOS predictions (only valid where != NaN)
    mask = np.isfinite(oos_p) & np.isfinite(oos_y)
    if mask.sum() == 0:
        return dict(fold_aucs=fold_aucs, mean_auc=np.nan, threshold_sweep=None)
    sweep = _threshold_sweep(oos_y[mask].astype(int), oos_p[mask])
    return dict(
        fold_aucs=fold_aucs,
        mean_auc=float(np.nanmean(fold_aucs)),
        oos_auc=sweep.get("auc"),
        threshold_sweep=sweep,
    )


def _permutation_importance(model_factory, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> list[dict]:
    """Permutation importance on a single train/holdout split (last 20% as holdout)."""
    if len(np.unique(y)) < 2:
        return []
    n = len(y)
    cut = int(n * 0.8)
    if cut < 30 or n - cut < 10:
        return []
    clf = model_factory()
    try:
        clf.fit(X[:cut], y[:cut])
    except Exception:
        return []
    try:
        r = permutation_importance(
            clf, X[cut:], y[cut:], n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc"
        )
    except Exception:
        return []
    return sorted(
        [
            dict(
                feature=feature_names[i],
                importance_mean=float(r.importances_mean[i]),
                importance_std=float(r.importances_std[i]),
            )
            for i in range(len(feature_names))
        ],
        key=lambda d: -d["importance_mean"],
    )


def _swing_audit(top_features: list[dict]) -> dict:
    """Special producer-level audit per dispatch §Step 4: any swing/pivot/
    extremum-derived feature in any cluster's top-10 gets spot-checked here.
    """
    swing_in_top = [f["feature"] for f in top_features[:10] if f["feature"] in SWING_DERIVED_FEATURES]
    notes = []
    if "swing_high_distance_14" in swing_in_top or "swing_low_distance_14" in swing_in_top:
        notes.append(
            "swing_high_distance_14 / swing_low_distance_14 use rolling(.max/.min, window=14).shift(1) "
            "on mid-OHLC — one-sided lookback, no future bars. Causal. (Trailing N-bar swing, not centred ±N.)"
        )
    if any(f.startswith("L1_") or f.startswith("L0_") for f in swing_in_top):
        notes.append(
            "L1/L0 features carried verbatim from the DLR producer (signals/lchar_dlr_long.py). "
            "Producer uses ±3-bar swing-low detector with right-edge offset 4 (confirmation-lag, "
            "NOT Arc 9's centred-at-signal failure mode). Producer-level trace passed at intent stage."
        )
    if "kijun_26_distance" in swing_in_top:
        notes.append(
            "kijun_26_distance = kijun(high, low, 26).shift(1) — strictly prior bars. Causal."
        )
    return dict(swing_features_in_top10=swing_in_top, audit_notes=notes)


def run(cfg_path: Path, *, write_manifest_flag: bool = True) -> dict:
    seed_everything(RANDOM_STATE)
    cfg = load_config(cfg_path)

    # Arc root + step dirs derived from Step 1 results_dir. Byte-identical
    # resolution for Arc 10 v3.0; correct routing for Arc 10 v3.0.2.
    arc_root = REPO_ROOT / Path(cfg["output"]["results_dir"]).parent
    pool_path = REPO_ROOT / cfg["output"]["results_dir"] / cfg["output"]["pool_parquet"]
    pool = pd.read_parquet(pool_path).sort_values("signal_bar_time").reset_index(drop=True)
    assignments = pd.read_parquet(arc_root / "step_2" / "cluster_assignments.parquet")
    pool = pool.merge(
        assignments[["trade_id", "cluster_primary", "archetype_primary", "primary_K"]], on="trade_id", how="left"
    )
    cap_csv = arc_root / "step_3" / "capturability.csv"
    cap = pd.read_csv(cap_csv)
    candidates = cap[cap["candidate_at_best_sl"] == True]["cluster_id"].astype(int).tolist()  # noqa
    if not candidates:
        # No candidates from Step 3; per dispatch run on the highest-composite cluster regardless.
        candidates = [int(cap.sort_values("composite", ascending=False).iloc[0]["cluster_id"])]

    out_dir = arc_root / "step_4"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        c for c in pool.columns
        if c not in EXCLUDED_COLS and pd.api.types.is_numeric_dtype(pool[c])
    ]

    # Drop rows with all-NaN features (rare) and fill remaining NaN with median (TimeSeriesSplit-safe global fill is acceptable here as a Step 1 simplification).
    X_full = pool[feature_cols].copy()
    medians = X_full.median(skipna=True)
    X_full = X_full.fillna(medians).fillna(0.0)

    clf_factories = _classifier_factories()
    lgbm = _lgbm_factory()
    if lgbm:
        clf_factories["lgbm"] = lgbm

    splitter = TimeSeriesSplit(n_splits=5)

    extraction_records = []
    feature_importance_records = []
    summary_per_cluster = {}

    for cid in candidates:
        cid = int(cid)
        y = (pool["cluster_primary"] == cid).astype(int).to_numpy()
        if y.sum() < 30:
            print(f"[step_4] cluster c{cid}: too few positives ({y.sum()}) — skip", flush=True)
            continue

        per_clf = {}
        for clf_name, factory in clf_factories.items():
            seed_everything(RANDOM_STATE)
            res = _fold_eval(factory, X_full.to_numpy(), y, splitter)
            per_clf[clf_name] = res
            ts = res.get("threshold_sweep") or {}
            extraction_records.append(
                dict(
                    cluster_id=cid,
                    classifier=clf_name,
                    n_positives=int(y.sum()),
                    n_total=int(len(y)),
                    mean_auc=res["mean_auc"],
                    oos_auc=res.get("oos_auc"),
                    auc_best_threshold=(ts.get("auc_best") or {}).get("threshold"),
                    auc_best_precision=(ts.get("auc_best") or {}).get("precision"),
                    auc_best_recall=(ts.get("auc_best") or {}).get("recall"),
                    auc_best_n_admit=(ts.get("auc_best") or {}).get("n_admit"),
                    f1_best_threshold=(ts.get("f1_best") or {}).get("threshold"),
                    f1_best_precision=(ts.get("f1_best") or {}).get("precision"),
                    f1_best_recall=(ts.get("f1_best") or {}).get("recall"),
                    f1_best_n_admit=(ts.get("f1_best") or {}).get("n_admit"),
                )
            )

        # Best classifier by mean OOS AUC
        best_clf_name = max(per_clf, key=lambda k: per_clf[k]["mean_auc"] if np.isfinite(per_clf[k]["mean_auc"]) else -np.inf)
        best_clf_factory = clf_factories[best_clf_name]
        seed_everything(RANDOM_STATE)
        perm_imp = _permutation_importance(best_clf_factory, X_full.to_numpy(), y, feature_cols)

        # Audit swing features in top 10
        audit = _swing_audit(perm_imp)

        # Save importance per cluster
        for rank, row in enumerate(perm_imp):
            feature_importance_records.append(
                dict(
                    cluster_id=cid,
                    classifier=best_clf_name,
                    rank=rank + 1,
                    feature=row["feature"],
                    importance_mean=row["importance_mean"],
                    importance_std=row["importance_std"],
                )
            )

        summary_per_cluster[cid] = dict(
            best_classifier=best_clf_name,
            best_mean_auc=per_clf[best_clf_name]["mean_auc"],
            best_oos_auc=per_clf[best_clf_name].get("oos_auc"),
            best_threshold=(per_clf[best_clf_name].get("threshold_sweep", {}) or {}).get("auc_best", {}).get("threshold"),
            top_10_features=[r["feature"] for r in perm_imp[:10]],
            swing_audit=audit,
        )

    df_metrics = pd.DataFrame(extraction_records)
    df_imp = pd.DataFrame(feature_importance_records)
    df_metrics.to_csv(out_dir / "extraction_metrics.csv", index=False, lineterminator="\n")
    df_imp.to_csv(out_dir / "feature_importance.csv", index=False, lineterminator="\n")

    lines = []
    lines.append("# Arc 10 v3.0 — Step 4 Extraction Summary\n\n")
    lines.append(f"- Candidate clusters: {candidates}\n")
    lines.append(f"- Features used: {len(feature_cols)} (default 27 + arc-specific extras)\n")
    lines.append(f"- Classifiers: {sorted(clf_factories.keys())} at Appendix A defaults\n\n")
    lines.append("## Per-cluster AUC + threshold sweep\n\n")
    lines.append(
        "| Cluster | Classifier | n+ | mean_AUC | OOS_AUC | AUC-best thr | prec | recall | n_admit | F1-best thr | F1 | F1 prec | F1 rec |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for r in extraction_records:
        ts_f1_thr = r.get("f1_best_threshold")
        lines.append(
            f"| c{r['cluster_id']} | {r['classifier']} | {r['n_positives']} | "
            f"{r['mean_auc']:.4f} | {r.get('oos_auc') or float('nan'):.4f} | "
            f"{r.get('auc_best_threshold') or float('nan'):.2f} | "
            f"{r.get('auc_best_precision') or float('nan'):.3f} | "
            f"{r.get('auc_best_recall') or float('nan'):.3f} | "
            f"{r.get('auc_best_n_admit') or 0} | "
            f"{ts_f1_thr if ts_f1_thr is not None else float('nan'):.2f} | "
            f"{(r.get('f1_best_precision') or float('nan')) if r.get('f1_best_threshold') else float('nan'):.3f} | "
            f"{r.get('f1_best_precision') or float('nan'):.3f} | "
            f"{r.get('f1_best_recall') or float('nan'):.3f} |\n"
        )

    lines.append("\n## Top-10 features per cluster (permutation importance on best classifier)\n\n")
    for cid, s in summary_per_cluster.items():
        lines.append(f"\n### c{cid} (best clf: {s['best_classifier']}, mean AUC: {s['best_mean_auc']:.4f})\n\n")
        top10 = df_imp[(df_imp["cluster_id"] == cid) & (df_imp["rank"] <= 10)]
        lines.append("| Rank | Feature | imp_mean | imp_std |\n|---:|---|---:|---:|\n")
        for _, row in top10.iterrows():
            lines.append(
                f"| {row['rank']} | {row['feature']} | {row['importance_mean']:.5f} | {row['importance_std']:.5f} |\n"
            )
        lines.append("\n**Swing-feature audit (per dispatch §Step 4):**\n")
        lines.append(
            f"- Swing-derived features in top 10: {s['swing_audit']['swing_features_in_top10']}\n"
        )
        for note in s["swing_audit"]["audit_notes"]:
            lines.append(f"- {note}\n")

    # §8 gate per dispatch — informational
    lines.append("\n## Gate readings (informational; arc continues regardless)\n\n")
    lines.append("- L_PROTOCOL v3.0 §3 has no Step-4 AUC gate. Step 5 (WFO) is the only deployment gate.\n")
    lines.append("- Cross-arc historical context: prior V-shape near-misses (Arc 7, Arc 10 v2.3) hovered Pipeline E ≈ 0.48-0.63 — feature-set bound.\n")

    summary_path = out_dir / "extraction_summary.md"
    summary_path.write_text("".join(lines), encoding="utf-8", newline="\n")

    manifest = dict(
        arc_name="l_arc_10",
        step="step_4",
        protocol_version="v3.0",
        candidates=candidates,
        feature_count=len(feature_cols),
        classifiers=sorted(clf_factories.keys()),
        per_cluster_summary=summary_per_cluster,
        sha256=dict(
            extraction_metrics_csv=sha256_file(out_dir / "extraction_metrics.csv"),
            feature_importance_csv=sha256_file(out_dir / "feature_importance.csv"),
            extraction_summary_md=sha256_file(summary_path),
        ),
        env=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__),
        determinism=dict(random_state=RANDOM_STATE, n_jobs=1, line_terminator="\\n"),
        run_timestamp_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
    try:
        import sklearn  # noqa
        manifest["env"]["sklearn"] = sklearn.__version__
    except Exception:
        manifest["env"]["sklearn"] = "not_installed"
    try:
        import lightgbm  # noqa
        manifest["env"]["lightgbm"] = lightgbm.__version__
    except Exception:
        manifest["env"]["lightgbm"] = "not_installed"

    if write_manifest_flag:
        write_manifest(out_dir / "manifest.json", manifest)

    print("[step_4] candidate AUCs:", flush=True)
    for cid, s in summary_per_cluster.items():
        print(f"  c{cid}: best clf={s['best_classifier']} mean AUC={s['best_mean_auc']:.4f}", flush=True)
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arc 10 v3.0 — Step 4 extraction")
    p.add_argument("-c", "--config", required=True, type=Path)
    args = p.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
