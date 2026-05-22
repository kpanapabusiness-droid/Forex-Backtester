"""Reconstruct run_summary.json from existing step_* artefacts when
scripts/l_arc_11/run.py crashed at the final write step.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = _REPO_ROOT / "results" / "l_arc_11"


def main() -> int:
    s1m = json.loads((ROOT / "step_1" / "manifest.json").read_text(encoding="utf-8"))
    pool = pd.read_parquet(ROOT / "step_1" / "pool.parquet")
    s3_cap = pd.read_csv(ROOT / "step_3" / "capturability.csv")
    s4_em = pd.read_csv(ROOT / "step_4" / "extraction_metrics.csv")
    s4_fi = pd.read_csv(ROOT / "step_4" / "feature_importance.csv")
    wfo = pd.read_csv(ROOT / "step_5" / "wfo_results.csv")
    holdout = pd.read_csv(ROOT / "step_5" / "holdout_results.csv")

    # Parse silhouettes + best K from cluster_summary.md
    cs_md = (ROOT / "step_2" / "cluster_summary.md").read_text(encoding="utf-8")
    silhouettes: dict[int, float] = {}
    k_selected = None
    for line in cs_md.splitlines():
        # | 2 | 0.3961 |
        if line.strip().startswith("| ") and "selected" not in line.lower():
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) == 2:
                try:
                    k = int(parts[0])
                    v = float(parts[1])
                    silhouettes[k] = v
                except ValueError:
                    pass
        if "K selected" in line or "K selected:" in line:
            for tok in line.split():
                try:
                    k_selected = int(tok)
                    break
                except ValueError:
                    continue
    if k_selected is None and silhouettes:
        k_selected = max(silhouettes, key=lambda k: silhouettes[k])

    # Step 3 per cluster
    step3_per_cluster = []
    for _, r in s3_cap.iterrows():
        step3_per_cluster.append({
            "cluster_id": int(r["cluster_id"]),
            "n_trades": int(r["n_trades"]),
            "shape_tag": str(r["shape_tag"]),
            "selected_sl_mult": float(r["selected_sl"]),
            "composite": float(r["capturability_composite"]),
            "reach_1r": float(r["reach_1r"]),
            "mfe_p50": float(r["mfe_p50"]),
            "ww_pp": float(r["wrong_way_pp"]),
            "is_candidate": bool(r["is_candidate"]),
        })
    candidate_cluster_ids = [c["cluster_id"] for c in step3_per_cluster if c["is_candidate"]]

    # Step 4 per cluster — aggregate from per-classifier per-fold metrics
    step4_per_cluster = []
    for cid in s4_em["cluster_id"].unique():
        sub = s4_em[s4_em["cluster_id"] == cid]
        clf_means = sub.groupby("classifier")["auc"].mean()
        if clf_means.empty:
            continue
        best_clf = clf_means.idxmax()
        best_auc = float(clf_means.max())
        best_thr_mean = float(sub[sub["classifier"] == best_clf]["threshold_auc_best"].mean())
        # Top 10 features
        fi_cid = s4_fi[s4_fi["cluster"] == cid] if "cluster" in s4_fi.columns else s4_fi[s4_fi["cluster_id"] == cid]
        if not fi_cid.empty:
            top10 = (
                fi_cid.groupby("feature")["importance"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .index.tolist()
            )
        else:
            top10 = []
        step4_per_cluster.append({
            "cluster_id": int(cid),
            "best_classifier": best_clf,
            "best_classifier_mean_auc": best_auc,
            "best_threshold": best_thr_mean,
            "n_trades": int(sub["n_admit_at_threshold"].sum())
                if "n_admit_at_threshold" in sub.columns else 0,
            "top_10_features": top10,
        })

    # Identify primary cluster (highest AUC)
    primary_cluster = None
    primary_classifier = None
    primary_threshold = 0.5
    if step4_per_cluster:
        primary = max(step4_per_cluster, key=lambda e: e["best_classifier_mean_auc"])
        primary_cluster = primary["cluster_id"]
        primary_classifier = primary["best_classifier"]
        primary_threshold = primary["best_threshold"]

    # search_results
    wfo_sorted = wfo.sort_values("worst_fold_ratio", ascending=False)
    search_results = [
        {
            "config_id": str(r["config_id"]),
            "verdict": str(r["verdict"]),
            "worst_fold_ratio": float(r["worst_fold_ratio"]),
            "worst_fold_roi": float(r["worst_fold_roi"]),
            "worst_fold_dd": float(r["worst_fold_dd"]),
            "mean_fold_ratio": float(r["mean_fold_ratio"]),
            "n_negative_folds": int(r["n_negative_folds"]),
            "min_trades_per_fold": int(r["min_trades_per_fold"]),
            "n_folds_evaluated": int(r["n_folds"]),
        }
        for _, r in wfo_sorted.iterrows()
    ]

    holdout_results = [
        {
            "config_id": str(r["config_id"]),
            "search_verdict": str(r["search_verdict"]),
            "holdout_verdict": str(r["holdout_verdict"]),
            "holdout_roi_pct": float(r["holdout_roi_pct"]),
            "holdout_dd_pct": float(r["holdout_dd_pct"]),
            "holdout_ratio": float(r["holdout_ratio"]),
            "deployable": bool(r["deployable"]),
        }
        for _, r in holdout.iterrows()
    ]

    # Arc verdict
    if not search_results:
        arc_verdict = "FAIL"
    else:
        best = search_results[0]
        h_match = next((h for h in holdout_results if h["config_id"] == best["config_id"]), None)
        if h_match is not None and h_match["deployable"]:
            arc_verdict = "PASS-DEPLOYABLE"
        elif h_match is not None and best["verdict"] == "pass_viable" and h_match["holdout_verdict"] in ("pass_deployable", "pass_viable"):
            arc_verdict = "PASS-VIABLE"
        else:
            arc_verdict = "FAIL"

    summary = {
        "arc_name": "l_arc_11",
        "verdict": arc_verdict,
        "pool_size": int(len(pool)),
        "pool_sha256": s1m["pool_sha256"],
        "k_selected": int(k_selected) if k_selected is not None else None,
        "silhouettes": {int(k): float(v) for k, v in silhouettes.items()},
        "candidate_cluster_ids": candidate_cluster_ids,
        "primary_cluster": primary_cluster,
        "primary_classifier": primary_classifier,
        "primary_threshold": primary_threshold,
        "n_configs_evaluated": len(search_results),
        "n_folds": 11,  # canonical builder produces 11; 10 evaluable after min_is_days=365
        "search_results": search_results,
        "holdout_results": holdout_results,
        "step_4_per_cluster": step4_per_cluster,
        "step_3_per_cluster": step3_per_cluster,
    }
    (ROOT / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"run_summary.json written. verdict={arc_verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
