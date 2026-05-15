"""Extract summary tables for the phase doc from the run outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/l_arc_1/step3_extractability")


def md_table(df: pd.DataFrame, cols=None, max_rows=None) -> str:
    if cols is not None:
        df = df[cols]
    if max_rows is not None:
        df = df.head(max_rows)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |"]
    lines.append("|" + "|".join("---" for _ in df.columns) + "|")
    for _, r in df.iterrows():
        vals = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append("NaN")
                elif abs(v) < 1e-3 or abs(v) > 1e6:
                    vals.append(f"{v:.3e}")
                else:
                    vals.append(f"{v:.4g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    # cluster summary
    cs = pd.read_csv(OUT / "cluster_summary_3a.csv")
    print("### Cluster size summary by K\n")
    for K in [2, 3, 4, 5, 6]:
        print(f"\n#### K = {K}\n")
        sub = cs[cs["K"] == K].copy()
        sub["frac_of_pool_pct"] = (sub["frac_of_pool"] * 100).round(2)
        print(md_table(sub[["K", "algo", "cluster_id", "n", "frac_of_pool_pct",
                            "below_15pct_floor", "silhouette_sample5k"]]))

    # ARI table
    print("\n### ARI table\n")
    ari = pd.read_csv(OUT / "cluster_ari_table_3a.csv")
    print(md_table(ari))

    # K=2 cluster path-distribution highlights
    print("\n### K=2 cluster distribution highlights (net_r, fwd_mfe_h24, fwd_mae_h24)\n")
    dist = pd.read_csv(OUT / "cluster_distributions_3a.csv")
    sub = dist[(dist["K"] == 2) & dist["metric"].isin([
        "net_r", "fwd_mfe_h24_atr", "fwd_mae_h24_atr",
        "fwd_mfe_to_mae_ratio_h24", "race_bars_plus1_minus_minus1",
    ])]
    print(md_table(sub[["K", "algo", "cluster_id", "metric", "n", "mean", "p50", "p95"]]))

    # stability (K=2 highlight)
    print("\n### Cluster stability (K=2)\n")
    st = pd.read_csv(OUT / "cluster_stability.csv")
    print(md_table(st[st["K"] == 2][["K", "algo", "cluster_id", "n_total",
                                       "meanR_cv_across_folds", "size_frac_range_normalised",
                                       "stability_flag_meanR_cv_ge_0.5",
                                       "stability_flag_size_range_gt_30pct"]]))

    # predictor AUC summary (Phase C)
    try:
        pa = pd.read_csv(OUT / "predictor_AUC_by_cluster.csv")
        print("\n### 3c predictor scan — pooled AUC at K=2 by model\n")
        sub = pa[pa["K"] == 2][["K", "algo", "cluster_id", "model",
                                "pooled_auc", "perfold_auc_mean", "perfold_auc_std"]]
        print(md_table(sub))

        print("\n### 3c predictor scan — pooled AUC at K=3 (top per-cluster)\n")
        sub = pa[pa["K"] == 3][["K", "algo", "cluster_id", "model",
                                "pooled_auc", "perfold_auc_mean", "perfold_auc_std"]]
        print(md_table(sub))
    except FileNotFoundError:
        print("\n(predictor_AUC_by_cluster.csv not yet present)\n")

    # calibration check
    print("\n### Calibration check status\n")
    try:
        text = (OUT / "calibration_check_status.txt").read_text(encoding="utf-8")
        print("```\n" + text + "\n```")
    except FileNotFoundError:
        print("(calibration_check_status.txt not yet present)")


if __name__ == "__main__":
    main()
