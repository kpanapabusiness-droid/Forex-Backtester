"""Calibration-FAIL §15 triage extraction.

Reads existing step 3 outputs (no recompute), produces seven supporting CSVs
under step3_triage/, and assembles the handover doc.

Per the resume prompt:
  - No re-run of Phase A–F computation.
  - No PHASE_L_ARC_1_STEP3.md.
  - No verdict.
  - No §15 outcome call.
  - No proposals.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "l_arc_1" / "step3_extractability"
TRI = OUT / "step3_triage"
SIGNALS = REPO / "results" / "l_arc_1" / "step2_descriptive" / "signals_features.csv"

TRI.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path, float_format: str = "%.10g") -> None:
    df.to_csv(path, index=False, float_format=float_format, lineterminator="\n")


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    hdr = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    out = [hdr, sep]
    for _, r in df.iterrows():
        vals = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append("NaN")
                elif abs(v) >= 1e-3 and abs(v) < 1e6:
                    vals.append(f"{v:.4g}")
                else:
                    vals.append(f"{v:.3e}")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


# =========================================================================
# Section 1 — Directory inventory
# =========================================================================
def section_1_directory_inventory() -> tuple[pd.DataFrame, str]:
    rows = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file():
            continue
        if p.parent.name == "step3_triage":
            continue  # exclude our own outputs
        rel = str(p.relative_to(OUT)).replace("\\", "/")
        stat = p.stat()
        rows.append(
            {
                "path": rel,
                "size_bytes": int(stat.st_size),
                "sha256": sha256(p),
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
        )
    df = pd.DataFrame(rows)
    write_csv(df, TRI / "directory_inventory.csv")
    md = "### Section 1 — Directory inventory\n\n" + md_table(df) + "\n"
    return df, md


# =========================================================================
# Section 2 — concurrent_signals_within_3h diagnostic
# =========================================================================
def section_2_csw3h_diagnostic() -> tuple[pd.DataFrame, str]:
    pred = pd.read_csv(OUT / "predictor_AUC_by_cluster.csv")
    hc = pd.read_csv(OUT / "look_elsewhere_haircut.csv")

    # csw3h ranks within each (K, algo, cid) signal-time scan ordered by BH p ascending.
    rows = []
    feature = "concurrent_signals_within_3h"
    hc_sig = hc[hc["scan"] == "signal_time"].copy()
    for (K, algo, cid), grp in hc_sig.groupby(["K", "algo", "cluster_id"]):
        grp_sorted = grp.sort_values("bh_p", ascending=True).reset_index(drop=True)
        match = grp_sorted[grp_sorted["feature"] == feature]
        if match.empty:
            continue
        rank_bh = int(match.index[0]) + 1
        row_hc = match.iloc[0]
        n_features_scanned = int(row_hc["n_features_in_scan"])
        # predictor AUCs per model for this (K, algo, cid)
        pred_sub = pred[(pred["K"] == K) & (pred["algo"] == algo) & (pred["cluster_id"] == cid)]
        for _, prow in pred_sub.iterrows():
            rows.append(
                {
                    "K": int(K),
                    "algo": algo,
                    "cluster_id": int(cid),
                    "model": prow["model"],
                    "ml_pooled_auc": float(prow["pooled_auc"]),
                    "ml_perfold_auc_mean": float(prow["perfold_auc_mean"]),
                    "ml_perfold_auc_std": float(prow["perfold_auc_std"]),
                    "univariate_auc": float(row_hc["univariate_auc"]),
                    "raw_p": float(row_hc["raw_p"]),
                    "bh_p": float(row_hc["bh_p"]),
                    "bh_tier": row_hc["tier"],
                    "rank_in_bh_ordered": rank_bh,
                    "n_features_scanned_for_this_cluster": n_features_scanned,
                    "permutation_count": 500,
                }
            )
    df = pd.DataFrame(rows)
    write_csv(df, TRI / "concurrent_signals_within_3h_diagnostic.csv")
    md = (
        "\n### Section 2 — `concurrent_signals_within_3h` diagnostic\n\n"
        "Per (K × algo × cluster_id × model). The ML AUCs are the pooled / per-fold AUCs\n"
        "from the **multivariate** model (all features). The univariate AUC, raw_p, bh_p,\n"
        "and bh_tier are the **univariate** measures of `concurrent_signals_within_3h`\n"
        "alone (the unit of multiple-comparison for the BH haircut per op spec §6.7).\n"
        "Permutation count = 500 (op spec §6.7 floor).\n\n" + md_table(df) + "\n"
    )
    return df, md


# =========================================================================
# Section 3 — Top-20 BH-cleared predictors at K=2 and K=3
# =========================================================================
def section_3_top20() -> tuple[pd.DataFrame, str]:
    hc = pd.read_csv(OUT / "look_elsewhere_haircut.csv")
    hc_sig = hc[hc["scan"] == "signal_time"].copy()
    pred = pd.read_csv(OUT / "predictor_AUC_by_cluster.csv")
    pred_max_auc = (
        pred.groupby(
            [
                "K",
                "algo",
                "cluster_id",
            ]
        )["pooled_auc"]
        .max()
        .reset_index()
    )
    pred_max_auc = pred_max_auc.rename(columns={"pooled_auc": "max_pooled_auc_across_models"})

    rows = []
    target_feature = "concurrent_signals_within_3h"
    for K in (2, 3):
        for algo in ("kmeans", "hierarchical"):
            for cid in sorted(
                hc_sig[(hc_sig["K"] == K) & (hc_sig["algo"] == algo)]["cluster_id"]
                .unique()
                .tolist()
            ):
                grp = hc_sig[
                    (hc_sig["K"] == K) & (hc_sig["algo"] == algo) & (hc_sig["cluster_id"] == cid)
                ].copy()
                grp = grp.sort_values("bh_p", ascending=True).reset_index(drop=True)
                grp["rank"] = grp.index + 1
                csw3h_rank = (
                    grp.loc[grp["feature"] == target_feature, "rank"].iloc[0]
                    if (grp["feature"] == target_feature).any()
                    else -1
                )
                top20 = grp.head(20)
                for _, r in top20.iterrows():
                    rows.append(
                        {
                            "K": K,
                            "algo": algo,
                            "cluster_id": int(cid),
                            "rank": int(r["rank"]),
                            "feature": r["feature"],
                            "univariate_auc": float(r["univariate_auc"]),
                            "raw_p": float(r["raw_p"]),
                            "bh_p": float(r["bh_p"]),
                            "tier": r["tier"],
                            "is_target_csw3h": bool(r["feature"] == target_feature),
                            "csw3h_rank_in_full_list": int(csw3h_rank),
                        }
                    )
    df = pd.DataFrame(rows)
    write_csv(df, TRI / "top20_BH_cleared_per_cluster.csv")
    # K=2 kmeans summary
    md_parts = ["\n### Section 3 — Top-20 BH-cleared predictors per cluster (K=2 + K=3)\n"]
    md_parts.append(
        "Sorted by BH-corrected p ascending. `csw3h_rank_in_full_list` shows where\n"
        "`concurrent_signals_within_3h` sits in the full ranked list for that cluster's\n"
        "scan (out of 37 signal-time features).\n"
    )
    for K in (2, 3):
        for algo in ("kmeans", "hierarchical"):
            for cid in sorted(
                df[(df["K"] == K) & (df["algo"] == algo)]["cluster_id"].unique().tolist()
            ):
                sub = df[(df["K"] == K) & (df["algo"] == algo) & (df["cluster_id"] == cid)]
                if sub.empty:
                    continue
                csw_rank = int(sub["csw3h_rank_in_full_list"].iloc[0])
                md_parts.append(
                    f"\n#### K={K} {algo} cluster_id={cid} (csw3h rank {csw_rank}/37)\n"
                )
                md_parts.append(
                    md_table(
                        sub[
                            [
                                "rank",
                                "feature",
                                "univariate_auc",
                                "raw_p",
                                "bh_p",
                                "tier",
                                "is_target_csw3h",
                            ]
                        ]
                    )
                )
                md_parts.append("")
    return df, "\n".join(md_parts) + "\n"


# =========================================================================
# Section 4 — Cluster identity audit
# =========================================================================
def section_4_cluster_identity() -> tuple[pd.DataFrame, str]:
    assignments = pd.read_csv(OUT / "cluster_assignments.csv")
    signals = pd.read_csv(SIGNALS)
    df = signals.merge(assignments, on="trade_id", how="left")
    len(df)

    rows = []
    for K in (2, 3, 4):
        for algo in ("kmeans", "hierarchical"):
            col = f"K{K}_{algo}"
            if col not in df.columns:
                continue
            for cid in sorted(df[col].unique().tolist()):
                mask = df[col] == cid
                sub = df[mask]
                exit_mix = sub["exit_reason"].value_counts(normalize=True)
                rows.append(
                    {
                        "K": K,
                        "algo": algo,
                        "cluster_id": int(cid),
                        "n_trades": int(mask.sum()),
                        "pct_of_pool": float(mask.mean()),
                        "mean_net_r": float(sub["net_r"].mean()),
                        "median_net_r": float(sub["net_r"].median()),
                        "median_mfe_held_atr": float(sub["mfe_held_atr"].median()),
                        "median_mae_held_atr": float(sub["mae_held_atr"].median()),
                        "median_fwd_mfe_h24_atr": float(sub["fwd_mfe_h24_atr"].median()),
                        "median_fwd_mae_h24_atr": float(sub["fwd_mae_h24_atr"].median()),
                        "median_fwd_mfe_h120_atr": float(sub["fwd_mfe_h120_atr"].median()),
                        "median_fwd_mae_h120_atr": float(sub["fwd_mae_h120_atr"].median()),
                        "median_race_condition": float(
                            sub["race_bars_plus1_minus_minus1"].median()
                        ),
                        "pct_exit_sl_hit": float(exit_mix.get("stop_loss", 0.0)),
                        "pct_exit_time_exit": float(exit_mix.get("time_exit", 0.0)),
                        "mean_concurrent_signals_within_3h": float(
                            sub["concurrent_signals_within_3h"].mean()
                        ),
                        "median_concurrent_signals_within_3h": float(
                            sub["concurrent_signals_within_3h"].median()
                        ),
                    }
                )
    cid_df = pd.DataFrame(rows)

    # Annotate which cluster the calibration check used.
    # Selection rule in run_step3.py:
    #   cid_high_mae = max([0, 1], key=lambda c: df.loc[lab == c, "fwd_mae_h24_atr"].mean())
    # = the cluster with the higher MEAN of fwd_mae_h24_atr at K=2.
    cid_df["cluster_used_for_calibration_check"] = False
    cid_df["calibration_check_selection_rule"] = ""
    for algo in ("kmeans", "hierarchical"):
        sub = cid_df[(cid_df["K"] == 2) & (cid_df["algo"] == algo)]
        # The selection rule used mean fwd_mae_h24 directly on the raw frame
        # (not the median we report here). Recompute mean from the raw df.
        means = {}
        for cid in (0, 1):
            mask = df[f"K2_{algo}"] == cid
            means[cid] = float(df.loc[mask, "fwd_mae_h24_atr"].mean())
        selected_cid = max(means, key=means.get)
        cid_df.loc[
            (cid_df["K"] == 2) & (cid_df["algo"] == algo) & (cid_df["cluster_id"] == selected_cid),
            "cluster_used_for_calibration_check",
        ] = True
        cid_df.loc[
            (cid_df["K"] == 2) & (cid_df["algo"] == algo) & (cid_df["cluster_id"] == selected_cid),
            "calibration_check_selection_rule",
        ] = "max mean(fwd_mae_h24_atr) within K=2 (per run_step3.py main())"

    # Characterisation: "high_mae_low_edge" vs "volatility_stratified"
    # high_mae_low_edge: low mean_net_r AND high pct_exit_sl_hit AND high median_fwd_mae
    # volatility_stratified: high both median_fwd_mfe AND median_fwd_mae, similar mean_net_r to pool
    pool_mean_r = float(df["net_r"].mean())
    pool_mfe24 = float(df["fwd_mfe_h24_atr"].median())
    pool_mae24 = float(df["fwd_mae_h24_atr"].median())
    chars = []
    for _, r in cid_df.iterrows():
        # heuristic descriptor — descriptive only per prompt
        edge_axis = r["mean_net_r"] - pool_mean_r  # negative = below-pool edge
        mae_axis = r["median_fwd_mae_h24_atr"] - pool_mae24  # positive = above-pool MAE
        mfe_axis = r["median_fwd_mfe_h24_atr"] - pool_mfe24  # positive = above-pool MFE
        # symmetric mfe + mae shift with mean_r near pool → volatility-stratified
        symmetric = abs(mfe_axis - mae_axis) < 0.30
        directional = mae_axis > 0.30 and edge_axis < -0.05
        if directional and not symmetric:
            chars.append("high_mae_low_edge")
        elif symmetric:
            chars.append("volatility_stratified")
        else:
            chars.append("mixed_or_neutral")
    cid_df["characterisation_descriptor"] = chars

    # Cluster size flag
    cid_df["filter_to_eligible_ge15pct"] = cid_df["pct_of_pool"] >= 0.15

    write_csv(cid_df, TRI / "cluster_identity_audit.csv")
    md_parts = ["\n### Section 4 — Cluster identity audit\n"]
    md_parts.append(
        "Selection rule used by `run_step3.py` for the calibration-check target:\n"
        "`max mean(fwd_mae_h24_atr)` within K=2.\n"
        "Characterisation descriptor heuristic:\n"
        "  - `high_mae_low_edge`: MAE↑ > pool by ≥0.30 ATR AND meanR < pool by ≥0.05.\n"
        "  - `volatility_stratified`: |MFE shift − MAE shift| < 0.30 ATR (symmetric).\n"
        "  - `mixed_or_neutral`: neither pure pattern.\n"
    )
    for K in (2, 3, 4):
        sub = cid_df[cid_df["K"] == K]
        if sub.empty:
            continue
        md_parts.append(f"\n#### K = {K}\n")
        md_parts.append(
            md_table(
                sub[
                    [
                        "algo",
                        "cluster_id",
                        "n_trades",
                        "pct_of_pool",
                        "mean_net_r",
                        "median_fwd_mfe_h24_atr",
                        "median_fwd_mae_h24_atr",
                        "median_race_condition",
                        "pct_exit_sl_hit",
                        "mean_concurrent_signals_within_3h",
                        "cluster_used_for_calibration_check",
                        "characterisation_descriptor",
                        "filter_to_eligible_ge15pct",
                    ]
                ]
            )
        )
        md_parts.append("")
    return cid_df, "\n".join(md_parts) + "\n"


# =========================================================================
# Section 5 — Feature matrix integrity check on concurrent_signals_within_3h
# =========================================================================
def section_5_feature_integrity() -> tuple[pd.DataFrame, str]:
    signals = pd.read_csv(SIGNALS, usecols=["concurrent_signals_within_3h"])
    col = signals["concurrent_signals_within_3h"]
    reported = {
        "mean": 10.24,
        "p50": 9.0,
        "p90": 19.0,
        "p95": 23.0,
        "p99": 30.0,
        "max": 44.0,
        "nan_count": 0,
    }
    actual = {
        "mean": float(col.mean()),
        "p50": float(col.quantile(0.50)),
        "p90": float(col.quantile(0.90)),
        "p95": float(col.quantile(0.95)),
        "p99": float(col.quantile(0.99)),
        "max": float(col.max()),
        "nan_count": int(col.isna().sum()),
    }
    rows = []
    for k, rep in reported.items():
        act = actual[k]
        match = (abs(act - rep) <= 0.05) if rep != 0 else (act == 0)
        rows.append(
            {
                "statistic": k,
                "step2_reported": rep,
                "step3_actual": act,
                "match_within_tolerance": bool(match),
            }
        )
    # zero count (info)
    rows.append(
        {
            "statistic": "zero_count",
            "step2_reported": float("nan"),
            "step3_actual": int((col == 0).sum()),
            "match_within_tolerance": None,
        }
    )
    df = pd.DataFrame(rows)
    write_csv(df, TRI / "feature_integrity_check.csv")
    md = (
        "\n### Section 5 — Feature matrix integrity check on `concurrent_signals_within_3h`\n\n"
        + md_table(df)
        + "\n"
    )
    return df, md


# =========================================================================
# Section 6 — Sibling-feature diagnostic
# =========================================================================
def section_6_sibling_features() -> tuple[pd.DataFrame, str]:
    hc = pd.read_csv(OUT / "look_elsewhere_haircut.csv")
    hc_sig = hc[hc["scan"] == "signal_time"].copy()
    siblings = [
        "concurrent_signals_same_bar",
        "concurrent_signals_within_3h",
        "currency_basket_3h_USD",
        "currency_basket_3h_EUR",
        "currency_basket_3h_JPY",
        "currency_basket_3h_GBP",
        "trade_overlap_at_execution_time",
        "sequential_same_pair_density_24h",
    ]
    rows = []
    for K in (2, 3, 4):
        for algo in ("kmeans", "hierarchical"):
            for cid in sorted(
                hc_sig[(hc_sig["K"] == K) & (hc_sig["algo"] == algo)]["cluster_id"]
                .unique()
                .tolist()
            ):
                grp = hc_sig[
                    (hc_sig["K"] == K) & (hc_sig["algo"] == algo) & (hc_sig["cluster_id"] == cid)
                ].copy()
                for sib in siblings:
                    row = grp[grp["feature"] == sib]
                    if row.empty:
                        rows.append(
                            {
                                "K": K,
                                "algo": algo,
                                "cluster_id": int(cid),
                                "feature": sib,
                                "univariate_auc": float("nan"),
                                "raw_p": float("nan"),
                                "bh_p": float("nan"),
                                "tier": "absent",
                            }
                        )
                        continue
                    r = row.iloc[0]
                    rows.append(
                        {
                            "K": K,
                            "algo": algo,
                            "cluster_id": int(cid),
                            "feature": sib,
                            "univariate_auc": float(r["univariate_auc"]),
                            "raw_p": float(r["raw_p"]),
                            "bh_p": float(r["bh_p"]),
                            "tier": r["tier"],
                        }
                    )
    df = pd.DataFrame(rows)
    write_csv(df, TRI / "sibling_family_diagnostic.csv")

    # Summary: count of cleared T1/T2 per (K, algo, cid) within the sibling family
    summary = (
        df.groupby(["K", "algo", "cluster_id", "tier"]).size().unstack(fill_value=0).reset_index()
    )
    md_parts = [
        "\n### Section 6 — Sibling-family diagnostic\n",
        "Per (K × algo × cluster_id), tier counts within the 8 cross-pair/portfolio "
        "features (op spec §5.15):\n\n",
        md_table(summary),
        "\n\n",
        "#### Full table (K=2 highlights)\n",
        md_table(
            df[df["K"] == 2][
                ["K", "algo", "cluster_id", "feature", "univariate_auc", "raw_p", "bh_p", "tier"]
            ]
        ),
        "\n",
    ]
    return df, "\n".join(md_parts) + "\n"


# =========================================================================
# Section 7 — K=2 split quality
# =========================================================================
def section_7_k2_split_quality() -> tuple[pd.DataFrame, str]:
    summary = pd.read_csv(OUT / "cluster_summary_3a.csv")
    ari_tbl = pd.read_csv(OUT / "cluster_ari_table_3a.csv")
    assignments = pd.read_csv(OUT / "cluster_assignments.csv")
    signals = pd.read_csv(SIGNALS)

    # Silhouette per algo at K=2
    rows = []
    for algo in ("kmeans", "hierarchical"):
        sub = summary[(summary["K"] == 2) & (summary["algo"] == algo)]
        sil = float(sub["silhouette_sample5k"].iloc[0])
        sizes = sorted(sub["n"].tolist())
        split_str = "/".join(str(int(round(100 * x / 45673))) for x in sizes)
        ari_row = ari_tbl[(ari_tbl["K"] == 2) & (ari_tbl["algo"] == algo)].iloc[0]
        rows.append(
            {
                "algo": algo,
                "silhouette_sample5k": sil,
                "size_smaller_n": int(min(sizes)),
                "size_larger_n": int(max(sizes)),
                "split_pct_smaller_larger": split_str,
                "ari_kmeans_vs_hierarchical_K2": float(ari_row["ari_kmeans_vs_hierarchical_sameK"]),
            }
        )

    # Between-cluster variance per clustering feature (top 5)
    cluster_features_num = (
        "mfe_held_atr",
        "mae_held_atr",
        "fwd_mfe_h24_atr",
        "fwd_mae_h24_atr",
        "fwd_mfe_h120_atr",
        "fwd_mae_h120_atr",
        "fwd_time_to_peak_mfe",
        "fwd_time_to_trough_mae",
        "fwd_oscillation_count",
        "fwd_monotonicity_ratio",
    )
    merged = signals.merge(
        assignments[["trade_id", "K2_kmeans", "K2_hierarchical"]], on="trade_id", how="left"
    )

    bcv_rows = []
    for algo in ("kmeans", "hierarchical"):
        col = f"K2_{algo}"
        for feat in cluster_features_num:
            grouped = merged.groupby(col)[feat]
            means = grouped.mean()
            grand = merged[feat].mean()
            sizes = grouped.size()
            between_var = float(((means - grand) ** 2 * sizes).sum() / sizes.sum())
            total_var = float(merged[feat].var(ddof=0))
            ratio = between_var / total_var if total_var > 0 else float("nan")
            bcv_rows.append(
                {
                    "algo": algo,
                    "feature": feat,
                    "between_cluster_variance": between_var,
                    "total_variance": total_var,
                    "bcv_to_total_ratio": ratio,
                }
            )
    bcv_df = pd.DataFrame(bcv_rows)
    # Top 5 per algo
    top_feats = []
    for algo in ("kmeans", "hierarchical"):
        sub = (
            bcv_df[bcv_df["algo"] == algo]
            .sort_values("bcv_to_total_ratio", ascending=False)
            .head(5)
        )
        top_feats.append(sub)
    top_df = pd.concat(top_feats, ignore_index=True)

    qual = pd.DataFrame(rows)
    write_csv(qual, TRI / "k2_split_quality.csv")
    write_csv(top_df, TRI / "k2_top5_drivers_by_bcv.csv")

    md_parts = [
        "\n### Section 7 — K=2 split quality\n",
        md_table(qual),
        "\n\n",
        "#### Top 5 clustering features by between-cluster variance ratio (per algo, K=2)\n",
        md_table(top_df),
        "\n",
    ]
    return qual, "\n".join(md_parts) + "\n"


# =========================================================================
# Missing-outputs detection
# =========================================================================
def detect_missing() -> str:
    expected_phase_F = [
        "look_elsewhere_haircut.csv",
        "calibration_check_status.txt",
    ]
    expected_postcalib = [
        "predictor_AUC_by_cluster_by_t.csv",
        "feature_importance_3d.csv",
        "auc_threshold_crossings_3d.csv",
        "cluster_effect_sizes.csv",
        "filter_dry_run.csv",
    ]
    expected_strat_axes = [
        "pair",
        "vol_regime",
        "session",
        "day_of_week",
        "hour_of_day",
        "hour_in_4h_bar",
        "hour_in_d1_bar",
        "pre_momentum_bin",
        "trigger_magnitude_decile",
    ]
    missing = []
    for f in expected_phase_F:
        if not (OUT / f).exists():
            missing.append(("Phase F", f))
    for f in expected_postcalib:
        if not (OUT / f).exists():
            missing.append(("Post-calibration (halted)", f))
    for axis in expected_strat_axes:
        for child in ("cluster_mix.csv", "conditional_forward_geometry.csv"):
            if not (OUT / "stratifications" / axis / child).exists():
                missing.append(("Phase E stratifications", f"stratifications/{axis}/{child}"))
    not_written = [
        ("Phase doc", "PHASE_L_ARC_1_STEP3.md (suppressed per calibration-FAIL halt rule)"),
        ("Step 3 sanity checks", "step3_sanity_checks.txt (gated on phase doc)"),
    ]
    lines = [
        "\n### §missing_outputs\n",
        "Outputs absent or incomplete due to the calibration-FAIL halt at Phase F:\n",
    ]
    for tag, f in missing:
        lines.append(f"- **{tag}:** `{f}`")
    lines.append("")
    lines.append("Outputs intentionally NOT written per the halt rule:")
    for tag, f in not_written:
        lines.append(f"- **{tag}:** `{f}`")
    lines.append("")
    return "\n".join(lines)


# =========================================================================
# Determinism receipt context
# =========================================================================
def determinism_summary() -> str:
    p = OUT / "determinism_check.txt"
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    n_total = text.count("    match: ")
    n_diff = text.count("    match: False")
    n_same = text.count("    match: True")

    body = [
        "\n### Section 11 — Determinism receipt (re-run vs run1)\n",
        f"Files checked: {n_total}",
        f"Match: {n_same}",
        f"Differ: {n_diff}",
        "",
        "Result reported: " + ("**PASS**" if n_diff == 0 else "**FAIL** (partial)") + "",
        "",
        "Two files differed across the re-run: `look_elsewhere_haircut.csv` and",
        "`calibration_check_status.txt`. Univariate AUCs are byte-identical;",
        "the differences are in the permutation-derived BH p-values, which retain",
        "Monte-Carlo variance because the per-feature permutation seed is composed",
        "with `hash(fname)`, and Python's string `hash` is per-process randomized",
        "unless `PYTHONHASHSEED` is pinned. The differences are:",
        "",
        "| File | run1 | run2 |",
        "|---|---|---|",
        "| K=2 kmeans cid=1 BH-p for csw3h | 0.4044 | 0.4431 |",
        "| K=2 hierarchical cid=1 BH-p for csw3h | 0.9993 | 0.9901 |",
        "",
        "**Tier assignment for `concurrent_signals_within_3h` is Tier 3 in both",
        "runs; the CALIBRATION STATUS is FAIL in both runs.** The halt outcome",
        "is therefore stable across runs. The non-determinism is in BH p-value",
        "Monte-Carlo precision, not in the calibration-check conclusion.",
        "Documented as a fixable narrow determinism gap (seed `hash(fname)` →",
        "deterministic hash like `int.from_bytes(hashlib.sha256(fname.encode())",
        ".digest()[:4], 'big')` in `run_step3.py`'s haircut loops).",
        "",
        "Full receipt: `determinism_check.txt`.",
        "",
    ]
    return "\n".join(body)


# =========================================================================
# Main
# =========================================================================
def main() -> int:
    print("running triage extraction (no recompute) …")
    df1, md1 = section_1_directory_inventory()
    df2, md2 = section_2_csw3h_diagnostic()
    df3, md3 = section_3_top20()
    df4, md4 = section_4_cluster_identity()
    df5, md5 = section_5_feature_integrity()
    df6, md6 = section_6_sibling_features()
    df7, md7 = section_7_k2_split_quality()
    missing_md = detect_missing()
    determinism_md = determinism_summary()

    header = """# step3_triage_handover — Arc 1 (redo) calibration-FAIL §15 triage

> **No verdict committed. No §15 outcome called. No filter / exit proposals.**
> This document extracts data from existing step 3 outputs to support the
> planning chat's §15 (a / b / c) triage. The standard `PHASE_L_ARC_1_STEP3.md`
> phase doc is suppressed per the calibration-FAIL halt rule.

---

## 1. Header

| Field | Value |
|---|---|
| Arc | L Arc 1 (redo) — protocol calibration check |
| Step | 3 — Extractability Assessment (halted at Phase F calibration check) |
| Protocol | `L_ARC_PROTOCOL.md` v1.0 |
| Operational spec | `L_ARC_OPERATIONAL_SPEC.md` v1.0 |
| Arc-open doc | `results/PHASE_L_ARC_1_OPEN.md` |
| Step 1 phase doc | `results/l_arc_1/step1_verbatim/PHASE_L_ARC_1_STEP1.md` |
| Step 2 phase doc | `results/l_arc_1/step2_descriptive/PHASE_L_ARC_1_STEP2.md` |
| Halt reason | Phase F BH haircut → `concurrent_signals_within_3h` Tier 3 across K=2 (both algos) |
| Halt trigger | step 3 prompt Phase C calibration-check rule + protocol §15 |
| Phase coverage | A, B, C, F (signal-time only) all complete; D, E, G, H not run |

---

## 2. Halt summary

Phases A–C ran to completion. Phase F (BH look-elsewhere haircut) ran for the
signal-time scan only (37 features × 18 clusters = 666 univariate AUC + 500-perm
p-value computations + BH correction per cluster). At the end of Phase F the
calibration check rule fired (per protocol §15 + step 3 prompt halt rule).

**On disk:** all Phase A–C outputs; Phase F's signal-time outputs
(`look_elsewhere_haircut.csv`, `calibration_check_status.txt`,
`multicollinearity_top20.csv`); the determinism receipt; and this triage bundle.

**Not on disk:** Phase D (held-bar predictor scan), Phase E (stratifications),
Phase G (cluster effect sizes), Phase H (filter dry-run), and
`PHASE_L_ARC_1_STEP3.md`. All suppressed per the halt rule.

---
"""

    "\n".join(
        [
            header,
            md1,
            determinism_md.replace(
                "Section 11", "Section 11 — placed at end"
            ),  # no-op rename; we keep the determinism block down below for clarity
        ]
    )
    # Reorder so determinism shows at the end with section number 11
    doc = "\n".join(
        [
            header,
            md1,
            missing_md,
            md2,
            md3,
            md4,
            md5,
            md6,
            md7,
            determinism_md,
            "\n### Section 12 — §verdict_pending — calibration-check FAIL\n",
            "No verdict is committed here.",
            "",
            "The Phase F outcome was Tier 3 for `concurrent_signals_within_3h` against",
            "the calibration-check target cluster (K=2 max-MAE cluster at both kmeans",
            "and hierarchical). Per `L_ARC_PROTOCOL.md` §15 the planner runs the",
            "outcome (a) / (b) / (c) investigation in a separate doc.",
            "",
            "No interpretation of the §15 outcome is offered here.",
            "",
        ]
    )

    out_md = OUT / "step3_triage_handover.md"
    out_md.write_text(doc, encoding="utf-8", newline="\n")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
