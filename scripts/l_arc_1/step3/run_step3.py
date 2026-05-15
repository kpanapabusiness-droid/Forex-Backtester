# ruff: noqa: E402  (sys.path.insert needed before project imports)
"""L Arc 1 Step 3 orchestrator — v1.1 amendment spec.

Runs Phases A–H per `L_ARC_PROTOCOL_v1.1_AMENDMENT.md`.

Amendments applied (numbered per amendment doc):
  1. Family-level calibration check (replaces feature-level)
  2. BH-tier reportorial, not eliminative (Tier 3 → Phase G/H eligible)
  3. HDBSCAN added as 3rd clustering method
  4. 3 new clustering features (fwd_realized_range_atr,
     fwd_fraction_time_above_entry, fwd_max_consecutive_directional_bars)
  5. PCA pre-check (Phase A.0)
  6. K range extended to 2..8
  7. Differentiated cluster-size floor (15% filter-TO, 5% exit/delayed, no floor filter-OUT)
  8. Delayed-entry candidate type (downstream — flagged in handover)
  9. Partial AUC at worst-decile cutoff
 10. Expected-R-volume filter ranking
 11. hashlib.sha256 permutation seeds (determinism fix)
 12. Cluster-target selection rule = lowest mean net_r (ties: max fwd_mae_h24)

Output: results/l_arc_1/step3_extractability/
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

# Allow `python -m scripts.l_arc_1.step3.run_step3` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="pandas")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from scripts.l_arc_1.step3 import _clustering as CL
from scripts.l_arc_1.step3 import _common as C
from scripts.l_arc_1.step3 import _data as D
from scripts.l_arc_1.step3 import _haircut as HC
from scripts.l_arc_1.step3 import _pca as PCA
from scripts.l_arc_1.step3 import _predictor as P


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.STRAT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"input signals sha256={C.sha256_file(C.SIGNALS_CSV)[:16]}…")
    df = D.load_signals()
    log(f"loaded signals: {len(df)} rows × {len(df.columns)} cols")

    # ---- v1.1 Amendment 4: compute new clustering features ----
    log("computing v1.1 Amendment 4 features from trade_paths")
    v11_cache = C.OUT_DIR / "v11_new_features.csv"
    new_feats = D.compute_v11_new_features(df, cache_path=v11_cache)
    df = df.merge(new_feats, on="trade_id", how="left")
    log(
        "  added 3 new features: fwd_realized_range_atr, "
        "fwd_fraction_time_above_entry, fwd_max_consecutive_directional_bars"
    )

    fold_id = df["fold_id"].values.astype(int)
    n = len(df)
    pool_mean_r = float(df["net_r"].mean())

    manifest = {
        "step": "3_extractability_v1_1",
        "input_signals_features_sha256": C.sha256_file(C.SIGNALS_CSV),
        "input_trade_paths_sha256": C.sha256_file(C.PATHS_CSV),
        "n_rows": int(n),
        "pool_mean_r": pool_mean_r,
        "base_seed": C.BASE_SEED,
        "n_permutations": C.N_PERMUTATIONS,
        "ward_subsample": C.WARD_SUBSAMPLE,
        "v11_amendments_applied": list(range(1, 13)),
    }

    # =================================================================
    # PHASE A.0 — PCA pre-check (Amendment 5)
    # =================================================================
    log("PHASE A.0: PCA pre-check on clustering features")
    X_cluster, cluster_feat_names = D.build_cluster_features(df)
    log(f"  cluster features matrix: {X_cluster.shape}")
    pc_summary, loadings, corr_mat = PCA.pca_diagnostic(X_cluster, cluster_feat_names)
    C.write_csv(pc_summary, C.OUT_DIR / "phase_a0_pca_summary.csv")
    C.write_csv(loadings, C.OUT_DIR / "phase_a0_pca_loadings.csv")
    C.write_csv(corr_mat, C.OUT_DIR / "phase_a0_correlation_matrix.csv")
    n_redundant = int(corr_mat["flagged_redundant_gt_0.85"].sum())
    log(
        f"  PC1+PC2+PC3 cumulative variance: {pc_summary['cumulative_variance_ratio'].iloc[-1]:.3f}"
    )
    log(f"  pairs flagged candidate-redundant (|r|>0.85): {n_redundant}")
    manifest["pca_pc3_cumulative"] = float(pc_summary["cumulative_variance_ratio"].iloc[-1])
    manifest["pca_redundant_pair_count"] = n_redundant

    # =================================================================
    # PHASE A — Cluster discovery (Amendments 3, 4, 6, 12)
    # =================================================================
    log("PHASE A: cluster discovery (k-means + hierarchical at K=2..8 + HDBSCAN)")
    cluster_assignments = pd.DataFrame({"trade_id": df["trade_id"].values})
    summary_rows = []

    for k in C.K_VALUES:
        km = CL.fit_kmeans(X_cluster, k, seed=C.BASE_SEED + k)
        cluster_assignments[f"K{k}_kmeans"] = km
        sil_km = CL.silhouette_sample(X_cluster, km, seed=C.BASE_SEED + k)
        for cid, sd in CL.cluster_size_distribution(km).items():
            summary_rows.append(
                {
                    "algo": "kmeans",
                    "K": k,
                    "cluster_id": int(cid),
                    "n": sd["n"],
                    "frac_of_pool": sd["frac"],
                    "below_15pct_filter_to_floor": bool(sd["frac"] < C.SIZE_FLOOR_FILTER_TO),
                    "below_5pct_exit_delayed_floor": bool(
                        sd["frac"] < C.SIZE_FLOOR_EXIT_OR_DELAYED_ENTRY
                    ),
                    "silhouette_sample5k": sil_km,
                }
            )

        hi = CL.fit_hierarchical_ward(X_cluster, k, seed=C.BASE_SEED + k)
        cluster_assignments[f"K{k}_hierarchical"] = hi
        sil_hi = CL.silhouette_sample(X_cluster, hi, seed=C.BASE_SEED + k + 1000)
        for cid, sd in CL.cluster_size_distribution(hi).items():
            summary_rows.append(
                {
                    "algo": "hierarchical",
                    "K": k,
                    "cluster_id": int(cid),
                    "n": sd["n"],
                    "frac_of_pool": sd["frac"],
                    "below_15pct_filter_to_floor": bool(sd["frac"] < C.SIZE_FLOOR_FILTER_TO),
                    "below_5pct_exit_delayed_floor": bool(
                        sd["frac"] < C.SIZE_FLOOR_EXIT_OR_DELAYED_ENTRY
                    ),
                    "silhouette_sample5k": sil_hi,
                }
            )

    # HDBSCAN (v1.1 Amendment 3)
    log("  fitting HDBSCAN")
    hd = CL.fit_hdbscan(X_cluster, seed=C.BASE_SEED)
    cluster_assignments["hdbscan_cluster_id"] = hd
    non_noise = hd != -1
    if non_noise.sum() > 100 and len(set(hd[non_noise])) > 1:
        sil_hd = CL.silhouette_sample(X_cluster[non_noise], hd[non_noise], seed=C.BASE_SEED + 5000)
    else:
        sil_hd = float("nan")
    n_hdbscan_clusters = int(len(set(hd[non_noise])))
    noise_count = int((hd == -1).sum())
    hdbscan_skipped_no_clusters = n_hdbscan_clusters < 2
    if hdbscan_skipped_no_clusters:
        log(
            "  HDBSCAN found < 2 non-noise clusters at min_cluster_size=5% pool; "
            "skipping HDBSCAN from target selection + Phase C/D (Amendment 3 reportorial)"
        )
    log(
        f"  HDBSCAN: {n_hdbscan_clusters} clusters, {noise_count} noise points ({100 * noise_count / n:.1f}%)"
    )
    for cid in sorted(set(hd.tolist())):
        mask = hd == cid
        frac = float(mask.mean())
        summary_rows.append(
            {
                "algo": "hdbscan",
                "K": int(n_hdbscan_clusters),
                "cluster_id": int(cid),
                "n": int(mask.sum()),
                "frac_of_pool": frac,
                "below_15pct_filter_to_floor": bool(frac < C.SIZE_FLOOR_FILTER_TO),
                "below_5pct_exit_delayed_floor": bool(frac < C.SIZE_FLOOR_EXIT_OR_DELAYED_ENTRY),
                "silhouette_sample5k": sil_hd,
            }
        )

    C.write_csv(cluster_assignments, C.OUT_DIR / "cluster_assignments.csv")
    cluster_summary = pd.DataFrame(summary_rows)
    C.write_csv(cluster_summary, C.OUT_DIR / "cluster_summary.csv")
    manifest["hdbscan_n_clusters"] = n_hdbscan_clusters
    manifest["hdbscan_noise_fraction"] = float(noise_count / n)

    # ARI: k-means vs hierarchical at same K; k-means K=2 baseline; HDBSCAN vs kmeans/hier
    ari_rows = []
    km_k2 = cluster_assignments["K2_kmeans"].values
    for k in C.K_VALUES:
        ari_rows.append(
            {
                "K_pair": k,
                "ari_kmeans_vs_hierarchical_sameK": CL.ari(
                    cluster_assignments[f"K{k}_kmeans"].values,
                    cluster_assignments[f"K{k}_hierarchical"].values,
                ),
                "ari_kmeans_vs_K2_kmeans_baseline": CL.ari(
                    km_k2, cluster_assignments[f"K{k}_kmeans"].values
                ),
                "ari_hierarchical_vs_K2_kmeans_baseline": CL.ari(
                    km_k2, cluster_assignments[f"K{k}_hierarchical"].values
                ),
            }
        )
    # HDBSCAN vs nearest K (= n_hdbscan_clusters)
    nearest_k = min(C.K_VALUES, key=lambda kk: abs(kk - n_hdbscan_clusters))
    ari_rows.append(
        {
            "K_pair": f"hdbscan_vs_kmeans_K{nearest_k}",
            "ari_kmeans_vs_hierarchical_sameK": CL.ari(
                hd, cluster_assignments[f"K{nearest_k}_kmeans"].values
            ),
            "ari_kmeans_vs_K2_kmeans_baseline": CL.ari(hd, km_k2),
            "ari_hierarchical_vs_K2_kmeans_baseline": CL.ari(
                hd, cluster_assignments[f"K{nearest_k}_hierarchical"].values
            ),
        }
    )
    C.write_csv(pd.DataFrame(ari_rows), C.OUT_DIR / "cluster_ari_table.csv")

    # Per-cluster distributions
    log("  computing per-cluster distributions")
    dist_metrics = (
        "net_r",
        "gross_r",
        "mfe_R",
        "mae_R",
        "fwd_mfe_h24_atr",
        "fwd_mae_h24_atr",
        "fwd_mfe_h120_atr",
        "fwd_mae_h120_atr",
        "fwd_mfe_h240_atr",
        "race_bars_plus1_minus_minus1",
        "fwd_mfe_to_mae_ratio_h24",
        "fwd_mfe_to_mae_ratio_h120",
        "fwd_realized_range_atr",
        "fwd_fraction_time_above_entry",
        "fwd_max_consecutive_directional_bars",
    )
    dist_rows = []
    for algo, col_pattern in (
        ("kmeans", "K{}_kmeans"),
        ("hierarchical", "K{}_hierarchical"),
    ):
        for k in C.K_VALUES:
            col = col_pattern.format(k)
            lab = cluster_assignments[col].values
            for cid in sorted(set(lab.tolist())):
                mask = lab == cid
                exit_mix = df.loc[mask, "exit_reason"].value_counts(normalize=True).to_dict()
                for m in dist_metrics:
                    stats = C.fmt_dist_stats(df.loc[mask, m].values)
                    dist_rows.append(
                        {"algo": algo, "K": k, "cluster_id": int(cid), "metric": m, **stats}
                    )
                dist_rows.append(
                    {
                        "algo": algo,
                        "K": k,
                        "cluster_id": int(cid),
                        "metric": "exit_reason_frac",
                        "frac_time_exit": float(exit_mix.get("time_exit", 0.0)),
                        "frac_stop_loss": float(exit_mix.get("stop_loss", 0.0)),
                    }
                )
    # HDBSCAN distributions
    for cid in sorted(set(hd.tolist())):
        mask = hd == cid
        for m in dist_metrics:
            stats = C.fmt_dist_stats(df.loc[mask, m].values)
            dist_rows.append(
                {
                    "algo": "hdbscan",
                    "K": n_hdbscan_clusters,
                    "cluster_id": int(cid),
                    "metric": m,
                    **stats,
                }
            )
        exit_mix = df.loc[mask, "exit_reason"].value_counts(normalize=True).to_dict()
        dist_rows.append(
            {
                "algo": "hdbscan",
                "K": n_hdbscan_clusters,
                "cluster_id": int(cid),
                "metric": "exit_reason_frac",
                "frac_time_exit": float(exit_mix.get("time_exit", 0.0)),
                "frac_stop_loss": float(exit_mix.get("stop_loss", 0.0)),
            }
        )
    C.write_csv(pd.DataFrame(dist_rows), C.OUT_DIR / "cluster_distributions.csv")

    # ---- Amendment 12: target cluster selection per (algo, K) ----
    log("PHASE A.4: cluster-target selection (Amendment 12: lowest mean net_r)")
    target_rows = []
    targets_by_algo_k = {}  # (algo, K) -> target_cluster_id
    for algo, col_pattern in (
        ("kmeans", "K{}_kmeans"),
        ("hierarchical", "K{}_hierarchical"),
    ):
        for k in C.K_VALUES:
            col = col_pattern.format(k)
            lab = cluster_assignments[col].values
            best_cid = None
            best_meanR = float("inf")
            best_tie_mae = float("-inf")
            for cid in sorted(set(lab.tolist())):
                mask = lab == cid
                meanR = float(df.loc[mask, "net_r"].mean())
                mae_med = float(df.loc[mask, "fwd_mae_h24_atr"].median())
                # lowest mean R; ties broken by highest median fwd_mae_h24
                if meanR < best_meanR - 1e-12:
                    best_meanR = meanR
                    best_cid = int(cid)
                    best_tie_mae = mae_med
                elif abs(meanR - best_meanR) <= 1e-12 and mae_med > best_tie_mae:
                    best_cid = int(cid)
                    best_tie_mae = mae_med
            target_rows.append(
                {
                    "algo": algo,
                    "K": k,
                    "target_cluster_id": int(best_cid),
                    "target_mean_net_r": best_meanR,
                    "target_median_fwd_mae_h24": best_tie_mae,
                    "selection_rule": "lowest mean net_r; ties broken by max median fwd_mae_h24",
                }
            )
            targets_by_algo_k[(algo, k)] = int(best_cid)
    # HDBSCAN target — only if it found ≥ 2 non-noise clusters
    if not hdbscan_skipped_no_clusters:
        best_cid = None
        best_meanR = float("inf")
        best_tie_mae = float("-inf")
        for cid in sorted(set(hd[non_noise].tolist())):
            mask = hd == cid
            meanR = float(df.loc[mask, "net_r"].mean())
            mae_med = float(df.loc[mask, "fwd_mae_h24_atr"].median())
            if meanR < best_meanR - 1e-12:
                best_meanR = meanR
                best_cid = int(cid)
                best_tie_mae = mae_med
            elif abs(meanR - best_meanR) <= 1e-12 and mae_med > best_tie_mae:
                best_cid = int(cid)
                best_tie_mae = mae_med
        target_rows.append(
            {
                "algo": "hdbscan",
                "K": n_hdbscan_clusters,
                "target_cluster_id": int(best_cid),
                "target_mean_net_r": best_meanR,
                "target_median_fwd_mae_h24": best_tie_mae,
                "selection_rule": "lowest mean net_r; ties broken by max median fwd_mae_h24",
            }
        )
        targets_by_algo_k[("hdbscan", n_hdbscan_clusters)] = int(best_cid)
    else:
        target_rows.append(
            {
                "algo": "hdbscan",
                "K": 0,
                "target_cluster_id": -999,
                "target_mean_net_r": float("nan"),
                "target_median_fwd_mae_h24": float("nan"),
                "selection_rule": "skipped — HDBSCAN found <2 non-noise clusters",
            }
        )
    manifest["hdbscan_skipped_no_clusters"] = bool(hdbscan_skipped_no_clusters)
    C.write_csv(pd.DataFrame(target_rows), C.OUT_DIR / "cluster_target_selection.csv")
    log(f"  selected target cluster ids: {targets_by_algo_k}")

    # =================================================================
    # PHASE B — Cluster stability
    # =================================================================
    log("PHASE B: cluster stability + feature-perturbation ARI")
    stab_rows = []
    iter_specs = (
        [("kmeans", k, f"K{k}_kmeans") for k in C.K_VALUES]
        + [("hierarchical", k, f"K{k}_hierarchical") for k in C.K_VALUES]
        + [("hdbscan", n_hdbscan_clusters, "hdbscan_cluster_id")]
    )
    for algo, k, col in iter_specs:
        lab = cluster_assignments[col].values
        for cid in sorted(set(lab.tolist())):
            mask_c = lab == cid
            n_total_c = int(mask_c.sum())
            per_fold_n, per_fold_meanR, per_fold_size_frac = [], [], []
            for f in range(1, C.N_FOLDS + 1):
                fmask = mask_c & (fold_id == f)
                n_in = int(fmask.sum())
                fold_total = int((fold_id == f).sum())
                per_fold_n.append(n_in)
                per_fold_size_frac.append(n_in / fold_total if fold_total > 0 else 0.0)
                per_fold_meanR.append(
                    float(df.loc[fmask, "net_r"].mean()) if n_in > 0 else float("nan")
                )
            meanR_arr = np.array(per_fold_meanR)
            size_arr = np.array(per_fold_size_frac)
            meanR_cv = (
                float(np.nanstd(meanR_arr, ddof=1) / abs(np.nanmean(meanR_arr)))
                if not np.isnan(meanR_arr).all() and abs(np.nanmean(meanR_arr)) > 1e-9
                else float("nan")
            )
            avg_size = float(np.mean(size_arr))
            size_range_norm = float(np.max(size_arr) - np.min(size_arr)) / max(avg_size, 1e-9)
            stab_rows.append(
                {
                    "algo": algo,
                    "K": k,
                    "cluster_id": int(cid),
                    "n_total": n_total_c,
                    **{f"n_fold{f}": per_fold_n[f - 1] for f in range(1, 8)},
                    **{f"meanR_fold{f}": per_fold_meanR[f - 1] for f in range(1, 8)},
                    **{f"size_frac_fold{f}": per_fold_size_frac[f - 1] for f in range(1, 8)},
                    "meanR_cv_across_folds": meanR_cv,
                    "size_frac_range_normalised": size_range_norm,
                    "stability_flag_meanR_cv_ge_0.5": bool(meanR_cv >= 0.5)
                    if not np.isnan(meanR_cv)
                    else None,
                    "stability_flag_size_range_gt_30pct": bool(size_range_norm > 0.30),
                }
            )
    C.write_csv(pd.DataFrame(stab_rows), C.OUT_DIR / "cluster_stability.csv")

    # Feature-perturbation ARI — k-means only across all K (Amendment doc text says
    # "15 features × 7 K × 3 algorithms — large output, OK" — we report k-means
    # and hierarchical here; HDBSCAN perturbation is computationally costly and
    # is omitted with documentation).
    log("PHASE B: feature-perturbation ARI")
    pert_rows = []
    base_feats = list(C.CLUSTER_FEATURES_NUMERIC)
    for drop_feat in base_feats:
        keep = [f for f in base_feats if f != drop_feat]
        num = df[keep].astype(float)
        num = num.fillna(num.median(numeric_only=True))
        mu = num.mean()
        sd = num.std(ddof=0).replace(0, 1.0)
        num_z = ((num - mu) / sd).values.astype(np.float64)
        cat_frames = []
        for cc in C.CLUSTER_FEATURES_CATEGORICAL:
            cat_frames.append(D._onehot(df[cc], cc).values.astype(np.float64))
        X_drop = np.concatenate([num_z] + cat_frames, axis=1)
        for k in C.K_VALUES:
            km_drop = CL.fit_kmeans(X_drop, k, seed=C.BASE_SEED + k)
            hi_drop = CL.fit_hierarchical_ward(X_drop, k, seed=C.BASE_SEED + k)
            pert_rows.append(
                {
                    "dropped_feature": drop_feat,
                    "K": k,
                    "ari_kmeans_vs_baseline": CL.ari(
                        cluster_assignments[f"K{k}_kmeans"].values, km_drop
                    ),
                    "ari_hierarchical_vs_baseline": CL.ari(
                        cluster_assignments[f"K{k}_hierarchical"].values, hi_drop
                    ),
                }
            )
    C.write_csv(pd.DataFrame(pert_rows), C.OUT_DIR / "feature_perturbation_ARI.csv")

    # =================================================================
    # PHASE C — Signal-time predictor scan (Amendments 9, 11)
    # =================================================================
    log("PHASE C: signal-time predictor scan (all targets, 4 models)")
    X_st, st_names, st_meta = D.build_signal_time_matrix(df, return_meta=True)
    log(f"  signal-time feature matrix: {X_st.shape}")
    C.write_csv(st_meta, C.OUT_DIR / "predictor_feature_set.csv")

    pred_rows_C = []
    importance_rows_C = []
    # All targets: (algo, K, target_cid)
    target_specs = []
    for (algo, k), cid in targets_by_algo_k.items():
        col = (
            "hdbscan_cluster_id"
            if algo == "hdbscan"
            else f"K{k}_kmeans"
            if algo == "kmeans"
            else f"K{k}_hierarchical"
        )
        target_specs.append((algo, k, cid, col))

    for algo, k, target_cid, col in target_specs:
        lab = cluster_assignments[col].values
        y = (lab == target_cid).astype(int)
        if y.sum() < 50 or y.sum() > n - 50:
            log(f"  skip {algo} K={k} target_cid={target_cid} (n_pos={y.sum()})")
            continue
        for mdl in ("logreg", "tree", "rf", "gb"):
            t0 = time.time()
            res = P.run_model_perfold(
                mdl, X_st, y, fold_id, seed=C.BASE_SEED + k * 100 + target_cid * 10
            )
            log(
                f"  3c {algo} K={k} cid={target_cid} {mdl}: "
                f"pooled_auc={res['pooled_auc']:.3f}, "
                f"partial_auc_d10={res['partial_auc_worst_decile']:.3f} ({time.time() - t0:.1f}s)"
            )
            pred_rows_C.append(
                {
                    "algo": algo,
                    "K": k,
                    "target_cluster_id": int(target_cid),
                    "model": mdl,
                    "n_positive": int(y.sum()),
                    "n_total": int(n),
                    "pooled_auc": res["pooled_auc"],
                    "partial_auc_worst_decile": res["partial_auc_worst_decile"],
                    **{
                        f"auc_fold{f}": res["perfold_auc"].get(f, float("nan")) for f in range(1, 8)
                    },
                    "perfold_auc_mean": float(np.nanmean(list(res["perfold_auc"].values()))),
                    "perfold_auc_std": (
                        float(np.nanstd(list(res["perfold_auc"].values()), ddof=1))
                        if len([v for v in res["perfold_auc"].values() if not np.isnan(v)]) > 1
                        else float("nan")
                    ),
                    "is_cv5_mean_auc_folds1to5": res["is_cv_mean"],
                    "is_cv5_std_auc_folds1to5": res["is_cv_std"],
                    "brier": res["calibration"]["brier"],
                    "logreg_C": res["model_meta"].get("C", float("nan"))
                    if mdl == "logreg"
                    else float("nan"),
                }
            )
            imp = res["importance"]
            top_idx = np.argsort(imp)[::-1][:20]
            for rnk, fi in enumerate(top_idx, start=1):
                importance_rows_C.append(
                    {
                        "algo": algo,
                        "K": k,
                        "target_cluster_id": int(target_cid),
                        "model": mdl,
                        "rank": rnk,
                        "feature": st_names[fi],
                        "importance": float(imp[fi]),
                    }
                )
    C.write_csv(pd.DataFrame(pred_rows_C), C.OUT_DIR / "predictor_AUC_by_cluster.csv")
    imp_df_C = pd.DataFrame(importance_rows_C)
    C.write_csv(imp_df_C, C.OUT_DIR / "feature_importance_3c.csv")

    # Multicollinearity on union of top-20 per target
    log("PHASE C: multicollinearity top-20")
    mc_rows = []
    for algo, k, target_cid, col in target_specs:
        sub = imp_df_C[
            (imp_df_C["algo"] == algo)
            & (imp_df_C["K"] == k)
            & (imp_df_C["target_cluster_id"] == target_cid)
        ]
        top_set = sorted(set(sub["feature"].tolist()))
        if len(top_set) < 2:
            continue
        cols = [st_names.index(f) for f in top_set]
        Xsub = X_st[:, cols]
        p_corr = np.corrcoef(Xsub, rowvar=False)
        Xrank = np.apply_along_axis(lambda x: ss.rankdata(x), 0, Xsub)
        s_corr = np.corrcoef(Xrank, rowvar=False)
        for i, fi in enumerate(top_set):
            for j, fj in enumerate(top_set):
                if j <= i:
                    continue
                mc_rows.append(
                    {
                        "algo": algo,
                        "K": k,
                        "target_cluster_id": int(target_cid),
                        "feature_a": fi,
                        "feature_b": fj,
                        "pearson": float(p_corr[i, j]),
                        "spearman": float(s_corr[i, j]),
                    }
                )
    C.write_csv(pd.DataFrame(mc_rows), C.OUT_DIR / "multicollinearity_top20.csv")

    # =================================================================
    # PHASE F — Look-elsewhere haircut signal-time (Amendments 1, 11)
    # =================================================================
    log("PHASE F: BH haircut (signal-time, hashlib seeds, 1000 perms)")
    # Raw features for haircut: numerics direct + categoricals rate-encoded
    raw_features = list(C.SIGNAL_TIME_NUMERIC) + list(C.SIGNAL_TIME_CATEGORICAL)
    haircut_rows_signal = []
    for algo, k, target_cid, col in target_specs:
        lab = cluster_assignments[col].values
        y = (lab == target_cid).astype(int)
        if y.sum() < 50 or y.sum() > n - 50:
            continue
        t0 = time.time()
        aucs, pvals = [], []
        for fname in raw_features:
            if fname in C.SIGNAL_TIME_CATEGORICAL:
                s_cat = df[fname].astype(str)
                feat_vals = pd.Series(y).groupby(s_cat).transform("mean").values.astype(float)
            else:
                feat_vals = df[fname].values.astype(float)
                if np.any(np.isnan(feat_vals)):
                    med = np.nanmedian(feat_vals)
                    feat_vals = np.where(np.isnan(feat_vals), med, feat_vals)
            seed_v = C.perm_seed_for_feature(
                f"{fname}|{algo}|K{k}|cid{target_cid}|t0",
                base_offset=C.BASE_SEED,
            )
            auc, p = HC.permutation_p_value(
                feat_vals,
                y,
                fold_id,
                n_perm=C.N_PERMUTATIONS,
                seed=seed_v,
            )
            aucs.append(auc)
            pvals.append(p)
        bh = HC.bh_correct(np.array(pvals))
        tiers = [HC.tier_from_bh(b) for b in bh]
        for fname, a, p, b, tt in zip(raw_features, aucs, pvals, bh, tiers):
            haircut_rows_signal.append(
                {
                    "scan": "signal_time",
                    "t_slice": 0,
                    "algo": algo,
                    "K": k,
                    "target_cluster_id": int(target_cid),
                    "feature": fname,
                    "univariate_auc": a,
                    "raw_p": p,
                    "bh_p": float(b),
                    "tier": tt,
                    "n_features_in_scan": int(len(raw_features)),
                }
            )
        log(f"  3c haircut {algo} K={k} cid={target_cid}: ({time.time() - t0:.1f}s)")

    haircut_signal_df = pd.DataFrame(haircut_rows_signal)

    # =================================================================
    # CALIBRATION CHECK (Amendment 1, family-level)
    # =================================================================
    log("CALIBRATION CHECK (Amendment 1, family-level)")
    calib_lines = [
        "CALIBRATION CHECK v1.1 — family-level (Amendment 1)",
        "Per L_ARC_PROTOCOL_v1.1_AMENDMENT.md §1 + arc-open §4 + op spec §6.7",
        "",
    ]
    calib_status_per_target = {}
    overall_pass = True
    for algo in ("kmeans", "hierarchical"):
        for k in (2,):
            target_cid = targets_by_algo_k.get((algo, k))
            if target_cid is None:
                continue
            sub = haircut_signal_df[
                (haircut_signal_df["algo"] == algo)
                & (haircut_signal_df["K"] == k)
                & (haircut_signal_df["target_cluster_id"] == target_cid)
            ]
            family_rows = sub[sub["feature"].isin(C.CALIBRATION_FAMILY_V11)]
            cleared = family_rows[family_rows["tier"].isin(("Tier1", "Tier2"))]
            pass_flag = len(cleared) >= 1
            calib_status_per_target[f"{algo}_K{k}"] = pass_flag
            calib_lines.append(f"--- {algo} K={k} target_cid={target_cid} ---")
            calib_lines.append("Family members (cross-pair / portfolio, §5.15):")
            for _, fr in family_rows.iterrows():
                marker = "  ✓" if fr["tier"] in ("Tier1", "Tier2") else "  ·"
                calib_lines.append(
                    f"{marker} {fr['feature']}: AUC={fr['univariate_auc']:.4f}, "
                    f"BH-p={fr['bh_p']:.4g}, Tier={fr['tier']}"
                )
            calib_lines.append(
                f"PASS at this target: {pass_flag} ({len(cleared)}/8 family members cleared Tier 1 or Tier 2)"
            )
            # historical carrier reporting
            hist = sub[sub["feature"] == C.CALIBRATION_FAMILY_HISTORICAL_CARRIER]
            if not hist.empty:
                h = hist.iloc[0]
                calib_lines.append(
                    f"Historical carrier ({C.CALIBRATION_FAMILY_HISTORICAL_CARRIER}): "
                    f"AUC={h['univariate_auc']:.4f}, BH-p={h['bh_p']:.4g}, "
                    f"Tier={h['tier']} (reportorial only)"
                )
            calib_lines.append("")
            if not pass_flag:
                overall_pass = False

    calib_lines.append(
        f"OVERALL CALIBRATION STATUS (PASS = all K=2 targets PASS): "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )
    C.write_text("\n".join(calib_lines) + "\n", C.OUT_DIR / "calibration_check_v1_1.txt")
    manifest["calibration_check_v1_1_status"] = "PASS" if overall_pass else "FAIL"
    manifest["calibration_check_v1_1_per_target"] = calib_status_per_target

    if not overall_pass:
        log("CALIBRATION CHECK v1.1 FAIL — halting (no phase doc written)")
        C.write_csv(haircut_signal_df, C.OUT_DIR / "look_elsewhere_haircut.csv")
        C.append_manifest(manifest, C.OUT_DIR / "run_manifest.json")
        return 2
    log("CALIBRATION CHECK v1.1 PASS — proceeding to Phase D, E, G, H")

    # =================================================================
    # PHASE D — Held-bar predictor scan (Amendment 9)
    # =================================================================
    # Per scope decision (documented in §schema_notes): Phase D runs for K=2
    # (kmeans + hierarchical) + HDBSCAN target. Higher-K targets omitted for
    # runtime feasibility within session budget. Op spec §6.1 designates K=2
    # as the canonical binary good/bad baseline.
    log("PHASE D: held-bar predictor scan (K=2 kmeans + K=2 hierarchical + HDBSCAN)")
    phase_d_targets = []
    for algo, k_target in [("kmeans", 2), ("hierarchical", 2)]:
        cid = targets_by_algo_k.get((algo, k_target))
        if cid is not None:
            col = f"K{k_target}_kmeans" if algo == "kmeans" else f"K{k_target}_hierarchical"
            phase_d_targets.append((algo, k_target, cid, col))
    if not hdbscan_skipped_no_clusters:
        cid_hd = targets_by_algo_k.get(("hdbscan", n_hdbscan_clusters))
        if cid_hd is not None:
            phase_d_targets.append(("hdbscan", n_hdbscan_clusters, cid_hd, "hdbscan_cluster_id"))

    pred_rows_D = []
    importance_rows_D = []
    haircut_rows_held = []
    auc_thresh_by = {}  # (algo, k, cid, mdl) → {thr: earliest_t}
    pauc_thresh_by = {}

    for t in C.HELD_BAR_TS:
        log(f"  building t={t} feature matrix")
        Xt, t_names, t_meta = D.build_t_matrix(df, t)
        for algo, k, target_cid, col in phase_d_targets:
            lab = cluster_assignments[col].values
            y = (lab == target_cid).astype(int)
            if y.sum() < 50 or y.sum() > n - 50:
                continue
            for mdl in ("logreg", "tree", "rf", "gb"):
                t0 = time.time()
                res = P.run_model_perfold(
                    mdl, Xt, y, fold_id, seed=C.BASE_SEED + k * 100 + target_cid * 10 + t
                )
                log(
                    f"  3d t={t} {algo} K={k} cid={target_cid} {mdl}: "
                    f"pooled_auc={res['pooled_auc']:.3f}, partial={res['partial_auc_worst_decile']:.3f} "
                    f"({time.time() - t0:.1f}s)"
                )
                pred_rows_D.append(
                    {
                        "t": t,
                        "algo": algo,
                        "K": k,
                        "target_cluster_id": int(target_cid),
                        "model": mdl,
                        "n_positive": int(y.sum()),
                        "pooled_auc": res["pooled_auc"],
                        "partial_auc_worst_decile": res["partial_auc_worst_decile"],
                        **{
                            f"auc_fold{f}": res["perfold_auc"].get(f, float("nan"))
                            for f in range(1, 8)
                        },
                        "perfold_auc_mean": float(np.nanmean(list(res["perfold_auc"].values()))),
                        "perfold_auc_std": (
                            float(np.nanstd(list(res["perfold_auc"].values()), ddof=1))
                            if len([v for v in res["perfold_auc"].values() if not np.isnan(v)]) > 1
                            else float("nan")
                        ),
                        "brier": res["calibration"]["brier"],
                    }
                )
                # threshold crossings
                key = (algo, k, int(target_cid), mdl)
                auc_thresh_by.setdefault(key, {0.60: None, 0.65: None, 0.70: None, 0.75: None})
                pauc_thresh_by.setdefault(key, {0.60: None, 0.65: None, 0.70: None, 0.75: None})
                pa = res["pooled_auc"]
                for thr in (0.60, 0.65, 0.70, 0.75):
                    if auc_thresh_by[key][thr] is None and not np.isnan(pa) and pa >= thr:
                        auc_thresh_by[key][thr] = t
                pb = res["partial_auc_worst_decile"]
                for thr in (0.60, 0.65, 0.70, 0.75):
                    if pauc_thresh_by[key][thr] is None and not np.isnan(pb) and pb >= thr:
                        pauc_thresh_by[key][thr] = t
                imp = res["importance"]
                top_idx = np.argsort(imp)[::-1][:20]
                for rnk, fi in enumerate(top_idx, start=1):
                    importance_rows_D.append(
                        {
                            "t": t,
                            "algo": algo,
                            "K": k,
                            "target_cluster_id": int(target_cid),
                            "model": mdl,
                            "rank": rnk,
                            "feature": t_names[fi],
                            "importance": float(imp[fi]),
                        }
                    )

        # Phase F haircut at this t
        log(f"  3d haircut at t={t}")
        hc_features_t = (
            list(C.SIGNAL_TIME_NUMERIC)
            + list(C.SIGNAL_TIME_CATEGORICAL)
            + ["first_bar_direction", "first_bar_range_atr", "first_bar_range_bin"]
            + [f"{fname}_t{t}" for fname in C.FWD_CTX_NUMERIC]
        )
        t_ctx = D.load_t_features(t).set_index("trade_id").reindex(df["trade_id"].values)
        for algo, k, target_cid, col in phase_d_targets:
            lab = cluster_assignments[col].values
            y = (lab == target_cid).astype(int)
            if y.sum() < 50 or y.sum() > n - 50:
                continue
            aucs, pvals = [], []
            for fname in hc_features_t:
                if fname in C.SIGNAL_TIME_NUMERIC:
                    feat_vals = df[fname].values.astype(float)
                elif fname in C.SIGNAL_TIME_CATEGORICAL or fname in (
                    "first_bar_direction",
                    "first_bar_range_bin",
                ):
                    s_cat = df[fname].astype(str)
                    feat_vals = pd.Series(y).groupby(s_cat).transform("mean").values.astype(float)
                elif fname == "first_bar_range_atr":
                    feat_vals = df["first_bar_range_atr"].values.astype(float)
                elif fname.endswith(f"_t{t}"):
                    base = fname[: -len(f"_t{t}")]
                    feat_vals = t_ctx[base].values.astype(float)
                else:
                    feat_vals = df[fname].values.astype(float)
                if np.any(np.isnan(feat_vals)):
                    med = np.nanmedian(feat_vals)
                    feat_vals = np.where(np.isnan(feat_vals), med, feat_vals)
                seed_v = C.perm_seed_for_feature(
                    f"{fname}|{algo}|K{k}|cid{target_cid}|t{t}",
                    base_offset=C.BASE_SEED,
                )
                auc, p = HC.permutation_p_value(
                    feat_vals,
                    y,
                    fold_id,
                    n_perm=C.N_PERMUTATIONS,
                    seed=seed_v,
                )
                aucs.append(auc)
                pvals.append(p)
            bh = HC.bh_correct(np.array(pvals))
            tiers = [HC.tier_from_bh(b) for b in bh]
            for fname, a, p, b, tt in zip(hc_features_t, aucs, pvals, bh, tiers):
                haircut_rows_held.append(
                    {
                        "scan": "held_bar",
                        "t_slice": t,
                        "algo": algo,
                        "K": k,
                        "target_cluster_id": int(target_cid),
                        "feature": fname,
                        "univariate_auc": a,
                        "raw_p": p,
                        "bh_p": float(b),
                        "tier": tt,
                        "n_features_in_scan": int(len(hc_features_t)),
                    }
                )

    C.write_csv(pd.DataFrame(pred_rows_D), C.OUT_DIR / "predictor_AUC_by_cluster_by_t.csv")
    C.write_csv(pd.DataFrame(importance_rows_D), C.OUT_DIR / "feature_importance_3d.csv")

    auc_th_rows = []
    for (algo, k, cid, mdl), thrs in sorted(auc_thresh_by.items()):
        pthrs = pauc_thresh_by.get((algo, k, cid, mdl), {})
        auc_th_rows.append(
            {
                "algo": algo,
                "K": k,
                "target_cluster_id": int(cid),
                "model": mdl,
                "earliest_t_pooled_auc_ge_0.60": thrs[0.60] if thrs[0.60] is not None else -1,
                "earliest_t_pooled_auc_ge_0.65": thrs[0.65] if thrs[0.65] is not None else -1,
                "earliest_t_pooled_auc_ge_0.70": thrs[0.70] if thrs[0.70] is not None else -1,
                "earliest_t_pooled_auc_ge_0.75": thrs[0.75] if thrs[0.75] is not None else -1,
                "earliest_t_partial_auc_ge_0.60": pthrs.get(0.60, -1)
                if pthrs.get(0.60) is not None
                else -1,
                "earliest_t_partial_auc_ge_0.65": pthrs.get(0.65, -1)
                if pthrs.get(0.65) is not None
                else -1,
                "earliest_t_partial_auc_ge_0.70": pthrs.get(0.70, -1)
                if pthrs.get(0.70) is not None
                else -1,
                "earliest_t_partial_auc_ge_0.75": pthrs.get(0.75, -1)
                if pthrs.get(0.75) is not None
                else -1,
            }
        )
    C.write_csv(pd.DataFrame(auc_th_rows), C.OUT_DIR / "auc_threshold_crossings_3d.csv")

    haircut_combined = pd.concat(
        [haircut_signal_df, pd.DataFrame(haircut_rows_held)], ignore_index=True
    )
    unique_features = haircut_combined["feature"].nunique() if len(haircut_combined) else 0
    haircut_combined["n_features_scanned_total"] = int(unique_features)
    C.write_csv(haircut_combined, C.OUT_DIR / "look_elsewhere_haircut.csv")
    manifest["n_features_scanned_total"] = int(unique_features)

    # =================================================================
    # Phase G/H eligible features (Amendment 2: top-K by raw AUC per target)
    # =================================================================
    log("Computing Phase G/H eligibility (Amendment 2: top-K by raw AUC per target)")
    gh_eligible_rows = []
    for algo, k, target_cid, _ in target_specs:
        sub = haircut_signal_df[
            (haircut_signal_df["algo"] == algo)
            & (haircut_signal_df["K"] == k)
            & (haircut_signal_df["target_cluster_id"] == target_cid)
        ].copy()
        if sub.empty:
            continue
        sub["abs_auc_dev"] = (sub["univariate_auc"] - 0.5).abs()
        top_k = C.phase_gh_top_k(int(sub["n_features_in_scan"].iloc[0]))
        top_sub = sub.sort_values("abs_auc_dev", ascending=False).head(top_k)
        for _, r in top_sub.iterrows():
            gh_eligible_rows.append(
                {
                    "algo": algo,
                    "K": k,
                    "target_cluster_id": int(target_cid),
                    "feature": r["feature"],
                    "univariate_auc": float(r["univariate_auc"]),
                    "bh_p": float(r["bh_p"]),
                    "tier": r["tier"],
                    "top_k_threshold": int(top_k),
                }
            )
    C.write_csv(pd.DataFrame(gh_eligible_rows), C.OUT_DIR / "phase_g_h_eligible_features.csv")

    # =================================================================
    # PHASE E — Stratifications
    # =================================================================
    log("PHASE E: stratifications across axes")
    primary_lab = cluster_assignments["K2_kmeans"].values
    strat_axes = {
        "pair": df["pair"].astype(str),
        "fold": df["fold_id"].astype(str),
        "exit_reason": df["exit_reason"].astype(str),
        "session": df["session"].astype(str),
        "day_of_week": df["day_of_week"].astype(str),
        "hour_of_day": df["hour_utc"].astype(str),
        "hour_in_4h_bar": df["hour_in_4h_bar"].astype(str),
        "hour_in_d1_bar": df["hour_in_d1_bar"].astype(str),
        "pre_momentum_bin": df["pre_momentum_bin"].astype(str),
        "trigger_magnitude_decile": df["trigger_magnitude_decile"].astype(str),
        "vol_regime": df["vol_regime"].astype(str),
    }
    for axis_name, axis_series in strat_axes.items():
        axis_dir = C.STRAT_DIR / axis_name
        axis_dir.mkdir(parents=True, exist_ok=True)
        mix_rows, geom_rows = [], []
        for level in sorted(axis_series.unique().tolist()):
            mask = (axis_series == level).values
            n_lvl = int(mask.sum())
            if n_lvl < 10:
                continue  # pool into _insufficient_n via the residual flag
            c0_frac = float((primary_lab[mask] == 0).mean())
            c1_frac = float((primary_lab[mask] == 1).mean())
            mix_rows.append(
                {
                    "level": level,
                    "n": n_lvl,
                    "cluster_0_frac_K2_kmeans": c0_frac,
                    "cluster_1_frac_K2_kmeans": c1_frac,
                    "flagged_lt_30": bool(n_lvl < 30),
                }
            )
            sub = df[mask]
            for cid in (0, 1):
                cmask = primary_lab[mask] == cid
                if cmask.sum() < 10:
                    continue
                sub_c = sub[cmask]
                geom_rows.append(
                    {
                        "level": level,
                        "cluster_id": cid,
                        "n": int(cmask.sum()),
                        "fwd_mfe_h24_p50": float(sub_c["fwd_mfe_h24_atr"].median()),
                        "fwd_mfe_h24_p75": float(sub_c["fwd_mfe_h24_atr"].quantile(0.75)),
                        "fwd_mfe_h24_p90": float(sub_c["fwd_mfe_h24_atr"].quantile(0.90)),
                        "fwd_mfe_h24_p95": float(sub_c["fwd_mfe_h24_atr"].quantile(0.95)),
                        "fwd_mae_h24_p50": float(sub_c["fwd_mae_h24_atr"].median()),
                        "fwd_mae_h24_p75": float(sub_c["fwd_mae_h24_atr"].quantile(0.75)),
                        "fwd_mae_h24_p90": float(sub_c["fwd_mae_h24_atr"].quantile(0.90)),
                        "race_p50": float(sub_c["race_bars_plus1_minus_minus1"].median()),
                        "fwd_mfe_to_mae_ratio_h24_p50": float(
                            sub_c["fwd_mfe_to_mae_ratio_h24"].median()
                        ),
                    }
                )
        C.write_csv(pd.DataFrame(mix_rows), axis_dir / "cluster_mix.csv")
        C.write_csv(pd.DataFrame(geom_rows), axis_dir / "conditional_forward_geometry.csv")

    # =================================================================
    # PHASE G — Cluster effect sizes
    # =================================================================
    log("PHASE G: cluster effect sizes per §8")
    pool_mfe24_p50 = float(df["fwd_mfe_h24_atr"].median())
    pool_mfe24_std = float(df["fwd_mfe_h24_atr"].std())
    pool_ratio_p50 = float(df["fwd_mfe_to_mae_ratio_h24"].median())
    pool_race_p50 = float(df["race_bars_plus1_minus_minus1"].median())
    pool_p1atr_240 = float(df["reached_plus_1.0_atr_within_240"].mean())
    pool_mfe120_p50 = float(df["fwd_mfe_h120_atr"].median())

    es_rows = []
    for algo, col_pattern in (
        ("kmeans", "K{}_kmeans"),
        ("hierarchical", "K{}_hierarchical"),
    ):
        for k in C.K_VALUES:
            col = col_pattern.format(k)
            lab = cluster_assignments[col].values
            for cid in sorted(set(lab.tolist())):
                mask = lab == cid
                sub = df.loc[mask]
                _emit_effect_size_row(
                    es_rows,
                    algo,
                    k,
                    int(cid),
                    mask,
                    sub,
                    pool_mfe24_p50,
                    pool_mfe24_std,
                    pool_mfe120_p50,
                    pool_ratio_p50,
                    pool_race_p50,
                    pool_p1atr_240,
                )
    # HDBSCAN
    for cid in sorted(set(hd.tolist())):
        mask = hd == cid
        sub = df.loc[mask]
        _emit_effect_size_row(
            es_rows,
            "hdbscan",
            n_hdbscan_clusters,
            int(cid),
            mask,
            sub,
            pool_mfe24_p50,
            pool_mfe24_std,
            pool_mfe120_p50,
            pool_ratio_p50,
            pool_race_p50,
            pool_p1atr_240,
        )
    C.write_csv(pd.DataFrame(es_rows), C.OUT_DIR / "cluster_effect_sizes.csv")

    # Summary table: pass count per K
    es_df = pd.DataFrame(es_rows)
    summary_es = (
        es_df.groupby(["algo", "K"])["pass_any_effect_size_metric"]
        .agg(["sum", "count"])
        .reset_index()
    )
    summary_es.columns = ["algo", "K", "n_clusters_pass_es", "n_clusters_total"]
    C.write_csv(summary_es, C.OUT_DIR / "cluster_effect_sizes_summary.csv")

    # =================================================================
    # PHASE H — Filter dry-run (Amendments 2, 10)
    # =================================================================
    log("PHASE H: filter dry-run with expected R-volume ranking")
    # Use phase_g_h_eligible_features as input (Amendment 2)
    gh_df = pd.DataFrame(gh_eligible_rows)
    # Primary axis: K=2 kmeans target (the most-natural extractable/non-extractable axis)
    target_cid_km_K2 = targets_by_algo_k.get(("kmeans", 2))
    extr_cid = 1 - target_cid_km_K2  # mirror (good) cluster
    p_extr_full = float((cluster_assignments["K2_kmeans"].values == extr_cid).mean())
    p_non_extr_full = float((cluster_assignments["K2_kmeans"].values == target_cid_km_K2).mean())

    eligible_for_h = gh_df[
        (gh_df["algo"] == "kmeans")
        & (gh_df["K"] == 2)
        & (gh_df["target_cluster_id"] == target_cid_km_K2)
    ]["feature"].tolist()

    dry_rows = []
    for fname in eligible_for_h:
        # find tier from haircut
        tier_row = haircut_signal_df[
            (haircut_signal_df["algo"] == "kmeans")
            & (haircut_signal_df["K"] == 2)
            & (haircut_signal_df["target_cluster_id"] == target_cid_km_K2)
            & (haircut_signal_df["feature"] == fname)
        ]
        bh_tier = tier_row["tier"].iloc[0] if not tier_row.empty else "absent"
        if fname in C.SIGNAL_TIME_NUMERIC:
            col = df[fname].astype(float).values
            for pct in (25, 50, 75):
                thr = float(np.percentile(col, pct))
                for direction in ("above", "below"):
                    mask = (col > thr) if direction == "above" else (col <= thr)
                    _emit_filter_row(
                        dry_rows,
                        fname,
                        "numeric",
                        pct,
                        thr,
                        direction,
                        mask,
                        df,
                        fold_id,
                        cluster_assignments["K2_kmeans"].values,
                        extr_cid,
                        target_cid_km_K2,
                        p_extr_full,
                        p_non_extr_full,
                        pool_mean_r,
                        bh_tier,
                    )
        else:
            s_cat = df[fname].astype(str)
            for level in sorted(s_cat.unique().tolist()):
                mask = (s_cat == level).values
                if mask.sum() == 0:
                    continue
                _emit_filter_row(
                    dry_rows,
                    fname,
                    "categorical",
                    float("nan"),
                    float("nan"),
                    f"level={level}",
                    mask,
                    df,
                    fold_id,
                    cluster_assignments["K2_kmeans"].values,
                    extr_cid,
                    target_cid_km_K2,
                    p_extr_full,
                    p_non_extr_full,
                    pool_mean_r,
                    bh_tier,
                )
    dry_df = pd.DataFrame(dry_rows)
    # Sort by Amendment 10: expected R-volume captured (descending)
    if not dry_df.empty:
        dry_df = dry_df.sort_values("expected_r_volume_captured", ascending=False).reset_index(
            drop=True
        )
    C.write_csv(dry_df, C.OUT_DIR / "filter_dry_run.csv")
    manifest["extractable_cluster_id_K2_kmeans"] = int(extr_cid)
    manifest["non_extractable_cluster_id_K2_kmeans"] = int(target_cid_km_K2)

    # =================================================================
    # Final manifest + output sha256
    # =================================================================
    log("manifest + output sha256 ledger")
    outputs = {}
    for p in sorted(C.OUT_DIR.rglob("*")):
        if p.is_file() and not str(p).startswith(str(C.OUT_DIR / "v1.0_archive")):
            rel = str(p.relative_to(C.OUT_DIR)).replace("\\", "/")
            outputs[rel] = C.sha256_file(p)
    manifest["outputs"] = outputs
    C.append_manifest(manifest, C.OUT_DIR / "run_manifest.json")
    log("DONE.")
    return 0


def _emit_effect_size_row(
    out_list,
    algo,
    k,
    cid,
    mask,
    sub,
    pool_mfe24_p50,
    pool_mfe24_std,
    pool_mfe120_p50,
    pool_ratio_p50,
    pool_race_p50,
    pool_p1atr_240,
):
    cluster_mfe24_p50 = float(sub["fwd_mfe_h24_atr"].median())
    cluster_mfe120_p50 = float(sub["fwd_mfe_h120_atr"].median())
    cluster_ratio_p50 = float(sub["fwd_mfe_to_mae_ratio_h24"].median())
    cluster_race_p50 = float(sub["race_bars_plus1_minus_minus1"].median())
    cluster_p1atr_240 = float(sub["reached_plus_1.0_atr_within_240"].mean())
    d_mfe24 = cluster_mfe24_p50 - pool_mfe24_p50
    d_ratio = cluster_ratio_p50 - pool_ratio_p50
    d_race = cluster_race_p50 - pool_race_p50
    d_p1atr = cluster_p1atr_240 - pool_p1atr_240
    pass_mfe24 = (
        abs(d_mfe24) >= C.ES_THRESHOLDS["delta_median_fwd_mfe_h24"]
        or abs(d_mfe24) >= C.ES_THRESHOLDS["delta_median_fwd_mfe_h24_stdfrac"] * pool_mfe24_std
    )
    pass_ratio = abs(d_ratio) >= C.ES_THRESHOLDS["delta_median_fwd_mfe_to_mae_ratio_h24"]
    pass_race = abs(d_race) >= C.ES_THRESHOLDS["delta_race_condition_median"]
    pass_p1atr = abs(d_p1atr) >= C.ES_THRESHOLDS["delta_p_reach_plus1atr_240"]
    pass_any = bool(pass_mfe24 or pass_ratio or pass_race or pass_p1atr)
    pass_dir = bool(pass_ratio or pass_race or pass_p1atr)
    brownian_only = bool(pass_mfe24 and not pass_dir)
    out_list.append(
        {
            "algo": algo,
            "K": int(k),
            "cluster_id": int(cid),
            "n": int(mask.sum()),
            "frac_of_pool": float(mask.mean()),
            "pool_mfe_h24_p50": pool_mfe24_p50,
            "cluster_mfe_h24_p50": cluster_mfe24_p50,
            "delta_median_fwd_mfe_h24": d_mfe24,
            "delta_median_fwd_mfe_h24_in_pool_stds": d_mfe24 / pool_mfe24_std
            if pool_mfe24_std > 0
            else float("nan"),
            "pass_delta_mfe_h24": pass_mfe24,
            "pool_mfe_h120_p50": pool_mfe120_p50,
            "cluster_mfe_h120_p50": cluster_mfe120_p50,
            "delta_median_fwd_mfe_h120": cluster_mfe120_p50 - pool_mfe120_p50,
            "pool_ratio_h24_p50": pool_ratio_p50,
            "cluster_ratio_h24_p50": cluster_ratio_p50,
            "delta_median_fwd_mfe_to_mae_ratio_h24": d_ratio,
            "pass_delta_ratio_h24": pass_ratio,
            "pool_race_p50": pool_race_p50,
            "cluster_race_p50": cluster_race_p50,
            "delta_race_condition_median_bars": d_race,
            "pass_delta_race": pass_race,
            "pool_p_reach_plus1atr_240": pool_p1atr_240,
            "cluster_p_reach_plus1atr_240": cluster_p1atr_240,
            "delta_p_reach_plus1atr_240": d_p1atr,
            "pass_delta_p_reach_plus1atr_240": pass_p1atr,
            "pass_any_effect_size_metric": pass_any,
            "pass_any_directional_metric": pass_dir,
            "brownian_path_characterisation_only": brownian_only,
        }
    )


def _emit_filter_row(
    out_list,
    fname,
    kind,
    pct,
    thr,
    direction,
    mask,
    df,
    fold_id,
    k2_lab,
    extr_cid,
    non_extr_cid,
    p_extr_full,
    p_non_extr_full,
    pool_mean_r,
    bh_tier,
):
    n_ret = int(mask.sum())
    if n_ret == 0:
        return
    perfold_n = [int(((fold_id == f) & mask).sum()) for f in range(1, 8)]
    perfold_mean_r = [
        float(df.loc[(fold_id == f) & mask, "net_r"].mean())
        if ((fold_id == f) & mask).sum() > 0
        else float("nan")
        for f in range(1, 8)
    ]
    pos_fold = sum(1 for v in perfold_mean_r if (not np.isnan(v)) and v > 0)
    mean_r_post = float(df.loc[mask, "net_r"].mean())
    delta_mean_r = mean_r_post - pool_mean_r
    # Amendment 10: expected R-volume captured = (mean R improvement) × n_retained
    expected_r_volume = delta_mean_r * n_ret
    p_ext_post = float((k2_lab[mask] == extr_cid).mean()) if n_ret > 0 else float("nan")
    p_nex_post = float((k2_lab[mask] == non_extr_cid).mean()) if n_ret > 0 else float("nan")
    out_list.append(
        {
            "feature": fname,
            "kind": kind,
            "threshold_pct": pct,
            "threshold_value": thr,
            "direction": direction,
            "n_retained": n_ret,
            "n_retained_min_fold": min(perfold_n),
            "mean_r_post_filter": mean_r_post,
            "delta_mean_r_vs_pool": delta_mean_r,
            "expected_r_volume_captured": expected_r_volume,
            "perfold_positive_count": pos_fold,
            "p_extractable_post": p_ext_post,
            "delta_p_extractable": p_ext_post - p_extr_full
            if not np.isnan(p_ext_post)
            else float("nan"),
            "p_non_extractable_post": p_nex_post,
            "delta_p_non_extractable": p_nex_post - p_non_extr_full
            if not np.isnan(p_nex_post)
            else float("nan"),
            "bh_tier": bh_tier,
        }
    )


if __name__ == "__main__":
    sys.exit(main())
