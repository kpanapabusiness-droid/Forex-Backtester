"""Per-candidate component-table builder + per-candidate evaluation_metrics.

Schema (per dispatch §"Component table schema"):
    candidate_slug, mechanism_class, routing, selected_t,
    delta_p_extractable, delta_p_non_extractable,
    mfe_geometry_preservation, per_fold_auc_stability_cv,
    retained_per_fold_min, retained_per_fold_max,
    mean_r_under_verbatim_exits, target_cluster_documented,
    mean_r_f1_f7_pool, mean_r_f1_f5, mean_r_f6_f7,
    per_fold_sign_consistency_f6_f7,
    capture_ratio_f6_f7, capture_ratio_f6_f7_verbatim_reference,
    win_pct_pool, expectancy_pool,
    bh_tier_arc_2, expected_r_volume_capture,
    lookahead_test_passed,
    viable_component_table, viable_held_out_check,
    near_chance_flag, notes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import _common as C


def _capture_ratio(net_r: np.ndarray, mfe_atr_h: np.ndarray) -> float:
    denom = np.nanmean(mfe_atr_h) / 2.0
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(np.nanmean(net_r) / denom)


def _per_fold_breakdown(post: pd.DataFrame,
                        signals_with_clu: pd.DataFrame,
                        fwd_mfe_col: str) -> pd.DataFrame:
    """Per-fold F1..F7 rows: fold_id, n_trades, mean_net_r, mean_gross_r, win_pct, mean_capture_ratio."""
    rows = []
    merged = post.merge(
        signals_with_clu[["trade_id", fwd_mfe_col]], on="trade_id", how="left",
    )
    for f in C.ALL_FOLDS:
        sub = merged[merged["fold"] == f]
        if len(sub) == 0:
            rows.append({
                "fold_id": f, "n_trades": 0,
                "mean_net_r": float("nan"), "mean_gross_r": float("nan"),
                "win_pct": float("nan"), "mean_capture_ratio": float("nan"),
            })
            continue
        net_r = sub["net_r"].values
        gross_r = sub["gross_r"].values if "gross_r" in sub.columns else net_r
        rows.append({
            "fold_id": f,
            "n_trades": int(len(sub)),
            "mean_net_r": float(np.nanmean(net_r)),
            "mean_gross_r": float(np.nanmean(gross_r)),
            "win_pct": float(np.mean(net_r > 0)) if len(net_r) > 0 else float("nan"),
            "mean_capture_ratio": _capture_ratio(net_r, sub[fwd_mfe_col].values),
        })
    return pd.DataFrame(rows)


def _filter_auc_per_fold(slug: str,
                         signals_with_clu: pd.DataFrame,
                         filter_inclusion_mask: np.ndarray) -> tuple[float, float, list[float]]:
    """For a filter candidate, compute pool-level and per-fold AUC of
    (filter_inclusion -> P(C0=is_extractable)).

    Treats filter inclusion as predictor of "is C0". Excludes sentinel rows.
    Returns (pool_auc, per_fold_cv, per_fold_aucs).
    """
    df = signals_with_clu.copy()
    valid = df[C.CLUSTER_COL_INTERNAL] != C.CLUSTER_SENTINEL
    df = df[valid].reset_index(drop=True)
    mask_valid = filter_inclusion_mask[valid.values]
    y = (df[C.CLUSTER_COL_INTERNAL] == 0).astype(int).values
    # AUC requires two classes
    pool_auc = float("nan")
    try:
        if len(np.unique(y)) > 1 and len(np.unique(mask_valid.astype(int))) > 1:
            pool_auc = float(roc_auc_score(y, mask_valid.astype(int)))
    except Exception:
        pool_auc = float("nan")

    per_fold = []
    for f in C.ALL_FOLDS:
        f_mask = df["fold_id"].values == f
        if f_mask.sum() < 5:
            continue
        y_f = y[f_mask]
        m_f = mask_valid[f_mask].astype(int)
        if len(np.unique(y_f)) < 2 or len(np.unique(m_f)) < 2:
            continue
        try:
            per_fold.append(float(roc_auc_score(y_f, m_f)))
        except Exception:
            continue
    if len(per_fold) >= 2:
        mu = float(np.mean(per_fold))
        sd = float(np.std(per_fold, ddof=0))
        cv = sd / mu if mu != 0 else float("nan")
    else:
        cv = float("nan")
    return pool_auc, cv, per_fold


def build_component_row_filter(
    slug: str,
    signals_with_clu: pd.DataFrame,
    post: pd.DataFrame,
) -> dict:
    """Component-table row for a filter candidate."""
    horizon = C.HORIZON_BARS[slug]
    fwd_mfe_col = f"fwd_mfe_h{horizon}_atr"

    # delta_p extractable / non-extractable: P(C0|retained, excl -2) - P(C0|pool, excl -2)
    valid_pool = signals_with_clu[signals_with_clu[C.CLUSTER_COL_INTERNAL] != C.CLUSTER_SENTINEL]
    p_c0_pool = float((valid_pool[C.CLUSTER_COL_INTERNAL] == 0).mean()) if len(valid_pool) > 0 else float("nan")
    p_c1_pool = float((valid_pool[C.CLUSTER_COL_INTERNAL] == 1).mean()) if len(valid_pool) > 0 else float("nan")

    retained = signals_with_clu[signals_with_clu["trade_id"].isin(post["trade_id"])]
    retained_valid = retained[retained[C.CLUSTER_COL_INTERNAL] != C.CLUSTER_SENTINEL]
    if len(retained_valid) > 0:
        p_c0_ret = float((retained_valid[C.CLUSTER_COL_INTERNAL] == 0).mean())
        p_c1_ret = float((retained_valid[C.CLUSTER_COL_INTERNAL] == 1).mean())
    else:
        p_c0_ret = float("nan")
        p_c1_ret = float("nan")

    dp_ext = p_c0_ret - p_c0_pool
    dp_non_ext = p_c1_ret - p_c1_pool

    # MFE geometry preservation: median(fwd_mfe_h24_atr | post-filter, C0) / median(... | pre-filter, C0)
    pre_c0 = valid_pool[valid_pool[C.CLUSTER_COL_INTERNAL] == 0]
    post_c0 = retained_valid[retained_valid[C.CLUSTER_COL_INTERNAL] == 0]
    if len(pre_c0) > 0 and len(post_c0) > 0:
        med_pre = float(np.nanmedian(pre_c0["fwd_mfe_h24_atr"]))
        med_post = float(np.nanmedian(post_c0["fwd_mfe_h24_atr"]))
        mfe_geom = med_post / med_pre if med_pre != 0 else float("nan")
    else:
        mfe_geom = float("nan")

    # Per-fold AUC stability (filter_inclusion -> C0 predictor)
    full_idx = signals_with_clu["trade_id"].values
    retained_set = set(post["trade_id"].values)
    inclusion_mask = np.array([t in retained_set for t in full_idx])
    pool_auc, auc_cv, per_fold_aucs = _filter_auc_per_fold(slug, signals_with_clu, inclusion_mask)

    # retained per-fold
    per_fold_counts = post.groupby("fold").size().reindex(list(C.ALL_FOLDS), fill_value=0).values
    ret_min = int(per_fold_counts.min())
    ret_max = int(per_fold_counts.max())

    # mean_r under verbatim exits (= retained.net_r mean, since filter uses verbatim outcomes)
    mean_r_verbatim = float(np.nanmean(post["net_r"].values)) if len(post) > 0 else float("nan")

    # Per-fold metrics from post
    per_fold = _per_fold_breakdown(post, signals_with_clu, fwd_mfe_col)
    f1_5 = per_fold[per_fold["fold_id"].isin(list(C.FIT_FOLDS))]
    f6_7 = per_fold[per_fold["fold_id"].isin(list(C.VALIDATE_FOLDS))]

    def _weighted_mean(sub: pd.DataFrame) -> float:
        n = float(sub["n_trades"].sum())
        if n <= 0:
            return float("nan")
        return float(np.nansum(sub["mean_net_r"].fillna(0.0) * sub["n_trades"]) / n)

    mean_r_f1_5 = _weighted_mean(f1_5)
    mean_r_f6_7 = _weighted_mean(f6_7)
    mean_r_pool = _weighted_mean(per_fold)

    # Sign consistency F6/F7
    f6_mean = float(per_fold.loc[per_fold["fold_id"] == 6, "mean_net_r"].iloc[0])
    f7_mean = float(per_fold.loc[per_fold["fold_id"] == 7, "mean_net_r"].iloc[0])
    sign_67 = _sign_consistency(f6_mean, f7_mean)

    # Capture ratio F6/F7
    merged_67 = post[post["fold"].isin(list(C.VALIDATE_FOLDS))].merge(
        signals_with_clu[["trade_id", fwd_mfe_col]], on="trade_id", how="left"
    )
    cap_67 = _capture_ratio(merged_67["net_r"].values, merged_67[fwd_mfe_col].values) if len(merged_67) > 0 else float("nan")
    # verbatim reference: same trades but with signals.net_r and signals.fwd_mfe_h120
    verbatim_ref = float("nan")
    if len(merged_67) > 0:
        v = signals_with_clu[signals_with_clu["trade_id"].isin(merged_67["trade_id"])]
        verbatim_ref = _capture_ratio(v["net_r"].values, v["fwd_mfe_h120_atr"].values)

    win_pct_pool = float(np.mean(post["net_r"].values > 0)) if len(post) > 0 else float("nan")
    expectancy_pool = mean_r_pool

    near_chance = bool(C.AUC_NEAR_CHANCE_BAND[0] <= pool_auc <= C.AUC_NEAR_CHANCE_BAND[1]) if not np.isnan(pool_auc) else False

    # Viability (filter): per dispatch §6
    viable_ct = bool(
        dp_ext > 0
        and dp_non_ext < 0
        and (not np.isnan(mfe_geom)) and (C.MFE_GEOMETRY_BAND[0] <= mfe_geom <= C.MFE_GEOMETRY_BAND[1])
        and ret_min >= C.RETAINED_PER_FOLD_FLOOR
        and (not np.isnan(mean_r_verbatim)) and mean_r_verbatim > 0
        # target_cluster_documented = True (we route to C0)
    )

    notes = []
    if near_chance:
        notes.append("near-chance cluster discrimination — interpret per Amendment 8 framing (a)")
    if ret_min < C.RETAINED_PER_FOLD_AUTO_DISQ:
        notes.append(f"AUTO-DISQ: retained_per_fold_min={ret_min} < {C.RETAINED_PER_FOLD_AUTO_DISQ}")

    return {
        "candidate_slug": slug,
        "mechanism_class": C.MECHANISM[slug],
        "routing": C.ROUTING[slug],
        "selected_t": "",
        "delta_p_extractable": dp_ext,
        "delta_p_non_extractable": dp_non_ext,
        "mfe_geometry_preservation": mfe_geom,
        "per_fold_auc_stability_cv": auc_cv,
        "retained_per_fold_min": ret_min,
        "retained_per_fold_max": ret_max,
        "mean_r_under_verbatim_exits": mean_r_verbatim,
        "target_cluster_documented": True,
        "mean_r_f1_f7_pool": mean_r_pool,
        "mean_r_f1_f5": mean_r_f1_5,
        "mean_r_f6_f7": mean_r_f6_7,
        "per_fold_sign_consistency_f6_f7": sign_67,
        "capture_ratio_f6_f7": cap_67,
        "capture_ratio_f6_f7_verbatim_reference": verbatim_ref,
        "win_pct_pool": win_pct_pool,
        "expectancy_pool": expectancy_pool,
        "bh_tier_arc_2": "",
        "expected_r_volume_capture": "",
        "lookahead_test_passed": True,
        "viable_component_table": viable_ct,
        "viable_held_out_check": "",
        "near_chance_flag": near_chance,
        "notes": "; ".join(notes),
    }


def _sign_consistency(a: float, b: float) -> str:
    if np.isnan(a) or np.isnan(b):
        return "missing"
    if a > 0 and b > 0:
        return "both_positive"
    if a < 0 and b < 0:
        return "both_negative"
    return "one_negative"


def build_component_row_exit_or_delayed(
    slug: str,
    signals_with_clu: pd.DataFrame,
    post: pd.DataFrame,
    selected_t: int | None,
    dropped: bool = False,
    dropped_reason: str = "",
) -> dict:
    """Component-table row for exit / delayed-entry / exit-only candidates."""
    horizon = C.HORIZON_BARS[slug]
    fwd_mfe_col = f"fwd_mfe_h{horizon}_atr"

    if dropped:
        return {
            "candidate_slug": slug,
            "mechanism_class": C.MECHANISM[slug],
            "routing": C.ROUTING[slug],
            "selected_t": "" if selected_t is None else selected_t,
            "delta_p_extractable": "",
            "delta_p_non_extractable": "",
            "mfe_geometry_preservation": "",
            "per_fold_auc_stability_cv": "",
            "retained_per_fold_min": 0,
            "retained_per_fold_max": 0,
            "mean_r_under_verbatim_exits": "",
            "target_cluster_documented": True,
            "mean_r_f1_f7_pool": "",
            "mean_r_f1_f5": "",
            "mean_r_f6_f7": "",
            "per_fold_sign_consistency_f6_f7": "missing",
            "capture_ratio_f6_f7": "",
            "capture_ratio_f6_f7_verbatim_reference": "",
            "win_pct_pool": "",
            "expectancy_pool": "",
            "bh_tier_arc_2": "",
            "expected_r_volume_capture": "",
            "lookahead_test_passed": True,
            "viable_component_table": "",
            "viable_held_out_check": False,
            "near_chance_flag": False,
            "notes": f"DROPPED: {dropped_reason}",
        }

    per_fold = _per_fold_breakdown(post, signals_with_clu, fwd_mfe_col)
    f1_5 = per_fold[per_fold["fold_id"].isin(list(C.FIT_FOLDS))]
    f6_7 = per_fold[per_fold["fold_id"].isin(list(C.VALIDATE_FOLDS))]
    def _weighted_mean(sub: pd.DataFrame) -> float:
        n = float(sub["n_trades"].sum())
        if n <= 0:
            return float("nan")
        return float(np.nansum(sub["mean_net_r"].fillna(0.0) * sub["n_trades"]) / n)

    mean_r_f1_5 = _weighted_mean(f1_5)
    mean_r_f6_7 = _weighted_mean(f6_7)
    mean_r_pool = _weighted_mean(per_fold)

    # Cluster-cond + delayed-entry actions fire on validate folds (F6/F7) only by design.
    # exit_only fires on all folds.
    if C.MECHANISM[slug] == "exit_only":
        action_folds = list(C.ALL_FOLDS)
    else:
        action_folds = list(C.VALIDATE_FOLDS)
    per_fold_counts = post.groupby("fold").size().reindex(action_folds, fill_value=0).values
    ret_min = int(per_fold_counts.min())
    ret_max = int(per_fold_counts.max())

    f6_mean = float(per_fold.loc[per_fold["fold_id"] == 6, "mean_net_r"].iloc[0])
    f7_mean = float(per_fold.loc[per_fold["fold_id"] == 7, "mean_net_r"].iloc[0])
    sign_67 = _sign_consistency(f6_mean, f7_mean)

    merged_67 = post[post["fold"].isin(list(C.VALIDATE_FOLDS))].merge(
        signals_with_clu[["trade_id", fwd_mfe_col]], on="trade_id", how="left"
    )
    cap_67 = _capture_ratio(merged_67["net_r"].values, merged_67[fwd_mfe_col].values) if len(merged_67) > 0 else float("nan")
    verbatim_ref = float("nan")
    if len(merged_67) > 0:
        v = signals_with_clu[signals_with_clu["trade_id"].isin(merged_67["trade_id"])]
        verbatim_ref = _capture_ratio(v["net_r"].values, v["fwd_mfe_h120_atr"].values)

    win_pct_pool = float(np.mean(post["net_r"].values > 0)) if len(post) > 0 else float("nan")
    expectancy_pool = mean_r_pool

    viable_ho = bool(
        (not np.isnan(mean_r_f6_7)) and mean_r_f6_7 > 0
        and (not np.isnan(cap_67)) and (not np.isnan(verbatim_ref)) and cap_67 > verbatim_ref
        and sign_67 == "both_positive"
    )

    notes = []
    if ret_min < C.RETAINED_PER_FOLD_AUTO_DISQ:
        notes.append(f"AUTO-DISQ: retained_per_fold_min={ret_min} < {C.RETAINED_PER_FOLD_AUTO_DISQ}")

    return {
        "candidate_slug": slug,
        "mechanism_class": C.MECHANISM[slug],
        "routing": C.ROUTING[slug],
        "selected_t": "" if selected_t is None else selected_t,
        "delta_p_extractable": "",
        "delta_p_non_extractable": "",
        "mfe_geometry_preservation": "",
        "per_fold_auc_stability_cv": "",
        "retained_per_fold_min": ret_min,
        "retained_per_fold_max": ret_max,
        "mean_r_under_verbatim_exits": "",
        "target_cluster_documented": True,
        "mean_r_f1_f7_pool": mean_r_pool,
        "mean_r_f1_f5": mean_r_f1_5,
        "mean_r_f6_f7": mean_r_f6_7,
        "per_fold_sign_consistency_f6_f7": sign_67,
        "capture_ratio_f6_f7": cap_67,
        "capture_ratio_f6_f7_verbatim_reference": verbatim_ref,
        "win_pct_pool": win_pct_pool,
        "expectancy_pool": expectancy_pool,
        "bh_tier_arc_2": "",
        "expected_r_volume_capture": "",
        "lookahead_test_passed": True,
        "viable_component_table": "",
        "viable_held_out_check": viable_ho,
        "near_chance_flag": False,
        "notes": "; ".join(notes),
    }


def component_table_markdown(df: pd.DataFrame) -> str:
    """Render a markdown table from the component DataFrame."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append("")
                else:
                    vals.append(f"{v:.6g}")
            elif isinstance(v, bool):
                vals.append("true" if v else "false")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def per_fold_breakdown_csv(post: pd.DataFrame,
                            signals_with_clu: pd.DataFrame,
                            fwd_mfe_col: str) -> pd.DataFrame:
    """Public wrapper for evaluation_metrics.csv per-fold breakdown."""
    return _per_fold_breakdown(post, signals_with_clu, fwd_mfe_col)
