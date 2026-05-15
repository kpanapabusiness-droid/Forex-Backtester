"""t-selection for candidates 1, 2, 3.

For each t in T_SWEEP:
  - Fit predictor with LOFO within F1..F5 (out-of-fold within fit folds).
  - Run candidate's action policy on F1..F5 predictions; compute mean capture_ratio.
  - Also report F6/F7 mean capture_ratio and mean_r under that t.
Select t* = argmax mean_capture_ratio_f1_f5.

Output schema (per dispatch):
    t, mean_capture_ratio_f1_f5, mean_capture_ratio_f6_f7,
    mean_r_f1_f5, mean_r_f6_f7, fold_6_mean_r, fold_7_mean_r,
    n_trades_active_at_t_f1_f5, n_trades_active_at_t_f6_f7,
    frac_cluster_1_already_exited_at_t, selected
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _actions as A
from . import _common as C
from . import _data as D
from . import _predictor as P


def _capture_ratio(net_r: np.ndarray, mfe_atr_h: np.ndarray) -> float:
    """Mean (net_r) / mean(fwd_mfe_h_atr / 2.0). Pool-level capture ratio."""
    denom = np.nanmean(mfe_atr_h) / 2.0
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(np.nanmean(net_r) / denom)


def _per_fold_cap_and_r(post: pd.DataFrame,
                        signals_with_clu: pd.DataFrame,
                        fwd_mfe_col: str) -> dict:
    """Compute fold-aggregated mean_r and mean capture for a post-mechanism frame.

    Returns dict with keys mean_r, capture_ratio per fold subset.
    """
    if post.empty:
        return {"mean_r": float("nan"), "capture_ratio": float("nan"), "n": 0}
    merged = post.merge(
        signals_with_clu[["trade_id", fwd_mfe_col]],
        on="trade_id", how="left",
    )
    return {
        "mean_r": float(np.nanmean(merged["net_r"].values)),
        "capture_ratio": _capture_ratio(merged["net_r"].values, merged[fwd_mfe_col].values),
        "n": int(len(merged)),
    }


def run_t_selection_for_candidate(
    slug: str,
    signals_with_clu: pd.DataFrame,
    paths: pd.DataFrame,
    tautology_df: pd.DataFrame,
) -> pd.DataFrame:
    """For each t in T_SWEEP: fit predictor, run action, compute metrics.

    Returns DataFrame (one row per t). Selects t* = argmax mean_cap_ratio_f1_f5.
    """
    rows = []

    fwd_mfe_col = "fwd_mfe_h120_atr" if C.HORIZON_BARS[slug] == 120 else "fwd_mfe_h240_atr"

    for t in C.T_SWEEP:
        preds = P.fit_predict_cluster_all_folds(signals_with_clu, t)

        if preds.empty:
            rows.append({
                "t": t,
                "mean_capture_ratio_f1_f5": float("nan"),
                "mean_capture_ratio_f6_f7": float("nan"),
                "mean_r_f1_f5": float("nan"),
                "mean_r_f6_f7": float("nan"),
                "fold_6_mean_r": float("nan"),
                "fold_7_mean_r": float("nan"),
                "n_trades_active_at_t_f1_f5": 0,
                "n_trades_active_at_t_f6_f7": 0,
                "frac_cluster_1_already_exited_at_t": float("nan"),
                "selected": False,
            })
            continue

        # Run action separately on F1..F5 preds vs F6..F7 preds
        f1_5_preds = preds[preds["fold"].isin(list(C.FIT_FOLDS))]
        f6_7_preds = preds[preds["fold"].isin(list(C.VALIDATE_FOLDS))]

        if slug == "exit_cluster_cond_gb":
            post_15 = A.run_exit_cluster_cond(signals_with_clu, paths, f1_5_preds, t)
            post_67 = A.run_exit_cluster_cond(signals_with_clu, paths, f6_7_preds, t)
        elif slug == "exit_cluster_cond_gb_h240":
            post_15 = A.run_exit_cluster_cond_h240(signals_with_clu, paths, f1_5_preds, t)
            post_67 = A.run_exit_cluster_cond_h240(signals_with_clu, paths, f6_7_preds, t)
        elif slug == "delayed_entry_t_gb":
            held_ctx_t = D.load_held_ctx(t)
            post_15 = A.run_delayed_entry(signals_with_clu, paths, held_ctx_t, f1_5_preds, t)
            post_67 = A.run_delayed_entry(signals_with_clu, paths, held_ctx_t, f6_7_preds, t)
        else:
            raise ValueError(f"unsupported slug for t-selection: {slug}")

        m15 = _per_fold_cap_and_r(post_15, signals_with_clu, fwd_mfe_col)
        m67 = _per_fold_cap_and_r(post_67, signals_with_clu, fwd_mfe_col)

        if not post_67.empty:
            f6 = post_67[post_67["fold"] == 6]["net_r"]
            f7 = post_67[post_67["fold"] == 7]["net_r"]
            f6_r = float(np.nanmean(f6.values)) if len(f6) > 0 else float("nan")
            f7_r = float(np.nanmean(f7.values)) if len(f7) > 0 else float("nan")
        else:
            f6_r = float("nan")
            f7_r = float("nan")

        # Tautology rate
        tauto = tautology_df[(tautology_df["slug"] == slug) & (tautology_df["t"] == t)]
        frac_already = float(tauto["frac_cluster_1_already_exited"].iloc[0]) if len(tauto) > 0 else float("nan")

        rows.append({
            "t": t,
            "mean_capture_ratio_f1_f5": m15["capture_ratio"],
            "mean_capture_ratio_f6_f7": m67["capture_ratio"],
            "mean_r_f1_f5": m15["mean_r"],
            "mean_r_f6_f7": m67["mean_r"],
            "fold_6_mean_r": f6_r,
            "fold_7_mean_r": f7_r,
            "n_trades_active_at_t_f1_f5": m15["n"],
            "n_trades_active_at_t_f6_f7": m67["n"],
            "frac_cluster_1_already_exited_at_t": frac_already,
            "selected": False,
        })

    df = pd.DataFrame(rows)

    # Pick t* = argmax mean_capture_ratio_f1_f5 (ignoring NaN)
    valid = df.dropna(subset=["mean_capture_ratio_f1_f5"])
    if not valid.empty:
        i_star = valid["mean_capture_ratio_f1_f5"].astype(float).idxmax()
        t_star = int(df.loc[i_star, "t"])
        df["selected"] = df["t"] == t_star
    return df
