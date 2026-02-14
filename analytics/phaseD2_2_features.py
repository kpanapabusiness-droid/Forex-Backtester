"""
Phase D-2.2 — Feature diagnostics: causal features at t vs Zone B/C opportunity.

Pure functions for feature computation, binning, and ranking.
No ROI, PnL, backtest, or ML. Deterministic, side-effect free.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "ret_1",
    "ret_5",
    "ret_20",
    "mom_slope_5",
    "mom_slope_20",
    "atr_14",
    "atrp_14",
    "atr_z_252",
    "atr_pctile_252",
    "true_range",
    "tr_atr_ratio",
    "range_20",
    "range_20_atr",
    "bb_width_20",
    "bb_width_pctile_252",
    "pos_in_range_20",
    "dist_to_high_20_atr",
    "dist_to_low_20_atr",
    "breakout_up_20",
    "breakout_dn_20",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
]

ZONE_B = "zone_b_3r_20"
ZONE_C = "zone_c_6r_40"
MIN_POS_ZONE_B = 200
MIN_POS_ZONE_C = 100


def _safe_div(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Divide; return NaN where denom is zero or invalid."""
    denom = np.where(np.isfinite(denom) & (denom != 0), denom, np.nan)
    return np.where(np.isfinite(denom), num / denom, np.nan)


def stable_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Percentile rank of each value within its trailing window (0..1).
    Uses scipy.stats.rankdata conceptually; implemented with rolling apply.
    """
    if window < 2 or len(series) < window:
        return pd.Series(np.nan, index=series.index)

    def _pctile(x: np.ndarray) -> float:
        if len(x) < window or not np.any(np.isfinite(x)):
            return np.nan
        val = x[-1]
        if not np.isfinite(val):
            return np.nan
        finite = x[np.isfinite(x)]
        n = len(finite)
        if n == 0:
            return np.nan
        rank = np.sum(finite <= val)
        return (rank - 1) / max(n - 1, 1) if n > 1 else 0.0

    out = series.rolling(window=window, min_periods=window).apply(
        _pctile, raw=True
    )
    return out


def compute_features_for_pair(
    df_ohlc: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """
    Compute all v1 features at time t for each row. Causal only (bars <= t).

    Returns DataFrame with columns: pair, date, dataset_split, + all FEATURE_NAMES.
    """
    pair = cfg.get("pair", "")
    atr_period = int(cfg.get("atr_period", 14))
    date_start = cfg.get("date_range", {}).get("start", "2019-01-01")
    date_end = cfg.get("date_range", {}).get("end", "2026-01-01")
    discovery_end = cfg.get("split", {}).get("discovery_end", "2022-12-31")

    df = df_ohlc.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    from core.utils import calculate_atr, slice_df_by_dates

    df_slice, _ = slice_df_by_dates(df, date_start, date_end, inclusive="both")
    if df_slice.empty or len(df_slice) < 2:
        return _empty_features_df(pair)

    df_slice = calculate_atr(df_slice, period=atr_period)

    o = pd.to_numeric(df_slice["open"], errors="coerce").to_numpy(dtype="float64")
    h = pd.to_numeric(df_slice["high"], errors="coerce").to_numpy(dtype="float64")
    low_arr = pd.to_numeric(df_slice["low"], errors="coerce").to_numpy(dtype="float64")
    c = pd.to_numeric(df_slice["close"], errors="coerce").to_numpy(dtype="float64")
    atr = pd.to_numeric(df_slice["atr"], errors="coerce").to_numpy(dtype="float64")
    dates = df_slice["date"].tolist()
    n = len(df_slice)

    prev_close = np.roll(c, 1)
    prev_close[0] = np.nan

    ret_1 = _safe_div(c - prev_close, prev_close)
    ret_5 = np.full(n, np.nan)
    ret_20 = np.full(n, np.nan)
    for i in range(5, n):
        if np.isfinite(c[i - 5]) and c[i - 5] != 0:
            ret_5[i] = (c[i] / c[i - 5]) - 1.0
    for i in range(20, n):
        if np.isfinite(c[i - 20]) and c[i - 20] != 0:
            ret_20[i] = (c[i] / c[i - 20]) - 1.0

    def _slope(arr: np.ndarray, start: int, length: int) -> float:
        x = np.arange(length, dtype=float)
        y = arr[start - length : start]
        if not np.all(np.isfinite(y)) or np.var(y) == 0:
            return np.nan
        return float(np.polyfit(x, y, 1)[0])

    mom_slope_5 = np.array([_slope(c, i, 5) if i >= 5 else np.nan for i in range(n)])
    mom_slope_20 = np.array([_slope(c, i, 20) if i >= 20 else np.nan for i in range(n)])

    atr_14 = atr.copy()
    close_safe = np.where(np.isfinite(c) & (c != 0), c, np.nan)
    atrp_14 = _safe_div(atr_14, close_safe)

    atr_252_mu = pd.Series(atr_14).rolling(252, min_periods=252).mean().to_numpy()
    atr_252_std = pd.Series(atr_14).rolling(252, min_periods=252).std().to_numpy()
    atr_z_252 = np.full(n, np.nan)
    for i in range(252, n):
        if np.isfinite(atr_252_std[i]) and atr_252_std[i] > 0:
            atr_z_252[i] = (atr_14[i] - atr_252_mu[i]) / atr_252_std[i]

    atr_pctile_252 = stable_percentile_rank(
        pd.Series(atr_14, index=df_slice.index), 252
    ).to_numpy()

    tr = np.maximum(
        h - low_arr,
        np.maximum(
            np.abs(h - np.roll(c, 1)),
            np.abs(low_arr - np.roll(c, 1)),
        ),
    )
    tr[0] = h[0] - low_arr[0]
    true_range = tr

    atr_safe = np.where(np.isfinite(atr_14) & (atr_14 > 0), atr_14, np.nan)
    tr_atr_ratio = _safe_div(true_range, atr_safe)

    max_high_20 = pd.Series(h).rolling(20, min_periods=20).max().to_numpy()
    min_low_20 = pd.Series(low_arr).rolling(20, min_periods=20).min().to_numpy()
    range_20 = _safe_div(max_high_20 - min_low_20, close_safe)
    range_20_atr = _safe_div(max_high_20 - min_low_20, atr_safe)

    bb_std_20 = pd.Series(c).rolling(20, min_periods=20).std().to_numpy()
    bb_width_20 = np.full(n, np.nan)
    for i in range(20, n):
        if np.isfinite(close_safe[i]) and close_safe[i] != 0:
            bb_width_20[i] = (4.0 * bb_std_20[i]) / close_safe[i]

    bb_width_series = pd.Series(bb_width_20, index=df_slice.index)
    bb_width_pctile_252 = stable_percentile_rank(bb_width_series, 252).to_numpy()

    denom_range = max_high_20 - min_low_20
    denom_range = np.where(
        np.isfinite(denom_range) & (denom_range > 0), denom_range, np.nan
    )
    pos_in_range_20 = _safe_div(c - min_low_20, denom_range)
    pos_in_range_20 = np.clip(pos_in_range_20, 0.0, 1.0)

    dist_to_high_20_atr = _safe_div(max_high_20 - c, atr_safe)
    dist_to_low_20_atr = _safe_div(c - min_low_20, atr_safe)

    max_high_19 = np.full(n, np.nan)
    min_low_19 = np.full(n, np.nan)
    for i in range(19, n):
        max_high_19[i] = np.nanmax(h[i - 19 : i])
        min_low_19[i] = np.nanmin(low_arr[i - 19 : i])
    breakout_up_20 = (
        (c > max_high_19).astype(float)
        if np.all(np.isfinite(max_high_19[19:]))
        else np.full(n, np.nan)
    )
    breakout_dn_20 = (
        (c < min_low_19).astype(float)
        if np.all(np.isfinite(min_low_19[19:]))
        else np.full(n, np.nan)
    )
    for i in range(19):
        breakout_up_20[i] = np.nan
        breakout_dn_20[i] = np.nan

    hl = h - low_arr
    hl = np.where(np.isfinite(hl) & (hl > 0), hl, np.nan)
    body_pct = _safe_div(np.abs(c - o), hl)
    upper_wick_pct = _safe_div(h - np.maximum(o, c), hl)
    lower_wick_pct = _safe_div(np.minimum(o, c) - low_arr, hl)

    cutoff = pd.Timestamp(discovery_end) if discovery_end else pd.Timestamp("2022-12-31")

    rows = []
    for i in range(n):
        date_val = dates[i]
        dataset_split = "discovery" if pd.Timestamp(date_val) <= cutoff else "validation"
        rows.append({
            "pair": pair,
            "date": date_val,
            "dataset_split": dataset_split,
            "ret_1": ret_1[i],
            "ret_5": ret_5[i],
            "ret_20": ret_20[i],
            "mom_slope_5": mom_slope_5[i],
            "mom_slope_20": mom_slope_20[i],
            "atr_14": atr_14[i],
            "atrp_14": atrp_14[i],
            "atr_z_252": atr_z_252[i],
            "atr_pctile_252": atr_pctile_252[i],
            "true_range": true_range[i],
            "tr_atr_ratio": tr_atr_ratio[i],
            "range_20": range_20[i],
            "range_20_atr": range_20_atr[i],
            "bb_width_20": bb_width_20[i],
            "bb_width_pctile_252": bb_width_pctile_252[i],
            "pos_in_range_20": pos_in_range_20[i],
            "dist_to_high_20_atr": dist_to_high_20_atr[i],
            "dist_to_low_20_atr": dist_to_low_20_atr[i],
            "breakout_up_20": breakout_up_20[i],
            "breakout_dn_20": breakout_dn_20[i],
            "body_pct": body_pct[i],
            "upper_wick_pct": upper_wick_pct[i],
            "lower_wick_pct": lower_wick_pct[i],
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["pair", "date"]).reset_index(drop=True)
    return out


def _empty_features_df(pair: str) -> pd.DataFrame:
    cols = ["pair", "date", "dataset_split"] + FEATURE_NAMES
    return pd.DataFrame(columns=cols)


def compute_bin_edges_from_discovery(
    df: pd.DataFrame,
    feature: str,
    n_bins: int,
    min_per_bin: int,
) -> np.ndarray | None:
    """
    Compute quantile-based bin edges from discovery split only.
    Returns edges (length n_bins+1) or None if insufficient data.
    """
    disc = df[df["dataset_split"] == "discovery"]
    vals = disc[feature].dropna()
    if len(vals) < n_bins * min_per_bin:
        return None
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(vals, q)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_bin_edges(series: pd.Series, edges: np.ndarray) -> pd.Series:
    """Assign bin index 0..n_bins-1 using edges. Values outside [edges[1], edges[-2]] go to edge bins."""
    if edges is None or len(edges) < 2:
        return pd.Series(np.nan, index=series.index)
    bins = np.digitize(series, edges) - 1
    bins = np.clip(bins, 0, len(edges) - 2)
    return pd.Series(bins, index=series.index)


def _auc_proxy(pos_vals: np.ndarray, neg_vals: np.ndarray) -> float:
    """Rank-based AUC: Mann-Whitney U / (n_pos * n_neg). Deterministic."""
    n_pos = len(pos_vals)
    n_neg = len(neg_vals)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pos_f = pos_vals[np.isfinite(pos_vals)]
    neg_f = neg_vals[np.isfinite(neg_vals)]
    n_pos = len(pos_f)
    n_neg = len(neg_f)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    count = 0
    for p in pos_f:
        count += np.sum(neg_f < p) + 0.5 * np.sum(neg_f == p)
    return count / (n_pos * n_neg)


def compute_feature_stats(
    df_joined: pd.DataFrame,
    target_col: str,
    split: str | None,
) -> pd.DataFrame:
    """
    For each feature, compute mean_pos, mean_neg, std_pos, std_neg, effect_size_d, auc_proxy.
    If split is not None, filter to that dataset_split.
    """
    mask = df_joined["dataset_split"] == split if split else pd.Series(True, index=df_joined.index)
    if split:
        sub = df_joined.loc[mask]
    else:
        sub = df_joined
    target = (sub[target_col].fillna(False).astype(bool)).to_numpy()
    n_pos = int(np.sum(target))
    n_neg = int(np.sum(~target))

    rows = []
    for feat in FEATURE_NAMES:
        if feat not in sub.columns:
            continue
        vals = sub[feat].to_numpy(dtype=float)
        pos_vals = vals[target]
        neg_vals = vals[~target]
        pos_vals = pos_vals[np.isfinite(pos_vals)]
        neg_vals = neg_vals[np.isfinite(neg_vals)]
        mean_pos = np.mean(pos_vals) if len(pos_vals) > 0 else np.nan
        mean_neg = np.mean(neg_vals) if len(neg_vals) > 0 else np.nan
        std_pos = np.std(pos_vals) if len(pos_vals) > 1 else 0.0
        std_neg = np.std(neg_vals) if len(neg_vals) > 1 else 0.0
        pooled_std = np.sqrt(
            (std_pos**2 * len(pos_vals) + std_neg**2 * len(neg_vals))
            / max(len(pos_vals) + len(neg_vals) - 2, 1)
        )
        if pooled_std > 0 and np.isfinite(mean_pos) and np.isfinite(mean_neg):
            effect_d = (mean_pos - mean_neg) / pooled_std
        else:
            effect_d = np.nan
        auc = _auc_proxy(pos_vals, neg_vals)
        rows.append({
            "feature": feat,
            "split": split or "global",
            "target": target_col,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "mean_pos": mean_pos,
            "mean_neg": mean_neg,
            "std_pos": std_pos,
            "std_neg": std_neg,
            "effect_size_d": effect_d,
            "auc_proxy": auc,
        })
    return pd.DataFrame(rows)


def compute_feature_rankings(
    df_joined: pd.DataFrame,
    bin_edges: dict[str, np.ndarray],
    target_col: str,
    split_cfg: dict,
) -> pd.DataFrame:
    """
    For each feature: binned lift, stability, score.
    Returns rankings with score, best_bin, lift_disc, lift_val, effect_size_d, etc.
    """
    disc = df_joined[df_joined["dataset_split"] == "discovery"]
    val = df_joined[df_joined["dataset_split"] == "validation"]
    target = (df_joined[target_col].fillna(False).astype(bool)).to_numpy()
    base_rate = np.mean(target)

    min_pos = MIN_POS_ZONE_B if target_col == ZONE_B else MIN_POS_ZONE_C

    rows = []
    for feat in FEATURE_NAMES:
        if feat not in df_joined.columns:
            continue
        edges = bin_edges.get(feat)
        if edges is None:
            rows.append({
                "feature": feat,
                "target": target_col,
                "score": np.nan,
                "best_bin": np.nan,
                "lift_disc": np.nan,
                "lift_val": np.nan,
                "stability_gap": np.nan,
                "effect_size_d_disc": np.nan,
                "effect_size_d_val": np.nan,
                "auc_disc": np.nan,
                "auc_val": np.nan,
                "n_pos_validation": 0,
                "notes": "no_bin_edges",
            })
            continue

        disc_bin = apply_bin_edges(disc[feat], edges)
        val_bin = apply_bin_edges(val[feat], edges)
        disc_target = (disc[target_col].fillna(False).astype(bool)).to_numpy()
        val_target = (val[target_col].fillna(False).astype(bool)).to_numpy()

        n_bins = len(edges) - 1
        best_bin = 0
        best_lift_disc = 1.0
        best_lift_val = 1.0
        for b in range(n_bins):
            m_d = disc_bin == b
            m_v = val_bin == b
            if m_d.sum() < 10:
                continue
            p_d = disc_target[m_d].mean()
            p_v = val_target[m_v].mean() if m_v.sum() > 0 else 0.0
            lift_d = (p_d / base_rate) if base_rate > 0 else 1.0
            lift_v = (p_v / base_rate) if base_rate > 0 else 1.0
            if lift_d > best_lift_disc:
                best_lift_disc = lift_d
                best_lift_val = lift_v
                best_bin = b

        stability_gap = abs(best_lift_val - best_lift_disc)
        stability_gap_clipped = min(stability_gap, 1.0)
        score = min(best_lift_disc, best_lift_val) * (1.0 - stability_gap_clipped)

        stats_disc = compute_feature_stats(df_joined, target_col, "discovery")
        stats_val = compute_feature_stats(df_joined, target_col, "validation")
        row_d = stats_disc[stats_disc["feature"] == feat]
        row_v = stats_val[stats_val["feature"] == feat]
        effect_d_disc = float(row_d["effect_size_d"].iloc[0]) if len(row_d) else np.nan
        effect_d_val = float(row_v["effect_size_d"].iloc[0]) if len(row_v) else np.nan
        auc_disc = float(row_d["auc_proxy"].iloc[0]) if len(row_d) else np.nan
        auc_val = float(row_v["auc_proxy"].iloc[0]) if len(row_v) else np.nan

        n_pos_val = int(val_target.sum())
        notes = []
        if n_pos_val < min_pos:
            notes.append("insufficient_sample")
        if stability_gap > 0.2:
            notes.append("unstable")

        rows.append({
            "feature": feat,
            "target": target_col,
            "score": score,
            "best_bin": best_bin,
            "lift_disc": best_lift_disc,
            "lift_val": best_lift_val,
            "stability_gap": stability_gap,
            "effect_size_d_disc": effect_d_disc,
            "effect_size_d_val": effect_d_val,
            "auc_disc": auc_disc,
            "auc_val": auc_val,
            "n_pos_validation": n_pos_val,
            "notes": ";".join(notes) if notes else "",
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out
