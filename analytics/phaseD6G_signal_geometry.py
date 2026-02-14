"""
Phase D-6G: Signal conditioning of clean opportunity geometry.

Measures how entry signals condition the probability of reaching Y before MAE X.
Phase E-1.2: Primary objective 3R_before_2R, frequency guards, pass criteria.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PHASE_D_DISCOVERY_END = "2022-12-31"
DIRECTIONS = ("long", "short")
PRIMARY_H = 40
FREQ_FLOOR = 24
FREQ_CEIL = 120
CLUSTERING_THRESH = 0.20
DISCOVERY_LIFT_MIN = 0.10
VALIDATION_LIFT_MIN = 0.05
VALIDATION_LIFT_RATIO_MIN = 0.5
P4R_DROP_MAX_PCT = 0.10


def _to_bool(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(bool)


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    valid_atr = _to_bool(df["valid_atr"]) if "valid_atr" in df.columns else pd.Series(True, index=df.index)
    valid_ref = _to_bool(df["valid_ref"]) if "valid_ref" in df.columns else pd.Series(True, index=df.index)
    valid_h40 = _to_bool(df["valid_h40"]) if "valid_h40" in df.columns else pd.Series(True, index=df.index)
    return valid_atr & valid_ref & valid_h40


def _get_mfe_col(direction: str, x: int) -> str:
    return f"clean_mfe_{direction}_x{x}_h40"


def compute_pooled_signal_lift(
    merged: pd.DataFrame,
    signal_col: str,
    x_vals: tuple[int, ...],
    y_vals: tuple[float, ...],
) -> pd.DataFrame:
    """
    merged must have: pair, date, valid_*, clean_mfe_*_h40, and signal_col in {-1,0,+1}.
    """
    mask = _valid_mask(merged)
    sub = merged[mask].copy()
    sub["_sig"] = pd.to_numeric(sub[signal_col], errors="coerce").fillna(0).clip(-1, 1).astype(int)
    long_day = sub["_sig"] == 1
    short_day = sub["_sig"] == -1

    rows = []
    for direction in DIRECTIONS:
        signal_day = long_day if direction == "long" else short_day
        for x in x_vals:
            col = _get_mfe_col(direction, x)
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            vals = vals[np.isfinite(vals)]
            n_total = len(vals)
            if n_total == 0:
                continue
            idx_signal = sub.loc[signal_day].index.intersection(vals.index)
            vals_signal = sub.loc[idx_signal, col].dropna()
            vals_signal = vals_signal[np.isfinite(vals_signal)]
            n_signal = len(vals_signal)
            for y in y_vals:
                n_hit_total = int((vals >= y).sum())
                baseline_rate_y = n_hit_total / n_total if n_total else 0.0
                n_hit_signal = int((vals_signal >= y).sum()) if n_signal else 0
                signal_rate_y = n_hit_signal / n_signal if n_signal else 0.0
                lift = round(signal_rate_y - baseline_rate_y, 6)
                denom = max(baseline_rate_y, 1e-12)
                ratio = round(signal_rate_y / denom, 6)
                rows.append({
                    "signal_name": signal_col,
                    "direction": direction,
                    "x": x,
                    "y": float(y),
                    "n_total": n_total,
                    "n_signal": n_signal,
                    "baseline_rate": round(baseline_rate_y, 6),
                    "signal_rate": round(signal_rate_y, 6),
                    "lift": lift,
                    "ratio": ratio,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["signal_name", "direction", "x", "y"]).reset_index(drop=True)
    return out


def compute_pooled_signal_lift_stability(
    merged: pd.DataFrame,
    signal_col: str,
    x_vals: tuple[int, ...],
    y_vals: tuple[float, ...],
    discovery_end: str = PHASE_D_DISCOVERY_END,
) -> pd.DataFrame:
    """Discovery vs validation stability for signal lift."""
    mask = _valid_mask(merged)
    sub = merged[mask].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    cutoff = pd.Timestamp(discovery_end)
    sub["_split"] = np.where(sub["date"] <= cutoff, "discovery", "validation")
    sub["_sig"] = pd.to_numeric(sub[signal_col], errors="coerce").fillna(0).clip(-1, 1).astype(int)

    rows = []
    for direction in DIRECTIONS:
        signal_day = sub["_sig"] == (1 if direction == "long" else -1)
        for x in x_vals:
            col = _get_mfe_col(direction, x)
            if col not in sub.columns:
                continue
            for y in y_vals:
                disc = sub[sub["_split"] == "discovery"]
                val = sub[sub["_split"] == "validation"]
                disc_vals = disc[col].dropna()
                disc_vals = disc_vals[np.isfinite(disc_vals)]
                val_vals = val[col].dropna()
                val_vals = val_vals[np.isfinite(val_vals)]
                n_disc = len(disc_vals)
                n_val = len(val_vals)
                disc_sig = disc.loc[signal_day[disc.index]].index.intersection(disc_vals.index)
                val_sig = val.loc[signal_day[val.index]].index.intersection(val_vals.index)
                disc_sig_vals = disc.loc[disc_sig, col].dropna()
                disc_sig_vals = disc_sig_vals[np.isfinite(disc_sig_vals)]
                val_sig_vals = val.loc[val_sig, col].dropna()
                val_sig_vals = val_sig_vals[np.isfinite(val_sig_vals)]
                n_disc_sig = len(disc_sig_vals)
                n_val_sig = len(val_sig_vals)
                base_disc = (disc_vals >= y).mean() if n_disc else 0
                base_val = (val_vals >= y).mean() if n_val else 0
                sig_disc = (disc_sig_vals >= y).mean() if n_disc_sig else 0
                sig_val = (val_sig_vals >= y).mean() if n_val_sig else 0
                lift_disc = sig_disc - base_disc
                lift_val = sig_val - base_val
                delta_lift = round(lift_val - lift_disc, 6)
                rows.append({
                    "signal_name": signal_col,
                    "direction": direction,
                    "x": x,
                    "y": float(y),
                    "n_total_discovery": n_disc,
                    "n_signal_discovery": n_disc_sig,
                    "baseline_rate_discovery": round(float(base_disc), 6),
                    "signal_rate_discovery": round(float(sig_disc), 6),
                    "lift_discovery": round(float(lift_disc), 6),
                    "n_total_validation": n_val,
                    "n_signal_validation": n_val_sig,
                    "baseline_rate_validation": round(float(base_val), 6),
                    "signal_rate_validation": round(float(sig_val), 6),
                    "lift_validation": round(float(lift_val), 6),
                    "delta_lift": delta_lift,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["signal_name", "direction", "x", "y"]).reset_index(drop=True)
    return out


def compute_leaderboard_signal_lift(
    lift_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    rank_by: tuple[int, float] = (1, 2.0),
    min_n_signal: int = 30,
) -> pd.DataFrame:
    """
    Rank signals by lift on (x,y) long+short combined.
    rank_by = (x, y); ties broken by validation lift.
    """
    if lift_df.empty or stability_df.empty:
        return pd.DataFrame(columns=["rank", "signal_name", "lift_combined", "lift_validation", "n_signal", "notes"])

    x0, y0 = rank_by
    sub = lift_df[(lift_df["x"] == x0) & (lift_df["y"] == y0)]
    if sub.empty:
        return pd.DataFrame(columns=["rank", "signal_name", "lift_combined", "lift_validation", "n_signal", "notes"])

    combined = []
    for sig in sub["signal_name"].unique():
        s = sub[sub["signal_name"] == sig]
        n_signal = int(s["n_signal"].sum())
        lift_long = s[s["direction"] == "long"]["lift"].mean() if len(s[s["direction"] == "long"]) else 0
        lift_short = s[s["direction"] == "short"]["lift"].mean() if len(s[s["direction"] == "short"]) else 0
        lift_combined = (lift_long + lift_short) / 2.0
        stab = stability_df[(stability_df["signal_name"] == sig) & (stability_df["x"] == x0) & (stability_df["y"] == y0)]
        lift_val = stab["lift_validation"].mean() if not stab.empty else 0
        combined.append({
            "signal_name": sig,
            "lift_combined": round(float(lift_combined), 6),
            "lift_validation": round(float(lift_val), 6),
            "n_signal": int(n_signal),
            "notes": "insufficient_sample" if n_signal < min_n_signal else "",
        })
    agg = pd.DataFrame(combined)
    agg = agg.sort_values(["lift_combined", "lift_validation"], ascending=[False, False]).reset_index(drop=True)
    agg["rank"] = range(1, len(agg) + 1)
    return agg[["rank", "signal_name", "lift_combined", "lift_validation", "n_signal", "notes"]]


def _derive_metrics_from_stability(
    stab_df: pd.DataFrame,
    signal_name: str,
    x: int,
    y: float,
) -> dict:
    """
    Derive weighted signal/baseline rates from pooled_signal_lift_stability for given (x,y).
    Ignores rows with n_signal==0. Returns dict with signal_rate_disc, baseline_rate_disc,
    signal_rate_val, baseline_rate_val, insufficient_sample (bool).
    """
    sub = stab_df[
        (stab_df["signal_name"] == signal_name) & (stab_df["x"] == x) & (stab_df["y"] == y)
    ].copy()
    if sub.empty:
        return {
            "signal_rate_disc": 0.0,
            "baseline_rate_disc": 0.0,
            "signal_rate_val": 0.0,
            "baseline_rate_val": 0.0,
            "insufficient_sample": True,
        }
    disc_rows = sub[sub["n_signal_discovery"] > 0]
    val_rows = sub[sub["n_signal_validation"] > 0]
    n_disc = disc_rows["n_signal_discovery"].sum()
    n_val = val_rows["n_signal_validation"].sum()

    if n_disc <= 0:
        sig_disc = 0.0
        base_disc = 0.0
        insufficient_disc = True
    else:
        weighted_sig = (disc_rows["signal_rate_discovery"] * disc_rows["n_signal_discovery"]).sum()
        weighted_base = (disc_rows["baseline_rate_discovery"] * disc_rows["n_signal_discovery"]).sum()
        sig_disc = weighted_sig / n_disc
        base_disc = weighted_base / n_disc
        insufficient_disc = False

    if n_val <= 0:
        sig_val = 0.0
        base_val = 0.0
        insufficient_val = True
    else:
        weighted_sig = (val_rows["signal_rate_validation"] * val_rows["n_signal_validation"]).sum()
        weighted_base = (val_rows["baseline_rate_validation"] * val_rows["n_signal_validation"]).sum()
        sig_val = weighted_sig / n_val
        base_val = weighted_base / n_val
        insufficient_val = False

    return {
        "signal_rate_disc": round(float(sig_disc), 6),
        "baseline_rate_disc": round(float(base_disc), 6),
        "signal_rate_val": round(float(sig_val), 6),
        "baseline_rate_val": round(float(base_val), 6),
        "insufficient_sample": insufficient_disc or insufficient_val,
    }


def compute_annual_signals_per_pair(
    merged: pd.DataFrame,
    signal_col: str,
    date_start: str = "2019-01-01",
    date_end: str = "2026-01-01",
) -> float:
    """annual_signals_per_pair = total_signals / years / num_pairs."""
    sub = merged.copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub["_sig"] = pd.to_numeric(sub[signal_col], errors="coerce").fillna(0).clip(-1, 1).astype(int)
    sig_count = (sub["_sig"] != 0).sum()
    n_pairs = sub["pair"].nunique()
    years = (pd.Timestamp(date_end) - pd.Timestamp(date_start)).days / 365.25
    if years <= 0 or n_pairs <= 0:
        return 0.0
    return float(sig_count) / years / n_pairs


def compute_clustering_ratio_simple(merged: pd.DataFrame, signal_col: str, window_bars: int = 3) -> float:
    """Clustering: fraction of non-zero signals within window_bars of prior same-direction."""
    sub = merged[merged[signal_col].notna()].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub["_sig"] = pd.to_numeric(sub[signal_col], errors="coerce").fillna(0).clip(-1, 1).astype(int)
    non_zero = sub[sub["_sig"] != 0].copy()
    if len(non_zero) <= 1:
        return 0.0
    non_zero["_dir"] = np.where(non_zero["_sig"] == 1, "long", "short")
    non_zero = non_zero.sort_values(["pair", "_dir", "date"]).reset_index(drop=True)
    clustered = 0
    for (pair, direction), grp in non_zero.groupby(["pair", "_dir"]):
        if len(grp) <= 1:
            continue
        dates = grp["date"].sort_values().tolist()
        for i in range(1, len(dates)):
            gap = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 1])).days
            if 0 < gap <= window_bars:
                clustered += 1
    return clustered / len(non_zero) if len(non_zero) > 0 else 0.0


def compute_leaderboard_geometry_lock(
    merged_by_signal: dict[str, pd.DataFrame],
    stability_df: pd.DataFrame,
    primary_objective: str = "3R_before_2R",
    discovery_end: str = PHASE_D_DISCOVERY_END,
    date_start: str = "2019-01-01",
    date_end: str = "2026-01-01",
) -> pd.DataFrame:
    """
    Leaderboard sorted by P_3R_before_2R with frequency guards and pass criteria.
    P_3R, P_4R, P_2R and lifts are derived from pooled_signal_lift_stability (x=2,y=3; x=2,y=4; x=1,y=2).
    """
    rows = []
    for signal_name, merged in merged_by_signal.items():
        if merged.empty:
            continue
        ann_sig = compute_annual_signals_per_pair(merged, signal_name, date_start, date_end)
        clust = compute_clustering_ratio_simple(merged, signal_name, 3)

        m3 = _derive_metrics_from_stability(stability_df, signal_name, 2, 3.0)
        m4 = _derive_metrics_from_stability(stability_df, signal_name, 2, 4.0)
        m2 = _derive_metrics_from_stability(stability_df, signal_name, 1, 2.0)

        p_disc = m3["signal_rate_disc"]
        p_val = m3["signal_rate_val"]
        base_disc = m3["baseline_rate_disc"]
        base_val = m3["baseline_rate_val"]
        denom_disc = max(base_disc, 1e-12)
        denom_val = max(base_val, 1e-12)
        lift_disc = (p_disc - base_disc) / denom_disc
        lift_val = (p_val - base_val) / denom_val

        p_4r_disc = m4["signal_rate_disc"]
        p_4r_val = m4["signal_rate_val"]

        reject_reasons = []
        if m3["insufficient_sample"]:
            reject_reasons.append("insufficient_sample_x2y3")
        if ann_sig < FREQ_FLOOR:
            reject_reasons.append(f"annual_signals_per_pair_{ann_sig:.1f}_below_{FREQ_FLOOR}")
        if ann_sig > FREQ_CEIL:
            reject_reasons.append(f"annual_signals_per_pair_{ann_sig:.1f}_above_{FREQ_CEIL}")
        if clust > CLUSTERING_THRESH:
            reject_reasons.append(f"clustering_ratio_{clust:.2f}_above_{CLUSTERING_THRESH}")

        if lift_disc < DISCOVERY_LIFT_MIN:
            reject_reasons.append(f"discovery_lift_{lift_disc:.3f}_below_{DISCOVERY_LIFT_MIN}")
        if lift_val < VALIDATION_LIFT_MIN or lift_val <= 0:
            reject_reasons.append(f"validation_lift_{lift_val:.3f}_fails")
        if lift_disc > 1e-9 and lift_val < VALIDATION_LIFT_RATIO_MIN * lift_disc:
            reject_reasons.append("validation_lift_below_half_discovery")

        denom_4r = max(p_4r_disc, 1e-12)
        p4r_drop = (p_4r_disc - p_4r_val) / denom_4r if denom_4r > 0 else 0.0
        if p4r_drop > P4R_DROP_MAX_PCT:
            reject_reasons.append(f"P_4R_before_2R_drop_{p4r_drop:.2%}_above_{P4R_DROP_MAX_PCT:.0%}")

        passed = len(reject_reasons) == 0
        rows.append({
            "signal_name": signal_name,
            "P_3R_before_2R_disc": round(p_disc, 6),
            "P_3R_before_2R_val": round(p_val, 6),
            "P_2R_before_1R_disc": round(m2["signal_rate_disc"], 6),
            "P_2R_before_1R_val": round(m2["signal_rate_val"], 6),
            "P_4R_before_2R_disc": round(p_4r_disc, 6),
            "P_4R_before_2R_val": round(p_4r_val, 6),
            "discovery_lift": round(lift_disc, 6),
            "validation_lift": round(lift_val, 6),
            "annual_signals_per_pair": round(ann_sig, 2),
            "clustering_ratio": round(clust, 4),
            "PASS": passed,
            "reject_reason": "; ".join(reject_reasons) if reject_reasons else "",
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("P_3R_before_2R_disc", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df
