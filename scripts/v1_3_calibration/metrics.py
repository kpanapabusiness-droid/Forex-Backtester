"""v1.3 calibration — metric computations.

Designed to run once per dataset on the normalised per-bar frame from
``load_paths.load_paths``. Each axis is one function returning a dict
of metric → value (scalars) plus side DataFrames where the spec calls
for full distributions.

Memory note: Arc 1's per-bar frame is ~22M rows. Per-trade aggregates
use a single ``groupby('trade_id', sort=False)`` pass per axis where
possible. Filtering to ``bar_offset <= 240`` up-front roughly halves
the data volume since metrics are spec'd "within 240 bars" except for
the per-trade-final-state aggregation.

R-unit convention: all values in SL-distance R-units, R = 2 × ATR.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

H_LIST = (1, 3, 6, 12, 24, 48, 120, 240)
TP_LEVELS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
TRAIL_WIDTHS = (0.3, 0.5, 0.75, 1.0, 1.5)
MFE_LOCK_LEVELS = (0.5, 1.0, 1.5, 2.0)
PERCENTILES = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)
COND_T = (1, 3, 5, 10, 20)
RACE_R = 1.0
WRONG_WAY_R = 1.5


def _per_trade_h240(paths: pd.DataFrame) -> pd.DataFrame:
    """Per-trade summary over bar_offset ≤ 240. One row per trade."""
    h240 = paths[paths["bar_offset"] <= 240]
    g = h240.groupby("trade_id", sort=False)
    # vectorised per-trade max/min and first-crossing bar offsets.
    out = pd.DataFrame({
        "max_mfe_h120":     h240[h240["bar_offset"] <= 120].groupby("trade_id", sort=False)["mfe_so_far_r"].max(),
        "max_mfe_h240":     g["mfe_so_far_r"].max(),
        "max_mae_h240":     g["mae_so_far_r"].min(),
    })
    # First-crossing bar offsets via masked groupby idxmin → bar_offset.
    def _first_cross_at_or_above(col: str, thresh: float) -> pd.Series:
        mask = h240[col] >= thresh
        sub = h240.loc[mask, ["trade_id", "bar_offset"]]
        return sub.groupby("trade_id", sort=False)["bar_offset"].min()

    def _first_cross_at_or_below(col: str, thresh: float) -> pd.Series:
        mask = h240[col] <= thresh
        sub = h240.loc[mask, ["trade_id", "bar_offset"]]
        return sub.groupby("trade_id", sort=False)["bar_offset"].min()

    out["first_1R"]    = _first_cross_at_or_above("mfe_so_far_r", 1.0)
    out["first_1_5R"]  = _first_cross_at_or_above("mfe_so_far_r", 1.5)
    out["first_2R"]    = _first_cross_at_or_above("mfe_so_far_r", 2.0)
    out["first_3R"]    = _first_cross_at_or_above("mfe_so_far_r", 3.0)
    out["first_5R"]    = _first_cross_at_or_above("mfe_so_far_r", 5.0)
    out["first_neg_1R"]   = _first_cross_at_or_below("mae_so_far_r", -1.0)
    out["first_neg_1_5R"] = _first_cross_at_or_below("mae_so_far_r", -1.5)

    # time_to_peak_mfe = bar_offset of the running MFE peak.
    peak_idx = h240.loc[h240.groupby("trade_id", sort=False)["mfe_so_far_r"].idxmax(), ["trade_id", "bar_offset"]]
    out["time_to_peak_mfe"] = peak_idx.set_index("trade_id")["bar_offset"]

    # Closing R at specific bar offsets (for Axis 2e conditional predictivity).
    for t in COND_T:
        row_at_t = h240.loc[h240["bar_offset"] == t, ["trade_id", "close_r"]]
        out[f"close_at_t{t}"] = row_at_t.set_index("trade_id")["close_r"]

    # close_r at bar_offset=240 (or the latest available bar within the cap).
    last_within_240 = h240.groupby("trade_id", sort=False).tail(1).set_index("trade_id")["close_r"]
    out["close_at_h240"] = last_within_240

    return out.reset_index()


def _per_trade_at_exit(paths: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per-trade snapshot at the LAST held bar (bar_offset == bars_held)."""
    # If bars_held > 240 we still want the last held bar within range.
    m = meta[["trade_id", "bars_held"]].copy()
    p = paths.merge(m, on="trade_id", how="inner", validate="many_to_one")
    exit_rows = p[p["bar_offset"] == p["bars_held"]]
    # Some Arc 1 trades have bars_held clamped to the max emitted offset
    # (forever-held). Tail-fallback for trades that don't have an exact
    # match (e.g. data_end truncation):
    matched = exit_rows.set_index("trade_id")
    missing = set(meta["trade_id"]) - set(matched.index)
    if missing:
        tail = (
            paths[paths["trade_id"].isin(missing)]
            .groupby("trade_id", sort=False).tail(1).set_index("trade_id")
        )
        matched = pd.concat([matched, tail], axis=0)
    return matched[["close_r", "mfe_so_far_r", "mae_so_far_r"]].rename(columns={
        "close_r":        "final_close_r",
        "mfe_so_far_r":   "mfe_at_exit",
        "mae_so_far_r":   "mae_at_exit",
    }).reset_index()


# ──────────────────────────────────────────────────────────────────────
# Axis 1 — Peak magnitude
# ──────────────────────────────────────────────────────────────────────

def axis1_peak_magnitude(per_trade: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    n = len(per_trade)
    mfe120 = per_trade["max_mfe_h120"]
    mfe240 = per_trade["max_mfe_h240"]
    metrics = {
        "n_trades":                  n,
        "pool_median_fwd_mfe_h120":  float(mfe120.median()),
        "pool_median_fwd_mfe_h240":  float(mfe240.median()),
        "pool_frac_reach_1R":        float((mfe240 >= 1.0).mean()),
        "pool_frac_reach_1_5R":      float((mfe240 >= 1.5).mean()),
        "pool_frac_reach_2R":        float((mfe240 >= 2.0).mean()),
        "pool_p50_fwd_mfe_h240":     float(mfe240.quantile(0.50)),
        "pool_p90_fwd_mfe_h240":     float(mfe240.quantile(0.90)),
        "pool_p95_fwd_mfe_h240":     float(mfe240.quantile(0.95)),
        "pool_p99_fwd_mfe_h240":     float(mfe240.quantile(0.99)),
    }
    # Full distribution side file.
    dist = pd.DataFrame({
        "percentile": list(PERCENTILES),
        "fwd_mfe_h240_R": [float(mfe240.quantile(p / 100)) for p in PERCENTILES],
    })
    return metrics, dist


# ──────────────────────────────────────────────────────────────────────
# Axis 2 — six exit families
# ──────────────────────────────────────────────────────────────────────

def axis2a_time_exit(paths: pd.DataFrame, per_trade: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Pure-time-exit with −1R SL. Capture ratio = mean(realised) / mean(peak)."""
    results = {}
    curve = []
    h240 = paths[paths["bar_offset"] <= 240]
    for h in H_LIST:
        peak_per_trade = (
            h240[h240["bar_offset"] <= h]
            .groupby("trade_id", sort=False)["mfe_so_far_r"].max()
        )
        # SL fires at first bar where mae_so_far_r ≤ −1; time-exit fires at h.
        sl_first = (
            h240[(h240["bar_offset"] <= h) & (h240["mae_so_far_r"] <= -1.0)]
            .groupby("trade_id", sort=False)["bar_offset"].min()
        )
        # close_r at bar_offset = h per trade (or last available bar within range).
        bar_at_h = h240[h240["bar_offset"] == h].set_index("trade_id")["close_r"]
        # Fallback: take the latest bar within h for trades with no exact bar_offset=h
        # (forward window truncation).
        last_within_h = (
            h240[h240["bar_offset"] <= h]
            .groupby("trade_id", sort=False).tail(1).set_index("trade_id")["close_r"]
        )
        bar_at_h = bar_at_h.reindex(last_within_h.index).fillna(last_within_h)

        realised = bar_at_h.copy()
        # Where SL was hit before time h, realised = −1.0
        realised.loc[realised.index.isin(sl_first.index)] = -1.0
        capture = float(realised.mean() / peak_per_trade.mean()) if peak_per_trade.mean() > 0 else float("nan")
        curve.append({"h": h, "capture_ratio": capture, "mean_realised_R": float(realised.mean()), "mean_peak_R": float(peak_per_trade.mean())})

    df = pd.DataFrame(curve)
    best = df.iloc[df["capture_ratio"].idxmax()]
    results["time_exit_best_h"] = int(best["h"])
    results["time_exit_best_capture"] = float(best["capture_ratio"])
    return results, df


def axis2b_trail_exit(paths: pd.DataFrame, per_trade: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Trail exit width W. Close-based detection (intrabar-low not available for Arc 1).

    Exits when close_r ≤ (mfe_so_far_r − W) at some bar t, OR SL (mae ≤ −1)
    fires first, OR cap at bar 240 (use close_r at last bar within cap).
    """
    h240 = paths[paths["bar_offset"] <= 240][["trade_id", "bar_offset", "close_r", "mfe_so_far_r", "mae_so_far_r"]].copy()
    curve = []
    for W in TRAIL_WIDTHS:
        h240["trail_hit"] = h240["close_r"] <= (h240["mfe_so_far_r"] - W)
        h240["sl_hit"]    = h240["mae_so_far_r"] <= -1.0
        trail_first = (
            h240.loc[h240["trail_hit"], ["trade_id", "bar_offset", "close_r"]]
            .groupby("trade_id", sort=False).head(1)
            .set_index("trade_id")
        )
        sl_first = (
            h240.loc[h240["sl_hit"], ["trade_id", "bar_offset"]]
            .groupby("trade_id", sort=False).head(1)
            .set_index("trade_id")
        )
        last_bar = (
            h240.groupby("trade_id", sort=False).tail(1)
            .set_index("trade_id")[["bar_offset", "close_r"]]
        )

        # Compose: choose earliest exit, mark realised R accordingly.
        all_ids = last_bar.index
        df = pd.DataFrame(index=all_ids)
        df["trail_bar"]  = trail_first["bar_offset"].reindex(all_ids)
        df["trail_r"]    = trail_first["close_r"].reindex(all_ids)
        df["sl_bar"]     = sl_first["bar_offset"].reindex(all_ids)
        df["cap_bar"]    = last_bar["bar_offset"]
        df["cap_r"]      = last_bar["close_r"]

        trail_first_arr = df["trail_bar"].fillna(np.inf).to_numpy()
        sl_first_arr    = df["sl_bar"].fillna(np.inf).to_numpy()
        cap_arr         = df["cap_bar"].to_numpy()

        # Pick whichever happens first.
        chosen_bar = np.minimum.reduce([trail_first_arr, sl_first_arr, cap_arr])
        realised = np.full(len(df), np.nan, dtype=np.float64)
        is_sl    = (sl_first_arr <= trail_first_arr) & (sl_first_arr <= cap_arr) & np.isfinite(sl_first_arr)
        is_trail = ~is_sl & (trail_first_arr <= cap_arr) & np.isfinite(trail_first_arr)
        is_cap   = ~is_sl & ~is_trail
        realised[is_sl]    = -1.0
        realised[is_trail] = df["trail_r"].to_numpy()[is_trail]
        realised[is_cap]   = df["cap_r"].to_numpy()[is_cap]

        peak = (
            h240.groupby("trade_id", sort=False)["mfe_so_far_r"].max()
            .reindex(all_ids).to_numpy()
        )
        mean_r = float(np.nanmean(realised))
        mean_p = float(np.nanmean(peak))
        capture = mean_r / mean_p if mean_p > 0 else float("nan")
        curve.append({"W": W, "capture_ratio": capture, "mean_realised_R": mean_r, "mean_peak_R": mean_p})

    df = pd.DataFrame(curve)
    best = df.iloc[df["capture_ratio"].idxmax()]
    return ({
        "trail_exit_best_W": float(best["W"]),
        "trail_exit_best_capture": float(best["capture_ratio"]),
    }, df)


def axis2c_tp_exit(paths: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fixed TP at level X. Uses running mfe_so_far_r — exact regardless of bar OHLC availability."""
    h240 = paths[paths["bar_offset"] <= 240]
    curve = []
    sl_first = (
        h240[h240["mae_so_far_r"] <= -1.0]
        .groupby("trade_id", sort=False)["bar_offset"].min()
    )
    last_bar = h240.groupby("trade_id", sort=False).tail(1).set_index("trade_id")[["bar_offset", "close_r"]]
    peak_240 = h240.groupby("trade_id", sort=False)["mfe_so_far_r"].max()
    all_ids = last_bar.index

    for X in TP_LEVELS:
        tp_first = (
            h240[h240["mfe_so_far_r"] >= X]
            .groupby("trade_id", sort=False)["bar_offset"].min()
        )
        tp_arr   = tp_first.reindex(all_ids).fillna(np.inf).to_numpy()
        sl_arr   = sl_first.reindex(all_ids).fillna(np.inf).to_numpy()
        cap_arr  = last_bar["bar_offset"].reindex(all_ids).to_numpy()
        cap_r    = last_bar["close_r"].reindex(all_ids).to_numpy()

        is_tp   = (tp_arr <= sl_arr) & (tp_arr <= cap_arr) & np.isfinite(tp_arr)
        is_sl   = ~is_tp & (sl_arr <= cap_arr) & np.isfinite(sl_arr)
        is_cap  = ~is_tp & ~is_sl
        realised = np.full(len(all_ids), np.nan, dtype=np.float64)
        realised[is_tp]  = X
        realised[is_sl]  = -1.0
        realised[is_cap] = cap_r[is_cap]
        peak = peak_240.reindex(all_ids).to_numpy()
        mean_r = float(np.nanmean(realised))
        mean_p = float(np.nanmean(peak))
        capture = mean_r / mean_p if mean_p > 0 else float("nan")
        curve.append({"X": X, "capture_ratio": capture, "mean_realised_R": mean_r, "mean_peak_R": mean_p})

    df = pd.DataFrame(curve)
    best = df.iloc[df["capture_ratio"].idxmax()]
    return ({
        "tp_exit_best_X": float(best["X"]),
        "tp_exit_best_capture": float(best["capture_ratio"]),
    }, df)


def axis2d_mfe_lock(paths: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """MFE-lock (breakeven). Trigger at X → SL → 0R.

    If mfe_so_far_r reaches X at bar t_trigger, the new SL becomes 0R.
    Exit when close_r ≤ 0 at any bar > t_trigger, OR original SL fires
    before trigger, OR time exit at bar 240.
    """
    h240 = paths[paths["bar_offset"] <= 240][["trade_id", "bar_offset", "close_r", "mfe_so_far_r", "mae_so_far_r"]].copy()
    sl_first_orig = (
        h240[h240["mae_so_far_r"] <= -1.0]
        .groupby("trade_id", sort=False)["bar_offset"].min()
    )
    last_bar = h240.groupby("trade_id", sort=False).tail(1).set_index("trade_id")[["bar_offset", "close_r"]]
    peak_240 = h240.groupby("trade_id", sort=False)["mfe_so_far_r"].max()
    all_ids = last_bar.index

    curve = []
    for X in MFE_LOCK_LEVELS:
        trig = (
            h240[h240["mfe_so_far_r"] >= X]
            .groupby("trade_id", sort=False)["bar_offset"].min()
        )
        # After trigger, the new SL is 0 — exit at the first bar > trigger where close_r ≤ 0.
        h240_m = h240.merge(trig.rename("trig_bar"), left_on="trade_id", right_index=True, how="left")
        post_trig_mask = (
            h240_m["trig_bar"].notna()
            & (h240_m["bar_offset"] >= h240_m["trig_bar"])
            & (h240_m["close_r"] <= 0.0)
        )
        lock_exit = (
            h240_m.loc[post_trig_mask, ["trade_id", "bar_offset"]]
            .groupby("trade_id", sort=False)["bar_offset"].min()
        )

        trig_arr      = trig.reindex(all_ids).fillna(np.inf).to_numpy()
        sl_orig_arr   = sl_first_orig.reindex(all_ids).fillna(np.inf).to_numpy()
        lock_arr      = lock_exit.reindex(all_ids).fillna(np.inf).to_numpy()
        cap_bar       = last_bar["bar_offset"].reindex(all_ids).to_numpy()
        cap_r         = last_bar["close_r"].reindex(all_ids).to_numpy()

        # Pre-trigger SL fires if sl_orig < trig.
        is_pre_sl     = np.isfinite(sl_orig_arr) & (sl_orig_arr < trig_arr) & (sl_orig_arr <= cap_bar)
        # Post-trigger lock fires if lock < cap and trig is finite.
        is_lock       = ~is_pre_sl & np.isfinite(lock_arr) & (lock_arr <= cap_bar)
        is_cap        = ~is_pre_sl & ~is_lock
        realised = np.full(len(all_ids), np.nan, dtype=np.float64)
        realised[is_pre_sl] = -1.0
        realised[is_lock]   = 0.0
        realised[is_cap]    = cap_r[is_cap]
        peak = peak_240.reindex(all_ids).to_numpy()
        mean_r = float(np.nanmean(realised))
        mean_p = float(np.nanmean(peak))
        capture = mean_r / mean_p if mean_p > 0 else float("nan")
        curve.append({"X": X, "capture_ratio": capture, "mean_realised_R": mean_r, "mean_peak_R": mean_p})

    df = pd.DataFrame(curve)
    best = df.iloc[df["capture_ratio"].idxmax()]
    return ({
        "mfe_lock_best_X": float(best["X"]),
        "mfe_lock_best_capture": float(best["capture_ratio"]),
    }, df)


def axis2e_conditional_predictivity(per_trade: pd.DataFrame) -> pd.DataFrame:
    """Descriptive: corr(close_r at t, final R) plus decile conditional means."""
    final_R = per_trade["close_at_h240"]
    rows = []
    for t in COND_T:
        col = f"close_at_t{t}"
        joined = pd.DataFrame({"x": per_trade[col], "y": final_R}).dropna()
        if len(joined) < 30:
            continue
        corr = float(joined["x"].corr(joined["y"]))
        deciles = pd.qcut(joined["x"], 10, labels=False, duplicates="drop")
        cond_means = joined.groupby(deciles)["y"].mean().tolist()
        rows.append({
            "t": t,
            "corr_close_t_vs_final": corr,
            "decile_means": str(cond_means),
            "n_trades": int(len(joined)),
        })
    return pd.DataFrame(rows)


def axis2f_reentry_descriptive(per_trade: pd.DataFrame) -> dict:
    """P(final R < max MFE − 1 AND max MFE > 2)."""
    mask = (per_trade["close_at_h240"] < per_trade["max_mfe_h240"] - 1.0) & (per_trade["max_mfe_h240"] > 2.0)
    return {"frac_reentry_candidates": float(mask.mean()), "n_candidates": int(mask.sum()), "n_total": int(len(per_trade))}


def axis2g_in_trade_smoothness(paths: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Local peak count + monotonicity within the HELD window."""
    # Restrict to held bars only.
    p = paths[paths["is_held"] == 1][["trade_id", "bar_offset", "close_r", "mfe_so_far_r"]].copy()
    p = p.sort_values(["trade_id", "bar_offset"])
    # local peak = bar where mfe_so_far_r strictly increased from prior bar
    p["mfe_prev"] = p.groupby("trade_id", sort=False)["mfe_so_far_r"].shift(1)
    p["is_peak"]  = (p["mfe_so_far_r"] > p["mfe_prev"]).fillna(False)
    peaks_per_trade = p.groupby("trade_id", sort=False)["is_peak"].sum()

    # Monotonicity in profit: among bars with close_r > 0, fraction where close_r advanced.
    p["close_prev"] = p.groupby("trade_id", sort=False)["close_r"].shift(1)
    in_profit       = p["close_r"] > 0
    advanced        = (p["close_r"] > p["close_prev"]).fillna(False) & in_profit
    retreat_or_eq   = in_profit & ~advanced
    mono_per_trade  = (
        advanced.groupby(p["trade_id"]).sum()
        / (in_profit.groupby(p["trade_id"]).sum().replace(0, np.nan))
    )

    out = pd.DataFrame({
        "local_peaks_count":          peaks_per_trade,
        "monotonicity_ratio_in_profit": mono_per_trade,
    }).reset_index()
    return out


def axis2h_time_to_peak_cv(per_trade: pd.DataFrame) -> dict:
    s = per_trade["time_to_peak_mfe"].dropna()
    mean = float(s.mean())
    std  = float(s.std())
    return {
        "time_to_peak_mean": mean,
        "time_to_peak_std":  std,
        "time_to_peak_cv":   (std / mean) if mean > 0 else float("nan"),
        "time_to_peak_p05":  float(s.quantile(0.05)),
        "time_to_peak_p25":  float(s.quantile(0.25)),
        "time_to_peak_p50":  float(s.quantile(0.50)),
        "time_to_peak_p75":  float(s.quantile(0.75)),
        "time_to_peak_p95":  float(s.quantile(0.95)),
    }


# ──────────────────────────────────────────────────────────────────────
# Axis 3 — path hostility
# ──────────────────────────────────────────────────────────────────────

def axis3_path_hostility(per_trade: pd.DataFrame, exit_snap: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    cap = 240
    # Set trade_id as index throughout so all comparisons align on the same key.
    pt   = per_trade.set_index("trade_id")
    snap = exit_snap.set_index("trade_id")
    joined = pt.join(snap[["final_close_r", "mfe_at_exit", "mae_at_exit"]], how="left")

    first_p   = joined["first_1R"].fillna(cap).clip(upper=cap)
    first_n   = joined["first_neg_1R"].fillna(cap).clip(upper=cap)
    race      = first_p - first_n

    first_p15 = joined["first_1_5R"].fillna(cap).clip(upper=cap)
    first_n15 = joined["first_neg_1_5R"].fillna(cap).clip(upper=cap)
    wrong_way_mask   = first_n15 < first_p15
    wrong_way_active = (first_p15 < cap) | (first_n15 < cap)

    winners = joined["final_close_r"] > 0
    ratio = (-joined["mae_at_exit"] / joined["mfe_at_exit"]).where(winners & (joined["mfe_at_exit"] > 0))
    peak_and_collapse = (joined["max_mfe_h240"] >= 1.0) & (joined["final_close_r"] < 0)

    metrics = {
        "race_condition_median": float(race.median()),
        "race_condition_mean":   float(race.mean()),
        "mae_mfe_ratio_winners_median": float(ratio.dropna().median()) if ratio.notna().any() else float("nan"),
        "pct_peak_and_collapse": float(peak_and_collapse.mean()),
        "pct_wrong_way": float((wrong_way_mask & wrong_way_active).mean()),
        "n_trades_with_race_signal": int(((first_p < cap) | (first_n < cap)).sum()),
    }
    dist = pd.DataFrame({
        "trade_id":             joined.index.values,
        "race_condition":       race.values,
        "mae_mfe_ratio_winners": ratio.values,
    })
    return metrics, dist


# ──────────────────────────────────────────────────────────────────────
# Shape tagging + mass-in-band (Steps 6 + 7)
# ──────────────────────────────────────────────────────────────────────

def shape_tag(per_trade: pd.DataFrame) -> dict:
    s = per_trade["max_mfe_h240"].dropna()
    p5, p50, p90, p95, p99 = (float(s.quantile(q)) for q in (0.05, 0.50, 0.90, 0.95, 0.99))
    skew = float(s.skew())
    kurt = float(s.kurt())
    sarle = (skew ** 2 + 1.0) / kurt if kurt and not np.isnan(kurt) else float("nan")

    if p50 < 0.5:
        tag = "no_magnitude"
    elif ((p95 / p50) < 1.5 if p50 > 0 else False) and ((p95 - p50) < (p50 - p5)):
        tag = "tight_unimodal"
    elif p99 > 3.0 * p90:
        tag = "heavy_right_tail"
    elif (not np.isnan(sarle)) and (sarle > 0.55):
        tag = "bimodal"
    elif (p50 > 0) and ((p95 / p50) > 3.0):
        tag = "scattered"
    else:
        tag = "unclassified"
    return {
        "p5": p5, "p50": p50, "p90": p90, "p95": p95, "p99": p99,
        "skew": skew, "kurt": kurt, "sarle": sarle, "tag": tag,
    }


def mass_in_band(per_trade: pd.DataFrame) -> dict:
    s = per_trade["max_mfe_h240"].dropna()
    n = len(s)
    return {
        "n_trades":           n,
        "band_0_to_0_5R":     float(((s >= 0)   & (s < 0.5)).sum() / n),
        "band_0_5_to_1R":     float(((s >= 0.5) & (s < 1.0)).sum() / n),
        "band_1_to_2R":       float(((s >= 1.0) & (s < 2.0)).sum() / n),
        "band_2_to_5R":       float(((s >= 2.0) & (s < 5.0)).sum() / n),
        "band_above_5R":      float((s >= 5.0).sum() / n),
    }


# ──────────────────────────────────────────────────────────────────────
# Per-dataset orchestrator (steps 3-7 minus per-cluster)
# ──────────────────────────────────────────────────────────────────────

def compute_dataset(
    name: str, paths: pd.DataFrame, meta: pd.DataFrame
) -> dict:
    """Compute the full v1.3 metric battery on one dataset.

    Returns a dict with one entry per axis / table; the caller writes
    them out to CSVs.
    """
    per_trade = _per_trade_h240(paths)
    per_trade = per_trade.merge(
        meta[["trade_id", "pair", "bars_held"]], on="trade_id", how="left"
    )
    exit_snap = _per_trade_at_exit(paths, meta)

    a1_metrics, a1_dist = axis1_peak_magnitude(per_trade)
    a2a_metrics, a2a_curve = axis2a_time_exit(paths, per_trade)
    a2b_metrics, a2b_curve = axis2b_trail_exit(paths, per_trade)
    a2c_metrics, a2c_curve = axis2c_tp_exit(paths)
    a2d_metrics, a2d_curve = axis2d_mfe_lock(paths)
    a2e_table              = axis2e_conditional_predictivity(per_trade)
    a2f_metrics            = axis2f_reentry_descriptive(per_trade)
    a2g_table              = axis2g_in_trade_smoothness(paths, meta)
    a2h_metrics            = axis2h_time_to_peak_cv(per_trade)
    a3_metrics, a3_dist    = axis3_path_hostility(per_trade, exit_snap)
    shape                  = shape_tag(per_trade)
    bands                  = mass_in_band(per_trade)

    return {
        "dataset":          name,
        "per_trade":        per_trade,
        "exit_snap":        exit_snap,
        "axis1":            a1_metrics,
        "axis1_dist":       a1_dist,
        "axis2a":           a2a_metrics,
        "axis2a_curve":     a2a_curve,
        "axis2b":           a2b_metrics,
        "axis2b_curve":     a2b_curve,
        "axis2c":           a2c_metrics,
        "axis2c_curve":     a2c_curve,
        "axis2d":           a2d_metrics,
        "axis2d_curve":     a2d_curve,
        "axis2e":           a2e_table,
        "axis2f":           a2f_metrics,
        "axis2g":           a2g_table,
        "axis2h":           a2h_metrics,
        "axis3":            a3_metrics,
        "axis3_dist":       a3_dist,
        "shape":            shape,
        "mass_bands":       bands,
    }


def aggregate_smoothness(per_dataset_smoothness: pd.DataFrame) -> dict:
    """Pool-level summary of axis 2g (in-trade smoothness)."""
    return {
        "local_peaks_pool_median": float(per_dataset_smoothness["local_peaks_count"].median()),
        "local_peaks_pool_mean":   float(per_dataset_smoothness["local_peaks_count"].mean()),
        "monotonicity_pool_median": float(
            per_dataset_smoothness["monotonicity_ratio_in_profit"].dropna().median()
        ),
    }
