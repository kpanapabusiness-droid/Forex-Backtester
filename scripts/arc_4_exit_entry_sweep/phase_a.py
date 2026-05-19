"""Phase A — deeper diagnostics on baseline data.

A1: alive vs full-window peak (trail extension)
A2: pre vs post trail-lock path behaviour
A3: win/loss metrics + winner final_r distribution
A4: per-cluster deep characterisation (match cluster 1 depth for 0/2/3)
A5: winners vs losers as outcome groups (with t-stats)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scistats

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from lib_sim import load_path_arrays, simulate_policy_vectorized, EXIT_CODE_TO_LABEL

REPO = THIS.parents[2]
OUT_DIR = REPO / "results" / "arc_4_exit_entry_sweep"
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "trades_all": REPO / "results" / "l_arc_4" / "step1" / "trades_all.csv",
    "clusters_K4": REPO / "results" / "l_arc_4" / "step2" / "clusters_K4.csv",
    "refit_per_trade": REPO / "results" / "l_arc_4" / "step5c" / "per_trade_simulated_refit_1.csv",
    "path_metrics": REPO / "results" / "arc_4_characterisation" / "per_trade_path_metrics.csv",
    "bar_0_20": REPO / "results" / "arc_4_characterisation" / "per_trade_bar_0_20.csv",
}

print("[info] loading inputs...")
trades_all = pd.read_csv(INPUTS["trades_all"]).sort_values("trade_id").reset_index(drop=True)
clusters = pd.read_csv(INPUTS["clusters_K4"])
refit = pd.read_csv(INPUTS["refit_per_trade"])
path_metrics = pd.read_csv(INPUTS["path_metrics"])
bar_0_20 = pd.read_csv(INPUTS["bar_0_20"])

trades_all["entry_time_dt"] = pd.to_datetime(trades_all["entry_time"])
trades_all["hour_of_day"] = trades_all["entry_time_dt"].dt.hour
trades_all["day_of_week"] = trades_all["entry_time_dt"].dt.day_name()


def session_from_hour(h):
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 13:
        return "london"
    if 13 <= h < 17:
        return "ny_overlap"
    return "ny_late"


trades_all["session"] = trades_all["hour_of_day"].apply(session_from_hour)

# Drop trades_all's step1-SL-only `final_r` and `exit_reason` to avoid clashing with §11 row 2 columns from path_metrics
trades_all_for_merge = trades_all.drop(columns=["final_r", "exit_reason", "exit_time", "exit_price", "bars_held", "spread_pips_exit"], errors="ignore")
master = trades_all_for_merge.merge(clusters, on="trade_id", how="left").merge(path_metrics, on="trade_id", how="left")
master = master.sort_values("trade_id").reset_index(drop=True)
refit_ids = set(refit["trade_id"].tolist())

# Slice definitions
slices_membership = {
    "A": master["trade_id"].tolist(),
    "B": master.loc[master["cluster_id"] == 1, "trade_id"].tolist(),
    "C": refit["trade_id"].tolist(),
    "D": master.loc[master["cluster_id"] != 1, "trade_id"].tolist(),
}
slice_master_idx = {k: master[master["trade_id"].isin(v)].index.to_numpy() for k, v in slices_membership.items()}

# Final_r source per slice
def slice_view(slice_name):
    if slice_name == "C":
        m = refit.copy().sort_values("trade_id").reset_index(drop=True)
        m = m.rename(columns={"peak_mfe_r": "peak_mfe_r_refit",
                              "exit_reason": "exit_reason_baseline",
                              "exit_bar": "bar_of_exit"})
        m = m.merge(clusters, on="trade_id", how="left")
        # Path metrics minus duplicates we already have; KEEP max_mae_r (2×ATR frame)
        pm = path_metrics.drop(columns=["final_r_s11r2", "exit_bar_s11r2", "exit_reason_s11r2",
                                        "giveback_r"],
                               errors="ignore")
        m = m.merge(pm, on="trade_id", how="left", suffixes=("", "_pm"))
        m = m.merge(trades_all[["trade_id", "pair", "atr_14_at_signal", "bar_range_at_signal",
                                "spread_pips_used", "sl_distance_pips", "hour_of_day",
                                "day_of_week", "session"]],
                    on="trade_id", how="left")
        # Canonical: refit final_r/exit; refit peak_mfe_r (3×ATR); path-metrics max_mae_r (2×ATR)
        m["peak_mfe_r"] = m["peak_mfe_r_refit"]
        return m
    else:
        m = master.copy()
        if slice_name == "B":
            m = m[m["cluster_id"] == 1]
        elif slice_name == "D":
            m = m[m["cluster_id"] != 1]
        m = m.rename(columns={"final_r_s11r2": "final_r", "exit_bar_s11r2": "bar_of_exit",
                              "exit_reason_s11r2": "exit_reason_baseline"})
        return m.reset_index(drop=True)


# For convenience: assemble per-trade-with-canonical-baseline frame for slices A/B/D
master_baseline = master.rename(columns={
    "final_r_s11r2": "final_r",
    "exit_bar_s11r2": "bar_of_exit",
    "exit_reason_s11r2": "exit_reason_baseline",
})

# Outcome bin assignment
def outcome_bin(final_r, max_mae_r):
    bins = []
    if final_r >= 1.5:
        bins.append(1)
    elif final_r >= 0.5:
        bins.append(2)
    elif final_r > -0.5:
        bins.append(3)
    elif final_r > -1.0:
        bins.append(4)
    else:
        bins.append(5)
    if final_r >= 0.5 and max_mae_r > -0.75:
        bins.append(6)
    return bins


def attach_bins(df):
    bins = df.apply(lambda r: outcome_bin(r["final_r"], r["max_mae_r"]), axis=1)
    df["bin_primary"] = bins.apply(lambda L: L[0])
    df["bin_t3_clean"] = bins.apply(lambda L: 6 in L).astype(int)
    return df


# Build slice views
sviews = {k: attach_bins(slice_view(k)) for k in ["A", "B", "C", "D"]}

# ============================================================
# A1: Trail extension after alive exit
# ============================================================

print("[info] A1: trail-extension analysis...")
a1_rows = []
for sname, df in sviews.items():
    # Trail-exit baseline: exit_reason in {trail, trail_breakeven, trail_hit}
    if sname == "C":
        trail_mask = df["exit_reason_baseline"].isin(["trail_hit", "trail"])
    else:
        trail_mask = df["exit_reason_baseline"].isin(["trail", "trail_breakeven"])
    trail_df = df[trail_mask].copy()
    if len(trail_df) == 0:
        continue
    # extension after exit
    if sname == "C":
        # Slice C peak_mfe_r is in 3*ATR frame; peak_mfe_r_full_window is in 2*ATR.
        # Convert peak_mfe_r (3*ATR) to 2*ATR for comparison.
        # In 3*ATR frame: peak_3atr = peak_price_move / (3*ATR). In 2*ATR: peak_2atr = peak_price_move / (2*ATR).
        # So peak_3atr * 1.5 = peak_2atr.
        trail_df = trail_df.copy()
        trail_df["peak_mfe_alive_2atr"] = trail_df["peak_mfe_r"].astype(float) * 1.5
        trail_df["extension_after_exit_r"] = trail_df["peak_mfe_r_full_window"] - trail_df["peak_mfe_alive_2atr"]
    else:
        trail_df["extension_after_exit_r"] = trail_df["peak_mfe_r_full_window"] - trail_df["peak_mfe_r"]

    # Per slice (overall)
    for cluster_id in [None, 0, 1, 2, 3]:
        if cluster_id is None:
            sub = trail_df
            cluster_label = "all"
        else:
            sub = trail_df[trail_df["cluster_id"] == cluster_id]
            cluster_label = f"cluster_{cluster_id}"
        if len(sub) == 0:
            continue
        ext = sub["extension_after_exit_r"].dropna().astype(float)
        if len(ext) == 0:
            continue
        for p in [10, 25, 50, 75, 90]:
            a1_rows.append({"slice": sname, "cluster": cluster_label,
                            "metric": "extension_after_exit_r",
                            "statistic": f"p{p}", "value": float(np.percentile(ext, p))})
        a1_rows.append({"slice": sname, "cluster": cluster_label,
                        "metric": "extension_after_exit_r", "statistic": "mean", "value": float(ext.mean())})
        a1_rows.append({"slice": sname, "cluster": cluster_label,
                        "metric": "extension_after_exit_r", "statistic": "std", "value": float(ext.std())})
        a1_rows.append({"slice": sname, "cluster": cluster_label,
                        "metric": "n_trail_exits", "statistic": "count", "value": int(len(sub))})
        a1_rows.append({"slice": sname, "cluster": cluster_label,
                        "metric": "pct_extended_ge_0.5R", "statistic": "fraction",
                        "value": float((ext >= 0.5).mean())})
        a1_rows.append({"slice": sname, "cluster": cluster_label,
                        "metric": "pct_extended_ge_1.0R", "statistic": "fraction",
                        "value": float((ext >= 1.0).mean())})
        # bars between alive-peak and full-peak — compute from path_metrics
        # Need bar_of_full_peak — we don't have this stored. Re-derive from path arrays.
        # For efficiency: compute mean bars-between for trades that extended >= 0.5R only,
        # using `peak_mfe_r_full_window` location which we'd need to recompute.
        # Skip for now; can be done from path arrays if requested.

a1_df = pd.DataFrame(a1_rows)
a1_df.to_csv(OUT_DIR / "a1_trail_extension.csv", index=False)
print(f"[done] a1_trail_extension.csv ({len(a1_df)} rows)")

# Now compute bars_between_alive_and_full_peak for trail extenders (slice B only for example)
# We need path arrays for this
print("[info] A1 bonus: computing bar_of_full_peak from path arrays...")
arr = load_path_arrays(OUT_DIR / "path_arrays.npz")
trade_id_arr = arr["trade_ids"]
mfe_mat = arr["mfe_so_far_r"]
# bar_of_full_peak per trade = first bar where mfe equals max
bar_of_full_peak = np.zeros(len(trade_id_arr), dtype=np.int32)
for i in range(len(trade_id_arr)):
    mfe_row = mfe_mat[i]
    valid = ~np.isnan(mfe_row)
    if valid.any():
        bar_of_full_peak[i] = int(np.argmax(mfe_row[valid]))
        # Note: mfe is running max so argmax over the row is the first bar
        # reaching the full peak
        # Use full row argmax (NaN is ignored via valid mask above; correct argmax:)
        nonnan_mfe = np.where(valid, mfe_row, -np.inf)
        bar_of_full_peak[i] = int(np.argmax(nonnan_mfe))
tid_to_bar_full = pd.Series(bar_of_full_peak, index=trade_id_arr, name="bar_of_full_peak")

# Compute bars_between for trail exits >= 0.5R extension
a1_bars_rows = []
for sname in ["A", "B", "C", "D"]:
    df = sviews[sname]
    if sname == "C":
        trail_mask = df["exit_reason_baseline"].isin(["trail_hit", "trail"])
    else:
        trail_mask = df["exit_reason_baseline"].isin(["trail", "trail_breakeven"])
    trail_df = df[trail_mask].copy()
    if sname == "C":
        trail_df["peak_mfe_alive_2atr"] = trail_df["peak_mfe_r"].astype(float) * 1.5
        trail_df["extension_after_exit_r"] = trail_df["peak_mfe_r_full_window"] - trail_df["peak_mfe_alive_2atr"]
    else:
        trail_df["extension_after_exit_r"] = trail_df["peak_mfe_r_full_window"] - trail_df["peak_mfe_r"]
    trail_df = trail_df.merge(tid_to_bar_full.reset_index().rename(columns={"index": "trade_id"}),
                              on="trade_id", how="left")
    trail_df["bars_between_alive_and_full_peak"] = trail_df["bar_of_full_peak"] - trail_df["bar_of_exit"]

    for cluster_id in [None, 0, 1, 2, 3]:
        if cluster_id is None:
            sub = trail_df[trail_df["extension_after_exit_r"] >= 0.5]
            clabel = "all_extenders_0.5R"
        else:
            sub = trail_df[(trail_df["cluster_id"] == cluster_id) & (trail_df["extension_after_exit_r"] >= 0.5)]
            clabel = f"cluster_{cluster_id}_extenders_0.5R"
        if len(sub) == 0:
            continue
        bb = sub["bars_between_alive_and_full_peak"].dropna().astype(float)
        if len(bb) == 0:
            continue
        a1_bars_rows.append({"slice": sname, "cluster": clabel,
                             "metric": "bars_between_alive_and_full_peak",
                             "statistic": "mean", "value": float(bb.mean())})
        for p in [25, 50, 75, 90]:
            a1_bars_rows.append({"slice": sname, "cluster": clabel,
                                 "metric": "bars_between_alive_and_full_peak",
                                 "statistic": f"p{p}", "value": float(np.percentile(bb, p))})

pd.DataFrame(a1_bars_rows).to_csv(OUT_DIR / "a1_bars_between_peaks.csv", index=False)
print("[done] a1_bars_between_peaks.csv")


# ============================================================
# A2: Pre vs post trail-lock path behaviour
# ============================================================

print("[info] A2: pre/post lock analysis...")

# Load full price data: high_r / low_r / close_r are in arr
close_mat = arr["close_r"]
N_trades = close_mat.shape[0]

def compute_path_window_metrics(close_row, start, end):
    """Compute path metrics over [start..end] (inclusive) of close_r row.
    end is the alive exit bar; window is [start..end] inclusive.
    Returns dict of metrics, or NaN dict if window too small.
    """
    if end < start or np.isnan(close_row[start]) or end - start < 1:
        return None
    window = close_row[start:end + 1]
    valid = ~np.isnan(window)
    if not valid.all():
        return None
    n = len(window)
    if n < 2:
        return None
    # Running max
    rmax = np.maximum.accumulate(window)
    # Pullbacks from running max
    pullbacks = rmax - window  # >= 0
    diffs = np.diff(window)
    abs_diffs = np.abs(diffs)
    var_diffs = np.var(diffs) if len(diffs) > 1 else np.nan
    # New highs
    new_high = (np.concatenate([[True], window[1:] > rmax[:-1]])).sum()
    # Local peaks
    local_peaks = 0
    if n >= 3:
        local_peaks = int(np.sum((window[1:-1] > window[:-2]) & (window[1:-1] > window[2:])))
    # Time within 0.25R of running max
    within_025 = float(np.mean(pullbacks <= 0.25))
    return {
        "n_bars": n,
        "pullback_mean": float(pullbacks.mean()),
        "pullback_median": float(np.median(pullbacks)),
        "pullback_max": float(pullbacks.max()),
        "n_new_highs": int(new_high),
        "local_peaks": local_peaks,
        "var_diff_close_r": float(var_diffs) if not np.isnan(var_diffs) else np.nan,
        "mean_abs_diff_close_r": float(abs_diffs.mean()),
        "time_within_0.25R_of_max": within_025,
    }


# For trail-triggered baseline trades, mfe_locked_bar is the lock bar
tid_to_idx_map = {int(t): i for i, t in enumerate(arr["trade_ids"])}

a2_rows = []
a2_paired_rows = []
pre_pull_for_slice_B = []
post_pull_for_slice_B = []
for sname, df in sviews.items():
    df_trail = df[df["trail_triggered"] == 1].copy() if "trail_triggered" in df.columns else df.iloc[0:0]
    if sname == "C":
        # slice C uses refit; trail_triggered not directly stored
        # use exit_reason='trail_hit' as proxy + mfe_locked_bar > -1 from path_metrics
        df_trail = df[df["exit_reason_baseline"] == "trail_hit"].copy()
        if "mfe_locked_bar" not in df_trail.columns:
            print(f"  [warn] slice C missing mfe_locked_bar — skipping A2 for C")
            continue
    if len(df_trail) == 0:
        continue
    for cluster_id in [None, 0, 1, 2, 3]:
        if cluster_id is None:
            sub = df_trail
            clabel = "all"
        else:
            sub = df_trail[df_trail["cluster_id"] == cluster_id]
            clabel = f"cluster_{cluster_id}"
        if len(sub) == 0:
            continue
        pre_rows = []
        post_rows = []
        for _, r in sub.iterrows():
            tid = int(r["trade_id"])
            if tid not in tid_to_idx_map:
                continue
            tidx = tid_to_idx_map[tid]
            lock_bar = int(r["mfe_locked_bar"]) if not pd.isna(r["mfe_locked_bar"]) else -1
            exit_b = int(r["bar_of_exit"])
            if lock_bar <= 0 or lock_bar > exit_b:
                continue
            close_row = close_mat[tidx]
            pre_m = compute_path_window_metrics(close_row, 0, lock_bar - 1)
            post_m = compute_path_window_metrics(close_row, lock_bar, exit_b)
            if pre_m and post_m:
                pre_rows.append(pre_m)
                post_rows.append(post_m)
                if sname == "B" and cluster_id == 1:
                    pre_pull_for_slice_B.append(pre_m["pullback_max"])
                    post_pull_for_slice_B.append(post_m["pullback_max"])
        if not pre_rows:
            continue
        pre_arr_df = pd.DataFrame(pre_rows)
        post_arr_df = pd.DataFrame(post_rows)
        for metric in ["pullback_mean", "pullback_median", "pullback_max", "n_new_highs", "local_peaks",
                       "var_diff_close_r", "mean_abs_diff_close_r", "time_within_0.25R_of_max"]:
            for window_name, source in [("pre_lock", pre_arr_df), ("post_lock", post_arr_df)]:
                col = source[metric].dropna()
                if len(col) == 0:
                    continue
                a2_rows.append({"slice": sname, "cluster": clabel, "window": window_name,
                                "metric": metric, "statistic": "mean", "value": float(col.mean())})
                a2_rows.append({"slice": sname, "cluster": clabel, "window": window_name,
                                "metric": metric, "statistic": "median", "value": float(col.median())})
                a2_rows.append({"slice": sname, "cluster": clabel, "window": window_name,
                                "metric": metric, "statistic": "std", "value": float(col.std())})
                a2_rows.append({"slice": sname, "cluster": clabel, "window": window_name,
                                "metric": metric, "statistic": "n", "value": int(len(col))})
            # Paired test
            pre_v = pre_arr_df[metric].dropna()
            post_v = post_arr_df[metric].dropna()
            common = pre_arr_df.index.intersection(post_arr_df.index)
            paired_pre = pre_arr_df.loc[common, metric].dropna()
            paired_post = post_arr_df.loc[common, metric].dropna()
            common2 = paired_pre.index.intersection(paired_post.index)
            paired_pre = paired_pre.loc[common2]
            paired_post = paired_post.loc[common2]
            if len(paired_pre) >= 5:
                t_stat, p_val = scistats.ttest_rel(paired_post, paired_pre)
                mean_diff = float((paired_post - paired_pre).mean())
                a2_paired_rows.append({"slice": sname, "cluster": clabel, "metric": metric,
                                       "n_paired": int(len(paired_pre)),
                                       "mean_diff_post_minus_pre": mean_diff,
                                       "t_stat": float(t_stat), "p_value": float(p_val)})

pd.DataFrame(a2_rows).to_csv(OUT_DIR / "a2_pre_post_lock.csv", index=False)
pd.DataFrame(a2_paired_rows).to_csv(OUT_DIR / "a2_paired_tests.csv", index=False)
print(f"[done] a2_pre_post_lock.csv ({len(a2_rows)}) + a2_paired_tests.csv ({len(a2_paired_rows)})")

# Plot for slice B cluster 1: pullback_max pre vs post
if pre_pull_for_slice_B and post_pull_for_slice_B:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.hist(pre_pull_for_slice_B, bins=40, alpha=0.5, label=f"pre-lock (N={len(pre_pull_for_slice_B)})", color="#2166ac")
    ax.hist(post_pull_for_slice_B, bins=40, alpha=0.5, label=f"post-lock (N={len(post_pull_for_slice_B)})", color="#b35806")
    ax.set_xlabel("max pullback magnitude (R, close-based)")
    ax.set_ylabel("count")
    ax.set_title("Slice B (cluster 1) — max pullback magnitude pre-lock vs post-lock")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "a2_pullback_distribution_pre_vs_post_slice_B.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("[done] a2_pullback_distribution_pre_vs_post_slice_B.png")


# ============================================================
# A3: Win/loss + profit factor metrics
# ============================================================

print("[info] A3: win/loss metrics...")

a3_rows = []
for sname, df in sviews.items():
    for cluster_id in [None, 0, 1, 2, 3]:
        if cluster_id is None:
            sub = df
            clabel = "all"
        else:
            sub = df[df["cluster_id"] == cluster_id]
            clabel = f"cluster_{cluster_id}"
        if len(sub) == 0:
            continue
        n = len(sub)
        bin1 = int((sub["bin_primary"] == 1).sum())
        bin2 = int((sub["bin_primary"] == 2).sum())
        bin5 = int((sub["bin_primary"] == 5).sum())
        bin6 = int(sub["bin_t3_clean"].sum())
        hit_rate = (bin1 + bin2 + bin6) / n if n else np.nan
        # Avoid double-counting bin6 with bin1/2 — bin6 is overlay. Hit rate uses unique trades with bin1∪2∪6
        unique_winners = sub[(sub["bin_primary"].isin([1, 2])) | (sub["bin_t3_clean"] == 1)]
        hit_rate = len(unique_winners) / n if n else np.nan
        win_loss = (bin1 + bin2) / bin5 if bin5 else np.nan
        mean_r = float(sub["final_r"].mean())
        pos = sub.loc[sub["final_r"] > 0, "final_r"].sum()
        neg = -sub.loc[sub["final_r"] < 0, "final_r"].sum()
        profit_factor = float(pos / neg) if neg > 0 else np.nan
        winners = sub[sub["bin_primary"].isin([1, 2])]
        winner_p50 = float(winners["final_r"].median()) if len(winners) else np.nan
        winner_p90 = float(winners["final_r"].quantile(0.9)) if len(winners) else np.nan
        a3_rows.append({"slice": sname, "cluster": clabel, "n": n,
                        "bin1": bin1, "bin2": bin2, "bin5": bin5, "bin6": bin6,
                        "hit_rate": hit_rate, "win_loss": win_loss,
                        "mean_r": mean_r, "profit_factor": profit_factor,
                        "winner_p50_r": winner_p50, "winner_p90_r": winner_p90})

pd.DataFrame(a3_rows).to_csv(OUT_DIR / "a3_win_loss_metrics.csv", index=False)
print(f"[done] a3_win_loss_metrics.csv ({len(a3_rows)} rows)")

# Plot: slice B trail-exit final_r distribution
sb = sviews["B"]
sb_trail = sb[sb["exit_reason_baseline"].isin(["trail", "trail_breakeven"])]
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
ax.hist(sb_trail["final_r"], bins=50, color="#1b7837", edgecolor="black", alpha=0.85)
ax.set_xlabel("final_r")
ax.set_ylabel("count")
ax.set_title(f"Slice B (cluster 1) trail exits — final_r distribution (N={len(sb_trail)})")
ax.axvline(0, color="black", linewidth=0.5)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(PLOT_DIR / "a3_winner_final_r_distribution_slice_B.png", dpi=150, facecolor="white")
plt.close(fig)
print("[done] a3 plot")


# ============================================================
# A4: Per-cluster deep characterisation
# ============================================================

print("[info] A4: per-cluster deep characterisation...")

PATH_SHAPE_METRICS = [
    "monotonicity_ratio_in_profit",
    "monotonicity_ratio_pre_peak",
    "local_peaks_count",
    "pullback_magnitude_median",
    "time_to_peak_mfe_relative",
    "velocity_first_t",
    "close_r_at_t1",
    "mae_so_far_r_at_t1",
    "mfe_so_far_r_at_t1",
]

a4_rows = []
# Use slice A (full population) for per-cluster characterisation
df_A = sviews["A"]
for cluster_id in [0, 1, 2, 3]:
    sub = df_A[df_A["cluster_id"] == cluster_id]
    n = len(sub)
    # Outcome
    bin1 = int((sub["bin_primary"] == 1).sum())
    bin2 = int((sub["bin_primary"] == 2).sum())
    bin5 = int((sub["bin_primary"] == 5).sum())
    bin6 = int(sub["bin_t3_clean"].sum())
    win_loss = (bin1 + bin2) / bin5 if bin5 else np.nan
    pos = sub.loc[sub["final_r"] > 0, "final_r"].sum()
    neg = -sub.loc[sub["final_r"] < 0, "final_r"].sum()
    pf = pos / neg if neg > 0 else np.nan
    a4_rows.extend([
        {"cluster": cluster_id, "metric_group": "outcome", "metric": "n", "value": n},
        {"cluster": cluster_id, "metric_group": "outcome", "metric": "bin1_share", "value": bin1 / n if n else np.nan},
        {"cluster": cluster_id, "metric_group": "outcome", "metric": "bin2_share", "value": bin2 / n if n else np.nan},
        {"cluster": cluster_id, "metric_group": "outcome", "metric": "bin5_share", "value": bin5 / n if n else np.nan},
        {"cluster": cluster_id, "metric_group": "outcome", "metric": "bin6_share", "value": bin6 / n if n else np.nan},
        {"cluster": cluster_id, "metric_group": "outcome", "metric": "win_loss", "value": win_loss},
        {"cluster": cluster_id, "metric_group": "outcome", "metric": "mean_r", "value": sub["final_r"].mean()},
        {"cluster": cluster_id, "metric_group": "outcome", "metric": "profit_factor", "value": pf},
    ])
    # Path-shape means
    for m in PATH_SHAPE_METRICS:
        if m in sub.columns:
            a4_rows.append({"cluster": cluster_id, "metric_group": "path_shape", "metric": m, "value": float(sub[m].mean())})
    # Bar-1/2/3 MAE for winners vs losers
    winners_ids = set(sub[sub["bin_primary"].isin([1, 2])]["trade_id"].tolist())
    losers_ids = set(sub[sub["bin_primary"] == 5]["trade_id"].tolist())
    for bar in [1, 2, 3]:
        for label, ids in [("winners", winners_ids), ("losers", losers_ids)]:
            bd = bar_0_20[(bar_0_20["bar_offset"] == bar) & (bar_0_20["trade_id"].isin(ids))]
            if len(bd):
                a4_rows.append({"cluster": cluster_id, "metric_group": f"mae_bar{bar}_{label}",
                                "metric": "mean", "value": float(bd["mae_so_far_r"].mean())})
                a4_rows.append({"cluster": cluster_id, "metric_group": f"mae_bar{bar}_{label}",
                                "metric": "p90", "value": float(bd["mae_so_far_r"].quantile(0.9))})
                a4_rows.append({"cluster": cluster_id, "metric_group": f"mae_bar{bar}_{label}",
                                "metric": "n", "value": int(len(bd))})
    # Exit reason
    for er, cnt in sub["exit_reason_baseline"].value_counts().items():
        a4_rows.append({"cluster": cluster_id, "metric_group": "exit_reason", "metric": str(er),
                        "value": int(cnt)})
        er_sub = sub[sub["exit_reason_baseline"] == er]
        a4_rows.append({"cluster": cluster_id, "metric_group": "exit_reason_final_r_mean",
                        "metric": str(er), "value": float(er_sub["final_r"].mean())})
        a4_rows.append({"cluster": cluster_id, "metric_group": "exit_reason_peak_mean",
                        "metric": str(er), "value": float(er_sub["peak_mfe_r"].mean())})
    # Entry-time
    if "atr_14_at_signal" in sub.columns:
        a4_rows.append({"cluster": cluster_id, "metric_group": "entry", "metric": "mean_atr_14", "value": float(sub["atr_14_at_signal"].mean())})
        a4_rows.append({"cluster": cluster_id, "metric_group": "entry", "metric": "mean_bar_range", "value": float(sub["bar_range_at_signal"].mean())})
        a4_rows.append({"cluster": cluster_id, "metric_group": "entry", "metric": "mean_spread", "value": float(sub["spread_pips_used"].mean())})
        top_pair = sub["pair"].value_counts().head(3).to_dict()
        for p, c in top_pair.items():
            a4_rows.append({"cluster": cluster_id, "metric_group": "entry", "metric": f"top_pair_{p}", "value": int(c)})
        top_hour = sub["hour_of_day"].value_counts().head(3).to_dict()
        for h, c in top_hour.items():
            a4_rows.append({"cluster": cluster_id, "metric_group": "entry", "metric": f"top_hour_{int(h):02d}", "value": int(c)})
        top_sess = sub["session"].value_counts().head(2).to_dict()
        for s, c in top_sess.items():
            a4_rows.append({"cluster": cluster_id, "metric_group": "entry", "metric": f"top_session_{s}", "value": int(c)})

pd.DataFrame(a4_rows).to_csv(OUT_DIR / "a4_per_cluster_metrics.csv", index=False)
print(f"[done] a4_per_cluster_metrics.csv ({len(a4_rows)} rows)")

# Per-cluster MAE winners vs losers plot
for cluster_id in [0, 1, 2, 3]:
    sub = df_A[df_A["cluster_id"] == cluster_id]
    winners_ids = set(sub[sub["bin_primary"].isin([1, 2])]["trade_id"].tolist())
    losers_ids = set(sub[sub["bin_primary"] == 5]["trade_id"].tolist())
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    for label, ids, colour in [("winners (bins 1+2)", winners_ids, "#1b7837"), ("losers (bin 5)", losers_ids, "#b35806")]:
        if not ids:
            continue
        bsub = bar_0_20[bar_0_20["trade_id"].isin(ids)]
        agg = bsub.groupby("bar_offset")["mae_so_far_r"].agg(["mean", lambda s: s.quantile(0.9)])
        agg.columns = ["mean", "p90"]
        ax.plot(agg.index, agg["mean"], label=f"{label} mean (N={len(ids)})", color=colour, linewidth=2)
        ax.plot(agg.index, agg["p90"], label=f"{label} p90", color=colour, linestyle="--", alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("bar_offset")
    ax.set_ylabel("MAE (R, 2×ATR)")
    ax.set_title(f"Cluster {cluster_id} — winners vs losers MAE by bar")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"a4_cluster_{cluster_id}_mae_winners_vs_losers.png", dpi=150, facecolor="white")
    plt.close(fig)
print("[done] a4 plots")


# ============================================================
# A5: Winners vs losers (outcome groups)
# ============================================================

print("[info] A5: winners vs losers (outcome groups)...")

a5_rows = []
PATH_AND_ENTRY = PATH_SHAPE_METRICS + ["atr_14_at_signal", "bar_range_at_signal", "spread_pips_used", "sl_distance_pips"]

for sname, df in sviews.items():
    winners = df[df["bin_primary"].isin([1, 2])]
    losers = df[df["bin_primary"] == 5]
    if len(winners) < 5 or len(losers) < 5:
        continue
    for m in PATH_AND_ENTRY:
        if m not in df.columns:
            continue
        w = winners[m].dropna()
        l_ = losers[m].dropna()
        if len(w) < 5 or len(l_) < 5:
            continue
        t_stat, p_val = scistats.ttest_ind(w, l_, equal_var=False)
        a5_rows.append({"slice": sname, "metric": m,
                        "winner_n": int(len(w)), "winner_mean": float(w.mean()), "winner_std": float(w.std()),
                        "loser_n": int(len(l_)), "loser_mean": float(l_.mean()), "loser_std": float(l_.std()),
                        "mean_diff_winner_minus_loser": float(w.mean() - l_.mean()),
                        "t_stat": float(t_stat), "p_value": float(p_val)})
    # Cluster composition
    for cid in [0, 1, 2, 3]:
        w_pct = (winners["cluster_id"] == cid).mean() if len(winners) else np.nan
        l_pct = (losers["cluster_id"] == cid).mean() if len(losers) else np.nan
        a5_rows.append({"slice": sname, "metric": f"cluster_{cid}_share",
                        "winner_n": int(len(winners)), "winner_mean": float(w_pct), "winner_std": np.nan,
                        "loser_n": int(len(losers)), "loser_mean": float(l_pct), "loser_std": np.nan,
                        "mean_diff_winner_minus_loser": float(w_pct - l_pct),
                        "t_stat": np.nan, "p_value": np.nan})
    # Bar-1/2/3 MAE/MFE
    for bar in [1, 2, 3]:
        for metric_col in ["mae_so_far_r", "mfe_so_far_r", "close_r"]:
            wb = bar_0_20[(bar_0_20["bar_offset"] == bar) & (bar_0_20["trade_id"].isin(winners["trade_id"]))][metric_col].dropna()
            lb = bar_0_20[(bar_0_20["bar_offset"] == bar) & (bar_0_20["trade_id"].isin(losers["trade_id"]))][metric_col].dropna()
            if len(wb) < 5 or len(lb) < 5:
                continue
            t_stat, p_val = scistats.ttest_ind(wb, lb, equal_var=False)
            a5_rows.append({"slice": sname, "metric": f"bar{bar}_{metric_col}",
                            "winner_n": int(len(wb)), "winner_mean": float(wb.mean()), "winner_std": float(wb.std()),
                            "loser_n": int(len(lb)), "loser_mean": float(lb.mean()), "loser_std": float(lb.std()),
                            "mean_diff_winner_minus_loser": float(wb.mean() - lb.mean()),
                            "t_stat": float(t_stat), "p_value": float(p_val)})

pd.DataFrame(a5_rows).to_csv(OUT_DIR / "a5_winners_vs_losers.csv", index=False)
print(f"[done] a5_winners_vs_losers.csv ({len(a5_rows)} rows)")

print("[done] Phase A complete")
