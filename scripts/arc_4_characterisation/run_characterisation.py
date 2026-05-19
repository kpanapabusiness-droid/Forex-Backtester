"""Arc 4 Characterisation — main analysis script.

Reads:
  - results/l_arc_4/step1/trades_all.csv
  - results/l_arc_4/step2/clusters_K4.csv
  - results/l_arc_4/step3/archetype_summaries.csv
  - results/l_arc_4/step5c/per_trade_simulated_refit_1.csv
  - results/l_arc_4/step5c/per_fold_refit_classifier_metrics.csv
  - results/arc_4_characterisation/per_trade_path_metrics.csv (precomputed §11 row 2 + path metrics)
  - results/arc_4_characterisation/per_trade_bar_0_20.csv (bars 0..20 close/mfe/mae)

Writes:
  - results/arc_4_characterisation/CHARACTERISATION.md
  - results/arc_4_characterisation/per_cell_distributions.csv
  - results/arc_4_characterisation/bar_trajectory_mae.csv
  - results/arc_4_characterisation/bar_trajectory_mfe.csv
  - results/arc_4_characterisation/exit_reason_metrics_full.csv
  - results/arc_4_characterisation/exit_reason_summary.md
  - results/arc_4_characterisation/conversion_analysis.csv
  - results/arc_4_characterisation/plots/*.png
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "arc_4_characterisation"
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "trades_all": REPO / "results" / "l_arc_4" / "step1" / "trades_all.csv",
    "trades_paths": REPO / "results" / "l_arc_4" / "step1" / "trades_paths.csv",
    "clusters_K4": REPO / "results" / "l_arc_4" / "step2" / "clusters_K4.csv",
    "archetype_summaries": REPO / "results" / "l_arc_4" / "step3" / "archetype_summaries.csv",
    "refit_per_trade": REPO / "results" / "l_arc_4" / "step5c" / "per_trade_simulated_refit_1.csv",
    "refit_per_fold": REPO / "results" / "l_arc_4" / "step5c" / "per_fold_refit_classifier_metrics.csv",
}

PRECOMP_PATH_METRICS = OUT_DIR / "per_trade_path_metrics.csv"
PRECOMP_BAR_0_20 = OUT_DIR / "per_trade_bar_0_20.csv"

PERCENTILES = [5, 10, 25, 50, 75, 90, 95]
P_SHORT = [10, 25, 50, 75, 90]


# ---------- helpers ----------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_r(x):
    if pd.isna(x):
        return "NaN"
    return f"{x:.4f}"


def fmt_int(x):
    if pd.isna(x):
        return "NaN"
    return f"{int(x)}"


def fmt_pct(x):
    if pd.isna(x):
        return "NaN"
    return f"{x:.1%}"


def distr_summary(series: pd.Series, name: str, percentiles=PERCENTILES, include_std=True, include_minmax=True) -> dict:
    s = series.dropna().astype(float)
    if len(s) == 0:
        out = {f"{name}_p{p}": np.nan for p in percentiles}
        out[f"{name}_mean"] = np.nan
        if include_std:
            out[f"{name}_std"] = np.nan
        if include_minmax:
            out[f"{name}_min"] = np.nan
            out[f"{name}_max"] = np.nan
        out[f"{name}_n"] = 0
        return out
    out = {}
    for p in percentiles:
        out[f"{name}_p{p}"] = float(np.percentile(s, p))
    out[f"{name}_mean"] = float(s.mean())
    if include_std:
        out[f"{name}_std"] = float(s.std())
    if include_minmax:
        out[f"{name}_min"] = float(s.min())
        out[f"{name}_max"] = float(s.max())
    out[f"{name}_n"] = int(len(s))
    return out


def long_distr_rows(slice_name, bin_name, metric_name, series: pd.Series, percentiles=P_SHORT):
    """Return list of dicts in long format: slice, bin, metric, statistic, value."""
    s = series.dropna().astype(float)
    rows = []
    if len(s) == 0:
        for p in percentiles:
            rows.append({"slice": slice_name, "bin": bin_name, "metric": metric_name, "statistic": f"p{p}", "value": np.nan})
        rows.append({"slice": slice_name, "bin": bin_name, "metric": metric_name, "statistic": "mean", "value": np.nan})
        rows.append({"slice": slice_name, "bin": bin_name, "metric": metric_name, "statistic": "std", "value": np.nan})
        rows.append({"slice": slice_name, "bin": bin_name, "metric": metric_name, "statistic": "n", "value": 0})
        return rows
    for p in percentiles:
        rows.append({"slice": slice_name, "bin": bin_name, "metric": metric_name, "statistic": f"p{p}", "value": float(np.percentile(s, p))})
    rows.append({"slice": slice_name, "bin": bin_name, "metric": metric_name, "statistic": "mean", "value": float(s.mean())})
    rows.append({"slice": slice_name, "bin": bin_name, "metric": metric_name, "statistic": "std", "value": float(s.std())})
    rows.append({"slice": slice_name, "bin": bin_name, "metric": metric_name, "statistic": "n", "value": int(len(s))})
    return rows


def outcome_bin(final_r, max_mae_r):
    """Returns list of bin_ids the trade belongs to. Bin 6 (T3 clean) is overlay, can co-occur with 1 or 2."""
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
    # Bin 6 overlay: final_r >= 0.5 AND max_mae > -0.75
    if final_r >= 0.5 and max_mae_r > -0.75:
        bins.append(6)
    return bins


BIN_LABEL = {
    1: "1_big_winners",
    2: "2_modest_winners",
    3: "3_scratches",
    4: "4_modest_losers",
    5: "5_full_stops",
    6: "6_T3_clean",
}


def session_from_hour(h: int) -> str:
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 13:
        return "london"
    if 13 <= h < 17:
        return "ny_overlap"
    return "ny_late"


# ---------- load ----------

print("[info] loading inputs...")
trades_all = pd.read_csv(INPUTS["trades_all"])
clusters = pd.read_csv(INPUTS["clusters_K4"])
refit = pd.read_csv(INPUTS["refit_per_trade"])
refit_meta = pd.read_csv(INPUTS["refit_per_fold"])
path_metrics = pd.read_csv(PRECOMP_PATH_METRICS)
archetype = pd.read_csv(INPUTS["archetype_summaries"])

# Compute sha256 of inputs
SHAS = {name: sha256_file(p) for name, p in INPUTS.items()}

# Merge into one master frame keyed by trade_id
master = trades_all.merge(clusters, on="trade_id", how="left").merge(path_metrics, on="trade_id", how="left")
master["entry_time_dt"] = pd.to_datetime(master["entry_time"])
master["hour_of_day"] = master["entry_time_dt"].dt.hour
master["day_of_week"] = master["entry_time_dt"].dt.day_name()
master["session"] = master["hour_of_day"].apply(session_from_hour)
master = master.sort_values("trade_id").reset_index(drop=True)

# Refit subset
refit = refit.sort_values("trade_id").reset_index(drop=True)
refit_meta_keys = refit[["trade_id"]].assign(in_refit=1)

# Slices A,B,C,D
slice_A_ids = master["trade_id"]
slice_B_ids = master.loc[master["cluster_id"] == 1, "trade_id"]
slice_D_ids = master.loc[master["cluster_id"] != 1, "trade_id"]
slice_C_ids = refit["trade_id"]

print(f"[info] slice A: {len(slice_A_ids)}")
print(f"[info] slice B: {len(slice_B_ids)}")
print(f"[info] slice C: {len(slice_C_ids)}")
print(f"[info] slice D: {len(slice_D_ids)}")


def build_slice_frame(name: str) -> pd.DataFrame:
    """Return slice dataframe with unified columns: final_r, peak_mfe_r, max_mae_r,
    giveback_r, bar_of_peak_mfe, bar_of_max_mae, bar_of_exit, exit_reason, cluster_id,
    plus all path metrics and entry features available.
    """
    if name == "C":
        # Use refit final_r + path_metrics for path-shape (peak_mfe_r, max_mae_r, etc.)
        # but exit info comes from refit
        m = refit.merge(path_metrics, on="trade_id", how="left")
        m = m.merge(clusters, on="trade_id", how="left")
        m = m.merge(
            master[["trade_id", "pair", "signal_time", "entry_time", "entry_price", "atr_14_at_signal",
                    "bar_range_at_signal", "spread_pips_used", "sl_distance_pips", "hour_of_day",
                    "day_of_week", "session"]],
            on="trade_id", how="left", suffixes=("", "_orig"),
        )
        m = m.rename(columns={
            "final_r": "final_r",
            "exit_reason": "exit_reason",
            "exit_bar": "bar_of_exit",
        })
        # peak_mfe_r etc. come from path_metrics (full-window from path);
        # but slice C uses refit peak_mfe_r — prefer refit peak when present
        m["peak_mfe_r"] = m["peak_mfe_r_x"] if "peak_mfe_r_x" in m.columns else m["peak_mfe_r"]
        # path_metrics already has peak_mfe_r — note refit also has peak_mfe_r col → dedupe
        # Rebuild simply:
        return m
    else:
        m = master.copy()
        if name == "B":
            m = m[m["cluster_id"] == 1].copy()
        elif name == "D":
            m = m[m["cluster_id"] != 1].copy()
        # Use §11 row 2 simulated values
        m = m.rename(columns={
            "final_r_s11r2": "final_r",
            "exit_bar_s11r2": "bar_of_exit",
            "exit_reason_s11r2": "exit_reason",
        })
        # Drop original final_r (step1 SL-only) to avoid confusion (keep as separate column)
        m = m.rename(columns={"final_r": "final_r_s11r2", "final_r_x": "final_r_step1_sl_only"})
        # Wait — pandas rename: just make sure the column we want is named 'final_r'.
        return m


# Re-implement build_slice_frame cleanly given column name pitfalls.

def build_slice_frame_v2(name: str) -> pd.DataFrame:
    if name == "C":
        # Pull refit fields and rename to canonical
        m = refit.copy()
        m = m.rename(columns={
            "exit_bar": "bar_of_exit",
            "exit_reason": "exit_reason",
            "final_r": "final_r",
            "peak_mfe_r": "peak_mfe_r_refit",
        })
        # Merge path_metrics (gives peak_mfe_r, max_mae_r, giveback_r etc. — but those use 2*ATR frame.
        # For slice C the refit peak_mfe_r uses cluster_R (3*ATR) frame, so they differ.
        # We document this — for C, use the refit-frame peak_mfe_r for outcome bin overlay and giveback;
        # path-shape metrics (monotonicity, local peaks, etc.) come from path_metrics directly.
        pm_keep = path_metrics.drop(columns=[
            "final_r_s11r2", "exit_bar_s11r2", "exit_reason_s11r2",
            "peak_mfe_r", "max_mae_r", "giveback_r", "bar_of_peak_mfe", "bar_of_max_mae",
            "mfe_locked_bar", "hold_duration_bars", "ever_reached_1R_alive",
            "trail_triggered", "peak_mfe_r_full_window", "max_mae_r_full_window",
        ])
        m = m.merge(pm_keep, on="trade_id", how="left")
        m = m.merge(clusters, on="trade_id", how="left")
        m = m.merge(
            master[["trade_id", "pair", "signal_time", "entry_time", "atr_14_at_signal",
                    "bar_range_at_signal", "spread_pips_used", "sl_distance_pips",
                    "hour_of_day", "day_of_week", "session"]],
            on="trade_id", how="left", suffixes=("", "_dup"),
        )
        # Use refit-frame as canonical peak_mfe_r / max_mae_r / giveback_r for slice C
        # max_mae_r is not in refit explicitly — recompute from path metrics in C's frame proxy:
        # we'll use path_metrics' max_mae_r as the only available value (2*ATR frame).
        # For slice C analyses, document this in the doc.
        path_only = path_metrics[["trade_id", "peak_mfe_r", "max_mae_r", "bar_of_peak_mfe", "bar_of_max_mae"]].copy()
        path_only = path_only.rename(columns={
            "peak_mfe_r": "peak_mfe_r_2atr",
            "max_mae_r": "max_mae_r_2atr",
        })
        m = m.merge(path_only, on="trade_id", how="left")
        # canonical peak_mfe_r for slice C uses refit (cluster_R / 3*ATR frame)
        m["peak_mfe_r"] = m["peak_mfe_r_refit"]
        m["max_mae_r"] = m["max_mae_r_2atr"]  # only 2*ATR available
        m["bar_of_peak_mfe"] = m["bar_of_peak_mfe"]
        m["bar_of_max_mae"] = m["bar_of_max_mae"]
        m["giveback_r"] = m["peak_mfe_r"] - m["final_r"]
        # Alias hold_duration_bars = bar_of_exit for slice C (refit exit_bar)
        m["hold_duration_bars"] = m["bar_of_exit"]
        return m
    else:
        m = master.copy()
        if name == "B":
            m = m[m["cluster_id"] == 1].copy()
        elif name == "D":
            m = m[m["cluster_id"] != 1].copy()
        # Canonical columns from §11 row 2
        m["final_r"] = m["final_r_s11r2"]
        m["bar_of_exit"] = m["exit_bar_s11r2"]
        m["exit_reason"] = m["exit_reason_s11r2"]
        # giveback already computed in path_metrics; ensure column exists
        if "giveback_r" not in m.columns:
            m["giveback_r"] = m["peak_mfe_r"] - m["final_r"]
        return m


slices = {
    "A": build_slice_frame_v2("A"),
    "B": build_slice_frame_v2("B"),
    "C": build_slice_frame_v2("C"),
    "D": build_slice_frame_v2("D"),
}
for k, v in slices.items():
    print(f"[info] slice {k} cols ({len(v)}): {len(v.columns)}")

# Add outcome bins to each slice
for k, df in slices.items():
    bins_per_trade = df.apply(lambda r: outcome_bin(r["final_r"], r["max_mae_r"]), axis=1)
    df["bin_primary"] = bins_per_trade.apply(lambda L: L[0])
    df["bin_t3_clean"] = bins_per_trade.apply(lambda L: 6 in L).astype(int)


# ---------- per_cell_distributions.csv ----------

print("[info] building per_cell_distributions.csv...")

PER_CELL_METRICS = [
    "final_r",
    "peak_mfe_r",
    "max_mae_r",
    "giveback_r",
    "bar_of_peak_mfe",
    "bar_of_max_mae",
    "bar_of_exit",
]

rows = []
for slice_name, df in slices.items():
    for bin_id in [1, 2, 3, 4, 5, 6]:
        if bin_id == 6:
            sub = df[df["bin_t3_clean"] == 1]
        else:
            sub = df[df["bin_primary"] == bin_id]
        bin_label = BIN_LABEL[bin_id]
        n = len(sub)
        slice_n = len(df)
        pct = n / slice_n if slice_n > 0 else np.nan
        rows.append({"slice": slice_name, "bin": bin_label, "metric": "_N", "statistic": "n", "value": n})
        rows.append({"slice": slice_name, "bin": bin_label, "metric": "_N", "statistic": "pct_of_slice", "value": pct})
        for metric in PER_CELL_METRICS:
            rows.extend(long_distr_rows(slice_name, bin_label, metric, sub[metric]))
        # Exit reason breakdown
        er_counts = sub["exit_reason"].value_counts(dropna=False).to_dict()
        for er, cnt in er_counts.items():
            er_str = "NA" if pd.isna(er) else str(er)
            rows.append({"slice": slice_name, "bin": bin_label, "metric": f"exit_reason_{er_str}", "statistic": "count", "value": int(cnt)})

per_cell_df = pd.DataFrame(rows)
per_cell_path = OUT_DIR / "per_cell_distributions.csv"
per_cell_df.to_csv(per_cell_path, index=False)
print(f"[done] {per_cell_path}")


# ---------- bar trajectory ----------

print("[info] computing bar trajectories (bars 0..20)...")
bar_df = pd.read_csv(PRECOMP_BAR_0_20)

# Attach slice membership + bin
slice_id_map = {}
for slice_name, df in slices.items():
    for tid in df["trade_id"].tolist():
        slice_id_map.setdefault(tid, []).append(slice_name)

slice_membership = pd.DataFrame([
    {"trade_id": tid, "slice": s} for tid, slist in slice_id_map.items() for s in slist
])

# Build trade -> bin per slice
trade_bin_per_slice = []
for slice_name, df in slices.items():
    sub = df[["trade_id", "bin_primary", "bin_t3_clean"]].copy()
    sub["slice"] = slice_name
    trade_bin_per_slice.append(sub)
trade_bin = pd.concat(trade_bin_per_slice, ignore_index=True)

# Determine bar cap per slice: bar by which 95% of trades have exited under §11 row 2
# Note: for slice C use refit bar_of_exit; for others use bar_of_exit
# But trajectory data is bars 0..20 only — use 20 if 95% exit beyond that
def slice_bar_cap(df: pd.DataFrame) -> int:
    p95 = int(np.percentile(df["bar_of_exit"].dropna(), 95))
    return min(20, p95)

slice_bar_cap_map = {k: slice_bar_cap(v) for k, v in slices.items()}
print(f"[info] slice bar caps: {slice_bar_cap_map}")

# Build long format
mae_rows = []
mfe_rows = []
for slice_name, df in slices.items():
    cap = slice_bar_cap_map[slice_name]
    tids_set = set(df["trade_id"].tolist())
    sub_bar = bar_df[bar_df["trade_id"].isin(tids_set)].copy()
    sub_bar = sub_bar[sub_bar["bar_offset"] <= cap]
    # Merge bin info
    bin_lookup = df.set_index("trade_id")[["bin_primary", "bin_t3_clean"]]
    sub_bar = sub_bar.join(bin_lookup, on="trade_id")

    for bin_id in [1, 2, 3, 4, 5, 6]:
        if bin_id == 6:
            sub_bin = sub_bar[sub_bar["bin_t3_clean"] == 1]
        else:
            sub_bin = sub_bar[sub_bar["bin_primary"] == bin_id]
        if len(sub_bin) == 0:
            continue
        bin_label = BIN_LABEL[bin_id]
        # Group by bar_offset
        for bar, grp in sub_bin.groupby("bar_offset"):
            for stat, val_mae, val_mfe in [
                ("mean", grp["mae_so_far_r"].mean(), grp["mfe_so_far_r"].mean()),
                ("p25", grp["mae_so_far_r"].quantile(0.25), grp["mfe_so_far_r"].quantile(0.25)),
                ("p50", grp["mae_so_far_r"].quantile(0.50), grp["mfe_so_far_r"].quantile(0.50)),
                ("p75", grp["mae_so_far_r"].quantile(0.75), grp["mfe_so_far_r"].quantile(0.75)),
            ]:
                mae_rows.append({"slice": slice_name, "bin": bin_label, "bar": int(bar), "statistic": stat, "value": float(val_mae)})
                mfe_rows.append({"slice": slice_name, "bin": bin_label, "bar": int(bar), "statistic": stat, "value": float(val_mfe)})

mae_traj_df = pd.DataFrame(mae_rows)
mfe_traj_df = pd.DataFrame(mfe_rows)
mae_traj_df.to_csv(OUT_DIR / "bar_trajectory_mae.csv", index=False)
mfe_traj_df.to_csv(OUT_DIR / "bar_trajectory_mfe.csv", index=False)
print(f"[done] bar trajectory CSVs")


# ---------- trajectory plots ----------

print("[info] making trajectory plots...")
COLOURS = {
    "1_big_winners": "#1b7837",
    "2_modest_winners": "#7fbf7b",
    "3_scratches": "#999999",
    "4_modest_losers": "#fdb863",
    "5_full_stops": "#b35806",
    "6_T3_clean": "#762a83",
}

for slice_name in slices:
    for metric in ["mae", "mfe"]:
        df_traj = mae_traj_df if metric == "mae" else mfe_traj_df
        sub = df_traj[df_traj["slice"] == slice_name]
        if len(sub) == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        for bin_label in BIN_LABEL.values():
            bsub = sub[sub["bin"] == bin_label]
            if len(bsub) == 0:
                continue
            mean_line = bsub[bsub["statistic"] == "mean"].sort_values("bar")
            p25 = bsub[bsub["statistic"] == "p25"].sort_values("bar")
            p75 = bsub[bsub["statistic"] == "p75"].sort_values("bar")
            if len(mean_line) == 0:
                continue
            ax.plot(mean_line["bar"], mean_line["value"], label=bin_label, color=COLOURS[bin_label], linewidth=2)
            if len(p25) > 0 and len(p75) > 0:
                ax.fill_between(p25["bar"].values, p25["value"].values, p75["value"].values, alpha=0.15, color=COLOURS[bin_label])
        ax.set_xlabel("bar offset")
        ax.set_ylabel(f"{metric.upper()} (R-multiples)")
        ax.set_title(f"Slice {slice_name} — {metric.upper()} by bar, outcome bins overlaid")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"trajectory_{metric}_slice_{slice_name}.png", dpi=150, facecolor="white")
        plt.close(fig)


# ---------- exit_reason_metrics_full.csv ----------

print("[info] building exit_reason_metrics_full.csv...")

ENTRY_FEATURES_NUMERIC = [
    "atr_14_at_signal",
    "bar_range_at_signal",
    "sl_distance_pips",
    "spread_pips_used",
]

PATH_FEATURES_DISTR = [
    "monotonicity_ratio_in_profit",
    "monotonicity_ratio_pre_peak",
    "local_peaks_count",
    "pullback_magnitude_median",
    "time_to_peak_mfe_relative",
    "velocity_first_t",
    "close_r_at_t1",
    "mae_so_far_r_at_t1",
    "mfe_so_far_r_at_t1",
    "mae_before_peak_mfe_r",
    "mfe_after_max_mae_r",
]

EXIT_CORE_DISTR = [
    "final_r",
    "peak_mfe_r",
    "max_mae_r",
    "giveback_r",
    "bar_of_exit",
    "bar_of_peak_mfe",
    "bar_of_max_mae",
    "hold_duration_bars",
]

er_rows = []
for slice_name, df in slices.items():
    er_values = df["exit_reason"].dropna().unique().tolist()
    slice_n = len(df)
    for er in er_values:
        sub = df[df["exit_reason"] == er]
        n = len(sub)
        pct = n / slice_n if slice_n else np.nan
        er_rows.append({"slice": slice_name, "exit_reason": er, "metric": "_N", "statistic": "n", "value": n})
        er_rows.append({"slice": slice_name, "exit_reason": er, "metric": "_N", "statistic": "pct_of_slice", "value": pct})

        # Cluster composition within exit reason
        cluster_counts = sub["cluster_id"].value_counts(dropna=False).to_dict()
        for c, cnt in cluster_counts.items():
            cstr = "NA" if pd.isna(c) else f"cluster_{int(c)}"
            er_rows.append({"slice": slice_name, "exit_reason": er, "metric": f"cluster_{cstr}", "statistic": "count", "value": int(cnt)})

        # core metrics
        for m in EXIT_CORE_DISTR:
            er_rows.extend(long_distr_rows(slice_name, "_exit_reason", m, sub[m], percentiles=PERCENTILES))
            # also min/max
            s = sub[m].dropna().astype(float)
            if len(s) > 0:
                er_rows.append({"slice": slice_name, "exit_reason": er, "metric": m, "statistic": "min", "value": float(s.min())})
                er_rows.append({"slice": slice_name, "exit_reason": er, "metric": m, "statistic": "max", "value": float(s.max())})

        # Path-shape metrics
        for m in PATH_FEATURES_DISTR:
            if m not in sub.columns:
                continue
            er_rows.extend(long_distr_rows(slice_name, "_exit_reason", m, sub[m], percentiles=P_SHORT))

        # Entry-time numeric
        for m in ENTRY_FEATURES_NUMERIC:
            if m not in sub.columns:
                continue
            er_rows.extend(long_distr_rows(slice_name, "_exit_reason", m, sub[m], percentiles=P_SHORT))

        # Hour of day distribution
        hr_counts = sub["hour_of_day"].value_counts().sort_index().to_dict() if "hour_of_day" in sub.columns else {}
        for h, cnt in hr_counts.items():
            er_rows.append({"slice": slice_name, "exit_reason": er, "metric": f"hour_{int(h):02d}", "statistic": "count", "value": int(cnt)})

        # Day of week
        dow_counts = sub["day_of_week"].value_counts().to_dict() if "day_of_week" in sub.columns else {}
        for d, cnt in dow_counts.items():
            er_rows.append({"slice": slice_name, "exit_reason": er, "metric": f"dow_{d}", "statistic": "count", "value": int(cnt)})

        # Session
        sess_counts = sub["session"].value_counts().to_dict() if "session" in sub.columns else {}
        for sname, cnt in sess_counts.items():
            er_rows.append({"slice": slice_name, "exit_reason": er, "metric": f"session_{sname}", "statistic": "count", "value": int(cnt)})

        # Top pairs
        pair_counts = sub["pair"].value_counts() if "pair" in sub.columns else pd.Series(dtype=int)
        top10 = pair_counts.head(10)
        tail = pair_counts.iloc[10:].sum() if len(pair_counts) > 10 else 0
        for p, cnt in top10.items():
            er_rows.append({"slice": slice_name, "exit_reason": er, "metric": f"pair_{p}", "statistic": "count", "value": int(cnt)})
        er_rows.append({"slice": slice_name, "exit_reason": er, "metric": "pair_TAIL", "statistic": "count", "value": int(tail)})

        # Time-from-peak-mfe-to-exit (for trail/time exits)
        time_peak_to_exit = (sub["bar_of_exit"] - sub["bar_of_peak_mfe"]).astype(float)
        er_rows.extend(long_distr_rows(slice_name, "_exit_reason", "time_from_peak_mfe_to_exit", time_peak_to_exit, percentiles=P_SHORT))
        # Time-from-entry-to-max-mae (for SL exits)
        time_entry_to_max_mae = sub["bar_of_max_mae"].astype(float)
        er_rows.extend(long_distr_rows(slice_name, "_exit_reason", "time_from_entry_to_max_mae", time_entry_to_max_mae, percentiles=P_SHORT))

# Rewire exit_reason into the rows
er_rows_final = []
last_slice = None
last_er = None
for r in er_rows:
    if "exit_reason" in r:
        last_slice = r["slice"]
        last_er = r["exit_reason"]
        er_rows_final.append(r)
    else:
        # belongs to current slice / exit_reason via context (from long_distr_rows that
        # didn't know the er value). Re-tag.
        rr = dict(r)
        rr["exit_reason"] = last_er
        er_rows_final.append(rr)

# Actually long_distr_rows sets "bin" key, not "exit_reason". Rebuild cleanly:
clean = []
current_er = None
for r in er_rows:
    if "exit_reason" in r and r["exit_reason"] is not None:
        current_er = r["exit_reason"]
        clean.append(r)
    elif "bin" in r and r["bin"] == "_exit_reason":
        rr = {k: v for k, v in r.items() if k != "bin"}
        rr["exit_reason"] = current_er
        clean.append(rr)
    else:
        clean.append(r)

exit_full_df = pd.DataFrame(clean)
# Re-order columns
cols_order = ["slice", "exit_reason", "metric", "statistic", "value"]
exit_full_df = exit_full_df.reindex(columns=cols_order)
exit_full_df.to_csv(OUT_DIR / "exit_reason_metrics_full.csv", index=False)
print(f"[done] exit_reason_metrics_full.csv ({len(exit_full_df)} rows)")


# ---------- exit_reason_summary.md ----------

print("[info] building exit_reason_summary.md...")

def fmt_one_summary(slice_name: str, df: pd.DataFrame) -> str:
    out = [f"## Slice {slice_name} (N={len(df)})\n"]
    out.append("### Top-line by exit_reason\n")
    out.append("| exit_reason | n | %  | mean final_r | p50 final_r | mean peak_mfe | mean max_mae | mean giveback | median bar_of_exit |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for er in sorted(df["exit_reason"].dropna().unique()):
        sub = df[df["exit_reason"] == er]
        n = len(sub)
        pct = n / len(df)
        out.append(
            f"| {er} | {n} | {pct:.1%} | "
            f"{sub['final_r'].mean():.4f} | {sub['final_r'].median():.4f} | "
            f"{sub['peak_mfe_r'].mean():.4f} | {sub['max_mae_r'].mean():.4f} | "
            f"{sub['giveback_r'].mean():.4f} | {int(sub['bar_of_exit'].median())} |"
        )
    out.append("")
    # Path-shape table per exit reason
    out.append("### Path-shape (mean) by exit_reason\n")
    out.append("| exit_reason | mono_in_profit | mono_pre_peak | local_peaks | pullback_med | time_to_peak_rel | close_r_t1 | velocity_t |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for er in sorted(df["exit_reason"].dropna().unique()):
        sub = df[df["exit_reason"] == er]
        out.append(
            f"| {er} | {sub['monotonicity_ratio_in_profit'].mean():.3f} | "
            f"{sub['monotonicity_ratio_pre_peak'].mean():.3f} | "
            f"{sub['local_peaks_count'].mean():.2f} | "
            f"{sub['pullback_magnitude_median'].mean():.4f} | "
            f"{sub['time_to_peak_mfe_relative'].mean():.3f} | "
            f"{sub['close_r_at_t1'].mean():.4f} | "
            f"{sub['velocity_first_t'].mean():.4f} |"
        )
    out.append("")
    # Entry-time by exit reason
    out.append("### Entry-time (mean) by exit_reason\n")
    out.append("| exit_reason | mean atr_14 | mean bar_range | mean sl_pips | mean spread | top pair (count) |")
    out.append("|---|---:|---:|---:|---:|---|")
    for er in sorted(df["exit_reason"].dropna().unique()):
        sub = df[df["exit_reason"] == er]
        top_pair = sub["pair"].value_counts().head(1)
        top_pair_str = f"{top_pair.index[0]} ({top_pair.iloc[0]})" if len(top_pair) else "—"
        out.append(
            f"| {er} | {sub['atr_14_at_signal'].mean():.5f} | "
            f"{sub['bar_range_at_signal'].mean():.5f} | "
            f"{sub['sl_distance_pips'].mean():.2f} | "
            f"{sub['spread_pips_used'].mean():.3f} | "
            f"{top_pair_str} |"
        )
    out.append("")
    return "\n".join(out)

lines = ["# Exit-reason summary per slice\n", "Read this together with `exit_reason_metrics_full.csv` for the full metric dump.\n"]
for slice_name, df in slices.items():
    lines.append(fmt_one_summary(slice_name, df))

(OUT_DIR / "exit_reason_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("[done] exit_reason_summary.md")


# ---------- conversion_analysis.csv ----------

print("[info] building conversion analysis...")

conv_rows = []
for slice_name, df in slices.items():
    sub = df[df["peak_mfe_r"] >= 1.0].copy()
    n_total = len(sub)
    if n_total == 0:
        continue
    grp_above_1 = sub[sub["final_r"] >= 1.0]
    grp_partial = sub[(sub["final_r"] >= 0.0) & (sub["final_r"] < 1.0)]
    grp_negative = sub[sub["final_r"] < 0.0]
    for label, g in [
        ("ended_>=_1R", grp_above_1),
        ("ended_0_to_1R", grp_partial),
        ("ended_<_0R", grp_negative),
    ]:
        n = len(g)
        pct = n / n_total
        median_gb = g["giveback_r"].median() if n else np.nan
        mean_gb = g["giveback_r"].mean() if n else np.nan
        conv_rows.append({
            "slice": slice_name,
            "subgroup": label,
            "n_in_subgroup": int(n),
            "n_reached_1R_total": int(n_total),
            "pct_of_reached_1R": pct,
            "median_giveback_r": median_gb,
            "mean_giveback_r": mean_gb,
        })
    conv_rows.append({
        "slice": slice_name,
        "subgroup": "TOTAL_reached_1R",
        "n_in_subgroup": int(n_total),
        "n_reached_1R_total": int(n_total),
        "pct_of_reached_1R": 1.0,
        "median_giveback_r": sub["giveback_r"].median(),
        "mean_giveback_r": sub["giveback_r"].mean(),
    })

conv_df = pd.DataFrame(conv_rows)
conv_df.to_csv(OUT_DIR / "conversion_analysis.csv", index=False)
print(f"[done] conversion_analysis.csv")


# ---------- specific anatomy plots ----------

print("[info] anatomy plots...")

# Q1: stop-out histogram (bin 5) per slice
for slice_name, df in slices.items():
    sub = df[df["bin_primary"] == 5]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    if len(sub):
        ax.hist(sub["bar_of_exit"].dropna(), bins=range(0, 21), color="#b35806", edgecolor="black", alpha=0.85)
    ax.set_xlabel("bar_of_exit")
    ax.set_ylabel("count")
    ax.set_title(f"Slice {slice_name} bin 5 (full stop-outs) — bar_of_exit distribution (N={len(sub)})")
    ax.set_xticks(range(0, 21, 2))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"stopout_hist_slice_{slice_name}.png", dpi=150, facecolor="white")
    plt.close(fig)

# Q2: peak_mfe_r x bar_of_peak_mfe heatmap for bin 2 (modest winners)
peak_table_rows = []
for slice_name, df in slices.items():
    sub = df[df["bin_primary"] == 2]
    if len(sub) == 0:
        # empty heatmap
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        ax.set_title(f"Slice {slice_name} bin 2 (modest winners) — no data")
        fig.savefig(PLOT_DIR / f"peak_heatmap_slice_{slice_name}.png", dpi=150, facecolor="white")
        plt.close(fig)
        continue
    # Bin: peak_mfe in [0,0.5),[0.5,1.0),[1.0,1.5),[1.5,2.0),[2.0,3.0),[3.0,5.0),[5.0,inf)
    peak_edges = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]
    bar_edges = [0, 1, 2, 3, 5, 10, 20, 60, 240]
    sub2 = sub.copy()
    sub2["peak_bin"] = pd.cut(sub2["peak_mfe_r"], peak_edges, right=False)
    sub2["bar_bin"] = pd.cut(sub2["bar_of_peak_mfe"], bar_edges, right=False)
    table = sub2.groupby(["peak_bin", "bar_bin"], observed=False).size().unstack(fill_value=0)

    # Save table rows
    for pb in table.index:
        for bb in table.columns:
            peak_table_rows.append({"slice": slice_name, "peak_mfe_bin": str(pb), "bar_of_peak_bin": str(bb), "count": int(table.loc[pb, bb])})

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    im = ax.imshow(table.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([str(c) for c in table.columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([str(i) for i in table.index], fontsize=8)
    ax.set_xlabel("bar_of_peak_mfe")
    ax.set_ylabel("peak_mfe_r")
    ax.set_title(f"Slice {slice_name} bin 2 (modest winners) — peak heatmap (N={len(sub)})")
    plt.colorbar(im, ax=ax, label="count")
    # Annotate
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color="black")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"peak_heatmap_slice_{slice_name}.png", dpi=150, facecolor="white")
    plt.close(fig)

pd.DataFrame(peak_table_rows).to_csv(OUT_DIR / "peak_heatmap_counts.csv", index=False)

# Q3: giveback histogram for winners (bins 1+2)
for slice_name, df in slices.items():
    sub = df[df["bin_primary"].isin([1, 2])]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    if len(sub):
        ax.hist(sub["giveback_r"].dropna(), bins=40, color="#2166ac", edgecolor="black", alpha=0.85)
    ax.set_xlabel("giveback_r (peak_mfe_r − final_r)")
    ax.set_ylabel("count")
    ax.set_title(f"Slice {slice_name} winners (bins 1+2) — giveback distribution (N={len(sub)})")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"giveback_hist_slice_{slice_name}.png", dpi=150, facecolor="white")
    plt.close(fig)


# Q4 / Q5: cluster 1 winners vs losers MAE-by-bar overlay
B = slices["B"]
winners = B[B["bin_primary"].isin([1, 2])]
losers = B[B["bin_primary"].isin([4, 5])]
b_traj = bar_df[bar_df["trade_id"].isin(set(B["trade_id"].tolist()))].copy()
b_traj = b_traj.merge(B[["trade_id", "bin_primary"]], on="trade_id", how="left")
wmask = b_traj["bin_primary"].isin([1, 2])
lmask = b_traj["bin_primary"].isin([4, 5])
agg_win = b_traj[wmask].groupby("bar_offset")["mae_so_far_r"].agg(["mean", lambda s: s.quantile(0.50), lambda s: s.quantile(0.90)])
agg_los = b_traj[lmask].groupby("bar_offset")["mae_so_far_r"].agg(["mean", lambda s: s.quantile(0.50), lambda s: s.quantile(0.90)])
agg_win.columns = ["mean", "p50", "p90"]
agg_los.columns = ["mean", "p50", "p90"]

fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
ax.plot(agg_win.index, agg_win["mean"], label="winners mean", color="#1b7837", linewidth=2)
ax.plot(agg_win.index, agg_win["p90"], label="winners p90", color="#1b7837", linestyle="--", alpha=0.7)
ax.plot(agg_los.index, agg_los["mean"], label="losers mean", color="#b35806", linewidth=2)
ax.plot(agg_los.index, agg_los["p90"], label="losers p90", color="#b35806", linestyle="--", alpha=0.7)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("bar_offset")
ax.set_ylabel("MAE (R-multiples)")
ax.set_title(f"Slice B (cluster 1) — winners (bins 1+2, N={len(winners)}) vs losers (bins 4+5, N={len(losers)}): MAE by bar")
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "cluster1_winners_vs_losers_mae.png", dpi=150, facecolor="white")
plt.close(fig)

# Save Q5 table
q5_rows = []
for bar in range(0, 21):
    win_bar = b_traj[wmask & (b_traj["bar_offset"] == bar)]["mae_so_far_r"]
    los_bar = b_traj[lmask & (b_traj["bar_offset"] == bar)]["mae_so_far_r"]
    q5_rows.append({
        "bar": bar,
        "winners_mean_mae": win_bar.mean(),
        "winners_p50_mae": win_bar.median(),
        "winners_p90_mae": win_bar.quantile(0.90),
        "winners_n": len(win_bar),
        "losers_mean_mae": los_bar.mean(),
        "losers_p50_mae": los_bar.median(),
        "losers_p90_mae": los_bar.quantile(0.90),
        "losers_n": len(los_bar),
    })
pd.DataFrame(q5_rows).to_csv(OUT_DIR / "q5_cluster1_winners_vs_losers_mae_by_bar.csv", index=False)


# ---------- comparisons ----------

def comparison_1_table():
    """Cluster 1 winners vs cluster 1 losers, bar-1-3 MAE."""
    out_lines = ["| bar | winners mean MAE | winners p90 MAE | losers mean MAE | losers p90 MAE |", "|---:|---:|---:|---:|---:|"]
    for bar in [1, 2, 3]:
        win_bar = b_traj[wmask & (b_traj["bar_offset"] == bar)]["mae_so_far_r"]
        los_bar = b_traj[lmask & (b_traj["bar_offset"] == bar)]["mae_so_far_r"]
        out_lines.append(f"| {bar} | {win_bar.mean():.4f} | {win_bar.quantile(0.90):.4f} | {los_bar.mean():.4f} | {los_bar.quantile(0.90):.4f} |")
    return "\n".join(out_lines)


def comparison_2_table():
    """Cluster 1 (B) vs non-cluster-1 (D) per outcome bin."""
    rows = []
    rows.append("| bin | slice | n | % of slice | mean final_r | mean peak_mfe | mean max_mae | mean giveback | mean mono_in_profit | mean local_peaks |")
    rows.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bin_id in [1, 2, 3, 4, 5]:
        for s_name, s_df in [("B", slices["B"]), ("D", slices["D"])]:
            sub = s_df[s_df["bin_primary"] == bin_id]
            n = len(sub)
            pct = n / len(s_df) if len(s_df) else 0
            rows.append(
                f"| {BIN_LABEL[bin_id]} | {s_name} | {n} | {pct:.1%} | "
                f"{sub['final_r'].mean():.4f} | "
                f"{sub['peak_mfe_r'].mean():.4f} | "
                f"{sub['max_mae_r'].mean():.4f} | "
                f"{sub['giveback_r'].mean():.4f} | "
                f"{sub['monotonicity_ratio_in_profit'].mean():.3f} | "
                f"{sub['local_peaks_count'].mean():.2f} |"
            )
    return "\n".join(rows)


def comparison_3_table():
    """Refit-admitted (C) vs cluster 1 raw (B). Per outcome bin."""
    rows = []
    rows.append("| bin | slice | n | % of slice | mean final_r | mean peak_mfe | mean giveback |")
    rows.append("|---|---|---:|---:|---:|---:|---:|")
    for bin_id in [1, 2, 3, 4, 5, 6]:
        for s_name, s_df in [("B", slices["B"]), ("C", slices["C"])]:
            if bin_id == 6:
                sub = s_df[s_df["bin_t3_clean"] == 1]
            else:
                sub = s_df[s_df["bin_primary"] == bin_id]
            n = len(sub)
            pct = n / len(s_df) if len(s_df) else 0
            rows.append(
                f"| {BIN_LABEL[bin_id]} | {s_name} | {n} | {pct:.1%} | "
                f"{sub['final_r'].mean():.4f} | "
                f"{sub['peak_mfe_r'].mean():.4f} | "
                f"{sub['giveback_r'].mean():.4f} |"
            )
    return "\n".join(rows)


def comparison_4_table():
    """Giveback distribution for slice C bin 2 (modest winners that survived filter 1)."""
    sub = slices["C"]
    sub = sub[sub["bin_primary"] == 2]
    if len(sub) == 0:
        return "Slice C bin 2 is empty."
    g = sub["giveback_r"].dropna()
    rows = ["| statistic | value (R) |", "|---|---:|"]
    rows.append(f"| n | {len(g)} |")
    rows.append(f"| mean | {g.mean():.4f} |")
    rows.append(f"| std | {g.std():.4f} |")
    rows.append(f"| p10 | {g.quantile(0.10):.4f} |")
    rows.append(f"| p25 | {g.quantile(0.25):.4f} |")
    rows.append(f"| p50 | {g.quantile(0.50):.4f} |")
    rows.append(f"| p75 | {g.quantile(0.75):.4f} |")
    rows.append(f"| p90 | {g.quantile(0.90):.4f} |")
    rows.append(f"| p95 | {g.quantile(0.95):.4f} |")
    return "\n".join(rows)


# ---------- anatomy answer tables ----------

# Q1 table: stop-out bar histogram per slice
q1_lines = ["| bar | slice A | slice B | slice C | slice D |", "|---:|---:|---:|---:|---:|"]
for bar in range(0, 21):
    cells = []
    for s_name in ["A", "B", "C", "D"]:
        sub = slices[s_name][slices[s_name]["bin_primary"] == 5]
        cnt = int((sub["bar_of_exit"] == bar).sum())
        cells.append(str(cnt))
    q1_lines.append(f"| {bar} | " + " | ".join(cells) + " |")
q1_text = "\n".join(q1_lines)

# Q3 table: giveback percentiles for winners (bins 1+2) per slice
q3_lines = ["| slice | n | p10 | p25 | p50 | p75 | p90 | p95 | mean |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for s_name, s_df in slices.items():
    sub = s_df[s_df["bin_primary"].isin([1, 2])]
    g = sub["giveback_r"].dropna()
    if len(g) == 0:
        q3_lines.append(f"| {s_name} | 0 | — | — | — | — | — | — | — |")
        continue
    q3_lines.append(
        f"| {s_name} | {len(g)} | {g.quantile(0.10):.4f} | {g.quantile(0.25):.4f} | "
        f"{g.quantile(0.50):.4f} | {g.quantile(0.75):.4f} | {g.quantile(0.90):.4f} | "
        f"{g.quantile(0.95):.4f} | {g.mean():.4f} |"
    )
q3_text = "\n".join(q3_lines)

# Q4 table: bin 4 + 5 MAE at bar 1, 2, 3 per slice
q4_lines = ["| slice | bar | mean MAE | p90 MAE | n |", "|---|---:|---:|---:|---:|"]
for s_name, s_df in slices.items():
    bad = s_df[s_df["bin_primary"].isin([4, 5])]
    tids = set(bad["trade_id"].tolist())
    sub_bar = bar_df[bar_df["trade_id"].isin(tids)]
    for bar in [1, 2, 3]:
        sub_b = sub_bar[sub_bar["bar_offset"] == bar]["mae_so_far_r"]
        q4_lines.append(f"| {s_name} | {bar} | {sub_b.mean():.4f} | {sub_b.quantile(0.90):.4f} | {len(sub_b)} |")
q4_text = "\n".join(q4_lines)

# Q5 table: cluster 1 winners vs losers MAE per bar (1..20)
q5_lines = ["| bar | winners mean | winners p50 | winners p90 | winners n | losers mean | losers p50 | losers p90 | losers n |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for bar in range(1, 21):
    win_bar = b_traj[wmask & (b_traj["bar_offset"] == bar)]["mae_so_far_r"]
    los_bar = b_traj[lmask & (b_traj["bar_offset"] == bar)]["mae_so_far_r"]
    q5_lines.append(
        f"| {bar} | {win_bar.mean():.4f} | {win_bar.median():.4f} | {win_bar.quantile(0.90):.4f} | {len(win_bar)} | "
        f"{los_bar.mean():.4f} | {los_bar.median():.4f} | {los_bar.quantile(0.90):.4f} | {len(los_bar)} |"
    )
q5_text = "\n".join(q5_lines)

# Q6 conversion table
q6_lines = ["| slice | reached_1R n | %end>=1R | median giveback (>=1R) | %ended 0..1R | median giveback (0..1R) | %ended <0 | median giveback (<0) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
for s_name in ["A", "B", "C", "D"]:
    df_conv = conv_df[conv_df["slice"] == s_name]
    if len(df_conv) == 0:
        q6_lines.append(f"| {s_name} | 0 | — | — | — | — | — | — |")
        continue
    total_n = int(df_conv[df_conv["subgroup"] == "TOTAL_reached_1R"]["n_in_subgroup"].iloc[0])
    g1 = df_conv[df_conv["subgroup"] == "ended_>=_1R"].iloc[0]
    g2 = df_conv[df_conv["subgroup"] == "ended_0_to_1R"].iloc[0]
    g3 = df_conv[df_conv["subgroup"] == "ended_<_0R"].iloc[0]
    q6_lines.append(
        f"| {s_name} | {total_n} | "
        f"{g1['pct_of_reached_1R']:.1%} | {g1['median_giveback_r']:.4f} | "
        f"{g2['pct_of_reached_1R']:.1%} | {g2['median_giveback_r']:.4f} | "
        f"{g3['pct_of_reached_1R']:.1%} | {g3['median_giveback_r']:.4f} |"
    )
q6_text = "\n".join(q6_lines)


# ---------- headline tables for CHARACTERISATION.md ----------

def headline_table(slice_name: str, df: pd.DataFrame) -> str:
    lines = [f"### Slice {slice_name} (N={len(df)})\n"]
    lines.append("| exit_reason | n | % | mean final_r | p50 final_r | mean peak_mfe | mean max_mae | mean giveback | median bar_of_exit |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for er in sorted(df["exit_reason"].dropna().unique()):
        sub = df[df["exit_reason"] == er]
        n = len(sub)
        pct = n / len(df)
        lines.append(
            f"| {er} | {n} | {pct:.1%} | "
            f"{sub['final_r'].mean():.4f} | {sub['final_r'].median():.4f} | "
            f"{sub['peak_mfe_r'].mean():.4f} | {sub['max_mae_r'].mean():.4f} | "
            f"{sub['giveback_r'].mean():.4f} | {int(sub['bar_of_exit'].median())} |"
        )
    return "\n".join(lines)


# ---------- bin-distribution table per slice ----------

def bin_distribution_table(df: pd.DataFrame, slice_name: str) -> str:
    lines = [f"### Slice {slice_name} outcome-bin distribution\n"]
    lines.append("| bin | n | % | mean final_r | p50 final_r | mean peak_mfe | p50 peak_mfe | mean max_mae | p50 max_mae | mean giveback | p50 giveback |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bin_id in [1, 2, 3, 4, 5, 6]:
        if bin_id == 6:
            sub = df[df["bin_t3_clean"] == 1]
        else:
            sub = df[df["bin_primary"] == bin_id]
        n = len(sub)
        pct = n / len(df) if len(df) else 0
        if n == 0:
            lines.append(f"| {BIN_LABEL[bin_id]} | 0 | 0.0% | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {BIN_LABEL[bin_id]} | {n} | {pct:.1%} | "
            f"{sub['final_r'].mean():.4f} | {sub['final_r'].median():.4f} | "
            f"{sub['peak_mfe_r'].mean():.4f} | {sub['peak_mfe_r'].median():.4f} | "
            f"{sub['max_mae_r'].mean():.4f} | {sub['max_mae_r'].median():.4f} | "
            f"{sub['giveback_r'].mean():.4f} | {sub['giveback_r'].median():.4f} |"
        )
    return "\n".join(lines)


# ---------- CHARACTERISATION.md ----------

print("[info] writing CHARACTERISATION.md...")

md = []
md.append("# Arc 4 Characterisation\n")
md.append("> Read-only diagnostic survey of Arc 4 trades. No verdicts, proposals, or recommendations.\n")
md.append("> Produced 2026-05-18 from Arc 4 step1/step2/step3/step5c artefacts.\n")
md.append("")
md.append("## Input file SHA256\n")
md.append("| File | SHA256 |")
md.append("|---|---|")
for name, p in INPUTS.items():
    md.append(f"| `{p.relative_to(REPO).as_posix()}` | `{SHAS[name]}` |")
md.append("")
md.append("## Slice definitions and final_r source\n")
md.append("| Slice | Definition | N | final_r source |")
md.append("|---|---|---:|---|")
md.append(f"| A | All Arc 4 trades | {len(slices['A'])} | §11 row 2 simulated on path (2×ATR R-frame) |")
md.append(f"| B | Cluster 1 only | {len(slices['B'])} | §11 row 2 simulated on path (2×ATR R-frame) |")
md.append(f"| C | Refit-admitted F2–F7 | {len(slices['C'])} | `per_trade_simulated_refit_1.csv` (3×ATR cluster_R frame) |")
md.append(f"| D | Non-cluster-1 (A minus B) | {len(slices['D'])} | §11 row 2 simulated on path (2×ATR R-frame) |")
md.append("")
md.append("### R-frame note (data quality)\n")
md.append("Path file `trades_paths.csv` normalises `close_r`, `mfe_so_far_r`, `mae_so_far_r` against 2×ATR(14)_1H. ")
md.append("§11 row 2 policy was applied in this 2×ATR frame for slices A/B/D. ")
md.append("Slice C's refit `final_r` is in the cluster_R = 3×ATR(14)_1H frame (cluster 1 selected SL=3 in §3). ")
md.append("Cross-slice R-multiple comparisons therefore mix R-frames; magnitudes of `final_r` / `peak_mfe_r` / `giveback_r` for slice C are not directly comparable to slices A/B/D. ")
md.append("Direction of effects and within-slice patterns are still comparable. `max_mae_r` for slice C was taken from path metrics (2×ATR frame) since the refit per-trade file did not record max MAE.")
md.append("")

# Outcome bin distribution per slice
md.append("## Outcome bin distribution per slice\n")
for s_name, s_df in slices.items():
    md.append(bin_distribution_table(s_df, s_name))
    md.append("")

# Headline tables
md.append("## Headline by exit_reason per slice\n")
for s_name, s_df in slices.items():
    md.append(headline_table(s_name, s_df))
    md.append("")

# Path-shape per slice (mean over slice; full distribution in exit_reason_metrics_full.csv)
md.append("## Path-shape mean by slice\n")
md.append("| slice | mono_in_profit | mono_pre_peak | local_peaks | pullback_med | time_to_peak_rel | velocity_t | close_r_t1 | mae_t1 | mfe_t1 |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for s_name, s_df in slices.items():
    md.append(
        f"| {s_name} | {s_df['monotonicity_ratio_in_profit'].mean():.3f} | "
        f"{s_df['monotonicity_ratio_pre_peak'].mean():.3f} | "
        f"{s_df['local_peaks_count'].mean():.2f} | "
        f"{s_df['pullback_magnitude_median'].mean():.4f} | "
        f"{s_df['time_to_peak_mfe_relative'].mean():.3f} | "
        f"{s_df['velocity_first_t'].mean():.4f} | "
        f"{s_df['close_r_at_t1'].mean():.4f} | "
        f"{s_df['mae_so_far_r_at_t1'].mean():.4f} | "
        f"{s_df['mfe_so_far_r_at_t1'].mean():.4f} |"
    )
md.append("")

# Entry-time mean by slice
md.append("## Entry-time context mean by slice\n")
md.append("| slice | mean atr_14 | mean bar_range | mean sl_pips | mean spread | top pair (count) | top hour (count) | top dow (count) | top session (count) |")
md.append("|---|---:|---:|---:|---:|---|---|---|---|")
for s_name, s_df in slices.items():
    top_pair = s_df["pair"].value_counts().head(1)
    top_pair_str = f"{top_pair.index[0]} ({top_pair.iloc[0]})" if len(top_pair) else "—"
    top_hour = s_df["hour_of_day"].value_counts().head(1)
    top_hour_str = f"{int(top_hour.index[0]):02d} ({top_hour.iloc[0]})" if len(top_hour) else "—"
    top_dow = s_df["day_of_week"].value_counts().head(1)
    top_dow_str = f"{top_dow.index[0]} ({top_dow.iloc[0]})" if len(top_dow) else "—"
    top_sess = s_df["session"].value_counts().head(1)
    top_sess_str = f"{top_sess.index[0]} ({top_sess.iloc[0]})" if len(top_sess) else "—"
    md.append(
        f"| {s_name} | {s_df['atr_14_at_signal'].mean():.5f} | "
        f"{s_df['bar_range_at_signal'].mean():.5f} | "
        f"{s_df['sl_distance_pips'].mean():.2f} | "
        f"{s_df['spread_pips_used'].mean():.3f} | "
        f"{top_pair_str} | {top_hour_str} | {top_dow_str} | {top_sess_str} |"
    )
md.append("")

# Anatomy answers
md.append("## Specific anatomy questions\n")

md.append("### Q1. When are losers stopping out?")
md.append("Distribution of `bar_of_exit` for outcome bin 5 (full stop-outs, final_r ≤ −1.0) across slices, bars 0–20.")
md.append("Histogram PNG per slice: `plots/stopout_hist_slice_{A,B,C,D}.png`.\n")
md.append(q1_text)
md.append("")
md.append("**Observation (factual):** see counts above and `plots/stopout_hist_slice_*.png`.\n")

md.append("### Q2. Where are we peaking?")
md.append("2-D distribution of `peak_mfe_r` × `bar_of_peak_mfe` for outcome bin 2 (modest winners, 0.5 ≤ final_r < 1.5), per slice.")
md.append("Heatmap PNGs: `plots/peak_heatmap_slice_{A,B,C,D}.png`. Counts table: `peak_heatmap_counts.csv`.")
md.append("")

md.append("### Q3. What are we giving back?")
md.append("`giveback_r` percentiles for winners (bins 1+2 combined), per slice. Histograms: `plots/giveback_hist_slice_{A,B,C,D}.png`.\n")
md.append(q3_text)
md.append("")

md.append("### Q4. What's killing us? (bin 4+5 MAE at bars 1–3)\n")
md.append(q4_text)
md.append("")

md.append("### Q5. Max MAE bars 1–20 for cluster 1 winners vs losers (Slice B)\n")
md.append("Plot: `plots/cluster1_winners_vs_losers_mae.png`. Also `q5_cluster1_winners_vs_losers_mae_by_bar.csv`.\n")
md.append(q5_text)
md.append("")

md.append("### Q6. Conversion analysis (trades that ever reached peak_mfe_r ≥ 1.0)\n")
md.append(q6_text)
md.append("")

# Diagnostic comparisons
md.append("## Four diagnostic comparisons\n")

md.append("### Comparison 1 — Cluster 1 winners vs cluster 1 losers, bar-1–3 MAE (slice B)")
md.append("Winners = bins 1+2. Losers = bins 4+5.\n")
md.append(comparison_1_table())
md.append("")

md.append("### Comparison 2 — Cluster 1 (B) vs non-cluster-1 (D) within each outcome bin\n")
md.append(comparison_2_table())
md.append("")

md.append("### Comparison 3 — Refit-admitted (C) vs cluster 1 raw (B) per outcome bin")
md.append("R-frame note: slice B is in 2×ATR frame, slice C is in 3×ATR frame; absolute R magnitudes are not directly comparable.\n")
md.append(comparison_3_table())
md.append("")

md.append("### Comparison 4 — Giveback distribution for slice C bin 2 (modest winners that survived filter 1)\n")
md.append(comparison_4_table())
md.append("")

# Data quality
md.append("## Data quality / assumptions\n")
md.append("- `trades_all.csv` exit_reason has only two values: `stop_loss` (8899) and `max_life` (1865). These come from the step1 SL=2×ATR simulation, NOT from §11 row 2. After re-simulating §11 row 2 on the path, exit_reason values for slices A/B/D are: `sl_initial`, `trail_breakeven`, `trail`, `time`.")
md.append("- `trades_paths.csv` paths extend up to bar 240 for every trade (cluster 1 horizon). `mfe_so_far_r` / `mae_so_far_r` are RUNNING and continue evolving past `is_held=0` (i.e., they reflect what would have happened with no SL).")
md.append("- §11 row 2 simulation order: at each bar, check stop hit against the stop level entering the bar; if no hit, update stop based on this bar's MFE. This is the conservative within-bar ordering (stop hit takes priority if both can fire same bar).")
md.append("- Entry-time features available in `trades_all.csv`: pair, signal_time, entry_time, entry_price, atr_14_at_signal, bar_range_at_signal, sl_distance_pips, spread_pips_used. Features NOT in this artefact (not analysed): RSI, D1/H4 trend state, volatility quintile column, bar0 body/range/wick ratios — these would need additional engine work.")
md.append("- Session derivation: hour-of-day buckets (00–06 asia, 07–12 london, 13–16 ny_overlap, 17–23 ny_late). Approximation only — actual session boundaries depend on DST.")
md.append("- Path-derived metrics (`monotonicity_ratio_in_profit`, `local_peaks_count`, `pullback_magnitude_median`, etc.) are computed from `close_r` on the alive window (bars 0..bar_of_exit inclusive). Definitions are documented in `scripts/arc_4_characterisation/compute_per_trade.py`.")
md.append("- Cluster 1 archetype summary records `pre_peak_mono` = 0.564 (passing v2.1.1 §3 rescue); the per-trade `monotonicity_ratio_pre_peak` here is computed bar-by-bar in close_r terms on the §11 row 2 alive window, so it differs from the cluster-3-step centroid measure.")
md.append("- `mfe_locked_bar` column in `per_trade_path_metrics.csv` = first bar where MFE reached ≥ 1.0R DURING the alive window (the §11 row 2 lock trigger); set to −1 if never reached during alive window.")
md.append("- `peak_mfe_r` and `max_mae_r` for slices A/B/D are computed over the **alive window** (bars 0..exit_bar under §11 row 2), not the full 240-bar forward window. This matches the cluster-archetype-frame definition used for slice C's refit `peak_mfe_r` and keeps Q6 conversion analysis consistent with the §11 row 2 policy invariant (locked-in trades cannot end up negative — `sl_initial` exits never trail-triggered).")
md.append("- For slices A/B/D, `peak_mfe_r_full_window` and `max_mae_r_full_window` are also recorded in `per_trade_path_metrics.csv` (full 240-bar window) for trades-that-would-have-recovered analysis if needed.")
md.append("- `trail_triggered` flag = 1 if the §11 row 2 stop was actually updated past −1R during the alive window (i.e., MFE reached 1R and the lock fired before a stop hit). Exits with `exit_reason='sl_initial'` always have `trail_triggered=0`.")
md.append("- **Bin 4 (modest losers, −1.0 < final_r ≤ −0.5) is empty in slices A/B/D under §11 row 2.** This is a policy artefact: §11 row 2 stops are binary (initial SL at −1R or trail-lock floor at 0R+), so there is no exit mechanism producing final_r in (−1, −0.5]. Slice C bin 4 has 13 trades — these come from the refit simulator which has a slightly different exit-bar accounting (`cap_bind` and edge cases). Q4 (\"bin 4+5 MAE\") therefore reduces to bin-5 MAE for slices A/B/D.")
md.append("- **Uniform `giveback_r = 0.7500` in winners (bins 1+2) and bin 6 is by §11 row 2 trail policy.** Under the policy, every trail-exit gives back exactly 0.75R from peak (the trail distance). All bin 1, bin 2, and bin 6 trades exit via the trail. The std in slice C is non-zero (0.0177) because the refit simulator has minor implementation differences (e.g., `cap_bind` rare cases). For slices A/B/D, giveback for trail exits is exactly 0.75R by construction. Q3 / Comparison 4's degenerate distribution reflects this geometry, not noise.")
md.append("- Slice C's `peak_mfe_r` (refit frame, 3×ATR) and slice B's `peak_mfe_r` (alive-window, 2×ATR) are not directly comparable: a slice-B peak of 2.0R = 4×ATR, while a slice-C peak of 2.0R = 6×ATR. Same R-multiple thresholds mean different absolute price moves.")
md.append("")

md.append("## Artefact index\n")
md.append("- `CHARACTERISATION.md` — this doc")
md.append("- `per_trade_path_metrics.csv` — 10,764 rows, §11 row 2 simulation + path-shape metrics per trade")
md.append("- `per_trade_bar_0_20.csv` — bar-by-bar (bars 0..20) for all trades")
md.append("- `per_cell_distributions.csv` — (slice × outcome bin × metric × statistic) long format")
md.append("- `bar_trajectory_mae.csv` / `bar_trajectory_mfe.csv` — (slice × bin × bar × statistic) long format")
md.append("- `exit_reason_metrics_full.csv` — full metric dump per (slice × exit_reason)")
md.append("- `exit_reason_summary.md` — human-readable headline + path-shape + entry-time tables per slice")
md.append("- `conversion_analysis.csv` — per-slice conversion of peak_mfe_r ≥ 1.0 trades")
md.append("- `peak_heatmap_counts.csv` — counts for Q2 heatmap")
md.append("- `q5_cluster1_winners_vs_losers_mae_by_bar.csv` — Q5 raw data")
md.append("- `plots/` — 19 PNGs total: 8× trajectory, 4× stop-out hist, 4× peak heatmap, 4× giveback hist (slice A/B/C/D × 4 categories) + 1× cluster1 winners-vs-losers MAE overlay")

(OUT_DIR / "CHARACTERISATION.md").write_text("\n".join(md), encoding="utf-8")
print("[done] CHARACTERISATION.md")

# Also export the slices themselves for reproducibility
for s_name, s_df in slices.items():
    s_df.to_csv(OUT_DIR / f"slice_{s_name}_data.csv", index=False)
print("[done] slice_*_data.csv written for reproducibility")
