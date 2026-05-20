"""Assemble EXIT_ENTRY_SWEEP.md from all phase artefacts."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "arc_4_exit_entry_sweep"

INPUTS = {
    "trades_all": REPO / "results" / "l_arc_4" / "step1" / "trades_all.csv",
    "trades_paths": REPO / "results" / "l_arc_4" / "step1" / "trades_paths.csv",
    "clusters_K4": REPO / "results" / "l_arc_4" / "step2" / "clusters_K4.csv",
    "refit_per_trade": REPO / "results" / "l_arc_4" / "step5c" / "per_trade_simulated_refit_1.csv",
    "path_metrics": REPO / "results" / "arc_4_characterisation" / "per_trade_path_metrics.csv",
    "bar_0_20": REPO / "results" / "arc_4_characterisation" / "per_trade_bar_0_20.csv",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


SHAS = {name: sha256_file(p) for name, p in INPUTS.items()}

# Load all phase outputs
a1 = pd.read_csv(OUT_DIR / "a1_trail_extension.csv")
a1_bars = pd.read_csv(OUT_DIR / "a1_bars_between_peaks.csv")
a2 = pd.read_csv(OUT_DIR / "a2_pre_post_lock.csv")
a2_paired = pd.read_csv(OUT_DIR / "a2_paired_tests.csv")
a3 = pd.read_csv(OUT_DIR / "a3_win_loss_metrics.csv")
a4 = pd.read_csv(OUT_DIR / "a4_per_cluster_metrics.csv")
a5 = pd.read_csv(OUT_DIR / "a5_winners_vs_losers.csv")
b1 = pd.read_csv(OUT_DIR / "b1_trail_trigger_sweep.csv")
b2 = pd.read_csv(OUT_DIR / "b2_trail_width_sweep.csv")
b3 = pd.read_csv(OUT_DIR / "b3_grid_2d.csv")
c1 = pd.read_csv(OUT_DIR / "c1_delayed_entry_sweep.csv")
c2 = pd.read_csv(OUT_DIR / "c2_two_bar_confirmation.csv")
c3 = pd.read_csv(OUT_DIR / "c3_combined_per_fold.csv")


def pivot_b_metric(df, slice_name, cluster_label, metric_name):
    sub = df[(df["slice"] == slice_name) & (df["cluster"] == cluster_label) & (df["metric"] == metric_name)]
    if "trigger" in sub.columns and "width" in sub.columns:
        return sub.pivot(index="trigger", columns="width", values="value")
    return None


def pivot_c1_metric(df, slice_name, cluster_label):
    sub = df[(df["slice"] == slice_name) & (df["cluster"] == cluster_label)]
    piv = sub.pivot(index="threshold", columns="metric", values="value")
    piv.index = piv.index.astype(float)
    return piv.sort_index()


def f4(x):
    if pd.isna(x):
        return "—"
    return f"{x:.4f}"


def f3(x):
    if pd.isna(x):
        return "—"
    return f"{x:.3f}"


def fi(x):
    if pd.isna(x):
        return "—"
    return f"{int(x)}"


md = []
md.append("# Arc 4 Exit Policy + Delayed Entry Sweep\n")
md.append("> Read-only re-simulation on existing trade paths. No engine changes, no live runs.")
md.append("> Off-protocol speculative work. Outputs are advisory.")
md.append("> Produced 2026-05-18 from Arc 4 artefacts + arc_4_characterisation derivatives.\n")
md.append("R-frame: 2×ATR(14)_signal_bar throughout all path-derived re-simulation. Slice C reporting of baseline `peak_mfe_r` retains the refit-file 3×ATR cluster_R frame; that frame mismatch is flagged where applicable.")
md.append("")
md.append("## Input file SHA256\n")
md.append("| File | SHA256 |")
md.append("|---|---|")
for name, p in INPUTS.items():
    md.append(f"| `{p.relative_to(REPO).as_posix()}` | `{SHAS[name]}` |")
md.append("")

# ============================================================
# Phase A
# ============================================================
md.append("## Phase A — Deeper diagnostics\n")

# A1
md.append("### A1. Trail extension after alive exit\n")
md.append("For trail-exited trades (baseline §11 row 2), measure `peak_mfe_full − peak_mfe_alive` (R, 2×ATR).\n")
md.append("**Headline: % of trail exits that extended ≥ 0.5R and ≥ 1.0R after exit, per slice (all clusters)**\n")
md.append("| slice | n_trail_exits | % ext ≥ 0.5R | % ext ≥ 1.0R | mean ext (R) | p50 ext (R) | p90 ext (R) |")
md.append("|---|---:|---:|---:|---:|---:|---:|")
for sname in ["A", "B", "C", "D"]:
    n_row = a1[(a1["slice"] == sname) & (a1["cluster"] == "all") & (a1["metric"] == "n_trail_exits")]
    if len(n_row) == 0:
        continue
    n = int(n_row["value"].iloc[0])
    def gv(metric, stat):
        r = a1[(a1["slice"] == sname) & (a1["cluster"] == "all") & (a1["metric"] == metric) & (a1["statistic"] == stat)]
        return r["value"].iloc[0] if len(r) else np.nan
    md.append(f"| {sname} | {n} | {gv('pct_extended_ge_0.5R', 'fraction'):.2%} | {gv('pct_extended_ge_1.0R', 'fraction'):.2%} | {f4(gv('extension_after_exit_r', 'mean'))} | {f4(gv('extension_after_exit_r', 'p50'))} | {f4(gv('extension_after_exit_r', 'p90'))} |")
md.append("")
md.append("**Per-cluster (slice A scope, cluster 1 highlighted)**\n")
md.append("| cluster | n_trail | % ext ≥ 0.5R | % ext ≥ 1.0R | mean ext | mean bars between alive & full peak (extenders ≥ 0.5R) |")
md.append("|---|---:|---:|---:|---:|---:|")
for cid in [0, 1, 2, 3]:
    n_row = a1[(a1["slice"] == "A") & (a1["cluster"] == f"cluster_{cid}") & (a1["metric"] == "n_trail_exits")]
    if len(n_row) == 0:
        continue
    n = int(n_row["value"].iloc[0])
    def gv(metric, stat):
        r = a1[(a1["slice"] == "A") & (a1["cluster"] == f"cluster_{cid}") & (a1["metric"] == metric) & (a1["statistic"] == stat)]
        return r["value"].iloc[0] if len(r) else np.nan
    bb_row = a1_bars[(a1_bars["slice"] == "A") & (a1_bars["cluster"] == f"cluster_{cid}_extenders_0.5R") & (a1_bars["metric"] == "bars_between_alive_and_full_peak") & (a1_bars["statistic"] == "mean")]
    bb_mean = bb_row["value"].iloc[0] if len(bb_row) else np.nan
    md.append(f"| {cid} | {n} | {gv('pct_extended_ge_0.5R', 'fraction'):.2%} | {gv('pct_extended_ge_1.0R', 'fraction'):.2%} | {f4(gv('extension_after_exit_r', 'mean'))} | {f3(bb_mean)} |")
md.append("")
md.append("CSVs: `a1_trail_extension.csv`, `a1_bars_between_peaks.csv`.")
md.append("")

# A2
md.append("### A2. Path behaviour before vs after trail activation\n")
md.append("Trail-activated trades only; pre-lock window = bars [0, lock_bar−1]; post-lock = [lock_bar, exit_bar].\n")
md.append("**Paired t-stats for cluster 1 (slice A): paired difference post − pre per trade**\n")
md.append("| metric | n_paired | mean diff (post−pre) | t-stat | p-value |")
md.append("|---|---:|---:|---:|---:|")
for metric in ["pullback_mean", "pullback_median", "pullback_max", "n_new_highs", "local_peaks",
               "var_diff_close_r", "mean_abs_diff_close_r", "time_within_0.25R_of_max"]:
    r = a2_paired[(a2_paired["slice"] == "A") & (a2_paired["cluster"] == "cluster_1") & (a2_paired["metric"] == metric)]
    if len(r) == 0:
        continue
    row = r.iloc[0]
    md.append(f"| {metric} | {int(row['n_paired'])} | {f4(row['mean_diff_post_minus_pre'])} | {f3(row['t_stat'])} | {row['p_value']:.4g} |")
md.append("")
md.append("Plot: `plots/a2_pullback_distribution_pre_vs_post_slice_B.png`.")
md.append("")
md.append("CSVs: `a2_pre_post_lock.csv`, `a2_paired_tests.csv`.")
md.append("")

# A3
md.append("### A3. Win/loss + profit factor per slice/cluster\n")
md.append("Baseline §11 row 2 policy.\n")
md.append("| slice | cluster | n | hit_rate | win_loss | mean_r | profit_factor | winner_p50_r | winner_p90_r |")
md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for _, r in a3.sort_values(["slice", "cluster"]).iterrows():
    md.append(f"| {r['slice']} | {r['cluster']} | {int(r['n'])} | {r['hit_rate']:.2%} | {f3(r['win_loss'])} | {f4(r['mean_r'])} | {f3(r['profit_factor'])} | {f4(r['winner_p50_r'])} | {f4(r['winner_p90_r'])} |")
md.append("")
md.append("Plot: `plots/a3_winner_final_r_distribution_slice_B.png` (slice B trail-exit final_r).")
md.append("")
md.append("CSV: `a3_win_loss_metrics.csv`.")
md.append("")

# A4
md.append("### A4. Per-cluster deep characterisation (slice A scope)\n")

# Side-by-side table
md.append("**Side-by-side: bin shares, win:loss, mean R, bar-1 winner/loser MAE**\n")
md.append("| cluster | n | bin1 share | bin5 share | win:loss | mean_r | bar1 winner MAE mean | bar1 loser MAE mean |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for cid in [0, 1, 2, 3]:
    def gv(group, metric):
        r = a4[(a4["cluster"] == cid) & (a4["metric_group"] == group) & (a4["metric"] == metric)]
        return r["value"].iloc[0] if len(r) else np.nan
    n = int(gv("outcome", "n"))
    b1s = gv("outcome", "bin1_share")
    b5s = gv("outcome", "bin5_share")
    wl = gv("outcome", "win_loss")
    mr = gv("outcome", "mean_r")
    wmae = gv("mae_bar1_winners", "mean")
    lmae = gv("mae_bar1_losers", "mean")
    md.append(f"| {cid} | {n} | {b1s:.2%} | {b5s:.2%} | {f3(wl)} | {f4(mr)} | {f4(wmae)} | {f4(lmae)} |")
md.append("")
md.append("Per-cluster path-shape means:\n")
md.append("| cluster | mono_in_profit | mono_pre_peak | local_peaks | pullback_med | time_to_peak_rel | velocity_t | close_r_t1 | mae_t1 | mfe_t1 |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for cid in [0, 1, 2, 3]:
    def gv(metric):
        r = a4[(a4["cluster"] == cid) & (a4["metric_group"] == "path_shape") & (a4["metric"] == metric)]
        return r["value"].iloc[0] if len(r) else np.nan
    md.append(f"| {cid} | {f3(gv('monotonicity_ratio_in_profit'))} | {f3(gv('monotonicity_ratio_pre_peak'))} | {f3(gv('local_peaks_count'))} | {f4(gv('pullback_magnitude_median'))} | {f3(gv('time_to_peak_mfe_relative'))} | {f4(gv('velocity_first_t'))} | {f4(gv('close_r_at_t1'))} | {f4(gv('mae_so_far_r_at_t1'))} | {f4(gv('mfe_so_far_r_at_t1'))} |")
md.append("")
md.append("CSV: `a4_per_cluster_metrics.csv`. Plots: `plots/a4_cluster_{0,1,2,3}_mae_winners_vs_losers.png`.")
md.append("")

# A5
md.append("### A5. Winners vs losers as outcome groups (slice B, top features by t-stat)\n")
md.append("Winners = bin1 ∪ bin2; losers = bin5. **Note:** descriptive use of outcome labels — no filter built from this analysis.\n")
md.append("**Top 5 features by |t-stat| for slice B**\n")
md.append("| metric | winner mean | loser mean | mean diff | t-stat | p-value |")
md.append("|---|---:|---:|---:|---:|---:|")
sub_b = a5[(a5["slice"] == "B") & a5["t_stat"].notna()].copy()
sub_b["abs_t"] = sub_b["t_stat"].abs()
top5 = sub_b.sort_values("abs_t", ascending=False).head(5)
for _, r in top5.iterrows():
    md.append(f"| {r['metric']} | {f4(r['winner_mean'])} | {f4(r['loser_mean'])} | {f4(r['mean_diff_winner_minus_loser'])} | {f3(r['t_stat'])} | {r['p_value']:.4g} |")
md.append("")
md.append("CSV: `a5_winners_vs_losers.csv`.")
md.append("")

# ============================================================
# Phase B
# ============================================================
md.append("## Phase B — Exit policy sweeps\n")
md.append("All sweeps in 2×ATR R-frame. Baseline reproduced exactly to per-trade level (10,764 / 10,764 match).\n")

# B1 headline
md.append("### B1. Trail-trigger sweep (width fixed at 0.75R) — slice B cluster 1\n")
md.append("| trigger | n | hit_rate | win_loss | mean_r | profit_factor | bin1_share | Δ mean_r vs baseline |")
md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
piv_b1 = b1[(b1["slice"] == "B") & (b1["cluster"] == "cluster_1")].pivot_table(index="trigger", columns="metric", values="value")
for t in sorted(piv_b1.index):
    r = piv_b1.loc[t]
    md.append(f"| {t} | {int(r['n'])} | {r['hit_rate']:.2%} | {f3(r['win_loss'])} | {f4(r['mean_r'])} | {f3(r['profit_factor'])} | {r['bin1_share']:.2%} | {f4(r['mean_delta_final_vs_baseline'])} |")
md.append("")

# B2 headline
md.append("### B2. Trail-width sweep (trigger fixed at 1.0R) — slice B cluster 1\n")
md.append("| width | n | hit_rate | win_loss | mean_r | profit_factor | bin1_share | Δ mean_r vs baseline |")
md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
piv_b2 = b2[(b2["slice"] == "B") & (b2["cluster"] == "cluster_1")].pivot_table(index="width", columns="metric", values="value")
for w in sorted(piv_b2.index):
    r = piv_b2.loc[w]
    md.append(f"| {w} | {int(r['n'])} | {r['hit_rate']:.2%} | {f3(r['win_loss'])} | {f4(r['mean_r'])} | {f3(r['profit_factor'])} | {r['bin1_share']:.2%} | {f4(r['mean_delta_final_vs_baseline'])} |")
md.append("")

# B3 best combinations
md.append("### B3. 2D grid (trigger × width) — best combinations for slice B cluster 1\n")
sub = b3[(b3["slice"] == "B") & (b3["cluster"] == "cluster_1")]
for label, metric in [("mean R per trade", "mean_r"), ("win:loss", "win_loss"), ("bin1 share", "bin1_share")]:
    s = sub[sub["metric"] == metric].sort_values("value", ascending=False).head(3)
    md.append(f"**Top 3 by {label}**:\n")
    md.append("| rank | trigger | width | value |")
    md.append("|---:|---:|---:|---:|")
    for i, (_, r) in enumerate(s.iterrows(), start=1):
        md.append(f"| {i} | {r['trigger']} | {r['width']} | {f4(r['value'])} |")
    md.append("")
# baseline reference
base_mr = sub[(sub["metric"] == "mean_r") & (sub["trigger"] == 1.0) & (sub["width"] == 0.75)]["value"].iloc[0]
base_wl = sub[(sub["metric"] == "win_loss") & (sub["trigger"] == 1.0) & (sub["width"] == 0.75)]["value"].iloc[0]
base_b1 = sub[(sub["metric"] == "bin1_share") & (sub["trigger"] == 1.0) & (sub["width"] == 0.75)]["value"].iloc[0]
md.append(f"**Baseline reference (trigger=1.0, width=0.75):** mean_r={f4(base_mr)} | win:loss={f3(base_wl)} | bin1_share={base_b1:.2%}\n")
md.append("Plots: `plots/b3_heatmap_mean_r_slice_{A,B,C}_{cluster_1,all}.png`.")
md.append("")

# B3 full pivot for cluster 1 mean_r
md.append("**Full grid: mean_r per trade — slice B cluster 1**\n")
piv = pivot_b_metric(b3, "B", "cluster_1", "mean_r")
md.append("| trigger \\ width | " + " | ".join(f"{w}" for w in piv.columns) + " |")
md.append("|---|" + "|".join(["---:"] * len(piv.columns)) + "|")
for t in sorted(piv.index):
    md.append(f"| **{t}** | " + " | ".join(f4(piv.loc[t, w]) for w in piv.columns) + " |")
md.append("")

md.append("CSVs: `b1_trail_trigger_sweep.csv`, `b2_trail_width_sweep.csv`, `b3_grid_2d.csv`.")
md.append("")

# ============================================================
# Phase C
# ============================================================
md.append("## Phase C — Delayed entry sweeps\n")

md.append("### C1. Bar-1 confirmation threshold (entry at bar 2, baseline exit policy) — slice B cluster 1\n")
md.append("R-frame: 2×ATR(signal_bar). `close_r_at_t1` is the bar-1 close as a fraction of R from original entry. Admit if ≥ threshold; otherwise skip. Admitted trades re-entered at bar-2 open, simulated through end of path with §11 row 2 (trigger=1.0, width=0.75) against the new entry.\n")
md.append("| threshold | n_admitted | admission_rate | mean_r_admitted | mean_r_full_pop | win_loss | recall_winners | specificity_losers | precision |")
md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
piv_c1 = pivot_c1_metric(c1, "B", "cluster_1")
for thr in sorted(piv_c1.index):
    r = piv_c1.loc[thr]
    md.append(f"| {thr} | {int(r['n_admitted'])} | {r['admission_rate']:.2%} | {f4(r['mean_r_admitted'])} | {f4(r['mean_r_full_population'])} | {f3(r['win_loss_admitted'])} | {r['recall_baseline_winners']:.2%} | {r['specificity_baseline_losers']:.2%} | {r['precision_admitted_winners']:.2%} |")
md.append("")

# All slices C1 headline
md.append("**C1 mean_r_full_population by slice × threshold (overall, not cluster-restricted)**\n")
piv_full = c1[(c1["cluster"] == "all") & (c1["metric"] == "mean_r_full_population")].pivot(index="threshold", columns="slice", values="value")
piv_full.index = piv_full.index.astype(float)
piv_full = piv_full.sort_index()
md.append("| threshold | " + " | ".join(piv_full.columns) + " |")
md.append("|---:|" + "|".join(["---:"] * len(piv_full.columns)) + "|")
for thr in piv_full.index:
    md.append(f"| {thr} | " + " | ".join(f4(piv_full.loc[thr, s]) for s in piv_full.columns) + " |")
md.append("")

md.append("CSV: `c1_delayed_entry_sweep.csv`.")
md.append("")

# C2
md.append("### C2. Bar-1 + bar-2 confirmation (supplementary) — slice B cluster 1\n")
md.append("Entry at bar 3 if `close_r_at_t1 ≥ t1` AND `close_r_at_t2 ≥ t2`.\n")
md.append("| t1 | t2 | n_admitted | admission_rate | mean_r_admitted | mean_r_full_pop | win_loss | precision |")
md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
sub_c2 = c2[(c2["slice"] == "B") & (c2["cluster"] == "cluster_1")]
piv_c2 = sub_c2.pivot(index="threshold", columns="metric", values="value")
for thr in piv_c2.index:
    r = piv_c2.loc[thr]
    # parse t1/t2 from descriptor like "t1=0.0;t2=0.1"
    parts = dict(p.split("=") for p in thr.split(";"))
    md.append(f"| {parts['t1']} | {parts['t2']} | {int(r['n_admitted'])} | {r['admission_rate']:.2%} | {f4(r['mean_r_admitted'])} | {f4(r['mean_r_full_population'])} | {f3(r['win_loss_admitted'])} | {r['precision_admitted_winners']:.2%} |")
md.append("")
md.append("CSV: `c2_two_bar_confirmation.csv`.")
md.append("")

# C3
md.append("### C3. Combined: best exit policy × best delayed entry, per fold (slice C cluster 1)\n")
# load best from c3 csv
best_trigger = None
best_width = None
best_threshold = None
with open(OUT_DIR / "c3_combined_per_fold.csv") as f:
    pass
# We'll just describe in doc
b3_b_c1 = b3[(b3["slice"] == "B") & (b3["cluster"] == "cluster_1") & (b3["metric"] == "mean_r")].sort_values("value", ascending=False).iloc[0]
best_trigger = float(b3_b_c1["trigger"])
best_width = float(b3_b_c1["width"])
c1_b_c1 = c1[(c1["slice"] == "B") & (c1["cluster"] == "cluster_1") & (c1["metric"] == "mean_r_full_population")].sort_values("value", ascending=False).iloc[0]
best_threshold = float(c1_b_c1["threshold"])

md.append(f"Best exit policy for slice B cluster 1 by mean R: **trigger = {best_trigger}, width = {best_width}** (mean R = {f4(b3_b_c1['value'])}).")
md.append(f"Best delayed entry threshold by mean_r_full_population (slice B cluster 1): **threshold = {best_threshold}** (mean_r_full_pop = {f4(c1_b_c1['value'])}).\n")
md.append("Combined: delayed entry at bar 2 (threshold {0}) + best exit (trigger {1}, width {2}). Slice C cluster 1 trades only (refit-admitted F2–F7).\n".format(best_threshold, best_trigger, best_width))

md.append("| fold | n_baseline | n_admitted | adm_rate | baseline mean R | combined mean R (admitted) | combined mean R (full pop) | baseline ann ROI % | combined ann ROI % | baseline DD R | combined DD R |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for _, r in c3.iterrows():
    md.append(
        f"| {r['fold']} | {fi(r['n_baseline'])} | {fi(r['n_combined_admitted'])} | "
        f"{r['admission_rate']:.2%} | {f4(r['baseline_mean_r'])} | "
        f"{f4(r['combined_mean_r_admitted'])} | {f4(r['combined_mean_r_full_pop'])} | "
        f"{f3(r['baseline_ann_roi_pct_at_0.20pct_risk'])} | {f3(r['combined_ann_roi_pct_at_0.20pct_risk'])} | "
        f"{f3(r['baseline_max_dd_r_seq'])} | {f3(r['combined_max_dd_r_seq'])} |"
    )
md.append("")
md.append("CSV: `c3_combined_per_fold.csv`.")
md.append("")
md.append("**Notes on C3 metrics:**")
md.append("- `baseline ann ROI` and DD use a coarse approximation: total R × 0.20% risk × 12/9 months (fold OOS ≈ 9 months). This is a sequence-DD on closed-trade cumulative PnL, not the convention (b) MTM DD that Step 5B-spread used. For exact pass-deployable comparison, the protocol-grade WFO engine run on the new policy would be required — this is read-only re-simulation, not an engine run.")
md.append("- The combined-policy max DD is sequence-ordered by trade_id, not entry timestamp; the column should be read as a relative comparison between baseline and combined within this approximation, not as absolute drawdown estimates.")
md.append("")

# ============================================================
# Data quality
# ============================================================
md.append("## Data quality / assumptions\n")
md.append("- **Baseline reproducibility check:** vectorized simulator reproduced the per-trade-by-per-trade §11 row 2 simulation from `per_trade_path_metrics.csv` exactly (10,764/10,764 trades match on final_r, exit_bar, exit_reason).")
md.append("- **R-frame.** All path-based re-simulation runs in 2×ATR(14)_signal_bar (the path file's normalization). Slice C `peak_mfe_r` baseline numbers are in 3×ATR cluster_R frame from the refit per-trade file; A1 converts them to 2×ATR via the multiplier 1.5 for cross-slice extension comparison.")
md.append("- **Path arrays cache.** `path_arrays.npz` (~110MB compressed) holds dense [10764, 241] matrices for close_r/mfe_so_far_r/mae_so_far_r/high_r/low_r/open_r. Every trade has full 241 bars (path file produced full forward window even after step1 SL hit).")
md.append("- **Within-bar ordering.** Conservative: stop hit fires against stop level entering the bar, before any MFE update. Same as baseline simulator.")
md.append("- **§11 row 2 invariants.** sl_initial exits always have `trail_triggered = 0` and `final_r = −1.0`. Trail exits have giveback = 0.75R exactly by construction (under baseline width). Bin 4 (modest losers, −1 < r ≤ −0.5) is policy-inaccessible under §11 row 2.")
md.append("- **Delayed entry R-frame.** R kept at 2×ATR(signal_bar) under delayed entry to preserve cross-policy comparability. The new \"close_r\" relative to bar-2 entry is computed as `close_r_original − close_r_at_t1`; MFE/MAE under delayed entry are running max/min of this close-based series (intrabar high/low after delayed entry are NOT used in re-derived MFE/MAE — this is a close-based approximation for the delayed-entry path).")
md.append("- **C2 cluster scope.** C2 was run for `all` and `cluster_1` only per prompt (focus). Full-cluster grid available in `c2_two_bar_confirmation.csv`.")
md.append("- **C3 fold annualised ROI / DD** use coarse approximations (see C3 notes above). Sequence-DD is by trade_id order, not by entry timestamp. For protocol-grade evaluation a proper WFO engine run is required.")
md.append("- **A2 close-based metrics.** Pre/post-lock pullback magnitude and other path metrics are computed from `close_r` (close-based), not intrabar high/low. This matches the §11 row 2 lock-bar definition (lock_bar is the first bar where running MFE reaches 1R), but means intrabar excursion is not captured in A2 specifically.")
md.append("- **A4 cluster path-shape metrics** are computed in the §11 row 2 alive-window frame from `per_trade_path_metrics.csv`. These are not centroid features from §3 clustering (which used a different feature set in 1×ATR frame).")
md.append("- **A3 hit_rate definition.** Hit rate = unique trades that are bin1 ∪ bin2 ∪ bin6 (T3-clean overlay), as a fraction of N. Bin6 can overlap bin1 or bin2; hit_rate counts unique trade ids.")
md.append("- **Recall / specificity / precision** in C1/C2 are computed against the baseline §11 row 2 outcome labels (bin1+bin2 = winners; bin5 = losers). They are descriptive only — outcome labels are not used in filter design.")

md.append("")
md.append("## Artefact index\n")
md.append("- `EXIT_ENTRY_SWEEP.md` — this doc")
md.append("- `path_arrays.npz` — dense numpy cache from `trades_paths.csv`")
md.append("- Phase A: `a1_trail_extension.csv`, `a1_bars_between_peaks.csv`, `a2_pre_post_lock.csv`, `a2_paired_tests.csv`, `a3_win_loss_metrics.csv`, `a4_per_cluster_metrics.csv`, `a5_winners_vs_losers.csv`")
md.append("- Phase B: `b1_trail_trigger_sweep.csv`, `b2_trail_width_sweep.csv`, `b3_grid_2d.csv`")
md.append("- Phase C: `c1_delayed_entry_sweep.csv`, `c2_two_bar_confirmation.csv`, `c3_combined_per_fold.csv`")
md.append("- `plots/` — A2 pullback hist, A3 final_r hist, A4 per-cluster MAE plots (4), B3 heatmaps (6 = 3 slices × 2 cluster scopes)")

(OUT_DIR / "EXIT_ENTRY_SWEEP.md").write_text("\n".join(md), encoding="utf-8")
print(f"[done] EXIT_ENTRY_SWEEP.md ({len(md)} lines)")
