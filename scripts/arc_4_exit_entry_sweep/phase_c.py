"""Phase C — delayed entry sweeps.

C1: bar-1 confirmation threshold sweep (entry at bar 2 if close_r_at_t1 >= threshold)
C2: bar-1 + bar-2 confirmation (entry at bar 3 if both pass) — 2D grid
C3: combined best exit policy × best delayed entry, per fold for cluster 1
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from lib_sim import load_path_arrays, simulate_policy_vectorized, EXIT_CODE_TO_LABEL

REPO = THIS.parents[2]
OUT_DIR = REPO / "results" / "arc_4_exit_entry_sweep"
PLOT_DIR = OUT_DIR / "plots"

print("[info] loading...")
arr = load_path_arrays(OUT_DIR / "path_arrays.npz")
trade_ids = arr["trade_ids"]
n_trades = len(trade_ids)

clusters = pd.read_csv(REPO / "results" / "l_arc_4" / "step2" / "clusters_K4.csv").set_index("trade_id")
cluster_of = clusters.reindex(trade_ids)["cluster_id"].to_numpy()

refit = pd.read_csv(REPO / "results" / "l_arc_4" / "step5c" / "per_trade_simulated_refit_1.csv")
refit_ids_set = set(refit["trade_id"].tolist())
refit_mask = np.array([tid in refit_ids_set for tid in trade_ids])

slice_mask = {
    "A": np.ones(n_trades, dtype=bool),
    "B": cluster_of == 1,
    "C": refit_mask,
    "D": cluster_of != 1,
}

# Baseline (§11 row 2, original entry at bar 0)
print("[info] baseline at original entry...")
baseline = simulate_policy_vectorized(
    close_r=arr["close_r"],
    mfe_so_far_r=arr["mfe_so_far_r"],
    mae_so_far_r=arr["mae_so_far_r"],
    initial_sl=-1.0,
    trail_trigger=1.0,
    trail_width=0.75,
    time_exit_bar=240,
)
baseline_final = baseline["final_r"]


def outcome_bins(final_r, max_mae_r):
    primary = np.where(
        final_r >= 1.5, 1,
        np.where(final_r >= 0.5, 2,
                 np.where(final_r > -0.5, 3,
                          np.where(final_r > -1.0, 4, 5)))
    )
    t3_clean = ((final_r >= 0.5) & (max_mae_r > -0.75)).astype(int)
    return primary, t3_clean


def aggregate_admission_sim(slice_name, cluster_label, threshold_descriptor,
                            admitted_mask_within_slice, total_in_slice,
                            final_r_admitted, peak_admitted, mae_admitted,
                            baseline_final_in_slice, baseline_winners_mask,
                            baseline_losers_mask):
    """Aggregate metrics for a (slice, cluster, threshold) cell."""
    n_admitted = int(admitted_mask_within_slice.sum())
    admission_rate = n_admitted / total_in_slice if total_in_slice else np.nan
    rows = []
    rows.append({"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
                 "metric": "n_admitted", "value": n_admitted})
    rows.append({"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
                 "metric": "n_slice_total", "value": int(total_in_slice)})
    rows.append({"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
                 "metric": "admission_rate", "value": float(admission_rate)})
    if n_admitted == 0:
        rows.append({"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
                     "metric": "mean_r_admitted", "value": np.nan})
        rows.append({"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
                     "metric": "mean_r_full_population", "value": 0.0})
        return rows
    primary, t3 = outcome_bins(final_r_admitted, mae_admitted)
    bin1 = int((primary == 1).sum())
    bin2 = int((primary == 2).sum())
    bin5 = int((primary == 5).sum())
    bin6 = int(t3.sum())
    hit_rate = ((primary == 1) | (primary == 2) | (t3 == 1)).mean()
    win_loss = (bin1 + bin2) / bin5 if bin5 else np.nan
    mean_r_adm = float(np.nanmean(final_r_admitted))
    pos = final_r_admitted[final_r_admitted > 0].sum()
    neg = -final_r_admitted[final_r_admitted < 0].sum()
    pf = float(pos / neg) if neg > 0 else np.nan
    # Per-trade across full population: admitted contribute their final_r, non-admitted contribute 0
    mean_r_full = float(final_r_admitted.sum() / total_in_slice) if total_in_slice else np.nan

    # Winner recall: % of baseline winners (bins 1+2) that survived this filter
    # baseline_winners_mask is for the full slice
    admitted_idx_in_slice = np.where(admitted_mask_within_slice)[0]
    recall_winners = float(admitted_mask_within_slice[baseline_winners_mask].mean()) if baseline_winners_mask.any() else np.nan
    # Specificity: % of baseline losers (bin 5) rejected
    rejected_mask = ~admitted_mask_within_slice
    specificity_losers = float(rejected_mask[baseline_losers_mask].mean()) if baseline_losers_mask.any() else np.nan
    # Precision: among admitted, what fraction are baseline winners (bin1+bin2)
    if n_admitted > 0:
        precision = float(baseline_winners_mask[admitted_mask_within_slice].mean())
    else:
        precision = np.nan

    rows.extend([
        {"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
         "metric": "bin1_admitted", "value": bin1},
        {"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
         "metric": "bin2_admitted", "value": bin2},
        {"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
         "metric": "bin5_admitted", "value": bin5},
        {"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
         "metric": "bin6_admitted", "value": bin6},
        {"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
         "metric": "hit_rate_admitted", "value": float(hit_rate)},
        {"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
         "metric": "win_loss_admitted", "value": float(win_loss) if not np.isnan(win_loss) else np.nan},
        {"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
         "metric": "mean_r_admitted", "value": mean_r_adm},
        {"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
         "metric": "profit_factor_admitted", "value": pf},
        {"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
         "metric": "mean_r_full_population", "value": mean_r_full},
        {"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
         "metric": "recall_baseline_winners", "value": recall_winners},
        {"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
         "metric": "specificity_baseline_losers", "value": specificity_losers},
        {"slice": slice_name, "cluster": cluster_label, "threshold": threshold_descriptor,
         "metric": "precision_admitted_winners", "value": precision},
    ])
    return rows


# Precompute baseline winners/losers masks for filtering precision/recall/specificity
baseline_primary, baseline_t3 = outcome_bins(baseline_final, baseline["max_mae_alive"])
baseline_winners_full = (baseline_primary == 1) | (baseline_primary == 2)
baseline_losers_full = baseline_primary == 5


# ============================================================
# C1: bar-1 confirmation sweep
# ============================================================

print("[info] C1 sweep...")
# close_r at bar 1 (already in 2*ATR R-frame from path arrays)
close_r_at_t1 = arr["close_r"][:, 1]  # N x 1
# Note: bar 0 close_r is close at bar 0 relative to entry_price (small value, ~0.01R)
# For delayed entry: enter at bar 2 open. R-frame stays 2*ATR(signal_bar), simulate from bar 2.

# Use lib_sim with start_bar=2 and entry_close_r = close_r[:, 1] (the bar-1 close as new origin)
# Then simulate from bar 2 onward

thresholds_c1 = [-0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
c1_rows = []
c1_simulated = {}  # threshold -> (admitted_mask_global, final_r_global_or_nan)

for threshold in thresholds_c1:
    admit_global = close_r_at_t1 >= threshold  # boolean over all trades
    # Re-simulate from bar 2 for admitted trades only — but vectorized still runs on all
    sim = simulate_policy_vectorized(
        close_r=arr["close_r"],
        mfe_so_far_r=arr["mfe_so_far_r"],
        mae_so_far_r=arr["mae_so_far_r"],
        initial_sl=-1.0,
        trail_trigger=1.0,
        trail_width=0.75,
        time_exit_bar=240,
        start_bar=2,
        entry_close_r=close_r_at_t1,
    )
    # Mask non-admitted
    sim_final = np.where(admit_global, sim["final_r"], np.nan)
    sim_peak = np.where(admit_global, sim["peak_mfe_alive"], np.nan)
    sim_mae = np.where(admit_global, sim["max_mae_alive"], np.nan)
    c1_simulated[threshold] = (admit_global, sim_final)

    for sname, smask in slice_mask.items():
        for cluster_id in [None, 0, 1, 2, 3]:
            if cluster_id is None:
                cmask = smask
                clabel = "all"
            else:
                cmask = smask & (cluster_of == cluster_id)
                clabel = f"cluster_{cluster_id}"
            total = int(cmask.sum())
            if total == 0:
                continue
            admit_in_slice = admit_global & cmask
            admit_mask_within_slice = admit_global[cmask]
            baseline_winners_slice = baseline_winners_full[cmask]
            baseline_losers_slice = baseline_losers_full[cmask]
            final_r_adm = sim["final_r"][admit_in_slice]
            peak_adm = sim["peak_mfe_alive"][admit_in_slice]
            mae_adm = sim["max_mae_alive"][admit_in_slice]
            rows = aggregate_admission_sim(
                sname, clabel, str(threshold),
                admit_mask_within_slice, total,
                final_r_adm, peak_adm, mae_adm,
                baseline_final[cmask], baseline_winners_slice, baseline_losers_slice,
            )
            c1_rows.extend(rows)
    print(f"  threshold {threshold}: admitted {admit_global.sum()}/{n_trades}")

pd.DataFrame(c1_rows).to_csv(OUT_DIR / "c1_delayed_entry_sweep.csv", index=False)
print(f"[done] c1_delayed_entry_sweep.csv ({len(c1_rows)})")

# ============================================================
# C2: bar-1 + bar-2 confirmation (run regardless — definition of done lists it)
# ============================================================

print("[info] C2 sweep...")
close_r_at_t2 = arr["close_r"][:, 2]
c2_rows = []
t1_thr = [0.0, 0.1, 0.2]
t2_thr = [0.0, 0.1, 0.2]

for thr1 in t1_thr:
    for thr2 in t2_thr:
        admit_global = (close_r_at_t1 >= thr1) & (close_r_at_t2 >= thr2)
        sim = simulate_policy_vectorized(
            close_r=arr["close_r"],
            mfe_so_far_r=arr["mfe_so_far_r"],
            mae_so_far_r=arr["mae_so_far_r"],
            initial_sl=-1.0,
            trail_trigger=1.0,
            trail_width=0.75,
            time_exit_bar=240,
            start_bar=3,
            entry_close_r=close_r_at_t2,
        )
        for sname, smask in slice_mask.items():
            for cluster_id in [None, 1]:  # focus on overall + cluster 1
                if cluster_id is None:
                    cmask = smask
                    clabel = "all"
                else:
                    cmask = smask & (cluster_of == cluster_id)
                    clabel = f"cluster_{cluster_id}"
                total = int(cmask.sum())
                if total == 0:
                    continue
                admit_mask_within_slice = admit_global[cmask]
                baseline_winners_slice = baseline_winners_full[cmask]
                baseline_losers_slice = baseline_losers_full[cmask]
                admit_in_slice = admit_global & cmask
                final_r_adm = sim["final_r"][admit_in_slice]
                peak_adm = sim["peak_mfe_alive"][admit_in_slice]
                mae_adm = sim["max_mae_alive"][admit_in_slice]
                rows = aggregate_admission_sim(
                    sname, clabel, f"t1={thr1};t2={thr2}",
                    admit_mask_within_slice, total,
                    final_r_adm, peak_adm, mae_adm,
                    baseline_final[cmask], baseline_winners_slice, baseline_losers_slice,
                )
                c2_rows.extend(rows)
        print(f"  t1={thr1} t2={thr2}: admitted {admit_global.sum()}")

pd.DataFrame(c2_rows).to_csv(OUT_DIR / "c2_two_bar_confirmation.csv", index=False)
print(f"[done] c2_two_bar_confirmation.csv ({len(c2_rows)})")


# ============================================================
# C3: combined best exit × best delayed entry, per fold for cluster 1
# ============================================================

print("[info] C3: combined best exit × best delayed entry on slice C scope (F2-F7)...")
# Pick best (trigger, width) for cluster 1 by mean R
b3 = pd.read_csv(OUT_DIR / "b3_grid_2d.csv")
sub = b3[(b3["slice"] == "B") & (b3["cluster"] == "cluster_1") & (b3["metric"] == "mean_r")]
best_row = sub.sort_values("value", ascending=False).iloc[0]
best_trigger = float(best_row["trigger"])
best_width = float(best_row["width"])
print(f"  best exit policy for slice B cluster 1: trigger={best_trigger}, width={best_width}, mean R={best_row['value']:.4f}")

# Pick best threshold from C1 by mean_r_full_population (i.e., expected ROI contribution)
c1 = pd.read_csv(OUT_DIR / "c1_delayed_entry_sweep.csv")
sub_c1 = c1[(c1["slice"] == "B") & (c1["cluster"] == "cluster_1") & (c1["metric"] == "mean_r_full_population")]
best_thr_row = sub_c1.sort_values("value", ascending=False).iloc[0]
best_threshold = float(best_thr_row["threshold"])
print(f"  best delayed entry threshold for slice B cluster 1: {best_threshold} (mean_r_full_population={best_thr_row['value']:.4f})")

# Combined simulation on slice C (refit-admitted F2-F7)
admit_global = close_r_at_t1 >= best_threshold
combined_sim = simulate_policy_vectorized(
    close_r=arr["close_r"],
    mfe_so_far_r=arr["mfe_so_far_r"],
    mae_so_far_r=arr["mae_so_far_r"],
    initial_sl=-1.0,
    trail_trigger=best_trigger,
    trail_width=best_width,
    time_exit_bar=240,
    start_bar=2,
    entry_close_r=close_r_at_t1,
)

# Baseline (§11 row 2 default) for slice C cluster 1 trades
# We need to attach to per-fold info from refit
fold_assignments = refit.set_index("trade_id")["fold"]
fold_of = pd.Series(np.full(n_trades, -1), index=trade_ids)
fold_of.loc[fold_assignments.index] = fold_assignments.values

# Restrict to slice C cluster 1 trades
slice_C_c1_mask = slice_mask["C"] & (cluster_of == 1)
print(f"  slice C cluster 1 trades: {int(slice_C_c1_mask.sum())}")

# Per-fold metrics
c3_rows = []
for fold in [2, 3, 4, 5, 6, 7]:
    fold_mask = slice_C_c1_mask & (fold_of.values == fold)
    n_in_fold = int(fold_mask.sum())
    # Baseline: original entry, baseline §11 row 2
    base_final_fold = baseline_final[fold_mask]
    base_mean = float(np.nanmean(base_final_fold)) if n_in_fold else np.nan
    # Combined: delayed entry + best exit
    admit_fold = admit_global & fold_mask
    n_admit = int(admit_fold.sum())
    comb_final_admit = combined_sim["final_r"][admit_fold]
    comb_mean_admit = float(np.nanmean(comb_final_admit)) if n_admit else np.nan
    comb_mean_full = float(np.nansum(comb_final_admit) / n_in_fold) if n_in_fold else np.nan
    # Annualised ROI at 0.20% risk — approximate: fold OOS window ~9 months for F2..F7
    # Use generic: ann_roi = sum(final_r) * risk_pct, scaled to 12 months
    fold_months = 9  # approximate
    base_total_r = float(np.nansum(base_final_fold))
    comb_total_r = float(np.nansum(comb_final_admit))
    risk_pct = 0.20  # 0.20% per trade
    base_ann_roi = base_total_r * risk_pct * (12 / fold_months)
    comb_ann_roi = comb_total_r * risk_pct * (12 / fold_months)
    # Quick max DD: convention (a) closed-trade ordering (sort by exit_time approximated by exit_bar+entry_ts)
    # Use cumulative sum of final_r as PnL trajectory
    # For baseline:
    if n_in_fold > 0:
        # Order by entry timestamp - we don't have it indexed here; use trade order
        bf = base_final_fold[~np.isnan(base_final_fold)]
        cum = np.cumsum(bf)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        base_dd_r = float(dd.max())
    else:
        base_dd_r = np.nan
    if n_admit > 0:
        cf = comb_final_admit[~np.isnan(comb_final_admit)]
        cum = np.cumsum(cf)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        comb_dd_r = float(dd.max())
    else:
        comb_dd_r = np.nan
    base_dd_pct = base_dd_r * risk_pct if not np.isnan(base_dd_r) else np.nan
    comb_dd_pct = comb_dd_r * risk_pct if not np.isnan(comb_dd_r) else np.nan
    c3_rows.append({
        "fold": fold,
        "n_baseline": n_in_fold,
        "n_combined_admitted": n_admit,
        "admission_rate": n_admit / n_in_fold if n_in_fold else np.nan,
        "baseline_mean_r": base_mean,
        "combined_mean_r_admitted": comb_mean_admit,
        "combined_mean_r_full_pop": comb_mean_full,
        "baseline_total_r": base_total_r,
        "combined_total_r": comb_total_r,
        "baseline_ann_roi_pct_at_0.20pct_risk": base_ann_roi,
        "combined_ann_roi_pct_at_0.20pct_risk": comb_ann_roi,
        "baseline_max_dd_r_seq": base_dd_r,
        "combined_max_dd_r_seq": comb_dd_r,
        "baseline_max_dd_pct_at_0.20pct_risk": base_dd_pct,
        "combined_max_dd_pct_at_0.20pct_risk": comb_dd_pct,
    })

# Add overall row
all_mask = slice_C_c1_mask
base_final_all = baseline_final[all_mask]
admit_all = admit_global & all_mask
comb_final_all = combined_sim["final_r"][admit_all]
c3_rows.append({
    "fold": "ALL_F2_F7",
    "n_baseline": int(all_mask.sum()),
    "n_combined_admitted": int(admit_all.sum()),
    "admission_rate": float(admit_all.sum() / all_mask.sum()) if all_mask.sum() else np.nan,
    "baseline_mean_r": float(np.nanmean(base_final_all)),
    "combined_mean_r_admitted": float(np.nanmean(comb_final_all)) if admit_all.sum() else np.nan,
    "combined_mean_r_full_pop": float(np.nansum(comb_final_all) / all_mask.sum()),
    "baseline_total_r": float(np.nansum(base_final_all)),
    "combined_total_r": float(np.nansum(comb_final_all)),
    "baseline_ann_roi_pct_at_0.20pct_risk": np.nan,
    "combined_ann_roi_pct_at_0.20pct_risk": np.nan,
    "baseline_max_dd_r_seq": np.nan,
    "combined_max_dd_r_seq": np.nan,
    "baseline_max_dd_pct_at_0.20pct_risk": np.nan,
    "combined_max_dd_pct_at_0.20pct_risk": np.nan,
})

pd.DataFrame(c3_rows).to_csv(OUT_DIR / "c3_combined_per_fold.csv", index=False)
print(f"[done] c3_combined_per_fold.csv")
print(f"[info] best policy: trigger={best_trigger}, width={best_width}, threshold={best_threshold}")
print("[done] Phase C complete")
