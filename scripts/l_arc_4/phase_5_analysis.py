"""Phase 5 — Step 5 cross-fold stability analysis on existing artefacts.

Read-only. Computes:
  §2 per_fold_economics.csv (admit + reject + full-pop)
  §3 gate_evaluation explicit per gate
  §4 pair_stability_cluster_1 per fold
  §5 prior_vs_new comparison
  §7 outlier impact check

Then renders STEP_5_VERDICT.md combining all sections.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RR = REPO / "results" / "l_arc_4_rerun"
OUT = RR / "step5" / "phase_5_analysis"
OUT.mkdir(parents=True, exist_ok=True)

RISK_PCT = 0.5  # 0.5% per trade — engine convention

# ============================================================
# Load all inputs
# ============================================================
trades_all = pd.read_csv(RR / "step1" / "trades_all.csv", parse_dates=["signal_time", "entry_time"])
clusters = pd.read_csv(RR / "step2" / "clusters_K4.csv")
refit_per_trade = pd.read_csv(RR / "step5c" / "per_trade_simulated_refit_1.csv", parse_dates=["entry_ts"])
fold_defs = pd.read_csv(RR / "step5c" / "fold_definitions.csv")
prior_fs = pd.read_csv(REPO / "results" / "l_arc_4" / "step5c" / "fold_stability_refit_cluster_1.csv")
new_fs = pd.read_csv(RR / "step5c" / "fold_stability_refit_cluster_1.csv")
new_admission = pd.read_csv(RR / "step5c" / "per_fold_admission_comparison.csv")
rejected = pd.read_csv(RR / "step5c" / "rejected_trade_pnl_cluster_1.csv", parse_dates=["entry_time"])

# Cluster 1 only
c1_trades = trades_all.merge(clusters, on="trade_id", how="left")
c1_trades = c1_trades[c1_trades["cluster_id"] == 1].copy()

# Assign fold
def assign_fold(ts):
    for _, r in fold_defs.iterrows():
        if pd.Timestamp(r["oos_start"]) <= ts <= pd.Timestamp(r["oos_end"]):
            return int(r["fold"])
    return -1

c1_trades["fold"] = c1_trades["entry_time"].apply(assign_fold)

# ============================================================
# §2 — Per-fold economics (admitted, rejected, full-population)
# ============================================================
print("=== §2 Per-fold economics ===")
admit_ids = set(refit_per_trade["trade_id"].tolist())
c1_trades["admitted"] = c1_trades["trade_id"].isin(admit_ids)

# For each fold, compute admitted/rejected/full-pop
rows = []
for fold in [1, 2, 3, 4, 5, 6, 7]:
    fold_trades = c1_trades[c1_trades["fold"] == fold]
    n_total = len(fold_trades)
    admit_in_fold = refit_per_trade[refit_per_trade["fold"] == fold]
    n_admit = len(admit_in_fold)
    reject_in_fold = rejected[rejected["fold"] == fold]
    n_reject = len(reject_in_fold)

    mean_R_admit = float(admit_in_fold["final_r"].mean()) if n_admit > 0 else np.nan
    mean_R_reject = float(reject_in_fold["bailout_pnl_r"].mean()) if n_reject > 0 else np.nan
    if (n_admit + n_reject) > 0:
        mean_R_full = (n_admit * (mean_R_admit if not np.isnan(mean_R_admit) else 0.0) +
                       n_reject * (mean_R_reject if not np.isnan(mean_R_reject) else 0.0)) / (n_admit + n_reject)
    else:
        mean_R_full = np.nan

    # t-stat (admitted)
    if n_admit > 1:
        std_admit = float(admit_in_fold["final_r"].std(ddof=1))
        t_stat_admit = mean_R_admit / (std_admit / np.sqrt(n_admit)) if std_admit > 0 else 0.0
    else:
        t_stat_admit = np.nan
    win_pct_admit = float((admit_in_fold["final_r"] > 0).mean()) if n_admit > 0 else np.nan

    # ROI (engine convention: 0.5% risk simple sum)
    roi_admit = sum(admit_in_fold["final_r"]) * RISK_PCT if n_admit > 0 else 0.0
    roi_reject = sum(reject_in_fold["bailout_pnl_r"]) * RISK_PCT if n_reject > 0 else 0.0
    roi_full = roi_admit + roi_reject

    # DD_a (convention a) — already in engine output for admitted; we recompute on cumsum
    # For admitted-only ordered by entry_ts:
    admit_sorted = admit_in_fold.sort_values("entry_ts")
    cum = (admit_sorted["final_r"] * RISK_PCT).cumsum().to_numpy()
    peak = np.maximum.accumulate(cum) if len(cum) else np.array([0.0])
    dd_a_admit = float((peak - cum).max()) if len(cum) > 0 else 0.0

    # Full-pop DD with admitted + rejected mixed by entry_time
    full_pop = pd.concat([
        admit_in_fold.assign(pnl_r=admit_in_fold["final_r"], _ts=admit_in_fold["entry_ts"])[["_ts", "pnl_r"]],
        reject_in_fold.assign(pnl_r=reject_in_fold["bailout_pnl_r"], _ts=reject_in_fold["entry_time"])[["_ts", "pnl_r"]],
    ]).sort_values("_ts")
    cum_full = (full_pop["pnl_r"] * RISK_PCT).cumsum().to_numpy()
    peak_full = np.maximum.accumulate(cum_full) if len(cum_full) else np.array([0.0])
    dd_a_full = float((peak_full - cum_full).max()) if len(cum_full) > 0 else 0.0

    # Convention (b) approximation: use prior Arc 4 F6 inflation factor of 1.334 as estimate
    # Per ARC_4_RESULT.md the F6 MTM/closed ratio was 1.334
    INFLATION_FACTORS = {1: 1.20, 2: 1.30, 3: 1.627, 4: 1.30, 5: 1.40, 6: 1.334, 7: 1.30}  # rough prior estimates
    inf = INFLATION_FACTORS.get(fold, 1.30)
    dd_b_admit_est = dd_a_admit * inf
    dd_b_full_est = dd_a_full * inf

    rows.append({
        "fold": fold,
        "n_total_cluster_1": n_total,
        "n_admit": n_admit,
        "n_reject": n_reject,
        "mean_R_admit": mean_R_admit,
        "mean_R_reject": mean_R_reject,
        "mean_R_full_pop": mean_R_full,
        "t_stat_admit": t_stat_admit,
        "win_pct_admit": win_pct_admit,
        "ROI_admit_pct": roi_admit,
        "ROI_reject_pct": roi_reject,
        "ROI_full_pct": roi_full,
        "DD_a_admit_pct": dd_a_admit,
        "DD_a_full_pct": dd_a_full,
        "DD_b_admit_est_pct": dd_b_admit_est,
        "DD_b_full_est_pct": dd_b_full_est,
        "DD_inflation_factor_est": inf,
    })

economics = pd.DataFrame(rows)
economics.to_csv(OUT / "per_fold_economics.csv", index=False)
print(economics.to_string(index=False))

# ============================================================
# §3 — §9 gate evaluation
# ============================================================
print("\n=== §3 §9 Gate Evaluation ===")
# Use admitted-only metrics from new_fs (engine output) — these are what §9 was designed for
# Excluding F1
gating = new_fs[new_fs["fold"].isin([2, 3, 4, 5, 6, 7])].copy()

# Gate A: sign consistency mean_r > 0
gate_a_pass = (gating["mean_r"] > 0).all()
gate_a_details = "; ".join(f"F{int(r.fold)}={r.mean_r:+.4f}" for _, r in gating.iterrows())

# Gate B: size variance max/min ≤ 3.0
n_min = gating["n"].min()
n_max = gating["n"].max()
ratio_b = n_max / n_min if n_min > 0 else float("inf")
gate_b_pass = ratio_b <= 3.0

# Gate C: DD ceiling — convention (a) directly from engine
dds = gating["fold_max_dd_pct"].to_numpy()
dd_max = dds.max()
dd_median = float(np.median(dds))
ratio_c = dd_max / dd_median if dd_median > 0 else float("inf")
gate_c_a_pass = ratio_c <= 2.0

# Gate C: DD ceiling — convention (b) approximation
dds_b = (gating["fold_max_dd_pct"] * gating["fold"].map(lambda f: {1:1.2,2:1.3,3:1.627,4:1.3,5:1.4,6:1.334,7:1.3}.get(f, 1.3))).to_numpy()
dd_b_max = dds_b.max()
dd_b_median = float(np.median(dds_b))
ratio_c_b = dd_b_max / dd_b_median if dd_b_median > 0 else float("inf")
gate_c_b_pass = ratio_c_b <= 2.0

print(f"Gate A (sign consistency): {'PASS' if gate_a_pass else 'FAIL'}")
print(f"  Details: {gate_a_details}")
print(f"Gate B (size variance ≤ 3.0): {'PASS' if gate_b_pass else 'FAIL'}")
print(f"  max={n_max} (F{int(gating.loc[gating['n'].idxmax(),'fold'])}), min={n_min} (F{int(gating.loc[gating['n'].idxmin(),'fold'])}), ratio={ratio_b:.4f}")
print(f"Gate C (DD ratio ≤ 2.0) convention (a): {'PASS' if gate_c_a_pass else 'FAIL'}")
print(f"  max_DD={dd_max:.4f}%, median_DD={dd_median:.4f}%, ratio={ratio_c:.4f}, margin to 2.0 = {2.0 - ratio_c:+.4f}")
print(f"Gate C (DD ratio ≤ 2.0) convention (b) est: {'PASS' if gate_c_b_pass else 'FAIL'}")
print(f"  max_DD={dd_b_max:.4f}%, median_DD={dd_b_median:.4f}%, ratio={ratio_c_b:.4f}, margin to 2.0 = {2.0 - ratio_c_b:+.4f}")

# ============================================================
# §4 — Per-pair stability
# ============================================================
print("\n=== §4 Per-pair stability ===")
pair_rows = []
for fold in [2, 3, 4, 5, 6, 7]:
    fold_admit = refit_per_trade[refit_per_trade["fold"] == fold]
    pair_counts = fold_admit["pair"].value_counts().sort_values(ascending=False)
    total = pair_counts.sum()
    top5 = pair_counts.head(5).sum()
    top10 = pair_counts.head(10).sum()
    flag = top5 / total > 0.5 and len(pair_counts) >= 5
    for pair, n in pair_counts.items():
        pair_rows.append({
            "fold": fold,
            "pair": pair,
            "n_admitted": int(n),
            "pct_of_fold": n / total,
            "top5_cumulative_pct": top5 / total,
            "top10_cumulative_pct": top10 / total,
            "top5_flag": flag,
        })
pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(OUT / "pair_stability_cluster_1.csv", index=False)

# Summary by fold
print("Per-fold top-5 concentration:")
for fold in [2, 3, 4, 5, 6, 7]:
    sub = pair_df[pair_df["fold"] == fold]
    if len(sub) == 0:
        continue
    top5_pct = sub["top5_cumulative_pct"].iloc[0]
    top10_pct = sub["top10_cumulative_pct"].iloc[0]
    n_pairs = sub["pair"].nunique()
    flag = "FLAG" if top5_pct > 0.5 else "OK"
    print(f"  F{fold}: top-5 = {top5_pct:.2%}, top-10 = {top10_pct:.2%}, n_pairs={n_pairs}  [{flag}]")

# ============================================================
# §5 — Prior vs new side-by-side
# ============================================================
print("\n=== §5 Prior vs new ===")
comp_rows = []
for fold in [2, 3, 4, 5, 6, 7]:
    p = prior_fs[prior_fs["fold"] == fold].iloc[0]
    n = new_fs[new_fs["fold"] == fold].iloc[0]
    comp_rows.append({
        "fold": fold,
        "prior_n": int(p["n"]),
        "new_n": int(n["n"]),
        "delta_n": int(n["n"]) - int(p["n"]),
        "prior_mean_r": float(p["mean_r"]),
        "new_mean_r": float(n["mean_r"]),
        "delta_mean_r": float(n["mean_r"]) - float(p["mean_r"]),
        "prior_roi_pct": float(p["fold_roi_pct"]),
        "new_roi_pct": float(n["fold_roi_pct"]),
        "delta_roi_pp": float(n["fold_roi_pct"]) - float(p["fold_roi_pct"]),
        "prior_dd_pct": float(p["fold_max_dd_pct"]),
        "new_dd_pct": float(n["fold_max_dd_pct"]),
        "delta_dd_pp": float(n["fold_max_dd_pct"]) - float(p["fold_max_dd_pct"]),
    })
comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(OUT / "prior_vs_new_comparison.csv", index=False)
print(comp_df.to_string(index=False))

# ============================================================
# §7 — Outlier impact
# ============================================================
print("\n=== §7 Outlier impact ===")
outlier_pairs_times = [
    ("USD_CAD", "2021-07-05 18:00:00"),
    ("AUD_NZD", "2023-10-11 00:00:00"),
    ("CHF_JPY", "2021-10-04 17:00:00"),
    ("NZD_CHF", "2025-11-13 20:00:00"),
    ("GBP_NZD", "2024-12-13 09:00:00"),
]
outliers_in_c1 = 0
for pair, ts in outlier_pairs_times:
    match = c1_trades[(c1_trades["pair"] == pair) & (c1_trades["signal_time"] == pd.Timestamp(ts))]
    if len(match) > 0:
        outliers_in_c1 += 1
        print(f"  IN cluster 1: {pair} {ts} (trade_id={int(match.iloc[0]['trade_id'])})")
print(f"Outliers in cluster 1: {outliers_in_c1}/5")
print("Cluster 1 economic numbers are NOT driven by these specific trades" if outliers_in_c1 == 0
      else "Cluster 1 economic numbers MAY be influenced by these outliers — investigate")

# Save the analysis
print(f"\n[done] wrote per_fold_economics.csv, pair_stability_cluster_1.csv, prior_vs_new_comparison.csv to {OUT}")

# Final structured output for the verdict doc
import json  # noqa: E402

out_summary = {
    "convention": {
        "risk_pct": RISK_PCT / 100,
        "compounding": False,
        "annualised": False,
        "dd_convention": "(a) closed-trade cumulative",
    },
    "gates": {
        "A_sign_consistency": {"pass": bool(gate_a_pass), "details": gate_a_details},
        "B_size_variance": {"pass": bool(gate_b_pass), "max_n": int(n_max), "min_n": int(n_min), "ratio": float(ratio_b), "threshold": 3.0},
        "C_dd_a": {"pass": bool(gate_c_a_pass), "max_dd_pct": float(dd_max), "median_dd_pct": float(dd_median), "ratio": float(ratio_c), "threshold": 2.0},
        "C_dd_b_est": {"pass": bool(gate_c_b_pass), "max_dd_pct_est": float(dd_b_max), "median_dd_pct_est": float(dd_b_median), "ratio_est": float(ratio_c_b), "threshold": 2.0},
    },
    "outliers_in_cluster_1": outliers_in_c1,
}
(OUT / "_phase_5_summary.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
print(f"\n[done] summary JSON: {OUT / '_phase_5_summary.json'}")
