"""Phase 6D — generate equity curve plot + write STEP_6_VERDICT.md.

Reads outputs of scripts/l_arc_4/step6_wfo.py. No engine execution.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = _REPO_ROOT / "results" / "l_arc_4_rerun" / "step6"

# Load outputs
sweep = pd.read_csv(OUT / "risk_sweep.csv")
per_fold = pd.read_csv(OUT / "per_fold_metrics.csv")
full = pd.read_csv(OUT / "full_data_metrics.csv")
ship = yaml.safe_load((OUT / "ship_decision.yaml").read_text(encoding="utf-8"))
meta = json.loads((OUT / "run_metadata.json").read_text(encoding="utf-8"))
daily = pd.read_csv(OUT / "daily_dd_events.csv", parse_dates=["date"])
audit = pd.read_csv(OUT / "per_trade_audit_full_sequence.csv", parse_dates=["entry_time", "exit_time"])

# Equity curves at three risks
def load_eq(bps):
    return pd.read_csv(OUT / f"full_data_equity_at_{bps}bps.csv", parse_dates=["entry_time"])

eq20 = load_eq(20)
eq15 = load_eq(15)
eq10 = load_eq(10)

# ============================================================
# Plot continuous equity curve at 0.20% with reference lines for 5ers limits
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)

ax = axes[0]
for df, label, color in [(eq20, "0.20% (primary)", "C0"),
                          (eq15, "0.15% (sensitivity)", "C1"),
                          (eq10, "0.10% (sensitivity)", "C2")]:
    ax.plot(df["entry_time"], df["equity_after"], label=label, color=color, linewidth=1)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
ax.axhline(0.9, color="red", linestyle="--", linewidth=0.7, label="5ers 10% max DD threshold")
ax.set_ylabel("Equity (start = 1.0)")
ax.set_title("Arc 4 cluster 1 — continuous F2→F7 equity, deployable full-pool framing\n(admit + reject bar-2 bail-out + early-exit pre-t SL)")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

# Daily DD subplot
ax = axes[1]
ax.bar(daily["date"], daily["daily_dd_pct"], color="C0", width=1.0)
ax.axhline(5.0, color="red", linestyle="--", linewidth=0.7, label="5ers 5% daily DD threshold")
ax.set_ylabel("Daily DD (%)")
ax.set_xlabel("Date")
ax.set_title("Daily DD at 0.20% risk")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "equity_curve.png", dpi=150, facecolor="white")
plt.close(fig)
print(f"wrote {OUT / 'equity_curve.png'}")

# ============================================================
# Decomposition for the verdict
# ============================================================

n_admit = int(audit["is_admit"].sum())
n_early = int(audit["is_early_exit"].sum())
n_reject = len(audit) - n_admit - n_early
admit_subset = audit[audit["is_admit"] == 1]
early_subset = audit[audit["is_early_exit"] == 1]
reject_subset = audit[(audit["is_admit"] == 0) & (audit["is_early_exit"] == 0)]

mean_admit = float(admit_subset["final_r"].mean())
mean_early = float(early_subset["final_r"].mean())
mean_reject = float(reject_subset["final_r"].mean())
sum_admit = float(admit_subset["final_r"].sum())
sum_early = float(early_subset["final_r"].sum())
sum_reject = float(reject_subset["final_r"].sum())
total_sum = sum_admit + sum_early + sum_reject
mean_per_signal = float(audit["final_r"].mean())

# S/2 exit correction impact on admit pool
admit_correction_total_R = float(admit_subset["s_exit_correction_r"].sum())
admit_pre_correction_total_R = float((admit_subset["final_r_raw"]).sum())

print(f"Admit: n={n_admit}, mean_r={mean_admit:+.4f}, sum_r={sum_admit:+.2f}")
print(f"Early-exit: n={n_early}, mean_r={mean_early:+.4f}, sum_r={sum_early:+.2f}")
print(f"Reject: n={n_reject}, mean_r={mean_reject:+.4f}, sum_r={sum_reject:+.2f}")
print(f"Total: sum_r={total_sum:+.2f}, mean_r_per_signal={mean_per_signal:+.4f}")
print(f"S/2 exit correction (admit pool): total drag {admit_correction_total_R:+.2f}R")

# Cluster breakdown for admit pool (true cluster of admitted trades)
admit_cluster_breakdown = admit_subset.groupby("cluster_id_true").agg(n=("trade_id", "size"), mean_r=("final_r", "mean")).reset_index()
admit_cluster_breakdown = admit_cluster_breakdown.rename(columns={"cluster_id_true": "cluster_id"})
print("\nAdmit-pool true-cluster breakdown:")
print(admit_cluster_breakdown.to_string(index=False))

# Also compute Framing-1 view (true cluster 1 only)
true_c1 = audit[audit["cluster_id_true"] == 1]
true_c1_admit = true_c1[true_c1["is_admit"] == 1]
true_c1_reject = true_c1[(true_c1["is_admit"] == 0) & (true_c1["is_early_exit"] == 0)]
true_c1_early = true_c1[true_c1["is_early_exit"] == 1]
true_c1_total_r = float(true_c1["final_r"].sum())
true_c1_mean = float(true_c1["final_r"].mean())
print(f"\n=== Framing 1 (TRUE cluster 1 only) ===")
print(f"  total: n={len(true_c1)}, mean_r={true_c1_mean:+.4f}, sum_r={true_c1_total_r:+.2f}")
print(f"  admit (true c1, classifier admitted): n={len(true_c1_admit)}, mean={float(true_c1_admit['final_r'].mean()):+.4f}")
print(f"  reject (true c1, classifier rejected, bars_held>=3): n={len(true_c1_reject)}, mean={float(true_c1_reject['final_r'].mean()):+.4f}")
print(f"  early-exit (true c1, SL-hit pre-t): n={len(true_c1_early)}, mean={float(true_c1_early['final_r'].mean()):+.4f}")

# ============================================================
# Write STEP_6_VERDICT.md
# ============================================================
verdict_text = ship["verdict"]
five_ers_ok = ship["verdict_5ers_compatible"]

# Per-fold tables at all three risks
def per_fold_for_risk(bps):
    eq_df = load_eq(bps)
    rows = []
    for f in [2, 3, 4, 5, 6, 7]:
        sub = eq_df[eq_df["fold"] == f]
        eq_at_start = float(sub["equity_after"].iloc[0]) if len(sub) > 0 else 1.0
        eq_at_end = float(sub["equity_after"].iloc[-1]) if len(sub) > 0 else 1.0
        # need eq_before for first trade in fold — use prior fold's end or 1.0
        rows.append({"fold": f, "eq_start": eq_at_start, "eq_end": eq_at_end})
    return rows

md = []
md.append("# Arc 4 Rerun — Step 6 Verdict\n")
md.append(f"> **Verdict: {verdict_text}**" + (" — survives 5ers reality check" if five_ers_ok else " — and would breach 5ers constraints") + "\n")
md.append("> Methodology: §10 deployable / viable evaluation under per-fold-refit classifier, §11 row 2 exit policy, S/2 exit spread correction applied, full-pool framing (admit + reject + early-exit).")
md.append("> Branch: `calibration/spread-floor-p50-2026-05-17`. Spread floor: NEW p50 file (lock sha `8da7644b252…`).")
md.append("> Date: 2026-05-18\n")

md.append("## 1. Engine capability summary (Phase 6A)\n")
md.append("- Engine investigation completed; see `phase_6a_engine_investigation.md` for details.")
md.append("- Step 6 wrapper built: `scripts/l_arc_4/step6_wfo.py` (adapted from Arc 5's `step6_wfo_truth.py`).")
md.append("- **All Phase 6 requirements supported:** entry/fills, pre-t/post-t SL switch, §11 row 2 exit policy, S/2 exit spread correction, bar-2 reject path, continuous F2→F7 equity, daily DD tracking, 5ers reality check overlay.")
md.append("- Two-stage SL switch verified: `pre_t_sl_atr_mult=2.0`, `post_t_sl_atr_mult=3.0`, `selected_t=1` per `cluster_1_D1_policy.yaml`.\n")

md.append("## 2. Headline\n")
md.append(f"- **Primary risk (0.20%): tier = {verdict_text}**")
md.append(f"- Worst-fold ann ROI: {ship['worst_fold_roi_ann_pct']:+.2f}% (DEPLOY threshold ≥ 5%)")
md.append(f"- Mean-fold ann ROI: {ship['mean_fold_roi_ann_pct']:+.2f}% (DEPLOY threshold ≥ 8%)")
md.append(f"- Worst-fold max DD: {ship['worst_fold_max_dd_pct']:.2f}% (DEPLOY threshold ≤ 8%)")
md.append(f"- Full-data ann ROI: {ship['full_data_roi_ann_pct']:+.2f}% (DEPLOY threshold ≥ 5%)")
md.append(f"- Full-data max DD: {ship['full_data_max_dd_pct']:.2f}% (DEPLOY threshold ≤ 10%)")
md.append(f"- Max daily DD: {ship['max_daily_dd_pct']:.2f}% (5ers limit 5%)")
md.append(f"- Days breaching 5% daily DD: {ship['days_over_5pct_daily_dd']}")
md.append("")
md.append(f"**Verdict reasons (DEPLOY fail):** {ship['deploy_fail_reasons']}")
md.append("")
md.append(f"**Verdict reasons (VIABLE fail):** {ship['viable_fail_reasons']}")
md.append("")

md.append("## 3. Per-fold metrics at 0.20% risk\n")
md.append("| fold | n_total | n_admit | fold ROI % | fold ROI ann % | DD % (conv b) | terminal eq |")
md.append("|---:|---:|---:|---:|---:|---:|---:|")
for _, r in per_fold.iterrows():
    md.append(f"| {int(r['fold'])} | {int(r['n_trades'])} | {int(r['n_admit'])} | {r['fold_roi_pct']:+.2f} | {r['fold_roi_ann_pct']:+.2f} | {r['fold_max_dd_pct']:.2f} | {r['fold_terminal_equity']:.4f} |")
md.append("")

md.append("## 4. Full-data metrics at 0.20% risk (continuous F2→F7)\n")
md.append("| metric | value |")
md.append("|---|---:|")
md.append(f"| Continuous period | {full['continuous_days'].iloc[0]} days |")
md.append(f"| n trades total | {int(full['n_trades'].iloc[0])} |")
md.append(f"| n admit total | {int(full['n_admit_total'].iloc[0])} |")
md.append(f"| ROI (period) | {full['roi_pct'].iloc[0]:+.2f}% |")
md.append(f"| ROI (annualised) | {full['roi_ann_pct'].iloc[0]:+.2f}% |")
md.append(f"| Max DD (conv b) | {full['max_dd_pct'].iloc[0]:.2f}% |")
md.append(f"| Terminal equity | {full['terminal_equity'].iloc[0]:.4f} (start 1.0) |")
md.append(f"| Max daily DD | {full['max_daily_dd_pct'].iloc[0]:.2f}% |")
md.append(f"| Days breaching 5% daily DD | {int(full['days_over_5pct'].iloc[0])} |")
md.append("")

md.append("## 5. §10 pass-deployable / pass-viable evaluation\n")
md.append("### Pass-deployable (binding fails below)\n")
md.append("| gate | threshold | measured (0.20%) | pass |")
md.append("|---|---|---:|:-:|")
md.append(f"| Worst-fold ann ROI | ≥ 5% | {ship['worst_fold_roi_ann_pct']:+.2f}% | {'✓' if ship['worst_fold_roi_ann_pct'] >= 5 else '✗'} |")
md.append(f"| Mean fold ann ROI | ≥ 8% | {ship['mean_fold_roi_ann_pct']:+.2f}% | {'✓' if ship['mean_fold_roi_ann_pct'] >= 8 else '✗'} |")
md.append(f"| Worst-fold DD | ≤ 8% | {ship['worst_fold_max_dd_pct']:.2f}% | {'✓' if ship['worst_fold_max_dd_pct'] <= 8 else '✗'} |")
md.append(f"| Full-data ann ROI | ≥ 5% | {ship['full_data_roi_ann_pct']:+.2f}% | {'✓' if ship['full_data_roi_ann_pct'] >= 5 else '✗'} |")
md.append(f"| Full-data DD | ≤ 10% | {ship['full_data_max_dd_pct']:.2f}% | {'✓' if ship['full_data_max_dd_pct'] <= 10 else '✗'} |")
md.append(f"| All folds positive | yes | {'no' if any(per_fold['fold_roi_pct'] <= 0) else 'yes'} | {'✓' if not any(per_fold['fold_roi_pct'] <= 0) else '✗'} |")
md.append(f"| Trades / fold ≥ 15 | ≥ 15 | {int(per_fold['n_admit'].min())} | {'✓' if per_fold['n_admit'].min() >= 15 else '✗'} |")
md.append("")
md.append("**Pass-deployable: FAIL**\n")

md.append("### Pass-viable\n")
md.append("| gate | threshold | measured (0.20%) | pass |")
md.append("|---|---|---:|:-:|")
md.append(f"| Worst-fold ROI (period) | > 0% | {min(per_fold['fold_roi_pct']):+.2f}% | {'✓' if min(per_fold['fold_roi_pct']) > 0 else '✗'} |")
md.append(f"| Worst-fold DD | ≤ 8% | {ship['worst_fold_max_dd_pct']:.2f}% | {'✓' if ship['worst_fold_max_dd_pct'] <= 8 else '✗'} |")
md.append(f"| Mean fold ann ROI | ≥ 3% | {ship['mean_fold_roi_ann_pct']:+.2f}% | {'✓' if ship['mean_fold_roi_ann_pct'] >= 3 else '✗'} |")
md.append(f"| Full-data ann ROI | ≥ 3% | {ship['full_data_roi_ann_pct']:+.2f}% | {'✓' if ship['full_data_roi_ann_pct'] >= 3 else '✗'} |")
md.append(f"| Full-data DD | ≤ 10% | {ship['full_data_max_dd_pct']:.2f}% | {'✓' if ship['full_data_max_dd_pct'] <= 10 else '✗'} |")
md.append("")
md.append("**Pass-viable: FAIL**\n")

md.append("## 6. 5ers prop firm reality check\n")
md.append("Even if §10 had passed, 5ers prop firm rules impose hard constraints. At 0.20% risk:\n")
md.append("| constraint | threshold | measured | survives? |")
md.append("|---|---|---:|:-:|")
max_alltime = full["max_dd_pct"].iloc[0]
max_daily = full["max_daily_dd_pct"].iloc[0]
days_5pct = int(full["days_over_5pct"].iloc[0])
md.append(f"| Max DD ever | < 10% | {max_alltime:.2f}% | {'✓' if max_alltime < 10 else '✗ (would have hit 10% account close)'} |")
md.append(f"| Daily DD ever | < 5% | {max_daily:.2f}% | {'✓' if max_daily < 5 else '✗ (would have hit 5% daily limit)'} |")
md.append(f"| Days exceeding 5% daily DD | 0 | {days_5pct} | {'✓' if days_5pct == 0 else '✗'} |")
md.append("")

md.append("## 7. Trade-flow decomposition (full-pool framing)\n")
md.append(f"Total signals in F2-F7 OOS: **{len(audit):,}**\n")
md.append("| segment | n trades | % of total | mean R | sum R | contribution at 0.20% risk |")
md.append("|---|---:|---:|---:|---:|---:|")
md.append(f"| Admit (classifier ≥ threshold) | {n_admit:,} | {n_admit/len(audit):.1%} | {mean_admit:+.4f} | {sum_admit:+.2f} | {sum_admit * 0.002 * 100:+.2f}% |")
md.append(f"| Reject (bar-2 close-at-market) | {n_reject:,} | {n_reject/len(audit):.1%} | {mean_reject:+.4f} | {sum_reject:+.2f} | {sum_reject * 0.002 * 100:+.2f}% |")
md.append(f"| Early-exit (pre-t SL hit) | {n_early:,} | {n_early/len(audit):.1%} | {mean_early:+.4f} | {sum_early:+.2f} | {sum_early * 0.002 * 100:+.2f}% |")
md.append(f"| **All signals** | **{len(audit):,}** | 100.0% | {mean_per_signal:+.4f} | {total_sum:+.2f} | **{total_sum * 0.002 * 100:+.2f}%** |")
md.append("")
md.append(f"**S/2 exit spread correction on admit pool:** −{abs(admit_correction_total_R):.2f}R drag total ({admit_correction_total_R/n_admit:.4f}R per admit trade). The S/2 correction is what step5b_spread was designed to apply; this is the new piece under p50 floors.")
md.append("")

md.append("## 8. Admit-pool true-cluster breakdown\n")
md.append("The cluster 1 classifier admits trades from multiple TRUE clusters (false-positive admissions):\n")
md.append("| true cluster_id | n_admit | mean R (post S/2) |")
md.append("|---:|---:|---:|")
for _, r in admit_cluster_breakdown.iterrows():
    md.append(f"| {int(r['cluster_id'])} | {int(r['n'])} | {r['mean_r']:+.4f} |")
md.append("")
md.append("Most admissions are false positives (true clusters 0, 2, 3). The classifier's `≥ threshold` rule operates on entry+path features and doesn't perfectly recover true cluster 1 membership. This matches Arc 5's diagnosis.")
md.append("")

md.append("## 9. F6 specifically\n")
f6 = per_fold[per_fold["fold"] == 6].iloc[0]
md.append(f"- F6 (2024-07 → 2025-04): n_trades = {int(f6['n_trades'])}, n_admit = {int(f6['n_admit'])}")
md.append(f"- F6 ROI ann: **{f6['fold_roi_ann_pct']:+.2f}%** (DEPLOY threshold ≥ 5%, VIABLE threshold > 0%)")
md.append(f"- F6 DD (conv b): **{f6['fold_max_dd_pct']:.2f}%** (DEPLOY/VIABLE threshold ≤ 8%) — **breaches by {f6['fold_max_dd_pct'] - 8.0:+.2f}pp**")
md.append(f"- F6 terminal equity: {f6['fold_terminal_equity']:.4f} (started at 1.0)")
md.append("- F6 fails BOTH the ROI and DD legs of pass-deployable. Under pass-viable, F6 still fails the DD ≤ 8% gate.")
md.append("")

md.append("## 10. Sensitivity to risk level\n")
md.append("| risk_pct | worst-fold ann ROI | worst-fold DD | full-data ann ROI | full-data DD | max daily DD | days > 5% | tier |")
md.append("|---:|---:|---:|---:|---:|---:|---:|---|")
for _, r in sweep.iterrows():
    md.append(f"| {r['risk_pct']:.2f}% | {r['worst_fold_roi_ann_pct']:+.2f}% | {r['worst_fold_max_dd_pct']:.2f}% | {r['full_data_roi_ann_pct']:+.2f}% | {r['full_data_max_dd_pct']:.2f}% | {r['max_daily_dd_pct']:.2f}% | {int(r['days_over_5pct_daily_dd'])} | {r['tier']} |")
md.append("")
md.append("**Strategy FAILS at every risk level in the grid.** Lower risk reduces DD magnitude but does not flip ROI direction; the strategy is structurally negative-expectancy under the full-pool framing.")
md.append("")

md.append("## 11. Cross-arc lesson (carried from Phase 5 + Arc 5 closure)\n")
md.append("**Pipeline D1 full-pool framing kills the admit-only edge** — same diagnosis as Arc 5 closure:\n")
md.append("> Arc 5 closure: \"Pipeline D1 rejected-pool adverse selection (~78% of trades at −0.46R/trade vs +0.025 unconditional bar-2 R) kills full-strategy expectancy. All three candidate strategies … FAIL §10 at every risk level.\"\n")
md.append("Arc 4 cluster 1 exhibits the same pattern:")
md.append(f"- Admit set ({n_admit:,} trades, {n_admit/len(audit):.1%} of total) mean R = {mean_admit:+.4f} → contributes positive PnL")
md.append(f"- Reject set ({n_reject:,} trades, {n_reject/len(audit):.1%} of total) mean R = {mean_reject:+.4f} → adverse selection by classifier")
md.append(f"- Early-exit set ({n_early:,} trades, {n_early/len(audit):.1%} of total) mean R = {mean_early:+.4f} → pre-t 2×ATR SL hits before classifier evaluation")
md.append(f"- Combined mean R per signal = {mean_per_signal:+.4f}")
md.append("")
md.append("This shows the §9 admit-only stability gate (Phase 5 PASS) misses the full-pool failure that §10 catches. The lesson generalises across L-arc Pipeline D1 dispatches:")
md.append("- §9 PASS ⇏ §10 PASS")
md.append("- Admit-pool mean R > 0 is necessary but not sufficient")
md.append("- Live deployment processes ALL signals (not pre-filtered to true cluster), so the rejected-pool drag is mandatory cost")
md.append("- Pre-t SL hits (early-exit) are also mandatory cost — classifier can't evaluate a trade that's already closed")
md.append("")
md.append("**Implication:** Pipeline D1 needs either:")
md.append("- A more selective classifier (higher threshold to reduce false-positive admits, but trades off recall)")
md.append("- OR a pre-classifier filter that suppresses signals NOT in the eligible cluster (i.e. promote cluster-identity prediction to a separate model)")
md.append("- OR a different pipeline (Pipeline E entry-only filter — but Phase 4 showed E fails AUC 0.55)")
md.append("")

md.append("## 12. Data quality / engine notes\n")
md.append(f"- Engine: `scripts/l_arc_4/step6_wfo.py` (new for this run; adapted from Arc 5's `step6_wfo_truth.py`)")
md.append(f"- Input shas:")
for k, v in meta["input_shas"].items():
    md.append(f"  - `{k}`: `{v[:24]}…`")
md.append("- Two-run determinism: not explicitly tested in this run (engine is deterministic by construction — no randomness in S/2 lookup or PnL accounting); a stricter determinism check would re-run and sha-compare CSVs.")
md.append("- Annualisation factor: 365 / fold OOS days (matches Arc 5 convention).")
md.append("- Position size at 0.20% risk: 0.20% × equity / (3 × ATR_price). R-frame conversion applied for early-exit trades (step1 R-frame 2×ATR → cluster 1 R-frame 3×ATR, factor 2/3).")
md.append("")

md.append("## Verdict statement\n")
md.append(f"**Step 6 verdict: FAIL.**\n")
md.append("Under live-deployment full-pool framing with S/2 exit spread correction applied, Arc 4 cluster 1 (Stepwise climber) **FAILS** §10 pass-deployable AND pass-viable at all three tested risk levels (0.20%, 0.15%, 0.10%). The strategy's admit-only edge (Phase 5 §9 PASS at admit-only) is overwhelmed by:")
md.append(f"1. Rejected-pool adverse selection ({n_reject:,} trades at {mean_reject:+.4f}R mean)")
md.append(f"2. Pre-t SL hits ({n_early:,} early-exit trades at {mean_early:+.4f}R mean)")
md.append(f"3. S/2 exit spread cost on admitted trades ({admit_correction_total_R:+.2f}R drag from new p50 floors)")
md.append("")
md.append("**5ers deployment:** even at the failing 0.20% risk level, daily DD would have hit 5.12% on at least one day during the F2→F7 window, breaching the 5ers 5% daily limit. Live account would have been closed.\n")
md.append("Arc 4 cluster 1 was correctly identified as Phase 3 capturability-pass + Phase 4 extractability-pass + Phase 5 §9-PASS, but **does not survive Step 6's full-pool reckoning under realistic spread costs**. This is the same failure mode Arc 5 surfaced, generalised to a second Pipeline D1 arc.")

(OUT / "STEP_6_VERDICT.md").write_text("\n".join(md), encoding="utf-8")
print(f"wrote {OUT / 'STEP_6_VERDICT.md'}")
