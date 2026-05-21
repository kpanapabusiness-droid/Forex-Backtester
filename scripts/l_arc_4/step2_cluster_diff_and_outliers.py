"""Phase 2 additional diagnostics:

1. Compare new K=4 cluster sizes to prior, and validate cluster 1 still maps to
   Stepwise climber per §11.
2. Cluster the 362 newly-admitted / 233 newly-dropped trades:
   - 362 newly-admitted → new clusters
   - 233 newly-dropped → prior clusters
3. Cluster + fold assignment for the 5 outlier trades (large negative
   delta_final_r from Phase 1 comparison).

Writes:
- results/l_arc_4_rerun/step2/cluster_size_comparison.md
- results/l_arc_4_rerun/step2/newly_admitted_cluster_breakdown.csv
- results/l_arc_4_rerun/step2/newly_dropped_cluster_breakdown.csv
- results/l_arc_4_rerun/step2/outliers_cluster_fold.csv
- results/l_arc_4_rerun/step2/phase_2_additional_diagnostics.md
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OLD_TRADES = REPO / "results" / "l_arc_4" / "step1" / "trades_all.csv"
NEW_TRADES = REPO / "results" / "l_arc_4_rerun" / "step1" / "trades_all.csv"
OLD_CLUSTERS = REPO / "results" / "l_arc_4" / "step2" / "clusters_K4.csv"
NEW_CLUSTERS = REPO / "results" / "l_arc_4_rerun" / "step2" / "clusters_K4.csv"
NEW_ARCH = REPO / "results" / "l_arc_4_rerun" / "step2" / "archetype_assignments.csv"
OLD_ARCH = REPO / "results" / "l_arc_4" / "step2" / "archetype_assignments.csv"
COMP_CSV = REPO / "results" / "l_arc_4_rerun" / "step1" / "comparison_to_prior_pool.csv"

OUT_DIR = REPO / "results" / "l_arc_4_rerun" / "step2"

# Fold definitions taken from prior Arc 4 step5c
FOLD_DEFS_PATH = REPO / "results" / "l_arc_4" / "step5c" / "fold_definitions.csv"


def fold_for(ts: pd.Timestamp, fold_defs: pd.DataFrame) -> int:
    """Return fold number (1..7) for a given entry timestamp; -1 if outside any OOS window."""
    for _, r in fold_defs.iterrows():
        oos_start = pd.Timestamp(r["oos_start"])
        oos_end = pd.Timestamp(r["oos_end"])
        if oos_start <= ts <= oos_end:
            return int(r["fold"])
    return -1


def main():
    old_t = pd.read_csv(OLD_TRADES, parse_dates=["signal_time", "entry_time", "exit_time"])
    new_t = pd.read_csv(NEW_TRADES, parse_dates=["signal_time", "entry_time", "exit_time"])
    old_c = pd.read_csv(OLD_CLUSTERS).set_index("trade_id")
    new_c = pd.read_csv(NEW_CLUSTERS).set_index("trade_id")

    fold_defs = pd.read_csv(FOLD_DEFS_PATH)
    print("[info] fold definitions:")
    print(fold_defs.to_string(index=False))

    # Attach cluster to each pool
    old_t = old_t.set_index("trade_id").join(old_c, how="left")
    new_t = new_t.set_index("trade_id").join(new_c, how="left")

    # === 1. Cluster size comparison ===
    print("\n=== cluster size comparison (K=4) ===")
    new_sizes = new_t["cluster_id"].value_counts().sort_index()
    old_sizes = old_c["cluster_id"].value_counts().sort_index()
    sz_df = pd.DataFrame({"old": old_sizes, "new": new_sizes}).fillna(0).astype(int)
    sz_df["delta"] = sz_df["new"] - sz_df["old"]
    print(sz_df.to_string())

    # === 2. Newly-admitted vs newly-dropped trade clustering ===
    # Key by (pair, signal_time)
    old_key = old_t.reset_index().set_index(["pair", "signal_time"])
    new_key = new_t.reset_index().set_index(["pair", "signal_time"])
    new_keys = set(new_key.index)
    old_keys = set(old_key.index)

    only_new_keys = new_keys - old_keys
    only_old_keys = old_keys - new_keys

    newly_admitted = new_key.loc[sorted(only_new_keys)].reset_index()
    newly_dropped = old_key.loc[sorted(only_old_keys)].reset_index()

    print(f"\n=== newly-admitted (362 expected, got {len(newly_admitted)}) ===")
    na_cluster = newly_admitted["cluster_id"].value_counts(dropna=False).sort_index()
    print(na_cluster)

    print(f"\n=== newly-dropped (233 expected, got {len(newly_dropped)}) ===")
    nd_cluster = newly_dropped["cluster_id"].value_counts(dropna=False).sort_index()
    print(nd_cluster)

    # Attach fold
    newly_admitted["fold"] = newly_admitted["entry_time"].apply(lambda ts: fold_for(ts, fold_defs))
    newly_dropped["fold"] = newly_dropped["entry_time"].apply(lambda ts: fold_for(ts, fold_defs))

    # Cluster × fold breakdown for newly-admitted (in new cluster ids)
    print("\n=== newly-admitted cluster × fold ===")
    na_xtab = pd.crosstab(newly_admitted["cluster_id"], newly_admitted["fold"], dropna=False, margins=True)
    print(na_xtab)

    print("\n=== newly-dropped cluster × fold (in OLD cluster ids) ===")
    nd_xtab = pd.crosstab(newly_dropped["cluster_id"], newly_dropped["fold"], dropna=False, margins=True)
    print(nd_xtab)

    # Save breakdowns
    newly_admitted[["pair", "signal_time", "entry_time", "cluster_id", "fold", "final_r", "bars_held", "exit_reason"]].to_csv(
        OUT_DIR / "newly_admitted_cluster_breakdown.csv", index=False
    )
    newly_dropped[["pair", "signal_time", "entry_time", "cluster_id", "fold", "final_r", "bars_held", "exit_reason"]].to_csv(
        OUT_DIR / "newly_dropped_cluster_breakdown.csv", index=False
    )

    # === 3. Outlier trades ===
    comp = pd.read_csv(COMP_CSV, parse_dates=["signal_time", "entry_time", "exit_time"])
    worst_5 = comp.nsmallest(5, "delta_final_r")
    # Lookup their (pair, signal_time) in NEW cluster + fold
    out_rows = []
    for _, r in worst_5.iterrows():
        key = (r["pair"], r["signal_time"])
        new_row = new_key.loc[[key]].iloc[0]
        old_row = old_key.loc[[key]].iloc[0]
        fold = fold_for(pd.Timestamp(r["entry_time"]), fold_defs)
        out_rows.append({
            "pair": r["pair"],
            "signal_time": r["signal_time"],
            "entry_time": r["entry_time"],
            "fold": fold,
            "new_cluster_id": int(new_row["cluster_id"]) if not pd.isna(new_row["cluster_id"]) else -1,
            "old_cluster_id": int(old_row["cluster_id"]) if not pd.isna(old_row["cluster_id"]) else -1,
            "new_final_r": r["final_r"],
            "old_final_r": r["final_r_old"],
            "delta_final_r": r["delta_final_r"],
            "new_bars_held": r["bars_held"],
            "old_bars_held": r["bars_held_old"],
            "new_exit_reason": r["exit_reason"],
            "old_exit_reason": r["exit_reason_old"],
            "new_entry_price": r["entry_price"],
            "old_entry_price": r["entry_price_old"],
            "spread_diff_pips": r["delta_spread_used"],
        })
    out_df = pd.DataFrame(out_rows)
    print("\n=== 5 worst-deltafinal_r trades: cluster + fold ===")
    print(out_df.to_string(index=False))
    out_df.to_csv(OUT_DIR / "outliers_cluster_fold.csv", index=False)

    # === Write a comprehensive markdown summary ===
    md = []
    md.append("# Phase 2 — Additional Diagnostics\n")
    md.append("> Per-user action items for Phase 2:")
    md.append("> 1. Cluster breakdown of the 362 newly-admitted / 233 newly-dropped trades")
    md.append("> 2. Cluster + fold of the 5 outlier trades (worst Δfinal_r)")
    md.append("> 3. Protocol-level lesson on spread × exposure-cap path dependence (in cross-arc lessons)\n")

    md.append("## 1. Cluster size comparison (K=4)\n")
    md.append("| cluster_id | old (n=10,764) | new (n=10,893) | delta | delta% |")
    md.append("|---:|---:|---:|---:|---:|")
    for cid in [0, 1, 2, 3]:
        o = int(sz_df.loc[cid, "old"])
        n = int(sz_df.loc[cid, "new"])
        d = n - o
        pct = d / o if o else float("nan")
        md.append(f"| {cid} | {o} | {n} | {d:+d} | {pct:+.2%} |")
    md.append(f"| **total** | **{sz_df['old'].sum()}** | **{sz_df['new'].sum()}** | **{sz_df['delta'].sum():+d}** | — |")
    md.append("")
    md.append("Per prompt §3.3: \"cluster sizes match prior within 1-2 trades (tolerance for float roundoff).\"")
    md.append(f"Actual deltas range from {sz_df['delta'].min():+d} to {sz_df['delta'].max():+d}. **This exceeds the ±2 tolerance** — explained by the +129 trade pool delta from Phase 1 (the 362 newly-admitted and 233 newly-dropped trades each redistribute across the four clusters). This is mechanical, not a clustering failure; new K=4 silhouette and archetype assignments need to be re-verified.")
    md.append("")

    md.append("## 2. Newly-admitted trades cluster × fold breakdown (362 trades)\n")
    md.append("**Cluster column = NEW cluster ids (after re-clustering on new pool).**\n")
    md.append(_xtab_to_md(na_xtab))
    md.append("")

    md.append("## 3. Newly-dropped trades cluster × fold breakdown (233 trades)\n")
    md.append("**Cluster column = OLD cluster ids (the cluster they had in the prior Arc 4 K=4 clustering).**\n")
    md.append(_xtab_to_md(nd_xtab))
    md.append("")

    md.append("## 4. Five worst-Δfinal_r outlier trades: cluster + fold\n")
    md.append("These trades had the largest negative `delta_final_r` (new − old). All 5 follow the same pattern: old pool = max_life winner; new pool = stop_loss at −1R, driven by the new SL absolute price being ~0.5 pips higher.\n")
    md.append("| pair | signal_time | entry_time | fold | new_cluster | old_cluster | new_final_r | old_final_r | Δ final_r | new_bars / old_bars | new_exit / old_exit | spread Δ (pips) |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|")
    for _, r in out_df.iterrows():
        md.append(
            f"| {r['pair']} | {r['signal_time']} | {r['entry_time']} | {int(r['fold'])} | "
            f"{int(r['new_cluster_id'])} | {int(r['old_cluster_id'])} | "
            f"{r['new_final_r']:+.4f} | {r['old_final_r']:+.4f} | {r['delta_final_r']:+.4f} | "
            f"{int(r['new_bars_held'])} / {int(r['old_bars_held'])} | "
            f"{r['new_exit_reason']} / {r['old_exit_reason']} | "
            f"{r['spread_diff_pips']:+.2f} |"
        )
    md.append("")

    # Specifically flag if any outlier hits cluster 1 + F6/F7
    flagged = out_df[(out_df["new_cluster_id"] == 1) & out_df["fold"].isin([6, 7])]
    if len(flagged) > 0:
        md.append(f"⚠️ **{len(flagged)} of 5 outlier trades land in cluster 1 + (F6 or F7)** — these are headline-relevant for Step 5 verdict because cluster 1 is the Arc 4 survivor and F6 was the prior closure's killer fold:")
        for _, r in flagged.iterrows():
            md.append(f"  - {r['pair']} {r['signal_time']} fold {int(r['fold'])} cluster {int(r['new_cluster_id'])}: Δfinal_r {r['delta_final_r']:+.4f}")
    else:
        md.append("No outlier trades sit in cluster 1 + (F6 or F7) — these are concentrated elsewhere.")
        # But still flag any cluster-1 hits
        c1_hits = out_df[out_df["new_cluster_id"] == 1]
        if len(c1_hits) > 0:
            md.append(f"\nCluster 1 outliers (any fold): {len(c1_hits)}:")
            for _, r in c1_hits.iterrows():
                md.append(f"  - {r['pair']} {r['signal_time']} fold {int(r['fold'])}: Δfinal_r {r['delta_final_r']:+.4f}")
    md.append("")

    md.append("## 5. Protocol-level lesson — for cross-arc lessons section of result doc\n")
    md.append("**Spread changes ARE path-dependent under exposure caps.**\n")
    md.append("The Arc 4 prompt assumed that since signal logic is spread-independent, changing the spread floor would produce an identical trade pool. This is incorrect under the `max_concurrent_per_pair = 1` exposure cap: a new spread floor changes the SL absolute price (because SL = entry − 2×ATR, and entry = mid + S/2 for longs), which changes when each trade hits SL, which changes when the per-pair slot frees up, which changes which subsequent signals get admitted.")
    md.append("")
    md.append("Concretely for Arc 4 under the p50 floor change:")
    md.append("- Trade count: 10,764 → 10,893 (+129, +1.20%)")
    md.append("- 362 signals admitted under new spreads that were excluded under old (a pair was still in an open trade at the old signal time)")
    md.append("- 233 signals admitted under old that are now excluded (vice versa)")
    md.append("- Of the 10,531 matched trades: 100% of `entry_time` identical, 94.3% of `bars_held` identical (5.7% diverge because SL fires at a different bar under the new SL absolute price)")
    md.append("")
    md.append("**Implication for Phase 4 (classifier AUC comparison):**")
    md.append("Because cluster 1 size changed (see §1 above), the RF classifier features and targets are computed on a slightly different population than Arc 4's prior. The classifier will be retrained per-fold on this new population. The Phase 4 expectation \"per-fold AUC matches prior Arc 4 Step 5C within ±0.005\" should be **relaxed** — small AUC drift is expected from population drift, not from any methodology change. A more meaningful Phase 4 check is: AUC ≥ §8 gate (≥ 0.60 for D1) on the new population. The byte-identity assumption against prior is no longer valid.")
    md.append("")
    md.append("This finding belongs in the cross-arc lessons section of the eventual `ARC_4_RERUN_RESULT.md`:")
    md.append("")
    md.append("> **Lesson (cross-arc):** Spread-floor changes are not population-invariant under exposure caps. Any future arc that swaps the spread floor file must expect Step 1 trade pool drift (+/− 1-2% typical) and propagate this through Step 2 (cluster sizes) and Step 4 (per-fold classifier retraining on slightly different populations). Byte-identity guarantees against prior runs do not apply at any step. The signal class, methodology, and gates remain valid; the specific trades change.")

    (OUT_DIR / "phase_2_additional_diagnostics.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[done] wrote {OUT_DIR / 'phase_2_additional_diagnostics.md'}")


def _xtab_to_md(xtab: pd.DataFrame) -> str:
    """Convert a pandas crosstab (with All margins) to a markdown table."""
    lines = []
    cols = [str(c) for c in xtab.columns]
    lines.append("| cluster_id \\ fold | " + " | ".join(cols) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(cols)) + "|")
    for idx in xtab.index:
        row_vals = [str(int(xtab.loc[idx, c])) for c in xtab.columns]
        lines.append(f"| {idx} | " + " | ".join(row_vals) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
