"""Compare new Step 1 trade pool (results/l_arc_4_rerun/step1) to prior Arc 4
trade pool (results/l_arc_4/step1) and produce a comparison CSV + summary text.

Read-only. Writes to results/l_arc_4_rerun/step1/comparison_to_prior_pool.{csv,md}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
NEW = REPO / "results" / "l_arc_4_rerun" / "step1" / "trades_all.csv"
OLD = REPO / "results" / "l_arc_4" / "step1" / "trades_all.csv"
OUT_CSV = REPO / "results" / "l_arc_4_rerun" / "step1" / "comparison_to_prior_pool.csv"
OUT_MD = REPO / "results" / "l_arc_4_rerun" / "step1" / "comparison_to_prior_pool.md"

new = pd.read_csv(NEW)
old = pd.read_csv(OLD)
new["signal_time"] = pd.to_datetime(new["signal_time"])
old["signal_time"] = pd.to_datetime(old["signal_time"])

print(f"new pool: {len(new):,}")
print(f"old pool: {len(old):,}")
print(f"delta: {len(new) - len(old)}")

# Key = (pair, signal_time)
new_key = new.set_index(["pair", "signal_time"])
old_key = old.set_index(["pair", "signal_time"])

new_keys = set(new_key.index)
old_keys = set(old_key.index)
both = new_keys & old_keys
only_new = new_keys - old_keys
only_old = old_keys - new_keys

print(f"in both pools: {len(both):,}")
print(f"only in new: {len(only_new):,}")
print(f"only in old: {len(only_old):,}")

# Build matched comparison
matched = new_key.loc[sorted(both)].join(
    old_key.loc[sorted(both)], rsuffix="_old"
)
print(f"matched rows: {len(matched):,}")

# Per-row checks
matched["entry_time_match"] = matched["entry_time"] == matched["entry_time_old"]
matched["exit_time_match"] = matched["exit_time"] == matched["exit_time_old"]
matched["exit_reason_match"] = matched["exit_reason"] == matched["exit_reason_old"]
matched["bars_held_match"] = matched["bars_held"] == matched["bars_held_old"]
matched["delta_entry_price"] = matched["entry_price"] - matched["entry_price_old"]
matched["delta_exit_price"] = matched["exit_price"] - matched["exit_price_old"]
matched["delta_sl_distance_pips"] = matched["sl_distance_pips"] - matched["sl_distance_pips_old"]
matched["delta_spread_used"] = matched["spread_pips_used"] - matched["spread_pips_used_old"]
matched["delta_spread_exit"] = matched["spread_pips_exit"] - matched["spread_pips_exit_old"]
matched["delta_final_r"] = matched["final_r"] - matched["final_r_old"]

# Reset to flat columns
matched = matched.reset_index()

# Summaries
print()
print("=== invariance checks on matched trades ===")
for col in ["entry_time_match", "exit_time_match", "exit_reason_match", "bars_held_match"]:
    rate = matched[col].mean()
    print(f"  {col}: {matched[col].sum():,}/{len(matched):,} ({rate:.4%})")

print()
print("=== delta distributions (matched trades) ===")
for col in ["delta_entry_price", "delta_exit_price", "delta_sl_distance_pips",
            "delta_spread_used", "delta_spread_exit", "delta_final_r"]:
    s = matched[col].dropna()
    print(f"  {col}:")
    print(f"    mean={s.mean():.6f}  std={s.std():.6f}  min={s.min():.6f}  max={s.max():.6f}")
    for p in [10, 25, 50, 75, 90]:
        print(f"    p{p}: {np.percentile(s, p):.6f}", end="  ")
    print()

# Save
matched.to_csv(OUT_CSV, index=False)
print(f"\nwrote {OUT_CSV}")

# Per-pair n_trade summary
print()
print("=== per-pair n_trades (new vs old) ===")
old_per_pair = old["pair"].value_counts().sort_index()
new_per_pair = new["pair"].value_counts().sort_index()
df_pair = pd.DataFrame({"old": old_per_pair, "new": new_per_pair}).fillna(0).astype(int)
df_pair["delta"] = df_pair["new"] - df_pair["old"]
print(df_pair.to_string())
print(f"\nsum old: {df_pair['old'].sum()}  sum new: {df_pair['new'].sum()}  delta: {df_pair['delta'].sum()}")

# Write markdown summary
md = []
md.append("# Step 1 trade pool comparison: rerun vs prior\n")
md.append(f"New pool (l_arc_4_rerun): **{len(new):,}** trades")
md.append(f"Prior pool (l_arc_4): **{len(old):,}** trades")
md.append(f"Delta: **{len(new) - len(old):+,} ({(len(new)-len(old))/len(old):+.2%})**")
md.append("")
md.append("## Set membership by (pair, signal_time)\n")
md.append(f"| set | n |")
md.append(f"|---|---:|")
md.append(f"| trades in BOTH pools | {len(both):,} |")
md.append(f"| only in new (admitted by new, NOT by old) | {len(only_new):,} |")
md.append(f"| only in old (admitted by old, NOT by new) | {len(only_old):,} |")
md.append("")
md.append("## Invariance checks on matched trades\n")
md.append("Per prompt: entry_time / exit_time / exit_reason / bars_held expected identical (path-based, spread-independent). Result:\n")
md.append("| field | n_matched | n_identical | pct |")
md.append("|---|---:|---:|---:|")
for col, lbl in [("entry_time_match", "entry_time"),
                 ("exit_time_match", "exit_time"),
                 ("exit_reason_match", "exit_reason"),
                 ("bars_held_match", "bars_held")]:
    n = int(matched[col].sum())
    md.append(f"| {lbl} | {len(matched):,} | {n:,} | {n/len(matched):.4%} |")
md.append("")
md.append("## Delta distributions on matched trades\n")
md.append("Per prompt: entry_price / exit_price / final_r expected to differ under new spreads.\n")
md.append("| metric | mean | std | p10 | p25 | p50 | p75 | p90 | min | max |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for col, lbl in [
    ("delta_entry_price", "Δ entry_price"),
    ("delta_exit_price", "Δ exit_price"),
    ("delta_sl_distance_pips", "Δ sl_distance_pips"),
    ("delta_spread_used", "Δ spread_pips_used (entry)"),
    ("delta_spread_exit", "Δ spread_pips_exit"),
    ("delta_final_r", "Δ final_r"),
]:
    s = matched[col].dropna()
    md.append(
        f"| {lbl} | {s.mean():.6f} | {s.std():.6f} | "
        f"{np.percentile(s,10):.6f} | {np.percentile(s,25):.6f} | "
        f"{np.percentile(s,50):.6f} | {np.percentile(s,75):.6f} | "
        f"{np.percentile(s,90):.6f} | {s.min():.6f} | {s.max():.6f} |"
    )
md.append("")
md.append("## Per-pair n_trades\n")
md.append("| pair | old | new | delta |")
md.append("|---|---:|---:|---:|")
for pair in df_pair.index:
    md.append(f"| {pair} | {df_pair.loc[pair, 'old']} | {df_pair.loc[pair, 'new']} | {df_pair.loc[pair, 'delta']:+d} |")
md.append(f"| **total** | **{df_pair['old'].sum()}** | **{df_pair['new'].sum()}** | **{df_pair['delta'].sum():+d}** |")
md.append("")

OUT_MD.write_text("\n".join(md), encoding="utf-8")
print(f"wrote {OUT_MD}")
