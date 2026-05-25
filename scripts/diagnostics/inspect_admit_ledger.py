"""Print first admit-attempts that scaled per risk level for sanity check."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIAG_ROOT = ROOT / "results" / "analysis" / "risk_leak_diagnosis"


def main():
    for fold_id in (5, 10, 12):
        a = pd.read_parquet(DIAG_ROOT / f"fold{fold_id}_r0.0050" / "admit_attempts_ledger.parquet")
        b = pd.read_parquet(DIAG_ROOT / f"fold{fold_id}_r0.0200" / "admit_attempts_ledger.parquet")
        admitted_a = a[a["admitted_to_pending"]]
        admitted_b = b[b["admitted_to_pending"]]
        print(f"\n=== FOLD {fold_id} — admits ===")
        print(f"  r=0.5%: {len(admitted_a)} admits / {len(a)} attempts")
        print(f"  r=2.0%: {len(admitted_b)} admits / {len(b)} attempts")
        print(f"  Identical signal_times: {list(admitted_a['signal_time']) == list(admitted_b['signal_time'])}")
        print(f"  Identical (pair, signal_time) tuples: "
              f"{set(zip(admitted_a['pair'], admitted_a['signal_time'])) == set(zip(admitted_b['pair'], admitted_b['signal_time']))}")
        # Size ratio at corresponding rows
        if len(admitted_a) == len(admitted_b):
            ratios = (admitted_b["base_size"].values / admitted_a["base_size"].values)
            ratios_finite = ratios[(ratios > 0) & ~pd.isna(ratios)]
            if len(ratios_finite):
                print(f"  base_size ratio r=2%/r=0.5%: min={ratios_finite.min():.4f}, "
                      f"max={ratios_finite.max():.4f}, mean={ratios_finite.mean():.4f} "
                      f"(expected 4.0 for linear scaling)")
            print(f"  First admit row diff (sized columns):")
            for col in ("base_size", "account_balance_at_signal", "proba", "mult"):
                print(f"    {col}: r=0.5%={admitted_a.iloc[0][col]}, r=2%={admitted_b.iloc[0][col]}")


if __name__ == "__main__":
    sys.exit(main() or 0)
