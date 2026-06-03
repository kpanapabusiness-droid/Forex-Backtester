"""Diff admit/fill/lifecycle ledgers between risk levels.

For each fold tested, compares two risk levels and reports:
  - Are admit_attempts byte-identical (modulo size/balance fields)?
  - Are fill_attempts byte-identical?
  - Are position_lifecycle byte-identical?
  - If divergent: first divergent (timestamp, pair) and what differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Columns that legitimately differ across risk levels (sizing only)
SIZE_DEPENDENT = {
    "base_size", "account_balance_at_signal", "effective_size",
    "account_balance_before", "pnl", "size", "delta_balance",
}
# Columns whose VALUE should be invariant under risk_pct scaling
RISK_INVARIANT = {
    "signal_time", "pair", "proba", "mult", "mult_band", "atr_at_entry",
    "sl_price", "admitted_to_pending", "reject_reason",
    "fill_time", "risk_multiplier", "fill_result", "n_open_before",
    "position_id", "parent_position_id", "entry_time", "exit_time",
    "entry_price", "exit_price", "exit_reason", "n_open_positions",
    "delta_open", "time",
}


def compare_ledger(p_a: Path, p_b: Path, name: str) -> dict:
    """Return diff report between two ledger parquets."""
    out: dict = {"ledger": name, "ok": False}
    if not p_a.exists() or not p_b.exists():
        out["status"] = "missing"
        return out
    a = pd.read_parquet(p_a)
    b = pd.read_parquet(p_b)
    out["n_rows_a"] = len(a)
    out["n_rows_b"] = len(b)
    if len(a) != len(b):
        out["status"] = "row_count_diff"
        out["delta"] = len(b) - len(a)
        return out

    diffs: list[dict] = []
    for col in a.columns:
        if col in SIZE_DEPENDENT:
            continue
        if col not in b.columns:
            diffs.append({"col": col, "issue": "missing_in_b"})
            continue
        ne_mask = (a[col].astype(object) != b[col].astype(object))
        # NaN equality: treat (NaN, NaN) as equal
        try:
            both_nan = a[col].isna() & b[col].isna()
            ne_mask = ne_mask & ~both_nan
        except (TypeError, ValueError):
            pass
        n_diff = int(ne_mask.sum())
        if n_diff > 0:
            first_idx = ne_mask[ne_mask].index[0]
            diffs.append({
                "col": col, "n_diff": n_diff,
                "first_row_idx": int(first_idx),
                "first_a": str(a.loc[first_idx, col]),
                "first_b": str(b.loc[first_idx, col]),
            })
    out["risk_invariant_diffs"] = diffs
    out["status"] = "byte_identical_invariant" if not diffs else "divergent"
    out["ok"] = not diffs
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("results/analysis/risk_leak_diagnosis"))
    parser.add_argument("--folds", type=str, default="5,10,12",
                        help="Fold IDs to diff (must match diagnostic output)")
    parser.add_argument("--risks", type=str, default="0.0050,0.0200")
    args = parser.parse_args(argv)

    fold_ids = [int(x) for x in args.folds.split(",")]
    risks = args.risks.split(",")
    if len(risks) != 2:
        raise ValueError("--risks must be two comma-separated values")

    all_ok = True
    for fid in fold_ids:
        print(f"\n=== FOLD {fid} ({risks[0]} vs {risks[1]}) ===")
        # Normalise risk-level token to the 4-decimal format used by the
        # diagnostic output (r0.0050, r0.0200, ...).
        ra = f"{float(risks[0]):.4f}"
        rb = f"{float(risks[1]):.4f}"
        a_dir = args.root / f"fold{fid}_r{ra}"
        b_dir = args.root / f"fold{fid}_r{rb}"
        for name in ("admit_attempts_ledger", "fill_attempts_ledger",
                     "position_lifecycle_ledger", "per_bar_state_ledger"):
            result = compare_ledger(a_dir / f"{name}.parquet",
                                    b_dir / f"{name}.parquet", name)
            print(f"  {name}: status={result['status']}, "
                  f"rows={result.get('n_rows_a', '?')}/{result.get('n_rows_b', '?')}, "
                  f"invariant_diffs={len(result.get('risk_invariant_diffs', []))}")
            for d in result.get("risk_invariant_diffs", []):
                print(f"     diff in {d['col']}: n={d['n_diff']}, "
                      f"first_row={d['first_row_idx']}, "
                      f"a={d['first_a']}, b={d['first_b']}")
            if not result["ok"]:
                all_ok = False
    print()
    print("RESULT:", "ALL RISK-INVARIANT COLUMNS BYTE-IDENTICAL" if all_ok
          else "AT LEAST ONE DIVERGENCE FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
