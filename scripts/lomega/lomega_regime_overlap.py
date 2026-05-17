"""Cross-variant bin overlap analysis for the regime-conditional summary.

Sidecar diagnostic: feeds Section 5 ("Cross-variant consistency") of
LOMEGA_REGIME_CONDITIONAL_SUMMARY.md. Not part of the headline dispatch
output — the dispatch's required deliverables (bin_membership.csv,
per_bin_per_fold_auc.csv, variant_summary.csv, timeframe_summary.csv) are
emitted by lomega_regime_conditional.py.

Computes per-TF crosstabs:
  - Variant A (vol quintile) × Variant C (year-quarter)
  - Variant B (d1 trend) × Variant C (year-quarter)
  - Variant A × Variant B
Highlights cells where one variant's bin is over-represented relative to
independence (raw counts + row-normalised conditional probabilities +
top-N |obs - expected| / expected with min 100 obs).

Run after lomega_regime_conditional.py has produced bin_membership.csv.

Usage:
    py scripts/lomega/lomega_regime_overlap.py

Output: results/lomega/regime_conditional/cross_variant_overlap.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
_BASE = _REPO / "results" / "lomega" / "regime_conditional"


def load_bins(tf: str) -> dict[str, pd.Series]:
    base = _BASE / f"timeframe_{tf}"
    return {
        "A": pd.read_csv(base / "variant_a_vol_percentile" / "bin_membership.csv",
                         parse_dates=["time"]),
        "B": pd.read_csv(base / "variant_b_d1_trend" / "bin_membership.csv",
                         parse_dates=["time"]),
        "C": pd.read_csv(base / "variant_c_year_quarter" / "bin_membership.csv",
                         parse_dates=["time"]),
    }


def write_xtab(f, name: str, left: pd.Series, right: pd.Series):
    f.write(f"\n=== {name} ===\n")
    ct = pd.crosstab(left, right, normalize="index")  # row-normalised
    f.write("Row-normalised (each row sums to 1.0): P(right | left)\n")
    f.write(ct.round(4).to_string())
    f.write("\n\nRaw counts:\n")
    f.write(pd.crosstab(left, right).to_string())
    f.write("\n")


def deviations(left_name: str, right_name: str, left: pd.Series, right: pd.Series,
               top_n: int = 6) -> str:
    """Report cells whose joint frequency deviates most from independence."""
    counts = pd.crosstab(left, right)
    row_marg = counts.sum(axis=1)
    col_marg = counts.sum(axis=0)
    total = counts.values.sum()
    # Expected under independence
    expected = np.outer(row_marg.values, col_marg.values) / total
    excess = counts.values - expected
    rel = np.where(expected > 0, excess / expected, 0.0)
    cells = []
    for i, r in enumerate(counts.index):
        for j, c in enumerate(counts.columns):
            if counts.values[i, j] >= 100:
                cells.append((str(r), str(c), int(counts.values[i, j]),
                              float(expected[i, j]), float(rel[i, j])))
    cells.sort(key=lambda x: abs(x[4]), reverse=True)
    out = [f"\nTop {top_n} cells with largest |observed - expected| / expected "
           f"(min 100 obs):"]
    out.append(f"  {left_name:25s}  {right_name:15s}  obs       exp     rel_excess")
    for r, c, obs, exp, rel_v in cells[:top_n]:
        out.append(f"  {r:25s}  {c:15s}  {obs:6d}  {exp:8.0f}  {rel_v:+.2f}")
    return "\n".join(out) + "\n"


def main():
    out_path = _BASE / "cross_variant_overlap.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for tf in ("1h", "4h"):
            f.write(f"\n{'=' * 70}\n  Timeframe: {tf}\n{'=' * 70}\n")
            bins = load_bins(tf)
            a = bins["A"].set_index(["pair", "time"])["bin"]
            b = bins["B"].set_index(["pair", "time"])["bin"]
            c = bins["C"].set_index(["pair", "time"])["bin"]
            # Drop missing / insufficient
            keep_a = a[~a.isin(["missing"])]
            keep_b = b[~b.isin(["missing"])]
            keep_c = c[~c.isin(["missing", "insufficient_data"])]
            # Align on common index
            common = keep_a.index.intersection(keep_b.index).intersection(keep_c.index)
            a2 = a.loc[common]
            b2 = b.loc[common]
            c2 = c.loc[common]
            f.write(f"\nRows in common (after drop missing / insufficient): {len(common):,}\n")

            write_xtab(f, "A (vol quintile) × C (year-quarter)", a2, c2)
            f.write(deviations("A_vol_quintile", "C_quarter", a2, c2, top_n=8))

            write_xtab(f, "B (d1 trend) × C (year-quarter)", b2, c2)
            f.write(deviations("B_d1_trend", "C_quarter", b2, c2, top_n=8))

            write_xtab(f, "A (vol quintile) × B (d1 trend)", a2, b2)
            f.write(deviations("A_vol_quintile", "B_d1_trend", a2, b2, top_n=6))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
