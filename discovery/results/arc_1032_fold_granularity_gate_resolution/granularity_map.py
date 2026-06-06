"""Arc 1032 — fold-granularity -> all-blocks-positive map (operator path-A decision-support).

The corpus's SOLE remaining deployability lever is the operator's path-A gate-governance call
(arcs 2016/2017/2019/3021/1023/2021/3022 + cosim item E): the 4-component book is mean-positive,
cost-robust, temporally stable, ~3 independent bets, but fails the per-CALENDAR-YEAR all-folds-
positive gate below the components' noise floor. Arc 2016 flagged "the AFP calendar-year gate is
applied below its noise floor — reconsider fold/gate resolution" but NO arc quantified it: at what
fold GRANULARITY does the IS book become all-blocks-positive, and is that robust to block alignment?

This converts the qualitative flag into a number. It is a re-analysis of the CANONICAL COMMITTED
per-year component ROIs (cosim item-E doc, reproduced byte-exact through MultiPairBacktester;
arc 1020/2019 frozen risk-parity weights) — NO new engine run, NO OOS touched (the book fails IS
AFP, so OOS stays frozen, §4/§5g). Robust across both weightings and ALL block alignments.
"""
from __future__ import annotations

import numpy as np

YEARS = list(range(2011, 2021))  # 2011-2020 IS per-year folds

# Canonical committed per-year ROIs (%) — cosim item-E validation doc ("mine" col, reproduced
# byte-exact vs arc-1020 recorded; gap/me_long/me_short exact, fbr within ~1.4% at 0.107 weight).
COMP = {
    "gap":      [-0.07, +8.23, -2.06, +2.94, -4.19, +3.20, +0.53, -6.79, +7.45, -2.39],
    "me_long":  [+0.40, +0.29, +0.96, -0.23, -1.14, -0.51, +0.34, +0.90, +1.16, +0.15],
    "fbr":      [+8.97, +3.12, +0.86, +0.24, +3.36, +4.04, +0.80, -4.39, -0.17, +3.99],
    "me_short": [+3.39, +1.69, -0.90, +0.98, +0.40, -0.91, -0.68, +0.86, +1.29, +0.71],
}
RP_WEIGHTS = {"gap": 0.078, "me_long": 0.531, "fbr": 0.107, "me_short": 0.284}  # arc-2019 frozen
EQ_WEIGHTS = {k: 0.25 for k in COMP}


def book_vector(weights):
    arr = np.zeros(len(YEARS))
    for k, w in weights.items():
        arr += w * np.array(COMP[k])
    return arr


def blocks_at_granularity(vec, block_len, offset):
    """Partition the 10-year vector into contiguous blocks of `block_len` starting at `offset`.
    Block ROI = mean of constituent per-year ROIs (equal-time). Partial leading/trailing blocks
    (from a nonzero offset) are included as their own (shorter) blocks — a stricter, alignment-fair
    accounting (every year belongs to exactly one block)."""
    out, i = [], 0
    # leading partial block
    if offset > 0:
        out.append(vec[:offset].mean())
        i = offset
    while i < len(vec):
        out.append(vec[i:i + block_len].mean())
        i += block_len
    return out


def all_blocks_positive_robust(vec, block_len):
    """All-blocks-positive across EVERY alignment offset in [0, block_len). Returns
    (robust_AFP, frac_alignments_pass, worst_block_over_alignments)."""
    passes, worst = [], []
    for off in range(block_len):
        blks = blocks_at_granularity(vec, block_len, off)
        passes.append(all(b > 0 for b in blks))
        worst.append(min(blks))
    return all(passes), float(np.mean(passes)), float(min(worst))


def main():
    for wname, weights in [("risk_parity", RP_WEIGHTS), ("equal", EQ_WEIGHTS)]:
        vec = book_vector(weights)
        print("=" * 84)
        print(f"BOOK ({wname})  per-year ROI (%):")
        print("  " + "  ".join(f"{y}:{v:+.2f}" for y, v in zip(YEARS, vec)))
        print(f"  mean={vec.mean():+.3f}%  n_neg_years={int((vec<0).sum())}  "
              f"neg_years={[y for y,v in zip(YEARS,vec) if v<0]}")
        print(f"\n  {'granularity':<16}{'robust_AFP':<12}{'frac_align_pass':<18}{'worst_block(any align)':<22}{'n_blocks(aligned)'}")
        for bl, label in [(1, "calendar-year"), (2, "2-year"), (3, "3-year"), (5, "5-year (halves)"), (10, "whole-IS")]:
            robust, frac, worst = all_blocks_positive_robust(vec, bl)
            n_blocks = len(blocks_at_granularity(vec, bl, 0))
            print(f"  {label:<16}{str(robust):<12}{frac:<18.2f}{worst:<+22.3f}{n_blocks}")
        print()
    print("=" * 84)
    print("Read: robust_AFP=True means all-blocks-positive under EVERY block alignment at that")
    print("granularity. The operator's path-A 'coarsen the fold gate' lever works only where")
    print("robust_AFP flips True with >2 blocks (a gate with <3 blocks is not a meaningful gate).")


if __name__ == "__main__":
    main()
