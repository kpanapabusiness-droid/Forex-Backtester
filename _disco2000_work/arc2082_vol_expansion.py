"""arc 2082 — volatility-EXPANSION breakout continuation (the dispatch's explicit positive-skew
example), full 28-pair universe, take-the-loss + trailing exits, judged on the tail-removed metric.
Final closer for the positive-skew continuation thread (after arc 2081's Donchian + shock). CALLS
canonical; reuses the arc2081 harness + the BUILT tail_removed_expectancy guard. IS only; OOS untouched.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "_disco2000_work")

from core.sim.panel import Panel
from core.wfo.folds import build_v3_folds
from discovery.tools.vol_expansion_breakout_signals import VolExpansionBreakoutSignal
from discovery.tools.tail_removed_expectancy import tail_removed_expectancy
from arc2081_continuation_skew import ALL28, BACKUP, CACHE, collect, report, split_metrics


def main():
    panel = Panel.from_pairs(ALL28, tf="H4", histdata_root=BACKUP, cache_root=CACHE,
                             boundary_convention="5ers_eet")
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    print(f"=== arc 2082 vol-EXPANSION breakout (28 pairs H4, {len(is_folds)} IS folds) ===")
    for vm in (1.5, 2.0):
        for direction in ("long", "short"):
            for ep in ("sl_plus_trailing_atr", "sl_partial_close_1r_runner_trail"):
                sig = VolExpansionBreakoutSignal(lookback=40, vol_mult=vm, direction=direction)
                fr, R, P = collect(sig, panel, is_folds, ep, f"volexp{vm}_{direction}")
                report(f"volexp{vm} {direction} + {ep}", fr, R)
                if len(R):
                    tm = tail_removed_expectancy(R)
                    print(f"  GUARD: {tm.summary()}")
                    split_metrics(R, P)


if __name__ == "__main__":
    main()
