"""arc 2011 — leave-one-pair-out + outlier-exclusion robustness on the reject-short drift cell.

The pooled drift (+0.26 ATR @ shadow>=1.25) looked structure-control-passing, but per-pair Q4 showed
4/7 with TWO thin outliers (AUDUSD n=25 +1.26, USDJPY n=31 +1.49). arc-1013 discipline: a real edge is
leave-one-pair-out ALL-positive. If dropping one pair collapses the pooled drift, it is an outlier-
carried thin-tail artifact (arc 1010/3002), not a robust edge — and capture being coin-flip (0.47-0.51)
already flags the slow-drift-killed-by-stop signature (arc 3003).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.observe_long_capture import observe_long_capture
from _disco2000_work.arc2011_observe_failed_breakout_short import build_cond, PAIRS, BACKUP


def main() -> None:
    panel = Panel.from_pairs(PAIRS, tf="H4", histdata_root=BACKUP, cache_root="data/cache",
                             boundary_convention="5ers_eet")
    obs = observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24, direction="short")
    obs = obs.set_index(["pair", "signal_time"])
    cond = pd.concat([build_cond(panel.pair_dfs[p]).assign(pair=p) for p in PAIRS]).set_index(["pair", "signal_time"])
    j = obs.join(cond, how="inner")
    st = j.index.get_level_values("signal_time")
    j = j[(st >= pd.Timestamp("2010-01-01", tz="UTC")) & (st <= pd.Timestamp("2020-12-31", tz="UTC"))]
    j = j[np.isfinite(j["fwd_drift_atr"]) & np.isfinite(j["atr_c"])]

    for s in (1.0, 1.25):
        cell = j[j["swept_high"] & j["reject"] & (j["upper_shadow"] >= s)]
        pooled = cell["fwd_drift_atr"].mean()
        cap = cell["capture"].mean()
        print(f"\n=== shadow>={s}: n={len(cell)} pooled drift={pooled:+.4f} cap={cap:.4f} ===")
        print("  leave-one-pair-out pooled drift (arc-1013 needs ALL > 0):")
        for drop in PAIRS:
            sub = cell[cell.index.get_level_values("pair") != drop]
            d = sub["fwd_drift_atr"].mean()
            flag = "" if d > 0 else "   <-- NEGATIVE (LOO fails)"
            print(f"    drop {drop}: n={len(sub):4d} drift={d:+.4f}{flag}")
        # exclude the two thin outlier pairs (AUDUSD, USDJPY) together
        robust = cell[~cell.index.get_level_values("pair").isin(["AUDUSD", "USDJPY"])]
        print(f"  EXCLUDE both outliers (AUDUSD+USDJPY): n={len(robust)} "
              f"drift={robust['fwd_drift_atr'].mean():+.4f} cap={robust['capture'].mean():.4f}")
        # median (thin-tail check: mean >> median => carried by a few big moves)
        print(f"  pooled MEDIAN drift={cell['fwd_drift_atr'].median():+.4f} (mean {pooled:+.4f}; "
              f"mean>>median => thin-tail-carried)")


if __name__ == "__main__":
    main()
