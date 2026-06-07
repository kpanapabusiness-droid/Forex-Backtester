"""arc 2081 addendum — the corpus's STRONGEST continuation (forward-confirmed shock, arc 3019)
re-measured under a TRAILING (tail-preserving) exit + the tail-removed metric on the FULL 28-pair
universe. arc 3019 ran it on 7 majors and froze tp_3r (tail-CAPPING) for OOS; this checks whether a
genuine positive-skew edge was masked by that exit choice. IS only (OOS untouched). Reuses the arc2081
harness (CALLS canonical; experiment-side = entry mask + winsorization only)."""
from __future__ import annotations

import numpy as np

from core.sim.panel import Panel
from core.wfo.folds import build_v3_folds
from discovery.tools.shock_continuation_signals import ShockContinuationSignal

import sys
sys.path.insert(0, "_disco2000_work")
from arc2081_continuation_skew import ALL28, BACKUP, CACHE, collect, report, split_metrics


def main():
    panel = Panel.from_pairs(ALL28, tf="H4", histdata_root=BACKUP, cache_root=CACHE,
                             boundary_convention="5ers_eet")
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    print(f"=== arc 2081 ADDENDUM: shock-continuation (full 28-pair, {len(is_folds)} IS folds) ===")

    for shock_atr in (2.0, 3.0):
        for direction in ("long", "short"):
            for ep in ("sl_plus_trailing_atr", "sl_partial_close_1r_runner_trail"):
                sig = ShockContinuationSignal(shock_atr=shock_atr, direction=direction, confirm=True)
                fr, R, P = collect(sig, panel, is_folds, ep, f"shock{shock_atr}_{direction}")
                tm = report(f"shock{shock_atr} {direction} + {ep}", fr, R)
                if tm:
                    split_metrics(R, P)


if __name__ == "__main__":
    main()
