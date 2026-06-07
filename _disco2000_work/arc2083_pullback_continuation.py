"""arc 2083 — Trend-PULLBACK continuation under the honest engine + trailing + TAIL-REMOVED.

The FAVORABLE-FIRST residual of the operator's positive-skew continuation thread. arc 1074 killed the
BREAKOUT continuation as ADVERSE-FIRST (false-breakout whipsaw -> -1R wall, median R ~ -0.9). The
pullback-resume entry is favorable-first by construction (enter on the resume AFTER a retracement, stop
beyond the pullback extreme) — the one continuation geometry whose median R could exceed -0.9. arc 2012
tested a pullback entry only on the capture/drift cheap-kill lens (blind to skew); this spends the honest
engine under the tail-preserving trailing exits, judged on per-trade MEAN R + median-per-fold + TAIL-REMOVED.

CALLS canonical (Panel/ArcFoldRunner/A1/build_v3_folds/apply_cost_model). Experiment-side = the pullback
entry MASK (new SignalModule) + arc-2063 winsorization arithmetic (via arc-2081 harness). Never realizes P&L.

Pre-registered kill-rule (arc doc, written before this run): mean > 0 net of costs AND survive the +2R cap
/ top-5% removal; mean-positive ONLY via the top-K winners -> KILL. Beats-null-but-net-negative -> KILL.
"""
from __future__ import annotations

import numpy as np

from core.sim.panel import Panel
from core.wfo.folds import build_v3_folds

from discovery.tools.pullback_continuation_signal import PullbackContinuationSignal
from discovery.tools.trend_entry_signals import PeriodicLongSignal

# reuse the arc-2081 measurement harness verbatim (collect / report / split_metrics)
import importlib.util, pathlib
_h = pathlib.Path(__file__).with_name("arc2081_continuation_skew.py")
_spec = importlib.util.spec_from_file_location("arc2081_harness", _h)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

ALL28 = H.ALL28
EXITS = ["sl_only", "sl_plus_trailing_atr", "sl_plus_trailing_swing", "sl_partial_close_1r_runner_trail"]


def main():
    panel = Panel.from_pairs(ALL28, tf="H4", histdata_root=H.BACKUP, cache_root=H.CACHE,
                             boundary_convention="5ers_eet")
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    print(f"=== arc 2083 trend-PULLBACK continuation (28 pairs H4, {len(is_folds)} IS folds) ===")
    print(f"R-unit = ${H.R_DOLLARS:.0f}; FundedNext costs ON; SL=2*ATR; risk={H.RISK_PCT}")

    best = None
    for direction in ("long", "short"):
        sig = PullbackContinuationSignal(direction=direction)
        print(f"\n########## PULLBACK-RESUME CONTINUATION {direction.upper()} (full 28-pair universe) ##########")
        for ep in EXITS:
            fr, R, P = H.collect(sig, panel, is_folds, ep, f"pb_{direction}")
            tm = H.report(f"pullback {direction} + {ep}", fr, R)
            H.split_metrics(R, P)
            if tm and (best is None or tm["mean"] > best[1]):
                best = (f"{direction} {ep}", tm["mean"], R, P)

    # Null: being-long-anytime under the trailing exits (does the pullback ENTRY add value?)
    print("\n########## PERIODIC-LONG NULL (being-long-anytime, same exit) ##########")
    for ep in ("sl_plus_trailing_atr", "sl_partial_close_1r_runner_trail"):
        fr, R, P = H.collect(PeriodicLongSignal(period=30, warmup=120), panel, is_folds, ep, "periodic")
        H.report(f"periodic30 NULL + {ep}", fr, R)

    if best:
        print(f"\n>>> BEST cell by per-trade mean R: {best[0]}  (mean={best[1]:+.4f}R)")
        print(">>> majors/crosses split for the best cell:")
        H.split_metrics(best[2], best[3])


if __name__ == "__main__":
    main()
