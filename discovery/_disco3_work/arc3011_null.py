"""arc 3011 — fair same-exit NULL baseline (KILL vs PORTFOLIO determination, §11).

Does the failed-breakout short's ~0 net IS edge beat a random-entry short under the SAME exit?
Uses BUILT build_null_signal_evaluation (matched fire-count, random eligible bars, scoring canonical).

Run:  PYTHONPATH=. py discovery/_disco3_work/arc3011_null.py
"""
from __future__ import annotations

from datetime import date
import numpy as np

from core.sim.panel import Panel
from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import run_config_over_folds
from discovery.tools.failed_breakout_signals import FailedBreakoutRejectionShortSignal
from discovery.tools.null_entry_baseline import build_null_signal_evaluation

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
HISTDATA = r"C:\Users\panap\histdata_backup"


def mean_worst(stats):
    rois = [s.roi_pct for s in stats]
    return float(np.mean(rois)), min(rois), sum(1 for r in rois if r > 0), len(rois)


def main():
    panel = Panel.from_pairs(PAIRS, tf="H4", histdata_root=HISTDATA,
                             cache_root="data/cache", boundary_convention="5ers_eet")
    sig = FailedBreakoutRejectionShortSignal(swing_lookback=40, min_shadow_atr=1.25)
    build_arc_pool(sig, {"H4": panel}, ArcPoolConfig(
        arc_name="arc_3011_failed_breakout_short", sl_atr_mult=2.0, hold_bars=120, risk_pct=0.005,
        window_start=date(2010, 1, 1), window_end=date(2020, 12, 31)))
    sig_eval = sig.evaluate({"H4": panel})
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    cfg = A1Config(config_id="best_drift", exit_policy="sl_plus_trailing_atr", sl_atr_mult=2.0,
                   risk_pct=0.005, trail_enabled=False)

    real = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=sig_eval, panels={"H4": panel})
    rm, rw, rp, rn = mean_worst(run_config_over_folds(real, is_folds, cfg))
    print(f"\n=== arc 3011 NULL (exit=sl_plus_trailing_atr SL2.0) ===")
    print(f"  REAL short : mean {rm:+.3f}%  worst {rw:+.3f}%  {rp}/{rn} pos")

    nmeans = []
    for seed in (42, 7, 123):
        null_eval = build_null_signal_evaluation(sig_eval, seed=seed, warmup=100)
        nr = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=null_eval, panels={"H4": panel})
        nm, nw, npz, nn = mean_worst(run_config_over_folds(nr, is_folds, cfg))
        nmeans.append(nm)
        print(f"  NULL seed={seed:3d}: mean {nm:+.3f}%  worst {nw:+.3f}%  {npz}/{nn} pos")
    print(f"  NULL mean-of-means {np.mean(nmeans):+.3f}%   REAL-minus-NULL {rm - np.mean(nmeans):+.3f} pp")


if __name__ == "__main__":
    main()
