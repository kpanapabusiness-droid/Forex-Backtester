"""arc 2015 — ROBUSTNESS of the month-end SHORT side (the Arc-10 / arc-2013 defense).

The §5f engine run gave the short side mean +0.683% (best exit partial_runner), beats null +0.92pp, and
POSITIVE in both binding folds (2015 +0.40, 2018 +0.86). Before claiming PORTFOLIO I must rule out the
arc-2013 failure mode: a mean-positive short that is THIN REGIME-LUCK / pair-carried (the observation
flagged GBPUSD+USDJPY carrying, NZDUSD dragging). Checks:
  1. Leave-one-pair-out (does the edge + the 2015/2018 positivity survive dropping ANY pair?).
  2. Threshold robustness (0.75 / 1.0 / 1.25) — one-cell or a plateau?
  3. 2015 & 2018 per-pair decomposition — broad, or 1-2 trades / 1-2 pairs?
All scored on the canonical engine (ArcFoldRunner), best exit = sl_partial_close_1r_runner_trail.
"""
from __future__ import annotations

import numpy as np
from datetime import date

from core.sim.panel import Panel
from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import judge_all_folds_positive, run_config_over_folds

from discovery.tools.month_end_signals import MonthEndReversionShortSignal

USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
BACKUP = r"C:\Users\panap\histdata_backup"
EXIT = "sl_partial_close_1r_runner_trail"
# fold_id -> OOS year (anchor: fold 6 = 2015; folds 2..11 = 2011..2020)
FOLD_YEAR = {fid: 2011 + (fid - 2) for fid in range(2, 12)}


def score(eval_, panel_pairs, folds):
    runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=eval_, panels={"D1": panel_pairs})
    cfg = A1Config(config_id="r", exit_policy=EXIT, sl_atr_mult=2.0)
    return run_config_over_folds(runner, folds, cfg)


def main():
    full = Panel.from_pairs(USD_MAJORS, tf="D1", histdata_root=BACKUP, cache_root="data/cache",
                            boundary_convention="5ers_eet")
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]

    print("=== 1. Leave-one-pair-out (best exit) ===")
    for drop in [None] + USD_MAJORS:
        pairs = [p for p in USD_MAJORS if p != drop]
        panel = Panel.from_pairs(pairs, tf="D1", histdata_root=BACKUP, cache_root="data/cache",
                                 boundary_convention="5ers_eet")
        sig = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2)
        stats = score(sig.evaluate({"D1": panel}), panel, is_folds)
        rois = [fs.roi_pct for fs in stats]
        by_year = {FOLD_YEAR[fs.fold_id]: fs.roi_pct for fs in stats}
        lbl = "FULL" if drop is None else f"-{drop}"
        print(f"  {lbl:9s} mean={np.mean(rois)*100:+6.3f}%  2015={by_year.get(2015,float('nan'))*100:+.2f} "
              f"2018={by_year.get(2018,float('nan'))*100:+.2f}  neg={sum(1 for r in rois if r<0)}/{len(rois)}")

    print("\n=== 2. Threshold robustness (full universe, best exit) ===")
    for thr in (0.75, 1.0, 1.25, 1.5):
        sig = MonthEndReversionShortSignal(threshold_atr=thr, into_bars=2)
        stats = score(sig.evaluate({"D1": full}), full, is_folds)
        rois = [fs.roi_pct for fs in stats]
        by_year = {FOLD_YEAR[fs.fold_id]: fs.roi_pct for fs in stats}
        mintr = min(fs.n_trades for fs in stats)
        print(f"  thr={thr:.2f} mean={np.mean(rois)*100:+6.3f}% 2015={by_year[2015]*100:+.2f} 2018={by_year[2018]*100:+.2f} "
              f"neg={sum(1 for r in rois if r<0)}/{len(rois)} min_trades/fold={mintr}")

    print("\n=== 3. 2015 & 2018 per-pair decomposition (pool final_r, gross) ===")
    pool = build_arc_pool(MonthEndReversionShortSignal(threshold_atr=1.0), {"D1": full}, ArcPoolConfig(
        arc_name="arc_2015_decomp", sl_atr_mult=2.0, hold_bars=120, risk_pct=0.005,
        window_start=date(2010, 1, 1), window_end=date(2020, 12, 31)))
    import pandas as pd
    tr = pool.trades.copy()
    tr["year"] = pd.to_datetime(tr["signal_time"]).dt.year
    for yr in (2015, 2018):
        sub = tr[tr["year"] == yr]
        print(f"  {yr}: n={len(sub)} mean_final_r={sub['final_r'].mean():+.3f} win={(sub['final_r']>0).mean():.3f}")
        print(sub.groupby("pair")["final_r"].agg(["count", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
