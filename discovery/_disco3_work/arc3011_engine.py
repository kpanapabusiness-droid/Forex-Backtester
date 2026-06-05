"""arc 3011 — Failed-breakout REJECTION short: honest engine triage + §5f exit/SL menu.

The observation showed a non-coin-flip entry (structure-confirmed gross drift +0.278 ATR), so §5f
MANDATES the honest engine with the registered exit menu before any FAIL. Standard entry point
(TOOL_REGISTRY): build_arc_pool -> ArcFoldRunner -> build_v3_folds + build_oos_year_folds ->
run_config_over_folds -> judge_all_folds_positive. Scored solely by MultiPairBacktester, FundedNext ON.

Run:  PYTHONPATH=. py discovery/_disco3_work/arc3011_engine.py
"""
from __future__ import annotations

from datetime import date

import numpy as np

from core.sim.panel import Panel
from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import (
    build_oos_year_folds, judge_all_folds_positive, run_config_over_folds,
)
from discovery.tools.failed_breakout_signals import FailedBreakoutRejectionShortSignal

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
HISTDATA = r"C:\Users\panap\histdata_backup"

EXITS = [
    "sl_only", "sl_plus_tp_2r", "sl_plus_tp_3r",
    "sl_plus_trailing_atr", "sl_plus_trailing_swing",
    "sl_partial_close_1r_runner_trail",
]
SLS = [1.5, 2.0, 3.0]


def fold_rois(stats):
    return [f"{s.roi_pct:+.2f}" for s in stats]


def summarize(stats, label):
    rois = [s.roi_pct for s in stats]
    n = sum(s.n_trades for s in stats)
    pos = sum(1 for r in rois if r > 0)
    worst = min(rois) if rois else float("nan")
    mean = float(np.mean(rois)) if rois else float("nan")
    print(f"  {label:46s} mean {mean:+.2f}%  worst {worst:+.2f}%  {pos}/{len(rois)} pos  n={n}")
    return mean, worst, pos, len(rois)


def main():
    panel = Panel.from_pairs(
        PAIRS, tf="H4", histdata_root=HISTDATA,
        cache_root="data/cache", boundary_convention="5ers_eet",
    )
    sig = FailedBreakoutRejectionShortSignal(swing_lookback=40, min_shadow_atr=1.25)

    pool = build_arc_pool(sig, {"H4": panel}, ArcPoolConfig(
        arc_name="arc_3011_failed_breakout_short", sl_atr_mult=2.0, hold_bars=120, risk_pct=0.005,
        window_start=date(2010, 1, 1), window_end=date(2020, 12, 31),
    ))
    tr = pool.trades
    print(f"\n=== arc 3011 POOL: n_trades={len(tr)} pool_sha={pool.pool_sha256[:12]} ===")
    if len(tr) < 50:
        print("POOL FLOOR FAIL (<50) -> KILL")
        return
    # honest capture sanity (+1R-before-SL = bars_to_1r_mfe non-NaN)
    cap = float(tr["bars_to_1r_mfe"].notna().mean())
    mean_r = float(tr["final_r"].mean())
    print(f"  pool capture(+1R-before-SL) {cap:.4f}  mean final_r {mean_r:+.4f}R")

    sig_eval = sig.evaluate({"H4": panel})
    runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=sig_eval, panels={"H4": panel})

    all_folds = build_v3_folds().folds
    is_folds = [f for f in all_folds if f.is_days >= 365]
    print(f"\n  IS folds: {len(is_folds)}")

    # --- triage on 3 representative folds (cheap), default partial/runner ---
    rep = [f for f in is_folds if any(str(y) in str(getattr(f, 'oos_start', '')) for y in (2013, 2016, 2019))]
    rep = rep if rep else is_folds[2:9:3]
    print("\n[TRIAGE — 3 representative folds]")
    for exit_policy in ("sl_partial_close_1r_runner_trail", "sl_only", "sl_plus_trailing_atr"):
        cfg = A1Config(config_id=f"tri_{exit_policy}", exit_policy=exit_policy, sl_atr_mult=2.0,
                       risk_pct=0.005, trail_enabled=False)
        stats = run_config_over_folds(runner, rep, cfg)
        summarize(stats, exit_policy)

    # --- §5f full IS exit/SL menu sweep (all IS folds) ---
    print("\n[§5f EXIT/SL MENU — full IS folds, all-folds-positive gate]")
    best = None
    for sl in SLS:
        for exit_policy in EXITS:
            cfg = A1Config(config_id=f"{exit_policy}_sl{sl}", exit_policy=exit_policy,
                           sl_atr_mult=sl, risk_pct=0.005, trail_enabled=False)
            stats = run_config_over_folds(runner, is_folds, cfg)
            mean, worst, pos, ntot = summarize(stats, f"{exit_policy} SL{sl}")
            v = judge_all_folds_positive(stats)
            if v.all_folds_positive:
                print(f"      ^^ IS ALL-FOLDS-POSITIVE: {fold_rois(stats)}")
            if best is None or worst > best[1]:
                best = (f"{exit_policy}_sl{sl}", worst, mean, pos, ntot)
    print(f"\n  BEST IS worst-fold: {best}")


if __name__ == "__main__":
    main()
