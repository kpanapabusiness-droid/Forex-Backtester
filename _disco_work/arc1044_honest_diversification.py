"""arc 1044 — Honest-exit DIVERSIFICATION: does the 4-way book beat me_long SOLO?

arc 1042/1043 showed under honest §5f exits the book deploy mean is ~half (+0.27%)
and NOT statistically significant, with 2 of 4 legs (gap, me_short) mean-negative and
me_long the lone robust leg (RP weight 0.531). The multi-component PORTFOLIO thesis
(arcs 1006-2019, the whole reason for 4 legs) rests on arc-2019's "~3 independent
bets" diversification pillar — computed on COMMITTED exits. This arc asks the sharpest
honest question: under honest exits, does the BOOK add risk-adjusted value over its one
robust leg me_long ALONE, or does the portfolio honestly collapse to one thin leg?

Computes (honest §5f series, fold-bootstrap seed 42): me_long-SOLO vs 4-way BOOK
mean/sd/t; the honest-series effective-number-of-bets (ENB = (Σλ)²/Σλ² of the 4-leg
fold-ROI covariance, arc-2019 method); and the honest pairwise fold-ROI correlations.

100% canonical scoring; BUILT nested_exit_selection + combine_fold_roi. OOS untouched.

Run:  PYTHONPATH=. py _disco_work/arc1044_honest_diversification.py
"""
from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path

import numpy as np

from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.wfo.discovery_measure import run_config_over_folds
from core.wfo.folds import build_v3_folds
from discovery.tools.combine_fold_roi import combine_fold_rois, fit_weights
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.gap_signals import WeekendGapFillLongSignal
from discovery.tools.month_end_signals import (
    MonthEndReversionLongSignal,
    MonthEndReversionShortSignal,
)
from discovery.tools.nested_exit_selection import (
    metric_afp_then_mean,
    metric_mean_roi,
    metric_worst_then_mean,
    nested_walk_forward_select,
)
from discovery.tools.time_exit_predicate import make_time_exit_predicate

HISTDATA_ROOT = Path(os.environ.get("COSIM_HISTDATA_ROOT", r"C:/Users/panap/histdata_backup"))
CACHE_ROOT = Path(os.environ.get("COSIM_CACHE_ROOT", r"C:/Users/panap/Documents/Forex-Backtester/data/cache"))
BOUNDARY = "5ers_eet"
JPY_CROSSES = ["EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"]
USD_MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
EXITS = ["sl_only", "sl_plus_tp_2r", "sl_plus_tp_3r", "sl_plus_trailing_atr",
         "sl_plus_trailing_swing", "sl_partial_close_1r_runner_trail"]
SLS = [1.5, 2.0, 2.5]
GAP_HORIZONS = [12, 18, 24, 36, 48]
NAMES = ("gap", "me_long", "fbr", "me_short")
N_BOOT = 10000
SEED = 42


def _load(pairs, tf):
    return Panel.from_pairs(pairs, tf, histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
                            use_cache=True, boundary_convention=BOUNDARY)


def _attach(sig, panel, n):
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n)
    return dataclasses.replace(sig, per_pair={
        p: dataclasses.replace(st, exit_predicate=pred) for p, st in sig.per_pair.items()})


def _boot(fold_rois, label):
    arr = np.array(fold_rois, dtype=float) * 100.0
    n = len(arr); rng = np.random.default_rng(SEED)
    means = arr[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    mean = arr.mean(); sd = arr.std(ddof=1)
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    lo, hi = np.percentile(means, [2.5, 97.5])
    sharpe = mean / sd if sd > 0 else float("inf")
    print(f"  {label:<26} mean={mean:+.3f}%  sd={sd:.3f}  t={t:+.2f}  Sharpe(fold)={sharpe:+.3f}  "
          f"CI=[{lo:+.3f},{hi:+.3f}]  P(<0)={float((means<0).mean()):.3f}")
    return dict(mean=mean, sd=sd, t=t, sharpe=sharpe)


def _enb(series_by_name):
    """Effective number of bets = (Σλ)²/Σλ² of the 4-leg fold-ROI covariance (arc 2019)."""
    M = np.array([series_by_name[n] for n in NAMES], dtype=float)  # 4 x 10
    cov = np.cov(M)  # 4x4
    lam = np.linalg.eigvalsh(cov)
    lam = lam[lam > 1e-15]
    enb = (lam.sum() ** 2) / (lam ** 2).sum()
    corr = np.corrcoef(M)
    return enb, corr


def main():
    print(f"N_boot={N_BOOT} seed={SEED}\n")
    h4_usd = _load(USD_MAJORS, "H4"); h4_jpy = _load(JPY_CROSSES, "H4"); d1 = _load(USD_MAJORS, "D1")
    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]

    gap_sig = WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36).evaluate({"H4": h4_jpy})
    me_long_sig = _attach(MonthEndReversionLongSignal(1.0, 2).evaluate({"D1": d1}), d1, 2)
    fbr_sig = FailedBreakdownReclaimLongSignal(40, 1.25).evaluate({"H4": h4_usd})
    me_short_sig = MonthEndReversionShortSignal(1.0, 2).evaluate({"D1": d1})
    runners = {"me_long": ArcFoldRunner(A1Architecture(), me_long_sig, {"D1": d1}),
               "fbr": ArcFoldRunner(A1Architecture(), fbr_sig, {"H4": h4_usd}),
               "me_short": ArcFoldRunner(A1Architecture(), me_short_sig, {"D1": d1})}

    print("scoring grids (canonical)...")
    scored = {n: {} for n in NAMES}
    for hz in GAP_HORIZONS:
        gr = ArcFoldRunner(A1Architecture(), _attach(gap_sig, h4_jpy, hz), {"H4": h4_jpy})
        for sl in SLS:
            scored["gap"][f"timeexit{hz}|sl{sl}"] = list(run_config_over_folds(
                gr, is_folds, A1Config(config_id=f"gap_t{hz}_sl{sl}", sl_atr_mult=sl,
                                       trail_enabled=False, exit_policy=None)))
    for name in ("me_long", "fbr", "me_short"):
        for ex in EXITS:
            for sl in SLS:
                scored[name][f"{ex}|sl{sl}"] = list(run_config_over_folds(
                    runners[name], is_folds,
                    A1Config(config_id=f"{name}_{ex}_sl{sl}", sl_atr_mult=sl, trail_enabled=False, exit_policy=ex)))

    metrics = [("mean_roi", metric_mean_roi), ("afp_then_mean", metric_afp_then_mean),
               ("worst_then_mean", metric_worst_then_mean)]
    honest = {}
    for mn, mf in metrics:
        honest[mn] = {n: [c.roi_pct for c in sorted(
            nested_walk_forward_select(scored[n], selection=mf, selection_name=mn, min_prior_folds=2).per_fold,
            key=lambda c: c.fold_id)] for n in NAMES}

    print("\n" + "=" * 86)
    print("HONEST-EXIT DIVERSIFICATION: 4-way BOOK vs me_long SOLO (fold-bootstrap, seed 42)")
    print("=" * 86)
    for mn, _ in metrics:
        hs = honest[mn]
        w_commit = fit_weights([hs[n] for n in NAMES], mode="risk_parity")  # honest-refit RP
        book = list(combine_fold_rois([hs[n] for n in NAMES], w_commit).combined_roi)
        enb, corr = _enb(hs)
        print(f"\n-- honest metric: {mn} --  (honest-refit RP wts: " +
              ", ".join(f"{n}={w:.3f}" for n, w in zip(NAMES, w_commit)) + ")")
        r_solo = _boot(hs["me_long"], "me_long SOLO")
        r_book = _boot(book, "4-way BOOK (RP)")
        better = "BOOK > solo" if r_book["sharpe"] > r_solo["sharpe"] else "SOLO >= book"
        print(f"  -> ENB(honest 4-leg) = {enb:.2f}/4   |   risk-adjusted: {better} "
              f"(book Sharpe {r_book['sharpe']:+.3f} vs solo {r_solo['sharpe']:+.3f})")
        print(f"     honest pairwise fold-ROI corr:")
        for i in range(4):
            print("       " + " ".join(f"{corr[i][j]:+.2f}" for j in range(4)) + f"   {NAMES[i]}")


if __name__ == "__main__":
    main()
