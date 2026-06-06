"""arc 1043 — Honest-exit book-mean SIGNIFICANCE (the F3 owed by arc 1042).

arc 1023 established the COMMITTED-exit 4-way book mean is statistically significant
(t=2.66, p~0.026, fold-bootstrap CI [+0.22%,+1.09%], P(mean<0)=0) — the strongest
quantitative pillar of the operator's path-A deploy case (with 2019/2021/3022). arc
1042 showed the HONEST §5f deploy mean is ~half (+0.27% RP commit-wts) and flagged
(F3) that the t=2.66 significance likely does NOT survive honest exits (estimated
t~1.2). This arc COMPUTES it: fold-bootstrap the honest-exit book per-fold ROI
series for its mean CI / P(mean<0) / implied t, by the SAME method as arc 1023/2019
(resample the 10 fold ROIs w/ replacement, seed 42), vs the committed series.

100% canonical scoring; selection via BUILT nested_exit_selection; combination via
BUILT combine_fold_roi. No gate reimplemented. OOS NOT touched (IS book mean only).

Run:  PYTHONPATH=. py _disco_work/arc1043_honest_book_significance.py
"""
from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path

import numpy as np

from core.arc.signal_protocol import SignalEvaluation
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


def _fold_bootstrap(fold_rois, label):
    """Resample the per-fold ROIs (decimal) w/ replacement; report mean stats (arc 1023 method)."""
    arr = np.array(fold_rois, dtype=float) * 100.0  # %
    n = len(arr)
    rng = np.random.default_rng(SEED)
    means = arr[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    mean = arr.mean()
    sd = arr.std(ddof=1)
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_neg = float((means < 0).mean())
    print(f"  {label:<34} mean={mean:+.3f}%  sd={sd:.3f}  t={t:+.2f}  "
          f"95%CI=[{lo:+.3f},{hi:+.3f}]  P(mean<0)={p_neg:.3f}  neg-folds={(arr<=0).sum()}/{n}")
    return dict(mean=mean, t=t, ci=(lo, hi), p_neg=p_neg)


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

    def cfg(tag, ex, sl):
        return A1Config(config_id=f"{tag}_{ex}_sl{sl}", sl_atr_mult=sl, trail_enabled=False, exit_policy=ex)

    # ── score grids (IS) ──────────────────────────────────────────────────
    print("scoring grids (canonical)...")
    scored = {n: {} for n in NAMES}
    # gap: pure-time horizon x SL
    for hz in GAP_HORIZONS:
        gr = ArcFoldRunner(A1Architecture(), _attach(gap_sig, h4_jpy, hz), {"H4": h4_jpy})
        for sl in SLS:
            scored["gap"][f"timeexit{hz}|sl{sl}"] = list(run_config_over_folds(
                gr, is_folds, A1Config(config_id=f"gap_t{hz}_sl{sl}", sl_atr_mult=sl,
                                       trail_enabled=False, exit_policy=None)))
    for name in ("me_long", "fbr", "me_short"):
        for ex in EXITS:
            for sl in SLS:
                scored[name][f"{ex}|sl{sl}"] = list(run_config_over_folds(runners[name], is_folds, cfg(name, ex, sl)))

    committed_label = {"gap": "timeexit24|sl2.0", "me_long": "sl_only|sl2.0",
                       "fbr": "sl_plus_trailing_atr|sl2.0", "me_short": "sl_partial_close_1r_runner_trail|sl2.0"}
    committed_series = {n: [fs.roi_pct for fs in sorted(scored[n][committed_label[n]], key=lambda fs: fs.fold_id)]
                        for n in NAMES}

    # ── honest series per metric ──────────────────────────────────────────
    metrics = [("mean_roi", metric_mean_roi), ("afp_then_mean", metric_afp_then_mean),
               ("worst_then_mean", metric_worst_then_mean)]
    honest = {}
    for mn, mf in metrics:
        honest[mn] = {n: [c.roi_pct for c in sorted(
            nested_walk_forward_select(scored[n], selection=mf, selection_name=mn, min_prior_folds=2).per_fold,
            key=lambda c: c.fold_id)] for n in NAMES}

    # ── weights ──────────────────────────────────────────────────────────
    w_commit = fit_weights([committed_series[n] for n in NAMES], mode="risk_parity")
    w_eq = [0.25] * 4

    def book(series, w):
        return list(combine_fold_rois([series[n] for n in NAMES], w).combined_roi)

    print("\n" + "=" * 78)
    print("FOLD-BOOTSTRAP of the book MEAN (arc-1023 method: resample 10 fold ROIs, seed 42)")
    print("=" * 78)
    print("\n-- COMMITTED exits (reproduce arc 1023 t=2.66) --")
    _fold_bootstrap(book(committed_series, w_commit), "committed RP")
    _fold_bootstrap(book(committed_series, w_eq), "committed EQUAL")

    print("\n-- HONEST §5f exits (the deploy number) --")
    summary = {}
    for mn, _ in metrics:
        w_h = fit_weights([honest[mn][n] for n in NAMES], mode="risk_parity")
        r_commit = _fold_bootstrap(book(honest[mn], w_commit), f"honest {mn} RP(commit-wts)")
        r_honest = _fold_bootstrap(book(honest[mn], w_h), f"honest {mn} RP(refit-wts)")
        summary[mn] = (r_commit, r_honest)

    print("\n" + "=" * 78)
    print("SIGNIFICANCE VERDICT (does the t=2.66 deploy pillar survive honest exits?)")
    print("=" * 78)
    for mn, _ in metrics:
        rc, rh = summary[mn]
        surv = "SURVIVES (t>2, P<0.05)" if rc["t"] > 2 and rc["p_neg"] < 0.05 else \
               ("marginal" if rc["t"] > 1.6 else "LOST (n.s.)")
        print(f"  [{mn:<15}] commit-wts t={rc['t']:+.2f} P(<0)={rc['p_neg']:.3f} -> {surv}")


if __name__ == "__main__":
    main()
