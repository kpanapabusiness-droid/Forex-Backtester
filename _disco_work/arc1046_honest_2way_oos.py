"""arc 1046 — Honest frozen-exit OOS: does the 2-way {me_long+fbr} beat the 4-way out-of-sample?

arc 1045 (IS) found the clean 2-way {me_long + fbr} is a materially better honest
deploy object than the 4-way: ~2x Sharpe, and it RECOVERS the statistical significance
the honest 4-way lost (afp metric t=+2.12, CI excludes zero), because gap + me_short are
net drags under honest exits. But that was IS only. This arc asks the durability
question: under the §5f FROZEN-exit discipline (choose each leg's exit on ALL IS, fit RP
weights on IS, FREEZE both, then score the 2021+ holdout ONCE — never re-selected, §4),
does the 2-way's IS superiority persist OOS?

Honest-frozen OOS is a CHARACTERIZATION data-point (per arc 1042), NOT a book-OOS gate —
the combined-book AFP holdout gate stays the operator's §5g firewall (the 2-way fails IS
AFP). We freeze on IS, measure OOS once, compare 2-way vs 4-way at honest-refit RP weights.

100% canonical scoring; BUILT nested_exit_selection (frozen_label) + combine_fold_roi.

Run:  PYTHONIOENCODING=utf-8 PYTHONPATH=. py _disco_work/arc1046_honest_2way_oos.py
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
from core.wfo.discovery_measure import build_oos_year_folds, run_config_over_folds
from core.wfo.folds import build_v3_folds
from discovery.tools.combine_fold_roi import combine_fold_rois, fit_weights
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.gap_signals import WeekendGapFillLongSignal
from discovery.tools.month_end_signals import (
    MonthEndReversionLongSignal,
    MonthEndReversionShortSignal,
)
from discovery.tools.nested_exit_selection import (
    freeze_best_over_folds,
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
NAMES4 = ("gap", "me_long", "fbr", "me_short")
SEED = 42
N_BOOT = 10000


def _load(pairs, tf):
    return Panel.from_pairs(pairs, tf, histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
                            use_cache=True, boundary_convention=BOUNDARY)


def _attach(sig, panel, n):
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n)
    return dataclasses.replace(sig, per_pair={
        p: dataclasses.replace(st, exit_predicate=pred) for p, st in sig.per_pair.items()})


def _series_by_label(fold_stats_by_label):
    return {lab: {fs.fold_id: fs.roi_pct for fs in fss} for lab, fss in fold_stats_by_label.items()}


def _stats(fold_rois, label):
    arr = np.array(fold_rois, dtype=float) * 100.0
    n = len(arr); mean = arr.mean(); sd = arr.std(ddof=1)
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    sharpe = mean / sd if sd > 0 else float("inf")
    rng = np.random.default_rng(SEED)
    means = arr[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    p_neg = float((means < 0).mean())
    print(f"  {label:<34} mean={mean:+.3f}%  sd={sd:.3f}  t={t:+.2f}  Sharpe={sharpe:+.3f}  "
          f"worst={arr.min():+.3f}%  neg={int((arr<0).sum())}/{n}  AFP={all(arr>0)}  P(<0)={p_neg:.3f}")
    return dict(mean=mean, sharpe=sharpe, worst=arr.min(), afp=bool(all(arr > 0)))


def main():
    print(f"arc 1046 — honest FROZEN-exit OOS, 2-way vs 4-way  (seed={SEED})\n")
    h4_usd = _load(USD_MAJORS, "H4"); h4_jpy = _load(JPY_CROSSES, "H4"); d1 = _load(USD_MAJORS, "D1")
    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    oos_folds = build_oos_year_folds(start_year=2021)
    print(f"IS folds: {[f.oos_start.year for f in is_folds]}")
    print(f"OOS folds: {sorted(getattr(f, 'fold_id', i) for i, f in enumerate(oos_folds))}\n")

    gap_sig = WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36).evaluate({"H4": h4_jpy})
    me_long_sig = _attach(MonthEndReversionLongSignal(1.0, 2).evaluate({"D1": d1}), d1, 2)
    fbr_sig = FailedBreakdownReclaimLongSignal(40, 1.25).evaluate({"H4": h4_usd})
    me_short_sig = MonthEndReversionShortSignal(1.0, 2).evaluate({"D1": d1})
    runners = {"me_long": ArcFoldRunner(A1Architecture(), me_long_sig, {"D1": d1}),
               "fbr": ArcFoldRunner(A1Architecture(), fbr_sig, {"H4": h4_usd}),
               "me_short": ArcFoldRunner(A1Architecture(), me_short_sig, {"D1": d1})}

    print("scoring grids IS + OOS (canonical)...")
    scored_is = {n: {} for n in NAMES4}
    scored_oos = {n: {} for n in NAMES4}
    for hz in GAP_HORIZONS:
        gr = ArcFoldRunner(A1Architecture(), _attach(gap_sig, h4_jpy, hz), {"H4": h4_jpy})
        for sl in SLS:
            lab = f"timeexit{hz}|sl{sl}"
            cfg = A1Config(config_id=f"gap_t{hz}_sl{sl}", sl_atr_mult=sl, trail_enabled=False, exit_policy=None)
            scored_is["gap"][lab] = list(run_config_over_folds(gr, is_folds, cfg))
            scored_oos["gap"][lab] = list(run_config_over_folds(gr, oos_folds, cfg))
    for name in ("me_long", "fbr", "me_short"):
        for ex in EXITS:
            for sl in SLS:
                lab = f"{ex}|sl{sl}"
                cfg = A1Config(config_id=f"{name}_{ex}_sl{sl}", sl_atr_mult=sl, trail_enabled=False, exit_policy=ex)
                scored_is[name][lab] = list(run_config_over_folds(runners[name], is_folds, cfg))
                scored_oos[name][lab] = list(run_config_over_folds(runners[name], oos_folds, cfg))

    metrics = [("mean_roi", metric_mean_roi), ("afp_then_mean", metric_afp_then_mean),
               ("worst_then_mean", metric_worst_then_mean)]

    for mn, mf in metrics:
        print("\n" + "=" * 96)
        print(f"FROZEN metric: {mn}  (per leg: exit chosen on ALL IS = frozen_label; scored ONCE on OOS)")
        print("=" * 96)
        # Freeze each leg's exit on IS, capture its IS evaluable series (for RP weights) + its OOS series.
        is_ev = {}; oos_fr = {}; frozen_lab = {}
        oos_series_lookup = {n: _series_by_label(scored_oos[n]) for n in NAMES4}
        for n in NAMES4:
            res = nested_walk_forward_select(scored_is[n], selection=mf, selection_name=mn, min_prior_folds=2)
            frozen_lab[n] = res.frozen_label
            is_ev[n] = [c.roi_pct for c in sorted(res.evaluable, key=lambda c: c.fold_id)]
            ol = oos_series_lookup[n][res.frozen_label]
            oos_fr[n] = [ol[fid] for fid in sorted(ol)]
            print(f"  {n:<9} frozen exit = {res.frozen_label:<28} OOS years scored: {sorted(ol)}")

        # RP weights fit on the IS honest series (FROZEN), applied to OOS.
        w2 = fit_weights([is_ev["me_long"], is_ev["fbr"]], mode="risk_parity")
        w4 = fit_weights([is_ev[n] for n in NAMES4], mode="risk_parity")
        print(f"  frozen RP wts: 2-way me_long={w2[0]:.3f} fbr={w2[1]:.3f} | "
              f"4-way " + " ".join(f"{n}={w:.3f}" for n, w in zip(NAMES4, w4)))

        book2_oos = list(combine_fold_rois([oos_fr["me_long"], oos_fr["fbr"]], w2).combined_roi)
        book4_oos = list(combine_fold_rois([oos_fr[n] for n in NAMES4], w4).combined_roi)
        print("  --- OOS (frozen, scored once) ---")
        _stats(oos_fr["me_long"], "me_long SOLO (OOS)")
        _stats(book4_oos, "4-way BOOK RP (OOS)")
        _stats(book2_oos, "2-way me_long+fbr RP (OOS)")
        # per-fold OOS for the 2-way
        oos_years = sorted(oos_series_lookup["me_long"][frozen_lab["me_long"]])
        print("  per-OOS-year (me_long / fbr / 2-way):")
        for y, rm, rf, rc in zip(oos_years, oos_fr["me_long"], oos_fr["fbr"], book2_oos):
            print(f"     {y}:  {rm*100:+7.3f}  {rf*100:+7.3f}  ->  {rc*100:+7.3f}%")


if __name__ == "__main__":
    main()
