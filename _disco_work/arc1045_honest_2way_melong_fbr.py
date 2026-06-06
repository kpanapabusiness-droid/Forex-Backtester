"""arc 1045 — Honest-exit 2-way {me_long + fbr} book.

The §5f audit (arcs 1042/2042) showed that under honest nested-WFO exit selection
TWO of the four book legs (gap, me_short) flip MEAN-NEGATIVE — they are
exit-selection artifacts. The two robust legs are me_long (exit-honest, the heaviest
RP leg) and fbr (~40% haircut but stays positive, and the load-bearing diversifier:
honest corr −0.44…−0.65 with me_long). me_long carries 2018 (+0.90), fbr carries
2015 (+3.17) — the two folds that block every combined book.

QUESTION (deploy-relevant, never answered): is the cleanest honest deploy object the
2-way {me_long + fbr} rather than the noise-diluted 4-way? Does DROPPING the two
negative-mean exit-artifact legs IMPROVE the book (higher Sharpe, fewer neg folds,
closer to all-folds-positive)? And — since me_long and fbr are the two binding-fold
rescuers — does any convex weighting of JUST those two clear all-folds-positive?

100% canonical scoring; BUILT nested_exit_selection + combine_fold_roi. OOS untouched
(IS characterization only). Honest series = the EVALUABLE (non-warmup) folds per §5f
(2013-2020, 8 folds; warmup 2011/2012 dropped — no honest in-sample exit choice yet).

Run:  PYTHONPATH=. py _disco_work/arc1045_honest_2way_melong_fbr.py
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
NAMES4 = ("gap", "me_long", "fbr", "me_short")
N_BOOT = 10000
SEED = 42


def _load(pairs, tf):
    return Panel.from_pairs(pairs, tf, histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
                            use_cache=True, boundary_convention=BOUNDARY)


def _attach(sig, panel, n):
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n)
    return dataclasses.replace(sig, per_pair={
        p: dataclasses.replace(st, exit_predicate=pred) for p, st in sig.per_pair.items()})


def _stats(fold_rois):
    """mean/sd/t/Sharpe/worst/n_neg over a per-fold ROI series (decimal -> %)."""
    arr = np.array(fold_rois, dtype=float) * 100.0
    n = len(arr); mean = arr.mean(); sd = arr.std(ddof=1)
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    sharpe = mean / sd if sd > 0 else float("inf")
    rng = np.random.default_rng(SEED)
    means = arr[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_neg = float((means < 0).mean())
    return dict(mean=mean, sd=sd, t=t, sharpe=sharpe, worst=arr.min(),
                n_neg=int((arr < 0).sum()), n=n, lo=lo, hi=hi, p_neg=p_neg)


def _show(label, s):
    print(f"  {label:<30} mean={s['mean']:+.3f}%  sd={s['sd']:.3f}  t={s['t']:+.2f}  "
          f"Sharpe={s['sharpe']:+.3f}  worst={s['worst']:+.3f}%  neg={s['n_neg']}/{s['n']}  "
          f"CI=[{s['lo']:+.2f},{s['hi']:+.2f}]  P(<0)={s['p_neg']:.3f}")


def main():
    print(f"arc 1045 — honest 2-way me_long+fbr  (N_boot={N_BOOT} seed={SEED})\n")
    h4_usd = _load(USD_MAJORS, "H4"); h4_jpy = _load(JPY_CROSSES, "H4"); d1 = _load(USD_MAJORS, "D1")
    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    year_by_fid = {f.fold_id: f.oos_start.year for f in is_folds}

    gap_sig = WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36).evaluate({"H4": h4_jpy})
    me_long_sig = _attach(MonthEndReversionLongSignal(1.0, 2).evaluate({"D1": d1}), d1, 2)
    fbr_sig = FailedBreakdownReclaimLongSignal(40, 1.25).evaluate({"H4": h4_usd})
    me_short_sig = MonthEndReversionShortSignal(1.0, 2).evaluate({"D1": d1})
    runners = {"me_long": ArcFoldRunner(A1Architecture(), me_long_sig, {"D1": d1}),
               "fbr": ArcFoldRunner(A1Architecture(), fbr_sig, {"H4": h4_usd}),
               "me_short": ArcFoldRunner(A1Architecture(), me_short_sig, {"D1": d1})}

    print("scoring grids (canonical)...")
    scored = {n: {} for n in NAMES4}
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

    # Honest EVALUABLE (non-warmup) per-fold series per leg per metric, with fold years.
    honest = {}      # metric -> name -> [roi per evaluable fold]
    years = {}       # metric -> [year per evaluable fold]
    for mn, mf in metrics:
        honest[mn] = {}
        for n in NAMES4:
            res = nested_walk_forward_select(scored[n], selection=mf, selection_name=mn, min_prior_folds=2)
            ev = sorted(res.evaluable, key=lambda c: c.fold_id)
            honest[mn][n] = [c.roi_pct for c in ev]
            years[mn] = [year_by_fid[c.fold_id] for c in ev]

    for mn, _ in metrics:
        hs = honest[mn]; yrs = years[mn]
        print("\n" + "=" * 92)
        print(f"HONEST metric: {mn}   (evaluable folds: {yrs})")
        print("=" * 92)

        # --- solo + book stats ---
        s_me = _stats(hs["me_long"]); s_fbr = _stats(hs["fbr"])
        _show("me_long SOLO", s_me)
        _show("fbr SOLO", s_fbr)

        w4 = fit_weights([hs[n] for n in NAMES4], mode="risk_parity")
        book4 = list(combine_fold_rois([hs[n] for n in NAMES4], w4).combined_roi)
        _show("4-way BOOK (RP)", _stats(book4))

        for mode in ("equal", "risk_parity"):
            w2 = fit_weights([hs["me_long"], hs["fbr"]], mode=mode)
            book2 = list(combine_fold_rois([hs["me_long"], hs["fbr"]], w2).combined_roi)
            _show(f"2-way me_long+fbr ({mode}) w_me={w2[0]:.2f}", _stats(book2))

        # --- convex weight scan over the 2-way (does ANY weighting clear AFP?) ---
        best_worst = -1e9; best_w = None; afp_any = False; afp_ws = []
        for i in range(0, 101):
            wme = i / 100.0
            comb = list(combine_fold_rois([hs["me_long"], hs["fbr"]], [wme, 1 - wme]).combined_roi)
            worst = min(comb) * 100.0
            if worst > best_worst:
                best_worst = worst; best_w = wme
            if all(r > 0 for r in comb):
                afp_any = True; afp_ws.append(wme)
        print(f"  convex scan w_me in [0,1]: AFP achievable = {afp_any}"
              + (f"  (w_me in [{min(afp_ws):.2f},{max(afp_ws):.2f}])" if afp_any else "")
              + f"   best worst-fold = {best_worst:+.3f}% at w_me={best_w:.2f}")

        # --- per-fold table for the best-worst 2-way (binding folds) ---
        comb_best = list(combine_fold_rois([hs["me_long"], hs["fbr"]], [best_w, 1 - best_w]).combined_roi)
        print(f"  per-fold @ best w_me={best_w:.2f} (me_long / fbr / 2-way):")
        for y, rm, rf, rc in zip(yrs, hs["me_long"], hs["fbr"], comb_best):
            mark = "  <-- NEG" if rc <= 0 else ""
            print(f"     {y}:  {rm*100:+7.3f}  {rf*100:+7.3f}  ->  {rc*100:+7.3f}%{mark}")

        # honest pairwise corr me_long vs fbr
        corr = np.corrcoef(np.array([hs["me_long"], hs["fbr"]]) * 100.0)[0][1]
        print(f"  honest corr(me_long, fbr) = {corr:+.3f}")


if __name__ == "__main__":
    main()
