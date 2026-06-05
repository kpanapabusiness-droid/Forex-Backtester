"""arc 2015 — 4-WAY PORTFOLIO combination: does the month-end SHORT (robustly +2018) break the 2018 wall?

The 3-way book (gap 1006 + me-long 1011 + fbr 1013) is provably blocked (arcs 2008/3009/1015): 0/5151
convex weightings all-folds-positive, blocked by 2015 (only fbr +) and 2018 (only me-long +), mutually
exclusive. arc 2015 found a NEW component — month-end reversion SHORT — that is ROBUSTLY +2018 (survives
all LOO + thresholds; the corpus's first scalable +2018 leg). If a robustly-+2018 4th leg breaks the
2015-vs-2018 mutual exclusivity (2018 no longer needs only me-long; 2015 stays carried by fbr), the 4-way
book could clear all-folds-positive — the deployable gate.

Reproduces each component at its EXACT committed config (portfolio-candidates/*/config.yaml), VERIFIES the
3 known headlines (gap +0.685%, me-long +0.232%, fbr +1.854%) before trusting the combination, computes the
4-component per-fold correlation, and runs the 3-way + 4-way combination (equal + risk_parity), judged
all-folds-positive on the COMBINED book. Linear combine_fold_roi is the established portfolio method
(2008/3009/1015); faithful here (gap on JPY crosses disjoint from USD majors; me-long/me-short disjoint
event-timing down vs up months; me/fbr mostly disjoint timing).
"""
from __future__ import annotations

import dataclasses
import numpy as np
from datetime import date

from core.sim.panel import Panel
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import run_config_over_folds

from discovery.tools.gap_signals import WeekendGapFillLongSignal
from discovery.tools.month_end_signals import MonthEndReversionLongSignal, MonthEndReversionShortSignal
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.time_exit_predicate import make_time_exit_predicate
from discovery.tools.combine_fold_roi import combine_fold_rois, fit_weights, rois_from_fold_stats

BACKUP = r"C:\Users\panap\histdata_backup"
JPY = ["EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"]
USD = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
FOLD_YEAR = {fid: 2011 + (fid - 2) for fid in range(2, 12)}


def inject_time_exit(eval_, panel, pairs, n_bars):
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in pairs}, n_bars=n_bars)
    new_pp = {p: dataclasses.replace(st, exit_predicate=pred) for p, st in eval_.per_pair.items()}
    return dataclasses.replace(eval_, per_pair=new_pp)


def run(eval_, panel, tf, cfg, folds):
    runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=eval_, panels={tf: panel})
    return run_config_over_folds(runner, folds, cfg)


def main():
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    pj = Panel.from_pairs(JPY, tf="H4", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    pu4 = Panel.from_pairs(USD, tf="H4", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    pud = Panel.from_pairs(USD, tf="D1", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")

    # --- component 1: gap-fill 1006 (JPY crosses H4, thr0.5 gap36, sl_only + 24-bar time, trail OFF)
    ge = WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36).evaluate({"H4": pj})
    ge = inject_time_exit(ge, pj, JPY, 24)
    gstats = run(ge, pj, "H4", A1Config(config_id="gap", exit_policy="sl_only", sl_atr_mult=2.0, trail_enabled=False), is_folds)

    # --- component 2: me-long 1011 (USD majors D1, thr1.0, sl_only + 2-bar time, trail OFF)
    le = MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": pud})
    le = inject_time_exit(le, pud, USD, 2)
    lstats = run(le, pud, "D1", A1Config(config_id="me_long", exit_policy="sl_only", sl_atr_mult=2.0, trail_enabled=False), is_folds)

    # --- component 3: fbr 1013 (USD majors H4, K40 sh1.25, sl_plus_trailing_atr, trail ON)
    fe = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": pu4})
    fstats = run(fe, pu4, "H4", A1Config(config_id="fbr", exit_policy="sl_plus_trailing_atr", sl_atr_mult=2.0, trail_enabled=True), is_folds)

    # --- component 4 (NEW): me-short 2015 (USD majors D1, thr1.0, partial_runner §5f best, trail ON)
    se = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": pud})
    sstats = run(se, pud, "D1", A1Config(config_id="me_short", exit_policy="sl_partial_close_1r_runner_trail", sl_atr_mult=2.0, trail_enabled=True), is_folds)

    comps = {"gap": gstats, "me_long": lstats, "fbr": fstats, "me_short": sstats}
    print("=== component headline means (VERIFY vs committed: gap +0.685, me_long +0.232, fbr +1.854) ===")
    for k, s in comps.items():
        rois = [fs.roi_pct for fs in s]
        by = {FOLD_YEAR[fs.fold_id]: fs.roi_pct for fs in s}
        print(f"  {k:9s} mean={np.mean(rois)*100:+6.3f}%  2015={by[2015]*100:+.2f} 2018={by[2018]*100:+.2f}  neg={sum(1 for r in rois if r<0)}/{len(rois)}  n_is={sum(fs.n_trades for fs in s)}")

    R = {k: np.array(rois_from_fold_stats(s)) for k, s in comps.items()}
    print("\n=== per-fold ROI correlation (4 components) ===")
    keys = list(R)
    print("           " + "  ".join(f"{k:>8s}" for k in keys))
    for a in keys:
        print(f"  {a:8s} " + "  ".join(f"{np.corrcoef(R[a],R[b])[0,1]:+8.3f}" for b in keys))

    def book(parts, mode):
        rl = [R[p] for p in parts]
        w = fit_weights(rl, mode)
        comb = combine_fold_rois(rl, w).combined_roi
        afp = all(r > 0 for r in comb)
        by = {FOLD_YEAR[2 + i]: r for i, r in enumerate(comb)}
        return w, comb, afp, by

    for parts, label in [(["gap", "me_long", "fbr"], "3-WAY (baseline, arc 1015)"),
                         (["gap", "me_long", "fbr", "me_short"], "4-WAY (+me_short)")]:
        print(f"\n=== {label} ===")
        for mode in ("equal", "risk_parity"):
            w, comb, afp, by = book(parts, mode)
            print(f"  {mode:12s} w={[round(x,3) for x in w]}")
            print(f"     AFP={afp} mean={np.mean(comb)*100:+.3f}% worst={min(comb)*100:+.3f}% neg={sum(1 for r in comb if r<0)}/{len(comb)}")
            print(f"     per-year: " + " ".join(f"{FOLD_YEAR[2+i]}:{r*100:+.2f}" for i, r in enumerate(comb)))


if __name__ == "__main__":
    main()


def convex_search():
    """Grid-search convex weights over the 4 components (step 0.05) — best achievable worst-fold.
    Mirrors arc 2008's 0/5151 convex search; the decisive 'is the 4-way book clearable' test."""
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    pj = Panel.from_pairs(JPY, tf="H4", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    pu4 = Panel.from_pairs(USD, tf="H4", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    pud = Panel.from_pairs(USD, tf="D1", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    ge = inject_time_exit(WeekendGapFillLongSignal(0.5, gap_hours=36).evaluate({"H4": pj}), pj, JPY, 24)
    gstats = run(ge, pj, "H4", A1Config(config_id="g", exit_policy="sl_only", sl_atr_mult=2.0, trail_enabled=False), is_folds)
    le = inject_time_exit(MonthEndReversionLongSignal(1.0, 2).evaluate({"D1": pud}), pud, USD, 2)
    lstats = run(le, pud, "D1", A1Config(config_id="l", exit_policy="sl_only", sl_atr_mult=2.0, trail_enabled=False), is_folds)
    fstats = run(FailedBreakdownReclaimLongSignal(40, 1.25).evaluate({"H4": pu4}), pu4, "H4", A1Config(config_id="f", exit_policy="sl_plus_trailing_atr", sl_atr_mult=2.0, trail_enabled=True), is_folds)
    sstats = run(MonthEndReversionShortSignal(1.0, 2).evaluate({"D1": pud}), pud, "D1", A1Config(config_id="s", exit_policy="sl_partial_close_1r_runner_trail", sl_atr_mult=2.0, trail_enabled=True), is_folds)
    R = [np.array(rois_from_fold_stats(s)) for s in (gstats, lstats, fstats, sstats)]
    names = ["gap", "me_long", "fbr", "me_short"]

    grid = [i / 20 for i in range(21)]
    best = None; n_pass = 0; n_tot = 0
    for a in grid:
        for b in grid:
            if a + b > 1: continue
            for c in grid:
                d = 1 - a - b - c
                if d < -1e-9 or d > 1 + 1e-9: continue
                w = np.array([a, b, c, d]); n_tot += 1
                comb = sum(wi * Ri for wi, Ri in zip(w, R))
                mn = comb.min()
                if mn > 0: n_pass += 1
                if best is None or mn > best[0]:
                    best = (mn, w.copy(), comb.copy())
    print(f"\n=== 4-way CONVEX SEARCH (step 0.05): {n_pass}/{n_tot} weightings all-folds-positive ===")
    mn, w, comb = best
    print(f"  best max-min worst-fold = {mn*100:+.3f}%  at w={dict(zip(names, [round(float(x),3) for x in w]))}")
    print(f"  per-year: " + " ".join(f"{FOLD_YEAR[2+i]}:{r*100:+.2f}" for i, r in enumerate(comb)))
    neg = [FOLD_YEAR[2+i] for i, r in enumerate(comb) if r <= 0]
    print(f"  binding (<=0) folds at best weighting: {neg}")

convex_search()
