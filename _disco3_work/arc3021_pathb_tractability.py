"""arc 3021 — PATH-B TRACTABILITY: how many decorrelated positive-mean legs would the
per-year all-folds-positive (AFP) gate actually require?

The corpus has converged (arcs 2016/2017/1023/2019/2021): the 4-component book is a sound
~3-independent-bet (ENB=3.32), mean-positive (t=2.66, P(mean<0)=0.004), temporally-robust
PORTFOLIO whose AFP FAILURE is purely the per-year gate sitting below the legs' noise floor.
arc 2019 concluded a 5th *reversion* leg "can't help" (it tested a specific leg / fold-painting
logic). The operator's stated lever is the path-A (change the gate) vs path-B (denser book) call,
left QUALITATIVE by every prior arc.

This diagnostic QUANTIFIES path-B and SHARPENS arc 2019's claim. Portfolio math: averaging N
decorrelated positive-mean legs shrinks the book's per-fold variance (~1/N_eff) -> book per-fold
Sharpe = (per-leg Sharpe)*sqrt(N_eff) -> P(AFP over 10 folds) = Phi(book_Sharpe)^10 rises with N.
So AFP IS reachable at SOME N even with thin legs -- the real question is: at what N, and is that
N tractable given the corpus finds ~1 PORTFOLIO leg per ~14 arcs?

Calibrated from the REAL 4 legs (reproduced via the CANONICAL apparatus, same configs as arcs
2019/2021). DIAGNOSTIC: no new component, no OOS, no gate loosened, no council (a measurement
informing a measurement). ROI at the arc-1024-confirmed 0.5% deployable risk (FRACTION).
"""
from __future__ import annotations

import dataclasses
import numpy as np

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


def inject_time_exit(eval_, panel, pairs, n_bars):
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in pairs}, n_bars=n_bars)
    new_pp = {p: dataclasses.replace(st, exit_predicate=pred) for p, st in eval_.per_pair.items()}
    return dataclasses.replace(eval_, per_pair=new_pp)


def run(eval_, panel, tf, cfg, folds):
    runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=eval_, panels={tf: panel})
    return run_config_over_folds(runner, folds, cfg)


def reproduce_legs():
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    pj = Panel.from_pairs(JPY, tf="H4", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    pu4 = Panel.from_pairs(USD, tf="H4", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    pud = Panel.from_pairs(USD, tf="D1", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")

    ge = inject_time_exit(WeekendGapFillLongSignal(0.5, gap_hours=36).evaluate({"H4": pj}), pj, JPY, 24)
    g = run(ge, pj, "H4", A1Config(config_id="gap", exit_policy="sl_only", sl_atr_mult=2.0, trail_enabled=False), is_folds)
    le = inject_time_exit(MonthEndReversionLongSignal(1.0, 2).evaluate({"D1": pud}), pud, USD, 2)
    l = run(le, pud, "D1", A1Config(config_id="me_long", exit_policy="sl_only", sl_atr_mult=2.0, trail_enabled=False), is_folds)
    f = run(FailedBreakdownReclaimLongSignal(40, 1.25).evaluate({"H4": pu4}), pu4, "H4", A1Config(config_id="fbr", exit_policy="sl_plus_trailing_atr", sl_atr_mult=2.0, trail_enabled=True), is_folds)
    s = run(MonthEndReversionShortSignal(1.0, 2).evaluate({"D1": pud}), pud, "D1", A1Config(config_id="me_short", exit_policy="sl_partial_close_1r_runner_trail", sl_atr_mult=2.0, trail_enabled=True), is_folds)
    return {"gap": np.array(rois_from_fold_stats(g)), "me_long": np.array(rois_from_fold_stats(l)),
            "fbr": np.array(rois_from_fold_stats(f)), "me_short": np.array(rois_from_fold_stats(s))}


def p_afp(per_leg_sharpe, N, n_folds=10, rho=0.0, sims=20000, seed=42):
    """P(all n_folds book-fold ROIs > 0) for an equal-weight book of N i.i.d.(rho-equicorr) legs,
    each leg per-fold ROI standardized to mean=per_leg_sharpe, sd=1 (so mean/sd = Sharpe)."""
    rng = np.random.RandomState(seed)
    hits = 0
    for _ in range(sims):
        # leg-fold matrix [N, n_folds], equicorrelation rho across legs within a fold
        z = rng.standard_normal((N, n_folds))
        if rho > 0:
            common = rng.standard_normal((1, n_folds))
            z = np.sqrt(1 - rho) * z + np.sqrt(rho) * common
        legf = per_leg_sharpe + z            # mean shift = Sharpe (sd=1)
        book_fold = legf.mean(axis=0)        # equal-weight average over legs
        if np.all(book_fold > 0):
            hits += 1
    return hits / sims


def main():
    R = reproduce_legs()
    keys = list(R)
    print("=== (0) component reproduction (VERIFY gap +0.685 me_long +0.232 fbr +1.854 me_short +0.683) ===")
    sharpes = {}
    for k in keys:
        v = R[k]
        sh = v.mean() / v.std(ddof=1)
        sharpes[k] = sh
        print(f"  {k:9s} mean={v.mean()*100:+6.3f}%  sd={v.std(ddof=1)*100:5.3f}%  per-fold Sharpe={sh:+.3f}  neg={int((v<0).sum())}/{len(v)}")

    # empirical pairwise correlation (decorrelation check, ties arc 2019 ENB)
    M = np.vstack([R[k] for k in keys])
    C = np.corrcoef(M)
    iu = np.triu_indices(len(keys), 1)
    mean_corr = C[iu].mean()
    print(f"\n  mean pairwise leg corr = {mean_corr:+.3f}  (arc 2019: -0.366..+0.406, ENB 3.32/4)")

    # the typical-leg Sharpe to calibrate synthetic legs (use the 4 real legs)
    s_arr = np.array([sharpes[k] for k in keys])
    print(f"\n  per-leg per-fold Sharpe: min {s_arr.min():.3f}  median {np.median(s_arr):.3f}  mean {s_arr.mean():.3f}  max {s_arr.max():.3f}")

    # required book Sharpe for AFP analytically: Phi(S)^10 = target
    from scipy.stats import norm
    for tgt in (0.5, 0.9):
        S_req = norm.ppf(tgt ** (1 / 10))
        print(f"  analytic: P(AFP over 10 folds)={tgt} needs book per-fold Sharpe >= {S_req:.3f}")

    print("\n=== (1) P(AFP) vs N decorrelated legs (rho=0), at calibrated per-leg Sharpe levels ===")
    print("  (book per-fold Sharpe grows ~ per_leg_Sharpe * sqrt(N))")
    for lab, sh in [("median-leg", float(np.median(s_arr))), ("mean-leg", float(s_arr.mean())),
                    ("best-leg(fbr-class)", float(s_arr.max())), ("optimistic 0.50", 0.50)]:
        row = []
        for N in (4, 6, 8, 10, 15, 20, 30, 50):
            row.append(f"N={N}:{p_afp(sh, N, rho=0.0):.2f}")
        print(f"  {lab:22s} (Sharpe {sh:+.3f})  " + "  ".join(row))

    print("\n=== (2) effect of residual correlation (rho>0 raises required N) ===")
    sh = float(np.median(s_arr))
    for rho in (0.0, 0.1, 0.2):
        row = [f"N={N}:{p_afp(sh, N, rho=rho):.2f}" for N in (8, 15, 30, 50, 80)]
        print(f"  median-leg Sharpe {sh:+.3f}, rho={rho}:  " + "  ".join(row))

    print("\n=== (3) N* required for P(AFP)>=0.9 (decorrelated), by per-leg Sharpe ===")
    for lab, sh in [("median-leg", float(np.median(s_arr))), ("mean-leg", float(s_arr.mean())),
                    ("best-leg", float(s_arr.max())), ("optimistic 0.50", 0.50)]:
        Nstar = None
        for N in range(2, 201):
            if p_afp(sh, N, rho=0.0, sims=4000) >= 0.9:
                Nstar = N
                break
        print(f"  {lab:22s} (Sharpe {sh:+.3f}):  N* ~ {Nstar if Nstar else '>200'} legs")

    print("\n  corpus production rate: 4 PORTFOLIO legs in ~55 arcs  (~1 per 14 arcs)")


if __name__ == "__main__":
    main()
