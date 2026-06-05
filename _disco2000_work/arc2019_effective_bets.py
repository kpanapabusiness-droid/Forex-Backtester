"""arc 2019 — DIAGNOSTIC: how many INDEPENDENT bets is the 4-component book? (council-driven)

The discovery council (arc 2019) converged on the one thing arcs 2016/2017 left unmeasured: the
EFFECTIVE NUMBER OF INDEPENDENT BETS in the existing 4-component book, and its TAIL co-movement.
The fork it resolves:
  - If the book is effectively rank-~1 (one reversion factor sliced 4 ways), "0/all convex
    weightings all-folds-positive" is MECHANICALLY INEVITABLE -> no 5th thin leg or gate reframe
    helps -> the lever is the operator gate-governance call.
  - If genuinely multi-bet in the BODY but rank-low in the TAIL (all bleed together in strong-USD
    risk-off 2018), then no 5th REVERSION leg can fix it (shares the tail); only a NON-reversion
    factor (trend/continuation = coin-flip dead) or the gate call remains.

Reproduces the 4 committed components EXACTLY (reusing arc 2015's config), then computes:
  (1) per-fold ROI correlation matrix (4x4)
  (2) eigenvalue spectrum + effective number of bets ENB = (sum L)^2 / sum(L^2) (participation ratio)
  (3) bootstrap CI of ENB (resample the 10 folds) -- is the (de)correlation even resolvable?
  (4) TAIL co-movement: co-negativity per fold; corr restricted to worst vs best folds
  (5) book pooled per-fold mean CI (equal + risk-parity, frozen weights): is the MEAN robustly >0?

DIAGNOSTIC, no new component, no OOS spent. Scale-invariant ratios (corr, ENB, sigma-units) are
robust to the arc-3017 risk_pct convention flag. Reuses CANONICAL measurement only.
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
FOLD_YEAR = {fid: 2011 + (fid - 2) for fid in range(2, 12)}
RNG = np.random.RandomState(42)


def inject_time_exit(eval_, panel, pairs, n_bars):
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in pairs}, n_bars=n_bars)
    new_pp = {p: dataclasses.replace(st, exit_predicate=pred) for p, st in eval_.per_pair.items()}
    return dataclasses.replace(eval_, per_pair=new_pp)


def run(eval_, panel, tf, cfg, folds):
    runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=eval_, panels={tf: panel})
    return run_config_over_folds(runner, folds, cfg)


def enb(corr):
    """Effective number of bets = participation ratio of the correlation-matrix eigenvalues."""
    w = np.linalg.eigvalsh(corr)
    w = np.clip(w, 0, None)
    return (w.sum() ** 2) / (np.square(w).sum()), np.sort(w)[::-1]


def main():
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    pj = Panel.from_pairs(JPY, tf="H4", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    pu4 = Panel.from_pairs(USD, tf="H4", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")
    pud = Panel.from_pairs(USD, tf="D1", histdata_root=BACKUP, cache_root="data/cache", boundary_convention="5ers_eet")

    ge = inject_time_exit(WeekendGapFillLongSignal(0.5, gap_hours=36).evaluate({"H4": pj}), pj, JPY, 24)
    gstats = run(ge, pj, "H4", A1Config(config_id="gap", exit_policy="sl_only", sl_atr_mult=2.0, trail_enabled=False), is_folds)
    le = inject_time_exit(MonthEndReversionLongSignal(1.0, 2).evaluate({"D1": pud}), pud, USD, 2)
    lstats = run(le, pud, "D1", A1Config(config_id="me_long", exit_policy="sl_only", sl_atr_mult=2.0, trail_enabled=False), is_folds)
    fstats = run(FailedBreakdownReclaimLongSignal(40, 1.25).evaluate({"H4": pu4}), pu4, "H4", A1Config(config_id="fbr", exit_policy="sl_plus_trailing_atr", sl_atr_mult=2.0, trail_enabled=True), is_folds)
    sstats = run(MonthEndReversionShortSignal(1.0, 2).evaluate({"D1": pud}), pud, "D1", A1Config(config_id="me_short", exit_policy="sl_partial_close_1r_runner_trail", sl_atr_mult=2.0, trail_enabled=True), is_folds)

    comps = {"gap": gstats, "me_long": lstats, "fbr": fstats, "me_short": sstats}
    keys = list(comps)
    R = {k: np.array(rois_from_fold_stats(s)) for k, s in comps.items()}
    M = np.vstack([R[k] for k in keys])  # 4 x 10

    print("=== component headlines (VERIFY: gap +0.685 me_long +0.232 fbr +1.854 me_short +0.683) ===")
    for k in keys:
        by = {FOLD_YEAR[fs.fold_id]: fs.roi_pct for fs in comps[k]}
        print(f"  {k:9s} mean={R[k].mean()*100:+6.3f}%  neg={int((R[k]<0).sum())}/{len(R[k])}  2015={by[2015]*100:+.2f} 2018={by[2018]*100:+.2f}")

    corr = np.corrcoef(M)
    print("\n=== (1) per-fold ROI correlation (n=10 folds) ===")
    print("           " + "  ".join(f"{k:>8s}" for k in keys))
    for i, a in enumerate(keys):
        print(f"  {a:8s} " + "  ".join(f"{corr[i,j]:+8.3f}" for j in range(len(keys))))

    e_point, evals = enb(corr)
    print(f"\n=== (2) eigenvalue spectrum + effective number of bets ===")
    print(f"  eigenvalues (desc): {[round(float(x),3) for x in evals]}")
    print(f"  ENB (participation ratio) = {e_point:.3f}  (1=rank-1/one bet, 4=fully independent)")
    print(f"  top eigenvalue explains {evals[0]/evals.sum()*100:.1f}% of cross-fold variance")

    print("\n=== (3) bootstrap CI of ENB (resample 10 folds, 5000x, seed 42) ===")
    boots = []
    for _ in range(5000):
        idx = RNG.randint(0, M.shape[1], M.shape[1])
        Mb = M[:, idx]
        if np.any(Mb.std(axis=1) == 0):
            continue
        cb = np.corrcoef(Mb)
        eb, _ = enb(cb)
        boots.append(eb)
    boots = np.array(boots)
    print(f"  ENB mean={boots.mean():.3f}  median={np.median(boots):.3f}  95% CI=[{np.percentile(boots,2.5):.3f}, {np.percentile(boots,97.5):.3f}]")
    print(f"  P(ENB<2)={np.mean(boots<2):.3f}  P(ENB>3)={np.mean(boots>3):.3f}")

    print("\n=== (4) TAIL co-movement (the real driver of gate failure) ===")
    n_neg = (M < 0).sum(axis=0)  # how many components negative per fold
    for i in range(M.shape[1]):
        yr = FOLD_YEAR[2 + i]
        negs = [keys[k] for k in range(4) if M[k, i] < 0]
        print(f"  {yr}: {int(n_neg[i])}/4 neg  ({','.join(negs) if negs else 'none'})")
    # correlation restricted to the book's worst vs best folds (equal-weight book)
    book_eq = M.mean(axis=0)
    order = np.argsort(book_eq)
    worst = order[:5]; best = order[5:]
    cw = np.corrcoef(M[:, worst]); cb = np.corrcoef(M[:, best])
    iu = np.triu_indices(4, 1)
    print(f"  mean pairwise corr in book's 5 WORST folds = {cw[iu].mean():+.3f}  (tail co-movement)")
    print(f"  mean pairwise corr in book's 5 BEST  folds = {cb[iu].mean():+.3f}")
    print(f"  mean pairwise corr ALL folds              = {corr[iu].mean():+.3f}")

    print("\n=== (5) book pooled per-fold MEAN CI (frozen weights; resample folds 5000x) ===")
    rl = [R[k] for k in keys]
    for mode in ("equal", "risk_parity"):
        w = fit_weights(rl, mode)
        comb = np.array(combine_fold_rois(rl, w).combined_roi)
        bm = np.array([comb[RNG.randint(0, len(comb), len(comb))].mean() for _ in range(5000)])
        print(f"  {mode:12s} mean={comb.mean()*100:+.3f}%  worst={comb.min()*100:+.3f}%  neg={int((comb<0).sum())}/{len(comb)}  "
              f"mean 95% CI=[{np.percentile(bm,2.5)*100:+.3f}%, {np.percentile(bm,97.5)*100:+.3f}%]  P(mean<0)={np.mean(bm<0):.3f}")


if __name__ == "__main__":
    main()
