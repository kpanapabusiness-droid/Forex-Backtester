"""arc 2021 — ADVERSARIAL temporal-stability stress of the 4-component book's mean-positive edge.

Arc 2019 established the book is a sound ~3-bet, mean-positive PORTFOLIO (risk-parity +0.589%,
P(mean<0)=0.004) whose all-folds-positive failure is a gate-resolution artifact -> the lever is the
operator's gate-governance call. BEFORE anyone leans on that mean for a deploy decision, the first
adversarial question (never asked by arcs 2016/2017/1023/2019, all of which characterized NOISE, not
TIME): is the edge STABLE across the IS decade, or front-loaded in an early-era (2011-2015) regime that
has since decayed? A decayed edge makes path-A (gate-resolution) hopeless and should downgrade the book;
a stable edge genuinely strengthens the deploy case (pending the operator's call).

CONSERVATIVE BIAS: try to BREAK the mean (show it is front-loaded / late-half-dead). Reproduces the 4
committed components EXACTLY (reusing arc 2019's frozen configs), then:
  (1) per-component + book per-year ROI (2011-2020); verify headlines.
  (2) EARLY (2011-2015) vs LATE (2016-2020) half: mean, n-neg, sign — per component and risk-parity book.
  (3) bootstrap each half's book-mean CI + the early-minus-late difference CI -> is decay significant?
  (4) the decision-critical number: is the LATE-half book mean robustly > 0 (would the edge plausibly
      persist into 2021+)?

DIAGNOSTIC, no new component, no OOS spent, no gate loosened. Reuses CANONICAL measurement only.
Scale-invariant where possible; ROI levels are at the arc-1024-confirmed 0.5% deployable risk (FRACTION).
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
FOLD_YEAR = {fid: 2011 + (fid - 2) for fid in range(2, 12)}   # fid 2..11 -> 2011..2020
RNG = np.random.RandomState(42)
EARLY = set(range(2011, 2016))   # 2011-2015
LATE = set(range(2016, 2021))    # 2016-2020


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

    ge = inject_time_exit(WeekendGapFillLongSignal(0.5, gap_hours=36).evaluate({"H4": pj}), pj, JPY, 24)
    gstats = run(ge, pj, "H4", A1Config(config_id="gap", exit_policy="sl_only", sl_atr_mult=2.0, trail_enabled=False), is_folds)
    le = inject_time_exit(MonthEndReversionLongSignal(1.0, 2).evaluate({"D1": pud}), pud, USD, 2)
    lstats = run(le, pud, "D1", A1Config(config_id="me_long", exit_policy="sl_only", sl_atr_mult=2.0, trail_enabled=False), is_folds)
    fstats = run(FailedBreakdownReclaimLongSignal(40, 1.25).evaluate({"H4": pu4}), pu4, "H4", A1Config(config_id="fbr", exit_policy="sl_plus_trailing_atr", sl_atr_mult=2.0, trail_enabled=True), is_folds)
    sstats = run(MonthEndReversionShortSignal(1.0, 2).evaluate({"D1": pud}), pud, "D1", A1Config(config_id="me_short", exit_policy="sl_partial_close_1r_runner_trail", sl_atr_mult=2.0, trail_enabled=True), is_folds)

    comps = {"gap": gstats, "me_long": lstats, "fbr": fstats, "me_short": sstats}
    keys = list(comps)
    R = {k: np.array(rois_from_fold_stats(s)) for k, s in comps.items()}
    years = np.array([FOLD_YEAR[fs.fold_id] for fs in gstats])
    early_mask = np.array([y in EARLY for y in years])
    late_mask = np.array([y in LATE for y in years])

    print("=== component headlines (VERIFY: gap +0.685 me_long +0.232 fbr +1.854 me_short +0.683) ===")
    for k in keys:
        print(f"  {k:9s} mean={R[k].mean()*100:+6.3f}%  neg={int((R[k]<0).sum())}/{len(R[k])}")
    print(f"  years order: {years.tolist()}")

    rl = [R[k] for k in keys]
    w_rp = fit_weights(rl, "risk_parity")
    w_eq = fit_weights(rl, "equal")
    book_rp = np.array(combine_fold_rois(rl, w_rp).combined_roi)
    book_eq = np.array(combine_fold_rois(rl, w_eq).combined_roi)

    print("\n=== (1) per-year book ROI (risk-parity, frozen weights) ===")
    for y, r in sorted(zip(years, book_rp)):
        print(f"  {y}: {r*100:+.3f}%   {'EARLY' if y in EARLY else 'LATE'}")

    print("\n=== (2) EARLY (2011-2015) vs LATE (2016-2020) half — per component + book ===")
    print(f"  {'series':12s} {'EARLY mean':>11s} {'neg':>5s}   {'LATE mean':>11s} {'neg':>5s}   {'decay(E-L)':>11s}")
    for k in keys:
        em, lm = R[k][early_mask].mean(), R[k][late_mask].mean()
        en, ln = int((R[k][early_mask] < 0).sum()), int((R[k][late_mask] < 0).sum())
        print(f"  {k:12s} {em*100:+10.3f}% {en:>3d}/5   {lm*100:+10.3f}% {ln:>3d}/5   {(em-lm)*100:+10.3f}%")
    for name, bk in [("book_rp", book_rp), ("book_eq", book_eq)]:
        em, lm = bk[early_mask].mean(), bk[late_mask].mean()
        en, ln = int((bk[early_mask] < 0).sum()), int((bk[late_mask] < 0).sum())
        print(f"  {name:12s} {em*100:+10.3f}% {en:>3d}/5   {lm*100:+10.3f}% {ln:>3d}/5   {(em-lm)*100:+10.3f}%")

    print("\n=== (3) bootstrap each half's book MEAN CI + early-minus-late decay CI (5000x, seed 42) ===")
    for name, bk in [("book_rp", book_rp), ("book_eq", book_eq)]:
        e, l = bk[early_mask], bk[late_mask]
        be = np.array([e[RNG.randint(0, 5, 5)].mean() for _ in range(5000)])
        bl = np.array([l[RNG.randint(0, 5, 5)].mean() for _ in range(5000)])
        bd = be - bl
        print(f"  {name}: EARLY mean={e.mean()*100:+.3f}% CI=[{np.percentile(be,2.5)*100:+.3f},{np.percentile(be,97.5)*100:+.3f}] "
              f"P(<0)={np.mean(be<0):.3f}")
        print(f"  {name}: LATE  mean={l.mean()*100:+.3f}% CI=[{np.percentile(bl,2.5)*100:+.3f},{np.percentile(bl,97.5)*100:+.3f}] "
              f"P(<0)={np.mean(bl<0):.3f}   <-- decision-critical: does the edge persist?")
        print(f"  {name}: DECAY (early-late)={bd.mean()*100:+.3f}% CI=[{np.percentile(bd,2.5)*100:+.3f},{np.percentile(bd,97.5)*100:+.3f}] "
              f"P(decay>0)={np.mean(bd>0):.3f}")


if __name__ == "__main__":
    main()
