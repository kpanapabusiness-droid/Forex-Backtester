"""arc 2058 — honest-engine §5f for the turn-of-quarter USD-reversion SHORT (XXXUSD majors).

Obs (arc2058_turn_of_quarter_usd) cleared §5d's coin-flip bar: post-quarter-end usd_long drift +0.296 ATR
(median +0.251, frac+ 0.601) >> ~0.085 ATR D1 cost, quarter-end-SPECIFIC (control other-month-ends ~0) ->
NON-coin-flip, so §5f REQUIRES the honest engine under the registered exit menu before any verdict (the
gross drift could still be gutted by the currency-exposure cap / cost — the arc-1017 lesson). Disposition:
all-folds-positive IS+OOS -> PASS; else mean-positive net of costs + beats null -> PORTFOLIO; else KILL.

Universe = the 4 XXXUSD majors (the clean short leg; USDXXX leg was weak/mixed in obs). Scored ENTIRELY
via the canonical apparatus (Panel, build_arc_pool, ArcFoldRunner, build_v3_folds, build_oos_year_folds,
discovery_measure); nothing reimplemented.
"""
from __future__ import annotations

import numpy as np
from datetime import date

from core.sim.panel import Panel
from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import judge_all_folds_positive, run_config_over_folds, build_oos_year_folds

from discovery.tools.quarter_end_signals import QuarterEndUsdReversionShortSignal
from discovery.tools.month_end_signals import MonthEndReversionShortSignal
from discovery.tools.null_entry_baseline import build_null_signal_evaluation
from discovery.tools.combine_fold_roi import rois_from_fold_stats

XXXUSD = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]
ALL7 = XXXUSD + ["USDJPY", "USDCAD", "USDCHF"]
BACKUP = r"C:\Users\panap\histdata_backup"
EXITS = ["sl_only", "sl_plus_tp_2r", "sl_plus_tp_3r", "sl_plus_trailing_atr",
         "sl_plus_trailing_swing", "sl_partial_close_1r_runner_trail"]


def score(runner, folds, tag, exit_policy):
    cfg = A1Config(config_id=tag, exit_policy=exit_policy, sl_atr_mult=2.0)
    stats = run_config_over_folds(runner, folds, cfg)
    rois = [fs.roi_pct for fs in stats]
    v = judge_all_folds_positive(stats)
    return stats, rois, v


def per_fold_str(stats):
    return ", ".join(f"{fs.fold_id}:{fs.roi_pct*100:+.2f}({fs.n_trades})" for fs in stats)


def main():
    qe_panel = Panel.from_pairs(XXXUSD, tf="D1", histdata_root=BACKUP, cache_root="data/cache",
                                boundary_convention="5ers_eet")
    panel = Panel.from_pairs(ALL7, tf="D1", histdata_root=BACKUP, cache_root="data/cache",
                             boundary_convention="5ers_eet")  # for the me_short correlation
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    oos_folds = build_oos_year_folds(start_year=2021)
    print(f"=== arc 2058 turn-of-quarter USD-reversion SHORT (XXXUSD) — {len(is_folds)} IS / {len(oos_folds)} OOS folds ===")

    sig = QuarterEndUsdReversionShortSignal()
    sig_eval = sig.evaluate({"D1": qe_panel})
    pool = build_arc_pool(sig, {"D1": qe_panel}, ArcPoolConfig(
        arc_name="arc_2058_qe_short", sl_atr_mult=2.0, hold_bars=120, risk_pct=0.005,
        window_start=date(2010, 1, 1), window_end=date(2020, 12, 31)))
    fr = pool.trades["final_r"]
    print(f"POOL n={len(pool.trades)} mean final_r={fr.mean():+.3f} win={(fr>0).mean():.3f}")

    runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=sig_eval, panels={"D1": qe_panel})

    print("\n--- A. §5f exit menu (IS folds) ---")
    best = None
    for ep in EXITS:
        stats, rois, v = score(runner, is_folds, f"qe_{ep}", ep)
        mean = float(np.mean(rois)); nneg = sum(1 for r in rois if r < 0)
        print(f"  {ep:32s} AFP={str(v.all_folds_positive):5s} mean={mean*100:+6.3f}% worst={v.worst_fold_roi*100:+6.3f}% neg={nneg}/{len(rois)}")
        key = (v.all_folds_positive, mean)
        if best is None or key > best[0]:
            best = (key, ep, stats, mean, v)
    _, bep, bstats, bmean, bv = best
    print(f"  --> best IS exit: {bep} mean={bmean*100:+.3f}% AFP={bv.all_folds_positive} worst={bv.worst_fold_roi*100:+.3f}%")
    print("      per-fold:", per_fold_str(bstats))

    print("\n--- B. fair same-side NULL (random-entry SHORT, best exit) ---")
    null_means = []
    for seed in (42, 43, 44):
        ne = build_null_signal_evaluation(sig_eval, seed=seed, warmup=100)
        nr = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=ne, panels={"D1": qe_panel})
        _, nrois, _ = score(nr, is_folds, f"qe_null{seed}", bep)
        null_means.append(float(np.mean(nrois)))
    print(f"  NULL mean (3 seeds) = {np.mean(null_means)*100:+.3f}%  | REAL = {bmean*100:+.3f}%  | excess = {(bmean-np.mean(null_means))*100:+.3f}pp")

    print("\n--- C. OOS (frozen best IS exit, measure-once) ---")
    ostats, orois, ov = score(runner, oos_folds, f"qe_oos_{bep}", bep)
    print(f"  OOS {bep} mean={np.mean(orois)*100:+.3f}% AFP={ov.all_folds_positive} worst={ov.worst_fold_roi*100:+.3f}% neg={sum(1 for r in orois if r<0)}/{len(orois)}")
    print("      per-fold:", per_fold_str(ostats))

    print("\n--- D. correlation vs me_short (1019) per-fold (decorrelation check) ---")
    ms = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2)
    ms_eval = ms.evaluate({"D1": panel})
    ms_runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=ms_eval, panels={"D1": panel})
    ms_stats, _, _ = score(ms_runner, is_folds, "me_short_partial", "sl_partial_close_1r_runner_trail")
    qe_r = np.array(rois_from_fold_stats(bstats)); ms_r = np.array(rois_from_fold_stats(ms_stats))
    if len(qe_r) == len(ms_r) and qe_r.std() > 0 and ms_r.std() > 0:
        print(f"  corr(qe_short, me_short) per-fold IS = {np.corrcoef(qe_r, ms_r)[0,1]:+.3f}")
    print("      qe per-fold:", ", ".join(f"{r*100:+.2f}" for r in qe_r))
    print("      ms per-fold:", ", ".join(f"{r*100:+.2f}" for r in ms_r))


if __name__ == "__main__":
    main()
