"""arc 2015 — honest-engine §5f for the month-end reversion SHORT side + the BIDIRECTIONAL book.

Observation (arc2015_observe) cleared §5d: short cap 0.551 (>0.50), POSITIVE in both binding folds
(2015 drift +0.437; 2018 cap 0.818 / drift +0.293) where the long-only 1011 is negative → non-coin-flip,
so §5f REQUIRES the honest engine under the registered exit menu before any verdict.

Scores (all via the canonical apparatus — Panel, build_arc_pool, ArcFoldRunner, build_v3_folds,
discovery_measure; never reimplemented):
  A. SHORT side §5f exit menu, IS per-fold + all-folds-positive judge + mean.
  B. Fair same-side NULL for the short (best exit) — does the structure beat random-entry shorts?
  C. LONG side (1011) per-fold under a common exit, for the combination.
  D. BIDIRECTIONAL book = long + short combined per-fold ROI (disjoint event timing → the linear
     combine_fold_roi is faithful), judged all-folds-positive on the COMBINED book.

Disposition (§11): all-folds-positive IS (short alone OR combined) → measure OOS; else mean-positive net
of costs → PORTFOLIO; else → KILL.
"""
from __future__ import annotations

import numpy as np
from datetime import date

from core.sim.panel import Panel
from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import judge_all_folds_positive, run_config_over_folds

from discovery.tools.month_end_signals import MonthEndReversionLongSignal, MonthEndReversionShortSignal
from discovery.tools.null_entry_baseline import build_null_signal_evaluation
from discovery.tools.combine_fold_roi import combine_fold_rois, fit_weights, rois_from_fold_stats

USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
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
    panel = Panel.from_pairs(USD_MAJORS, tf="D1", histdata_root=BACKUP, cache_root="data/cache",
                             boundary_convention="5ers_eet")
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    print(f"=== arc 2015 month-end SHORT + bidirectional — {len(is_folds)} IS folds ===")

    short_sig = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2)
    short_eval = short_sig.evaluate({"D1": panel})
    pool = build_arc_pool(short_sig, {"D1": panel}, ArcPoolConfig(
        arc_name="arc_2015_me_short", sl_atr_mult=2.0, hold_bars=120, risk_pct=0.005,
        window_start=date(2010, 1, 1), window_end=date(2020, 12, 31)))
    fr = pool.trades["final_r"]
    print(f"SHORT pool n={len(pool.trades)} mean final_r={fr.mean():+.3f} win={(fr>0).mean():.3f}")

    short_runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=short_eval, panels={"D1": panel})

    print("\n--- A. SHORT side §5f exit menu (IS) ---")
    best = None
    for ep in EXITS:
        stats, rois, v = score(short_runner, is_folds, f"me_short_{ep}", ep)
        mean = float(np.mean(rois)); nneg = sum(1 for r in rois if r < 0)
        print(f"  {ep:32s} AFP={str(v.all_folds_positive):5s} mean={mean*100:+6.3f}% worst={v.worst_fold_roi*100:+6.3f}% neg={nneg}/{len(rois)}")
        key = (v.all_folds_positive, mean)
        if best is None or key > best[0]:
            best = (key, ep, stats, mean, v)
    _, sep, sstats, smean, sv = best
    print(f"  --> SHORT best: {sep} mean={smean*100:+.3f}% AFP={sv.all_folds_positive} worst={sv.worst_fold_roi*100:+.3f}%")
    print("      per-fold:", per_fold_str(sstats))

    print("\n--- B. fair same-side NULL (random-entry SHORT, best exit) ---")
    null_means = []
    for seed in (42, 43, 44):
        ne = build_null_signal_evaluation(short_eval, seed=seed, warmup=100)
        nr = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=ne, panels={"D1": panel})
        _, nrois, _ = score(nr, is_folds, f"me_short_null{seed}", sep)
        null_means.append(float(np.mean(nrois)))
    print(f"  NULL mean (3 seeds) = {np.mean(null_means)*100:+.3f}%  | REAL short = {smean*100:+.3f}%  | excess = {(smean-np.mean(null_means))*100:+.3f}pp")

    print("\n--- C. LONG side (1011) per-fold, common exit sl_only ---")
    long_sig = MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2)
    long_eval = long_sig.evaluate({"D1": panel})
    long_runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=long_eval, panels={"D1": panel})
    lstats, lrois, lv = score(long_runner, is_folds, "me_long_sl_only", "sl_only")
    print(f"  LONG sl_only mean={np.mean(lrois)*100:+.3f}% AFP={lv.all_folds_positive} worst={lv.worst_fold_roi*100:+.3f}%")
    print("      per-fold:", per_fold_str(lstats))

    print("\n--- D. BIDIRECTIONAL book = LONG(sl_only) + SHORT(best) combined per-fold ROI ---")
    lr = rois_from_fold_stats(lstats); sr = rois_from_fold_stats(sstats)
    for mode in ("equal", "risk_parity"):
        w = fit_weights([lr, sr], mode)
        cb = combine_fold_rois([lr, sr], w)
        comb = cb.combined_roi
        nneg = sum(1 for r in comb if r < 0)
        afp = all(r > 0 for r in comb)
        print(f"  {mode:12s} w={[round(x,3) for x in w]} AFP={afp} mean={np.mean(comb)*100:+.3f}% worst={min(comb)*100:+.3f}% neg={nneg}/{len(comb)}")
        print("      per-fold:", ", ".join(f"{i}:{r*100:+.2f}" for i, r in enumerate(comb)))


if __name__ == "__main__":
    main()
