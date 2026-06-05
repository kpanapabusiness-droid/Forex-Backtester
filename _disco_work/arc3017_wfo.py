"""Arc 3017 honest-engine WFO -- Month-End Reversion SHORT (the decisive §5f test).

Non-coin-flip entry (capture 0.5508>0.50, structure control passes) -> §5f mandates the engine
with the registered exit menu swept BEFORE any FAIL. Scored solely by MultiPairBacktester via
ArcFoldRunner, FundedNext costs ON, SL-first take-the-loss.

Reports per-fold IS ROI series for each exit, all-folds-positive verdict, and -- the whole point --
the 2015 & 2018 fold sign (the binding portfolio folds). OOS preserved (not touched unless IS
all-folds-positive). Plus the fair same-side null (arc 2013 dir-aware) on the best config.
"""
from __future__ import annotations

import dataclasses
from datetime import date

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool
from core.arc.signal_protocol import SignalEvaluation
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import judge_all_folds_positive, run_config_over_folds
from discovery.tools.month_end_signals import MonthEndReversionShortSignal
from discovery.tools.time_exit_predicate import make_time_exit_predicate
from discovery.tools.null_entry_baseline import build_null_signal_evaluation

PAIRS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
panel = Panel.from_pairs(PAIRS, tf="D1", histdata_root=r"C:\Users\panap\histdata_backup",
                         cache_root="data/cache", boundary_convention="5ers_eet")

sig = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2)

# Pool sanity (n_trades, gross mean R)
pool = build_arc_pool(sig, {"D1": panel}, ArcPoolConfig(
    arc_name="arc_3017_month_end_reversion_short", sl_atr_mult=2.0, hold_bars=120, risk_pct=0.005,
    window_start=date(2010, 1, 1), window_end=date(2020, 12, 31)))
print(f"POOL: n_trades={len(pool.trades)}  gross mean_final_r={pool.trades['final_r'].mean():+.4f}")

sig_eval = sig.evaluate({"D1": panel})
is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
# map fold -> OOS year for the acceptance test
fold_year = {}
for f in is_folds:
    try:
        fold_year[f.fold_id] = pd.Timestamp(f.oos_start).year
    except Exception:
        fold_year[f.fold_id] = None
print("fold->oos_year:", fold_year)


def eval_with_time_exit(base_eval, n_bars):
    pp = {p: dataclasses.replace(st, exit_predicate=make_time_exit_predicate(
        {p: panel.pair_dfs[p]}, n_bars=n_bars)) for p, st in base_eval.per_pair.items()}
    return SignalEvaluation(primary_tf=base_eval.primary_tf, per_pair=pp,
                            signal_name=base_eval.signal_name + f"_te{n_bars}",
                            causal_lineage=base_eval.causal_lineage, direction=base_eval.direction)


def run_and_report(label, eval_obj, exit_policy):
    runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=eval_obj, panels={"D1": panel})
    cfg = A1Config(config_id=f"arc3017_{label}", exit_policy=exit_policy, sl_atr_mult=2.0, risk_pct=0.005)
    stats = run_config_over_folds(runner, is_folds, cfg)
    v = judge_all_folds_positive(stats)
    rois = {fold_year.get(s.fold_id, s.fold_id): s.roi_pct for s in stats}
    series = [rois[y] for y in sorted(rois)]
    npos = sum(1 for r in series if r > 0)
    mean = float(np.mean(series))
    worst = float(np.min(series))
    y2015 = rois.get(2015, float("nan"))
    y2018 = rois.get(2018, float("nan"))
    print(f"\n{label:36s} folds_pos={npos}/{len(series)} mean={mean:+.3f}% worst={worst:+.3f}% "
          f"AFP={v.all_folds_positive}  2015={y2015:+.3f}% 2018={y2018:+.3f}%")
    print("   per-fold(by oos yr): " + " ".join(f"{y}:{rois[y]:+.2f}" for y in sorted(rois)))
    return mean, npos, series, rois


print("\n=== REGISTERED EXIT MENU (sl_atr 2.0) ===")
results = {}
for ep in ("sl_only", "sl_plus_tp_2r", "sl_plus_tp_3r", "sl_plus_trailing_atr",
           "sl_plus_trailing_swing", "sl_partial_close_1r_runner_trail"):
    results[ep] = run_and_report(ep, sig_eval, ep)

print("\n=== TIME-EXIT variants (sl_only + N-bar close, mirrors arc 1011) ===")
for nb in (2, 3, 5):
    results[f"te{nb}"] = run_and_report(f"sl_only_te{nb}", eval_with_time_exit(sig_eval, nb), "sl_only")

# Fair null on the best-mean config
best = max(results, key=lambda k: results[k][0])
print(f"\n=== FAIR SAME-SIDE NULL vs best config ({best}) ===")
null_eval = build_null_signal_evaluation(sig_eval, seed=42, warmup=100)
if best.startswith("te"):
    nb = int(best[2:])
    null_eval = eval_with_time_exit(null_eval, nb)
    null_mean, *_ = run_and_report(f"NULL_{best}", null_eval, "sl_only")
else:
    null_mean, *_ = run_and_report(f"NULL_{best}", null_eval, best)
print(f"\n>>> real({best}) mean={results[best][0]:+.3f}%  null mean={null_mean:+.3f}%  "
      f"excess={results[best][0]-null_mean:+.3f}pp")
