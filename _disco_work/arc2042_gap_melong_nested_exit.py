"""arc 2042 — §5f NESTED walk-forward EXIT/SL selection on gap (1006) + me_long (1011).

Completes the book's exit-honesty audit started by arc 2040 (fbr: ~40% optimism,
stays +) and arc 2041 (me_short: optimism FLIPS the sign to NEGATIVE). gap and
me_long were SKIPPED by 2040/2041 because their committed exits carry a TIME-EXIT
predicate (the reversion/fill mechanism plays out over a fixed horizon, not at
+1R) — so the clean 6-registry-exit grid does not map directly. This arc handles
that: the honest §5f exit menu for a time-exit reversion leg is the exit knobs
that were full-sample-chosen = {exit_policy} x {SL} AND {time horizon}.

Committed configs (scripts/cosim_validation/validate_4way_book.py):
  gap     : WeekendGapFillLongSignal(0.5, 36) H4 JPY + time-exit 24 + exit_policy=None, SL2.0
  me_long : MonthEndReversionLongSignal(1.0, 2) D1 USD + time-exit 2 + sl_only,        SL2.0

Two analyses per leg:
  (A) PRIMARY  — 6 registry exit_policy x SL{1.5,2.0,2.5} at the COMMITTED horizon
                 (apples-to-apples with 2040/2041's 18-cfg grid): was the
                 exit_policy/SL choice a full-sample best pick?
  (B) HORIZON  — committed exit_policy x SL{1.5,2.0,2.5} x time-horizon grid:
                 was the time horizon a full-sample best pick? (the binding knob
                 for these reversion legs)
Each analysis -> the BUILT nested_exit_selection walk-forward (3 metrics) + a
frozen pick scored ONCE on 2021+. Scoring 100% canonical (A1+MultiPairBacktester
+FundedNext). No OOS tuned.

Run:  PYTHONPATH=. py _disco_work/arc2042_gap_melong_nested_exit.py
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from core.arc.signal_protocol import SignalEvaluation
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.wfo.discovery_measure import build_oos_year_folds, run_config_over_folds
from core.wfo.folds import build_v3_folds
from discovery.tools.gap_signals import WeekendGapFillLongSignal
from discovery.tools.month_end_signals import MonthEndReversionLongSignal
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

# arc-1020 recorded per-year IS ROI (%) — committed-config anchors.
GAP_ANCHOR = {2011: -0.07, 2012: 8.23, 2013: -2.06, 2014: 2.94, 2015: -4.19,
              2016: 3.20, 2017: 0.53, 2018: -6.79, 2019: 7.45, 2020: -2.39}
MELONG_ANCHOR = {2011: 0.40, 2012: 0.29, 2013: 0.96, 2014: -0.23, 2015: -1.14,
                 2016: -0.51, 2017: 0.34, 2018: 0.90, 2019: 1.16, 2020: 0.15}

METRICS = [("mean_roi", metric_mean_roi), ("afp_then_mean", metric_afp_then_mean),
           ("worst_then_mean", metric_worst_then_mean)]


def _attach_time_exit(sig: SignalEvaluation, panel: Panel, n_bars: int) -> SignalEvaluation:
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n_bars)
    new_per_pair = {pair: dataclasses.replace(state, exit_predicate=pred)
                    for pair, state in sig.per_pair.items()}
    return dataclasses.replace(sig, per_pair=new_per_pair)


def _cfg(label: str, exit_policy, sl: float) -> A1Config:
    return A1Config(config_id=label, sl_atr_mult=sl, trail_enabled=False, exit_policy=exit_policy)


def _print_grid(scored, yr_of):
    print(f"  {'config':<44} {'meanIS%':>8} {'pos/10':>7} {'worst%':>8}")
    rows = []
    for label, stats in scored.items():
        rois = [fs.roi_pct * 100 for fs in stats]
        rows.append((label, sum(rois) / len(rois), sum(1 for r in rois if r > 0), min(rois)))
    for label, mean, npos, worst in sorted(rows, key=lambda r: -r[1]):
        print(f"  {label:<44} {mean:+8.2f} {npos:>5}/10 {worst:+8.2f}")
    best = max(rows, key=lambda r: r[1])
    print(f"  -> full-sample best-MEAN pick = {best[0]} ({best[1]:+.2f}%) [§5f-forbidden]")
    return best


def _nested(scored, yr_of, oos_runner_for, oos_folds):
    """Run the 3-metric nested WFO + frozen OOS scoring over an already-scored grid.

    oos_runner_for(label) -> (runner, A1Config) to score the frozen pick on 2021+.
    """
    frozen = {}
    for name, fn in METRICS:
        res = nested_walk_forward_select(scored, selection=fn, selection_name=name, min_prior_folds=2)
        frozen[name] = res.frozen_label
        ev = [c.roi_pct * 100 for c in res.per_fold if not c.is_warmup]
        print(f"\n  -- metric {name} --  frozen pick = {res.frozen_label}")
        print(f"     HONEST nested ({res.n_evaluable_folds} folds): mean {sum(ev)/len(ev):+.3f}%  "
              f"worst {res.worst_fold_roi*100:+.2f}%  neg {res.n_negative_folds}  AFP={res.all_folds_positive}")
    print("\n  FROZEN holdout (2021+), scored ONCE per metric:")
    for name in (m[0] for m in METRICS):
        runner, cfg = oos_runner_for(frozen[name])
        oos = run_config_over_folds(runner, oos_folds, cfg)
        rois = [fs.roi_pct * 100 for fs in oos]
        print(f"   [{name}] {frozen[name]}: " + " ".join(f"{fs.fold_id}:{r:+.2f}%" for fs, r in zip(oos, rois)))
        print(f"     OOS mean {sum(rois)/len(rois):+.3f}%  worst {min(rois):+.2f}%  "
              f"pos {sum(1 for r in rois if r>0)}/{len(rois)}")


def audit_leg(name, base_sig, panel, panels_arg, committed_exit, committed_h,
              horizon_grid, anchor, is_folds, oos_folds, yr_of):
    print("\n" + "#" * 80)
    print(f"# {name}: committed exit_policy={committed_exit!r}  horizon={committed_h}  SL2.0")
    print("#" * 80)

    # one runner per horizon (time-exit predicate lives on the signal eval)
    runners = {}
    for h in sorted(set(horizon_grid) | {committed_h}):
        ev = _attach_time_exit(base_sig, panel, h)
        runners[h] = ArcFoldRunner(A1Architecture(), ev, panels_arg)

    # ---- fidelity anchor: committed config ----
    anc_runner = runners[committed_h]
    anc_stats = run_config_over_folds(anc_runner, is_folds, _cfg(f"{name}_anchor", committed_exit, 2.0))
    print("\nFIDELITY ANCHOR (committed config) vs arc-1020 recorded:")
    maxdiff = 0.0
    anc_vals = {}
    for fs in anc_stats:
        yr = yr_of[fs.fold_id]; mine = fs.roi_pct * 100; rec = anchor[yr]; d = mine - rec
        anc_vals[yr] = mine; maxdiff = max(maxdiff, abs(d))
        print(f"  {yr} | mine {mine:+7.2f} | rec {rec:+7.2f} | diff {d:+6.2f}")
    print(f"  committed mean IS ROI = {sum(anc_vals.values())/len(anc_vals):+.3f}%  (max|diff| = {maxdiff:.3f}pp)")

    # ---- (A) PRIMARY: 6 exit_policy x 3 SL at committed horizon ----
    print(f"\n=== (A) PRIMARY exit_policy x SL grid @ committed horizon {committed_h} (the §5f exit-fishing view) ===")
    scoredA = {}
    for ex in EXITS:
        for sl in SLS:
            lbl = f"{ex}|sl{sl}"
            scoredA[lbl] = list(run_config_over_folds(runners[committed_h], is_folds, _cfg(lbl, ex, sl)))
    _print_grid(scoredA, yr_of)
    print("\n  §5f NESTED selection (A):")

    def oosA(label):
        ex, sltag = label.split("|"); sl = float(sltag[2:])
        return runners[committed_h], _cfg(label, ex, sl)
    _nested(scoredA, yr_of, oosA, oos_folds)

    # ---- (B) HORIZON: committed exit_policy x SL x horizon ----
    print(f"\n=== (B) HORIZON grid: exit_policy={committed_exit!r} x SL x time-horizon {horizon_grid} ===")
    scoredB = {}
    for h in horizon_grid:
        for sl in SLS:
            lbl = f"h{h}|sl{sl}"
            scoredB[lbl] = list(run_config_over_folds(runners[h], is_folds, _cfg(lbl, committed_exit, sl)))
    _print_grid(scoredB, yr_of)
    print("\n  §5f NESTED selection (B):")

    def oosB(label):
        htag, sltag = label.split("|"); h = int(htag[1:]); sl = float(sltag[2:])
        return runners[h], _cfg(label, committed_exit, sl)
    _nested(scoredB, yr_of, oosB, oos_folds)


def main() -> None:
    print(f"histdata_root={HISTDATA_ROOT}\ncache_root={CACHE_ROOT}\n")
    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    oos_folds = build_oos_year_folds(start_year=2021)
    yr_of = {f.fold_id: f.oos_start.year for f in is_folds}

    # ---- gap (H4 JPY crosses) ----
    h4 = Panel.from_pairs(JPY_CROSSES, "H4", histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
                          use_cache=True, boundary_convention=BOUNDARY)
    gap_sig = WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36).evaluate({"H4": h4})
    audit_leg("gap", gap_sig, h4, {"H4": h4}, committed_exit=None, committed_h=24,
              horizon_grid=[12, 18, 24, 36], anchor=GAP_ANCHOR,
              is_folds=is_folds, oos_folds=oos_folds, yr_of=yr_of)

    # ---- me_long (D1 USD majors) ----
    d1 = Panel.from_pairs(USD_MAJORS, "D1", histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
                          use_cache=True, boundary_convention=BOUNDARY)
    me_sig = MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1})
    audit_leg("me_long", me_sig, d1, {"D1": d1}, committed_exit="sl_only", committed_h=2,
              horizon_grid=[1, 2, 3, 5], anchor=MELONG_ANCHOR,
              is_folds=is_folds, oos_folds=oos_folds, yr_of=yr_of)


if __name__ == "__main__":
    main()
