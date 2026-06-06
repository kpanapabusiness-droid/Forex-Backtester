"""arc 1042 — Honest-exit 4-way BOOK mean.

Completes the §5f nested-WFO exit-selection audit across ALL FOUR book components
(arc 2040 did fbr, 2041 did me_short; gap + me_long were untested), then recomputes
the book's deploy-relevant MEAN / worst-fold / AFP with EVERY component scored under
its HONEST nested-WFO-selected exit instead of its committed full-sample exit. The
committed +0.59% RP book mean is what operator path-A would deploy; this measures
whether it is inflated by component-level exit-fishing (2041 showed me_short's
committed +0.683% FLIPS NEGATIVE under honest selection).

Scoring 100% canonical (ArcFoldRunner -> A1 -> MultiPairBacktester, FundedNext).
Selection via BUILT `nested_exit_selection`; book combination via BUILT
`combine_fold_roi`. No gate reimplemented; OOS frozen-scored ONCE (never re-selected).

Run:  PYTHONPATH=. py _disco_work/arc1042_honest_exit_book.py
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
METRICS = [("mean_roi", metric_mean_roi), ("afp_then_mean", metric_afp_then_mean),
           ("worst_then_mean", metric_worst_then_mean)]

# arc-1020 recorded committed per-year IS ROI (%) — reproduction anchor.
ANCHOR = {
    2011: dict(gap=-0.07, me_long=+0.40, fbr=+7.55, me_short=+3.39),
    2012: dict(gap=+8.23, me_long=+0.29, fbr=+3.05, me_short=+1.69),
    2013: dict(gap=-2.06, me_long=+0.96, fbr=+0.91, me_short=-0.90),
    2014: dict(gap=+2.94, me_long=-0.23, fbr=+0.19, me_short=+0.98),
    2015: dict(gap=-4.19, me_long=-1.14, fbr=+3.17, me_short=+0.40),
    2016: dict(gap=+3.20, me_long=-0.51, fbr=+2.55, me_short=-0.91),
    2017: dict(gap=+0.53, me_long=+0.34, fbr=+1.23, me_short=-0.68),
    2018: dict(gap=-6.79, me_long=+0.90, fbr=-4.20, me_short=+0.86),
    2019: dict(gap=+7.45, me_long=+1.16, fbr=+0.05, me_short=+1.29),
    2020: dict(gap=-2.39, me_long=+0.15, fbr=+4.03, me_short=+0.71),
}


def _load(pairs, tf):
    return Panel.from_pairs(pairs, tf, histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
                            use_cache=True, boundary_convention=BOUNDARY)


def _attach_time_exit(sig: SignalEvaluation, panel: Panel, n_bars: int) -> SignalEvaluation:
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n_bars)
    return dataclasses.replace(sig, per_pair={
        p: dataclasses.replace(st, exit_predicate=pred) for p, st in sig.per_pair.items()})


def _score_grid(runner, folds, grid):
    """grid: dict label -> A1Config (signal already on runner). -> dict label -> [FoldStats]."""
    out = {}
    for label, cfg in grid.items():
        out[label] = list(run_config_over_folds(runner, folds, cfg))
    return out


def _anchor_report(name, scored, committed_label, yr_of):
    stats = scored[committed_label]
    rec = {y: ANCHOR[y][name] for y in ANCHOR}
    mx = 0.0
    for fs in stats:
        d = fs.roi_pct * 100 - rec[yr_of[fs.fold_id]]
        mx = max(mx, abs(d))
    mean = sum(fs.roi_pct * 100 for fs in stats) / len(stats)
    print(f"  {name:<9} committed={committed_label:<42} meanIS={mean:+.3f}%  "
          f"max|diff vs arc-1020|={mx:.3f}pp")
    return mean


def _grid_summary(name, scored):
    rows = []
    for label, stats in scored.items():
        rois = [fs.roi_pct * 100 for fs in stats]
        rows.append((label, sum(rois) / len(rois), sum(1 for r in rois if r > 0), min(rois)))
    best = max(rows, key=lambda r: r[1])
    print(f"  [{name}] full-sample best-MEAN (=§5f-forbidden fish) = {best[0]} "
          f"({best[1]:+.2f}%, {best[2]}/10, worst {best[3]:+.2f}%)")
    return rows


def main():
    print(f"histdata_root={HISTDATA_ROOT}\ncache_root={CACHE_ROOT}\n")
    h4_usd = _load(USD_MAJORS, "H4")
    h4_jpy = _load(JPY_CROSSES, "H4")
    d1 = _load(USD_MAJORS, "D1")

    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    oos_folds = build_oos_year_folds(start_year=2021)
    yr_of = {f.fold_id: f.oos_start.year for f in is_folds}

    # ── component signal evaluations (committed) ─────────────────────────
    gap_sig = WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36).evaluate({"H4": h4_jpy})
    me_long_base = MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1})
    fbr_sig = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": h4_usd})
    me_short_sig = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1})

    # me_long carries a 2-bar reversion-horizon time exit (mechanism, arc 2024) on EVERY config.
    me_long_sig = _attach_time_exit(me_long_base, d1, 2)

    runners = {
        "gap": ArcFoldRunner(A1Architecture(), gap_sig, {"H4": h4_jpy}),
        "me_long": ArcFoldRunner(A1Architecture(), me_long_sig, {"D1": d1}),
        "fbr": ArcFoldRunner(A1Architecture(), fbr_sig, {"H4": h4_usd}),
        "me_short": ArcFoldRunner(A1Architecture(), me_short_sig, {"D1": d1}),
    }

    # ── grids ────────────────────────────────────────────────────────────
    # gap: the chosen hyperparameter is the OVERSHOOT-harvest HORIZON (arc 1006/1007:
    #   the edge is overshoot, capping at target is worse) -> pure-time (exit_policy=None)
    #   x horizon x SL. Committed = horizon 24 / SL 2.0.
    gap_grid = {}
    for hz in GAP_HORIZONS:
        for sl in SLS:
            gap_grid[f"timeexit{hz}|sl{sl}"] = A1Config(
                config_id=f"gap_t{hz}_sl{sl}", sl_atr_mult=sl, trail_enabled=False, exit_policy=None)
    gap_committed = "timeexit24|sl2.0"
    # gap registry-exit side-check (at committed horizon 24): confirm registry SL-exits
    #   don't beat pure-time overshoot. NOT part of the nested grid (different family).
    gap_side = {f"{ex}|sl2.0": A1Config(config_id=f"gap_{ex}", sl_atr_mult=2.0,
                                        trail_enabled=False, exit_policy=ex) for ex in EXITS}

    # me_long / fbr / me_short: registry 6 exits x 3 SL (apples-to-apples w/ 2040/2041).
    def registry_grid(tag):
        return {f"{ex}|sl{sl}": A1Config(config_id=f"{tag}_{ex}_sl{sl}", sl_atr_mult=sl,
                                         trail_enabled=False, exit_policy=ex)
                for ex in EXITS for sl in SLS}

    me_long_grid = registry_grid("melong")
    fbr_grid = registry_grid("fbr")
    me_short_grid = registry_grid("meshort")

    committed_label = {
        "gap": gap_committed, "me_long": "sl_only|sl2.0",
        "fbr": "sl_plus_trailing_atr|sl2.0", "me_short": "sl_partial_close_1r_runner_trail|sl2.0"}

    # ── score every grid over IS folds (canonical) ───────────────────────
    print("Scoring grids over IS folds 2011-2020 (canonical A1 + MultiPairBacktester)...")
    scored = {}
    # gap requires the time-exit predicate re-attached per horizon (different n_bars)
    print("  gap (pure-time horizon x SL)...")
    scored["gap"] = {}
    for hz in GAP_HORIZONS:
        gsig = _attach_time_exit(gap_sig, h4_jpy, hz)
        grunner = ArcFoldRunner(A1Architecture(), gsig, {"H4": h4_jpy})
        for sl in SLS:
            lbl = f"timeexit{hz}|sl{sl}"
            scored["gap"][lbl] = list(run_config_over_folds(grunner, is_folds, gap_grid[lbl]))
    print("  gap registry side-check (horizon 24)...")
    gsig24 = _attach_time_exit(gap_sig, h4_jpy, 24)
    grunner24 = ArcFoldRunner(A1Architecture(), gsig24, {"H4": h4_jpy})
    gap_side_scored = _score_grid(grunner24, is_folds, gap_side)
    print("  me_long (registry x SL, 2-bar horizon fixed)...")
    scored["me_long"] = _score_grid(runners["me_long"], is_folds, me_long_grid)
    print("  fbr (registry x SL)...")
    scored["fbr"] = _score_grid(runners["fbr"], is_folds, fbr_grid)
    print("  me_short (registry x SL)...")
    scored["me_short"] = _score_grid(runners["me_short"], is_folds, me_short_grid)

    # ── anchor + grid summaries ──────────────────────────────────────────
    print("\n=== REPRODUCTION ANCHOR (committed configs vs arc-1020) ===")
    committed_mean = {}
    for name in ("gap", "me_long", "fbr", "me_short"):
        committed_mean[name] = _anchor_report(name, scored[name], committed_label[name], yr_of)
    print("\n=== FULL-SAMPLE best-mean (the exit-fishing view) ===")
    for name in ("gap", "me_long", "fbr", "me_short"):
        _grid_summary(name, scored[name])
    print("  [gap registry side-check @ horizon 24]:")
    for label, stats in sorted(gap_side_scored.items(),
                               key=lambda kv: -sum(fs.roi_pct for fs in kv[1])):
        rois = [fs.roi_pct * 100 for fs in stats]
        print(f"    {label:<42} meanIS {sum(rois)/len(rois):+7.3f}%  "
              f"{sum(1 for r in rois if r>0)}/10  worst {min(rois):+.2f}%")

    # ── nested selection per component (3 metrics) ───────────────────────
    # Store honest per-fold ROI series keyed by metric for the book combination.
    honest_series = {m: {} for m, _ in METRICS}   # metric -> name -> [roi_pct per fold]
    frozen_pick = {m: {} for m, _ in METRICS}
    print("\n" + "=" * 80)
    print("§5f NESTED WALK-FORWARD EXIT SELECTION (honest: per fold, choose on PRIOR folds)")
    print("=" * 80)
    for name in ("gap", "me_long", "fbr", "me_short"):
        print(f"\n##### {name} #####")
        for mname, mfn in METRICS:
            res = nested_walk_forward_select(scored[name], selection=mfn,
                                             selection_name=mname, min_prior_folds=2)
            # per-fold honest series, ordered by fold_id
            series = [c.roi_pct for c in sorted(res.per_fold, key=lambda c: c.fold_id)]
            honest_series[mname][name] = series
            frozen_pick[mname][name] = res.frozen_label
            ev_mean = sum(c.roi_pct for c in res.evaluable) / len(res.evaluable) * 100
            print(f"  [{mname:<15}] frozen={res.frozen_label:<40} "
                  f"honest meanIS(eval)={ev_mean:+.3f}%  worst={res.worst_fold_roi*100:+.2f}%  "
                  f"neg={res.n_negative_folds}  AFP={res.all_folds_positive}")

    # committed per-fold series (for the committed-book baseline)
    committed_series = {name: [fs.roi_pct for fs in
                               sorted(scored[name][committed_label[name]], key=lambda fs: fs.fold_id)]
                        for name in ("gap", "me_long", "fbr", "me_short")}
    order_years = [yr_of[fid] for fid in sorted(yr_of)]
    warmup_years = set(order_years[:2])   # first 2 folds = nested warmup

    # ── BOOK assembly: committed vs honest ───────────────────────────────
    def book_stats(series_by_name, weights):
        rois = combine_fold_rois([series_by_name[n] for n in
                                  ("gap", "me_long", "fbr", "me_short")], weights).combined_roi
        # verdict over evaluable (non-warmup) folds; mean over all for the deploy number
        eval_idx = [i for i, y in enumerate(order_years) if y not in warmup_years]
        eval_rois = [rois[i] for i in eval_idx]
        return dict(mean=sum(rois) / len(rois) * 100,
                    eval_mean=sum(eval_rois) / len(eval_rois) * 100,
                    worst=min(eval_rois) * 100, afp=all(r > 0 for r in eval_rois),
                    n_neg=sum(1 for r in eval_rois if r <= 0), per_fold=rois)

    print("\n" + "=" * 80)
    print("BOOK MEAN — committed exits  vs  HONEST nested exits (the deploy number)")
    print("=" * 80)
    w_committed_rp = fit_weights([committed_series[n] for n in
                                  ("gap", "me_long", "fbr", "me_short")], mode="risk_parity")
    w_equal = [0.25, 0.25, 0.25, 0.25]
    print("committed RP weights (frozen IS): " +
          ", ".join(f"{n}={w:.3f}" for n, w in zip(("gap", "me_long", "fbr", "me_short"), w_committed_rp)))

    cb_rp = book_stats(committed_series, w_committed_rp)
    cb_eq = book_stats(committed_series, w_equal)
    print(f"\nCOMMITTED book  RP : mean(all10)={cb_rp['mean']:+.3f}%  eval_mean={cb_rp['eval_mean']:+.3f}%  "
          f"worst={cb_rp['worst']:+.2f}%  AFP={cb_rp['afp']}  neg={cb_rp['n_neg']}")
    print(f"COMMITTED book  EQ : mean(all10)={cb_eq['mean']:+.3f}%  eval_mean={cb_eq['eval_mean']:+.3f}%  "
          f"worst={cb_eq['worst']:+.2f}%  AFP={cb_eq['afp']}  neg={cb_eq['n_neg']}")

    for mname, _ in METRICS:
        hs = honest_series[mname]
        # honest RP weights re-fit on the honest series (the honest book re-derives vols)
        w_h_rp = fit_weights([hs[n] for n in ("gap", "me_long", "fbr", "me_short")], mode="risk_parity")
        hb_rp = book_stats(hs, w_h_rp)
        hb_rp_cw = book_stats(hs, w_committed_rp)  # honest series, COMMITTED weights (isolate exit effect)
        hb_eq = book_stats(hs, w_equal)
        print(f"\n[honest {mname}]")
        print(f"  honest RP wts: " + ", ".join(f"{n}={w:.3f}" for n, w in
              zip(("gap", "me_long", "fbr", "me_short"), w_h_rp)))
        print(f"  HONEST book RP(honest wts) : mean={hb_rp['mean']:+.3f}%  eval_mean={hb_rp['eval_mean']:+.3f}%  "
              f"worst={hb_rp['worst']:+.2f}%  AFP={hb_rp['afp']}  neg={hb_rp['n_neg']}")
        print(f"  HONEST book RP(commit wts) : mean={hb_rp_cw['mean']:+.3f}%  eval_mean={hb_rp_cw['eval_mean']:+.3f}%  "
              f"worst={hb_rp_cw['worst']:+.2f}%  AFP={hb_rp_cw['afp']}  neg={hb_rp_cw['n_neg']}")
        print(f"  HONEST book EQUAL          : mean={hb_eq['mean']:+.3f}%  eval_mean={hb_eq['eval_mean']:+.3f}%  "
              f"worst={hb_eq['worst']:+.2f}%  AFP={hb_eq['afp']}  neg={hb_eq['n_neg']}")

    # ── OOS freeze: each component's frozen exit, scored ONCE, combined ───
    print("\n" + "=" * 80)
    print("FROZEN-EXIT OOS BOOK (2021+) — each component's nested-frozen exit, scored ONCE (§4)")
    print("=" * 80)
    for mname, _ in METRICS:
        oos_series = {}
        for name in ("gap", "me_long", "fbr", "me_short"):
            frozen = frozen_pick[mname][name]
            if name == "gap":
                hz = int(frozen.split("|")[0].replace("timeexit", "")); sl = float(frozen.split("|")[1][2:])
                gsig = _attach_time_exit(gap_sig, h4_jpy, hz)
                gr = ArcFoldRunner(A1Architecture(), gsig, {"H4": h4_jpy})
                cfg = A1Config(config_id=f"gap_oos_t{hz}_sl{sl}", sl_atr_mult=sl, trail_enabled=False, exit_policy=None)
                stats = run_config_over_folds(gr, oos_folds, cfg)
            else:
                ex, sltag = frozen.split("|"); sl = float(sltag[2:])
                cfg = A1Config(config_id=f"{name}_oos_{ex}_sl{sl}", sl_atr_mult=sl, trail_enabled=False, exit_policy=ex)
                stats = run_config_over_folds(runners[name], oos_folds, cfg)
            oos_series[name] = [fs.roi_pct for fs in stats]
        w_oos_rp = fit_weights([honest_series[mname][n] for n in
                                ("gap", "me_long", "fbr", "me_short")], mode="risk_parity")  # IS-frozen wts
        oos_rp = combine_fold_rois([oos_series[n] for n in ("gap", "me_long", "fbr", "me_short")], w_oos_rp).combined_roi
        print(f"  [honest {mname}] frozen exits: " +
              ", ".join(f"{n}={frozen_pick[mname][n]}" for n in ("gap", "me_long", "fbr", "me_short")))
        print(f"     OOS book RP per-fold: " + " ".join(f"{r*100:+.2f}" for r in oos_rp))
        print(f"     OOS book RP mean={sum(oos_rp)/len(oos_rp)*100:+.3f}%  worst={min(oos_rp)*100:+.2f}%  "
              f"AFP={all(r>0 for r in oos_rp)}")


if __name__ == "__main__":
    main()
