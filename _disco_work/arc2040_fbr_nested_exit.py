"""arc 2040 — §5f NESTED WALK-FORWARD EXIT/SL SELECTION on fbr (the load-bearing leg).

fbr (arc 1013) committed `sl_plus_trailing_atr`/SL2.0 chosen as the single
full-IS-best-MEAN exit — exactly the full-sample best-pick §5f forbids. This
driver re-derives fbr's exit HONESTLY: score the 6 registry exits × SL{1.5,2.0,
2.5} over the IS folds via the canonical runner, then NESTED-WFO select (per
fold, choose the config best over strictly-earlier folds; score that fold), and
freeze the all-IS pick onto the 2021+ holdout once. No OOS in selection.

Scoring is 100% canonical (A1Architecture -> MultiPairBacktester, FundedNext
netted). Selection is the BUILT `nested_exit_selection` arithmetic. Geometry/
selection only; realizes no P&L.

Run:  PYTHONPATH=. py _disco_work/arc2040_fbr_nested_exit.py
"""

from __future__ import annotations

import os
from pathlib import Path

from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.wfo.discovery_measure import build_oos_year_folds, run_config_over_folds
from core.wfo.folds import build_v3_folds
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.nested_exit_selection import (
    metric_afp_then_mean,
    metric_mean_roi,
    metric_worst_then_mean,
    nested_walk_forward_select,
)

HISTDATA_ROOT = Path(os.environ.get("COSIM_HISTDATA_ROOT", r"C:/Users/panap/histdata_backup"))
CACHE_ROOT = Path(os.environ.get("COSIM_CACHE_ROOT", r"C:/Users/panap/Documents/Forex-Backtester/data/cache"))
BOUNDARY = "5ers_eet"
USD_MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]

EXITS = [
    "sl_only",
    "sl_plus_tp_2r",
    "sl_plus_tp_3r",
    "sl_plus_trailing_atr",
    "sl_plus_trailing_swing",
    "sl_partial_close_1r_runner_trail",
]
SLS = [1.5, 2.0, 2.5]

# arc-1020 recorded fbr per-year IS ROI (%) — the committed-config anchor.
FBR_ANCHOR = {2011: 7.55, 2012: 3.05, 2013: 0.91, 2014: 0.19, 2015: 3.17,
              2016: 2.55, 2017: 1.23, 2018: -4.20, 2019: 0.05, 2020: 4.03}


def main() -> None:
    print(f"histdata_root={HISTDATA_ROOT}\ncache_root={CACHE_ROOT}\n")
    h4 = Panel.from_pairs(USD_MAJORS, "H4", histdata_root=HISTDATA_ROOT,
                          cache_root=CACHE_ROOT, use_cache=True, boundary_convention=BOUNDARY)
    sig = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": h4})
    runner = ArcFoldRunner(A1Architecture(), sig, {"H4": h4})

    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]   # 2011-2020
    oos_folds = build_oos_year_folds(start_year=2021)

    def cfg(exit_policy: str, sl: float) -> A1Config:
        return A1Config(config_id=f"fbr_{exit_policy}_sl{sl}", sl_atr_mult=sl,
                        trail_enabled=False, exit_policy=exit_policy)

    # ── score the full grid over IS folds (canonical) ────────────────────
    print("Scoring grid (6 exits x 3 SL) over IS folds 2011-2020 (canonical A1+MultiPairBacktester)...")
    scored_is: dict[str, list] = {}
    for ex in EXITS:
        for sl in SLS:
            label = f"{ex}|sl{sl}"
            scored_is[label] = list(run_config_over_folds(runner, is_folds, cfg(ex, sl)))
        print(f"  scored {ex} @ SL{SLS}")

    # ── fidelity anchor: committed config (sl_plus_trailing_atr / SL2.0) ──
    anchor = {fs.fold_id: fs.roi_pct * 100 for fs in scored_is["sl_plus_trailing_atr|sl2.0"]}
    print("\nFIDELITY ANCHOR — sl_plus_trailing_atr/SL2.0 (trail_off) vs arc-1020 recorded fbr:")
    print("  year |   mine |    rec |   diff")
    yr_of = {f.fold_id: f.oos_start.year for f in is_folds}
    maxdiff = 0.0
    for fs in scored_is["sl_plus_trailing_atr|sl2.0"]:
        yr = yr_of[fs.fold_id]
        rec = FBR_ANCHOR[yr]
        d = anchor[fs.fold_id] - rec
        maxdiff = max(maxdiff, abs(d))
        print(f"  {yr} | {anchor[fs.fold_id]:+6.2f} | {rec:+6.2f} | {d:+6.2f}")
    mean_committed = sum(anchor.values()) / len(anchor)
    print(f"  committed-config mean IS ROI = {mean_committed:+.3f}%  (max|diff vs rec| = {maxdiff:.3f}pp)")

    # ── full grid summary (full-sample mean + folds-positive — the FISHING view) ──
    print("\nFULL-GRID summary (full-IS mean / folds-pos / worst) — the EXIT-FISHING view §5f forbids:")
    print(f"  {'config':<42} {'meanIS%':>8} {'pos/10':>7} {'worst%':>8}")
    grid_rows = []
    for label, stats in scored_is.items():
        rois = [fs.roi_pct * 100 for fs in stats]
        mean = sum(rois) / len(rois)
        npos = sum(1 for r in rois if r > 0)
        grid_rows.append((label, mean, npos, min(rois)))
    for label, mean, npos, worst in sorted(grid_rows, key=lambda r: -r[1]):
        print(f"  {label:<42} {mean:+8.2f} {npos:>5}/10 {worst:+8.2f}")
    best_fish = max(grid_rows, key=lambda r: r[1])
    print(f"  -> full-sample best-MEAN pick = {best_fish[0]} ({best_fish[1]:+.2f}%) [the §5f-forbidden number]")

    # ── §5f NESTED walk-forward selection (3 selection metrics) ──────────
    print("\n" + "=" * 78)
    print("§5f NESTED WALK-FORWARD EXIT/SL SELECTION (honest: per fold, choose on PRIOR folds only)")
    print("=" * 78)
    metrics = [
        ("mean_roi", metric_mean_roi),
        ("afp_then_mean", metric_afp_then_mean),
        ("worst_then_mean", metric_worst_then_mean),
    ]
    frozen_by_metric = {}
    for name, fn in metrics:
        res = nested_walk_forward_select(scored_is, selection=fn, selection_name=name, min_prior_folds=2)
        frozen_by_metric[name] = res.frozen_label
        print(f"\n-- selection metric: {name} --   frozen-all-IS pick = {res.frozen_label}")
        print(f"  {'year':>5} {'selected (chosen on prior folds)':<40} {'ROI%':>8} {'nprior':>6} {'warmup':>6}")
        ev_rois = []
        for c in res.per_fold:
            yr = yr_of[c.fold_id]
            tag = "WARMUP" if c.is_warmup else ""
            if not c.is_warmup:
                ev_rois.append(c.roi_pct * 100)
            print(f"  {yr:>5} {c.selected_label:<40} {c.roi_pct*100:+8.2f} {c.n_prior_folds:>6} {tag:>6}")
        mean_ev = sum(ev_rois) / len(ev_rois) if ev_rois else float("nan")
        print(f"  HONEST nested series (evaluable {res.n_evaluable_folds} folds): "
              f"mean {mean_ev:+.3f}%  worst {res.worst_fold_roi*100:+.2f}%  "
              f"neg {res.n_negative_folds}  all_folds_positive={res.all_folds_positive}")

    # ── freeze the (mean-metric) pick onto the OOS holdout ONCE ──────────
    print("\n" + "=" * 78)
    print("FROZEN holdout scoring (2021+) — frozen pick applied ONCE per OOS year, NOT re-selected (§4)")
    print("=" * 78)
    for name in ("mean_roi", "afp_then_mean", "worst_then_mean"):
        frozen = frozen_by_metric[name]
        ex, sltag = frozen.split("|")
        sl = float(sltag[2:])
        oos_stats = run_config_over_folds(runner, oos_folds, cfg(ex, sl))
        rois = [fs.roi_pct * 100 for fs in oos_stats]
        npos = sum(1 for r in rois if r > 0)
        print(f"\n  [{name}] frozen={frozen}")
        for fs, r in zip(oos_stats, rois):
            print(f"    OOS {fs.fold_id}: {r:+.2f}% (n={fs.n_trades})")
        print(f"    OOS mean {sum(rois)/len(rois):+.3f}%  worst {min(rois):+.2f}%  "
              f"pos {npos}/{len(rois)}  all_folds_positive={all(r>0 for r in rois)}")


if __name__ == "__main__":
    main()
