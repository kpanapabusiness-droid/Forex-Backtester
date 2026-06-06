"""arc 2041 — §5f NESTED walk-forward EXIT/SL selection on me_short (the short leg).

Tests whether arc-2040's fbr finding (committed headline ~40% full-sample
exit-selection optimism) generalizes to me_short (1019). me_short's committed
exit IS a registry exit (`sl_partial_close_1r_runner_trail`, NO time predicate in
the book's validate_4way_book config), so the 6-exit registry grid maps cleanly
— same clean comparison as fbr. Scoring 100% canonical; selection via the BUILT
`nested_exit_selection`. SHORT signal (engine short-symmetric, PR#273).

Run:  PYTHONPATH=. py _disco_work/arc2041_meshort_nested_exit.py
"""

from __future__ import annotations

import os
from pathlib import Path

from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.wfo.discovery_measure import build_oos_year_folds, run_config_over_folds
from core.wfo.folds import build_v3_folds
from discovery.tools.month_end_signals import MonthEndReversionShortSignal
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

EXITS = ["sl_only", "sl_plus_tp_2r", "sl_plus_tp_3r", "sl_plus_trailing_atr",
         "sl_plus_trailing_swing", "sl_partial_close_1r_runner_trail"]
SLS = [1.5, 2.0, 2.5]

# arc-1020 recorded me_short per-year IS ROI (%) — committed-config anchor.
MESHORT_ANCHOR = {2011: 3.39, 2012: 1.69, 2013: -0.90, 2014: 0.98, 2015: 0.40,
                  2016: -0.91, 2017: -0.68, 2018: 0.86, 2019: 1.29, 2020: 0.71}


def main() -> None:
    print(f"histdata_root={HISTDATA_ROOT}\ncache_root={CACHE_ROOT}\n")
    d1 = Panel.from_pairs(USD_MAJORS, "D1", histdata_root=HISTDATA_ROOT,
                          cache_root=CACHE_ROOT, use_cache=True, boundary_convention=BOUNDARY)
    sig = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1})
    runner = ArcFoldRunner(A1Architecture(), sig, {"D1": d1})

    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    oos_folds = build_oos_year_folds(start_year=2021)
    yr_of = {f.fold_id: f.oos_start.year for f in is_folds}

    def cfg(exit_policy: str, sl: float) -> A1Config:
        return A1Config(config_id=f"meshort_{exit_policy}_sl{sl}", sl_atr_mult=sl,
                        trail_enabled=False, exit_policy=exit_policy)

    print("Scoring grid (6 exits x 3 SL) over IS folds 2011-2020 (canonical A1+MultiPairBacktester)...")
    scored_is: dict[str, list] = {}
    for ex in EXITS:
        for sl in SLS:
            scored_is[f"{ex}|sl{sl}"] = list(run_config_over_folds(runner, is_folds, cfg(ex, sl)))
        print(f"  scored {ex} @ SL{SLS}")

    # fidelity anchor: committed partial-runner / SL2.0
    anchor = {fs.fold_id: fs.roi_pct * 100 for fs in scored_is["sl_partial_close_1r_runner_trail|sl2.0"]}
    print("\nFIDELITY ANCHOR — partial_close_1r_runner_trail/SL2.0 vs arc-1020 recorded me_short:")
    maxdiff = 0.0
    for fs in scored_is["sl_partial_close_1r_runner_trail|sl2.0"]:
        yr = yr_of[fs.fold_id]; rec = MESHORT_ANCHOR[yr]; d = anchor[fs.fold_id] - rec
        maxdiff = max(maxdiff, abs(d))
        print(f"  {yr} | mine {anchor[fs.fold_id]:+6.2f} | rec {rec:+6.2f} | diff {d:+6.2f}")
    mean_committed = sum(anchor.values()) / len(anchor)
    print(f"  committed-config mean IS ROI = {mean_committed:+.3f}%  (max|diff vs rec| = {maxdiff:.3f}pp)")

    print("\nFULL-GRID summary (full-IS mean / folds-pos / worst) — the EXIT-FISHING view §5f forbids:")
    print(f"  {'config':<42} {'meanIS%':>8} {'pos/10':>7} {'worst%':>8}")
    grid_rows = []
    for label, stats in scored_is.items():
        rois = [fs.roi_pct * 100 for fs in stats]
        grid_rows.append((label, sum(rois) / len(rois), sum(1 for r in rois if r > 0), min(rois)))
    for label, mean, npos, worst in sorted(grid_rows, key=lambda r: -r[1]):
        print(f"  {label:<42} {mean:+8.2f} {npos:>5}/10 {worst:+8.2f}")
    best_fish = max(grid_rows, key=lambda r: r[1])
    print(f"  -> full-sample best-MEAN pick = {best_fish[0]} ({best_fish[1]:+.2f}%) [§5f-forbidden]")

    print("\n" + "=" * 78)
    print("§5f NESTED WALK-FORWARD EXIT/SL SELECTION (honest: per fold, choose on PRIOR folds only)")
    print("=" * 78)
    frozen_by_metric = {}
    for name, fn in [("mean_roi", metric_mean_roi), ("afp_then_mean", metric_afp_then_mean),
                     ("worst_then_mean", metric_worst_then_mean)]:
        res = nested_walk_forward_select(scored_is, selection=fn, selection_name=name, min_prior_folds=2)
        frozen_by_metric[name] = res.frozen_label
        print(f"\n-- metric {name} --  frozen-all-IS pick = {res.frozen_label}")
        ev = []
        for c in res.per_fold:
            tag = "WARMUP" if c.is_warmup else ""
            if not c.is_warmup:
                ev.append(c.roi_pct * 100)
            print(f"  {yr_of[c.fold_id]:>5} {c.selected_label:<40} {c.roi_pct*100:+8.2f} {tag:>6}")
        print(f"  HONEST nested ({res.n_evaluable_folds} folds): mean {sum(ev)/len(ev):+.3f}%  "
              f"worst {res.worst_fold_roi*100:+.2f}%  neg {res.n_negative_folds}  AFP={res.all_folds_positive}")

    print("\n" + "=" * 78)
    print("FROZEN holdout scoring (2021+) — applied ONCE per OOS year, NOT re-selected (§4)")
    print("=" * 78)
    for name in ("mean_roi", "afp_then_mean", "worst_then_mean"):
        frozen = frozen_by_metric[name]; ex, sltag = frozen.split("|"); sl = float(sltag[2:])
        oos_stats = run_config_over_folds(runner, oos_folds, cfg(ex, sl))
        rois = [fs.roi_pct * 100 for fs in oos_stats]
        print(f"\n  [{name}] frozen={frozen}: " +
              " ".join(f"{fs.fold_id}:{r:+.2f}%" for fs, r in zip(oos_stats, rois)))
        print(f"    OOS mean {sum(rois)/len(rois):+.3f}%  worst {min(rois):+.2f}%  "
              f"pos {sum(1 for r in rois if r>0)}/{len(rois)}  AFP={all(r>0 for r in rois)}")


if __name__ == "__main__":
    main()
