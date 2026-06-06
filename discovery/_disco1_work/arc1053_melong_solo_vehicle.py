"""arc 1053 — Deployment-vehicle feasibility of me_long-SOLO (the actual honest deploy object).

DIAGNOSTIC, OOS-PRESERVING. Reads geometry off ALREADY-SCORED canonical curves; never
tunes, never gates, never spends OOS (characterization only, arc-1042/1033/2045 precedent).
Uses BUILT tools only (cosim_book_fold canonical + equity_risk_profile + propfirm_feasibility);
no new canonical code, no engine reimplementation.

WHY: arc 1046 showed that under honest frozen §5f exits, the 4-way book collapses to me_long-SOLO
out-of-sample (fbr goes mean-negative OOS; adding any leg degrades it). me_long-solo (committed
= honest exit per 1042/2042: sl_only / 2-bar D1 / SL2.0) is the real honest deploy object. But the
deployment-vehicle profiles (arc 2033 prop-firm / arc 1033 Calmar-underwater / arc 2045 honest 2-way)
were all computed on the 4-way and 2-way BOOKS — never on me_long-SOLO. That standalone number is the
one the operator's path-A deploy call needs and it has never been computed. This fills exactly that gap.

me_long config: MonthEndReversionLongSignal(1.0, into_bars=2), D1 USD majors, sl_only + 2-bar time-exit,
SL 2.0 ATR — IDENTICAL to the committed arc-1011/1020 leg (anchors to the cosim item-E record).
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pandas as pd

from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.wfo.cosim_book import CoSimComponent, cosim_book_fold, mark_prices_from_panel
from core.wfo.folds import Fold, build_v3_folds
from core.wfo.discovery_measure import build_oos_year_folds
from discovery.tools.time_exit_predicate import make_time_exit_predicate
from discovery.tools.month_end_signals import MonthEndReversionLongSignal
from discovery.tools.equity_risk_profile import compute_risk_profile
from discovery.tools.propfirm_feasibility import (
    compute_sharpe, feasibility_horizon_years, _CHALLENGES,
)

HISTDATA_ROOT = Path(os.environ.get("COSIM_HISTDATA_ROOT", r"C:/Users/panap/histdata_backup"))
CACHE_ROOT = Path(os.environ.get("COSIM_CACHE_ROOT", "data/cache"))
BOUNDARY = "5ers_eet"
SB = 100_000.0
USD = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
_OFFSET = 100_000

# committed arc-1020 IS per-year ROI % for me_long (reproduction cross-check)
REC_IS = {2011: 0.40, 2012: 0.29, 2013: 0.96, 2014: -0.23, 2015: -1.14,
          2016: -0.51, 2017: 0.34, 2018: 0.90, 2019: 1.16, 2020: 0.15}


def _reid(trades, fold_idx):
    base = fold_idx * _OFFSET
    out = []
    for t in trades:
        out.append(dataclasses.replace(
            t, position_id=base + int(t.position_id),
            parent_position_id=(None if t.parent_position_id is None
                                else base + int(t.parent_position_id))))
    return out


def main():
    print("loading D1 USD-majors panel (warm cache)...", flush=True)
    d1 = Panel.from_pairs(USD, "D1", histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
                          use_cache=True, boundary_convention=BOUNDARY)
    pred = make_time_exit_predicate({p: d1.pair_dfs[p] for p in d1.pairs}, 2)
    sig = MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1})
    sig = dataclasses.replace(sig, per_pair={
        p: dataclasses.replace(s, exit_predicate=pred) for p, s in sig.per_pair.items()})
    cfg = A1Config(config_id="arc_1011", sl_atr_mult=2.0, trail_enabled=False, exit_policy="sl_only")
    runner = ArcFoldRunner(A1Architecture(), sig, {"D1": d1})
    marks = mark_prices_from_panel(d1)

    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    oos_folds = list(build_oos_year_folds(2021))
    all_folds = is_folds + oos_folds

    print("scoring me_long per-year (canonical A1 + MultiPairBacktester)...", flush=True)
    trades, roi = [], {}
    for fi, fold in enumerate(all_folds):
        fs = runner(fold, cfg)
        roi[fold.oos_start.year] = fs.roi_pct * 100.0
        trades.extend(_reid(runner.last_result.run_result.closed_trades, fi))
    trades = tuple(trades)

    print("\nreproduction cross-check (IS per-year ROI % vs arc-1020 record):")
    dev = 0.0
    for yr in range(2011, 2021):
        dev = max(dev, abs(roi[yr] - REC_IS[yr]))
        print(f"  {yr}: mine {roi[yr]:+.2f}  rec {REC_IS[yr]:+.2f}")
    print(f"  max|mine-rec| = {dev:.3f} pp")
    print("\nOOS per-year ROI % (2021+, characterization — NOT tuned):")
    for yr in sorted(y for y in roi if y >= 2021):
        print(f"  {yr}: {roi[yr]:+.2f}")
    is_rois = [roi[y] for y in range(2011, 2021)]
    oos_rois = [roi[y] for y in sorted(y for y in roi if y >= 2021)]
    print(f"\n  IS  mean {sum(is_rois)/len(is_rois):+.3f}%  ({sum(r>0 for r in is_rois)}/{len(is_rois)} pos)")
    print(f"  OOS mean {sum(oos_rois)/len(oos_rois):+.3f}%  ({sum(r>0 for r in oos_rois)}/{len(oos_rois)} pos)")

    # contiguous single-component curve via canonical cosim
    decade = Fold(fold_id=99, is_start=all_folds[0].is_start, is_end=all_folds[0].is_end,
                  oos_start=all_folds[0].oos_start, oos_end=all_folds[-1].oos_end)
    cc = [CoSimComponent(name="me_long", closed_trades=trades, mark_prices=marks, starting_balance=SB)]
    book = cosim_book_fold(cc, decade, weights=[1.0], starting_balance=SB, apply_exposure_cap=True)
    eq = book.net_equity

    def profile(label, lo, hi):
        prof = compute_risk_profile(eq, lo=lo, hi=hi)
        shp = compute_sharpe(eq, lo=lo, hi=hi)
        print(f"\n==== {label} ====")
        print(f"  ann_return {prof.ann_return_pct:+.3f}%  maxDD {prof.max_dd_pct:.3f}%  "
              f"Calmar {prof.calmar:.3f}  longest_underwater {prof.longest_underwater_days}d "
              f"({prof.time_underwater_frac*100:.0f}%)")
        print(f"  Sharpe {shp.daily_sharpe_ann:.3f}  Sortino {shp.daily_sortino_ann:.3f}  "
              f"vol {shp.daily_vol_ann_pct:.3f}%/yr")
        return prof

    ts = lambda s: pd.Timestamp(s, tz="UTC")
    p_is = profile("IS 2011-2020", ts("2011-01-01"), ts("2020-12-31"))
    p_oos = profile("OOS 2021-2026 (characterization)", ts("2021-01-01"), None)
    p_full = profile("FULL 2011-2026 contiguous", ts("2011-01-01"), None)

    print("\n==== prop-firm-challenge feasibility (T_min years, at the DD limit) ====")
    for label, P, _dd, D in _CHALLENGES:
        t_is = feasibility_horizon_years(p_is.calmar, P, D)
        t_full = feasibility_horizon_years(p_full.calmar, P, D)
        print(f"  {label}: P={P}/D={D}  T_min(IS Calmar {p_is.calmar:.2f})={t_is:.2f}yr  "
              f"T_min(full {p_full.calmar:.2f})={t_full:.2f}yr")


if __name__ == "__main__":
    main()
