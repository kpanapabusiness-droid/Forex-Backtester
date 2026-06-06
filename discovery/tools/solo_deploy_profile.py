"""Standalone deployment-vehicle geometry of a SINGLE committed book component
(EXPERIMENT / analysis).

GEOMETRY ONLY on an ALREADY-COMPUTED contiguous co-sim equity curve — it never
realizes P&L, never scores a trade, never touches the gate, never spends OOS. It is
the 1-leg companion to arc 1033's `equity_risk_profile._build_4way_contiguous`
(4-way) and arc 2045's 2-way profile: arc 1046 identified `me_long`-SOLO as the
honest OOS-survivor deploy object, but the deployment GEOMETRY axis (max-DD / Calmar /
time-underwater / prop-firm T_min / daily-cap headroom) was only ever computed for the
4-way (1033/2033) and 2-way (2045) books — never the solo object the operator would
actually deploy.

This builds ONE component's contiguous 2011-2020 IS equity curve via the SAME
canonical machinery as the 4-way driver (per-year-scored through A1 +
MultiPairBacktester, re-id'd, concatenated, superimposed by the canonical
`cosim_book_fold` with a single CoSimComponent at weight 1.0) and profiles it with
the BUILT `compute_risk_profile` + `compute_sharpe` + `feasibility_horizon_years`.

Default component = `me_long` at its committed/robust deploy exit (`sl_only` + 2-bar
time-exit, D1 USD majors — the arc 1042/1046 exit). The component spec is a small
table below; pass a name to profile a different one.

IS-only; OOS (2021+) NEVER touched.

Reproduce:  PYTHONPATH=. py discovery/tools/solo_deploy_profile.py [me_long|fbr|gap|me_short]
Created by: arc 2053.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import pandas as pd

from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.wfo.cosim_book import CoSimComponent, cosim_book_fold, mark_prices_from_panel
from core.wfo.folds import Fold, build_v3_folds
from discovery.tools.equity_risk_profile import compute_risk_profile
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.gap_signals import WeekendGapFillLongSignal
from discovery.tools.month_end_signals import (
    MonthEndReversionLongSignal,
    MonthEndReversionShortSignal,
)
from discovery.tools.propfirm_feasibility import (
    compute_sharpe,
    feasibility_horizon_years,
)
from discovery.tools.time_exit_predicate import make_time_exit_predicate

_PER_YEAR_OFFSET = 100_000  # << cosim's 10M component offset; > intra-year pid range
SB = 100_000.0
JPY = ["EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"]
USD = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]

# Representative published prop-firm challenge structures (same as arc 2033).
_CHALLENGES = [
    ("FundedNext Stellar 2-step P1", 8.0, 10.0),
    ("FundedNext Stellar 1-step   ", 10.0, 6.0),
    ("5ers Hyper-growth (rep.)    ", 8.0, 5.0),
]


def _reid(trades, fold_idx):
    base = fold_idx * _PER_YEAR_OFFSET
    out = []
    for t in trades:
        out.append(dataclasses.replace(
            t,
            position_id=base + int(t.position_id),
            parent_position_id=(None if t.parent_position_id is None
                                else base + int(t.parent_position_id)),
        ))
    return out


def _load(pairs, tf, histdata_root, cache_root, boundary):
    return Panel.from_pairs(pairs, tf, histdata_root=histdata_root, cache_root=cache_root,
                            use_cache=True, boundary_convention=boundary)


def _attach_te(sig, panel, n):
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n)
    return dataclasses.replace(sig, per_pair={
        p: dataclasses.replace(s, exit_predicate=pred) for p, s in sig.per_pair.items()})


def build_component(name, histdata_root, cache_root, boundary):
    """Return (signal_eval, panels_dict, A1Config, mark_panel) for one committed component.

    Configs are IDENTICAL to scripts/cosim_validation/validate_4way_book.py /
    equity_risk_profile._build_4way_contiguous (the committed item-E reproduction).
    """
    if name in ("gap",):
        h4_jpy = _load(JPY, "H4", histdata_root, cache_root, boundary)
        sig = _attach_te(WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36).evaluate({"H4": h4_jpy}), h4_jpy, 24)
        return sig, {"H4": h4_jpy}, A1Config(config_id="arc_1006", sl_atr_mult=2.0, trail_enabled=False, exit_policy=None), h4_jpy
    if name == "me_long":
        d1 = _load(USD, "D1", histdata_root, cache_root, boundary)
        sig = _attach_te(MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1}), d1, 2)
        return sig, {"D1": d1}, A1Config(config_id="arc_1011", sl_atr_mult=2.0, trail_enabled=False, exit_policy="sl_only"), d1
    if name == "fbr":
        h4_usd = _load(USD, "H4", histdata_root, cache_root, boundary)
        sig = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": h4_usd})
        # committed exit (sl_plus_trailing_atr, trail_enabled=True double-trail) — the 1013 committed config.
        return sig, {"H4": h4_usd}, A1Config(config_id="arc_1013", sl_atr_mult=2.0, trail_enabled=True, exit_policy="sl_plus_trailing_atr"), h4_usd
    if name == "me_short":
        d1 = _load(USD, "D1", histdata_root, cache_root, boundary)
        sig = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1})
        return sig, {"D1": d1}, A1Config(config_id="arc_1019", sl_atr_mult=2.0, trail_enabled=False, exit_policy="sl_partial_close_1r_runner_trail"), d1
    raise ValueError(f"unknown component {name!r}")


def build_solo_contiguous(name, histdata_root, cache_root, boundary):
    """One component's contiguous 2011-2020 IS curve via canonical cosim (weight=1.0)."""
    sig, panels, cfg, mark_panel = build_component(name, histdata_root, cache_root, boundary)
    marks = mark_prices_from_panel(mark_panel)
    runner = ArcFoldRunner(A1Architecture(), sig, panels)

    folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    decade = Fold(fold_id=99, is_start=folds[0].is_start, is_end=folds[0].is_end,
                  oos_start=folds[0].oos_start, oos_end=folds[-1].oos_end)

    decade_trades = []
    per_year_roi = {}
    for fi, fold in enumerate(folds):
        fs = runner(fold, cfg)
        per_year_roi[fold.oos_start.year] = fs.roi_pct * 100.0
        decade_trades.extend(_reid(runner.last_result.run_result.closed_trades, fi))
    decade_trades = tuple(decade_trades)

    def cosim(cap):
        cc = [CoSimComponent(name=name, closed_trades=decade_trades, mark_prices=marks, starting_balance=SB)]
        return cosim_book_fold(cc, decade, weights=[1.0], starting_balance=SB, apply_exposure_cap=cap)

    return per_year_roi, cosim


def main():  # pragma: no cover - analysis driver
    name = sys.argv[1] if len(sys.argv) > 1 else "me_long"
    histdata_root = Path(os.environ.get("COSIM_HISTDATA_ROOT", r"C:/Users/panap/histdata_backup"))
    cache_root = Path(os.environ.get("COSIM_CACHE_ROOT", r"C:/Users/panap/Documents/Forex-Backtester/data/cache"))
    boundary = "5ers_eet"
    lo = pd.Timestamp("2011-01-01", tz="UTC")
    hi = pd.Timestamp("2020-12-31 23:59:59", tz="UTC")

    print(f"building {name}-SOLO contiguous 2011-2020 IS curve (canonical A1 + MultiPairBacktester)...")
    per_year_roi, cosim = build_solo_contiguous(name, histdata_root, cache_root, boundary)

    print(f"\nper-year ROI % ({name}-solo, committed exit):")
    for yr in range(2011, 2021):
        print(f"  {yr}: {per_year_roi[yr]:+.3f}%")
    rois = [per_year_roi[yr] for yr in range(2011, 2021)]
    import statistics
    print(f"  mean {statistics.mean(rois):+.3f}% | sd {statistics.pstdev(rois):.3f}% | "
          f"neg-folds {sum(1 for r in rois if r < 0)}/10 | worst {min(rois):+.3f}%")

    print("\n" + "=" * 78)
    print(f"{name.upper()}-SOLO -- CONTIGUOUS 2011-2020 IS DEPLOYMENT-VEHICLE GEOMETRY")
    print("(1-leg companion to arc 1033 4-way / arc 2045 2-way)")
    print("=" * 78)
    for cap_label, cap in [("cap-OFF (monotone bound)", False), ("cap-ON  (faithful)      ", True)]:
        book = cosim(cap)
        pdd = book.per_day_max_dd_eet
        worst_day = float(pdd["day_max_dd_base_pct"].max() * 100.0) if len(pdd) else float("nan")
        prof = compute_risk_profile(book.net_equity, lo=lo, hi=hi, worst_day_dd_pct=worst_day)
        shp = compute_sharpe(book.net_equity, lo=lo, hi=hi)
        print(f"\n[{cap_label}]  n_dropped={book.n_dropped}  daily-5%-cap breached: {book.daily_dd_breached}")
        print("  " + prof.as_row())
        print("  " + shp.as_row())
        print(f"  deepest DD: peak {prof.peak_before_trough.date()} -> trough {prof.trough_date.date()}")
        print("  prop-firm T_min (=(target/maxDD)/Calmar yr, at DD limit; safe ~2x):")
        for clabel, P, D in _CHALLENGES:
            T = feasibility_horizon_years(prof.calmar, P, D)
            print(f"    {clabel}  [P {P:.0f}% / maxDD {D:.0f}%, P/D={P/D:.2f}]  -> T_min {T:6.1f} yr")


if __name__ == "__main__":  # pragma: no cover
    main()
