"""4-way BOOK deployment-vehicle geometry across IS *and* OOS (EXPERIMENT / analysis).

GEOMETRY ONLY on an ALREADY-COMPUTED contiguous co-sim equity curve — it never
realizes P&L, never scores a trade, never touches the gate, never *selects* on OOS.
It is the BOOK (4-leg) companion to arc 2053/2055's `solo_deploy_profile.py`
(1-leg, me_long, IS+OOS) and arc 1033's `equity_risk_profile._build_4way_contiguous`
(4-leg, IS only).

WHY (arc 2056). The deploy-object vehicle matrix had one missing cell. arc 1033/2054
profiled the 4-way book's IS vehicle geometry (maxDD 1.59% / Calmar 0.36/0.24); arc
2055 profiled me_long-SOLO's OOS vehicle geometry (kinder than IS — Calmar 0.31 vs
0.06, because me_long's IS-binding 2014-16 strong-USD block did not recur on the
holdout); arc 1042 spent the frozen-exit OOS book *mean* (+0.2-0.3%/yr) but NEVER the
OOS book's per-year series + contiguous vehicle geometry (maxDD / Calmar / underwater /
prop-firm T_min). This fills that cell: does the FULL book's OOS vehicle also beat its
IS (like me_long-solo, 2055), or does the OOS collapse of fbr/gap/me_short (1046) drag
the book's OOS vehicle below its IS? Directly informs the operator's path-A
"deploy the book vs deploy me_long-solo" call.

§4 COMPLIANCE. Weights are RISK-PARITY *frozen on IS* (`fit_weights(...).FREEZE these
for OOS`) and applied UNCHANGED to the OOS curve — nothing is fit to the holdout. The
exits are the committed/frozen per-component configs (identical to
`validate_4way_book.py` / `solo_deploy_profile.build_component`). The per-component
frozen OOS per-year series were already spent in arc 1042 (the OOS book mean); reading
vehicle geometry off the already-spent frozen series adds NO selection (arc 2055
precedent). The OOS curve is measure-once CHARACTERIZATION, never a tuning target.

Reproduce:  PYTHONPATH=. py discovery/tools/book_deploy_profile.py
Created by: arc 2056.
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

import pandas as pd

from core.runners.arc_fold_runner import ArcFoldRunner
from core.architectures.a1_system_level_filter import A1Architecture
from core.wfo.cosim_book import CoSimComponent, cosim_book_fold, mark_prices_from_panel
from core.wfo.discovery_measure import build_oos_year_folds
from core.wfo.folds import Fold, build_v3_folds
from discovery.tools.combine_fold_roi import fit_weights
from discovery.tools.equity_risk_profile import compute_risk_profile
from discovery.tools.propfirm_feasibility import compute_sharpe, feasibility_horizon_years
from discovery.tools.solo_deploy_profile import (
    _CHALLENGES,
    _PER_YEAR_OFFSET,
    SB,
    _reid,
    build_component,
)

NAMES = ["gap", "me_long", "fbr", "me_short"]


def _folds(window):
    if window == "oos":
        return list(build_oos_year_folds(start_year=2021))
    return [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]


def build_book(window, histdata_root, cache_root, boundary):
    """Per-component contiguous trade book + per-year ROIs for one window.

    Returns (per_year_roi[name][year], decade_trades[name], marks[name], decade Fold).
    The committed/frozen exit config is IDENTICAL in both windows (build_component);
    only the fold window changes.
    """
    folds = _folds(window)
    decade = Fold(fold_id=99, is_start=folds[0].is_start, is_end=folds[0].is_end,
                  oos_start=folds[0].oos_start, oos_end=folds[-1].oos_end)

    per_year_roi = {n: {} for n in NAMES}
    decade_trades = {n: [] for n in NAMES}
    marks = {}
    for n in NAMES:
        sig, panels, cfg, mark_panel = build_component(n, histdata_root, cache_root, boundary)
        marks[n] = mark_prices_from_panel(mark_panel)
        runner = ArcFoldRunner(A1Architecture(), sig, panels)
        for fi, fold in enumerate(folds):
            fs = runner(fold, cfg)
            per_year_roi[n][fold.oos_start.year] = fs.roi_pct * 100.0
            decade_trades[n].extend(_reid(runner.last_result.run_result.closed_trades, fi))
        decade_trades[n] = tuple(decade_trades[n])
    return per_year_roi, decade_trades, marks, decade


def _cosim(weights, cap, decade_trades, marks, decade):
    cc = [CoSimComponent(name=n, closed_trades=decade_trades[n],
                         mark_prices=marks[n], starting_balance=SB) for n in NAMES]
    return cosim_book_fold(cc, decade, weights=weights, starting_balance=SB, apply_exposure_cap=cap)


def _profile_window(label, per_year_roi, decade_trades, marks, decade, weights_named, lo, hi):
    years = sorted(next(iter(per_year_roi.values())))
    print(f"\nper-year book-component ROI % ({label}):")
    for yr in years:
        row = " ".join(f"{n}={per_year_roi[n][yr]:+.2f}" for n in NAMES)
        print(f"  {yr}: {row}")

    print("\n" + "=" * 80)
    print(f"4-WAY BOOK -- CONTIGUOUS {years[0]}-{years[-1]} {label} DEPLOYMENT-VEHICLE GEOMETRY")
    print("=" * 80)
    for wlabel, w in weights_named:
        # the book's own per-year ROI under these weights (for mean/neg/worst)
        book_yr = [sum(w[i] * per_year_roi[NAMES[i]][yr] for i in range(4)) for yr in years]
        print(f"\n--- weights [{wlabel}]: " + ", ".join(f"{n}={w[i]:.3f}" for i, n in enumerate(NAMES)))
        print(f"  book per-year ROI: mean {statistics.mean(book_yr):+.3f}% | "
              f"sd {statistics.pstdev(book_yr):.3f}% | neg {sum(1 for r in book_yr if r < 0)}/{len(book_yr)} | "
              f"worst {min(book_yr):+.3f}%")
        for cap_label, cap in [("cap-OFF (monotone bound)", False), ("cap-ON  (faithful)      ", True)]:
            book = _cosim(w, cap, decade_trades, marks, decade)
            pdd = book.per_day_max_dd_eet
            worst_day = float(pdd["day_max_dd_base_pct"].max() * 100.0) if len(pdd) else float("nan")
            prof = compute_risk_profile(book.net_equity, lo=lo, hi=hi, worst_day_dd_pct=worst_day)
            shp = compute_sharpe(book.net_equity, lo=lo, hi=hi)
            print(f"  [{cap_label}] n_dropped={book.n_dropped} daily-5%-cap breached: {book.daily_dd_breached}")
            print("    " + prof.as_row())
            print("    " + shp.as_row())
            print(f"    deepest DD: peak {prof.peak_before_trough.date()} -> trough {prof.trough_date.date()}")
            print("    prop-firm T_min (=(target/maxDD)/Calmar yr, at DD limit; safe ~2x):")
            for clabel, P, D in _CHALLENGES:
                T = feasibility_horizon_years(prof.calmar, P, D)
                print(f"      {clabel} [P {P:.0f}% / maxDD {D:.0f}%] -> T_min {T:6.1f} yr")


def main():  # pragma: no cover - analysis driver
    histdata_root = Path(os.environ.get("COSIM_HISTDATA_ROOT", r"C:/Users/panap/histdata_backup"))
    cache_root = Path(os.environ.get("COSIM_CACHE_ROOT", r"C:/Users/panap/Documents/Forex-Backtester/data/cache"))
    boundary = "5ers_eet"

    # ---- IS window: build, derive FROZEN risk-parity weights ----
    print("building IS (2011-2020) book (canonical A1 + MultiPairBacktester)...")
    is_roi, is_trades, is_marks, is_decade = build_book("is", histdata_root, cache_root, boundary)
    comp_is_rois = [[is_roi[n][yr] / 100.0 for yr in range(2011, 2021)] for n in NAMES]
    w_rp = fit_weights(comp_is_rois, mode="risk_parity")   # FROZEN — applied unchanged to OOS
    w_eq = [0.25] * 4
    print("\nFROZEN IS risk-parity weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(NAMES, w_rp)))
    weights_named = [("equal", w_eq), ("risk-parity(frozen-IS)", w_rp)]

    is_lo = pd.Timestamp("2011-01-01", tz="UTC")
    is_hi = pd.Timestamp("2020-12-31 23:59:59", tz="UTC")
    _profile_window("IS", is_roi, is_trades, is_marks, is_decade, weights_named, is_lo, is_hi)

    # ---- OOS window: SAME frozen weights + SAME committed exits, nothing selected ----
    print("\n\nbuilding OOS (2021+) book (SAME committed/frozen configs; weights FROZEN from IS)...")
    oos_roi, oos_trades, oos_marks, oos_decade = build_book("oos", histdata_root, cache_root, boundary)
    oos_years = sorted(next(iter(oos_roi.values())))
    oos_lo = pd.Timestamp(f"{oos_years[0]}-01-01", tz="UTC")
    oos_hi = pd.Timestamp(f"{oos_years[-1]}-12-31 23:59:59", tz="UTC")
    _profile_window("OOS", oos_roi, oos_trades, oos_marks, oos_decade, weights_named, oos_lo, oos_hi)


if __name__ == "__main__":  # pragma: no cover
    main()
