"""N-leg SUBSET deployment-vehicle geometry across IS *and* OOS (EXPERIMENT / analysis).

GEOMETRY ONLY on an ALREADY-COMPUTED contiguous co-sim equity curve — it never
realizes P&L, never scores a trade, never touches the gate, never *selects* on OOS.
It is the parameterized generalization of arc 2056's `book_deploy_profile.py` (which
hard-codes the 4-leg NAMES) and arc 2053/2055's `solo_deploy_profile.py` (1-leg). The
ONLY change is that NAMES is an arbitrary subset of {gap, me_long, fbr, me_short};
every measurement primitive (build_component, cosim_book_fold, fit_weights,
compute_risk_profile, compute_sharpe, feasibility_horizon_years) is reused UNCHANGED.

WHY (arc 2059). The deploy-object vehicle matrix had solo (1-leg, me_long: 2053 IS /
2055 OOS) and the full 4-way book (2054 IS / 2056 OOS) but NOT the interpolating
2-leg cell. The operator's rational path-A deploy choice is me_long (the sole OOS-mean-
robust anchor, 1046) PLUS at most one decorrelating partner for drawdown-depth cover —
the question is whether any such 2-leg book DOMINATES: better OOS Calmar than me_long-
solo (0.31, 2055) WITHOUT collapsing to the 4-way book's halved OOS mean (~+0.25%/yr,
2056, dragged by THREE weak legs). This fills the missing 2-leg cell.

§4 COMPLIANCE. Weights are RISK-PARITY *frozen on IS* (`fit_weights(...)`) over the
SUBSET's IS per-year ROIs and applied UNCHANGED to the OOS curve — nothing is fit to
the holdout. Exits are the committed/frozen per-component configs (build_component,
identical to validate_4way_book.py / solo_deploy_profile.build_component). The
per-component frozen OOS per-year series were already spent in arc 1042/2055/2056;
reading vehicle geometry off the already-spent frozen series adds NO selection (arc
2055/2056 precedent). The OOS curve is measure-once CHARACTERIZATION, never a target.

Reproduce:  PYTHONPATH=. SUBSET="me_long,fbr" py discovery/tools/subset_deploy_profile.py
Created by: arc 2059.
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
    SB,
    _reid,
    build_component,
)

ALL_NAMES = ["gap", "me_long", "fbr", "me_short"]


def _folds(window):
    if window == "oos":
        return list(build_oos_year_folds(start_year=2021))
    return [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]


def build_book(names, window, histdata_root, cache_root, boundary):
    """Per-component contiguous trade book + per-year ROIs for one window (SUBSET)."""
    folds = _folds(window)
    decade = Fold(fold_id=99, is_start=folds[0].is_start, is_end=folds[0].is_end,
                  oos_start=folds[0].oos_start, oos_end=folds[-1].oos_end)

    per_year_roi = {n: {} for n in names}
    decade_trades = {n: [] for n in names}
    marks = {}
    for n in names:
        sig, panels, cfg, mark_panel = build_component(n, histdata_root, cache_root, boundary)
        marks[n] = mark_prices_from_panel(mark_panel)
        runner = ArcFoldRunner(A1Architecture(), sig, panels)
        for fi, fold in enumerate(folds):
            fs = runner(fold, cfg)
            per_year_roi[n][fold.oos_start.year] = fs.roi_pct * 100.0
            decade_trades[n].extend(_reid(runner.last_result.run_result.closed_trades, fi))
        decade_trades[n] = tuple(decade_trades[n])
    return per_year_roi, decade_trades, marks, decade


def _cosim(names, weights, cap, decade_trades, marks, decade):
    cc = [CoSimComponent(name=n, closed_trades=decade_trades[n],
                         mark_prices=marks[n], starting_balance=SB) for n in names]
    return cosim_book_fold(cc, decade, weights=weights, starting_balance=SB, apply_exposure_cap=cap)


def _profile_window(names, label, per_year_roi, decade_trades, marks, decade, weights_named, lo, hi):
    nn = len(names)
    years = sorted(next(iter(per_year_roi.values())))
    print(f"\nper-year book-component ROI % ({label}):")
    for yr in years:
        row = " ".join(f"{n}={per_year_roi[n][yr]:+.2f}" for n in names)
        print(f"  {yr}: {row}")

    print("\n" + "=" * 80)
    print(f"{nn}-LEG BOOK [{','.join(names)}] -- CONTIGUOUS {years[0]}-{years[-1]} {label} DEPLOY GEOMETRY")
    print("=" * 80)
    for wlabel, w in weights_named:
        book_yr = [sum(w[i] * per_year_roi[names[i]][yr] for i in range(nn)) for yr in years]
        print(f"\n--- weights [{wlabel}]: " + ", ".join(f"{n}={w[i]:.3f}" for i, n in enumerate(names)))
        print(f"  book per-year ROI: mean {statistics.mean(book_yr):+.3f}% | "
              f"sd {statistics.pstdev(book_yr):.3f}% | neg {sum(1 for r in book_yr if r < 0)}/{len(book_yr)} | "
              f"worst {min(book_yr):+.3f}%")
        for cap_label, cap in [("cap-OFF (monotone bound)", False), ("cap-ON  (faithful)      ", True)]:
            book = _cosim(names, w, cap, decade_trades, marks, decade)
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
    names = [s.strip() for s in os.environ.get("SUBSET", "me_long,fbr").split(",") if s.strip()]
    assert all(n in ALL_NAMES for n in names), f"unknown component in {names}; valid={ALL_NAMES}"
    assert len(names) >= 1
    nn = len(names)
    histdata_root = Path(os.environ.get("COSIM_HISTDATA_ROOT", r"C:/Users/panap/histdata_backup"))
    cache_root = Path(os.environ.get("COSIM_CACHE_ROOT", r"C:/Users/panap/Documents/Forex-Backtester/data/cache"))
    boundary = "5ers_eet"

    print(f"SUBSET = {names}")
    print("building IS (2011-2020) subset book (canonical A1 + MultiPairBacktester)...")
    is_roi, is_trades, is_marks, is_decade = build_book(names, "is", histdata_root, cache_root, boundary)
    comp_is_rois = [[is_roi[n][yr] / 100.0 for yr in range(2011, 2021)] for n in names]
    w_rp = fit_weights(comp_is_rois, mode="risk_parity")   # FROZEN — applied unchanged to OOS
    w_eq = [1.0 / nn] * nn
    print("\nFROZEN IS risk-parity weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(names, w_rp)))
    weights_named = [("equal", w_eq), ("risk-parity(frozen-IS)", w_rp)]

    is_lo = pd.Timestamp("2011-01-01", tz="UTC")
    is_hi = pd.Timestamp("2020-12-31 23:59:59", tz="UTC")
    _profile_window(names, "IS", is_roi, is_trades, is_marks, is_decade, weights_named, is_lo, is_hi)

    print("\n\nbuilding OOS (2021+) subset book (SAME committed/frozen configs; weights FROZEN from IS)...")
    oos_roi, oos_trades, oos_marks, oos_decade = build_book(names, "oos", histdata_root, cache_root, boundary)
    oos_years = sorted(next(iter(oos_roi.values())))
    oos_lo = pd.Timestamp(f"{oos_years[0]}-01-01", tz="UTC")
    oos_hi = pd.Timestamp(f"{oos_years[-1]}-12-31 23:59:59", tz="UTC")
    _profile_window(names, "OOS", oos_roi, oos_trades, oos_marks, oos_decade, weights_named, oos_lo, oos_hi)


if __name__ == "__main__":  # pragma: no cover
    main()
