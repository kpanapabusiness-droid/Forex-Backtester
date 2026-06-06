"""arc 2045 — Deployment GEOMETRY of the honest 2-leg (me_long+fbr) book.

arc 2044 (+1044) established that under honest §5f exits the 4-way book's deploy object
reduces to the 2-leg me_long+fbr (gap & me_short flip mean-negative → pure drag; the
honest 4-leg book is mean-DRAGGED below the 2-leg). But the DEPLOYMENT GEOMETRY that
actually binds (arc 1033/2033: Calmar / time-underwater / prop-firm feasibility — NOT
depth or per-year sign) was only ever computed on the COMMITTED 4-leg book. This arc
profiles the honest deploy object — does dropping the two exit-fragile drag legs (gap,
me_short) AND using fbr's HONEST (frozen-nested-§5f) exit improve the operator's actual
deploy candidate's risk geometry?

Honest deploy object = me_long (committed sl_only/2-bar — exit-robust, committed≈honest,
arc 2042) + fbr (FROZEN honest §5f exit = freeze_best_over_folds, NOT the committed
sl_plus_trailing_atr full-sample pick that 2040 flagged ~40% optimistic). A FROZEN single
config per leg → a genuinely deployable contiguous account (no per-fold exit re-selection).

Method: 100% canonical scoring (A1→MultiPairBacktester, FundedNext, risk 0.005). fbr's
frozen honest exit via BUILT nested_exit_selection.freeze_best_over_folds (3 metrics).
Contiguous co-sim via the canonical cosim_book_fold (reusing equity_risk_profile's
_build pattern, specialized to 2 legs); geometry via BUILT compute_risk_profile;
vehicle feasibility via BUILT propfirm_feasibility. Committed 4-leg numbers cited from
arcs 1033/2033 (already reproduced many times) as the comparison baseline; me_long/fbr
per-year ROIs cross-checked against the committed record as the in-script anchor.
OOS NEVER touched (IS characterization; book fails IS AFP).

Run:  PYTHONPATH=. py _disco_work/arc2045_honest_2leg_geometry.py
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd

from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.wfo.cosim_book import CoSimComponent, cosim_book_fold, mark_prices_from_panel
from core.wfo.discovery_measure import run_config_over_folds
from core.wfo.folds import Fold, build_v3_folds
from discovery.tools.combine_fold_roi import fit_weights
from discovery.tools.equity_risk_profile import compute_risk_profile
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.month_end_signals import MonthEndReversionLongSignal
from discovery.tools.nested_exit_selection import (
    freeze_best_over_folds,
    metric_afp_then_mean,
    metric_mean_roi,
    metric_worst_then_mean,
)
from discovery.tools.propfirm_feasibility import compute_sharpe, feasibility_horizon_years
from discovery.tools.time_exit_predicate import make_time_exit_predicate

HISTDATA_ROOT = Path(os.environ.get("COSIM_HISTDATA_ROOT", r"C:/Users/panap/histdata_backup"))
CACHE_ROOT = Path(os.environ.get("COSIM_CACHE_ROOT", r"C:/Users/panap/Documents/Forex-Backtester/data/cache"))
BOUNDARY = "5ers_eet"
USD = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
SB = 100_000.0
EXITS = ["sl_only", "sl_plus_tp_2r", "sl_plus_tp_3r", "sl_plus_trailing_atr",
         "sl_plus_trailing_swing", "sl_partial_close_1r_runner_trail"]
SLS = [1.5, 2.0, 2.5]
_PER_YEAR_OFFSET = 100_000

# committed arc-1020 per-year ROI %: (me_long, fbr) — the in-script reproduction anchor
REC_MELONG = {2011: 0.40, 2012: 0.29, 2013: 0.96, 2014: -0.23, 2015: -1.14,
              2016: -0.51, 2017: 0.34, 2018: 0.90, 2019: 1.16, 2020: 0.15}
# fbr committed double-trail (arc 1020) vs trail-off (arc 2040/2043 +2.084); we use the HONEST exit
# so committed fbr is only a sanity reference, not the deploy object.


def _load(pairs, tf):
    return Panel.from_pairs(pairs, tf, histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
                            use_cache=True, boundary_convention=BOUNDARY)


def _attach_te(sig, panel, n):
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n)
    return dataclasses.replace(sig, per_pair={
        p: dataclasses.replace(s, exit_predicate=pred) for p, s in sig.per_pair.items()})


def _reid(trades, fold_idx):
    base = fold_idx * _PER_YEAR_OFFSET
    out = []
    for t in trades:
        out.append(dataclasses.replace(
            t, position_id=base + int(t.position_id),
            parent_position_id=(None if t.parent_position_id is None else base + int(t.parent_position_id))))
    return out


def main():
    print("ARC 2045 — honest 2-leg (me_long+fbr) book DEPLOYMENT GEOMETRY")
    print("committed 4-leg baseline (arc 1033/2033, RP): maxDD 1.59% | Calmar 0.24-0.36 | "
          "underwater ~4.7yr/98% | Sharpe ~0.4-0.7 | prop-firm T_min 1.4-8.4yr\n")
    h4_usd = _load(USD, "H4")
    d1 = _load(USD, "D1")
    h4_usd_only = Panel.from_frames({p: h4_usd.pair_dfs[p] for p in USD}, tf="H4", boundary_convention=BOUNDARY)

    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]

    # ── signals ──────────────────────────────────────────────────────────
    me_long_eval = _attach_te(MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1}), d1, 2)
    fbr_eval = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": h4_usd_only})

    # ── determine fbr's FROZEN honest §5f exit (the deployable single config) ──
    print("Determining fbr's frozen honest §5f exit (score 18-cfg grid over IS folds)...")
    fbr_runner = ArcFoldRunner(A1Architecture(), fbr_eval, {"H4": h4_usd_only})
    fbr_grid = {f"{ex}|sl{sl}": A1Config(config_id=f"fbr_{ex}_sl{sl}", sl_atr_mult=sl,
                                         trail_enabled=False, exit_policy=ex)
                for ex in EXITS for sl in SLS}
    fbr_scored = {lbl: list(run_config_over_folds(fbr_runner, is_folds, cfg)) for lbl, cfg in fbr_grid.items()}
    frozen = {}
    for mname, mfn in (("mean_roi", metric_mean_roi), ("afp_then_mean", metric_afp_then_mean),
                       ("worst_then_mean", metric_worst_then_mean)):
        fl = freeze_best_over_folds(fbr_scored, selection=mfn)
        full_mean = np.mean([fs.roi_pct for fs in fbr_scored[fl]]) * 100
        frozen[mname] = fl
        print(f"  frozen fbr exit [{mname:<14}] = {fl:<40} full-IS mean {full_mean:+.3f}%")
    # deploy choice = the conservative gate-aligned metric (mean_roi grabs the trailing_swing §5f trap, arc 2040)
    DEPLOY_METRIC = "afp_then_mean"
    fbr_deploy_label = frozen[DEPLOY_METRIC]
    ex, slstr = fbr_deploy_label.split("|sl")
    fbr_cfg = A1Config(config_id="fbr_honest_frozen", sl_atr_mult=float(slstr), trail_enabled=False, exit_policy=ex)
    print(f"\n  >> honest deploy fbr exit (metric={DEPLOY_METRIC}): {fbr_deploy_label}\n")

    me_long_cfg = A1Config(config_id="me_long_committed", sl_atr_mult=2.0, trail_enabled=False, exit_policy="sl_only")

    # ── per-year score the 2 legs; capture trades for the contiguous co-sim ──
    print("Per-year scoring me_long (committed) + fbr (frozen-honest) [canonical]...")
    comps = [("me_long", me_long_eval, {"D1": d1}, me_long_cfg, d1),
             ("fbr", fbr_eval, {"H4": h4_usd_only}, fbr_cfg, h4_usd_only)]
    names = [c[0] for c in comps]
    runners = {n: ArcFoldRunner(A1Architecture(), sig, panels) for (n, sig, panels, _c, _mp) in comps}
    cfgs = {n: c for (n, _s, _p, c, _mp) in comps}
    marks = {n: mark_prices_from_panel(mp) for (n, _s, _p, _c, mp) in comps}
    decade_trades = {n: [] for n in names}
    per_year_roi = {n: {} for n in names}
    for fi, fold in enumerate(is_folds):
        for n in names:
            fs = runners[n](fold, cfgs[n])
            per_year_roi[n][fold.oos_start.year] = fs.roi_pct * 100.0
            decade_trades[n].extend(_reid(runners[n].last_result.run_result.closed_trades, fi))
    decade_trades = {n: tuple(v) for n, v in decade_trades.items()}

    # anchor check: me_long reproduces committed
    print("\nme_long per-year ROI vs committed record (anchor):")
    dev = 0.0
    for yr in range(2011, 2021):
        d = abs(per_year_roi["me_long"][yr] - REC_MELONG[yr])
        dev = max(dev, d)
        print(f"  {yr}: me_long {per_year_roi['me_long'][yr]:+.2f} (rec {REC_MELONG[yr]:+.2f})  "
              f"fbr-honest {per_year_roi['fbr'][yr]:+.2f}")
    print(f"  me_long max |mine-rec| = {dev:.3f}pp  (anchor)")
    print(f"  fbr honest-frozen IS mean = {np.mean([per_year_roi['fbr'][y] for y in range(2011,2021)]):+.3f}% "
          f"(vs committed double-trail +1.854% / trail-off +2.084%)")

    # ── 2-leg RP + equal weights (frozen IS) ──
    comp_rois = [[per_year_roi[n][yr] / 100.0 for yr in range(2011, 2021)] for n in names]
    w_rp = fit_weights(comp_rois, mode="risk_parity")
    w_eq = [0.5, 0.5]
    print("\n2-leg risk-parity weights (frozen IS): " + ", ".join(f"{n}={w:.3f}" for n, w in zip(names, w_rp)))
    # per-fold book ROI (per-year-reset, for AFP + significance)
    for wlabel, w in (("risk-parity", w_rp), ("equal", w_eq)):
        book_roi = [sum(w[k] * comp_rois[k][i] for k in range(2)) * 100 for i in range(10)]
        mean = np.mean(book_roi)
        sd = np.std(book_roi, ddof=1)
        t = mean / (sd / np.sqrt(10))
        npos = sum(1 for r in book_roi if r > 0)
        print(f"  2-leg {wlabel:<11} per-fold%={[round(x,2) for x in book_roi]}")
        print(f"     mean={mean:+.3f}% sd={sd:.3f}% t={t:+.2f} pos={npos}/10 worst={min(book_roi):+.3f}%")

    # ── contiguous co-sim deployment geometry ──
    decade = Fold(fold_id=99, is_start=is_folds[0].is_start, is_end=is_folds[0].is_end,
                  oos_start=is_folds[0].oos_start, oos_end=is_folds[-1].oos_end)
    lo = pd.Timestamp("2011-01-01", tz="UTC")
    hi = pd.Timestamp("2020-12-31 23:59:59", tz="UTC")

    def cosim(weights, cap):
        cc = [CoSimComponent(name=n, closed_trades=decade_trades[n], mark_prices=marks[n], starting_balance=SB)
              for n in names]
        return cosim_book_fold(cc, decade, weights=weights, starting_balance=SB, apply_exposure_cap=cap)

    print("\n" + "=" * 78)
    print("HONEST 2-LEG (me_long + fbr-honest) — CONTIGUOUS 2011-2020 IS DEPLOYMENT GEOMETRY")
    print("=" * 78)
    for wlabel, w in (("equal      ", w_eq), ("risk-parity", w_rp)):
        for caplabel, cap in (("cap-OFF (monotone bound)", False), ("cap-ON  (faithful)      ", True)):
            book = cosim(w, cap)
            pdd = book.per_day_max_dd_eet
            worst_day = float(pdd["day_max_dd_base_pct"].max() * 100.0) if len(pdd) else float("nan")
            prof = compute_risk_profile(book.net_equity, lo=lo, hi=hi, worst_day_dd_pct=worst_day)
            shp = compute_sharpe(book.net_equity, lo=lo, hi=hi)
            print(f"\n[{wlabel}| {caplabel}] n_dropped={book.n_dropped} daily5%breach={book.daily_dd_breached}")
            print("  " + prof.as_row())
            print(f"  Sharpe(ann)={shp.daily_sharpe_ann:+.3f} Sortino={shp.daily_sortino_ann:+.3f}  "
                  f"deepest DD {prof.peak_before_trough.date()} -> {prof.trough_date.date()}")
            # prop-firm feasibility at a representative 10%/10% challenge
            try:
                tmin = feasibility_horizon_years(prof.calmar, 10.0, 10.0)
                print(f"  prop-firm T_min (10% target / 10% maxDD limit) = {tmin:.2f} yr  (4-leg was 1.4-8.4yr)")
            except Exception as e:
                print(f"  (feasibility n/a: {e})")

    print("\nDONE.")


if __name__ == "__main__":
    main()
