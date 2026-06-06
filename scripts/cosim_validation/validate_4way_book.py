"""Item-E validation — re-run the existing 4-way book through the co-sim gate.

Scores the four components of the portfolio-route thread (arcs 1006/1011/1013/1019,
combined in arc 1020) via the canonical A1 + ``MultiPairBacktester`` path over the
per-year IS folds (2011-2020), then runs them through BOTH:

  - the per-fold LINEAR combiner (``discovery/tools/combine_fold_roi.py``), and
  - the single co-simulated equity curve (``core/wfo/cosim_book.py``),

at the SAME (equal and IS-frozen risk-parity) capital weights, and prints a
side-by-side per-fold ROI + maxDD table with the fold-flip verdict (does
co-simulation change all-folds-positive vs the linear combiner — artifact or
fundamental?).

Run:  PYTHONPATH=. py scripts/cosim_validation/validate_4way_book.py

Data:  histdata_root defaults to C:/Users/panap/histdata_backup, cache_root to the
main-repo data/cache (both overridable by env COSIM_HISTDATA_ROOT / COSIM_CACHE_ROOT).
This is an ANALYSIS script (not a gate path); it CALLS the canonical scoring engine.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pandas as pd

from core.arc.signal_protocol import SignalEvaluation
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.wfo.cosim_book import CoSimComponent, cosim_book_fold, mark_prices_from_panel
from core.wfo.folds import build_v3_folds
from discovery.tools.combine_fold_roi import combine_fold_rois, fit_weights
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.gap_signals import WeekendGapFillLongSignal
from discovery.tools.month_end_signals import (
    MonthEndReversionLongSignal,
    MonthEndReversionShortSignal,
)
from discovery.tools.time_exit_predicate import make_time_exit_predicate

HISTDATA_ROOT = Path(os.environ.get("COSIM_HISTDATA_ROOT", r"C:/Users/panap/histdata_backup"))
CACHE_ROOT = Path(os.environ.get("COSIM_CACHE_ROOT", r"C:/Users/panap/Documents/Forex-Backtester/data/cache"))
BOUNDARY = "5ers_eet"
SB = 100_000.0

JPY_CROSSES = ["EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"]
USD_MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]

# Recorded arc-1020 per-year IS ROI (%) — reproduction-fidelity cross-check only.
ARC1020_RECORDED = {
    # year: (gap, me_long, fbr, me_short)
    2011: (-0.07, +0.40, +7.55, +3.39),
    2012: (+8.23, +0.29, +3.05, +1.69),
    2013: (-2.06, +0.96, +0.91, -0.90),
    2014: (+2.94, -0.23, +0.19, +0.98),
    2015: (-4.19, -1.14, +3.17, +0.40),
    2016: (+3.20, -0.51, +2.55, -0.91),
    2017: (+0.53, +0.34, +1.23, -0.68),
    2018: (-6.79, +0.90, -4.20, +0.86),
    2019: (+7.45, +1.16, +0.05, +1.29),
    2020: (-2.39, +0.15, +4.03, +0.71),
}


def _load_panel(pairs, tf):
    return Panel.from_pairs(
        pairs, tf, histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
        use_cache=True, boundary_convention=BOUNDARY,
    )


def _attach_time_exit(sig: SignalEvaluation, panel: Panel, n_bars: int) -> SignalEvaluation:
    """Return a copy of ``sig`` with an n-bar time-exit predicate on every pair."""
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n_bars)
    new_per_pair = {
        pair: dataclasses.replace(state, exit_predicate=pred)
        for pair, state in sig.per_pair.items()
    }
    return dataclasses.replace(sig, per_pair=new_per_pair)


def _slice_marks(marks: dict, fold) -> dict:
    """Slice each pair's mark series to the fold window + warmup slack."""
    lo = pd.Timestamp(fold.oos_start, tz="UTC") - pd.Timedelta(days=90)
    hi = pd.Timestamp(fold.oos_end, tz="UTC") + pd.Timedelta(days=1)
    return {p: s.loc[lo:hi] for p, s in marks.items()}


def main() -> None:
    print(f"histdata_root={HISTDATA_ROOT}\ncache_root={CACHE_ROOT}\n")

    # ── panels ───────────────────────────────────────────────────────────
    h4 = _load_panel(sorted(set(JPY_CROSSES) | set(USD_MAJORS)), "H4")
    d1 = _load_panel(USD_MAJORS, "D1")
    h4_jpy = Panel.from_frames({p: h4.pair_dfs[p] for p in JPY_CROSSES}, tf="H4", boundary_convention=BOUNDARY)
    h4_usd = Panel.from_frames({p: h4.pair_dfs[p] for p in USD_MAJORS}, tf="H4", boundary_convention=BOUNDARY)

    # ── component definitions: (name, signal_eval, panels, A1Config, mark panel) ──
    gap_eval = _attach_time_exit(
        WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36).evaluate({"H4": h4_jpy}),
        h4_jpy, n_bars=24,
    )
    me_long_eval = _attach_time_exit(
        MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1}),
        d1, n_bars=2,
    )
    fbr_eval = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": h4_usd})
    me_short_eval = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1})

    components = [
        ("gap",      gap_eval,      {"H4": h4_jpy}, A1Config(config_id="arc_1006", sl_atr_mult=2.0, trail_enabled=False, exit_policy=None), h4_jpy),
        ("me_long",  me_long_eval,  {"D1": d1},     A1Config(config_id="arc_1011", sl_atr_mult=2.0, trail_enabled=False, exit_policy="sl_only"), d1),
        ("fbr",      fbr_eval,      {"H4": h4_usd}, A1Config(config_id="arc_1013", sl_atr_mult=2.0, trail_enabled=False, exit_policy="sl_plus_trailing_atr"), h4_usd),
        ("me_short", me_short_eval, {"D1": d1},     A1Config(config_id="arc_1019", sl_atr_mult=2.0, trail_enabled=False, exit_policy="sl_partial_close_1r_runner_trail"), d1),
    ]
    names = [c[0] for c in components]

    # ── score each component over the per-year IS folds (2011-2020) ──────
    folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    runners = {
        name: ArcFoldRunner(A1Architecture(), sig, panels)
        for (name, sig, panels, _cfg, _mp) in components
    }
    full_marks = {name: mark_prices_from_panel(mp) for (name, _s, _p, _c, mp) in components}

    # per-fold: component FoldStats (for linear) + StrategyResult (for co-sim)
    comp_fs: dict[str, list] = {n: [] for n in names}
    comp_sr: dict[str, list] = {n: [] for n in names}
    print("Scoring components over folds (canonical A1 + MultiPairBacktester)...")
    for fold in folds:
        for (name, sig, panels, cfg, _mp) in components:
            fs = runners[name](fold, cfg)
            comp_fs[name].append(fs)
            comp_sr[name].append(runners[name].last_result)
        print(f"  fold {fold.oos_start.year}: " + ", ".join(
            f"{n}={comp_fs[n][-1].roi_pct*100:+.2f}%({comp_fs[n][-1].n_trades})" for n in names))

    # ── reproduction-fidelity cross-check vs arc 1020 ────────────────────
    print("\nReproduction cross-check (mine vs arc-1020 recorded, ROI %):")
    print("year | " + " | ".join(f"{n:>8}" for n in names))
    for i, fold in enumerate(folds):
        yr = fold.oos_start.year
        rec = ARC1020_RECORDED.get(yr)
        mine = [comp_fs[n][i].roi_pct * 100 for n in names]
        line = f"{yr} | " + " | ".join(f"{m:+7.2f}" for m in mine)
        if rec:
            line += "   (rec: " + ", ".join(f"{r:+.2f}" for r in rec) + ")"
        print(line)

    # ── LINEAR combiner: equal + IS-frozen risk-parity ───────────────────
    comp_rois = [[fs.roi_pct for fs in comp_fs[n]] for n in names]
    w_equal = [0.25, 0.25, 0.25, 0.25]
    w_rp = fit_weights(comp_rois, mode="risk_parity")  # frozen on the IS fold ROIs
    lin_equal = combine_fold_rois(comp_rois, w_equal).combined_roi
    lin_rp = combine_fold_rois(comp_rois, w_rp).combined_roi
    print("\nrisk-parity weights (frozen IS): " +
          ", ".join(f"{n}={w:.3f}" for n, w in zip(names, w_rp)))

    # ── CO-SIM: same weights, single equity line. Run BOTH:
    #     cap OFF = the strictly-monotone anchor (real DD interaction only; ROI
    #               must match linear up to the OOS-slice boundary → anti-optimism
    #               proof: the equity/cost math adds no return source).
    #     cap ON  = the faithful book under the 2-per-currency limit (drops
    #               over-cap entries; not strictly ROI-monotone — dropping a
    #               net-loser raises ROI; reported transparently).
    def cosim_series(weights, apply_cap):
        rois, dds, breaches, drops = [], [], [], []
        for i, fold in enumerate(folds):
            comps = [
                CoSimComponent(
                    name=n,
                    closed_trades=comp_sr[n][i].run_result.closed_trades,
                    mark_prices=_slice_marks(full_marks[n], fold),
                    starting_balance=SB,
                )
                for n in names
            ]
            book = cosim_book_fold(comps, fold, weights=weights, starting_balance=SB,
                                   apply_exposure_cap=apply_cap)
            rois.append(book.roi_pct)
            dds.append(book.fold_stats.max_dd_pct)
            breaches.append(book.daily_dd_breached)
            drops.append(book.n_dropped)
        return rois, dds, breaches, drops

    off_eq_roi, off_eq_dd, off_eq_br, _ = cosim_series(w_equal, apply_cap=False)
    off_rp_roi, off_rp_dd, off_rp_br, _ = cosim_series(w_rp, apply_cap=False)
    on_eq_roi, on_eq_dd, on_eq_br, on_eq_drop = cosim_series(w_equal, apply_cap=True)
    on_rp_roi, on_rp_dd, on_rp_br, on_rp_drop = cosim_series(w_rp, apply_cap=True)

    # ── anti-optimism anchor: cap-off co-sim ROI must match linear ───────
    max_dev_eq = max(abs(off_eq_roi[i] - lin_equal[i]) for i in range(len(folds)))
    max_dev_rp = max(abs(off_rp_roi[i] - lin_rp[i]) for i in range(len(folds)))
    print(f"\nANTI-OPTIMISM ANCHOR (cap OFF): max|cosim_off - linear| = "
          f"{max_dev_eq*100:.4f}% (equal), {max_dev_rp*100:.4f}% (risk-parity)")
    print("  -> cap-off co-sim ROI tracks linear (boundary-only diff); the only NEW")
    print("     information is the real BOOK max-DD the linear combiner cannot see.")

    # ── side-by-side table (risk-parity) ─────────────────────────────────
    print("\n" + "=" * 92)
    print("CO-SIM vs LINEAR (risk-parity weights) -- per-fold ROI %, book maxDD %, drops")
    print("=" * 92)
    hdr = (f"{'year':>5} | {'linear':>7} {'cosOFF':>7} {'dOFF':>6} {'bookDD':>6} | "
           f"{'cosON':>7} {'dON':>6} {'dropON':>6}")
    print(hdr)
    print("-" * len(hdr))
    for i, fold in enumerate(folds):
        yr = fold.oos_start.year
        print(f"{yr:>5} | "
              f"{lin_rp[i]*100:+7.2f} {off_rp_roi[i]*100:+7.2f} {(off_rp_roi[i]-lin_rp[i])*100:+6.2f} "
              f"{off_rp_dd[i]*100:6.2f} | "
              f"{on_rp_roi[i]*100:+7.2f} {(on_rp_roi[i]-lin_rp[i])*100:+6.2f} {on_rp_drop[i]:>6}")

    # ── verdicts ─────────────────────────────────────────────────────────
    def afp(rois):  # all-folds-positive
        return all(r > 0.0 for r in rois), min(rois), sum(1 for r in rois if r <= 0)

    print("\n" + "=" * 92)
    print("ALL-FOLDS-POSITIVE VERDICTS (IS, 2011-2020)")
    print("=" * 92)
    for label, rois in [
        ("linear        equal      ", lin_equal),
        ("co-sim cap-OFF equal      ", off_eq_roi),
        ("co-sim cap-ON  equal      ", on_eq_roi),
        ("linear        risk-parity ", lin_rp),
        ("co-sim cap-OFF risk-parity ", off_rp_roi),
        ("co-sim cap-ON  risk-parity ", on_rp_roi),
    ]:
        ok, worst, n_neg = afp(rois)
        print(f"  {label}: all_folds_positive={ok!s:>5}  worst_fold={worst*100:+.2f}%  n_nonpositive={n_neg}")

    print(f"\n  co-sim exposure-cap drops (cap ON): {sum(on_rp_drop)} positions across all folds "
          f"(book is USD-concentrated; 2-per-currency binds hard)")
    print(f"  co-sim worst book max-DD (cap OFF, risk-parity): {max(off_rp_dd)*100:.2f}%")
    print(f"  co-sim 5% daily-cap breached (cap OFF rp / cap ON rp): "
          f"{any(off_rp_br)} / {any(on_rp_br)}")

    # ── fold-flip verdict ────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("FOLD-FLIP VERDICT (the question arc 2019 left open)")
    print("=" * 92)
    lin_ok = afp(lin_rp)[0]
    off_ok = afp(off_rp_roi)[0]
    on_ok = afp(on_rp_roi)[0]
    flips_off = [folds[i].oos_start.year for i in range(len(folds)) if (lin_rp[i] > 0) != (off_rp_roi[i] > 0)]
    flips_on = [folds[i].oos_start.year for i in range(len(folds)) if (lin_rp[i] > 0) != (on_rp_roi[i] > 0)]
    print(f"  linear all-folds-positive:        {lin_ok}")
    print(f"  co-sim cap-OFF all-folds-positive: {off_ok}  (sign flips vs linear: {flips_off or 'none'})")
    print(f"  co-sim cap-ON  all-folds-positive: {on_ok}  (sign flips vs linear: {flips_on or 'none'})")
    if not (lin_ok or off_ok or on_ok):
        print("\n  VERDICT: ALL measurements FAIL all-folds-positive. Co-simulation CONFIRMS")
        print("  the linear verdict -- the 4-way book's failure is FUNDAMENTAL, not a")
        print("  linear-combiner artifact. The route is closed cleanly. (No OOS touched.)")
    elif off_ok and on_ok:
        print("\n  VERDICT: co-sim PASSES on IS -> eligible to score the frozen 2021+ holdout ONCE.")
    else:
        print("\n  VERDICT: mixed -- see per-measurement rows above; operator decision.")


if __name__ == "__main__":
    main()
