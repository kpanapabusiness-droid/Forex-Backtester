"""Deployment risk profile of a co-simulated equity curve (EXPERIMENT / analysis).

GEOMETRY ONLY on an ALREADY-COMPUTED equity curve — it never realizes P&L, never
scores a trade, never touches the gate. It reads a net-equity `pd.Series` produced
by the canonical co-sim book (`core/wfo/cosim_book.py`) and reports the standard
deployment risk statistics a per-fold ROI table structurally cannot see:

  - **Contiguous peak-to-trough max-DD** over the whole window (a drawdown that
    chains across calendar-year boundaries is invisible to a per-year-reset table;
    the co-sim doc reports per-FOLD bookDD only — each year resets the high-water
    mark, so the real decade trough is understated).
  - **Calmar** (annualized return / max-DD) — risk-INVARIANT in the linear band
    (arc 1024: ROI and DD both scale linearly with per-trade risk_pct, so their
    ratio is fixed), hence the single most transferable deploy number.
  - **Longest underwater duration** (calendar days from a high-water mark until the
    book recovers it) and time-underwater fraction.
  - **Daily-cap headroom** — worst single-day DD vs the 5% FundedNext cap.

The `__main__` reproduces the 4-component portfolio-route book (arcs
1006/1011/1013/1019, combined 1020; configs IDENTICAL to
`scripts/cosim_validation/validate_4way_book.py`) as ONE contiguous 2011-2020 IS
equity curve via the canonical `cosim_book_fold`, then prints its risk profile at
equal + IS-frozen risk-parity weights, cap-OFF (strictly-monotone bound) and cap-ON
(faithful). OOS (2021+) is NEVER touched — this is IS characterization only.

Reproduce:  PYTHONPATH=. py discovery/tools/equity_risk_profile.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskProfile:
    """Deployment risk statistics for one contiguous equity curve."""

    start: pd.Timestamp
    end: pd.Timestamp
    years: float
    total_return_pct: float
    ann_return_pct: float
    max_dd_pct: float            # peak-to-trough, positive %
    calmar: float                # ann_return_pct / max_dd_pct
    trough_date: pd.Timestamp    # date of the deepest drawdown
    peak_before_trough: pd.Timestamp
    longest_underwater_days: int
    time_underwater_frac: float  # fraction of bars below the running high-water mark
    worst_day_dd_pct: float      # deepest single-EET-day DD (for the 5% cap headroom)

    def as_row(self) -> str:
        return (
            f"ret {self.ann_return_pct:+.3f}%/yr | maxDD {self.max_dd_pct:.3f}% | "
            f"Calmar {self.calmar:.3f} | underwater {self.longest_underwater_days}d "
            f"({self.time_underwater_frac*100:.0f}% of time) | "
            f"worst-day {self.worst_day_dd_pct:.3f}%"
        )


def compute_risk_profile(
    equity: pd.Series,
    *,
    lo: pd.Timestamp | None = None,
    hi: pd.Timestamp | None = None,
    worst_day_dd_pct: float | None = None,
) -> RiskProfile:
    """Risk statistics for a net-equity curve, optionally window-sliced.

    `equity` is a strictly time-indexed net-equity `pd.Series` (UTC). `lo`/`hi`
    slice it to the characterization window (inclusive). `worst_day_dd_pct` is the
    deepest single-day DD if the caller has it from `compute_per_day_max_dd`
    (otherwise NaN — the curve alone cannot recover EET day buckets).
    """
    eq = equity.dropna().sort_index()
    if lo is not None:
        eq = eq.loc[eq.index >= lo]
    if hi is not None:
        eq = eq.loc[eq.index <= hi]
    if len(eq) < 2:
        raise ValueError("need >= 2 equity points for a risk profile")

    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0          # <= 0
    max_dd = float(-drawdown.min())
    trough_date = drawdown.idxmin()
    peak_before = eq.loc[:trough_date].idxmax()

    start, end = eq.index[0], eq.index[-1]
    years = (end - start).days / 365.25
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    ann_return = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    calmar = (ann_return * 100.0) / (max_dd * 100.0) if max_dd > 1e-12 else float("inf")

    # Longest underwater: calendar days from a high-water mark until the curve
    # recovers it. A bar is "at a peak" when drawdown ~ 0; the span between
    # consecutive peaks (or peak->end if never recovered) is an underwater stretch.
    underwater = (drawdown < -1e-9).to_numpy()
    times = eq.index
    longest_days = 0
    last_peak_time = times[0]
    for i in range(len(eq)):
        if not underwater[i]:                  # back at (or above) the high-water mark
            span = (times[i] - last_peak_time).days
            longest_days = max(longest_days, span)
            last_peak_time = times[i]
    # tail: an unrecovered drawdown running to the end of the window
    longest_days = max(longest_days, (times[-1] - last_peak_time).days)
    time_uw_frac = float(underwater.mean())

    return RiskProfile(
        start=start,
        end=end,
        years=years,
        total_return_pct=total_return * 100.0,
        ann_return_pct=ann_return * 100.0,
        max_dd_pct=max_dd * 100.0,
        calmar=calmar,
        trough_date=trough_date,
        peak_before_trough=peak_before,
        longest_underwater_days=int(longest_days),
        time_underwater_frac=time_uw_frac,
        worst_day_dd_pct=(float("nan") if worst_day_dd_pct is None else worst_day_dd_pct),
    )


# ───────────────────────── __main__: 4-way book contiguous curve ─────────────────────────
def _build_4way_contiguous():  # pragma: no cover - analysis driver
    """Reproduce the 4-component book as ONE contiguous 2011-2020 IS curve.

    Components + configs IDENTICAL to scripts/cosim_validation/validate_4way_book.py
    (the committed item-E reproduction). To reproduce the committed NON-compounding
    per-fold record exactly, each component is scored PER-YEAR (each fold resets to
    SB, sizing at risk_pct*SB — the committed convention), its trades re-id'd with a
    per-year offset (so concatenating across years does not collide position ids),
    and the decade of trades concatenated. `cosim_book_fold` then SUPERIMPOSES these
    constant-notional contributions onto one clock (it never re-sizes off a running
    balance — see _position_contribution), so the contiguous curve is the
    constant-fraction-of-initial-capital book whose per-year increments equal the
    committed per-fold linear ROIs, now with intra-year drawdown resolution.
    """
    import dataclasses
    import os
    from pathlib import Path

    from core.architectures.a1_system_level_filter import A1Architecture, A1Config
    from core.runners.arc_fold_runner import ArcFoldRunner
    from core.sim.account import ClosedTrade
    from core.sim.panel import Panel
    from core.wfo.cosim_book import CoSimComponent, cosim_book_fold, mark_prices_from_panel
    from core.wfo.folds import Fold, build_v3_folds
    from discovery.tools.combine_fold_roi import fit_weights

    _PER_YEAR_OFFSET = 100_000  # << cosim's 10M component offset; > intra-year pid range

    def _reid(trades, fold_idx):
        """Offset a fold's trade position ids so cross-year concat never collides."""
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
    from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
    from discovery.tools.gap_signals import WeekendGapFillLongSignal
    from discovery.tools.month_end_signals import (
        MonthEndReversionLongSignal,
        MonthEndReversionShortSignal,
    )
    from discovery.tools.time_exit_predicate import make_time_exit_predicate

    histdata_root = Path(os.environ.get("COSIM_HISTDATA_ROOT", r"C:/Users/panap/histdata_backup"))
    cache_root = Path(os.environ.get("COSIM_CACHE_ROOT", r"C:/Users/panap/Documents/Forex-Backtester/data/cache"))
    boundary = "5ers_eet"
    SB = 100_000.0
    JPY = ["EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"]
    USD = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    REC = {  # committed arc-1020 per-year ROI %: (gap, me_long, fbr, me_short)
        2011: (-0.07, 0.40, 7.55, 3.39), 2012: (8.23, 0.29, 3.05, 1.69),
        2013: (-2.06, 0.96, 0.91, -0.90), 2014: (2.94, -0.23, 0.19, 0.98),
        2015: (-4.19, -1.14, 3.17, 0.40), 2016: (3.20, -0.51, 2.55, -0.91),
        2017: (0.53, 0.34, 1.23, -0.68), 2018: (-6.79, 0.90, -4.20, 0.86),
        2019: (7.45, 1.16, 0.05, 1.29), 2020: (-2.39, 0.15, 4.03, 0.71),
    }

    def load(pairs, tf):
        return Panel.from_pairs(pairs, tf, histdata_root=histdata_root, cache_root=cache_root,
                                use_cache=True, boundary_convention=boundary)

    print("loading panels (warm cache)...")
    h4 = load(sorted(set(JPY) | set(USD)), "H4")
    d1 = load(USD, "D1")
    h4_jpy = Panel.from_frames({p: h4.pair_dfs[p] for p in JPY}, tf="H4", boundary_convention=boundary)
    h4_usd = Panel.from_frames({p: h4.pair_dfs[p] for p in USD}, tf="H4", boundary_convention=boundary)

    def attach_te(sig, panel, n):
        pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n)
        return dataclasses.replace(sig, per_pair={
            p: dataclasses.replace(s, exit_predicate=pred) for p, s in sig.per_pair.items()})

    gap_eval = attach_te(WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36).evaluate({"H4": h4_jpy}), h4_jpy, 24)
    me_long_eval = attach_te(MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1}), d1, 2)
    fbr_eval = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": h4_usd})
    me_short_eval = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1})

    comps = [
        ("gap", gap_eval, {"H4": h4_jpy}, A1Config(config_id="arc_1006", sl_atr_mult=2.0, trail_enabled=False, exit_policy=None), h4_jpy),
        ("me_long", me_long_eval, {"D1": d1}, A1Config(config_id="arc_1011", sl_atr_mult=2.0, trail_enabled=False, exit_policy="sl_only"), d1),
        ("fbr", fbr_eval, {"H4": h4_usd}, A1Config(config_id="arc_1013", sl_atr_mult=2.0, trail_enabled=False, exit_policy="sl_plus_trailing_atr"), h4_usd),
        ("me_short", me_short_eval, {"D1": d1}, A1Config(config_id="arc_1019", sl_atr_mult=2.0, trail_enabled=False, exit_policy="sl_partial_close_1r_runner_trail"), d1),
    ]
    names = [c[0] for c in comps]
    full_marks = {n: mark_prices_from_panel(mp) for (n, _s, _p, _c, mp) in comps}

    # per-year IS folds (2011-2020) — the committed non-compounding convention
    folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    decade = Fold(fold_id=99, is_start=folds[0].is_start, is_end=folds[0].is_end,
                  oos_start=folds[0].oos_start, oos_end=folds[-1].oos_end)

    print("scoring components per-year (canonical A1 + MultiPairBacktester)...")
    runners = {n: ArcFoldRunner(A1Architecture(), sig, panels)
               for (n, sig, panels, _c, _mp) in comps}
    cfgs = {n: cfg for (n, _s, _p, cfg, _mp) in comps}
    decade_trades = {n: [] for n in names}
    per_year_roi = {n: {} for n in names}
    for fi, fold in enumerate(folds):
        for n in names:
            fs = runners[n](fold, cfgs[n])
            per_year_roi[n][fold.oos_start.year] = fs.roi_pct * 100.0
            decade_trades[n].extend(_reid(runners[n].last_result.run_result.closed_trades, fi))
    decade_trades = {n: tuple(v) for n, v in decade_trades.items()}

    # reproduction cross-check: per-fold FoldStats ROI vs committed arc-1020 record
    print("\nreproduction cross-check (per-fold ROI % vs arc-1020 record):")
    maxdev = 0.0
    for yr in range(2011, 2021):
        mine = [per_year_roi[n][yr] for n in names]
        rec = REC[yr]
        maxdev = max(maxdev, max(abs(mine[k] - rec[k]) for k in range(4)))
        print(f"  {yr}: " + " ".join(f"{n}={per_year_roi[n][yr]:+.2f}(rec{rec[k]:+.2f})"
                                     for k, n in enumerate(names)))
    print(f"  max |mine - recorded| = {maxdev:.3f} pp")

    # IS-frozen risk-parity weights from the per-year component ROIs
    comp_rois = [[per_year_roi[n][yr] / 100.0 for yr in range(2011, 2021)] for n in names]
    w_rp = fit_weights(comp_rois, mode="risk_parity")
    w_eq = [0.25] * 4
    print("\nrisk-parity weights (frozen IS): " + ", ".join(f"{n}={w:.3f}" for n, w in zip(names, w_rp)))

    def cosim(weights, cap):
        cc = [CoSimComponent(name=n, closed_trades=decade_trades[n],
                             mark_prices=full_marks[n], starting_balance=SB) for n in names]
        return cosim_book_fold(cc, decade, weights=weights, starting_balance=SB, apply_exposure_cap=cap)

    return names, w_eq, w_rp, cosim


def main():  # pragma: no cover - analysis driver
    lo = pd.Timestamp("2011-01-01", tz="UTC")
    hi = pd.Timestamp("2020-12-31 23:59:59", tz="UTC")
    names, w_eq, w_rp, cosim = _build_4way_contiguous()

    print("\n" + "=" * 78)
    print("4-WAY BOOK -- CONTIGUOUS 2011-2020 IS DEPLOYMENT RISK PROFILE")
    print("(peak-to-trough across the decade -- NOT the per-year-reset bookDD)")
    print("=" * 78)
    for label, w in [("equal       ", w_eq), ("risk-parity ", w_rp)]:
        for cap_label, cap in [("cap-OFF (monotone bound)", False), ("cap-ON  (faithful book) ", True)]:
            book = cosim(w, cap)
            # worst single EET-day DD for the 5%-cap headroom
            pdd = book.per_day_max_dd_eet
            worst_day = float(pdd["day_max_dd_base_pct"].max() * 100.0) if len(pdd) else float("nan")
            prof = compute_risk_profile(book.net_equity, lo=lo, hi=hi, worst_day_dd_pct=worst_day)
            print(f"\n[{label}| {cap_label}]  n_dropped={book.n_dropped}")
            print("  " + prof.as_row())
            print(f"  deepest DD: peak {prof.peak_before_trough.date()} -> trough {prof.trough_date.date()}")
            print(f"  daily 5% cap breached: {book.daily_dd_breached}")


if __name__ == "__main__":  # pragma: no cover
    main()
