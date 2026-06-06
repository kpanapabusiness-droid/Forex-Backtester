"""arc 2043 — Honest-§5f book-mean SIGNIFICANCE recompute (discharge arc-1042 FLAG F3).

arc 1042 recomputed the 4-way book's deploy MEAN under each component's HONEST §5f
nested-WFO-selected exit (~+0.27% RP vs the committed +0.59%) but left F3 OWED:
"arc-1023/2019's t=2.66 significance pillar likely does not survive honest exits
(estimated t≈1.2); exact recompute owed."  arc 1023 established t=2.66 (p≈0.026),
fold-bootstrap CI [+0.22%,+1.09%], P(mean≤0)=0 — the load-bearing "statistically
mean-positive" pillar for the operator's path-A deploy case — on the COMMITTED-exit
series. This arc applies the EXACT arc-1023 significance battery (across-fold t-stat,
worst-fold z, fold-bootstrap CI seed 123 B=20000, P(mean≤0), binomial sign) to the
HONEST §5f per-fold book series, and to the committed series as a fidelity anchor /
independent reproduction.

Honest reading: the honestly no-lookahead-selected folds are the EVALUABLE (non-warmup)
folds only (the first 2 folds have no in-sample exit choice → nested falls back to the
frozen full-sample pick = NOT honest). So the honest series significance is on the 8
evaluable folds (2013-2020). The committed book is reported on BOTH the full 10 folds
(arc-1023/2019 basis) AND the same 8 evaluable folds (isolates the exit effect from the
n-change).

Scoring 100% canonical (ArcFoldRunner -> A1 -> MultiPairBacktester, FundedNext, default
risk_pct=0.005 = the linear deployable regime; t is risk-INVARIANT there, arc 1024).
Selection via BUILT nested_exit_selection; book combination via BUILT combine_fold_roi.
Significance is pure numpy arithmetic over canonical per-fold numbers (arc-1023 method).
No gate reimplemented. OOS NEVER touched (the book fails IS AFP; §5g operator firewall).

Run:  PYTHONPATH=. py _disco_work/arc2043_honest_book_significance.py
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np

from core.arc.signal_protocol import SignalEvaluation
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.wfo.discovery_measure import run_config_over_folds
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
NAMES = ("gap", "me_long", "fbr", "me_short")
SEED, B = 123, 20000


def _load(pairs, tf):
    return Panel.from_pairs(pairs, tf, histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
                            use_cache=True, boundary_convention=BOUNDARY)


def _attach_time_exit(sig: SignalEvaluation, panel: Panel, n_bars: int) -> SignalEvaluation:
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n_bars)
    return dataclasses.replace(sig, per_pair={
        p: dataclasses.replace(st, exit_predicate=pred) for p, st in sig.per_pair.items()})


def _score_grid(runner, folds, grid):
    return {label: list(run_config_over_folds(runner, folds, cfg)) for label, cfg in grid.items()}


def signif(series_pct, *, seed=SEED, B=B):
    """arc-1023 across-fold significance battery on a per-fold ROI%% vector."""
    a = np.asarray(series_pct, float)
    n = len(a)
    m = a.mean()
    s = a.std(ddof=1)
    se = s / np.sqrt(n)
    t = m / se if se > 0 else float("nan")
    rng = np.random.default_rng(seed)
    boot = np.array([a[rng.integers(0, n, n)].mean() for _ in range(B)])
    ci_lo, ci_hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    p_neg = float((boot <= 0).mean())
    npos = int((a > 0).sum())
    worst_z = a.min() / s if s > 0 else float("nan")
    return dict(n=n, mean=m, sd=s, t=t, ci=(ci_lo, ci_hi), p_neg=p_neg, npos=npos, worst_z=worst_z)


def _print_signif(tag, d):
    # two-sided 0.05 t-crit by df
    tcrit = {7: 2.365, 8: 2.306, 9: 2.262}.get(d["n"] - 1, 2.26)
    sig = "SIG+" if (d["t"] > tcrit and d["ci"][0] > 0) else ("~0" if d["ci"][0] <= 0 <= d["ci"][1] else "?")
    print(f"  {tag:<46} n={d['n']:2d} mean={d['mean']:+.3f}% sd={d['sd']:.3f}% "
          f"t={d['t']:+.2f} (crit {tcrit:.2f}) CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] "
          f"P(<=0)={d['p_neg']:.3f} pos={d['npos']}/{d['n']} worstZ={d['worst_z']:+.2f}  [{sig}]")


def main():
    print(f"histdata_root={HISTDATA_ROOT}\ncache_root={CACHE_ROOT}\nrisk_pct=A1 default 0.005 (linear regime; t risk-invariant, arc 1024)\n")
    h4_usd = _load(USD_MAJORS, "H4")
    h4_jpy = _load(JPY_CROSSES, "H4")
    d1 = _load(USD_MAJORS, "D1")

    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    yr_of = {f.fold_id: f.oos_start.year for f in is_folds}
    order_years = [yr_of[fid] for fid in sorted(yr_of)]
    warmup_idx = [0, 1]                      # first 2 folds = nested warmup (no honest choice)
    eval_idx = [i for i in range(len(order_years)) if i not in warmup_idx]
    print(f"IS fold years (order): {order_years}")
    print(f"warmup folds (excluded from honest verdict): {[order_years[i] for i in warmup_idx]}")
    print(f"evaluable (honest, no-lookahead) folds: {[order_years[i] for i in eval_idx]}\n")

    # ── committed component signal evaluations ───────────────────────────
    gap_sig = WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36).evaluate({"H4": h4_jpy})
    me_long_base = MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1})
    fbr_sig = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": h4_usd})
    me_short_sig = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1})
    me_long_sig = _attach_time_exit(me_long_base, d1, 2)

    runners = {
        "gap": ArcFoldRunner(A1Architecture(), gap_sig, {"H4": h4_jpy}),
        "me_long": ArcFoldRunner(A1Architecture(), me_long_sig, {"D1": d1}),
        "fbr": ArcFoldRunner(A1Architecture(), fbr_sig, {"H4": h4_usd}),
        "me_short": ArcFoldRunner(A1Architecture(), me_short_sig, {"D1": d1}),
    }

    def registry_grid(tag):
        return {f"{ex}|sl{sl}": A1Config(config_id=f"{tag}_{ex}_sl{sl}", sl_atr_mult=sl,
                                         trail_enabled=False, exit_policy=ex)
                for ex in EXITS for sl in SLS}

    committed_label = {"gap": "timeexit24|sl2.0", "me_long": "sl_only|sl2.0",
                       "fbr": "sl_plus_trailing_atr|sl2.0",
                       "me_short": "sl_partial_close_1r_runner_trail|sl2.0"}

    # ── score grids over IS folds (canonical) ────────────────────────────
    print("Scoring grids over IS folds 2011-2020 (canonical A1 + MultiPairBacktester)...")
    scored = {}
    print("  gap (pure-time horizon x SL)...")
    scored["gap"] = {}
    for hz in GAP_HORIZONS:
        gsig = _attach_time_exit(gap_sig, h4_jpy, hz)
        grunner = ArcFoldRunner(A1Architecture(), gsig, {"H4": h4_jpy})
        for sl in SLS:
            lbl = f"timeexit{hz}|sl{sl}"
            cfg = A1Config(config_id=f"gap_t{hz}_sl{sl}", sl_atr_mult=sl, trail_enabled=False, exit_policy=None)
            scored["gap"][lbl] = list(run_config_over_folds(grunner, is_folds, cfg))
    print("  me_long...")
    scored["me_long"] = _score_grid(runners["me_long"], is_folds, registry_grid("melong"))
    print("  fbr...")
    scored["fbr"] = _score_grid(runners["fbr"], is_folds, registry_grid("fbr"))
    print("  me_short...")
    scored["me_short"] = _score_grid(runners["me_short"], is_folds, registry_grid("meshort"))

    # ── committed per-fold series (anchor) ───────────────────────────────
    committed_series = {n: [fs.roi_pct for fs in
                            sorted(scored[n][committed_label[n]], key=lambda fs: fs.fold_id)]
                        for n in NAMES}
    print("\n=== REPRODUCTION ANCHOR (committed per-fold means; cf arc 1042) ===")
    for n in NAMES:
        mean = np.mean(committed_series[n]) * 100
        print(f"  {n:<9} committed meanIS={mean:+.3f}%  per-fold%={[round(r*100,2) for r in committed_series[n]]}")

    # ── nested honest series per metric ──────────────────────────────────
    honest_series = {m: {} for m, _ in METRICS}
    print("\n=== HONEST §5f nested per-fold series (full 10; warmup = first 2) ===")
    for n in NAMES:
        for mname, mfn in METRICS:
            res = nested_walk_forward_select(scored[n], selection=mfn, selection_name=mname, min_prior_folds=2)
            honest_series[mname][n] = [c.roi_pct for c in sorted(res.per_fold, key=lambda c: c.fold_id)]

    # ── book combination helper ──────────────────────────────────────────
    def book_series_pct(series_by_name, weights):
        rois = combine_fold_rois([series_by_name[n] for n in NAMES], weights).combined_roi
        return [r * 100 for r in rois]

    w_committed_rp = fit_weights([committed_series[n] for n in NAMES], mode="risk_parity")
    w_equal = [0.25, 0.25, 0.25, 0.25]
    print("\ncommitted RP weights (frozen IS): " +
          ", ".join(f"{n}={w:.3f}" for n, w in zip(NAMES, w_committed_rp)))

    # =====================================================================
    # PART 1 — COMMITTED book significance (FIDELITY ANCHOR: reproduce arc 1023/2019 t≈2.66)
    # =====================================================================
    print("\n" + "=" * 100)
    print("PART 1 — COMMITTED-exit book significance (anchor: arc 1023 t=2.66 / arc 2019 RP CI [+0.12,+1.09], P(<0)=0.004)")
    print("=" * 100)
    cb_rp = book_series_pct(committed_series, w_committed_rp)
    cb_eq = book_series_pct(committed_series, w_equal)
    # arc-2016 hand weights {gap 0, me_long .65, fbr .2, me_short .15} — the exact arc-1023 book
    w_2016 = [0.0, 0.65, 0.20, 0.15]
    cb_2016 = book_series_pct(committed_series, w_2016)
    print("  committed book per-fold%% (RP):", [round(x, 2) for x in cb_rp])
    print(f"  full 10 folds:")
    _print_signif("committed RP (n=10, arc-2019 basis)", signif(cb_rp))
    _print_signif("committed EQUAL (n=10)", signif(cb_eq))
    _print_signif("committed arc-2016 wts (n=10, arc-1023 basis)", signif(cb_2016))
    print(f"  same 8 EVALUABLE folds (apples-to-apples vs honest):")
    _print_signif("committed RP (n=8 eval)", signif([cb_rp[i] for i in eval_idx]))
    _print_signif("committed arc-2016 wts (n=8 eval)", signif([cb_2016[i] for i in eval_idx]))

    # =====================================================================
    # PART 2 — HONEST §5f book significance (the F3 recompute) — EVALUABLE folds
    # =====================================================================
    print("\n" + "=" * 100)
    print("PART 2 — HONEST §5f book significance (F3 recompute) — 8 EVALUABLE folds (2013-2020), the no-lookahead series")
    print("=" * 100)
    for mname, _ in METRICS:
        hs = honest_series[mname]
        w_h_rp = fit_weights([hs[n] for n in NAMES], mode="risk_parity")
        for wlabel, w in (("RP commit-wts", w_committed_rp), ("RP honest-refit", w_h_rp), ("EQUAL", w_equal)):
            full = book_series_pct(hs, w)
            ev = [full[i] for i in eval_idx]
            _print_signif(f"HONEST[{mname}] {wlabel}", signif(ev))
        print()

    # =====================================================================
    # PART 3 — per-component HONEST mean significance (which leg carries it)
    # =====================================================================
    print("=" * 100)
    print("PART 3 — per-component HONEST §5f mean significance (8 evaluable folds) — which leg carries the book's mean")
    print("=" * 100)
    for mname, _ in METRICS:
        print(f"  metric={mname}:")
        for n in NAMES:
            _print_signif(f"    {n}", signif([honest_series[mname][n][i] * 100 for i in eval_idx]))
    print("\n  (committed per-component, 8 eval folds, for contrast):")
    for n in NAMES:
        _print_signif(f"    {n} committed", signif([committed_series[n][i] * 100 for i in eval_idx]))


if __name__ == "__main__":
    main()
