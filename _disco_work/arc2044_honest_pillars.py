"""arc 2044 — Honest-§5f re-derivation of the path-A robustness pillars (discharge arc-1043 FLAG F2).

arc 2043 (+ 1043) resolved arc-1042 F3: the book's t=2.66 mean-significance pillar
does NOT survive honest §5f exits (honest t<=0.96, all CIs span zero). arc 1043 then
raised F2, explicitly owed:

  "arcs 2021 (temporal-stability), 3022 (cost-robustness kappa=3.32), 2019 (ENB /
   diversification, P(mean<0)=0.004) were ALL computed on the COMMITTED-exit series
   -> suspect by the same exit-optimism mechanism; honest-exit re-run of the
   characterization suite owed before path-A leans on them."

This arc discharges F2: it re-derives the THREE remaining path-A pillars on the HONEST
§5f per-component series, with the COMMITTED series as a fidelity anchor (reproduce
ENB 3.32 [2019], LATE-half +0.404% [2021], break-even kappa 3.32 / kappa=0 2-of-10-neg
[3022]).

  PART A — DIVERSIFICATION / ENB (arc 2019): per-fold component-ROI correlation matrix,
           ENB = participation ratio (sum(lam))^2/sum(lam^2), top-eigenvalue share,
           bootstrap ENB CI, tail co-movement (worst-5 vs best-5 folds), co-negativity.
           PLUS the decisive honest test: does the 4-leg book beat its 2 honest legs
           (me_long+fbr)? — i.e. are gap+me_short pure drag once exit-honest?
  PART B — TEMPORAL STABILITY (arc 2021): EARLY/LATE half book means + decay bootstrap.
  PART C — COST ROBUSTNESS (arc 3022): re-net each honest-selected exit's gross RunResult
           at a kappa sweep -> honest book mean(kappa) -> break-even kappa; n_neg at kappa=0.

Scoring 100% canonical (ArcFoldRunner -> A1 -> MultiPairBacktester, FundedNext, A1
default risk_pct=0.005 = the linear deployable regime; ratios/t risk-invariant, arc 1024).
Selection via BUILT nested_exit_selection (2040); book combination via BUILT combine_fold_roi
(2006); cost re-net via BUILT scaled_fundednext (3022) through the CANONICAL chokepoint
build_fold_stats_from_run. ENB/temporal arithmetic is inline numpy over canonical per-fold
numbers (no new BUILT tool, as arc 2043's significance battery). No gate reimplemented.
OOS NEVER touched (book fails IS AFP; §5g operator firewall).

Run:  PYTHONPATH=. py _disco_work/arc2044_honest_pillars.py
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np

from core.arc.signal_protocol import SignalEvaluation
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners._fold_stats_helpers import build_fold_stats_from_run
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.panel import Panel
from core.wfo.folds import build_v3_folds
from discovery.tools.combine_fold_roi import combine_fold_rois, fit_weights
from discovery.tools.cost_scaling import scaled_fundednext
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
START_BAL = 100_000.0
SEED, B = 123, 20000
KAPPAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


def _load(pairs, tf):
    return Panel.from_pairs(pairs, tf, histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
                            use_cache=True, boundary_convention=BOUNDARY)


def _attach_time_exit(sig: SignalEvaluation, panel: Panel, n_bars: int) -> SignalEvaluation:
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n_bars)
    return dataclasses.replace(sig, per_pair={
        p: dataclasses.replace(st, exit_predicate=pred) for p, st in sig.per_pair.items()})


def _score_grid_capture(runner, folds, grid):
    """Score every (label, fold) AND capture the gross RunResult per cell.

    Returns (scored, runs):
      scored[label] = [FoldStats...]   (canonical net @ FundedNext, kappa=1)
      runs[label][fold_id] = RunResult (for kappa-re-net via build_fold_stats_from_run)
    """
    scored, runs = {}, {}
    for label, cfg in grid.items():
        fs_list, rr_map = [], {}
        for fold in folds:
            fs = runner(fold, cfg)
            fs_list.append(fs)
            rr_map[fold.fold_id] = runner.last_result.run_result
        scored[label] = fs_list
        runs[label] = rr_map
    return scored, runs


def _renet_roi(run_result, fold, kappa):
    """ROI%% (decimal) of one RunResult re-netted at kappa x FundedNext via the canonical chokepoint."""
    fs = build_fold_stats_from_run(fold=fold, run_result=run_result,
                                   starting_balance=START_BAL, cost_model=scaled_fundednext(kappa))
    return fs.roi_pct


# ───────────────────────── significance / stat helpers (arc-1023 method) ─────────────
def signif(series_pct, *, seed=SEED, B=B):
    a = np.asarray(series_pct, float)
    n = len(a)
    m = a.mean()
    s = a.std(ddof=1)
    se = s / np.sqrt(n)
    t = m / se if se > 0 else float("nan")
    rng = np.random.default_rng(seed)
    boot = np.array([a[rng.integers(0, n, n)].mean() for _ in range(B)])
    return dict(n=n, mean=m, sd=s, t=t, ci=(np.percentile(boot, 2.5), np.percentile(boot, 97.5)),
                p_neg=float((boot <= 0).mean()), npos=int((a > 0).sum()),
                worst_z=(a.min() / s if s > 0 else float("nan")))


def _ps(tag, d):
    sigflag = "~0" if d["ci"][0] <= 0 <= d["ci"][1] else ("SIG+" if d["ci"][0] > 0 else "SIG-")
    print(f"  {tag:<42} n={d['n']:2d} mean={d['mean']:+.3f}% sd={d['sd']:.3f}% t={d['t']:+.2f} "
          f"CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] P(<=0)={d['p_neg']:.3f} pos={d['npos']}/{d['n']}  [{sigflag}]")


def enb_block(comp_series, names, seed=SEED, B=4000):
    """Diversification / ENB on a (len(names) x n_folds) per-fold ROI matrix (arc-2019 method)."""
    M = np.array([np.asarray(comp_series[n], float) for n in names])  # k x n
    C = np.corrcoef(M)
    eig = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
    enb = (eig.sum() ** 2) / (eig ** 2).sum()
    top_share = eig.max() / eig.sum()
    # bootstrap ENB over resampled folds
    n = M.shape[1]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        Mb = M[:, idx]
        if np.any(Mb.std(axis=1) < 1e-12):
            continue
        e = np.clip(np.linalg.eigvalsh(np.corrcoef(Mb)), 1e-12, None)
        boots.append((e.sum() ** 2) / (e ** 2).sum())
    boots = np.array(boots)
    ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if len(boots) else (np.nan, np.nan)
    return dict(C=C, enb=enb, top_share=top_share, enb_ci=ci, k=len(names))


def mean_pairwise_corr(comp_series, names, fold_idx):
    """Mean of the k-choose-2 pairwise correlations of components over a fold SUBSET."""
    if len(fold_idx) < 3:
        return float("nan")
    M = np.array([np.asarray(comp_series[n], float)[fold_idx] for n in names])
    C = np.corrcoef(M)
    iu = np.triu_indices(len(names), 1)
    vals = C[iu]
    return float(np.nanmean(vals))


def main():
    print(f"ARC 2044 — honest-§5f re-derivation of path-A pillars (ENB 2019 / temporal 2021 / cost-kappa 3022)")
    print(f"histdata_root={HISTDATA_ROOT}\ncache_root={CACHE_ROOT}")
    print("risk_pct=A1 default 0.005 (linear regime); scoring canonical; OOS NEVER touched\n")
    h4_usd = _load(USD_MAJORS, "H4")
    h4_jpy = _load(JPY_CROSSES, "H4")
    d1 = _load(USD_MAJORS, "D1")

    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    fold_by_id = {f.fold_id: f for f in is_folds}
    yr_of = {f.fold_id: f.oos_start.year for f in is_folds}
    order_ids = sorted(yr_of)
    order_years = [yr_of[i] for i in order_ids]
    warmup_idx = [0, 1]
    eval_idx = [i for i in range(len(order_years)) if i not in warmup_idx]
    print(f"IS fold years (order): {order_years}")
    print(f"evaluable (honest no-lookahead) folds: {[order_years[i] for i in eval_idx]}\n")

    # ── committed component signals ──────────────────────────────────────
    gap_sig = WeekendGapFillLongSignal(threshold_atr=0.5, gap_hours=36).evaluate({"H4": h4_jpy})
    me_long_base = MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1})
    fbr_sig = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": h4_usd})
    me_short_sig = MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1})
    me_long_sig = _attach_time_exit(me_long_base, d1, 2)

    def registry_grid(tag):
        return {f"{ex}|sl{sl}": A1Config(config_id=f"{tag}_{ex}_sl{sl}", sl_atr_mult=sl,
                                         trail_enabled=False, exit_policy=ex)
                for ex in EXITS for sl in SLS}

    committed_label = {"gap": "timeexit24|sl2.0", "me_long": "sl_only|sl2.0",
                       "fbr": "sl_plus_trailing_atr|sl2.0",
                       "me_short": "sl_partial_close_1r_runner_trail|sl2.0"}

    # ── score grids over IS folds (canonical) + capture RunResults ───────
    print("Scoring grids over IS folds 2011-2020 (canonical; capturing gross RunResults for cost re-net)...")
    scored, runs = {}, {}
    print("  gap (pure-time horizon x SL)...")
    scored["gap"], runs["gap"] = {}, {}
    for hz in GAP_HORIZONS:
        gsig = _attach_time_exit(gap_sig, h4_jpy, hz)
        grunner = ArcFoldRunner(A1Architecture(), gsig, {"H4": h4_jpy})
        for sl in SLS:
            lbl = f"timeexit{hz}|sl{sl}"
            cfg = A1Config(config_id=f"gap_t{hz}_sl{sl}", sl_atr_mult=sl, trail_enabled=False, exit_policy=None)
            fs_list, rr_map = [], {}
            for fold in is_folds:
                fs_list.append(grunner(fold, cfg))
                rr_map[fold.fold_id] = grunner.last_result.run_result
            scored["gap"][lbl] = fs_list
            runs["gap"][lbl] = rr_map
    for nm, sig, panels, tag in (
        ("me_long", me_long_sig, {"D1": d1}, "melong"),
        ("fbr", fbr_sig, {"H4": h4_usd}, "fbr"),
        ("me_short", me_short_sig, {"D1": d1}, "meshort"),
    ):
        print(f"  {nm}...")
        r = ArcFoldRunner(A1Architecture(), sig, panels)
        scored[nm], runs[nm] = _score_grid_capture(r, is_folds, registry_grid(tag))

    # committed per-fold series (anchor)
    committed_series = {n: [fs.roi_pct for fs in sorted(scored[n][committed_label[n]], key=lambda f: f.fold_id)]
                        for n in NAMES}
    print("\n=== REPRODUCTION ANCHOR (committed per-fold means; cf arc 1042/2043) ===")
    for n in NAMES:
        print(f"  {n:<9} committed meanIS={np.mean(committed_series[n])*100:+.3f}%  "
              f"per-fold%={[round(r*100,2) for r in committed_series[n]]}")

    # honest nested series + per-fold selected labels (per metric)
    honest_series = {m: {} for m, _ in METRICS}
    honest_labels = {m: {} for m, _ in METRICS}  # [m][name] -> {fold_id: selected_label}
    for n in NAMES:
        for mname, mfn in METRICS:
            res = nested_walk_forward_select(scored[n], selection=mfn, selection_name=mname, min_prior_folds=2)
            pf = sorted(res.per_fold, key=lambda c: c.fold_id)
            honest_series[mname][n] = [c.roi_pct for c in pf]
            honest_labels[mname][n] = {c.fold_id: c.selected_label for c in pf}

    def book_pct(series_by_name, weights):
        return [r * 100 for r in combine_fold_rois([series_by_name[n] for n in NAMES], weights).combined_roi]

    w_committed_rp = fit_weights([committed_series[n] for n in NAMES], mode="risk_parity")
    w_equal = [0.25, 0.25, 0.25, 0.25]
    print("\ncommitted RP weights (frozen IS): " + ", ".join(f"{n}={w:.3f}" for n, w in zip(NAMES, w_committed_rp)))

    PRIMARY = "afp_then_mean"  # gate-aligned conservative metric (mean_roi grabs the trailing_swing §5f trap)

    # =====================================================================
    # PART A — DIVERSIFICATION / ENB (arc 2019 honest re-derivation)
    # =====================================================================
    print("\n" + "=" * 100)
    print("PART A — DIVERSIFICATION / ENB  (anchor: arc 2019 ENB=3.32, top-eig 38.9%, CI[2.02,3.33])")
    print("=" * 100)

    print("\n[A0] per-component HONEST mean SIGN (8 evaluable folds) — does each leg keep positive expectancy?")
    for n in NAMES:
        d = signif([honest_series[PRIMARY][n][i] * 100 for i in eval_idx])
        c = signif([committed_series[n][i] * 100 for i in eval_idx])
        print(f"  {n:<9} committed {c['mean']:+.3f}%   honest[{PRIMARY}] {d['mean']:+.3f}%  "
              f"({'POSITIVE' if d['mean']>0 else 'NEGATIVE'} honest-mean)")

    print("\n[A1] ENB — committed series (anchor) vs honest series")
    eb_c10 = enb_block(committed_series, NAMES)
    print(f"  committed (n=10): ENB={eb_c10['enb']:.2f}/4  top-eig share={eb_c10['top_share']*100:.1f}%  "
          f"ENB-CI[{eb_c10['enb_ci'][0]:.2f},{eb_c10['enb_ci'][1]:.2f}]")
    comm_eval = {n: [committed_series[n][i] for i in eval_idx] for n in NAMES}
    eb_c8 = enb_block(comm_eval, NAMES)
    print(f"  committed (n=8 eval): ENB={eb_c8['enb']:.2f}/4  top-eig {eb_c8['top_share']*100:.1f}%  "
          f"CI[{eb_c8['enb_ci'][0]:.2f},{eb_c8['enb_ci'][1]:.2f}]")
    for mname, _ in METRICS:
        hs_eval = {n: [honest_series[mname][n][i] for i in eval_idx] for n in NAMES}
        eb = enb_block(hs_eval, NAMES)
        print(f"  honest[{mname:<14}] (n=8 eval): ENB={eb['enb']:.2f}/4  top-eig {eb['top_share']*100:.1f}%  "
              f"CI[{eb['enb_ci'][0]:.2f},{eb['enb_ci'][1]:.2f}]")

    print("\n[A2] pairwise component correlation matrix (honest primary, 8 eval folds)")
    hsp = {n: [honest_series[PRIMARY][n][i] for i in eval_idx] for n in NAMES}
    Cm = enb_block(hsp, NAMES)["C"]
    print("         " + "".join(f"{n:>9}" for n in NAMES))
    for i, n in enumerate(NAMES):
        print(f"  {n:<7}" + "".join(f"{Cm[i,j]:+9.2f}" for j in range(len(NAMES))))

    print("\n[A3] tail co-movement + co-negativity (honest primary RP book, 8 eval folds)")
    book_h = np.array(book_pct(hsp, w_committed_rp))  # already eval-sliced via hsp
    order = np.argsort(book_h)
    worst5, best5 = order[:5], order[-5:]
    print(f"  book worst-5 folds mean pairwise comp-corr = {mean_pairwise_corr(hsp, NAMES, worst5):+.3f}")
    print(f"  book best-5  folds mean pairwise comp-corr = {mean_pairwise_corr(hsp, NAMES, best5):+.3f}")
    coneg = [sum(1 for n in NAMES if hsp[n][i] < 0) for i in range(len(eval_idx))]
    print(f"  co-negativity per fold (# of 4 legs <0): {coneg}  max={max(coneg)}")

    print("\n[A4] DECISIVE — does the 4-leg honest book beat its 2 HONEST legs (me_long+fbr)?")
    print("     (if gap+me_short are mean-neg under honest exits, diversifying INTO them should DRAG)")
    honest2 = ("me_long", "fbr")
    for mname, _ in METRICS:
        hs_eval = {n: [honest_series[mname][n][i] for i in eval_idx] for n in NAMES}
        # 4-leg RP book
        w4 = fit_weights([hs_eval[n] for n in NAMES], mode="risk_parity")
        b4 = book_pct(hs_eval, w4)
        # 2-leg RP book (me_long+fbr only) via combine on the 2 series
        s2 = [np.asarray(hs_eval[n], float) for n in honest2]
        w2 = fit_weights(s2, mode="risk_parity")
        b2 = [r * 100 for r in combine_fold_rois(s2, w2).combined_roi]
        d4, d2 = signif(b4), signif(b2)
        print(f"  metric={mname:<14} 4-leg RP mean={d4['mean']:+.3f}% worst={min(b4):+.3f}% pos={d4['npos']}/8  | "
              f"2-leg(me_long+fbr) RP mean={d2['mean']:+.3f}% worst={min(b2):+.3f}% pos={d2['npos']}/8")

    # =====================================================================
    # PART B — TEMPORAL STABILITY (arc 2021 honest re-derivation)
    # =====================================================================
    print("\n" + "=" * 100)
    print("PART B — TEMPORAL STABILITY  (anchor: arc 2021 LATE-half RP +0.404%, P(<0)=0.060)")
    print("=" * 100)

    def half_split(book_series, years):
        yr = np.asarray(years)
        early_mask = yr <= np.median(yr)
        return np.asarray(book_series)[early_mask], np.asarray(book_series)[~early_mask]

    def temporal_report(tag, book_series, years):
        e, l = half_split(book_series, years)
        de, dl = signif(e), signif(l)
        # decay bootstrap (early_mean - late_mean)
        rng = np.random.default_rng(SEED)
        boot = np.array([e[rng.integers(0, len(e), len(e))].mean() - l[rng.integers(0, len(l), len(l))].mean()
                         for _ in range(B)])
        print(f"  {tag}")
        print(f"     EARLY mean={de['mean']:+.3f}% (n={de['n']}, P(<=0)={de['p_neg']:.3f})   "
              f"LATE mean={dl['mean']:+.3f}% (n={dl['n']}, P(<=0)={dl['p_neg']:.3f})")
        print(f"     decay(early-late)={boot.mean():+.3f}% CI[{np.percentile(boot,2.5):+.3f},{np.percentile(boot,97.5):+.3f}] "
              f"P(decay>0)={float((boot>0).mean()):.2f}")

    # committed anchor (full 10)
    cb_rp10 = book_pct(committed_series, w_committed_rp)
    temporal_report("committed RP (n=10, anchor cf arc 2021)", cb_rp10, order_years)
    # honest (8 eval) RP book, primary + all metrics
    eval_years = [order_years[i] for i in eval_idx]
    for mname, _ in METRICS:
        hs_eval = {n: [honest_series[mname][n][i] for i in eval_idx] for n in NAMES}
        wh = fit_weights([hs_eval[n] for n in NAMES], mode="risk_parity")
        temporal_report(f"honest[{mname}] RP (n=8 eval)", book_pct(hs_eval, wh), eval_years)

    # =====================================================================
    # PART C — COST ROBUSTNESS kappa (arc 3022 honest re-derivation)
    # =====================================================================
    print("\n" + "=" * 100)
    print("PART C — COST ROBUSTNESS kappa  (anchor: arc 3022 break-even kappa=3.32, kappa=0 -> 2/10 neg)")
    print("=" * 100)

    def book_mean_at_kappa(label_map, weights, fold_index_set):
        """Honest book per-fold ROI at cost multiplier kappa, then over kappas -> means + n_neg."""
        out = {}
        for kp in KAPPAS:
            comp_k = {}
            for n in NAMES:
                series = []
                for i in order_ids:  # fold_id ascending
                    lbl = label_map[n][i]
                    rr = runs[n][lbl][i]
                    series.append(_renet_roi(rr, fold_by_id[i], kp))
                comp_k[n] = series
            book = combine_fold_rois([comp_k[n] for n in NAMES], weights).combined_roi
            book = [book[idx] * 100 for idx in range(len(book))]
            sub = [book[idx] for idx in fold_index_set]
            out[kp] = (float(np.mean(sub)), sum(1 for r in sub if r < 0), len(sub))
        return out

    def breakeven_kappa(kappa_means):
        ks = sorted(kappa_means)
        prev_k, prev_m = ks[0], kappa_means[ks[0]][0]
        for k in ks[1:]:
            m = kappa_means[k][0]
            if prev_m > 0 >= m:
                # linear interpolate crossing
                frac = prev_m / (prev_m - m)
                return prev_k + frac * (k - prev_k)
            prev_k, prev_m = k, m
        return float("inf") if prev_m > 0 else 0.0

    # committed anchor: committed-label map (same exit every fold) on full 10 folds, RP weights
    committed_map = {n: {i: committed_label[n] for i in order_ids} for n in NAMES}
    full_idx = list(range(len(order_ids)))
    km_c = book_mean_at_kappa(committed_map, w_committed_rp, full_idx)
    print("\n[C-anchor] committed RP book (n=10) cost sweep:")
    for kp in KAPPAS:
        m, nn, tot = km_c[kp]
        print(f"    kappa={kp:<4} book mean={m:+.4f}%  n_neg={nn}/{tot}")
    print(f"    >> committed break-even kappa = {breakeven_kappa(km_c):.2f}   (anchor target ~3.32; kappa=0 n_neg target 2/10)")

    # honest: per-metric honest-selected per-fold labels; book over the 8 EVAL folds (honest series)
    eval_idset = eval_idx  # positions in order_ids that are evaluable
    for mname, _ in METRICS:
        # honest RP weights from the kappa=1 honest series (eval folds)
        hs_eval = {n: [honest_series[mname][n][i] for i in eval_idx] for n in NAMES}
        wh = fit_weights([hs_eval[n] for n in NAMES], mode="risk_parity")
        km_h = book_mean_at_kappa(honest_labels[mname], wh, eval_idset)
        print(f"\n[C-honest {mname}] RP book (n=8 eval) cost sweep:")
        for kp in KAPPAS:
            m, nn, tot = km_h[kp]
            print(f"    kappa={kp:<4} book mean={m:+.4f}%  n_neg={nn}/{tot}")
        print(f"    >> honest[{mname}] break-even kappa = {breakeven_kappa(km_h):.2f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
