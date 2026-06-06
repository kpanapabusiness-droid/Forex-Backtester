"""arc 1056 — honest-§5f PER-TRADE (within-fold) bootstrap of the deploy book mean.

Resolves the explicitly-owed thread from arc 1043: arc 1043 killed the path-A
"significant mean-positive" pillar only at n=10 FOLD-bootstrap resolution (honest §5f
exits → t≈1.3, every CI spans zero) and flagged that "a per-TRADE bootstrap
(arc-2016's more-powerful method) MIGHT tighten the honest CI." This computes it on
the HONEST nested-§5f series (NOT a single full-sample-best exit — that was the
Arc-10 trap the first cut of this driver fell into and the reproduction anchor
caught).

Decision-grade for the SINGLE remaining lever (operator path-A gate-governance). The
deploy object is the HONEST 2-leg me_long+fbr book (arc 2044 FLAG F1: gap & me_short
flip mean-negative under honest §5f). Reuses arc 2044's EXACT honest-§5f machinery:
18-cfg registry grid (6 exits × SL{1.5,2.0,2.5}, trail_off) per leg, me_long with the
2-bar time exit attached to the signal; `nested_walk_forward_select(afp_then_mean)`
picks per-fold the cfg best over STRICTLY-EARLIER folds; warmup folds 2011/2012
(<2 priors) are EXCLUDED → 8 honest no-lookahead evaluable folds 2013-2020.

METHOD (CALLS canonical; experiment-side = bootstrap arithmetic only, never realizes
P&L):
 1. Score each leg's 18-cfg grid over the 10 IS folds, capturing each cell's gross
    RunResult (→ per-trade pnl ledger).
 2. nested-select (afp_then_mean) → per-fold the SELECTED exit label; the honest fold
    ROI = that cell's ROI. Anchor: honest 2-leg book mean ≈ +0.36–0.41% (arc 2044).
 3. WITHIN-FOLD per-trade bootstrap of the book MEAN over the 8 evaluable folds: each
    iter resamples each fold's trades (under its SELECTED exit) w/ replacement → fold
    ROI → combine at IS-frozen RP weights → mean over the 8 folds. N=10000, seed 42.
 4. The arc-1023 GUARD (decides whether step-3's tighter CI is legitimate): across-fold
    sd of the 8 observed book fold-ROIs vs the typical within-fold sampling sd.
    across≈within ⇒ folds sampling-dominated ⇒ the per-trade CI is legitimate.
    across≫within ⇒ REAL regime variance ⇒ the per-trade tightening FABRICATES
    significance and the fold-level bootstrap (arc 1043) stands.
 5. Fold-level bootstrap anchor (arc 1043 method) on the same honest 2-leg book.

Single-use diagnostic; reads canonical outputs only (cf. arc 2016 — no new BUILT
tool). OOS NEVER touched.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd

from core.arc.signal_protocol import SignalEvaluation
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.runners._fold_stats_helpers import filter_oos_trades, slice_equity_to_oos
from core.sim.costs.model import CostModel, apply_cost_model
from core.sim.panel import Panel
from core.wfo.folds import build_v3_folds
from discovery.tools.combine_fold_roi import combine_fold_rois, fit_weights
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.month_end_signals import MonthEndReversionLongSignal
from discovery.tools.nested_exit_selection import (
    metric_afp_then_mean,
    nested_walk_forward_select,
)
from discovery.tools.time_exit_predicate import make_time_exit_predicate

HISTDATA_ROOT = Path(os.environ.get("COSIM_HISTDATA_ROOT", r"C:/Users/panap/histdata_backup"))
CACHE_ROOT = Path(os.environ.get("COSIM_CACHE_ROOT", r"C:/Users/panap/Documents/Forex-Backtester/data/cache"))
BOUNDARY = "5ers_eet"
USD = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
EXITS = ["sl_only", "sl_plus_tp_2r", "sl_plus_tp_3r", "sl_plus_trailing_atr",
         "sl_plus_trailing_swing", "sl_partial_close_1r_runner_trail"]
SLS = [1.5, 2.0, 2.5]
SB = 100_000.0
SEED, N_BOOT = 42, 10_000
NAMES = ("me_long", "fbr")


def _load(pairs, tf):
    return Panel.from_pairs(pairs, tf, histdata_root=HISTDATA_ROOT, cache_root=CACHE_ROOT,
                            use_cache=True, boundary_convention=BOUNDARY)


def _attach_te(sig: SignalEvaluation, panel: Panel, n: int) -> SignalEvaluation:
    pred = make_time_exit_predicate({p: panel.pair_dfs[p] for p in panel.pairs}, n)
    return dataclasses.replace(sig, per_pair={
        p: dataclasses.replace(st, exit_predicate=pred) for p, st in sig.per_pair.items()})


def _registry_grid(tag):
    return {f"{ex}|sl{sl}": A1Config(config_id=f"{tag}_{ex}_sl{sl}", sl_atr_mult=sl,
                                     trail_enabled=False, exit_policy=ex)
            for ex in EXITS for sl in SLS}


def main():
    print("ARC 1056 — honest-§5f PER-TRADE bootstrap of the 2-leg me_long+fbr deploy book")
    print(f"histdata_root={HISTDATA_ROOT}\ncache_root={CACHE_ROOT}")
    print("scoring canonical (A1→MultiPairBacktester, FundedNext, risk 0.005 linear); OOS NEVER touched\n")
    d1 = _load(USD, "D1")
    h4_usd = _load(USD, "H4")

    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    yr_of = {f.fold_id: f.oos_start.year for f in is_folds}
    order_ids = sorted(yr_of)                     # fold_id order (chronological)
    order_years = [yr_of[i] for i in order_ids]
    fold_by_id = {f.fold_id: f for f in is_folds}
    warmup_idx = [0, 1]                            # <2 priors → excluded from the honest verdict
    eval_idx = [i for i in range(len(order_years)) if i not in warmup_idx]
    eval_ids = [order_ids[i] for i in eval_idx]
    eval_years = [order_years[i] for i in eval_idx]
    print(f"IS fold years (order): {order_years}")
    print(f"evaluable (honest no-lookahead) folds: {eval_years}\n")

    me_long_sig = _attach_te(
        MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": d1}), d1, 2)
    fbr_sig = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": h4_usd})
    legs = [("me_long", me_long_sig, {"D1": d1}, "melong"),
            ("fbr", fbr_sig, {"H4": h4_usd}, "fbr")]

    # ── score 18-cfg grids, capture per-cell per-fold per-POSITION NET pnl ──
    # Gate fidelity (build_fold_stats_from_run): roi_pct = (net_equity sliced to the
    # OOS window).end/.start − 1. The OOS equity GAIN is the sum of net P&L REALIZED
    # (position EXIT) inside the OOS window, over the OOS-start equity. So the
    # bootstrap unit = per-POSITION net_pnl (gross − cost, apply_cost_model — the
    # canonical chokepoint) of positions whose final_exit_time ∈ OOS, denominator =
    # the OOS-start net equity. (Verified: this reproduces fs.roi_pct to ≤0.02pp;
    # ClosedTrade.pnl is GROSS and a partial+runner is ONE position with non-
    # independent legs → resample positions, not legs.)
    fn_cost = CostModel.fundednext()
    print("Scoring 18-cfg grids over IS folds 2011-2020 (capturing OOS-realized per-position NET pnl)...")
    scored = {n: {} for n in NAMES}                       # name -> label -> [FoldStats by fold order]
    pnl_cell = {n: {} for n in NAMES}                     # name -> label -> {fold_id: (net_pnl_array, denom)}
    for nm, sig, panels, tag in legs:
        print(f"  {nm}...")
        runner = ArcFoldRunner(A1Architecture(), sig, panels)
        for label, cfg in _registry_grid(tag).items():
            fs_list, cellp = [], {}
            for fold in is_folds:
                fs_list.append(runner(fold, cfg))
                rr = runner.last_result.run_result
                costed = apply_cost_model(rr, fn_cost)
                oos_start = pd.Timestamp(fold.oos_start, tz="UTC")
                oos_end = (pd.Timestamp(fold.oos_end, tz="UTC")
                           + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
                bd = costed.breakdown
                if len(bd):
                    ex = pd.to_datetime(bd["final_exit_time"], utc=True)
                    m = (ex >= oos_start) & (ex <= oos_end)
                    net = bd.loc[m, "net_pnl"].to_numpy(dtype=float)
                else:
                    net = np.empty(0, dtype=float)
                oos_eq = slice_equity_to_oos(costed.net_equity, fold)
                denom = float(oos_eq.iloc[0]) if len(oos_eq) else SB
                cellp[fold.fold_id] = (net, denom)
            scored[nm][label] = fs_list
            pnl_cell[nm][label] = cellp

    # ── honest nested §5f series + per-fold selected labels (afp_then_mean) ──
    sel_label = {n: {} for n in NAMES}                    # name -> {fold_id: selected_label}
    honest_full = {n: [] for n in NAMES}                  # name -> per-fold honest ROI (all 10, fold order)
    for n in NAMES:
        res = nested_walk_forward_select(scored[n], selection=metric_afp_then_mean,
                                         selection_name="afp_then_mean", min_prior_folds=2)
        for c in sorted(res.per_fold, key=lambda c: c.fold_id):
            sel_label[n][c.fold_id] = c.selected_label
            honest_full[n].append(c.roi_pct)

    # honest series restricted to the 8 evaluable folds
    honest_eval = {n: [honest_full[n][i] for i in eval_idx] for n in NAMES}
    print("\nhonest §5f nested series (afp_then_mean), evaluable folds 2013-2020:")
    for n in NAMES:
        ser = honest_eval[n]
        print(f"  {n:8s} mean {np.mean(ser)*100:+.3f}%  pos {sum(r>0 for r in ser)}/8  "
              f"per-fold% {[round(r*100,2) for r in ser]}")
        print("           selected exits: " +
              ", ".join(f"{eval_years[k]}:{sel_label[n][eval_ids[k]]}" for k in range(len(eval_idx))))

    # cross-check: per-position-net-pnl-derived ROI of the SELECTED cell == nested series ROI
    # (a residual = the daily-5%-DD / exposure cap, which the uncapped Σnet_pnl misses; at
    #  risk 0.005 it should rarely bind — flag any cell where it does)
    maxdev = 0.0
    for n in NAMES:
        for k, fid in enumerate(eval_ids):
            net, denom = pnl_cell[n][sel_label[n][fid]][fid]
            dev = abs(net.sum() / denom - honest_eval[n][k])
            if dev > 2e-3:
                print(f"    DIAG residual: {n} {eval_years[k]} {sel_label[n][fid]}  "
                      f"Σnet/denom={net.sum()/denom*100:+.3f}%  gate roi={honest_eval[n][k]*100:+.3f}%  "
                      f"diff={dev*100:.3f}pp  n_pos={len(net)}")
            maxdev = max(maxdev, dev)
    print(f"\n  per-position-net vs gate-ROI reproduction: max |diff| = {maxdev:.2e}")

    # ── IS-frozen risk-parity weights on the honest evaluable series ──
    w_rp = fit_weights([honest_eval[n] for n in NAMES], mode="risk_parity")
    w_eq = [0.5, 0.5]
    print("\nIS-frozen RP weights (honest series): " + ", ".join(f"{n}={w:.3f}" for n, w in zip(NAMES, w_rp)))

    for wlabel, w in [("equal      ", w_eq), ("risk-parity", w_rp)]:
        book = np.array(combine_fold_rois([honest_eval[n] for n in NAMES], w).combined_roi)
        pt = float(book.mean()) * 100.0
        across_sd = float(np.std(book, ddof=1)) * 100.0
        worst_i = int(np.argmin(book))
        print(f"\n{'='*78}\n[{wlabel}]  honest 2-leg book point mean {pt:+.4f}%/yr  "
              f"pos {int((book>0).sum())}/8  worst {book.min()*100:+.3f}% ({eval_years[worst_i]})")
        print(f"  across-fold sd (year-to-year variation) = {across_sd:.4f}%")

        # (3) WITHIN-FOLD per-trade bootstrap of the mean (each fold under its selected exit)
        rng = np.random.default_rng(SEED)
        means = np.empty(N_BOOT)
        per_fold_boot = {fid: np.empty(N_BOOT) for fid in eval_ids}
        for b in range(N_BOOT):
            bk = np.empty(len(eval_ids))
            for j, fid in enumerate(eval_ids):
                roi = 0.0
                for i, n in enumerate(NAMES):
                    p, denom = pnl_cell[n][sel_label[n][fid]][fid]
                    r = 0.0 if len(p) == 0 else rng.choice(p, size=len(p), replace=True).sum() / denom
                    roi += w[i] * r
                bk[j] = roi
                per_fold_boot[fid][b] = roi
            means[b] = bk.mean()
        ci = np.percentile(means, [2.5, 97.5]) * 100.0
        p_neg = float((means < 0).mean())
        t_pt = pt / (means.std() * 100.0)
        within_sds = np.array([per_fold_boot[fid].std() for fid in eval_ids]) * 100.0
        within_rms = float(np.sqrt((within_sds ** 2).mean()))
        sig = "SIG+" if ci[0] > 0 else ("SIG-" if ci[1] < 0 else "~0 (spans zero)")
        print(f"  WITHIN-FOLD per-trade bootstrap (N={N_BOOT}): "
              f"95% CI [{ci[0]:+.4f}%, {ci[1]:+.4f}%]  P(mean<0)={p_neg:.4f}  implied t={t_pt:+.3f}  [{sig}]")
        print(f"  GUARD — within-fold sampling sd (RMS) = {within_rms:.4f}%  vs across-fold sd "
              f"{across_sd:.4f}%  → ratio across/within = {across_sd/within_rms:.2f}")

        # (5) fold-level (CLUSTER) bootstrap anchor (arc 1043 method) — the textbook
        #     SE-of-mean for per-year-clustered data; each year's realized ROI already
        #     embeds its own trade-sampling noise.
        rng2 = np.random.default_rng(SEED)
        fmeans = np.array([rng2.choice(book, size=len(book), replace=True).mean() for _ in range(N_BOOT)])
        fci = np.percentile(fmeans, [2.5, 97.5]) * 100.0
        fp_neg = float((fmeans < 0).mean())
        ft = pt / (fmeans.std() * 100.0)
        fsig = "SIG+" if fci[0] > 0 else ("SIG-" if fci[1] < 0 else "~0 (spans zero)")
        print(f"  FOLD-level (cluster) bootstrap (arc-1043 method): "
              f"95% CI [{fci[0]:+.4f}%, {fci[1]:+.4f}%]  P(mean<0)={fp_neg:.4f}  implied t={ft:+.3f}  [{fsig}]")

        # (6) TWO-LEVEL (cluster + trade) bootstrap — the honest TOTAL uncertainty:
        #     resample years w/ replacement, AND resample trades within each drawn year.
        rng3 = np.random.default_rng(SEED)
        tmeans = np.empty(N_BOOT)
        for b in range(N_BOOT):
            drawn = rng3.choice(eval_ids, size=len(eval_ids), replace=True)
            acc = np.empty(len(drawn))
            for j, fid in enumerate(drawn):
                roi = 0.0
                for i, n in enumerate(NAMES):
                    p, denom = pnl_cell[n][sel_label[n][fid]][fid]
                    r = 0.0 if len(p) == 0 else rng3.choice(p, size=len(p), replace=True).sum() / denom
                    roi += w[i] * r
                acc[j] = roi
            tmeans[b] = acc.mean()
        tci = np.percentile(tmeans, [2.5, 97.5]) * 100.0
        tp_neg = float((tmeans < 0).mean())
        tt = pt / (tmeans.std() * 100.0)
        tsig = "SIG+" if tci[0] > 0 else ("SIG-" if tci[1] < 0 else "~0 (spans zero)")
        print(f"  TWO-LEVEL (cluster+trade) bootstrap [TOTAL uncertainty]: "
              f"95% CI [{tci[0]:+.4f}%, {tci[1]:+.4f}%]  P(mean<0)={tp_neg:.4f}  implied t={tt:+.3f}  [{tsig}]")


if __name__ == "__main__":
    main()
