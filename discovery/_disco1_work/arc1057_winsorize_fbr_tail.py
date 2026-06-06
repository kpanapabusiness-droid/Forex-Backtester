"""arc 1057 — TAIL-DEPENDENCE of the honest 2-leg deploy book mean (winsorize the fbr runner tail).

Resolves the thread arc 1056 explicitly owed (and the chat-1000s handoff pointer):
arc 1056 found the honest 2-leg me_long+fbr deploy mean (+0.41%/yr) is only BORDERLINE
significant (cluster bootstrap RP t=2.27 SIG+ / equal t=1.91 ~0) and that within-year
sampling sd (1.08% RP) EXCEEDS the across-year spread (0.55%) → the per-year ROIs are
fat-tail-(fbr-runner)-dominated, so the apparent significance "rests on a lucky tight
clustering of noisy annual numbers." The natural falsifier it named: cap the fat tail
and see whether the mean / significance survives.

QUESTION: how much of the honest 2-leg deploy mean and its borderline significance is
carried by a HANDFUL of un-repeatable fat-tail (fbr runner) positions? If winsorizing
the top few positions collapses the mean / t toward zero → the deploy edge is a
fat-tail mirage (a robustness KILL for the operator's path-A "significant mean" pillar);
if it survives → the edge is thicker than the within>across sd ratio feared.

because: a deploy mean carried by a few un-repeatable fat-tailed runner trades is not a
robustly-deployable edge (you cannot size to a mean that is one lucky +8R fbr-2024
runner away from zero). Winsorization (cap above a percentile) + leave-top-N are the
standard tail-robustness diagnostics; arc 1056 flagged the within>across-sd as the
fragility signal this arc quantifies directly.

METHOD (CALLS canonical; experiment side = winsorization + bootstrap ARITHMETIC only,
never realizes P&L, never touches the gate, OOS NEVER touched):
 1. Reproduce arc 1056 EXACTLY: score each leg's 18-cfg §5f registry grid over the 10
    IS folds (A1->MultiPairBacktester, FundedNext, risk 0.005), capture per-cell per-
    fold per-POSITION NET pnl (gross - apply_cost_model cost), nested_select
    (afp_then_mean) -> per-fold selected exit, 8 evaluable no-lookahead folds 2013-2020.
    Anchor: me_long +0.203%, fbr +1.001%; honest 2-leg book RP +0.411%/yr.
 2. TAIL CONCENTRATION: pool each leg's positions across the 8 evaluable folds (under
    their selected exits); report what % of the leg's total NET P&L comes from its
    top-1 / top-3 / top-5 / top-1% positions (the tail-dependence measure).
 3. WINSORIZE: cap each position's net P&L at the q-th percentile of its own leg's pooled
    positive net-P&L (q in {99,97.5,95,90}); recompute each fold's ROI -> recombine at the
    SAME IS-frozen RP + equal weights -> recompute book mean, folds-positive, worst fold,
    and the CLUSTER bootstrap t/CI (the correct SE-of-mean, arc 1056). Track survival.
 4. LEAVE-TOP-N: remove the single largest 1/2/3/5 positions of the whole book (by net
    P&L, across both legs) -> recompute the same. (The discrete complement to winsorize.)
 5. Per-leg attribution: redo (3) winsorizing ONLY fbr vs ONLY me_long, to confirm the
    tail dependence is the fbr runner (the hypothesis) not me_long.

Single-use diagnostic; reads canonical outputs only (cf. arc 2016/1056 — no new BUILT
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
from core.runners._fold_stats_helpers import slice_equity_to_oos
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


def _cluster_boot(book, pt, seed=SEED, n=N_BOOT):
    """Fold-level (cluster) bootstrap — the correct SE-of-mean for per-year-clustered data (arc 1056)."""
    rng = np.random.default_rng(seed)
    fmeans = np.array([rng.choice(book, size=len(book), replace=True).mean() for _ in range(n)])
    ci = np.percentile(fmeans, [2.5, 97.5]) * 100.0
    p_neg = float((fmeans < 0).mean())
    t = pt / (fmeans.std() * 100.0) if fmeans.std() > 0 else float("nan")
    sig = "SIG+" if ci[0] > 0 else ("SIG-" if ci[1] < 0 else "~0")
    return ci, p_neg, t, sig


def _book_from_cells(pnl_by_leg, sel_label, eval_ids, weights, transform=None):
    """Build the per-fold book ROI series from per-position net pnl, optional transform(name, pnl_array)."""
    per_leg = []
    for i, n in enumerate(NAMES):
        ser = []
        for fid in eval_ids:
            p, denom = pnl_by_leg[n][sel_label[n][fid]][fid]
            p = transform(n, p) if transform is not None else p
            ser.append(0.0 if len(p) == 0 else float(p.sum()) / denom)
        per_leg.append(ser)
    return np.array(combine_fold_rois(per_leg, weights).combined_roi)


def main():
    print("ARC 1057 — TAIL-DEPENDENCE of the honest 2-leg me_long+fbr deploy book mean (winsorize fbr tail)")
    print(f"histdata_root={HISTDATA_ROOT}\ncache_root={CACHE_ROOT}")
    print("scoring canonical (A1->MultiPairBacktester, FundedNext, risk 0.005 linear); OOS NEVER touched\n")
    d1 = _load(USD, "D1")
    h4_usd = _load(USD, "H4")

    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    yr_of = {f.fold_id: f.oos_start.year for f in is_folds}
    order_ids = sorted(yr_of)
    order_years = [yr_of[i] for i in order_ids]
    warmup_idx = [0, 1]
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

    fn_cost = CostModel.fundednext()
    print("Scoring 18-cfg grids over IS folds 2011-2020 (capturing OOS-realized per-position NET pnl)...")
    scored = {n: {} for n in NAMES}
    pnl_cell = {n: {} for n in NAMES}          # name -> label -> {fold_id: (net_pnl_array, denom)}
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

    # honest nested §5f series + per-fold selected labels
    sel_label = {n: {} for n in NAMES}
    honest_full = {n: [] for n in NAMES}
    for n in NAMES:
        res = nested_walk_forward_select(scored[n], selection=metric_afp_then_mean,
                                         selection_name="afp_then_mean", min_prior_folds=2)
        for c in sorted(res.per_fold, key=lambda c: c.fold_id):
            sel_label[n][c.fold_id] = c.roi_pct if False else c.selected_label
            honest_full[n].append(c.roi_pct)
    honest_eval = {n: [honest_full[n][i] for i in eval_idx] for n in NAMES}

    print("\nhonest §5f nested series (afp_then_mean), evaluable folds 2013-2020:")
    for n in NAMES:
        ser = honest_eval[n]
        print(f"  {n:8s} mean {np.mean(ser)*100:+.3f}%  pos {sum(r>0 for r in ser)}/8  "
              f"per-fold% {[round(r*100,2) for r in ser]}")
        print("           selected exits: " +
              ", ".join(f"{eval_years[k]}:{sel_label[n][eval_ids[k]]}" for k in range(len(eval_idx))))

    w_rp = fit_weights([honest_eval[n] for n in NAMES], mode="risk_parity")
    w_eq = [0.5, 0.5]
    print("\nIS-frozen RP weights (honest series): " + ", ".join(f"{n}={w:.3f}" for n, w in zip(NAMES, w_rp)))

    # ── (2) TAIL CONCENTRATION: pool each leg's positions under its selected exits ──
    print(f"\n{'='*88}\n(2) TAIL CONCENTRATION — pooled positions under the honest-selected exits (8 folds)")
    pooled = {}
    for n in NAMES:
        allp = np.concatenate([pnl_cell[n][sel_label[n][fid]][fid][0] for fid in eval_ids])
        pooled[n] = allp
        tot = allp.sum()
        s = np.sort(allp)[::-1]                       # descending
        npos = len(allp)
        top1 = s[:1].sum(); top3 = s[:3].sum(); top5 = s[:5].sum()
        ktop1pct = max(1, int(round(0.01 * npos)))
        toppct = s[:ktop1pct].sum()
        print(f"  {n:8s} n_pos={npos:4d}  Σnet=${tot:+,.0f}   max=${s[0]:+,.0f}")
        print(f"           top-1 = {top1/tot*100:5.1f}% of Σ | top-3 = {top3/tot*100:5.1f}% | "
              f"top-5 = {top5/tot*100:5.1f}% | top-1%(n={ktop1pct}) = {toppct/tot*100:5.1f}%")

    # ── baseline book + cluster bootstrap (reproduce arc 1056) ──
    def report(tag, book, weights_label):
        pt = float(book.mean()) * 100.0
        ci, p_neg, t, sig = _cluster_boot(book, pt)
        worst_i = int(np.argmin(book))
        print(f"  [{weights_label}] {tag:28s} mean {pt:+.4f}%/yr  pos {int((book>0).sum())}/8  "
              f"worst {book.min()*100:+.3f}%({eval_years[worst_i]})  cluster t={t:+.2f} "
              f"CI[{ci[0]:+.3f},{ci[1]:+.3f}] P(<0)={p_neg:.3f} [{sig}]")
        return pt, t

    for wlabel, w in [("equal", w_eq), ("risk-parity", w_rp)]:
        print(f"\n{'='*88}\n[{wlabel} weights]")
        base = _book_from_cells(pnl_cell, sel_label, eval_ids, w)
        pt0, t0 = report("BASELINE (no winsorize)", base, wlabel)

        # ── (3) WINSORIZE both legs at percentile q of each leg's pooled positive net pnl ──
        print(f"  -- (3) WINSORIZE both legs' positive tail at percentile q --")
        caps_by_q = {}
        for q in (99.0, 97.5, 95.0, 90.0):
            cap = {}
            for n in NAMES:
                pos = pooled[n][pooled[n] > 0]
                cap[n] = np.percentile(pos, q) if len(pos) else np.inf
            caps_by_q[q] = cap
            tf = lambda n, p: np.minimum(p, cap[n])
            bk = _book_from_cells(pnl_cell, sel_label, eval_ids, w, transform=tf)
            report(f"winsorize q{q:g} (cap fbr=${cap['fbr']:,.0f})", bk, wlabel)

        # ── (5) per-leg attribution: winsorize ONLY fbr, then ONLY me_long, at q95 ──
        print(f"  -- (5) per-leg attribution (winsorize ONE leg at q95) --")
        cap95 = caps_by_q[95.0]
        for only in NAMES:
            tf = lambda n, p, _only=only: (np.minimum(p, cap95[n]) if n == _only else p)
            bk = _book_from_cells(pnl_cell, sel_label, eval_ids, w, transform=tf)
            report(f"winsorize ONLY {only} q95", bk, wlabel)

        # ── (4) LEAVE-TOP-N: remove the N largest positions of the whole book ──
        print(f"  -- (4) LEAVE-TOP-N (remove the N largest book positions across both legs) --")
        # build a global ranked list of (net_pnl, leg, fold_id, idx_in_cell)
        catalog = []
        for n in NAMES:
            for fid in eval_ids:
                arr = pnl_cell[n][sel_label[n][fid]][fid][0]
                for j, v in enumerate(arr):
                    catalog.append((v, n, fid, j))
        catalog.sort(key=lambda r: r[0], reverse=True)
        for topn in (1, 2, 3, 5):
            drop = {(n, fid, j) for (_, n, fid, j) in catalog[:topn]}
            who = ", ".join(f"{n}/{yr_of[fid]}=${v:,.0f}" for (v, n, fid, j) in catalog[:topn])
            def tf(n, p, _fidmap=None):
                return p  # placeholder; need fold-aware removal below
            # fold-aware removal: rebuild per-leg series dropping the flagged positions
            per_leg = []
            for n in NAMES:
                ser = []
                for fid in eval_ids:
                    arr, denom = pnl_cell[n][sel_label[n][fid]][fid]
                    keep = np.array([v for j, v in enumerate(arr) if (n, fid, j) not in drop], dtype=float)
                    ser.append(0.0 if len(keep) == 0 else keep.sum() / denom)
                per_leg.append(ser)
            bk = np.array(combine_fold_rois(per_leg, w).combined_roi)
            report(f"drop top-{topn} [{who}]", bk, wlabel)


if __name__ == "__main__":
    main()
