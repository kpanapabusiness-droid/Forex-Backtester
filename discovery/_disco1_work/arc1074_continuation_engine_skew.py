"""arc 1074 — positive-skew CONTINUATION on the HONEST ENGINE (close arc-1064's deferred engine test).

Arc 1064 cheap-killed the trend-continuation positive-skew thread on GROSS fwd_drift, arguing the
SL-first engine is "strictly worse" so no engine was spent. That bound is wrong for a take-the-loss
trailing-runner exit: the -1R stop CAPS the gross left-tail losers (-3..-5 ATR) that drove arc-1064's
strongly-negative tail-removed gross mean. This driver spends the honest engine on arc-1064's EXACT
entry (TrendContinuationBreakoutSignal, both directions) under the positive-skew RUNNER exits only,
with §5f nested exit/SL selection, and judges by per-fold MEAN ROI + median-per-fold + TAIL-REMOVED
(+2R cap, drop top-5%/top-K) expectancy vs a fair same-side random-entry null.

CALLS canonical scoring (A1->MultiPairBacktester, FundedNext, risk 0.005); experiment side = the
entry signal + tail-removal/null ARITHMETIC only. Never realizes P&L for a gate. IS 2010-2020 only;
OOS touched ONLY if the IS guard (G1+G2+G3 + beats null) holds (printed decision). H4, 7 USD majors.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.costs.model import CostModel, apply_cost_model
from core.sim.panel import Panel
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import build_oos_year_folds
from discovery.tools.nested_exit_selection import (
    metric_afp_then_mean,
    nested_walk_forward_select,
)
from discovery.tools.null_entry_baseline import build_null_signal_evaluation
from discovery.tools.trend_continuation_signal import TrendContinuationBreakoutSignal

HIST = Path(os.environ.get("HIST", r"C:/Users/panap/histdata_backup"))
CACHE = Path(os.environ.get("CACHE", r"C:/Users/panap/Documents/Forex-Backtester/data/cache"))
BOUNDARY = "5ers_eet"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
RISK = 0.005
SB = 100_000.0
EXITS = ["sl_plus_trailing_atr", "sl_plus_trailing_swing", "sl_partial_close_1r_runner_trail"]
SLS = [1.5, 2.0, 2.5]
BINDING = [2015, 2018]


def _grid(tag):
    return {f"{ex}|sl{sl}": A1Config(config_id=f"{tag}_{ex}_sl{sl}", sl_atr_mult=sl,
                                     trail_enabled=False, exit_policy=ex)
            for ex in EXITS for sl in SLS}


def _score(runner, folds, cfg, fn_cost):
    """Return (fold_stats_list, {fold_id: (net_pnl_array, denom)}) over `folds` (OOS-window positions)."""
    fs_list, cellp = [], {}
    for fold in folds:
        fs = runner(fold, cfg)
        fs_list.append(fs)
        rr = runner.last_result.run_result
        costed = apply_cost_model(rr, fn_cost)
        oos_start = pd.Timestamp(fold.oos_start, tz="UTC")
        oos_end = pd.Timestamp(fold.oos_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        bd = costed.breakdown
        if len(bd):
            ex = pd.to_datetime(bd["final_exit_time"], utc=True)
            m = (ex >= oos_start) & (ex <= oos_end)
            net = bd.loc[m, "net_pnl"].to_numpy(float)
        else:
            net = np.empty(0, float)
        denom = SB  # constant-notional fold base (matches FoldStats roi denom convention)
        cellp[fold.fold_id] = (net, denom)
    return fs_list, cellp


def _honest_series(scored):
    """Nested §5f (afp_then_mean) -> {fold_id: selected_label}, ordered per-fold roi list, eval ids."""
    res = nested_walk_forward_select(scored, selection=metric_afp_then_mean,
                                     selection_name="afp_then_mean", min_prior_folds=2)
    sel = {c.fold_id: c.selected_label for c in res.per_fold}
    ordered = sorted(res.per_fold, key=lambda c: c.fold_id)
    return sel, ordered


def _tail_stats(pnl_by_fold, sel, eval_ids, yr_of):
    """Per-fold ROI + per-trade R distribution + tail-removed (+2R cap, drop top5%/topK)."""
    risk_d = RISK * SB
    per_fold_roi, per_fold_med_roi, pooled_r = [], [], []
    cap_roi, drop5_roi = [], []
    per_pair_unused = None
    for fid in eval_ids:
        net, denom = pnl_by_fold[sel[fid]][fid]
        roi = net.sum() / denom if len(net) else 0.0
        per_fold_roi.append(roi)
        r = net / risk_d
        per_fold_med_roi.append(float(np.median(r)) if len(r) else 0.0)
        pooled_r.append(r)
        # +2R cap
        capped = np.minimum(net, 2.0 * risk_d)
        cap_roi.append(capped.sum() / denom if len(net) else 0.0)
        # drop top-5% of POSITIVE positions in this fold
        if len(net):
            k = max(1, int(np.ceil(0.05 * len(net))))
            order = np.argsort(net)[::-1]
            keep = np.ones(len(net), bool)
            keep[order[:k]] = False
            drop5_roi.append(net[keep].sum() / denom)
        else:
            drop5_roi.append(0.0)
    allr = np.concatenate(pooled_r) if pooled_r else np.empty(0)
    return {
        "per_fold_roi": np.array(per_fold_roi),
        "per_fold_med_roi": np.array(per_fold_med_roi),
        "cap_roi": np.array(cap_roi),
        "drop5_roi": np.array(drop5_roi),
        "allr": allr,
        "eval_years": [yr_of[i] for i in eval_ids],
    }


def _global_drop_topk(pnl_by_fold, sel, eval_ids, k):
    """Mean per-fold ROI after removing the k largest positions across the whole pooled book."""
    catalog = []
    for fid in eval_ids:
        net, denom = pnl_by_fold[sel[fid]][fid]
        for j, v in enumerate(net):
            catalog.append((v, fid, j))
    catalog.sort(key=lambda r: r[0], reverse=True)
    drop = {(fid, j) for (_, fid, j) in catalog[:k]}
    rois = []
    for fid in eval_ids:
        net, denom = pnl_by_fold[sel[fid]][fid]
        keep = np.array([v for j, v in enumerate(net) if (fid, j) not in drop], float)
        rois.append(keep.sum() / denom if len(keep) else 0.0)
    return float(np.mean(rois)), catalog[:k]


def run_cell(panel, *, direction, donch, sf, ss, is_folds, yr_of, fn_cost, label):
    sig = TrendContinuationBreakoutSignal(direction=direction, lookback=donch,
                                          sma_fast=sf, sma_slow=ss).evaluate({"H4": panel})
    n_fires = sum(int(st.signal_mask.sum()) for st in sig.per_pair.values())
    runner = ArcFoldRunner(A1Architecture(), sig, {"H4": panel})
    scored, pnl_by_fold = {}, {}
    for lab, cfg in _grid(f"{direction}_{donch}").items():
        fs_list, cellp = _score(runner, is_folds, cfg, fn_cost)
        scored[lab] = fs_list
        pnl_by_fold[lab] = cellp
    sel, ordered = _honest_series(scored)
    eval_ids = [c.fold_id for c in ordered if not c.is_warmup]
    eval_years = [yr_of[i] for i in eval_ids]

    ts = _tail_stats(pnl_by_fold, sel, eval_ids, yr_of)
    roi = ts["per_fold_roi"]
    mean_roi, med_roi = float(roi.mean()), float(np.median(roi))
    pos = int((roi > 0).sum())
    afp = pos == len(roi)
    allr = ts["allr"]
    mean_r, med_r = float(allr.mean()), float(np.median(allr))
    cap_mean = float(ts["cap_roi"].mean())
    drop5_mean = float(ts["drop5_roi"].mean())
    medfold_pos = int((ts["per_fold_med_roi"] > 0).sum())
    d1_mean, top1 = _global_drop_topk(pnl_by_fold, sel, eval_ids, 1)
    d3_mean, _ = _global_drop_topk(pnl_by_fold, sel, eval_ids, 3)
    d5_mean, _ = _global_drop_topk(pnl_by_fold, sel, eval_ids, 5)

    print(f"\n{'='*92}\n{label}  | dir={direction} donch={donch} sma{sf}/{ss} | fires={n_fires} | "
          f"n_trades(pooled)={len(allr)}")
    print(f"  selected exits: " + ", ".join(f"{yr_of[i]}:{sel[i]}" for i in eval_ids))
    print(f"  per-fold ROI%: {[round(r*100,2) for r in roi]}  years {eval_years}")
    print(f"  G1  MEAN per-fold ROI {mean_roi*100:+.4f}%/yr | median-fold {med_roi*100:+.4f}% | "
          f"folds+ {pos}/{len(roi)}{'  [AFP]' if afp else ''}")
    print(f"      mean per-trade R {mean_r:+.4f} | median R {med_r:+.4f}")
    print(f"  G2  TAIL-REMOVED: +2R-cap mean ROI {cap_mean*100:+.4f}% ({'PASS' if cap_mean>0 else 'FAIL'}) | "
          f"drop-top5%/fold mean {drop5_mean*100:+.4f}% ({'PASS' if drop5_mean>0 else 'FAIL'})")
    print(f"      drop-top-K(global): K1 {d1_mean*100:+.4f}%  K3 {d3_mean*100:+.4f}%  K5 {d5_mean*100:+.4f}%  "
          f"(largest pos R={top1[0][0]/(RISK*SB):+.2f})")
    print(f"  G3  per-fold median ROI>0 in {medfold_pos}/{len(roi)} folds (majority needs >{len(roi)//2})")
    binding = {yr_of[i]: round(roi[k]*100, 3) for k, i in enumerate(eval_ids) if yr_of[i] in BINDING}
    print(f"      BINDING folds {binding}")
    g1 = mean_roi > 0 and mean_r > 0
    g2 = cap_mean > 0 and drop5_mean > 0 and d1_mean > 0 and d3_mean > 0
    g3 = medfold_pos > len(roi) // 2
    print(f"  >>> GUARD: G1 {'PASS' if g1 else 'FAIL'} | G2 {'PASS' if g2 else 'FAIL'} | "
          f"G3 {'PASS' if g3 else 'FAIL'}")
    return {"sel": sel, "eval_ids": eval_ids, "mean_roi": mean_roi, "afp": afp,
            "g1": g1, "g2": g2, "g3": g3, "scored": scored, "pnl_by_fold": pnl_by_fold,
            "sig": sig, "runner": runner}


def run_null(panel, cell, direction, donch, is_folds, yr_of, fn_cost):
    """Fair same-side random-entry null under the cell's frozen (all-fold-best) exit."""
    from discovery.tools.nested_exit_selection import freeze_best_over_folds
    frozen = freeze_best_over_folds(cell["scored"])
    cfg = _grid(f"{direction}_{donch}")[frozen]
    null_eval = build_null_signal_evaluation(cell["sig"], seed=42, warmup=210)
    runner = ArcFoldRunner(A1Architecture(), null_eval, {"H4": panel})
    fs_list, cellp = _score(runner, is_folds, cfg, fn_cost)
    eval_ids = cell["eval_ids"]
    real = np.array([cell["pnl_by_fold"][cell["sel"][i]][i][0].sum() / SB for i in eval_ids])
    nullr = np.array([cellp[i][0].sum() / SB for i in eval_ids])
    print(f"  NULL ({frozen}): real mean {real.mean()*100:+.4f}%/yr vs null mean {nullr.mean()*100:+.4f}%/yr "
          f"-> excess {(real.mean()-nullr.mean())*100:+.4f}pp ({'beats' if real.mean()>nullr.mean() else 'LOSES to'} null)")


def main():
    print("ARC 1074 — positive-skew CONTINUATION on the HONEST ENGINE (arc-1064's deferred engine test)")
    print(f"HIST={HIST} CACHE={CACHE}\nrunner exits={EXITS} SLs={SLS} risk={RISK}; OOS untouched unless guard holds\n")
    panel = Panel.from_pairs(PAIRS, "H4", histdata_root=HIST, cache_root=CACHE,
                             use_cache=True, boundary_convention=BOUNDARY)
    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    yr_of = {f.fold_id: f.oos_start.year for f in is_folds}
    print(f"IS folds (oos years): {[yr_of[f.fold_id] for f in is_folds]}")
    fn_cost = CostModel.fundednext()

    cells = [
        ("PRIMARY long  Donchian20 trend50/200", "long", 20, 50, 200),
        ("PRIMARY short Donchian20 trend50/200", "short", 20, 50, 200),
        ("ROBUST  long  Donchian55 trend50/200 (Turtle)", "long", 55, 50, 200),
        ("ROBUST  short Donchian55 trend50/200 (Turtle)", "short", 55, 50, 200),
    ]
    results = []
    for label, d, dn, sf, ss in cells:
        cell = run_cell(panel, direction=d, donch=dn, sf=sf, ss=ss,
                        is_folds=is_folds, yr_of=yr_of, fn_cost=fn_cost, label=label)
        run_null(panel, cell, d, dn, is_folds, yr_of, fn_cost)
        results.append((label, d, dn, cell))

    print(f"\n{'='*92}\nSUMMARY — IS guard decision (OOS spent ONLY for a cell passing G1+G2+G3)")
    any_pass = False
    for label, d, dn, cell in results:
        ok = cell["g1"] and cell["g2"] and cell["g3"]
        any_pass = any_pass or ok
        print(f"  {label:48s} G1={cell['g1']} G2={cell['g2']} G3={cell['g3']} AFP={cell['afp']} "
              f"-> {'SURVIVES IS -> measure OOS' if ok else 'KILL at IS (OOS preserved)'}")
    if not any_pass:
        print("\nNo cell survives the IS guard -> KILL at IS; OOS NOT touched (preserved frozen).")


if __name__ == "__main__":
    main()
