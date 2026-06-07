"""arc 2084 — positive-skew trend CONTINUATION on the DAILY timeframe (the Turtle/managed-futures home).

The operator-redirected positive-skew thread has been closed across breakout (1074/2081),
vol-expansion breakout (2082), pullback-resume favorable+adverse-first (1075/2083) and shock (2081)
— but EVERY one of those arcs ran on H4. The ONE axis none varied is the SAMPLING TIMEFRAME.

WHY D1 is genuinely distinct (the mechanistic *because*): a take-the-loss trailing exit's whipsaw
rate is governed by the ratio (trailing-stop width) / (intra-trend pullback size). On H4 the 2.ATR(H4)
trailing stop is small relative to a trend's ordinary daily pullbacks, so it gets hit by intra-trend
noise (arc 2081: median trade -0.81R = stopped out mid-move; winners can't run). Classic positive-skew
trend-following (Turtle: 20/55-day Donchian, 2N stop) is a DAILY phenomenon precisely because at lower
sampling frequency the stop sits ~sqrt(6)x wider relative to the trend and survives ordinary pullbacks,
letting the winner run to its multi-month fat-tail length. So the trailing-exit x take-the-loss
interaction that the positive-skew premise depends on is structurally different on D1 — and it was
never run on the honest engine under the mean + median-per-fold + tail-removed lens.

This driver = arc 1074's EXACT construction (TrendContinuationBreakoutSignal, both directions,
dual-SMA 50/200, Donchian {20,55}, runner exits x SL{1.5,2,2.5}, §5f nested, tail-removed guard,
fair same-side null) with the SOLE change being TF H4 -> D1 and the FULL 28-pair universe (trend
following is classically a broad-diversified play; arc 2000's own #1 open thread + arc 2081 showed
majors~=crosses on H4). CALLS canonical scoring (A1->MultiPairBacktester, FundedNext, risk 0.005);
experiment side = the entry signal + tail-removal/null arithmetic only. IS 2011-2020; OOS touched
ONLY if the IS guard (G1+G2+G3 + beats null) holds.
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
from discovery.tools.nested_exit_selection import (
    freeze_best_over_folds,
    metric_afp_then_mean,
    nested_walk_forward_select,
)
from discovery.tools.null_entry_baseline import build_null_signal_evaluation
from discovery.tools.trend_continuation_signal import TrendContinuationBreakoutSignal

HIST = Path(os.environ.get("HIST", r"C:/Users/panap/histdata_backup"))
CACHE = Path(os.environ.get("CACHE", r"C:/Users/panap/Documents/Forex-Backtester/data/cache"))
BOUNDARY = "5ers_eet"
TF = "D1"
PAIRS = [
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD", "GBPAUD",
    "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD", "NZDCAD", "NZDCHF", "NZDJPY",
    "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]
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
        cellp[fold.fold_id] = (net, SB)
    return fs_list, cellp


def _honest_series(scored):
    res = nested_walk_forward_select(scored, selection=metric_afp_then_mean,
                                     selection_name="afp_then_mean", min_prior_folds=2)
    sel = {c.fold_id: c.selected_label for c in res.per_fold}
    ordered = sorted(res.per_fold, key=lambda c: c.fold_id)
    return sel, ordered


def _tail_stats(pnl_by_fold, sel, eval_ids, yr_of):
    risk_d = RISK * SB
    per_fold_roi, per_fold_med_roi, pooled_r, cap_roi, drop5_roi = [], [], [], [], []
    for fid in eval_ids:
        net, denom = pnl_by_fold[sel[fid]][fid]
        per_fold_roi.append(net.sum() / denom if len(net) else 0.0)
        r = net / risk_d
        per_fold_med_roi.append(float(np.median(r)) if len(r) else 0.0)
        pooled_r.append(r)
        capped = np.minimum(net, 2.0 * risk_d)
        cap_roi.append(capped.sum() / denom if len(net) else 0.0)
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
    return float(np.mean(rois)), (catalog[:k] if catalog else [(0.0, -1, -1)])


def run_cell(panel, *, direction, donch, sf, ss, is_folds, yr_of, fn_cost, label):
    sig = TrendContinuationBreakoutSignal(direction=direction, lookback=donch,
                                          sma_fast=sf, sma_slow=ss, primary_tf=TF).evaluate({TF: panel})
    n_fires = sum(int(st.signal_mask.sum()) for st in sig.per_pair.values())
    runner = ArcFoldRunner(A1Architecture(), sig, {TF: panel})
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
    mean_r, med_r = float(allr.mean()) if len(allr) else 0.0, float(np.median(allr)) if len(allr) else 0.0
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
            "sig": sig}


def run_null(panel, cell, direction, donch, is_folds, yr_of, fn_cost):
    frozen = freeze_best_over_folds(cell["scored"])
    cfg = _grid(f"{direction}_{donch}")[frozen]
    null_eval = build_null_signal_evaluation(cell["sig"], seed=42, warmup=210)
    runner = ArcFoldRunner(A1Architecture(), null_eval, {TF: panel})
    fs_list, cellp = _score(runner, is_folds, cfg, fn_cost)
    eval_ids = cell["eval_ids"]
    real = np.array([cell["pnl_by_fold"][cell["sel"][i]][i][0].sum() / SB for i in eval_ids])
    nullr = np.array([cellp[i][0].sum() / SB for i in eval_ids])
    print(f"  NULL ({frozen}): real mean {real.mean()*100:+.4f}%/yr vs null mean {nullr.mean()*100:+.4f}%/yr "
          f"-> excess {(real.mean()-nullr.mean())*100:+.4f}pp ({'beats' if real.mean()>nullr.mean() else 'LOSES to'} null)")


def main():
    print("ARC 2084 — positive-skew trend CONTINUATION on the DAILY timeframe (Turtle/managed-futures home)")
    print(f"HIST={HIST} CACHE={CACHE}\nTF={TF} pairs={len(PAIRS)} runner exits={EXITS} SLs={SLS} risk={RISK}; OOS untouched unless guard holds\n")
    panel = Panel.from_pairs(PAIRS, TF, histdata_root=HIST, cache_root=CACHE,
                             use_cache=True, boundary_convention=BOUNDARY)
    is_folds = [f for f in build_v3_folds().folds if f.oos_start.year >= 2011]
    yr_of = {f.fold_id: f.oos_start.year for f in is_folds}
    print(f"IS folds (oos years): {[yr_of[f.fold_id] for f in is_folds]}")
    fn_cost = CostModel.fundednext()

    cells = [
        ("PRIMARY long  Donchian20 trend50/200", "long", 20, 50, 200),
        ("PRIMARY short Donchian20 trend50/200", "short", 20, 50, 200),
        ("TURTLE  long  Donchian55 trend50/200", "long", 55, 50, 200),
        ("TURTLE  short Donchian55 trend50/200", "short", 55, 50, 200),
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
        print(f"  {label:40s} G1={cell['g1']} G2={cell['g2']} G3={cell['g3']} AFP={cell['afp']} "
              f"-> {'SURVIVES IS -> measure OOS' if ok else 'KILL at IS (OOS preserved)'}")
    if not any_pass:
        print("\nNo cell survives the IS guard -> KILL at IS; OOS NOT touched (preserved frozen).")


if __name__ == "__main__":
    main()
