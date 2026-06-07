"""arc 2086 — STOP-WIDTH (SL-multiple) sweep on the positive-skew CONTINUATION entry.

The one axis every prior positive-skew arc (2081-2085) held fixed: SL = 2*ATR. The unified finding
across geometry/timeframe/direction/universe/exit was an INVARIANT median trade R ~= -0.9 -- a
stop-width artifact (2*ATR whipsaws continuation entries out before the trend develops). This sweeps
sl_atr_mult in {2,3,4,6} on the canonical Donchian-breakout continuation entry, full 28-pair, under the
tail-preserving trailing exits, judged on per-trade MEAN R + MEDIAN R + the TAIL-REMOVED guard.

Risk-normalization verified: A1Architecture risk-sizes from entry->SL (risk_pct), exit policies set
1R = sl_atr_mult*ATR, so a stop-out = -1R = $500 at every SL width and net_pnl/$500 is comparable across
widths. Widening SL: each loss still -1R, fewer stop-outs (median should rise), tail in R compresses.

CALLS canonical (Panel/ArcFoldRunner/A1/build_v3_folds/apply_cost_model); experiment-side = Donchian entry
mask (arc 2000) + SL parameter + winsorization arithmetic (arc 2063/2081). Never realizes P&L.

Pre-registered kill-rule (arc doc, before run): mean >0 net of costs AND survives +2R cap / top-5% removal
at the best SL width AND beats the being-long null; else KILL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.sim.costs.model import CostModel, apply_cost_model
from core.wfo.folds import build_v3_folds

from discovery.tools.trend_entry_signals import DonchianBreakoutLongSignal, PeriodicLongSignal

BACKUP = r"C:\Users\panap\histdata_backup"
CACHE = r"C:/Users/panap/Documents/Forex-Backtester/data/cache"
SB, RISK_PCT = 100_000.0, 0.005
R_DOLLARS = RISK_PCT * SB  # $500 -- linear risk per trade (-1R loss), SL-distance-normalized

MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP"]
ALL28 = ["AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY", "CHFJPY",
         "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD", "GBPAUD",
         "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD", "NZDCAD", "NZDCHF", "NZDJPY",
         "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]

SL_MULTS = [2.0, 3.0, 4.0, 6.0]
EXITS = ["sl_plus_trailing_atr", "sl_partial_close_1r_runner_trail"]


def collect(sig, panel, is_folds, exit_policy, sl_mult, tag):
    """Run one signal+exit+SL over the IS folds; collect per-fold ROI + per-trade net R (+ pair)."""
    sig_eval = sig.evaluate({"H4": panel})
    runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=sig_eval, panels={"H4": panel})
    cfg = A1Config(config_id=f"arc2086_{tag}_{exit_policy}_sl{sl_mult}", exit_policy=exit_policy,
                   sl_atr_mult=sl_mult)
    fn_cost = CostModel.fundednext()
    fold_roi = {}
    trade_R, trade_pair = [], []
    for fold in is_folds:
        fs = runner(fold, cfg)
        fold_roi[fold.oos_start.year] = (fs.roi_pct * 100.0, fs.n_trades)
        costed = apply_cost_model(runner.last_result.run_result, fn_cost)
        bd = costed.breakdown
        if len(bd):
            ex = pd.to_datetime(bd["final_exit_time"], utc=True)
            o0 = pd.Timestamp(fold.oos_start, tz="UTC")
            o1 = pd.Timestamp(fold.oos_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            m = (ex >= o0) & (ex <= o1)
            sub = bd.loc[m]
            trade_R.extend((sub["net_pnl"].to_numpy(float) / R_DOLLARS).tolist())
            pcol = "pair" if "pair" in sub.columns else ("symbol" if "symbol" in sub.columns else None)
            trade_pair.extend(sub[pcol].tolist() if pcol else [None] * len(sub))
    return fold_roi, np.array(trade_R, float), np.array(trade_pair, object)


def tail_metrics(R):
    if len(R) == 0:
        return None
    mean = R.mean()
    capped2 = np.minimum(R, 2.0).mean()
    k = max(1, int(round(0.05 * len(R))))
    thresh = np.sort(R)[-k]
    top5_removed = R[R < thresh].mean() if (R < thresh).any() else 0.0
    return dict(n=len(R), mean=mean, median=np.median(R), win=float((R > 0).mean()),
                cap2=capped2, top5rm=top5_removed, max=R.max(),
                surv2=capped2 / mean * 100 if mean else float("nan"),
                survtop5=top5_removed / mean * 100 if mean else float("nan"))


def report(label, fold_roi, R):
    yrs = sorted(fold_roi)
    rois = [fold_roi[y][0] for y in yrs]
    print(f"\n=== {label} ===")
    print(f"  per-fold ROI%: " + ", ".join(f"{y}:{fold_roi[y][0]:+.1f}" for y in yrs))
    print(f"  fold mean {np.mean(rois):+.3f}%  median {np.median(rois):+.3f}%  "
          f"neg {sum(r < 0 for r in rois)}/{len(rois)}  worst {min(rois):+.2f}%  "
          f"2015:{fold_roi.get(2015,(float('nan'),))[0]:+.1f}  2018:{fold_roi.get(2018,(float('nan'),))[0]:+.1f}")
    tm = tail_metrics(R)
    if tm:
        print(f"  per-trade R: n={tm['n']} mean={tm['mean']:+.4f} MEDIAN={tm['median']:+.4f} "
              f"win={tm['win']:.3f} max={tm['max']:+.1f}")
        print(f"  TAIL-REMOVED: +2R-cap={tm['cap2']:+.4f} ({tm['surv2']:.0f}%surv) | "
              f"top5%-removed={tm['top5rm']:+.4f} ({tm['survtop5']:.0f}%surv)")
    return tm


def main():
    panel = Panel.from_pairs(ALL28, tf="H4", histdata_root=BACKUP, cache_root=CACHE,
                             boundary_convention="5ers_eet")
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    print(f"=== arc 2086 STOP-WIDTH sweep, Donchian-40 continuation (28 pairs H4, {len(is_folds)} IS folds) ===")
    print(f"R-unit = ${R_DOLLARS:.0f}; FundedNext costs ON; risk={RISK_PCT}; SL sweep {SL_MULTS}")
    print("KEY QUESTION: does the median trade R climb off the invariant ~-0.9 wall as SL widens, "
          "and does the mean flip + survive the tail guard?")

    best = None
    for ep in EXITS:
        print(f"\n########## DONCHIAN-40 LONG x {ep} -- SL SWEEP ##########")
        for sl in SL_MULTS:
            fr, R, P = collect(DonchianBreakoutLongSignal(lookback=40, spacing_bars=6),
                               panel, is_folds, ep, sl, "don40")
            tm = report(f"donchian40 + {ep} + SL={sl}xATR", fr, R)
            if tm and (best is None or tm["mean"] > best[1]):
                best = (f"{ep} SL={sl}", tm["mean"], tm)

    print("\n########## PERIODIC-LONG NULL (being-long-anytime) -- SL {2,4} ##########")
    for sl in (2.0, 4.0):
        fr, R, P = collect(PeriodicLongSignal(period=30, warmup=120), panel, is_folds,
                           "sl_plus_trailing_atr", sl, "periodic")
        report(f"periodic30 NULL + trailing_atr + SL={sl}xATR", fr, R)

    if best:
        print(f"\n>>> BEST cell by per-trade mean R: {best[0]}  mean={best[1]:+.4f}R  "
              f"median={best[2]['median']:+.4f}  +2Rcap={best[2]['cap2']:+.4f}  top5rm={best[2]['top5rm']:+.4f}")
        verdict = "SURVIVES guard (escalate)" if (best[1] > 0 and best[2]['cap2'] > 0 and best[2]['top5rm'] > 0) else "KILL by pre-registered guard"
        print(f">>> PRE-REGISTERED GUARD: {verdict}")


if __name__ == "__main__":
    main()
