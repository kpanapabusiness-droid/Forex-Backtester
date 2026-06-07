"""arc 2081 — Positive-skew CONTINUATION book: MEAN + median-per-fold + TAIL-REMOVED.

The operator-redirected frontier (LESSONS 2026-06-06): a continuation/breakout book, take-the-loss +
TRAILING (tail-preserving) exit, judged on the lens the corpus never applied to a trend book — per-trade
MEAN R, median-per-fold, and the TAIL-REMOVED guard (+2R cap / top-5% drop). Full 28-pair universe (the
arc-2000 majors-only open thread). CALLS canonical (Panel/ArcFoldRunner/A1/build_v3_folds/apply_cost_model);
experiment-side = Donchian entry mask (arc 2000) + winsorization arithmetic (arc 2063). Never realizes P&L.

Pre-registered kill-rule (arc doc, written before this run): mean must be >0 net of costs AND survive the
+2R cap / top-5% removal; if mean-positive ONLY via the top-K winners -> KILL.
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
R_DOLLARS = RISK_PCT * SB  # $500 — linear risk per trade (-1R loss)

MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP"]
ALL28 = ["AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY", "CHFJPY",
         "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD", "GBPAUD",
         "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD", "NZDCAD", "NZDCHF", "NZDJPY",
         "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
CROSSES = [p for p in ALL28 if p not in MAJORS]
EXITS = ["sl_only", "sl_plus_trailing_atr", "sl_plus_trailing_swing", "sl_partial_close_1r_runner_trail"]


def collect(sig, panel, is_folds, exit_policy, tag):
    """Run one signal+exit over the IS folds; collect per-fold ROI + per-trade net R (+ pair)."""
    sig_eval = sig.evaluate({"H4": panel})
    runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=sig_eval, panels={"H4": panel})
    cfg = A1Config(config_id=f"arc2081_{tag}_{exit_policy}", exit_policy=exit_policy, sl_atr_mult=2.0)
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
    """Per-trade R distribution + the TAIL-REMOVED guard."""
    if len(R) == 0:
        return None
    mean = R.mean()
    capped2 = np.minimum(R, 2.0).mean()
    capped3 = np.minimum(R, 3.0).mean()
    k = max(1, int(round(0.05 * len(R))))
    thresh = np.sort(R)[-k]
    top5_removed = R[R < thresh].mean() if (R < thresh).any() else 0.0
    # also drop the single largest winner
    drop1 = np.sort(R)[:-1].mean()
    return dict(n=len(R), mean=mean, median=np.median(R), win=float((R > 0).mean()),
                cap2=capped2, cap3=capped3, top5rm=top5_removed, drop1=drop1,
                surv2=capped2 / mean * 100 if mean else float("nan"),
                survtop5=top5_removed / mean * 100 if mean else float("nan"),
                max=R.max())


def report(label, fold_roi, R):
    yrs = sorted(fold_roi)
    rois = [fold_roi[y][0] for y in yrs]
    print(f"\n=== {label} ===")
    print(f"  per-fold ROI%: " + ", ".join(f"{y}:{fold_roi[y][0]:+.2f}(n{fold_roi[y][1]})" for y in yrs))
    print(f"  fold mean {np.mean(rois):+.3f}%  median {np.median(rois):+.3f}%  "
          f"neg {sum(r < 0 for r in rois)}/{len(rois)}  worst {min(rois):+.3f}%")
    for y in (2015, 2018):
        if y in fold_roi:
            print(f"    {y}: {fold_roi[y][0]:+.3f}% (n{fold_roi[y][1]})")
    tm = tail_metrics(R)
    if tm:
        print(f"  per-trade R: n={tm['n']} mean={tm['mean']:+.4f} median={tm['median']:+.4f} "
              f"win={tm['win']:.3f} max={tm['max']:+.2f}")
        print(f"  TAIL-REMOVED: +2R-cap mean={tm['cap2']:+.4f} ({tm['surv2']:.0f}% survives) | "
              f"+3R-cap={tm['cap3']:+.4f} | top5%-removed={tm['top5rm']:+.4f} ({tm['survtop5']:.0f}%) | "
              f"drop-largest={tm['drop1']:+.4f}")
    return tm


def split_metrics(R, P):
    maj_mask = np.array([p in MAJORS for p in P], bool)
    for lab, mask in (("MAJORS", maj_mask), ("CROSSES", ~maj_mask)):
        Rs = R[mask]
        tm = tail_metrics(Rs)
        if tm:
            print(f"  [{lab}] n={tm['n']} mean={tm['mean']:+.4f} median={tm['median']:+.4f} "
                  f"+2Rcap={tm['cap2']:+.4f} ({tm['surv2']:.0f}%) top5rm={tm['top5rm']:+.4f}")


def main():
    panel = Panel.from_pairs(ALL28, tf="H4", histdata_root=BACKUP, cache_root=CACHE,
                             boundary_convention="5ers_eet")
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    print(f"=== arc 2081 positive-skew CONTINUATION book (28 pairs H4, {len(is_folds)} IS folds) ===")
    print(f"R-unit = risk_pct*SB = ${R_DOLLARS:.0f}; FundedNext costs ON; SL=2*ATR; risk={RISK_PCT}")

    print("\n########## DONCHIAN-40 BREAKOUT LONG (full 28-pair universe) ##########")
    best = None
    for ep in EXITS:
        fr, R, P = collect(DonchianBreakoutLongSignal(lookback=40, spacing_bars=6), panel, is_folds, ep, "don40")
        tm = report(f"donchian40 + {ep}", fr, R)
        split_metrics(R, P)
        if tm and (best is None or tm["mean"] > best[1]):
            best = (ep, tm["mean"], R, P)

    # Null: being-long-anytime under the two trailing exits (does the breakout beat generic long-vol?)
    print("\n########## PERIODIC-LONG NULL (being-long-anytime, same exit) ##########")
    for ep in ("sl_plus_trailing_atr", "sl_partial_close_1r_runner_trail"):
        fr, R, P = collect(PeriodicLongSignal(period=30, warmup=120), panel, is_folds, ep, "periodic")
        report(f"periodic30 NULL + {ep}", fr, R)

    if best:
        print(f"\n>>> BEST real exit by per-trade mean R: {best[0]}  (mean={best[1]:+.4f}R)")
        print(">>> majors/crosses split for the best exit:")
        split_metrics(best[2], best[3])


if __name__ == "__main__":
    main()
