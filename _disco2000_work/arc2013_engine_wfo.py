"""arc 2013 — honest-engine IS WFO for the UP-gap weekend SHORT on JPY crosses (§5f exit menu).

The observation cleared the §5d bar (honest i+1 short capture 0.518 >0.50 at thr 1.0, median +0.146,
beats the fair weekly-open null by +0.27 ATR) so it is a NON-coin-flip entry → §5f REQUIRES the honest
engine under the registered exit menu before any FAIL. This is the FIRST end-to-end SHORT engine run in
the programme (arc 2011: the short pool/engine path was observation-verified only). I CALL the canonical
apparatus (Panel, build_arc_pool, ArcFoldRunner, build_v3_folds, discovery_measure) — never reimplement.

Disposition logic (§11): all-folds-positive IS → measure OOS; else mean-positive net of costs →
PORTFOLIO candidate; else (incl. mean-negative) → KILL.
"""
from __future__ import annotations

import numpy as np

from core.sim.panel import Panel
from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import judge_all_folds_positive, run_config_over_folds
from datetime import date

from discovery.tools.gap_signals import WeekendUpGapShortSignal

CROSSES = ["EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"]
BACKUP = r"C:\Users\panap\histdata_backup"
EXITS = [
    "sl_only", "sl_plus_tp_2r", "sl_plus_tp_3r",
    "sl_plus_trailing_atr", "sl_plus_trailing_swing", "sl_partial_close_1r_runner_trail",
]
THRESHOLDS = [0.5, 1.0]


def main() -> None:
    panel = Panel.from_pairs(CROSSES, tf="H4", histdata_root=BACKUP, cache_root="data/cache",
                             boundary_convention="5ers_eet")
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    print(f"=== arc 2013 UP-gap SHORT honest IS WFO (JPY crosses) — {len(is_folds)} IS folds ===")

    for thr in THRESHOLDS:
        sig = WeekendUpGapShortSignal(threshold_atr=thr)
        sig_eval = sig.evaluate({"H4": panel})
        n_fires = sum(int(s.signal_mask.sum()) for s in sig_eval.per_pair.values())
        # quick pool sanity (characterization) — confirms the short pool builds + sign is right
        pool = build_arc_pool(sig, {"H4": panel}, ArcPoolConfig(
            arc_name=f"arc_2013_upgap_short_thr{thr}", sl_atr_mult=2.0, hold_bars=120, risk_pct=0.005,
            window_start=date(2010, 1, 1), window_end=date(2020, 12, 31)))
        fr = pool.trades["final_r"]
        print(f"\n##### threshold up-gap>=+{thr}  (fires={n_fires}, IS pool n={len(pool.trades)}, "
              f"mean final_r={fr.mean():+.3f}, win={float((fr>0).mean()):.3f}) #####")

        runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=sig_eval, panels={"H4": panel})
        best = None
        for exit_policy in EXITS:
            cfg = A1Config(config_id=f"arc2013_thr{thr}_{exit_policy}", exit_policy=exit_policy, sl_atr_mult=2.0)
            stats = run_config_over_folds(runner, is_folds, cfg)
            rois = [fs.roi_pct for fs in stats]
            v = judge_all_folds_positive(stats)
            mean_roi = float(np.mean(rois))
            n_neg = sum(1 for r in rois if r < 0)
            mintr = min(fs.n_trades for fs in stats)
            print(f"  {exit_policy:32s} AFP={str(v.all_folds_positive):5s} mean={mean_roi*100:+6.3f}% "
                  f"worst={v.worst_fold_roi*100:+6.3f}% neg={n_neg}/{len(rois)} min_trades/fold={mintr}")
            key = (v.all_folds_positive, mean_roi)
            if best is None or key > best[0]:
                best = (key, exit_policy, stats, mean_roi, v)

        # per-fold for the best config of this threshold
        _, ep, stats, mean_roi, v = best
        print(f"  --> BEST: {ep}  mean={mean_roi*100:+.3f}%  AFP={v.all_folds_positive}  worst={v.worst_fold_roi*100:+.3f}%")
        print("      per-fold ROI%:", ", ".join(f"{fs.fold_id}:{fs.roi_pct*100:+.2f}({fs.n_trades})" for fs in stats))


if __name__ == "__main__":
    main()
