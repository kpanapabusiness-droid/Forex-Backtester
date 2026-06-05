"""arc 2013 — fair same-side same-exit NULL for the UP-gap SHORT (thr 1.0), JPY crosses (arc-1009 std).

The real up-gap short is mean-positive net of costs under overshoot exits (trailing_atr +0.745%,
tp_3r +0.510%) but NOT all-folds-positive. The PORTFOLIO-vs-KILL decider (§11): does it BEAT a fair
random-entry SHORT null at the same fire-count + same exit + same crosses? (A random JPY-cross short
should LOSE — the JPY-basket drifts UP against shorts — so beating it is the real test.)
"""
from __future__ import annotations

import numpy as np

from core.sim.panel import Panel
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import run_config_over_folds

from discovery.tools.gap_signals import WeekendUpGapShortSignal
from discovery.tools.null_entry_baseline import build_null_signal_evaluation

CROSSES = ["EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"]
BACKUP = r"C:\Users\panap\histdata_backup"


def mean_roi(runner, folds, cfg):
    stats = run_config_over_folds(runner, folds, cfg)
    rois = [fs.roi_pct for fs in stats]
    return float(np.mean(rois)), sum(1 for r in rois if r < 0), len(rois)


def main() -> None:
    panel = Panel.from_pairs(CROSSES, tf="H4", histdata_root=BACKUP, cache_root="data/cache",
                             boundary_convention="5ers_eet")
    is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]
    sig_eval = WeekendUpGapShortSignal(threshold_atr=1.0).evaluate({"H4": panel})

    print("=== arc 2013 fair SHORT null (up-gap thr 1.0, JPY crosses, IS) ===")
    for exit_policy in ("sl_plus_trailing_atr", "sl_plus_tp_3r"):
        cfg = A1Config(config_id=f"arc2013_null_{exit_policy}", exit_policy=exit_policy, sl_atr_mult=2.0)
        real_runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=sig_eval, panels={"H4": panel})
        real_mean, real_neg, nf = mean_roi(real_runner, is_folds, cfg)
        null_means = []
        for seed in (42, 7, 123, 2024, 99):
            null_eval = build_null_signal_evaluation(sig_eval, seed=seed, warmup=100)
            null_runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=null_eval, panels={"H4": panel})
            nm, _, _ = mean_roi(null_runner, is_folds, cfg)
            null_means.append(nm)
        null_avg = float(np.mean(null_means))
        print(f"\n  exit={exit_policy}")
        print(f"    REAL up-gap short: mean={real_mean*100:+.3f}%  ({real_neg}/{nf} neg)")
        print(f"    NULL random short: mean={null_avg*100:+.3f}%  (5 seeds: "
              + ", ".join(f"{m*100:+.2f}" for m in null_means) + ")")
        print(f"    LIFT real - null = {(real_mean-null_avg)*100:+.3f}%")


if __name__ == "__main__":
    main()
