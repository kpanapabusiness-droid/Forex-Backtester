"""Arc 1025 §5f honest-engine — does a DENSER fbr resolve folds / beat null / clear 2018?

The obs sweep (arc1025_fbr_density_sweep.py) showed the fbr edge degrades MONOTONICALLY with
density; the one genuine "denser but still non-coin-flip" cell is K=40, shadow=1.00 (n=507,
~2x the committed ref's 237; cap 0.5227, struct +0.044, drift +0.098). §5f mandates the honest
exit-menu sweep before a FAIL on a non-coin-flip entry.

Question (HYP-A / arc-2017 option B): is the DENSER fbr a fold-RESOLVING thicker component?
A thicker component helps the book ONLY if its extra trades shrink per-fold SE faster than the
weaker mean grows it (i.e. SE/mean improves). Run the candidate AND the committed ref through
the SAME exit menu on the SAME IS folds + the fair same-exit null; compare:
  - all-folds-positive count, 2018 fold ROI (HYP-B), null-beat margin,
  - mean / across-fold-SD (the fold-resolution proxy: |mean| vs SD).

risk_pct = 0.005 (= 0.5% deployable, arc 1024 convention; reproduces corpus headlines). IS only.
"""
from __future__ import annotations

import numpy as np

from core.sim.panel import Panel
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import run_config_over_folds

from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal
from discovery.tools.null_entry_baseline import build_null_signal_evaluation

HISTDATA = r"C:\Users\panap\histdata_backup"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
IS_FOLDS = [f for f in build_v3_folds().folds if f.is_days >= 365]
FOLD_IDS = [f.fold_id for f in IS_FOLDS]
RISK_PCT = 0.005

# the 6 registered exits; trailing-atr also in its committed trail_enabled=True form (fbr's best)
EXITS = [
    ("sl_only", False), ("sl_plus_tp_2r", False), ("sl_plus_tp_3r", False),
    ("sl_plus_trailing_atr", False), ("sl_plus_trailing_atr", True),
    ("sl_plus_trailing_swing", False), ("sl_partial_close_1r_runner_trail", False),
]
CELLS = {"ref K40 s1.25": (40, 1.25), "DENSER K40 s1.00": (40, 1.00)}

print("Loading H4 USD majors panel ...")
panel = Panel.from_pairs(PAIRS, tf="H4", histdata_root=HISTDATA,
                         cache_root="data/cache", boundary_convention="5ers_eet")


def run_eval(ev, exit_policy, trail):
    cfg = A1Config(config_id="arc1025", sl_atr_mult=2.0, trail_enabled=trail,
                   risk_pct=RISK_PCT, exit_policy=exit_policy)
    runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=ev, panels={"H4": panel})
    stats = run_config_over_folds(runner, IS_FOLDS, cfg)
    return np.array([fs.roi_pct for fs in stats], float) * 100.0, np.array([fs.n_trades for fs in stats], int)


for cell, (K, SH) in CELLS.items():
    print("\n" + "=" * 100)
    print(f"CELL: {cell}   (swing_lookback={K}, min_shadow_atr={SH})")
    print("=" * 100)
    sig = FailedBreakdownReclaimLongSignal(swing_lookback=K, min_shadow_atr=SH)
    ev = sig.evaluate({"H4": panel})
    best = None
    for exit_policy, trail in EXITS:
        rois, ntr = run_eval(ev, exit_policy, trail)
        mean, sd = rois.mean(), rois.std(ddof=1)
        npos = int((rois > 0).sum())
        roi18 = rois[FOLD_IDS.index(2018)] if 2018 in FOLD_IDS else float("nan")
        tag = f"{exit_policy}{'+trail' if trail else ''}"
        # fold-resolution proxy: |mean|/SD (higher = more resolvable)
        resolv = abs(mean) / sd if sd > 0 else float("nan")
        print(f"  {tag:34s} mean={mean:+.3f}% {npos:2d}/{len(rois)} pos  2018={roi18:+.3f}%  "
              f"SD={sd:.3f}%  mean/SD={resolv:.3f}  ntot={ntr.sum()}")
        if best is None or mean > best[1]:
            best = (tag, mean, npos, rois, exit_policy, trail)
    # fair same-exit NULL under the cell's best exit
    btag, bmean, bnpos, brois, bexit, btrail = best
    null_ev = build_null_signal_evaluation(ev, seed=42)
    nrois, _ = run_eval(null_ev, bexit, btrail)
    print(f"  --> BEST exit: {btag}  mean={bmean:+.3f}%  {bnpos}/{len(brois)} pos")
    print(f"      per-fold ROI%: {dict(zip(FOLD_IDS, np.round(brois,3).tolist()))}")
    print(f"      fair NULL ({btag}) mean={nrois.mean():+.3f}%  -> real-minus-null = {bmean-nrois.mean():+.3f}pp")
