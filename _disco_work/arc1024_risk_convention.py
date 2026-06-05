"""Arc 1024 — DIAGNOSTIC: resolve arc-3017 FLAG-1 (the risk_pct "unit split").

arc 3017 FLAG-1 claims: `A1Config.risk_pct` is in PERCENT (0.5=0.5%), so passing 0.005 makes ROI
"100x-compressed" (0.005%), and the portfolio verdict is "risk-convention-dependent" because at risk
0.5 the daily-DD cap flips fold signs vs low-risk.

The code says otherwise: `core/sim/risk/live_balance.py` sizes `risk_amount = balance * risk_pct`
(default 0.01 = "1% per trade") — risk_pct is a FRACTION, no x100 anywhere. Prediction: at small
risk (no DD-cap), ROI scales ~LINEARLY in risk_pct, so 0.005 = 0.5% gives the committed headlines at a
NORMAL deployable risk; the "risk 0.5 explosion" is simply 50% risk blowing the account / DD cap, NOT a
convention ambiguity. This driver maps fbr's per-fold ROI across risk_pct to settle it.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from core.sim.panel import Panel
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.wfo.folds import build_v3_folds
from core.wfo.discovery_measure import run_config_over_folds
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal

HISTDATA = r"C:\Users\panap\histdata_backup"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
IS_FOLDS = [f for f in build_v3_folds().folds if f.is_days >= 365]

panel = Panel.from_pairs(PAIRS, tf="H4", histdata_root=HISTDATA,
                         cache_root="data/cache", boundary_convention="5ers_eet")
ev = FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": panel})

print("fbr (committed config: sl_plus_trailing_atr, trail_enabled=True) per-fold ROI vs risk_pct")
print("(committed HEADLINE = +1.854% mean, 9/10 pos, worst 2018 -4.196% — at risk_pct=0.005)")
print("="*90)
base = None
for rp in [0.0025, 0.005, 0.01, 0.02, 0.05, 0.5]:
    cfg = A1Config(config_id=f"fbr_rp{rp}", sl_atr_mult=2.0, trail_enabled=True,
                   risk_pct=rp, exit_policy="sl_plus_trailing_atr")
    runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=ev, panels={"H4": panel})
    rois = np.array([fs.roi_pct for fs in run_config_over_folds(runner, IS_FOLDS, cfg)], float) * 100
    mean, npos, worst = rois.mean(), int((rois > 0).sum()), rois.min()
    ratio = "" if base is None else f"  mean/base_per_unit_risk = {mean/(rp/0.0025):+.4f}% (linear if ~const)"
    if base is None:
        base = mean
    print(f"risk_pct={rp:<7} ({rp*100:g}% risk)  mean={mean:+9.3f}%  {npos}/10 pos  worst={worst:+9.3f}%{ratio}")
print("="*90)
print("If mean scales ~linearly across 0.0025-0.02 (no DD cap), risk_pct is a FRACTION and the")
print("0.005 headline is at 0.5% deployable risk. Nonlinearity should appear only at large risk (DD cap).")
