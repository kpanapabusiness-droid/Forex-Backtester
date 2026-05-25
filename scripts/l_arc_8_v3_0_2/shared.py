"""Arc 8 v3.0.2 — shared constants and helpers across step scripts.

Single source of truth for: pair set, data window, cache paths, signal
config, fixed risk/SL/forward-window parameters, boundary convention.
Each step script imports from here so a parameter change ripples through
the whole pipeline.
"""

from __future__ import annotations

from pathlib import Path

from core.signals.pullback_resume_hhhl import PullbackResumeParams

ARC_NAME: str = "l_arc_8_v3.0.2"
ARC_ID: str = "l_arc_8_v3_0_2"
RESULTS_DIRNAME: str = "l_arc_8_v3.0.2"

# Boundary convention — locked at 5ers_eet end-to-end per corrected
# dispatch §0.5 + Amendment 6. No UTC fallback. No KH-24 anchor parity
# justification (this isn't a KH-24 anchor test).
BOUNDARY_CONVENTION: str = "5ers_eet"

# 28 FX pairs — KH-24 set, matches Arc 8 original.
PAIRS_28: tuple[str, ...] = (
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
)

# Data window — v3.0.2 extends to clean year-end (2025-12-31). Matches
# L_PROTOCOL §2 Step 5 IS+holdout convention (2010-2020 IS / 2021-present
# holdout, one-shot).
WINDOW_START: str = "2010-01-01"
WINDOW_END_TARGET: str = "2025-12-31"

# Risk and trade construction (identical to Arc 8 original).
RISK_PCT: float = 0.005          # 0.5% — L_PROTOCOL v3 default
SL_ATR_MULT_STEP1: float = 2.0   # Step 1 pool-build SL
FORWARD_BARS: int = 240          # 240 H4 bars = 40 days forward window

# SL sweep at Step 3 — Arc 8 original set, ratified at prior intent §1 F3.
SL_SWEEP: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

# Caches — PR-E.2 temp cache root (pre-warmed with all 28 pairs of
# H4_5ers_eet / D1_5ers_eet / W1_5ers_eet from the prior _halted run).
# aggregator chooses convention-appropriate subpath: with
# boundary_convention="5ers_eet" → <CACHE_ROOT>/<TF>_5ers_eet/<PAIR>.parquet.
CACHE_ROOT: Path = Path(r"C:/Users/panap/AppData/Local/Temp/pr_e2_cache")
HISTDATA_ROOT: Path = Path("data/histdata")

# Result paths — dot-suffixed dirname (matches dispatch + closure template).
RESULTS_ROOT: Path = Path("results") / RESULTS_DIRNAME

# Feature cache (PR-D contract). Per-arc namespace.
FEATURE_CACHE_ROOT: Path = Path("data/cache/features") / ARC_ID

# Signal params (locked at arc-open per spec; identical to Arc 8).
SIGNAL_PARAMS: PullbackResumeParams = PullbackResumeParams()

# Stable signal_def string — drives feature-cache key (PR-D).
SIGNAL_DEF: str = (
    "pullback_resume_hhhl_long_v0.1"
    f"|swing={SIGNAL_PARAMS.swing_lookback}"
    f"|trend_window={SIGNAL_PARAMS.trend_window_bars}"
    f"|right_edge_lag={SIGNAL_PARAMS.right_edge_lag}"
    f"|pullback_atr_mult={SIGNAL_PARAMS.pullback_atr_mult}"
    f"|upper_half={SIGNAL_PARAMS.upper_half_threshold}"
    f"|spacing={SIGNAL_PARAMS.spacing_bars}"
    f"|atr_period={SIGNAL_PARAMS.atr_period}"
)

# Step 4 — PR-#185 holdout-window training filter cutoff.
TRAIN_END_ISO: str = "2021-01-01T00:00:00Z"
