"""Arc 8 — shared constants and helpers across step scripts.

Single source of truth for: pair set, data window, cache paths, signal
config, fixed risk/SL/forward-window parameters. Each step script imports
from here so a parameter change ripples through the whole pipeline.
"""

from __future__ import annotations

from pathlib import Path

from core.signals.pullback_resume_hhhl import PullbackResumeParams

ARC_NAME: str = "l_arc_8"
ARC_ID: str = "l_arc_8"

# 28 FX pairs — same set as KH-24, per dispatch §header + signal spec.
PAIRS_28: tuple[str, ...] = (
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
)

# Data window — dispatch §header wins per chat F2.
# Holdout end is data-limited (HistData runs to 2026-04-10 at the time of this
# arc). The v3 fold builder's "last complete month" default would land on
# 2026-04-30; effective ceiling is whatever the H4 parquet contains.
WINDOW_START: str = "2010-01-01"
WINDOW_END_TARGET: str = "2026-04-30"

# Risk and trade construction.
RISK_PCT: float = 0.005          # 0.5% — dispatch §header + signal spec
SL_ATR_MULT_STEP1: float = 2.0   # dispatch §"Step 1"
FORWARD_BARS: int = 240          # spec "Forward window" — 240 4H bars = 40 days

# SL sweep at Step 3 — dispatch's set (chat F3).
SL_SWEEP: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

# Caches — H4/D1/H1 already populated under the PR-E.2 temp cache from PR-E
# anchor work. Step 1 reads from here.
CACHE_ROOT: Path = Path(r"C:/Users/panap/AppData/Local/Temp/pr_e2_cache")
HISTDATA_ROOT: Path = Path("data/histdata")

# Result paths.
RESULTS_ROOT: Path = Path("results") / ARC_NAME

# Feature cache (PR-D contract).
FEATURE_CACHE_ROOT: Path = Path("data/cache/features") / ARC_ID

# Signal params (locked at arc-open per spec).
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
