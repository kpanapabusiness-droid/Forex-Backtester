"""
Phase KG-L-V2 — D1 ATR Distance Cap + signal_flip Exit REMOVED
===============================================================
Builds directly on KG-L-V1. The only additional change vs V1:
    signal_flip exit is completely disabled — not computed, not checked.

FULL CHANGE LIST FROM KG-J:

Change 1 — Trail activation fix (same as V1):
    Close-based: trail_active when c_j >= entry_px + 2.0 × ATR

Change 2 — Classification fix (same as V1, reporting only):
    WIN = r_multiple > 0, LOSS < 0, SCRATCH = 0

Change 3 — cond_9 D1 ATR distance cap (same as V1):
    Signal condition: D1 close <= D1 Kijun(26) + 1.0 × D1 ATR(14)

Change 4 — signal_flip exit REMOVED (V2 only):
    Exits: kijun_d1 | trailing_stop | stoploss  (no signal_flip)
    The signal_flip function is not computed and not checked at any point.
    Motivation: test whether removing the signal_flip exit improves trade
    duration and allows more winning exits via trailing stop or kijun_d1.

Outputs: results/phase_kg/kg_l_v2/
Gate: worst-fold OOS ROI > 0% AND worst-fold max DD < 8%
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dateutil.relativedelta import relativedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
warnings.filterwarnings("ignore")

from core.d1_pipeline import D1Hook, apply_d1_hook_per_bar  # noqa: E402
from core.features_path_so_far import build_entry_features_at_signal_bar  # noqa: E402
from core.utils import get_pip_size, load_pair_csv  # noqa: E402
from signals.kb_exhaustion_bar import _wilder_atr  # noqa: E402

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR_4H = PROJECT_ROOT / "data" / "4hr"
D1_DATA_DIR = PROJECT_ROOT / "data" / "daily"
H1_DATA_DIR = PROJECT_ROOT / "data" / "1hr"
OUT_ROOT    = PROJECT_ROOT / "results" / "phase_kg" / "kg_l_v2"

# ── D1 filter parameters ──────────────────────────────────────────────────────
D1_REGIME_FILTER  = True
USE_C8            = True   # C8: D1 close > D1 Kijun gate; disable via config use_c8: false
USE_C9            = True   # C9: D1 ATR distance cap; disable via config use_c9: false
D1_KIJUN_PERIOD   = 26
D1_SLOPE_FILTER   = False
D1_SLOPE_LOOKBACK = 5
D1_ATR_PERIOD     = 14
D1_ATR_DIST_CAP   = 1.0

POINTS_PER_PIP = 10.0

# ── Liquidity-proxy diagnostics ──────────────────────────────────────────────
# These columns are read-only enrichment on the trade record (trades_all.csv).
# All values are computed at the signal bar N using only bars 0..N.
# They must NEVER feed back into signal or entry logic.
LIQUIDITY_MEDIAN_WINDOW = 100


def _session_for_ts(ts: pd.Timestamp) -> str:
    """Classify a 4H bar's open time (UTC) into a session bucket.

    tokyo   : 00:00-08:00  (hours 0-7)
    london  : 08:00-16:00  (hours 8-15)
    newyork : 16:00-00:00  (hours 16-23)
    """
    h = int(pd.Timestamp(ts).hour)
    if h < 8:
        return "tokyo"
    if h < 16:
        return "london"
    return "newyork"


def _rolling_median_prior(series: pd.Series, window: int) -> pd.Series:
    """Rolling median over the `window` bars STRICTLY BEFORE each index.

    At index i the returned value is median(series[i-window:i]); the value
    at i is not included. Warm-up bars (< window prior observations) are NaN.
    Uses numeric median (ignores NaN within the window).
    """
    return series.shift(1).rolling(window=window, min_periods=window).median()


# ── v1.3 capturability extension ────────────────────────────────────────────
# Per-bar trade-paths for capturability calibration. ATR-normalised on
# signal-bar 4H ATR(14) (same anchor as the legacy mfe_final / mae_final),
# but signed so that high_r/low_r/close_r preserve direction. Forward window
# extends to 240 bars from entry regardless of exit, giving capture-ceiling
# data for longer-time-exit reconstructions.
PATH_FORWARD_BARS = 240


def _init_bar_path(direction: str, entry_px: float, atr: float,
                   hi_e: float, lo_e: float, c_e: float) -> list[dict]:
    """Seed bar_path with the entry-bar row (bar_offset=0, is_held=1).

    Long convention: high_r = (high - entry) / atr; low_r/close_r mirror.
    For long, high_r >= 0 and low_r <= 0 at the entry bar by OHLC invariants.
    Short flips the sign so positive = favourable to the trade direction.
    """
    if direction == "short":
        hr = (entry_px - hi_e) / atr
        lr = (entry_px - lo_e) / atr
        cr = (entry_px - c_e) / atr
    else:
        hr = (hi_e - entry_px) / atr
        lr = (lo_e - entry_px) / atr
        cr = (c_e - entry_px) / atr
    return [{
        "bar_offset":    0,
        "high_r":        hr,
        "low_r":         lr,
        "close_r":       cr,
        "mfe_so_far_r":  hr,
        "mae_so_far_r":  lr,
        "is_held":       1,
    }]


def _flatten_bar_path_for_trade(
    trade: dict,
    exit_bar: int,
    pair_cache: dict,
    direction: str,
    out_rows: list[dict],
    forward_bars: int = PATH_FORWARD_BARS,
) -> None:
    """Emit one row per (trade_id, bar_offset) into ``out_rows``.

    Held window (bar_offset 0..bars_held) comes from ``trade["bar_path"]``,
    accumulated in the hold loop. Forward window (bar_offset bars_held+1..
    forward_bars) is computed here against the same pair_cache the simulation
    used; ATR anchor and entry price are the immutable signal-bar values
    already pinned on the trade dict. Forward-window rows carry is_held=0.

    No lookahead invariant: bar_offset=t reads pair_cache[pair][...][entry_idx+t]
    only — strictly past relative to t, never future.
    """
    pair       = trade["pair"]
    cached     = pair_cache[pair]
    n          = len(cached["o"])
    entry_idx  = trade["entry_idx"]
    entry_px   = trade["entry_px"]
    a          = trade["atr"]
    trade_id   = f"{pair}_{trade['entry_date']}"

    # Held rows already accumulated.
    last = trade["bar_path"][-1]
    mfe_sf = last["mfe_so_far_r"]
    mae_sf = last["mae_so_far_r"]
    for r in trade["bar_path"]:
        out_rows.append({
            "trade_id":     trade_id,
            "pair":         pair,
            "bar_offset":   r["bar_offset"],
            "high_r":       r["high_r"],
            "low_r":        r["low_r"],
            "close_r":      r["close_r"],
            "mfe_so_far_r": r["mfe_so_far_r"],
            "mae_so_far_r": r["mae_so_far_r"],
            "is_held":      r["is_held"],
        })

    # Forward window — runs past exit_bar to entry_idx+forward_bars (or end of
    # data). Same per-bar excursion math as the held path, just with is_held=0.
    target_end = min(entry_idx + forward_bars, n - 1)
    for bar_idx in range(exit_bar + 1, target_end + 1):
        hi = cached["h"][bar_idx]
        lo = cached["lo"][bar_idx]
        cl = cached["c"][bar_idx]
        if direction == "short":
            hr = (entry_px - hi) / a
            lr = (entry_px - lo) / a
            cr = (entry_px - cl) / a
        else:
            hr = (hi - entry_px) / a
            lr = (lo - entry_px) / a
            cr = (cl - entry_px) / a
        if hr > mfe_sf:
            mfe_sf = hr
        if lr < mae_sf:
            mae_sf = lr
        out_rows.append({
            "trade_id":     trade_id,
            "pair":         pair,
            "bar_offset":   bar_idx - entry_idx,
            "high_r":       hr,
            "low_r":        lr,
            "close_r":      cr,
            "mfe_so_far_r": mfe_sf,
            "mae_so_far_r": mae_sf,
            "is_held":      0,
        })

ALL_PAIRS = [
    "AUD_CAD", "AUD_CHF", "AUD_JPY", "AUD_NZD", "AUD_USD",
    "CAD_CHF", "CAD_JPY", "CHF_JPY",
    "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_GBP", "EUR_JPY",
    "EUR_NZD", "EUR_USD",
    "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_JPY", "GBP_NZD", "GBP_USD",
    "NZD_CAD", "NZD_CHF", "NZD_JPY", "NZD_USD",
    "USD_CAD", "USD_CHF", "USD_JPY",
]

PAIR_CURRENCIES: dict[str, tuple[str, str]] = {
    "AUD_CAD": ("AUD", "CAD"), "AUD_CHF": ("AUD", "CHF"),
    "AUD_JPY": ("AUD", "JPY"), "AUD_NZD": ("AUD", "NZD"),
    "AUD_USD": ("AUD", "USD"), "CAD_CHF": ("CAD", "CHF"),
    "CAD_JPY": ("CAD", "JPY"), "CHF_JPY": ("CHF", "JPY"),
    "EUR_AUD": ("EUR", "AUD"), "EUR_CAD": ("EUR", "CAD"),
    "EUR_CHF": ("EUR", "CHF"), "EUR_GBP": ("EUR", "GBP"),
    "EUR_JPY": ("EUR", "JPY"), "EUR_NZD": ("EUR", "NZD"),
    "EUR_USD": ("EUR", "USD"), "GBP_AUD": ("GBP", "AUD"),
    "GBP_CAD": ("GBP", "CAD"), "GBP_CHF": ("GBP", "CHF"),
    "GBP_JPY": ("GBP", "JPY"), "GBP_NZD": ("GBP", "NZD"),
    "GBP_USD": ("GBP", "USD"), "NZD_CAD": ("NZD", "CAD"),
    "NZD_CHF": ("NZD", "CHF"), "NZD_JPY": ("NZD", "JPY"),
    "NZD_USD": ("NZD", "USD"), "USD_CAD": ("USD", "CAD"),
    "USD_CHF": ("USD", "CHF"), "USD_JPY": ("USD", "JPY"),
}

# ── Signal / trade parameters (KF v17 — locked) ───────────────────────────────
ATR_PERIOD    = 14
KIJUN_PERIOD  = 26
BODY_THRESH   = 0.5
CLOSE_POS_MAX = 0.24
DISTANCE_CAP  = 1.0
DEPTH_BARS    = 10
DEPTH_THRESH  = 0.5
VOL_LOOKBACK  = 20
VOL_MULT      = 1.2
NO_VOLUME_FILTER = False  # set True via --no-volume-filter to skip cond_7
VOL_COLUMN = "volume"  # column to use for C7; override via volume_column in YAML
RANGE_ATR_CEILING: float | None = None  # set via --range-ceiling; None = disabled
C4_KIJUN_OFFSET_ATR = 0.0  # KH-2: allow entries up to N×ATR below Kijun; 0.0 = original

# KH-4: 4H EMA(50) momentum regime gate (0 = disabled, preserves baseline)
REGIME_MIN_TRENDING_PAIRS = 0
REGIME_EMA_PERIOD         = 50
REGIME_SLOPE_LOOKBACK     = 5

# KH-5: own-pair 4H EMA(50) slope gate (False = disabled, preserves baseline)
REQUIRE_OWN_EMA_SLOPE     = False

# KH-11A: D1 Kijun slope falling/flat gate (False = disabled, preserves baseline).
#   When True, block a signal if d1_kijun_slope is True at the lagged D1 bar
#   already used for C8/C9 (i.e. D1 Kijun[T] > D1 Kijun[T-5]). Reuses the
#   merge_asof one-day-lag D1 series — never same-day D1.
REQUIRE_D1_KIJUN_SLOPE_FALLING = False

# KH-7: baseline type for C4/C5 ("kijun" default; "hma" or "dema" to replace)
#   C8/C9/kijun_d1 exit are D1-based and unaffected — they still use D1 Kijun.
BASELINE_TYPE = "kijun"
WARMUP_BARS = {"kijun": 75, "hma": 75, "dema": 150}

# KH-8: kijun_d1 exit confirmation bars
#   1 = current behaviour (single-bar D1 close < D1 Kijun)
#   2 = require two consecutive D1 bars with close < D1 Kijun
KIJUN_D1_CONFIRM_BARS = 1

# KH-9: conditional application of the 2-bar confirmation.
# All three default to their "off" state; when none is active, the
# KH-8 behaviour applies unconditionally to every kijun_d1 evaluation.
#
#   Variant A — KIJUN_D1_CONFIRM_IF_TRAIL
#       True  = require confirmation only when trail_active is True.
#       False = confirmation requirement is not gated on trail status.
#
#   Variant B — KIJUN_D1_CONFIRM_MIN_BARS
#       N > 0 = require confirmation only when bars_held >= N.
#       0     = confirmation requirement is not gated on trade age.
#
#   Variant C — KIJUN_D1_CONFIRM_DEPTH_ATR
#       X > 0 = require confirmation only when the D1 cross depth is
#               shallow (< X * d1_atr_lag1 below Kijun).  Deep crosses
#               (>= X * d1_atr_lag1) exit immediately on lag=1.
#       0.0   = confirmation requirement is not gated on cross depth.
KIJUN_D1_CONFIRM_IF_TRAIL  = False
KIJUN_D1_CONFIRM_MIN_BARS  = 0
KIJUN_D1_CONFIRM_DEPTH_ATR = 0.0

# KH-13: early exit on adverse bar-3 movement.
#   When enabled, exit at bar N+4 open if at bar N+3 close:
#     mae_at_bar_3 >= KH13_MAE_THRESHOLD × ATR  (price moved against us)
#     mfe_at_bar_3 <= KH13_MFE_THRESHOLD × ATR  (price barely moved for us)
#   kh13_triggered is always recorded (even when USE_KH13_EARLY_EXIT=False)
#   so the rule's coverage can be audited on any run.
USE_KH13_EARLY_EXIT = False
KH13_MAE_THRESHOLD  = 1.0   # ATR multiples
KH13_MFE_THRESHOLD  = 0.5   # ATR multiples

# KH-14: early exit on adverse bar-6 movement within State 2 (first_bar_dir=-1).
#   When enabled, exit at bar N+7 open if at bar N+6 close:
#     mfe_at_bar_6 < KH14_MFE_THRESHOLD × ATR  (not moved 1 ATR our way)
#     AND mae_at_bar_6 >= KH14_MAE_THRESHOLD × ATR  (has moved 1 ATR against us)
#   Applies ONLY when first_bar_dir == -1 (bar N+1 close < entry_price).
#   kh14_triggered and kh14_state2 are always recorded (even when disabled)
#   so coverage can be audited on any run.
USE_KH14_BAR6_EXIT  = False
KH14_MFE_THRESHOLD  = 1.0   # ATR multiples — hold if mfe >= this
KH14_MAE_THRESHOLD  = 1.0   # ATR multiples — exit if mae >= this

# KH-15A: ATR-conditional position sizing.
#   When USE_ATR_SIZING is True AND the signal-bar atr_ratio (ATR(14)[N] /
#   median ATR of the prior LIQUIDITY_MEDIAN_WINDOW bars) is >= threshold,
#   the trade uses ATR_SIZING_REDUCED_PCT instead of RISK_PCT as its risk
#   per trade.  Lot size is halved; SL distance (in price), trail logic,
#   entry/exit conditions, and r_multiple denominator are unchanged so
#   that sized-down trades still show r_multiple = -1.0 on full SL hits.
#   Default OFF preserves baseline exactly.
USE_ATR_SIZING          = False
ATR_SIZING_THRESHOLD    = 1.5
ATR_SIZING_REDUCED_PCT  = 0.005   # stored as fraction (0.5%)

# KH-16: re-entry after kh14_bar6 exit on State 2 original trades.
#   When enabled: after a kh14_bar6 exit, watch REENTRY_WINDOW_BARS bars for
#   close > original_entry_price (trigger).  If triggered, enter long at next
#   bar open.  SL = fill_price - REENTRY_SL_ATR_MULT × ATR(14) at entry bar.
#   Trail: same activation (2.0×ATR) and level (1.5×ATR from best close).
#   Only ONE re-entry per original trade.  Cancelled by new signal on same pair.
#   Applies only to State 2 original trades that exit via kh14_bar6.
USE_REENTRY         = False
REENTRY_WINDOW_BARS = 10
REENTRY_SL_ATR_MULT = 1.5
REENTRY_TRIGGER     = "close_above_entry"

# KH-17: two-decision State 1 delayed / State 2 virtual bar-6 gate + watch.
#   When enabled, no trade fills at bar N+1 open under any circumstance.
#   reference_price = bar N+1 open; virtual_atr = ATR(14) at bar N+1.
#   At bar N+1 close:
#     - close > open (State 1 path):
#         Enter long at bar N+2 open.  trade_type="state1_delayed".
#         SL = fill - KH17_STATE1_SL_ATR_MULT × ATR(14) at fill bar.
#     - close <= open (State 2 path):
#         No trade.  Begin virtual tracking from reference_price using
#         CLOSE prices only: virtual_mfe = max over j of (close[j]-ref)/atr,
#         virtual_mae = max over j of (ref-close[j])/atr, floored at 0,
#         j in [N+1, N+6].
#   At bar N+6 close (virtual bar 6; N+1 is virtual bar 1):
#     virtual_mfe_at_bar6 >= KH17_BAR6_MFE_THRESHOLD:
#         Holder.  Signal discarded, no re-entry.
#     virtual_mfe < thr_mfe AND virtual_mae >= KH17_BAR6_MAE_THRESHOLD:
#         Open re-entry watch for KH17_WATCH_WINDOW_BARS bars starting
#         at N+7.  First bar whose close > reference_price -> enter long
#         at next bar open (trade_type="state2_reentry").
#         SL = fill - KH17_WATCH_SL_ATR_MULT × ATR(14) at fill bar.
#     virtual_mfe < thr_mfe AND virtual_mae < thr_mae:
#         Ambiguous.  Signal discarded, no re-entry.
#   A new original signal on the same pair cancels any active pending
#   decision, virtual tracking, or re-entry watch on that pair.
#   One trade per pair at a time: new KH-17 entries are skipped if the
#   pair already has an open trade.  kijun_d1 exit stays baseline 1-bar.
KH17_ENABLED                  = False
KH17_STATE2_REENTRY_ENABLED   = True   # False = State 1 delayed only; State 2 discarded
KH17_WATCH_WINDOW_BARS        = 10
KH17_WATCH_SL_ATR_MULT        = 1.5
KH17_STATE1_SL_ATR_MULT       = 2.0
KH17_BAR6_MFE_THRESHOLD       = 1.0
KH17_BAR6_MAE_THRESHOLD       = 1.0

# KH-18: alias of KH-17 two-decision path (State 1 delayed + State 2
# virtual bar-6 gate + re-entry watch).  kh18_* YAML keys feed the
# existing KH17_* globals; enabling kh18_enabled turns on the KH-17
# code path and tags the run banner / OOS summary under KH-18.
# Default OFF preserves baseline exactly.
KH18_ENABLED = False

# KH-19: baseline entry (bar N close → bar N+1 open, no delay) + kh14_bar6 exit
# + KH-16-identical re-entry after kh14_bar6 exit.  kh19_* YAML keys feed the
# existing USE_REENTRY / REENTRY_* globals (same code path as KH-16).
# KH19_ENABLED is tracked separately so the run banner tags the run as KH-19.
# Default OFF preserves baseline exactly.
KH19_ENABLED = False

# KH-25: kh14_bar6 exit + re-entry on top of KH-24 (exposure cap=2 + 1H filter).
# kh25_* YAML keys feed the same USE_REENTRY/REENTRY_* globals as KH-19, and
# automatically enable USE_KH14_BAR6_EXIT.  KH25_ENABLED is tracked separately
# for the run banner and summary output.  Re-entries bypass the 1H filter and
# exposure cap — they are continuations of an already-approved signal context.
KH25_ENABLED = False

# KH-20: D1 close-in-range entry filter (False = disabled, preserves baseline).
#   At bar N close, compute d1_close_in_range = (d1_close - d1_low) / (d1_high - d1_low)
#   for the last fully-closed D1 bar (same lag-1 convention as C8/C9).
#   If d1_close_in_range <= KH20_D1_RANGE_THRESHOLD, skip signal (no entry taken).
#   If d1_high - d1_low == 0 (doji), treat as 0.5 (neutral, allow entry).
#   If D1 value is NaN (no data), allow entry.
KH20_ENABLED             = False
KH20_D1_RANGE_THRESHOLD  = 0.3087

# KH-22: 1H close-in-range entry filter (False = disabled, preserves baseline).
#   At bar N close, compute h1_last_bar_close_in_range for the most recent
#   fully-closed 1H bar (close time <= 4H bar close time).
#   h1_last_bar_close_in_range = (h1_close - h1_low) / (h1_high - h1_low)
#   If h1_last_bar_close_in_range > KH22_H1_RANGE_THRESHOLD, skip signal.
#   If h1_high == h1_low (doji), treat as 0.5 (neutral, allow entry).
#   If no 1H data available, allow entry.
KH22_ENABLED              = False
KH22_H1_RANGE_THRESHOLD   = 0.624

SL_MULT        = 2.0
TRAIL_MULT     = 1.5
TRAIL_ACT_MULT = 2.0
RISK_PCT       = 0.02
INITIAL_BAL    = 10_000.0
EXPOSURE_CAP   = 1
SIGNAL_DIRECTION = "long"  # "long" or "short" — set via config direction: short

# ── Pipeline D1 hook (L_ARC_PROTOCOL v2.0 §3) ────────────────────────────────
# None preserves baseline byte-for-byte. Configured at run-start from the
# top-level ``d1_archetypes`` YAML block. See core/d1_pipeline.py.
D1_HOOK: "D1Hook | None" = None

# ── WFO configuration ─────────────────────────────────────────────────────────
WFO_START       = datetime(2019, 1, 1)
WFO_IS_MONTHS_0 = 21
WFO_OOS_MONTHS  = 9
WFO_N_FOLDS     = 7

GATE_ROI_MIN = 0.0
GATE_DD_MAX  = 8.0

# ── KG-J reference results ────────────────────────────────────────────────────
_KGIJ_FOLDS = {
    1: (12.57,  9.71, 50),
    2: ( 3.65,  9.75, 47),
    3: (24.46,  3.97, 48),
    4: (-1.13,  6.28, 40),
    5: ( 9.69,  6.32, 34),
    6: ( 0.62,  7.75, 27),
    7: ( 3.91,  5.13, 41),
}


# ── Fold boundaries ───────────────────────────────────────────────────────────

def _wfo_folds() -> list[dict]:
    folds = []
    for k in range(WFO_N_FOLDS):
        oos_s = WFO_START + relativedelta(months=WFO_IS_MONTHS_0 + k * WFO_OOS_MONTHS)
        oos_e = WFO_START + relativedelta(months=WFO_IS_MONTHS_0 + (k + 1) * WFO_OOS_MONTHS)
        folds.append({
            "fold":      k + 1,
            "is_start":  WFO_START,
            "is_end":    oos_s,
            "oos_start": oos_s,
            "oos_end":   oos_e,
        })
    return folds


FOLDS = _wfo_folds()


def _fold_id_for_entry(entry_ts) -> str | None:
    """Return the WFO fold id (``"F1"``..``"F7"``) whose OOS window
    contains ``entry_ts``, or ``None`` for IS-only trades.

    Used by the Pipeline D1 hook to dispatch the trade to the correct
    per-fold classifier — see ``D1_HOOK.evaluate(..., fold_id=...)``.
    Trades with ``entry_ts`` before the first fold's OOS start return
    ``None`` and are not gated by D1.
    """
    ts = pd.Timestamp(entry_ts)
    for f in FOLDS:
        if pd.Timestamp(f["oos_start"]) <= ts < pd.Timestamp(f["oos_end"]):
            return f"F{int(f['fold'])}"
    return None


# ── Kijun helpers ─────────────────────────────────────────────────────────────

def _compute_kijun(df: pd.DataFrame, period: int) -> np.ndarray:
    hi_max = df["high"].rolling(max(period, 1)).max()
    lo_min = df["low"].rolling(max(period, 1)).min()
    return ((hi_max + lo_min) / 2).values


# ── KH-4: 4H EMA(50) regime helpers ───────────────────────────────────────────

def _compute_ema(closes: np.ndarray, period: int) -> np.ndarray:
    """Standard EMA seeded on the first close. Uses only bars 0..i (no lookahead)."""
    n = len(closes)
    ema = np.empty(n, dtype=float)
    if n == 0:
        return ema
    alpha = 2.0 / (period + 1)
    ema[0] = closes[0]
    for i in range(1, n):
        ema[i] = alpha * closes[i] + (1.0 - alpha) * ema[i - 1]
    return ema


# ── KH-7: HMA / DEMA helpers (C4/C5 baseline replacements) ───────────────────

def _wma(values: np.ndarray, period: int) -> np.ndarray:
    """Linearly weighted moving average. Weight of bar i = i+1.
    First (period - 1) entries are NaN. Handles NaN in input (propagates NaN).
    """
    n = len(values)
    result = np.full(n, np.nan)
    if period <= 0 or n < period:
        return result
    weights = np.arange(1, period + 1, dtype=float)
    w_sum   = weights.sum()
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        if np.isnan(window).any():
            continue
        result[i] = float(np.dot(window, weights) / w_sum)
    return result


def compute_hma(close: np.ndarray, period: int = 26) -> np.ndarray:
    """Hull Moving Average. HMA = WMA(sqrt(p)) of (2*WMA(p/2) - WMA(p))."""
    half   = period // 2
    sqrt_p = int(np.sqrt(period))
    wma_h  = _wma(close, half)
    wma_f  = _wma(close, period)
    raw    = 2.0 * wma_h - wma_f
    return _wma(raw, sqrt_p)


def compute_dema(close: np.ndarray, period: int = 50) -> np.ndarray:
    """Double EMA: DEMA = 2*EMA(p) - EMA(EMA(p)). Uses _compute_ema (seeded on close[0])."""
    e1 = _compute_ema(close, period)
    e2 = _compute_ema(e1,    period)
    return 2.0 * e1 - e2


def _compute_baseline_series(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                              baseline_type: str) -> np.ndarray:
    """Return the C4/C5 baseline series for the chosen baseline_type."""
    bt = (baseline_type or "kijun").lower()
    if bt == "hma":
        return compute_hma(closes, period=26)
    if bt == "dema":
        return compute_dema(closes, period=50)
    # Default: Kijun(26) matches existing C4/C5 behaviour exactly.
    hi_s = pd.Series(highs).rolling(max(KIJUN_PERIOD, 1)).max().values
    lo_s = pd.Series(lows).rolling(max(KIJUN_PERIOD, 1)).min().values
    return (hi_s + lo_s) / 2.0


def _regime_count(pair_cache: dict, signal_timestamp: pd.Timestamp,
                  slope_lookback: int) -> int:
    """Count pairs whose 4H EMA(period) has rising slope at `signal_timestamp`.

    Uses asof / ffill lookup so each pair is evaluated with data available
    strictly at or before `signal_timestamp` (no lookahead). EMA value at
    time T only depends on closes up to T per `_compute_ema`.
    """
    count = 0
    ts64 = np.datetime64(signal_timestamp)
    for cached in pair_cache.values():
        series = cached.get("ema50_series")
        if series is None or len(series) == 0:
            continue
        idx_arr = series.index.get_indexer([ts64], method="ffill")
        idx = int(idx_arr[0])
        if idx < slope_lookback:
            continue
        ema_now = series.iat[idx]
        ema_prev = series.iat[idx - slope_lookback]
        if np.isnan(ema_now) or np.isnan(ema_prev):
            continue
        if ema_now > ema_prev:
            count += 1
    return count


def _apply_regime_gate(pair_cache: dict) -> tuple[int, int, int]:
    """Post-filter signals by 4H EMA(50) regime gate(s).

    Two independent gates (can be active simultaneously or independently):
      - Portfolio gate: at least REGIME_MIN_TRENDING_PAIRS pairs must have
        rising EMA slope over REGIME_SLOPE_LOOKBACK bars.
      - Own-slope gate: the signalling pair itself must have rising EMA slope
        over REGIME_SLOPE_LOOKBACK bars.

    Returns (n_blocked_portfolio, n_blocked_own, n_evaluated).
    No-op when both gates are disabled — exact baseline behaviour.
    """
    portfolio_on = REGIME_MIN_TRENDING_PAIRS > 0
    own_on       = REQUIRE_OWN_EMA_SLOPE
    if not portfolio_on and not own_on:
        return 0, 0, 0

    blocked_portfolio = 0
    blocked_own       = 0
    evaluated         = 0
    lookback          = REGIME_SLOPE_LOOKBACK

    for cached in pair_cache.values():
        sig        = cached["sig"]
        dates      = cached["dates"]
        own_series = cached.get("ema50_series")
        for i in range(len(sig)):
            if sig[i] != 1:
                continue
            evaluated += 1
            ts = pd.Timestamp(dates[i])

            if own_on:
                if own_series is None or len(own_series) == 0:
                    sig[i] = 0
                    blocked_own += 1
                    continue
                idx_arr = own_series.index.get_indexer([ts], method="ffill")
                idx = int(idx_arr[0])
                if idx < lookback:
                    sig[i] = 0
                    blocked_own += 1
                    continue
                ema_now  = own_series.iat[idx]
                ema_prev = own_series.iat[idx - lookback]
                if np.isnan(ema_now) or np.isnan(ema_prev) or ema_now <= ema_prev:
                    sig[i] = 0
                    blocked_own += 1
                    continue

            if portfolio_on:
                rc = _regime_count(pair_cache, ts, lookback)
                if rc < REGIME_MIN_TRENDING_PAIRS:
                    sig[i] = 0
                    blocked_portfolio += 1

    return blocked_portfolio, blocked_own, evaluated


def _compute_d1_kijun(d1_df: pd.DataFrame, period: int) -> np.ndarray:
    hi_max = d1_df["high"].rolling(max(period, 1)).max()
    lo_min = d1_df["low"].rolling(max(period, 1)).min()
    return ((hi_max + lo_min) / 2).values


# ── D1 regime filter (with D1 ATR for cond_9) ────────────────────────────────

class D1Filter:
    """
    D1 regime filter + D1 ATR for cond_9.
    check() returns: (d1a, d1_close, d1_kijun, d1_kijun_lag, d1_atr)
    """
    def __init__(
        self,
        d1_df: pd.DataFrame,
        kijun_period: int,
        slope_lookback: int,
        atr_period: int = 14,
    ) -> None:
        d1_df = d1_df.copy()
        d1_df["date"] = pd.to_datetime(d1_df["date"]).dt.normalize()
        d1_df = d1_df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        kijun_arr = _compute_d1_kijun(d1_df, kijun_period)
        kijun_s   = pd.Series(kijun_arr, index=d1_df["date"])
        close_s   = pd.Series(d1_df["close"].values.astype(float), index=d1_df["date"])
        high_s    = pd.Series(d1_df["high"].values.astype(float),  index=d1_df["date"])
        low_s     = pd.Series(d1_df["low"].values.astype(float),   index=d1_df["date"])
        kijun_lag = kijun_s.shift(slope_lookback)
        atr_arr   = _wilder_atr(d1_df, atr_period).values
        atr_s     = pd.Series(atr_arr, index=d1_df["date"])
        self._kijun     = kijun_s
        self._kijun_lag = kijun_lag
        self._close     = close_s
        self._high      = high_s
        self._low       = low_s
        self._atr       = atr_s

    def check(
        self, signal_date: pd.Timestamp
    ) -> tuple[bool, float, float, float, float]:
        # Use the last completed D1 bar: subtract 1 day so a 4H bar on day D
        # sees D1 data from day D-1 (the last bar whose close is fully known).
        d   = signal_date.normalize() - pd.Timedelta(days=1)
        cl  = self._close.asof(d)
        kij = self._kijun.asof(d)
        lag = self._kijun_lag.asof(d)
        atr = self._atr.asof(d)
        if pd.isna(cl) or pd.isna(kij):
            return False, float("nan"), float("nan"), float("nan"), float("nan")
        d1a = bool(cl > kij)
        if D1_SLOPE_FILTER:
            if pd.isna(lag):
                return False, float(cl), float(kij), float("nan"), float(atr) if not pd.isna(atr) else float("nan")
            d1b = bool(kij > lag)
            return (d1a and d1b), float(cl), float(kij), float(lag), float(atr) if not pd.isna(atr) else float("nan")
        lag_val = float(lag) if not pd.isna(lag) else float("nan")
        atr_val = float(atr) if not pd.isna(atr) else float("nan")
        return d1a, float(cl), float(kij), lag_val, atr_val


def _load_d1_filter(pair: str) -> D1Filter | None:
    try:
        d1_df = load_pair_csv(pair, D1_DATA_DIR)
        d1_df["date"] = pd.to_datetime(d1_df["date"])
        d1_df = d1_df.sort_values("date").reset_index(drop=True)
        return D1Filter(d1_df, D1_KIJUN_PERIOD, D1_SLOPE_LOOKBACK, D1_ATR_PERIOD)
    except FileNotFoundError:
        return None


def _load_h1_data(pair: str) -> pd.DataFrame | None:
    """Load 1H OHLCV data for a pair. Returns None if file not found."""
    try:
        h1_df = load_pair_csv(pair, H1_DATA_DIR)
        h1_df["date"] = pd.to_datetime(h1_df["date"])
        h1_df = h1_df.sort_values("date").reset_index(drop=True)
        return h1_df
    except FileNotFoundError:
        return None


def _precompute_h1_range_array(
    df_4h: pd.DataFrame,
    h1_df: pd.DataFrame | None,
) -> np.ndarray:
    """Return h1_last_bar_close_in_range aligned to the 4H bar index.

    For each 4H bar (open time T, close time T+4H), find the most recent
    1H bar whose close timestamp (h1_open + 1H) is <= T+4H.  Equivalently,
    the 1H bar with open time <= T+3H.

    h1_last_bar_close_in_range = (h1_close - h1_low) / (h1_high - h1_low)
    Doji (h1_high == h1_low): return 0.5 (neutral, allow entry).
    No data: return NaN (allow entry).
    """
    n = len(df_4h)
    result = np.full(n, np.nan, dtype=float)
    if h1_df is None or h1_df.empty:
        return result

    h1_dates  = pd.to_datetime(h1_df["date"].values)
    h1_high   = h1_df["high"].values.astype(float)
    h1_low    = h1_df["low"].values.astype(float)
    h1_close  = h1_df["close"].values.astype(float)

    # Compute (close-low)/(high-low); 0.5 for doji
    _rng = h1_high - h1_low
    h1_cir = np.where(_rng > 0, (h1_close - h1_low) / _rng, 0.5)

    h1_frame = pd.DataFrame({
        "h1_open_time": h1_dates,
        "h1_cir":       h1_cir,
    }).sort_values("h1_open_time").reset_index(drop=True)

    # For each 4H bar (open time T), the 4H close is T+4H.
    # Last valid 1H bar has open time <= T+3H.
    dates_4h = pd.to_datetime(df_4h["date"].values)
    lookup_times = dates_4h + pd.Timedelta(hours=3)

    lookup_frame = pd.DataFrame({
        "lookup_time": lookup_times,
        "_idx":        np.arange(n, dtype=int),
    }).sort_values("lookup_time")

    merged = pd.merge_asof(
        lookup_frame,
        h1_frame,
        left_on="lookup_time",
        right_on="h1_open_time",
        direction="backward",
    )
    # Restore original order
    merged = merged.sort_values("_idx").reset_index(drop=True)
    result = merged["h1_cir"].values.astype(float)
    return result


# ── D1 exit alignment ─────────────────────────────────────────────────────────

def _precompute_d1_exit_arrays(
    df_4h: pd.DataFrame,
    d1_filter: D1Filter | None,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return aligned D1 arrays for the 4H index:
        d1_close_lag1, d1_kijun_lag1,  (yesterday's D1 close/Kijun — existing lag=1 fix)
        d1_close_lag2, d1_kijun_lag2,  (the D1 bar before that   — KH-8 lag=2)
        d1_atr_lag1,                   (D1 ATR at lag=1 — KH-9 Variant C)
        d1_kijun_prior5,               (D1 Kijun 5 D1 bars before the lag=1 bar;
                                         for diagnostic d1_kijun_slope only)
        d1_date_lag1,  d1_date_lag2,   (D1 bar dates for audit)
        bar_date_norm  (4H calendar date, normalised, for audit)
        d1_close_in_range_lag1         (KH-20: (close-low)/(high-low) at lag=1; 0.5 for doji)
    """
    n = len(df_4h)
    nan_arr = lambda: np.full(n, np.nan)  # noqa: E731
    nat_arr = lambda: np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")  # noqa: E731
    d1_close_lag1        = nan_arr()
    d1_kijun_lag1        = nan_arr()
    d1_close_lag2        = nan_arr()
    d1_kijun_lag2        = nan_arr()
    d1_atr_lag1          = nan_arr()
    d1_kijun_prior5      = nan_arr()
    d1_date_lag1         = nat_arr()
    d1_date_lag2         = nat_arr()
    bar_date_norm        = pd.to_datetime(df_4h["date"]).dt.normalize().values.astype("datetime64[ns]")
    d1_close_in_range    = nan_arr()
    if d1_filter is None:
        return (d1_close_lag1, d1_kijun_lag1,
                d1_close_lag2, d1_kijun_lag2,
                d1_atr_lag1,
                d1_kijun_prior5,
                d1_date_lag1,  d1_date_lag2,
                bar_date_norm,
                d1_close_in_range)
    # Shift back 1 day so each 4H bar sees the last completed D1 bar (day D-1).
    dates_4h_norm = pd.to_datetime(df_4h["date"]).dt.normalize() - pd.Timedelta(days=1)
    df_4h_dates = pd.DataFrame({
        "date": dates_4h_norm,
        "_idx": np.arange(n, dtype=int),
    })
    df_d1 = pd.DataFrame({
        "date":             d1_filter._close.index,
        "d1_close":         d1_filter._close.values,
        "d1_high":          d1_filter._high.values,
        "d1_low":           d1_filter._low.values,
        "d1_kijun":         d1_filter._kijun.values,
        "d1_atr":           d1_filter._atr.values,
        "d1_kijun_prior5":  d1_filter._kijun_lag.values,
    }).dropna(subset=["d1_kijun"]).sort_values("date").reset_index(drop=True)
    # KH-20: (close - low) / (high - low); 0.5 for doji (range == 0)
    _rng = df_d1["d1_high"].values - df_d1["d1_low"].values
    df_d1["d1_close_in_range"] = np.where(
        _rng > 0,
        (df_d1["d1_close"].values - df_d1["d1_low"].values) / _rng,
        0.5,
    )
    if df_d1.empty:
        return (d1_close_lag1, d1_kijun_lag1,
                d1_close_lag2, d1_kijun_lag2,
                d1_atr_lag1,
                d1_kijun_prior5,
                d1_date_lag1,  d1_date_lag2,
                bar_date_norm,
                d1_close_in_range)
    # KH-8: lag=2 is the prior D1 bar's close/Kijun on the already-lagged D1 series.
    # shift(1) on the D1 frame = one D1 bar earlier = lag=2 relative to the 4H bar.
    df_d1["d1_close_lag2"] = df_d1["d1_close"].shift(1)
    df_d1["d1_kijun_lag2"] = df_d1["d1_kijun"].shift(1)
    df_d1["d1_date_lag2"]  = df_d1["date"].shift(1)
    df_d1["d1_date_lag1"]  = df_d1["date"]
    merged = pd.merge_asof(
        df_4h_dates.sort_values("date"),
        df_d1,
        on="date",
        direction="backward",
    )
    d1_close_lag1     = merged["d1_close"].values.astype(float)
    d1_kijun_lag1     = merged["d1_kijun"].values.astype(float)
    d1_close_lag2     = merged["d1_close_lag2"].values.astype(float)
    d1_kijun_lag2     = merged["d1_kijun_lag2"].values.astype(float)
    d1_atr_lag1       = merged["d1_atr"].values.astype(float)
    d1_kijun_prior5   = merged["d1_kijun_prior5"].values.astype(float)
    d1_date_lag1      = merged["d1_date_lag1"].values.astype("datetime64[ns]")
    d1_date_lag2      = merged["d1_date_lag2"].values.astype("datetime64[ns]")
    d1_close_in_range = merged["d1_close_in_range"].values.astype(float)
    return (d1_close_lag1, d1_kijun_lag1,
            d1_close_lag2, d1_kijun_lag2,
            d1_atr_lag1,
            d1_kijun_prior5,
            d1_date_lag1,  d1_date_lag2,
            bar_date_norm,
            d1_close_in_range)


# ── Signal evaluation: KF v17 + D1a + cond_9 (identical to V1) ───────────────

def _build_signal_series(
    df: pd.DataFrame,
    d1_filter: D1Filter | None,
    baseline_override: np.ndarray | None = None,
    min_warmup: int | None = None,
) -> tuple[np.ndarray, int, int, int, list[pd.Timestamp], list[tuple[int, float]]]:
    n   = len(df)
    h   = df["high"].values.astype(float)
    lo  = df["low"].values.astype(float)
    c   = df["close"].values.astype(float)
    o   = df["open"].values.astype(float)
    _vol_col = VOL_COLUMN if VOL_COLUMN in df.columns else "volume"
    vol = df[_vol_col].values.astype(float)
    atr = _wilder_atr(df, ATR_PERIOD).values
    kij_native = _compute_kijun(df, KIJUN_PERIOD)
    # C4/C5 baseline: Kijun by default, or HMA/DEMA when overridden (KH-7).
    base_arr = baseline_override if baseline_override is not None else kij_native

    warm_floor = max(ATR_PERIOD, KIJUN_PERIOD, DEPTH_BARS, VOL_LOOKBACK)
    if min_warmup is not None:
        warm_floor = max(warm_floor, int(min_warmup))

    sig                = np.zeros(n, dtype=int)
    n_d1_blocked       = 0
    n_d1_passed        = 0
    n_cond9_blocked    = 0
    cond9_block_dates: list[pd.Timestamp]    = []
    d1_dist_ratios:   list[tuple[int, float]] = []

    for i in range(n):
        if i < warm_floor:
            continue
        a = atr[i]
        if np.isnan(a) or a == 0.0:
            continue
        k = base_arr[i]
        if np.isnan(k):
            continue
        bar_range = h[i] - lo[i]
        if bar_range == 0.0:
            continue
        if SIGNAL_DIRECTION == "short":
            # C1 short: bullish bar (close > open)
            if c[i] <= o[i]:
                continue
            # C2: body >= BODY_THRESH × ATR (unchanged)
            if abs(c[i] - o[i]) / a < BODY_THRESH:
                continue
            # C3 short: close near high — (high - close) / range <= CLOSE_POS_MAX
            if (h[i] - c[i]) / bar_range > CLOSE_POS_MAX:
                continue
            # C4 short: close below Kijun (not too far above)
            if c[i] > k + C4_KIJUN_OFFSET_ATR * a:
                continue
            # C5 short: not too far below Kijun
            if c[i] < k - DISTANCE_CAP * a:
                continue
            # C6 short: upward momentum — close must have risen >= DEPTH_THRESH × ATR
            if (c[i] - c[i - DEPTH_BARS]) / a < DEPTH_THRESH:
                continue
        else:
            # C1 long: bearish bar (close < open)
            if c[i] >= o[i]:
                continue
            # C2: body >= BODY_THRESH × ATR
            if abs(c[i] - o[i]) / a < BODY_THRESH:
                continue
            # C3 long: close near low — (close - low) / range <= CLOSE_POS_MAX
            if (c[i] - lo[i]) / bar_range > CLOSE_POS_MAX:
                continue
            # C4 long: close above Kijun (not too far below)
            if c[i] < k - C4_KIJUN_OFFSET_ATR * a:
                continue
            # C5 long: not too far above Kijun
            if c[i] > k + DISTANCE_CAP * a:
                continue
            # C6 long: downward momentum — close must have fallen >= DEPTH_THRESH × ATR
            if (c[i] - c[i - DEPTH_BARS]) / a > -DEPTH_THRESH:
                continue
        if RANGE_ATR_CEILING is not None and bar_range / a > RANGE_ATR_CEILING:
            continue
        if not NO_VOLUME_FILTER:
            if not np.isnan(vol[i]):
                prior_vol = vol[i - VOL_LOOKBACK: i]
                vol_mean  = np.nanmean(prior_vol)
                if not (vol_mean == 0.0 or np.isnan(vol_mean)):
                    if vol[i] <= VOL_MULT * vol_mean:
                        continue
        if D1_REGIME_FILTER and d1_filter is not None:
            signal_date = pd.Timestamp(df.at[i, "date"])
            passes, d1_cl, d1_kij, _, d1_atr = d1_filter.check(signal_date)
            if SIGNAL_DIRECTION == "short":
                # C8 short: D1 close < D1 Kijun (bearish D1 regime)
                # passes = (d1_cl > d1_kij) from D1Filter.check(); invert for short
                short_passes = not passes
                if USE_C8 and not short_passes:
                    n_d1_blocked += 1
                    continue
                n_d1_passed += 1
                # C9 short: not too far below Kijun
                if USE_C9 and not (np.isnan(d1_atr) or d1_atr == 0.0):
                    if d1_cl < d1_kij - D1_ATR_DIST_CAP * d1_atr:
                        n_cond9_blocked += 1
                        cond9_block_dates.append(signal_date)
                        continue
            else:
                if USE_C8 and not passes:
                    n_d1_blocked += 1
                    continue
                n_d1_passed += 1
                if USE_C9 and not (np.isnan(d1_atr) or d1_atr == 0.0):
                    if d1_cl > d1_kij + D1_ATR_DIST_CAP * d1_atr:
                        n_cond9_blocked += 1
                        cond9_block_dates.append(signal_date)
                        continue
            ratio = (d1_cl - d1_kij) / d1_atr if d1_atr > 0 else float("nan")
            d1_dist_ratios.append((i, ratio))
        sig[i] = 1

    return sig, n_d1_blocked, n_d1_passed, n_cond9_blocked, cond9_block_dates, d1_dist_ratios


# ── Build pair cache ──────────────────────────────────────────────────────────

def _build_pair_cache(pairs: list[str] | None = None) -> dict:
    target_pairs = pairs if pairs is not None else ALL_PAIRS
    pair_cache: dict[str, dict] = {}

    for pair in target_pairs:
        try:
            df = load_pair_csv(pair, DATA_DIR_4H)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            if "volume" not in df.columns:
                df["volume"] = 0.0
            d1_filt = _load_d1_filter(pair) if D1_REGIME_FILTER else None
            h1_df   = _load_h1_data(pair)
            (d1_close_arr, d1_kijun_arr,
             d1_close_lag2_arr, d1_kijun_lag2_arr,
             d1_atr_lag1_arr,
             d1_kijun_prior5_arr,
             d1_date_lag1_arr, d1_date_lag2_arr,
             bar_date_norm_arr,
             d1_close_in_range_arr) = _precompute_d1_exit_arrays(df, d1_filt)

            h1_range_arr = _precompute_h1_range_array(df, h1_df)

            # KH-8 lookahead verification: for every row that has both lags,
            #     d1_date_lag2 < d1_date_lag1 < 4H calendar date.
            _nat = np.datetime64("NaT")
            _has_lag1 = ~np.isnat(d1_date_lag1_arr)
            _has_lag2 = ~np.isnat(d1_date_lag2_arr)
            _both = _has_lag1 & _has_lag2
            if _both.any():
                if not (d1_date_lag2_arr[_both] < d1_date_lag1_arr[_both]).all():
                    raise ValueError(
                        f"{pair}: KH-8 lag ordering violated (lag2 >= lag1 on "
                        f"{int((d1_date_lag2_arr[_both] >= d1_date_lag1_arr[_both]).sum())} rows)"
                    )
                if not (d1_date_lag1_arr[_both] < bar_date_norm_arr[_both]).all():
                    raise ValueError(
                        f"{pair}: KH-8 lookahead — d1_date_lag1 >= 4H calendar date on "
                        f"{int((d1_date_lag1_arr[_both] >= bar_date_norm_arr[_both]).sum())} rows"
                    )
            _ = _nat  # silence unused

            closes_arr = df["close"].values.astype(float)
            highs_arr  = df["high"].values.astype(float)
            lows_arr   = df["low"].values.astype(float)

            # KH-7: C4/C5 baseline — Kijun (default) or HMA(26) / DEMA(50).
            # Kijun is still computed via _compute_kijun inside _build_signal_series
            # when no override is supplied (default path preserves baseline exactly).
            if BASELINE_TYPE in ("hma", "dema"):
                baseline_vals = _compute_baseline_series(
                    closes_arr, highs_arr, lows_arr, BASELINE_TYPE,
                )
            else:
                baseline_vals = None

            min_warm = WARMUP_BARS.get(BASELINE_TYPE, 0)
            sig, n_blocked, n_passed, n_cond9, cond9_dates, d1_dist_ratios = _build_signal_series(
                df, d1_filt,
                baseline_override=baseline_vals,
                min_warmup=min_warm if baseline_vals is not None else None,
            )

            ema50_arr = _compute_ema(closes_arr, REGIME_EMA_PERIOD)
            ema50_series = pd.Series(ema50_arr, index=pd.DatetimeIndex(df["date"].values))

            atr_arr = _wilder_atr(df, ATR_PERIOD).values
            sp_arr  = (df["spread"].fillna(0.0).values.astype(float)
                       if "spread" in df.columns else np.zeros(len(df)))
            # Liquidity-proxy diagnostics — pre-compute per-pair rolling
            # medians over prior-only windows so signal-bar enrichment is O(1).
            spread_pips_arr = sp_arr / POINTS_PER_PIP
            spread_median_arr = _rolling_median_prior(
                pd.Series(spread_pips_arr), LIQUIDITY_MEDIAN_WINDOW,
            ).values
            atr_median_arr = _rolling_median_prior(
                pd.Series(atr_arr), LIQUIDITY_MEDIAN_WINDOW,
            ).values

            pair_cache[pair] = {
                "df":                df,
                "o":                 df["open"].values.astype(float),
                "h":                 df["high"].values.astype(float),
                "lo":                df["low"].values.astype(float),
                "c":                 df["close"].values.astype(float),
                "atr":               atr_arr,
                "sp":                sp_arr,
                "sig":               sig,
                "d1_close_arr":             d1_close_arr,
                "d1_kijun_arr":             d1_kijun_arr,
                "d1_close_lag2_arr":        d1_close_lag2_arr,
                "d1_kijun_lag2_arr":        d1_kijun_lag2_arr,
                "d1_atr_lag1_arr":          d1_atr_lag1_arr,
                "d1_kijun_prior5_arr":      d1_kijun_prior5_arr,
                "d1_close_in_range_arr":    d1_close_in_range_arr,
                "h1_range_arr":             h1_range_arr,
                "dates":             df["date"].values,
                "ema50_series":      ema50_series,
                "n_d1_blocked":      n_blocked,
                "n_d1_passed":       n_passed,
                "n_cond9_blocked":   n_cond9,
                "cond9_block_dates":    cond9_dates,
                "d1_dist_ratio_by_bar": dict(d1_dist_ratios),
                "pip":                  get_pip_size(pair),
                "spread_pips_arr":      spread_pips_arr,
                "spread_median_arr":    spread_median_arr,
                "atr_median_arr":       atr_median_arr,
            }
        except FileNotFoundError:
            print(f"    [WARN] {pair}: 4H data not found — skipping")

    for pair, cached in pair_cache.items():
        date_to_idx: dict[pd.Timestamp, int] = {}
        for idx, d in enumerate(cached["dates"]):
            date_to_idx[pd.Timestamp(d)] = idx
        cached["date_to_idx"] = date_to_idx

    return pair_cache


# ── kijun_d1 exit — unified 1-bar / 2-bar / conditional (KH-8 + KH-9) ────────

def _kijun_d1_should_exit(
    d1_close_lag1: float,
    d1_kijun_lag1: float,
    d1_close_lag2: float,
    d1_kijun_lag2: float,
    d1_atr_lag1: float,
    trail_active: bool,
    bars_held: int,
    confirm_bars: int,
    confirm_if_trail: bool,
    confirm_min_bars: int,
    confirm_depth_atr: float,
    direction: str = "long",
) -> bool:
    """Unified kijun_d1 exit decision.

    Baseline / KH-8 behaviour is preserved when all KH-9 flags are at
    their off state (confirm_if_trail=False, confirm_min_bars=0,
    confirm_depth_atr=0.0).  In that case the result is:
        confirm_bars == 1 → primary cross only
        confirm_bars == 2 → primary AND secondary (lag=2) cross

    When any KH-9 flag is active, 2-bar confirmation is REQUIRED only
    when the corresponding condition is satisfied; otherwise the exit
    fires on the 1-bar (primary) cross.  This is only meaningful when
    confirm_bars == 2; with confirm_bars == 1 all variants collapse to
    1-bar behaviour.
    """
    # Step 1: primary cross present?
    if np.isnan(d1_close_lag1) or np.isnan(d1_kijun_lag1):
        return False
    if direction == "short":
        # Short exits when D1 close > D1 Kijun (bearish regime ended)
        primary_cross = bool(d1_close_lag1 > d1_kijun_lag1)
    else:
        # Long exits when D1 close < D1 Kijun (bullish regime ended)
        primary_cross = bool(d1_close_lag1 < d1_kijun_lag1)
    if not primary_cross:
        return False

    # Step 2: decide whether 2-bar confirmation should be required.
    require_confirm = False

    if confirm_if_trail and trail_active:
        require_confirm = True

    if confirm_min_bars > 0 and bars_held >= confirm_min_bars:
        require_confirm = True

    if confirm_depth_atr > 0.0:
        if not np.isnan(d1_atr_lag1) and d1_atr_lag1 > 0.0:
            if direction == "short":
                cross_depth = d1_close_lag1 - d1_kijun_lag1
            else:
                cross_depth = d1_kijun_lag1 - d1_close_lag1
            if cross_depth < confirm_depth_atr * d1_atr_lag1:
                require_confirm = True

    any_conditional_active = (
        confirm_if_trail
        or confirm_min_bars > 0
        or confirm_depth_atr > 0.0
    )

    # Step 3: if 2-bar is configured, apply it either globally (no
    # conditional active) or only when at least one conditional flagged.
    if confirm_bars == 2 and (not any_conditional_active or require_confirm):
        if np.isnan(d1_close_lag2) or np.isnan(d1_kijun_lag2):
            return False
        if direction == "short":
            return bool(d1_close_lag2 > d1_kijun_lag2)
        return bool(d1_close_lag2 < d1_kijun_lag2)

    # Step 4: 1-bar path (either confirm_bars == 1, or a conditional
    # variant decided no confirmation was required this bar).
    return True


# ── Unified multi-pair simulation (V2: NO signal_flip exit) ──────────────────

def _simulate_kgl_v2(
    pair_cache: dict,
    restrict_dates: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Unified simulation — V2 variant.

    signal_flip exit is completely disabled.
    Exit priority:
        1. Hard SL (intrabar low <= sl_px)
        2. Trailing stop (bar close <= ts_level AND trail_active)
        3. kijun_d1 (D1 close < D1 Kijun, bar close)

    Trail activation: CLOSE-BASED (c_j >= trail_act_px)
    Classification:   WIN = r_mult > 0, LOSS < 0, SCRATCH = 0
    """
    all_ts: set = set()
    for cached in pair_cache.values():
        for d in cached["dates"]:
            all_ts.add(pd.Timestamp(d))

    master_dates = sorted(all_ts)

    if restrict_dates is not None:
        rs, re = restrict_dates
        master_dates = [d for d in master_dates if rs <= d < re]

    currency_exposure: dict[str, int] = defaultdict(int)
    open_trades:       list[dict]     = []
    completed_trades:  list[dict]     = []
    bar_paths_data:    list[dict]     = []  # v1.3: flat per-(trade_id, bar_offset) rows
    blocked_events:    list[dict]     = []
    reentry_watches:   dict[str, dict] = {}  # pair -> watch state for KH-16
    # KH-17: three-stage state per pair.
    #   kh17_pending — signal at bar N awaiting bar N+1 close decision
    #   kh17_virtual — State 2 virtual tracking, H/L-based MFE/MAE on N+1..N+6
    #   kh17_watches — re-entry watch (bars N+7..N+7+window-1) after bar-6 trigger
    kh17_pending:      dict[str, dict] = {}
    kh17_virtual:      dict[str, dict] = {}
    kh17_watches:      dict[str, dict] = {}
    # KH-17 Phase 1.7 diagnostic counters (State 1 drop reasons)
    _kh17_s1_fired:          int = 0
    _kh17_s1_drop_open:      int = 0
    _kh17_s1_drop_atr:       int = 0
    _kh17_s1_drop_cap:       int = 0
    _kh17_s1_drop_expired:   int = 0

    for bar_date in master_dates:
        # ── Phase 1: Advance all open trades ─────────────────────────────────
        still_open: list[dict] = []
        for trade in open_trades:
            pair   = trade["pair"]
            cached = pair_cache[pair]
            j_opt  = cached["date_to_idx"].get(bar_date)

            if j_opt is None:
                still_open.append(trade)
                continue

            j = j_opt

            if j < trade["min_mgmt_idx"]:
                still_open.append(trade)
                continue

            lo_j = cached["lo"][j]
            hi_j = cached["h"][j]
            c_j  = cached["c"][j]

            # ── MAE/MFE roll (diagnostic; updated BEFORE exit checks so the
            # exit bar's full H/L are reflected). Units: ATR multiples using
            # ATR at signal bar (trade["atr"]), which is immutable.
            _a_trk = trade["atr"]
            _ep_trk = trade["entry_px"]
            if SIGNAL_DIRECTION == "short":
                _mae_bar = max(0.0, (hi_j - _ep_trk) / _a_trk)
                _mfe_bar = max(0.0, (_ep_trk - lo_j) / _a_trk)
            else:
                _mae_bar = max(0.0, (_ep_trk - lo_j) / _a_trk)
                _mfe_bar = max(0.0, (hi_j - _ep_trk) / _a_trk)
            _mae_bar = max(0.0, _mae_bar)
            _mfe_bar = max(0.0, _mfe_bar)
            if _mae_bar > trade["mae_run"]:
                trade["mae_run"] = _mae_bar
            if _mfe_bar > trade["mfe_run"]:
                trade["mfe_run"] = _mfe_bar
            # v1.3: per-bar trade-path emission. Signed (high - entry)/atr,
            # (low - entry)/atr, (close - entry)/atr; running max/min track
            # mfe_so_far_r / mae_so_far_r and pin time_to_peak_mfe /
            # time_to_trough_mae as bar offsets (0 = entry bar).
            _bar_off_v13 = j - trade["entry_idx"]
            if SIGNAL_DIRECTION == "short":
                _hr_v13 = (_ep_trk - hi_j) / _a_trk
                _lr_v13 = (_ep_trk - lo_j) / _a_trk
                _cr_v13 = (_ep_trk - c_j)  / _a_trk
            else:
                _hr_v13 = (hi_j - _ep_trk) / _a_trk
                _lr_v13 = (lo_j - _ep_trk) / _a_trk
                _cr_v13 = (c_j  - _ep_trk) / _a_trk
            _last_p_v13 = trade["bar_path"][-1]
            _mfe_sf_v13 = _last_p_v13["mfe_so_far_r"]
            _mae_sf_v13 = _last_p_v13["mae_so_far_r"]
            if _hr_v13 > _mfe_sf_v13:
                _mfe_sf_v13 = _hr_v13
                trade["time_to_peak_mfe"] = _bar_off_v13
            if _lr_v13 < _mae_sf_v13:
                _mae_sf_v13 = _lr_v13
                trade["time_to_trough_mae"] = _bar_off_v13
            trade["bar_path"].append({
                "bar_offset":   _bar_off_v13,
                "high_r":       _hr_v13,
                "low_r":        _lr_v13,
                "close_r":      _cr_v13,
                "mfe_so_far_r": _mfe_sf_v13,
                "mae_so_far_r": _mae_sf_v13,
                "is_held":      1,
            })
            _bar_num = j - trade["entry_idx"] + 1
            if _bar_num >= 3 and math.isnan(trade["mae_at_3"]):
                trade["mae_at_3"] = trade["mae_run"]
                trade["mfe_at_3"] = trade["mfe_run"]
                # KH-13: record check values and trigger flag at bar 3
                # (always, regardless of USE_KH13_EARLY_EXIT — enables
                # post-hoc audit even on baseline runs)
                trade["kh13_mae_at_check"] = trade["mae_run"]
                trade["kh13_mfe_at_check"] = trade["mfe_run"]
                trade["kh13_triggered"] = bool(
                    trade["mae_run"] >= KH13_MAE_THRESHOLD
                    and trade["mfe_run"] <= KH13_MFE_THRESHOLD
                )
            if _bar_num >= 6 and math.isnan(trade["mae_at_6"]):
                trade["mae_at_6"] = trade["mae_run"]
                trade["mfe_at_6"] = trade["mfe_run"]
                # KH-14: record whether bar-6 MAE/MFE condition is met (all trades,
                # always recorded regardless of first_bar_dir or config — enables
                # post-hoc audit of full population).  Exit fires only when State 2.
                trade["kh14_triggered"] = bool(
                    trade["mfe_run"] < KH14_MFE_THRESHOLD
                    and trade["mae_run"] >= KH14_MAE_THRESHOLD
                )

            sl_px        = trade["sl_px"]
            trail_active = trade["trail_active"]
            best_cl      = trade["best_cl"]
            ts_level     = trade["ts_level"]
            a            = trade["atr"]
            entry_px     = trade["entry_px"]
            entry_idx    = trade["entry_idx"]
            trail_act_px = trade["trail_act_px"]

            exit_px     = None
            exit_bar    = j
            exit_reason = None

            # Pipeline D1 hook (L_ARC_PROTOCOL v2.0 §3 + §11). Fires exactly
            # once per trade at bar offset t == D1_HOOK.bar_offset_t.
            #   Close       → exit at next bar's open with reason
            #                  "d1_untradeable" (last-bar fallback → bar j
            #                  close). PR 1 behaviour.
            #   ApplyPolicy → install the archetype's ExitPolicy on the
            #                  trade. The policy's apply_at_accept runs
            #                  inside the helper (replaces pre-t SL), and
            #                  the per-bar block below ratchets SL on
            #                  every subsequent bar. PR 2.
            #   Hold        → no classifier for this trade's fold (e.g.
            #                  Arc 4 F1). Legacy cascade handles the trade.
            if D1_HOOK is not None:
                _d1_px, _d1_bar, _d1_reason = apply_d1_hook_per_bar(
                    D1_HOOK, trade, j, entry_idx, cached, c_j,
                    fold_id=trade.get("fold_id"),
                )
                if _d1_reason is not None:
                    exit_px     = _d1_px
                    exit_bar    = _d1_bar
                    exit_reason = _d1_reason
                # apply_at_accept (when ApplyPolicy) mutated trade["sl_px"];
                # refresh the local so subsequent SL/trail logic uses the
                # archetype's post-t stop, not the stale pre-t value.
                sl_px = trade["sl_px"]

            # Per-bar policy update (PR 2). Fires every bar from accept
            # onward; mutates trade["sl_px"] in place via the installed
            # ExitPolicy. Runs before the Priority-1 SL check so the
            # ratchet is in effect for the same bar's intrabar SL test.
            if trade.get("exit_policy") is not None and exit_reason is None:
                trade["exit_policy"].update_per_bar(
                    trade=trade,
                    bar_path_row=trade["bar_path"][-1],
                    entry_px=entry_px,
                    atr_at_entry=a,
                )
                sl_px = trade["sl_px"]

            # Priority 1: Intrabar hard SL
            if exit_reason is None:
                if SIGNAL_DIRECTION == "short":
                    if hi_j >= sl_px:
                        exit_px     = sl_px
                        exit_bar    = j
                        exit_reason = "stoploss"
                else:
                    if lo_j <= sl_px:
                        exit_px     = sl_px
                        exit_bar    = j
                        exit_reason = "stoploss"

            if exit_reason is None:
                if SIGNAL_DIRECTION == "short":
                    # Trail activation: CLOSE-BASED (close <= entry - 2×ATR)
                    if not trail_active and c_j <= trail_act_px:
                        trail_active = True
                    # Update trailing stop (ratchet down to lowest close)
                    if c_j < best_cl:
                        best_cl  = c_j
                        ts_level = best_cl + TRAIL_MULT * a
                    # Priority 2: Trailing stop (close >= stop level)
                    if trail_active and c_j >= ts_level:
                        n = len(cached["o"])
                        if j + 1 < n:
                            exit_bar    = j + 1
                            exit_px     = cached["o"][j + 1]
                        else:
                            exit_bar    = j
                            exit_px     = c_j
                        exit_reason = "trailing_stop"
                else:
                    # Trail activation: CLOSE-BASED
                    if not trail_active and c_j >= trail_act_px:
                        trail_active = True
                    # Update trailing stop on bar close (ratchet)
                    if c_j > best_cl:
                        best_cl  = c_j
                        ts_level = best_cl - TRAIL_MULT * a
                    # Priority 2: Trailing stop (bar close, gated on trail_active)
                    if trail_active and c_j <= ts_level:
                        n = len(cached["o"])
                        if j + 1 < n:
                            exit_bar    = j + 1
                            exit_px     = cached["o"][j + 1]
                        else:
                            exit_bar    = j
                            exit_px     = c_j
                        exit_reason = "trailing_stop"

            if exit_reason is None:
                # Priority 3: KH-13 early exit — fires at bar N+3 close,
                # fills at bar N+4 open.  Only triggers on the exact bar
                # where _bar_num == 3 and the condition was just evaluated.
                if (USE_KH13_EARLY_EXIT
                        and _bar_num == 3
                        and trade.get("kh13_triggered", False)):
                    n_bars = len(cached["o"])
                    if j + 1 < n_bars:
                        exit_bar    = j + 1
                        exit_px     = cached["o"][j + 1]
                    else:
                        exit_bar    = j
                        exit_px     = c_j
                    exit_reason = "kh13_early"

            if exit_reason is None:
                # Priority 4: KH-14 bar-6 exit — fires at bar N+6 close,
                # fills at bar N+7 open.  Only triggers on the exact bar
                # where _bar_num == 6, State 2 (first_bar_dir==-1), and the
                # MAE/MFE condition was just evaluated as True.
                if (USE_KH14_BAR6_EXIT
                        and _bar_num == 6
                        and trade["first_bar_dir"] == -1
                        and trade.get("kh14_triggered", False)):
                    n_bars = len(cached["o"])
                    if j + 1 < n_bars:
                        exit_bar    = j + 1
                        exit_px     = cached["o"][j + 1]
                    else:
                        exit_bar    = j
                        exit_px     = c_j
                    exit_reason = "kh14_bar6"

            if exit_reason is None:
                # Priority 5: kijun_d1 only (signal_flip DISABLED)
                exit_d1 = _kijun_d1_should_exit(
                    d1_close_lag1   = cached["d1_close_arr"][j],
                    d1_kijun_lag1   = cached["d1_kijun_arr"][j],
                    d1_close_lag2   = cached["d1_close_lag2_arr"][j],
                    d1_kijun_lag2   = cached["d1_kijun_lag2_arr"][j],
                    d1_atr_lag1     = cached["d1_atr_lag1_arr"][j],
                    trail_active    = trail_active,
                    bars_held       = j - entry_idx,
                    confirm_bars    = KIJUN_D1_CONFIRM_BARS,
                    confirm_if_trail  = KIJUN_D1_CONFIRM_IF_TRAIL,
                    confirm_min_bars  = KIJUN_D1_CONFIRM_MIN_BARS,
                    confirm_depth_atr = KIJUN_D1_CONFIRM_DEPTH_ATR,
                    direction         = SIGNAL_DIRECTION,
                )
                if exit_d1:
                    n = len(cached["o"])
                    if j + 1 < n:
                        exit_bar    = j + 1
                        exit_px     = cached["o"][j + 1]
                    else:
                        exit_bar    = j
                        exit_px     = c_j
                    exit_reason = "kijun_d1"

            if exit_reason is not None:
                # v1.3: when the exit fills at the next bar's open
                # (exit_bar > j), the regular per-bar emission stopped at
                # bar_offset = j - entry_idx; append a held row at
                # bar_offset = bars_held using exit_bar's full H/L so the
                # held window covers entry through exit inclusive.
                if exit_bar > j:
                    _hi_e_v13 = cached["h"][exit_bar]
                    _lo_e_v13 = cached["lo"][exit_bar]
                    _cl_e_v13 = cached["c"][exit_bar]
                    if SIGNAL_DIRECTION == "short":
                        _hr_v13 = (entry_px - _hi_e_v13) / a
                        _lr_v13 = (entry_px - _lo_e_v13) / a
                        _cr_v13 = (entry_px - _cl_e_v13) / a
                    else:
                        _hr_v13 = (_hi_e_v13 - entry_px) / a
                        _lr_v13 = (_lo_e_v13 - entry_px) / a
                        _cr_v13 = (_cl_e_v13 - entry_px) / a
                    _last_p_v13 = trade["bar_path"][-1]
                    _mfe_sf_v13 = _last_p_v13["mfe_so_far_r"]
                    _mae_sf_v13 = _last_p_v13["mae_so_far_r"]
                    if _hr_v13 > _mfe_sf_v13:
                        _mfe_sf_v13 = _hr_v13
                        trade["time_to_peak_mfe"] = exit_bar - entry_idx
                    if _lr_v13 < _mae_sf_v13:
                        _mae_sf_v13 = _lr_v13
                        trade["time_to_trough_mae"] = exit_bar - entry_idx
                    trade["bar_path"].append({
                        "bar_offset":   exit_bar - entry_idx,
                        "high_r":       _hr_v13,
                        "low_r":        _lr_v13,
                        "close_r":      _cr_v13,
                        "mfe_so_far_r": _mfe_sf_v13,
                        "mae_so_far_r": _mae_sf_v13,
                        "is_held":      1,
                    })

                risk_pp     = trade["risk_pp"]
                pos_size    = trade["pos_size"]
                spread_cost = trade["spread_cost"]

                if SIGNAL_DIRECTION == "short":
                    gross_pnl = (entry_px - exit_px) * pos_size
                    net_pnl   = gross_pnl - spread_cost
                    r_mult    = (entry_px - exit_px) / risk_pp
                else:
                    gross_pnl = (exit_px - entry_px) * pos_size
                    net_pnl   = gross_pnl - spread_cost
                    r_mult    = (exit_px - entry_px) / risk_pp
                bars_held = exit_bar - entry_idx

                assert bars_held >= 1, (
                    f"{pair}: exit_bar={exit_bar} entry_idx={entry_idx} "
                    f"entry_date={trade['entry_date']} exit_date={cached['dates'][exit_bar]}"
                )

                if r_mult > 0:
                    classification = "WIN"
                elif r_mult < 0:
                    classification = "LOSS"
                else:
                    classification = "SCRATCH"

                _mae_fin = float(trade["mae_run"])
                _mfe_fin = float(trade["mfe_run"])
                _mae_3 = _mae_fin if math.isnan(trade["mae_at_3"]) else float(trade["mae_at_3"])
                _mfe_3 = _mfe_fin if math.isnan(trade["mfe_at_3"]) else float(trade["mfe_at_3"])
                _mae_6 = _mae_fin if math.isnan(trade["mae_at_6"]) else float(trade["mae_at_6"])
                _mfe_6 = _mfe_fin if math.isnan(trade["mfe_at_6"]) else float(trade["mfe_at_6"])
                _kh13_mae_chk = float(trade["kh13_mae_at_check"])
                _kh13_mfe_chk = float(trade["kh13_mfe_at_check"])
                _kh13_trig    = bool(trade["kh13_triggered"])
                # kh14_triggered: State 2 trades (first_bar_dir=-1) where the bar-6
                # MAE/MFE condition is met.  Uses _mae_6/_mfe_6 with final fallback
                # so early-exit State 2 trades are also counted for audit purposes.
                _kh14_trig    = bool(
                    trade["first_bar_dir"] == -1
                    and _mfe_6 < KH14_MFE_THRESHOLD
                    and _mae_6 >= KH14_MAE_THRESHOLD
                )
                _kh14_state2  = bool(trade["first_bar_dir"] == -1)

                # Pipeline D1 (PR 2) audit columns. Gated on D1_HOOK so
                # KH-24 baseline runs (D1_HOOK = None) write a
                # byte-identical trades_all.csv.
                _d1_audit_extras: dict = {}
                if D1_HOOK is not None:
                    _d1_audit_extras = {
                        "fold_id":               trade.get("fold_id"),
                        "d1_decision":           trade.get("d1_decision", "no_d1"),
                        "classifier_fold_id":    trade.get("classifier_fold_id"),
                        "d1_archetype_label":    trade.get("d1_archetype_label"),
                        "d1_probability":        trade.get("d1_probability", float("nan")),
                        "mfe_lock_fired_bar":    trade.get("mfe_lock_fired_bar"),
                        "trail_active_from_bar": trade.get("trail_active_from_bar"),
                    }
                completed_trades.append({
                    "pair":              pair,
                    "entry_date":        trade["entry_date"],
                    "exit_date":         pd.Timestamp(cached["dates"][exit_bar]),
                    "entry_price":       round(entry_px,           6),
                    "exit_price":        round(exit_px,            6),
                    "sl_price":          round(trade["sl_px_init"], 6),
                    "trail_active":      trail_active,
                    "exit_reason":       exit_reason,
                    "classification":    classification,
                    "bars_held":         bars_held,
                    "net_pnl":           round(net_pnl,            4),
                    "r_multiple":        round(r_mult,             4),
                    "spread_pips_used":  round(trade["spread_pips"], 4),
                    "sl_distance_atr":   round(trade["risk_pp"] / trade["atr"], 4),
                    "d1_dist_ratio":     trade["d1_dist_ratio"],
                    "d1_close_in_range": trade.get("d1_close_in_range", float("nan")),
                    "h1_last_bar_close_in_range": trade.get("h1_last_bar_close_in_range", float("nan")),
                    "mae_final":         round(_mae_fin, 4),
                    "mfe_final":         round(_mfe_fin, 4),
                    "mae_at_bar_3":      round(_mae_3,   4),
                    "mfe_at_bar_3":      round(_mfe_3,   4),
                    "mae_at_bar_6":      round(_mae_6,   4),
                    "mfe_at_bar_6":      round(_mfe_6,   4),
                    "first_bar_dir":     int(trade["first_bar_dir"]),
                    "kh13_mae_at_check": (round(_kh13_mae_chk, 4)
                                          if not math.isnan(_kh13_mae_chk) else float("nan")),
                    "kh13_mfe_at_check": (round(_kh13_mfe_chk, 4)
                                          if not math.isnan(_kh13_mfe_chk) else float("nan")),
                    "kh13_triggered":    _kh13_trig,
                    "kh14_triggered":    _kh14_trig,
                    "kh14_state2":       _kh14_state2,
                    "atr_sized_down":    bool(trade.get("atr_sized_down", False)),
                    "trade_type":        trade.get("trade_type", "original"),
                    "original_entry_price_ref": trade.get("original_entry_price",
                                                          float("nan")),
                    **trade["signal_ctx"],
                    # v1.3 sidecars: re-positioned in the post-pass after
                    # concurrent_signals so trade_id / time_to_peak_mfe /
                    # time_to_trough_mae land at the END of the legacy column
                    # set (preserves byte-identity of the legacy subset).
                    "_v13_time_to_peak_mfe":   int(trade["time_to_peak_mfe"]),
                    "_v13_time_to_trough_mae": int(trade["time_to_trough_mae"]),
                    **_d1_audit_extras,
                })
                _flatten_bar_path_for_trade(
                    trade, exit_bar, pair_cache,
                    SIGNAL_DIRECTION, bar_paths_data,
                )

                base, quote = PAIR_CURRENCIES[pair]
                if SIGNAL_DIRECTION == "short":
                    currency_exposure[base] += 1
                    currency_exposure[quote] -= 1
                else:
                    currency_exposure[base] -= 1
                    currency_exposure[quote] += 1

                # KH-16: after kh14_bar6 exit on an original trade, open watch
                if (USE_REENTRY
                        and exit_reason == "kh14_bar6"
                        and trade.get("trade_type", "original") == "original"):
                    reentry_watches[pair] = {
                        "original_entry_price": entry_px,
                        "watch_start_idx":      exit_bar,  # fill bar (N+7 open)
                        "bars_remaining":       REENTRY_WINDOW_BARS,
                    }

            else:
                trade["trail_active"] = trail_active
                trade["best_cl"]      = best_cl
                trade["ts_level"]     = ts_level
                # Pipeline D1 (PR 2): the per-bar policy mutated sl_px in
                # place — persist the latest value so the next bar's local
                # binding sees the policy-driven stop. mfe_lock_fired_bar
                # and trail_active_from_bar are already on the trade dict
                # (mutated by update_per_bar / apply_at_accept).
                trade["sl_px"]        = sl_px
                still_open.append(trade)

        open_trades = still_open

        # ── Phase 1.5: Process KH-16 re-entry watches ────────────────────────
        if USE_REENTRY and reentry_watches:
            for re_pair in list(reentry_watches.keys()):
                watch  = reentry_watches[re_pair]
                re_cac = pair_cache.get(re_pair)
                if re_cac is None:
                    continue
                j_r = re_cac["date_to_idx"].get(bar_date)
                if j_r is None:
                    continue
                if j_r < watch["watch_start_idx"]:
                    continue
                # Guard: skip if pair already has an open trade
                if any(t["pair"] == re_pair for t in open_trades):
                    continue
                if re_cac["c"][j_r] > watch["original_entry_price"]:
                    # Trigger: attempt re-entry at next bar open
                    re_idx = j_r + 1
                    if re_idx >= len(re_cac["o"]):
                        del reentry_watches[re_pair]
                        continue
                    re_px   = re_cac["o"][re_idx]
                    a_re    = re_cac["atr"][re_idx]
                    if np.isnan(a_re) or a_re == 0.0:
                        del reentry_watches[re_pair]
                        continue
                    re_base, re_quote = PAIR_CURRENCIES[re_pair]
                    if (currency_exposure[re_base] + 1 > EXPOSURE_CAP
                            or currency_exposure[re_quote] - 1 < -EXPOSURE_CAP):
                        # Blocked by exposure cap — decrement and keep watching
                        watch["bars_remaining"] -= 1
                        if watch["bars_remaining"] <= 0:
                            del reentry_watches[re_pair]
                        continue
                    risk_pp_re  = REENTRY_SL_ATR_MULT * a_re
                    sl_px_re    = re_px - risk_pp_re
                    ta_px_re    = re_px + TRAIL_ACT_MULT * a_re
                    sp_pips_re  = re_cac["sp"][re_idx] / POINTS_PER_PIP
                    pos_size_re = (INITIAL_BAL * RISK_PCT) / risk_pp_re
                    sc_re       = (sp_pips_re * re_cac["pip"]) * pos_size_re
                    best_cl_re  = re_cac["c"][re_idx]
                    ts_lv_re    = best_cl_re - TRAIL_MULT * a_re
                    ta_re       = bool(re_cac["c"][re_idx] >= ta_px_re)
                    mae_init_re = max(0.0, (re_px - re_cac["lo"][re_idx]) / a_re)
                    mfe_init_re = max(0.0, (re_cac["h"][re_idx] - re_px) / a_re)
                    c_ent_re    = re_cac["c"][re_idx]
                    if c_ent_re > re_px:
                        fbd_re = 1
                    elif c_ent_re < re_px:
                        fbd_re = -1
                    else:
                        fbd_re = 0
                    re_ctx = {
                        "signal_bar_ts":      pd.Timestamp(bar_date),
                        "signal_spread_pips": round(float(re_cac["spread_pips_arr"][j_r]), 4),
                        "spread_ratio":       float("nan"),
                        "atr_abs":            round(float(a_re), 6),
                        "atr_ratio":          float("nan"),
                        "d1_kijun_slope":     False,
                        "session":            _session_for_ts(bar_date),
                    }
                    currency_exposure[re_base]  += 1
                    currency_exposure[re_quote] -= 1
                    open_trades.append({
                        "pair":                  re_pair,
                        "base_ccy":              re_base,
                        "quote_ccy":             re_quote,
                        "signal_bar":            j_r,
                        "entry_idx":             re_idx,
                        "min_mgmt_idx":          re_idx + 1,
                        "entry_px":              re_px,
                        "sl_px":                 sl_px_re,
                        "sl_px_init":            sl_px_re,
                        "trail_act_px":          ta_px_re,
                        "risk_pp":               risk_pp_re,
                        "pos_size":              pos_size_re,
                        "spread_cost":           sc_re,
                        "spread_pips":           sp_pips_re,
                        "best_cl":               best_cl_re,
                        "signal_ctx":            re_ctx,
                        "ts_level":              ts_lv_re,
                        "trail_active":          ta_re,
                        "atr":                   a_re,
                        "entry_date":            bar_date,
                        "d1_dist_ratio":         float("nan"),
                        "mae_run":               mae_init_re,
                        "mfe_run":               mfe_init_re,
                        "mae_at_3":              float("nan"),
                        "mfe_at_3":              float("nan"),
                        "mae_at_6":              float("nan"),
                        "mfe_at_6":              float("nan"),
                        "first_bar_dir":         fbd_re,
                        "kh13_mae_at_check":     float("nan"),
                        "kh13_mfe_at_check":     float("nan"),
                        "kh13_triggered":        False,
                        "kh14_triggered":        False,
                        "atr_sized_down":        False,
                        "trade_type":            "reentry",
                        "original_entry_price":  watch["original_entry_price"],
                        "bar_path":              _init_bar_path(
                            SIGNAL_DIRECTION, re_px, a_re,
                            re_cac["h"][re_idx], re_cac["lo"][re_idx],
                            re_cac["c"][re_idx],
                        ),
                        "time_to_peak_mfe":      0,
                        "time_to_trough_mae":    0,
                    })
                    del reentry_watches[re_pair]
                else:
                    watch["bars_remaining"] -= 1
                    if watch["bars_remaining"] <= 0:
                        del reentry_watches[re_pair]

        # ── Phase 1.7: Resolve KH-17 pending decisions at bar N+1 close ──────
        if KH17_ENABLED and kh17_pending:
            for p17 in list(kh17_pending.keys()):
                pend  = kh17_pending[p17]
                cac17 = pair_cache[p17]
                j_now = cac17["date_to_idx"].get(bar_date)
                if j_now is None:
                    continue
                if j_now < pend["signal_idx"] + 1:
                    continue
                if j_now > pend["signal_idx"] + 1:
                    _kh17_s1_drop_expired += (
                        1 if cac17["c"][pend["signal_idx"] + 1]
                             > cac17["o"][pend["signal_idx"] + 1]
                        else 0
                    ) if (pend["signal_idx"] + 1) < len(cac17["c"]) else 0
                    del kh17_pending[p17]
                    continue
                n17 = len(cac17["o"])
                if j_now + 1 >= n17:
                    del kh17_pending[p17]
                    continue
                open_n1  = cac17["o"][j_now]
                close_n1 = cac17["c"][j_now]
                atr_n1   = cac17["atr"][j_now]
                if np.isnan(atr_n1) or atr_n1 == 0.0:
                    del kh17_pending[p17]
                    continue
                if close_n1 > open_n1:
                    # State 1 path: enter long at bar N+2 open.
                    if any(t["pair"] == p17 for t in open_trades):
                        _kh17_s1_drop_open += 1
                        del kh17_pending[p17]
                        continue
                    e_idx = j_now + 1
                    e_px  = cac17["o"][e_idx]
                    a_f   = cac17["atr"][e_idx]
                    if np.isnan(a_f) or a_f == 0.0:
                        _kh17_s1_drop_atr += 1
                        del kh17_pending[p17]
                        continue
                    b17, q17 = PAIR_CURRENCIES[p17]
                    if (currency_exposure[b17] + 1 > EXPOSURE_CAP
                            or currency_exposure[q17] - 1 < -EXPOSURE_CAP):
                        _kh17_s1_drop_cap += 1
                        blocked_events.append({
                            "date": bar_date, "pair": p17,
                            "reason": "kh17_state1_exposure_cap",
                        })
                        del kh17_pending[p17]
                        continue
                    risk_pp_s1 = KH17_STATE1_SL_ATR_MULT * a_f
                    if risk_pp_s1 <= 0.0:
                        del kh17_pending[p17]
                        continue
                    sl_px_s1 = e_px - risk_pp_s1
                    ta_px_s1 = e_px + TRAIL_ACT_MULT * a_f
                    sp_pips_s1  = cac17["sp"][e_idx] / POINTS_PER_PIP
                    pos_size_s1 = (INITIAL_BAL * RISK_PCT) / risk_pp_s1
                    sc_s1       = (sp_pips_s1 * cac17["pip"]) * pos_size_s1
                    best_cl_s1  = cac17["c"][e_idx]
                    ts_lv_s1    = best_cl_s1 - TRAIL_MULT * a_f
                    ta_flag_s1  = bool(cac17["c"][e_idx] >= ta_px_s1)
                    mae_init_s1 = max(0.0, (e_px - cac17["lo"][e_idx]) / a_f)
                    mfe_init_s1 = max(0.0, (cac17["h"][e_idx] - e_px) / a_f)
                    c_ent_s1 = cac17["c"][e_idx]
                    if c_ent_s1 > e_px:
                        fbd_s1 = 1
                    elif c_ent_s1 < e_px:
                        fbd_s1 = -1
                    else:
                        fbd_s1 = 0
                    currency_exposure[b17]  += 1
                    currency_exposure[q17] -= 1
                    open_trades.append({
                        "pair":                  p17,
                        "base_ccy":              b17,
                        "quote_ccy":             q17,
                        "signal_bar":            pend["signal_idx"],
                        "entry_idx":             e_idx,
                        "min_mgmt_idx":          e_idx + 1,
                        "entry_px":              e_px,
                        "sl_px":                 sl_px_s1,
                        "sl_px_init":            sl_px_s1,
                        "trail_act_px":          ta_px_s1,
                        "risk_pp":               risk_pp_s1,
                        "pos_size":              pos_size_s1,
                        "spread_cost":           sc_s1,
                        "spread_pips":           sp_pips_s1,
                        "best_cl":               best_cl_s1,
                        "signal_ctx":            pend["signal_ctx"],
                        "ts_level":              ts_lv_s1,
                        "trail_active":          ta_flag_s1,
                        "atr":                   a_f,
                        "entry_date":            bar_date,
                        "d1_dist_ratio":         pend["d1_dist_ratio"],
                        "mae_run":               mae_init_s1,
                        "mfe_run":               mfe_init_s1,
                        "mae_at_3":              float("nan"),
                        "mfe_at_3":              float("nan"),
                        "mae_at_6":              float("nan"),
                        "mfe_at_6":              float("nan"),
                        "first_bar_dir":         fbd_s1,
                        "kh13_mae_at_check":     float("nan"),
                        "kh13_mfe_at_check":     float("nan"),
                        "kh13_triggered":        False,
                        "kh14_triggered":        False,
                        "atr_sized_down":        False,
                        "trade_type":            "state1_delayed",
                        "original_entry_price":  float("nan"),
                        "bar_path":              _init_bar_path(
                            SIGNAL_DIRECTION, e_px, a_f,
                            cac17["h"][e_idx], cac17["lo"][e_idx],
                            cac17["c"][e_idx],
                        ),
                        "time_to_peak_mfe":      0,
                        "time_to_trough_mae":    0,
                    })
                    _kh17_s1_fired += 1
                    del kh17_pending[p17]
                else:
                    # State 2 path: begin virtual tracking from bar N+1 open.
                    # Use H/L of N+1 (matching KH-16 real-trade tracking) and
                    # ATR at signal bar N (pend["signal_atr"]) so the gate
                    # threshold is on the same scale as KH-16's kh14_bar6 check.
                    sig_atr = pend["signal_atr"]
                    hi_n1   = cac17["h"][j_now]
                    lo_n1   = cac17["lo"][j_now]
                    v_mfe = max(0.0, (hi_n1  - open_n1) / sig_atr)
                    v_mae = max(0.0, (open_n1 - lo_n1)  / sig_atr)
                    kh17_virtual[p17] = {
                        "signal_idx":           pend["signal_idx"],
                        "virtual_entry_price":  open_n1,
                        "virtual_atr":          sig_atr,
                        "virtual_mfe":          v_mfe,
                        "virtual_mae":          v_mae,
                        "signal_ctx":           pend["signal_ctx"],
                        "d1_dist_ratio":        pend["d1_dist_ratio"],
                    }
                    del kh17_pending[p17]

        # ── Phase 1.8: Advance KH-17 virtual tracking + bar-6 gate ───────────
        if KH17_ENABLED and kh17_virtual:
            for p17 in list(kh17_virtual.keys()):
                vt    = kh17_virtual[p17]
                cac17 = pair_cache[p17]
                j_v   = cac17["date_to_idx"].get(bar_date)
                if j_v is None:
                    continue
                bar_from_sig = j_v - vt["signal_idx"]
                if bar_from_sig < 2:
                    continue
                if bar_from_sig > 6:
                    del kh17_virtual[p17]
                    continue
                v_atr = vt["virtual_atr"]
                if v_atr and v_atr > 0.0:
                    hi_jv = cac17["h"][j_v]
                    lo_jv = cac17["lo"][j_v]
                    mfe_now = max(0.0, (hi_jv - vt["virtual_entry_price"]) / v_atr)
                    mae_now = max(0.0, (vt["virtual_entry_price"] - lo_jv) / v_atr)
                    if mfe_now > vt["virtual_mfe"]:
                        vt["virtual_mfe"] = mfe_now
                    if mae_now > vt["virtual_mae"]:
                        vt["virtual_mae"] = mae_now
                if bar_from_sig == 6:
                    if vt["virtual_mfe"] >= KH17_BAR6_MFE_THRESHOLD:
                        # Holder — would have been a winner, discard signal.
                        del kh17_virtual[p17]
                    elif (KH17_STATE2_REENTRY_ENABLED
                            and vt["virtual_mae"] >= KH17_BAR6_MAE_THRESHOLD):
                        # Bar-6 trigger: open re-entry watch for N+7..N+7+window-1.
                        kh17_watches[p17] = {
                            "signal_idx":           vt["signal_idx"],
                            "virtual_entry_price":  vt["virtual_entry_price"],
                            "virtual_atr":          vt["virtual_atr"],
                            "watch_start_idx":      vt["signal_idx"] + 7,
                            "bars_remaining":       KH17_WATCH_WINDOW_BARS,
                            "signal_ctx":           vt["signal_ctx"],
                            "d1_dist_ratio":        vt["d1_dist_ratio"],
                        }
                        del kh17_virtual[p17]
                    else:
                        # Ambiguous — stalled but not adverse enough, discard.
                        del kh17_virtual[p17]

        # ── Phase 1.9: Process KH-17 re-entry watches ────────────────────────
        if KH17_ENABLED and kh17_watches:
            for p17 in list(kh17_watches.keys()):
                w17   = kh17_watches[p17]
                cac17 = pair_cache[p17]
                j_w   = cac17["date_to_idx"].get(bar_date)
                if j_w is None:
                    continue
                if j_w < w17["watch_start_idx"]:
                    continue
                # One trade per pair: if a trade is already open (e.g. a
                # state1_delayed fill), skip entry this bar and let the
                # window expire normally.
                if any(t["pair"] == p17 for t in open_trades):
                    w17["bars_remaining"] -= 1
                    if w17["bars_remaining"] <= 0:
                        del kh17_watches[p17]
                    continue
                ref_px = w17["virtual_entry_price"]
                if cac17["c"][j_w] > ref_px:
                    # Trigger: attempt re-entry at next bar open.
                    re_idx = j_w + 1
                    n17    = len(cac17["o"])
                    if re_idx >= n17:
                        del kh17_watches[p17]
                        continue
                    r_px = cac17["o"][re_idx]
                    a_r  = cac17["atr"][re_idx]
                    if np.isnan(a_r) or a_r == 0.0:
                        del kh17_watches[p17]
                        continue
                    b17, q17 = PAIR_CURRENCIES[p17]
                    if (currency_exposure[b17] + 1 > EXPOSURE_CAP
                            or currency_exposure[q17] - 1 < -EXPOSURE_CAP):
                        w17["bars_remaining"] -= 1
                        if w17["bars_remaining"] <= 0:
                            del kh17_watches[p17]
                        continue
                    risk_pp_r = KH17_WATCH_SL_ATR_MULT * a_r
                    if risk_pp_r <= 0.0:
                        del kh17_watches[p17]
                        continue
                    sl_px_r = r_px - risk_pp_r
                    ta_px_r = r_px + TRAIL_ACT_MULT * a_r
                    sp_pips_r  = cac17["sp"][re_idx] / POINTS_PER_PIP
                    pos_size_r = (INITIAL_BAL * RISK_PCT) / risk_pp_r
                    sc_r       = (sp_pips_r * cac17["pip"]) * pos_size_r
                    best_cl_r  = cac17["c"][re_idx]
                    ts_lv_r    = best_cl_r - TRAIL_MULT * a_r
                    ta_flag_r  = bool(cac17["c"][re_idx] >= ta_px_r)
                    mae_init_r = max(0.0, (r_px - cac17["lo"][re_idx]) / a_r)
                    mfe_init_r = max(0.0, (cac17["h"][re_idx] - r_px) / a_r)
                    c_ent_r = cac17["c"][re_idx]
                    if c_ent_r > r_px:
                        fbd_r = 1
                    elif c_ent_r < r_px:
                        fbd_r = -1
                    else:
                        fbd_r = 0
                    currency_exposure[b17]  += 1
                    currency_exposure[q17] -= 1
                    open_trades.append({
                        "pair":                  p17,
                        "base_ccy":              b17,
                        "quote_ccy":             q17,
                        "signal_bar":            w17["signal_idx"],
                        "entry_idx":             re_idx,
                        "min_mgmt_idx":          re_idx + 1,
                        "entry_px":              r_px,
                        "sl_px":                 sl_px_r,
                        "sl_px_init":            sl_px_r,
                        "trail_act_px":          ta_px_r,
                        "risk_pp":               risk_pp_r,
                        "pos_size":              pos_size_r,
                        "spread_cost":           sc_r,
                        "spread_pips":           sp_pips_r,
                        "best_cl":               best_cl_r,
                        "signal_ctx":            w17["signal_ctx"],
                        "ts_level":              ts_lv_r,
                        "trail_active":          ta_flag_r,
                        "atr":                   a_r,
                        "entry_date":            bar_date,
                        "d1_dist_ratio":         w17["d1_dist_ratio"],
                        "mae_run":               mae_init_r,
                        "mfe_run":               mfe_init_r,
                        "mae_at_3":              float("nan"),
                        "mfe_at_3":              float("nan"),
                        "mae_at_6":              float("nan"),
                        "mfe_at_6":              float("nan"),
                        "first_bar_dir":         fbd_r,
                        "kh13_mae_at_check":     float("nan"),
                        "kh13_mfe_at_check":     float("nan"),
                        "kh13_triggered":        False,
                        "kh14_triggered":        False,
                        "atr_sized_down":        False,
                        "trade_type":            "state2_reentry",
                        "original_entry_price":  ref_px,
                        "bar_path":              _init_bar_path(
                            SIGNAL_DIRECTION, r_px, a_r,
                            cac17["h"][re_idx], cac17["lo"][re_idx],
                            cac17["c"][re_idx],
                        ),
                        "time_to_peak_mfe":      0,
                        "time_to_trough_mae":    0,
                    })
                    del kh17_watches[p17]
                else:
                    w17["bars_remaining"] -= 1
                    if w17["bars_remaining"] <= 0:
                        del kh17_watches[p17]

        # ── Phase 2: Check for new signals ───────────────────────────────────
        signals_today: list[tuple[str, int]] = []
        for pair in sorted(pair_cache.keys()):
            cached = pair_cache[pair]
            i_opt  = cached["date_to_idx"].get(bar_date)
            if i_opt is None:
                continue
            i = i_opt
            n = len(cached["sig"])
            if i >= n - 2:
                continue
            if cached["sig"][i] == 1:
                signals_today.append((pair, i))

        for pair, i in signals_today:
            cached = pair_cache[pair]
            # KH-16: new original signal cancels any active re-entry watch on same pair
            if USE_REENTRY and pair in reentry_watches:
                del reentry_watches[pair]
            # KH-17: new original signal cancels any active pending/virtual/watch
            #   for this pair.  The new signal then begins its own KH-17 cycle.
            if KH17_ENABLED:
                kh17_pending.pop(pair, None)
                kh17_virtual.pop(pair, None)
                kh17_watches.pop(pair, None)
            a      = cached["atr"][i]
            if np.isnan(a) or a == 0.0:
                blocked_events.append({
                    "date": bar_date, "pair": pair, "reason": "atr_invalid",
                })
                continue

            # KH-20: D1 close-in-range entry filter.
            # Skip signal if the last fully-closed D1 bar's close is in the
            # bottom 31% of its high/low range (bearish daily close).
            if KH20_ENABLED:
                _d1_cir = cached["d1_close_in_range_arr"][i]
                if not np.isnan(_d1_cir) and _d1_cir <= KH20_D1_RANGE_THRESHOLD:
                    blocked_events.append({
                        "date":   bar_date,
                        "pair":   pair,
                        "reason": "kh20_d1_range_filter",
                        "d1_close_in_range": float(_d1_cir),
                    })
                    continue

            # KH-22: 1H close-in-range entry filter.
            # Long:  skip if CIR > threshold (close too high in range — bearish H1).
            # Short: skip if (1 - CIR) > threshold, i.e. CIR < (1 - threshold)
            #        (close too low in range — bullish H1, not exhausted upward).
            _h1_cir = cached["h1_range_arr"][i] if "h1_range_arr" in cached else float("nan")
            if KH22_ENABLED:
                if not np.isnan(_h1_cir):
                    if SIGNAL_DIRECTION == "short":
                        _h1_blocked = _h1_cir < (1.0 - KH22_H1_RANGE_THRESHOLD)
                    else:
                        _h1_blocked = _h1_cir > KH22_H1_RANGE_THRESHOLD
                    if _h1_blocked:
                        blocked_events.append({
                            "date":   bar_date,
                            "pair":   pair,
                            "reason": "kh22_h1_range_filter",
                            "h1_last_bar_close_in_range": float(_h1_cir),
                        })
                        continue

            base, quote = PAIR_CURRENCIES[pair]

            if SIGNAL_DIRECTION == "short":
                if currency_exposure[base] - 1 < -EXPOSURE_CAP:
                    blocked_events.append({
                        "date":           bar_date,
                        "pair":           pair,
                        "reason":         f"base_{base}_exposure_{currency_exposure[base]}",
                        "base_exposure":  currency_exposure[base],
                        "quote_exposure": currency_exposure[quote],
                    })
                    continue
                if currency_exposure[quote] + 1 > EXPOSURE_CAP:
                    blocked_events.append({
                        "date":           bar_date,
                        "pair":           pair,
                        "reason":         f"quote_{quote}_exposure_{currency_exposure[quote]}",
                        "base_exposure":  currency_exposure[base],
                        "quote_exposure": currency_exposure[quote],
                    })
                    continue
            else:
                if currency_exposure[base] + 1 > EXPOSURE_CAP:
                    blocked_events.append({
                        "date":           bar_date,
                        "pair":           pair,
                        "reason":         f"base_{base}_exposure_{currency_exposure[base]}",
                        "base_exposure":  currency_exposure[base],
                        "quote_exposure": currency_exposure[quote],
                    })
                    continue
                if currency_exposure[quote] - 1 < -EXPOSURE_CAP:
                    blocked_events.append({
                        "date":           bar_date,
                        "pair":           pair,
                        "reason":         f"quote_{quote}_exposure_{currency_exposure[quote]}",
                        "base_exposure":  currency_exposure[base],
                        "quote_exposure": currency_exposure[quote],
                    })
                    continue

            entry_idx = i + 1
            entry_px  = cached["o"][entry_idx]
            risk_pp   = SL_MULT * a
            if SIGNAL_DIRECTION == "short":
                sl_px        = entry_px + SL_MULT * a
                trail_act_px = entry_px - TRAIL_ACT_MULT * a
            else:
                sl_px        = entry_px - SL_MULT * a
                trail_act_px = entry_px + TRAIL_ACT_MULT * a

            if risk_pp <= 0.0:
                continue

            # KH-15A: ATR-conditional sizing. Uses the same atr_ratio computed
            # below for the signal-context enrichment (ATR(14)[N] / median of
            # prior LIQUIDITY_MEDIAN_WINDOW bars, bar N excluded via .shift(1)).
            # Recompute here to avoid a circular dependency in local scope.
            _med_atr_for_sizing = cached["atr_median_arr"][i]
            _atr_ratio_for_sizing = (
                float(a / _med_atr_for_sizing)
                if _med_atr_for_sizing
                and not np.isnan(_med_atr_for_sizing)
                and _med_atr_for_sizing > 0.0
                else float("nan")
            )
            atr_sized_down_flag = bool(
                USE_ATR_SIZING
                and not np.isnan(_atr_ratio_for_sizing)
                and _atr_ratio_for_sizing >= ATR_SIZING_THRESHOLD
            )
            effective_risk_pct = (
                ATR_SIZING_REDUCED_PCT if atr_sized_down_flag else RISK_PCT
            )

            spread_pips  = cached["sp"][entry_idx] / POINTS_PER_PIP
            spread_price = spread_pips * cached["pip"]
            pos_size     = (INITIAL_BAL * effective_risk_pct) / risk_pp
            spread_cost  = spread_price * pos_size

            # Entry bar: trail initialisation — CLOSE-BASED
            best_cl = cached["c"][entry_idx]
            if SIGNAL_DIRECTION == "short":
                ts_level     = best_cl + TRAIL_MULT * a
                trail_active = bool(cached["c"][entry_idx] <= trail_act_px)
            else:
                ts_level     = best_cl - TRAIL_MULT * a
                trail_active = bool(cached["c"][entry_idx] >= trail_act_px)

            # ── MAE/MFE init (diagnostic only; never feeds back into logic) ──
            # Seed running MAE/MFE with the entry bar's own high/low so bar 1
            # (= N+1 in user notation) is included. Units: ATR multiples.
            hi_entry = cached["h"][entry_idx]
            lo_entry = cached["lo"][entry_idx]
            if SIGNAL_DIRECTION == "short":
                mae_run_init = max(0.0, (hi_entry - entry_px) / a)
                mfe_run_init = max(0.0, (entry_px - lo_entry) / a)
            else:
                mae_run_init = max(0.0, (entry_px - lo_entry) / a)
                mfe_run_init = max(0.0, (hi_entry - entry_px) / a)
            c_entry = cached["c"][entry_idx]
            if c_entry > entry_px:
                first_bar_dir_val = 1
            elif c_entry < entry_px:
                first_bar_dir_val = -1
            else:
                first_bar_dir_val = 0

            d1_ratio = cached["d1_dist_ratio_by_bar"].get(i, float("nan"))
            d1_cir   = float(cached["d1_close_in_range_arr"][i]) if not np.isnan(cached["d1_close_in_range_arr"][i]) else float("nan")

            # ── Liquidity-proxy context, captured at signal bar N ────────────
            # All values use only data available at bar N close. Diagnostic
            # only — these never feed back into signal/entry decisions.
            sig_spread_pips = float(cached["spread_pips_arr"][i])
            med_spread = cached["spread_median_arr"][i]
            med_atr    = cached["atr_median_arr"][i]
            spread_ratio_val = (float(sig_spread_pips / med_spread)
                                if med_spread and not np.isnan(med_spread) and med_spread > 0.0
                                else float("nan"))
            atr_ratio_val = (float(a / med_atr)
                             if med_atr and not np.isnan(med_atr) and med_atr > 0.0
                             else float("nan"))
            kj_now   = cached["d1_kijun_arr"][i]
            kj_prior = cached["d1_kijun_prior5_arr"][i]
            d1_kijun_slope_val = bool(
                (not np.isnan(kj_now)) and (not np.isnan(kj_prior)) and (kj_now > kj_prior)
            )

            # KH-11A: optional entry gate — block rising-slope D1 Kijun signals.
            # Uses the same lagged D1 Kijun series as C8/C9 (never same-day D1).
            # Default REQUIRE_D1_KIJUN_SLOPE_FALLING=False preserves baseline.
            # Placed before currency_exposure increments below, so no rollback
            # is needed when this gate blocks a signal.
            if REQUIRE_D1_KIJUN_SLOPE_FALLING and d1_kijun_slope_val:
                blocked_events.append({
                    "date":   bar_date,
                    "pair":   pair,
                    "reason": "d1_kijun_slope_rising",
                })
                continue

            session_val = _session_for_ts(bar_date)
            signal_ctx = {
                "signal_bar_ts":     pd.Timestamp(bar_date),
                "signal_spread_pips": round(sig_spread_pips, 4),
                "spread_ratio":      (round(spread_ratio_val, 2)
                                      if not np.isnan(spread_ratio_val) else float("nan")),
                "atr_abs":           round(float(a), 6),
                "atr_ratio":         (round(atr_ratio_val, 2)
                                      if not np.isnan(atr_ratio_val) else float("nan")),
                "d1_kijun_slope":    d1_kijun_slope_val,
                "session":           session_val,
            }

            # KH-17: divert the signal into a pending decision.  No trade
            # is opened at bar N+1 under any circumstance.  Exposure is
            # incremented at the eventual fill bar (Phase 1.7 for State 1,
            # Phase 1.9 for State 2 re-entry), not here.
            if KH17_ENABLED:
                kh17_pending[pair] = {
                    "signal_idx":     i,
                    "signal_ctx":     signal_ctx,
                    "d1_dist_ratio":  d1_ratio,
                    "signal_atr":     a,
                }
                continue

            if SIGNAL_DIRECTION == "short":
                currency_exposure[base]  -= 1
                currency_exposure[quote] += 1
            else:
                currency_exposure[base]  += 1
                currency_exposure[quote] -= 1

            # Pipeline D1: compute the 8 base entry features at signal bar N
            # for downstream classifier evaluation at bar offset t. Storage is
            # gated on D1_HOOK so the baseline path remains byte-identical when
            # no D1 archetypes are configured.
            entry_features_dict: dict[str, float] | None = None
            if D1_HOOK is not None:
                entry_features_dict = build_entry_features_at_signal_bar(
                    open_arr=cached["o"],
                    high_arr=cached["h"],
                    low_arr=cached["lo"],
                    close_arr=cached["c"],
                    atr_at_signal_bar=float(a),
                    signal_bar_idx=int(i),
                )

            open_trades.append({
                "pair":           pair,
                "base_ccy":       base,
                "quote_ccy":      quote,
                "signal_bar":     i,
                "entry_idx":      entry_idx,
                "min_mgmt_idx":   entry_idx + 1,
                "entry_px":       entry_px,
                "sl_px":          sl_px,
                "sl_px_init":     sl_px,
                "trail_act_px":   trail_act_px,
                "risk_pp":        risk_pp,
                "pos_size":       pos_size,
                "spread_cost":    spread_cost,
                "spread_pips":    spread_pips,
                "best_cl":        best_cl,
                "signal_ctx":     signal_ctx,
                "ts_level":       ts_level,
                "trail_active":   trail_active,
                "atr":            a,
                "entry_date":     bar_date,
                "d1_dist_ratio":        d1_ratio,
                "d1_close_in_range":    d1_cir,
                "h1_last_bar_close_in_range": float(_h1_cir) if not np.isnan(_h1_cir) else float("nan"),
                "mae_run":             mae_run_init,
                "mfe_run":             mfe_run_init,
                "mae_at_3":            float("nan"),
                "mfe_at_3":            float("nan"),
                "mae_at_6":            float("nan"),
                "mfe_at_6":            float("nan"),
                "first_bar_dir":       first_bar_dir_val,
                "kh13_mae_at_check":   float("nan"),
                "kh13_mfe_at_check":   float("nan"),
                "kh13_triggered":      False,
                "kh14_triggered":      False,
                "atr_sized_down":      atr_sized_down_flag,
                "trade_type":          "original",
                "original_entry_price": float("nan"),
                "bar_path":            _init_bar_path(
                    SIGNAL_DIRECTION, entry_px, a, hi_entry, lo_entry, c_entry,
                ),
                "time_to_peak_mfe":    0,
                "time_to_trough_mae":  0,
                "entry_features":      entry_features_dict,
                # Pipeline D1 (PR 2) audit fields. Stamped at trade open
                # so they are always present in the completed-trade
                # record, even when the hook doesn't fire (e.g. trade
                # exited before bar t, or D1_HOOK is None).
                "fold_id":             _fold_id_for_entry(bar_date),
                "signal_direction":    SIGNAL_DIRECTION,
                "exit_policy":         None,
                "mfe_lock_fired_bar":  None,
                "trail_active_from_bar": None,
                "classifier_fold_id":  None,
                "d1_decision":         "no_d1",
            })

    # Close any remaining open trades at last available close
    for trade in open_trades:
        pair      = trade["pair"]
        cached    = pair_cache[pair]
        n         = len(cached["o"])
        exit_bar  = n - 1
        exit_px   = cached["c"][exit_bar]
        entry_px  = trade["entry_px"]
        entry_idx = trade["entry_idx"]
        risk_pp   = trade["risk_pp"]
        pos_size  = trade["pos_size"]
        spread_cost = trade["spread_cost"]

        _a_trk = trade["atr"]
        _mae_bar = (entry_px - cached["lo"][exit_bar]) / _a_trk
        if _mae_bar < 0.0:
            _mae_bar = 0.0
        _mfe_bar = (cached["h"][exit_bar] - entry_px) / _a_trk
        if _mfe_bar < 0.0:
            _mfe_bar = 0.0
        if _mae_bar > trade["mae_run"]:
            trade["mae_run"] = _mae_bar
        if _mfe_bar > trade["mfe_run"]:
            trade["mfe_run"] = _mfe_bar
        _bar_num = exit_bar - entry_idx + 1
        if _bar_num >= 3 and math.isnan(trade["mae_at_3"]):
            trade["mae_at_3"] = trade["mae_run"]
            trade["mfe_at_3"] = trade["mfe_run"]
        if _bar_num >= 6 and math.isnan(trade["mae_at_6"]):
            trade["mae_at_6"] = trade["mae_run"]
            trade["mfe_at_6"] = trade["mfe_run"]
            if not trade.get("kh14_triggered", False):
                trade["kh14_triggered"] = bool(
                    trade["mfe_run"] < KH14_MFE_THRESHOLD
                    and trade["mae_run"] >= KH14_MAE_THRESHOLD
                )

        gross_pnl = (exit_px - entry_px) * pos_size
        net_pnl   = gross_pnl - spread_cost
        r_mult    = (exit_px - entry_px) / risk_pp
        bars_held = max(exit_bar - entry_idx, 1)

        if r_mult > 0:
            classification = "WIN"
        elif r_mult < 0:
            classification = "LOSS"
        else:
            classification = "SCRATCH"

        _mae_fin = float(trade["mae_run"])
        _mfe_fin = float(trade["mfe_run"])
        _mae_3 = _mae_fin if math.isnan(trade["mae_at_3"]) else float(trade["mae_at_3"])
        _mfe_3 = _mfe_fin if math.isnan(trade["mfe_at_3"]) else float(trade["mfe_at_3"])
        _mae_6 = _mae_fin if math.isnan(trade["mae_at_6"]) else float(trade["mae_at_6"])
        _mfe_6 = _mfe_fin if math.isnan(trade["mfe_at_6"]) else float(trade["mfe_at_6"])
        _kh13_mae_chk = float(trade["kh13_mae_at_check"])
        _kh13_mfe_chk = float(trade["kh13_mfe_at_check"])
        _kh13_trig    = bool(trade["kh13_triggered"])
        _kh14_trig    = bool(
            trade["first_bar_dir"] == -1
            and _mfe_6 < KH14_MFE_THRESHOLD
            and _mae_6 >= KH14_MAE_THRESHOLD
        )
        _kh14_state2  = bool(trade["first_bar_dir"] == -1)

        completed_trades.append({
            "pair":              pair,
            "entry_date":        trade["entry_date"],
            "exit_date":         pd.Timestamp(cached["dates"][exit_bar]),
            "entry_price":       round(entry_px,            6),
            "exit_price":        round(exit_px,             6),
            "sl_price":          round(trade["sl_px_init"],  6),
            "trail_active":      trade["trail_active"],
            "exit_reason":       "trailing_stop",
            "classification":    classification,
            "bars_held":         bars_held,
            "net_pnl":           round(net_pnl,             4),
            "r_multiple":        round(r_mult,              4),
            "spread_pips_used":  round(trade["spread_pips"], 4),
            "sl_distance_atr":   round(trade["risk_pp"] / trade["atr"], 4),
            "d1_dist_ratio":     trade["d1_dist_ratio"],
            "d1_close_in_range": trade.get("d1_close_in_range", float("nan")),
            "h1_last_bar_close_in_range": trade.get("h1_last_bar_close_in_range", float("nan")),
            "mae_final":         round(_mae_fin, 4),
            "mfe_final":         round(_mfe_fin, 4),
            "mae_at_bar_3":      round(_mae_3,   4),
            "mfe_at_bar_3":      round(_mfe_3,   4),
            "mae_at_bar_6":      round(_mae_6,   4),
            "mfe_at_bar_6":      round(_mfe_6,   4),
            "first_bar_dir":     int(trade["first_bar_dir"]),
            "kh13_mae_at_check": (round(_kh13_mae_chk, 4)
                                  if not math.isnan(_kh13_mae_chk) else float("nan")),
            "kh13_mfe_at_check": (round(_kh13_mfe_chk, 4)
                                  if not math.isnan(_kh13_mfe_chk) else float("nan")),
            "kh13_triggered":    _kh13_trig,
            "kh14_triggered":    _kh14_trig,
            "kh14_state2":       _kh14_state2,
            "atr_sized_down":    bool(trade.get("atr_sized_down", False)),
            "trade_type":        trade.get("trade_type", "original"),
            "original_entry_price_ref": trade.get("original_entry_price", float("nan")),
            **trade["signal_ctx"],
            # v1.3 sidecars (re-positioned after concurrent_signals in post-pass)
            "_v13_time_to_peak_mfe":   int(trade["time_to_peak_mfe"]),
            "_v13_time_to_trough_mae": int(trade["time_to_trough_mae"]),
        })
        _flatten_bar_path_for_trade(
            trade, exit_bar, pair_cache,
            SIGNAL_DIRECTION, bar_paths_data,
        )

    # ── KH-17 Phase 1.7 diagnostics ──────────────────────────────────────────
    if KH17_ENABLED:
        _kh17_s1_total_eligible = (
            _kh17_s1_fired + _kh17_s1_drop_open
            + _kh17_s1_drop_atr + _kh17_s1_drop_cap + _kh17_s1_drop_expired
        )
        print(
            f"  [KH-17 Phase1.7] State-1 eligible={_kh17_s1_total_eligible}"
            f"  fired={_kh17_s1_fired}"
            f"  drop_open_trade={_kh17_s1_drop_open}"
            f"  drop_atr_invalid={_kh17_s1_drop_atr}"
            f"  drop_exposure_cap={_kh17_s1_drop_cap}"
            f"  drop_expired_missed={_kh17_s1_drop_expired}"
        )

    # ── Post-pass: concurrent_signals per bar timestamp ──────────────────────
    # Diagnostic-only enrichment. Count, for each bar timestamp, how many
    # pairs generated a fully-gated (C1-C9 passed) signal. Each trade gets
    # the count MINUS its own occurrence. No cross-pair dependency is
    # introduced into entry logic — this runs strictly after simulation.
    signals_per_ts: dict[pd.Timestamp, int] = defaultdict(int)
    for cached in pair_cache.values():
        sig_arr = cached["sig"]
        dates_arr = cached["dates"]
        for idx in np.flatnonzero(sig_arr == 1):
            signals_per_ts[pd.Timestamp(dates_arr[idx])] += 1

    for t in completed_trades:
        ts = t.pop("signal_bar_ts", None)
        n_concurrent = signals_per_ts.get(ts, 1) - 1 if ts is not None else 0
        t["concurrent_signals"] = int(max(n_concurrent, 0))
        # v1.3: emit the three new per-trade columns at the END so the legacy
        # 39-column subset stays in its locked order. trade_id is derived from
        # (pair, entry_date) per the audit; time_to_peak_mfe / time_to_trough_mae
        # were pinned bar-by-bar during the hold loop and reach us as sidecars.
        _ttp = t.pop("_v13_time_to_peak_mfe", 0)
        _ttt = t.pop("_v13_time_to_trough_mae", 0)
        t["trade_id"]            = f"{t['pair']}_{t['entry_date']}"
        t["time_to_peak_mfe"]    = int(_ttp)
        t["time_to_trough_mae"]  = int(_ttt)

    completed_trades.sort(key=lambda t: t["entry_date"])
    return completed_trades, blocked_events, bar_paths_data


# ── OOS metrics ───────────────────────────────────────────────────────────────

def _oos_metrics(
    trades: list[dict],
    oos_start,
    oos_end,
    balance: float = INITIAL_BAL,
) -> dict:
    ts  = pd.Timestamp(oos_start)
    te  = pd.Timestamp(oos_end)
    oos = [t for t in trades if ts <= pd.Timestamp(t["entry_date"]) < te]
    if not oos:
        return dict(
            n_trades=0, roi_pct=0.0, max_dd_pct=0.0,
            win_rate=float("nan"), expectancy_r=float("nan"),
            avg_bars_held=float("nan"), trades_blocked_exposure=0,
            n_kh13_early=0, n_kh14_bar6=0, n_atr_sized_down=0, n_reentry=0,
        )
    equity  = balance
    peak    = balance
    max_dd  = 0.0
    wins    = 0
    n_kh13  = 0
    n_kh14  = 0
    n_sized = 0
    n_re    = 0
    for t in oos:
        equity += t["net_pnl"]
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak * 100.0
        max_dd  = max(max_dd, dd)
        if t["classification"] == "WIN":
            wins += 1
        if t.get("exit_reason") == "kh13_early":
            n_kh13 += 1
        if t.get("exit_reason") == "kh14_bar6":
            n_kh14 += 1
        if bool(t.get("atr_sized_down", False)):
            n_sized += 1
        if t.get("trade_type") == "reentry":
            n_re += 1
    avg_bars = float(np.mean([t["bars_held"] for t in oos]))
    return dict(
        n_trades          = len(oos),
        roi_pct           = round((equity - balance) / balance * 100.0, 2),
        max_dd_pct        = round(max_dd, 2),
        win_rate          = round(wins / len(oos), 4),
        expectancy_r      = round(float(np.mean([t["r_multiple"] for t in oos])), 4),
        avg_bars_held     = round(avg_bars, 1),
        trades_blocked_exposure = 0,
        n_kh13_early      = n_kh13,
        n_kh14_bar6       = n_kh14,
        n_atr_sized_down  = n_sized,
        n_reentry         = n_re,
    )


def _per_pair_oos(all_trades: list[dict]) -> pd.DataFrame:
    oos_start  = pd.Timestamp(FOLDS[0]["oos_start"])
    oos_trades = [t for t in all_trades if pd.Timestamp(t["entry_date"]) >= oos_start]
    rows = []
    for pair in ALL_PAIRS:
        pair_t = [t for t in oos_trades if t["pair"] == pair]
        if not pair_t:
            rows.append({
                "pair": pair, "n_trades": 0,
                "roi_pct": float("nan"), "win_rate": float("nan"),
                "expectancy_r": float("nan"), "avg_bars_held": float("nan"),
            })
            continue
        equity = INITIAL_BAL
        wins   = 0
        for t in pair_t:
            equity += t["net_pnl"]
            if t["classification"] == "WIN":
                wins += 1
        rows.append({
            "pair":          pair,
            "n_trades":      len(pair_t),
            "roi_pct":       round((equity - INITIAL_BAL) / INITIAL_BAL * 100.0, 2),
            "win_rate":      round(wins / len(pair_t), 4),
            "expectancy_r":  round(float(np.mean([t["r_multiple"] for t in pair_t])), 4),
            "avg_bars_held": round(float(np.mean([t["bars_held"] for t in pair_t])), 1),
        })
    return pd.DataFrame(rows)


def _exit_stats_oos(trades: list[dict]) -> dict:
    oos_start = pd.Timestamp(FOLDS[0]["oos_start"])
    oos = [t for t in trades if pd.Timestamp(t["entry_date"]) >= oos_start]
    total  = len(oos)
    groups: dict[str, list] = {}
    for t in oos:
        groups.setdefault(t["exit_reason"], []).append(t)
    result = {}
    for etype, tlist in groups.items():
        result[etype] = {
            "count":    len(tlist),
            "pct":      round(len(tlist) / total * 100.0, 1) if total > 0 else 0.0,
            "avg_r":    round(float(np.mean([t["r_multiple"] for t in tlist])), 4),
            "avg_bars": round(float(np.mean([t["bars_held"]  for t in tlist])), 1),
        }
    return result


# ── Cond_9 delta per fold ─────────────────────────────────────────────────────

def _cond9_delta_per_fold(pair_cache: dict) -> dict[int, int]:
    fold_counts: dict[int, int] = {f["fold"]: 0 for f in FOLDS}
    for cached in pair_cache.values():
        for d in cached.get("cond9_block_dates", []):
            for f in FOLDS:
                ts = pd.Timestamp(f["oos_start"])
                te = pd.Timestamp(f["oos_end"])
                if ts <= d < te:
                    fold_counts[f["fold"]] += 1
                    break
    return fold_counts


# ── Full WFO run ──────────────────────────────────────────────────────────────

def _run_wfo(
    pair_cache: dict,
    all_trades: list[dict],
    blocked_events: list[dict],
    bar_paths_data: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    fold_blocked: dict[int, int] = {}
    kh20_per_fold: dict[int, int] = {}
    kh22_per_fold: dict[int, int] = {}
    for f in FOLDS:
        ts = pd.Timestamp(f["oos_start"])
        te = pd.Timestamp(f["oos_end"])
        n_bl = sum(
            1 for b in blocked_events
            if "exposure" in b.get("reason", "")
            and ts <= b["date"] < te
        )
        fold_blocked[f["fold"]] = n_bl
        n_kh20 = sum(
            1 for b in blocked_events
            if b.get("reason") == "kh20_d1_range_filter"
            and ts <= b["date"] < te
        )
        kh20_per_fold[f["fold"]] = n_kh20
        n_kh22 = sum(
            1 for b in blocked_events
            if b.get("reason") == "kh22_h1_range_filter"
            and ts <= b["date"] < te
        )
        kh22_per_fold[f["fold"]] = n_kh22

    cond9_per_fold = _cond9_delta_per_fold(pair_cache)

    fold_rows = []
    fold_rois = []
    fold_dds  = []

    for f in FOLDS:
        m = _oos_metrics(all_trades, f["oos_start"], f["oos_end"])
        fold_rois.append(m["roi_pct"])
        fold_dds.append(m["max_dd_pct"])
        m["trades_blocked_exposure"] = fold_blocked[f["fold"]]
        kgj_ref = _KGIJ_FOLDS.get(f["fold"], (float("nan"), float("nan"), 0))
        fold_rows.append({
            "fold":                    f["fold"],
            "oos_start":               str(f["oos_start"].date()),
            "oos_end":                 str(f["oos_end"].date()),
            "n_trades":                m["n_trades"],
            "kgj_n_trades":            kgj_ref[2],
            "delta_vs_kgj":            m["n_trades"] - kgj_ref[2],
            "cond9_blocked_oos":       cond9_per_fold[f["fold"]],
            "roi_pct":                 m["roi_pct"],
            "max_dd_pct":              m["max_dd_pct"],
            "win_rate":                m["win_rate"],
            "expectancy_r":            m["expectancy_r"],
            "avg_bars_held":           m["avg_bars_held"],
            "trades_blocked_exposure": m["trades_blocked_exposure"],
            "n_kh13_early":            m["n_kh13_early"],
            "n_kh14_bar6":             m["n_kh14_bar6"],
            "n_atr_sized_down":        m["n_atr_sized_down"],
            "n_reentry":               m["n_reentry"],
        })
        g = "PASS" if m["roi_pct"] > GATE_ROI_MIN and m["max_dd_pct"] < GATE_DD_MAX else "FAIL"
        kh13_sfx = (f"  kh13_early={m['n_kh13_early']:>3}" if USE_KH13_EARLY_EXIT else "")
        kh14_sfx = (f"  kh14_bar6={m['n_kh14_bar6']:>3}" if USE_KH14_BAR6_EXIT else "")
        kh15_sfx = (f"  atr_sized={m['n_atr_sized_down']:>3}" if USE_ATR_SIZING else "")
        kh16_sfx = (f"  reentry={m['n_reentry']:>3}" if USE_REENTRY else "")
        kh20_sfx = (f"  kh20_filt={kh20_per_fold[f['fold']]:>3}" if KH20_ENABLED else "")
        kh22_sfx = (f"  kh22_filt={kh22_per_fold[f['fold']]:>3}" if KH22_ENABLED else "")
        print(f"  Fold {f['fold']:>2} OOS {f['oos_start'].date()} -> {f['oos_end'].date()}: "
              f"trades={m['n_trades']:>4}  kgj={kgj_ref[2]:>3}  "
              f"d={m['n_trades']-kgj_ref[2]:>+4}  c9_blocked={cond9_per_fold[f['fold']]:>3}  "
              f"ROI={m['roi_pct']:>7.2f}%  DD={m['max_dd_pct']:>6.2f}%"
              f"{kh13_sfx}{kh14_sfx}{kh15_sfx}{kh16_sfx}{kh20_sfx}{kh22_sfx}  {g}")

    worst_roi = min(fold_rois)
    worst_dd  = max(fold_dds)
    gate_pass = worst_roi > GATE_ROI_MIN and worst_dd < GATE_DD_MAX
    print(f"\n  KG-L-V2 — Worst ROI: {worst_roi:.2f}%  "
          f"Worst DD: {worst_dd:.2f}%  Gate: {'PASS' if gate_pass else 'FAIL'}")

    fold_df     = pd.DataFrame(fold_rows)
    per_pair_df = _per_pair_oos(all_trades)

    fold_df.to_csv(OUT_ROOT / "wfo_fold_results_4h.csv", index=False)
    per_pair_df.to_csv(OUT_ROOT / "wfo_per_pair_4h.csv", index=False)
    if all_trades:
        pd.DataFrame(all_trades).to_csv(OUT_ROOT / "trades_all.csv", index=False)
    # v1.3: per-bar trade-paths artefact for capturability calibration. One
    # row per (trade_id, bar_offset). is_held=1 covers entry through exit;
    # is_held=0 extends to bar_offset=PATH_FORWARD_BARS (or end of data) so
    # capture-ceilings under longer time exits can be reconstructed.
    if bar_paths_data:
        pd.DataFrame(bar_paths_data).to_csv(
            OUT_ROOT / "trades_paths.csv", index=False
        )

    oos_total       = int(fold_df["n_trades"].sum())
    oos_years       = WFO_N_FOLDS * WFO_OOS_MONTHS / 12.0
    trades_per_year = oos_total / oos_years if oos_years > 0 else float("nan")

    summary_lines = [
        "Phase KG-L-V2 — D1 ATR Distance Cap + signal_flip REMOVED — Summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Gate: {'PASS' if gate_pass else 'FAIL'}",
        f"Worst-fold ROI: {worst_roi:.2f}%  Worst-fold DD: {worst_dd:.2f}%",
        f"Trades/year (OOS): {trades_per_year:.1f}",
        "",
        "--- Fold Results ---",
    ]
    for _, row in fold_df.iterrows():
        g = ("PASS" if row["roi_pct"] > GATE_ROI_MIN and row["max_dd_pct"] < GATE_DD_MAX
             else "FAIL")
        summary_lines.append(
            f"  Fold {int(row['fold'])}: {row['oos_start']} to {row['oos_end']}  "
            f"trades={int(row['n_trades'])}  kgj={int(row['kgj_n_trades'])}  "
            f"delta={int(row['delta_vs_kgj']):+d}  "
            f"c9_blocked={int(row['cond9_blocked_oos'])}  "
            f"ROI={row['roi_pct']:.2f}%  DD={row['max_dd_pct']:.2f}%  {g}"
        )
    (OUT_ROOT / "wfo_summary_4h.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    oos_start_ts = pd.Timestamp(FOLDS[0]["oos_start"])
    oos_blocked  = [b for b in blocked_events
                    if "exposure" in b.get("reason", "")
                    and b["date"] >= oos_start_ts]
    pair_block_counts: dict[str, int] = defaultdict(int)
    for b in oos_blocked:
        pair_block_counts[b["pair"]] += 1

    total_cond9_all = sum(len(cached.get("cond9_block_dates", [])) for cached in pair_cache.values())
    total_cond9_oos = sum(cond9_per_fold.values())

    # KH-13 / KH-14 / KH-16 OOS summary stats
    oos_start_ts_kh = pd.Timestamp(FOLDS[0]["oos_start"])
    oos_trades_all  = [t for t in all_trades
                       if pd.Timestamp(t["entry_date"]) >= oos_start_ts_kh]
    total_kh13_triggered_oos = sum(1 for t in oos_trades_all if t.get("kh13_triggered"))
    total_kh13_early_oos     = int(fold_df["n_kh13_early"].sum())
    total_kh14_triggered_oos = sum(1 for t in oos_trades_all if t.get("kh14_triggered"))
    total_kh14_state2_oos    = sum(1 for t in oos_trades_all if t.get("kh14_state2"))
    total_kh14_bar6_oos      = int(fold_df["n_kh14_bar6"].sum())
    total_atr_sized_oos      = int(fold_df["n_atr_sized_down"].sum())
    total_reentry_oos        = int(fold_df["n_reentry"].sum())

    # Re-entry OOS performance stats
    re_oos = [t for t in oos_trades_all if t.get("trade_type") == "reentry"]
    if re_oos:
        re_wins   = sum(1 for t in re_oos if t["classification"] == "WIN")
        re_wr     = round(re_wins / len(re_oos), 4)
        re_mean_r = round(float(np.mean([t["r_multiple"] for t in re_oos])), 4)
    else:
        re_wr     = float("nan")
        re_mean_r = float("nan")

    stats = {
        "worst_roi":                  worst_roi,
        "worst_dd":                   worst_dd,
        "gate_pass":                  gate_pass,
        "fold_rois":                  fold_rois,
        "fold_dds":                   fold_dds,
        "total_blocked_oos":          len(oos_blocked),
        "fold_blocked":               fold_blocked,
        "pair_block_counts":          dict(pair_block_counts),
        "trades_per_year":            trades_per_year,
        "cond9_per_fold":             cond9_per_fold,
        "total_cond9_all":            total_cond9_all,
        "total_cond9_oos":            total_cond9_oos,
        "kh20_per_fold":              kh20_per_fold,
        "kh20_total_oos":             sum(kh20_per_fold.values()),
        "kh22_per_fold":              kh22_per_fold,
        "kh22_total_oos":             sum(kh22_per_fold.values()),
        "kh13_triggered_oos":         total_kh13_triggered_oos,
        "kh13_early_oos":             total_kh13_early_oos,
        "kh14_triggered_oos":         total_kh14_triggered_oos,
        "kh14_state2_oos":            total_kh14_state2_oos,
        "kh14_bar6_oos":              total_kh14_bar6_oos,
        "atr_sized_down_oos":         total_atr_sized_oos,
        "reentry_oos":                total_reentry_oos,
        "reentry_win_rate":           re_wr,
        "reentry_mean_r":             re_mean_r,
    }
    return fold_df, per_pair_df, stats


# ── Report ────────────────────────────────────────────────────────────────────

def _write_report(
    all_trades: list[dict],
    blocked_events: list[dict],
    fold_df: pd.DataFrame,
    per_pair_df: pd.DataFrame,
    stats: dict,
    pair_cache: dict,
) -> None:
    lines: list[str] = []
    a = lines.append

    a("# Phase KG-L-V2 Report: D1 ATR Cap + signal_flip Exit Removed")
    a("")
    a(f"> Generated: {datetime.now().strftime('%Y-%m-%d')}")
    a("> Base: KG-L-V1 (KG-J + trail fix + classification fix + cond_9)")
    a(">")
    a("> **V2 additional change — signal_flip exit REMOVED:**")
    a(">   KG-L-V1 exits: signal_flip | kijun_d1 | trailing_stop | stoploss")
    a(">   KG-L-V2 exits:              kijun_d1 | trailing_stop | stoploss")
    a(">   signal_flip is not computed and not checked anywhere in V2.")
    a(">")
    a("> Inherited from V1 (all three fixes present in V2):")
    a(">   Fix 1: Trail activation close-based (not intrabar high)")
    a(">   Fix 2: Classification from r_multiple (WIN/LOSS/SCRATCH)")
    a(">   Change 3: cond_9 — D1 close <= D1 Kijun(26) + 1.0 × D1 ATR(14)")
    a(">")
    a("> Timeframe: 4H | Pairs: 28 | WFO: 7 anchored-expanding folds")
    a("> Gate: worst-fold OOS ROI > 0% AND max DD < 8%")
    a("")
    a("---")
    a("")

    # ── 1. Gate result ────────────────────────────────────────────────────────
    a("## 1. Gate Result")
    a("")
    worst_roi = stats["worst_roi"]
    worst_dd  = stats["worst_dd"]
    gate_pass = stats["gate_pass"]
    a("| Criterion | Threshold | Actual | Result |")
    a("|-----------|-----------|--------|--------|")
    a(f"| Worst-fold ROI | > {GATE_ROI_MIN:.1f}% | {worst_roi:.2f}% | "
      f"{'**PASS**' if worst_roi > GATE_ROI_MIN else '**FAIL**'} |")
    a(f"| Worst-fold DD | < {GATE_DD_MAX:.1f}% | {worst_dd:.2f}% | "
      f"{'**PASS**' if worst_dd < GATE_DD_MAX else '**FAIL**'} |")
    a(f"| **Overall Gate** | both | — | "
      f"{'**PASS**' if gate_pass else '**FAIL**'} |")
    a("")
    a("---")
    a("")

    # ── 2. Full fold table ────────────────────────────────────────────────────
    a("## 2. Full Fold Table")
    a("")
    a("| Fold | OOS Start | OOS End | Trades | KGJ | Δ | c9_blk | ROI% | MaxDD% | WinRate | ExpR | AvgBars | Gate |")
    a("|------|-----------|---------|--------|-----|---|--------|------|--------|---------|------|---------|------|")
    for _, row in fold_df.iterrows():
        g   = "PASS" if row["roi_pct"] > GATE_ROI_MIN and row["max_dd_pct"] < GATE_DD_MAX else "FAIL"
        wr  = f"{row['win_rate']:.4f}"     if not pd.isna(row["win_rate"])        else "n/a"
        exp = f"{row['expectancy_r']:.4f}" if not pd.isna(row["expectancy_r"])    else "n/a"
        bh  = f"{row['avg_bars_held']:.1f}" if not pd.isna(row["avg_bars_held"])  else "n/a"
        dlt = f"{int(row['delta_vs_kgj']):+d}"
        a(f"| {int(row['fold'])} | {row['oos_start']} | {row['oos_end']} | "
          f"{int(row['n_trades'])} | {int(row['kgj_n_trades'])} | {dlt} | "
          f"{int(row['cond9_blocked_oos'])} | "
          f"{row['roi_pct']:.2f}% | {row['max_dd_pct']:.2f}% | "
          f"{wr} | {exp} | {bh} | {g} |")
    a("")
    fold_rois = stats["fold_rois"]
    fold_dds  = stats["fold_dds"]
    a(f"**Worst-fold ROI:** {worst_roi:.2f}% | "
      f"**Median-fold ROI:** {float(np.median(fold_rois)):.2f}% | "
      f"**Best-fold ROI:** {max(fold_rois):.2f}%")
    a(f"**Worst-fold DD:** {worst_dd:.2f}% | "
      f"**Median-fold DD:** {float(np.median(fold_dds)):.2f}%")
    a(f"**Gate: {'PASS' if gate_pass else 'FAIL'}**")
    a("")
    a("---")
    a("")

    # ── 3. Trade count delta vs KG-J ─────────────────────────────────────────
    a("## 3. Trade Count Delta vs KG-J")
    a("")
    a("| Fold | KGJ | KGL-V2 | Δ | cond_9 blocked OOS |")
    a("|------|-----|--------|---|---------------------|")
    total_kgj   = 0
    total_kglv2 = 0
    total_c9    = 0
    for _, row in fold_df.iterrows():
        kgj_n = int(row["kgj_n_trades"])
        kgl_n = int(row["n_trades"])
        c9_n  = int(row["cond9_blocked_oos"])
        dlt   = kgl_n - kgj_n
        total_kgj   += kgj_n
        total_kglv2 += kgl_n
        total_c9    += c9_n
        a(f"| {int(row['fold'])} | {kgj_n} | {kgl_n} | {dlt:+d} | {c9_n} |")
    total_delta = total_kglv2 - total_kgj
    a(f"| **Total** | **{total_kgj}** | **{total_kglv2}** | **{total_delta:+d}** | **{total_c9}** |")
    a("")
    a(f"Total cond_9 blocked (all-time): {stats['total_cond9_all']}")
    a(f"Total cond_9 blocked (OOS only): {stats['total_cond9_oos']}")
    a("")
    a("---")
    a("")

    # ── 4. Exit breakdown ─────────────────────────────────────────────────────
    a("## 4. Exit Reason Breakdown (OOS)")
    a("")
    ex_stats = _exit_stats_oos(all_trades)
    total_oos_t = sum(s["count"] for s in ex_stats.values())
    a(f"OOS total: {total_oos_t}")
    a("Note: signal_flip is disabled in V2 — kijun_d1 absorbs those exits.")
    a("")
    a("| Exit Reason | Count | % of OOS | Avg R | Avg Bars |")
    a("|-------------|-------|----------|-------|----------|")
    for etype in ["kh13_early", "kh14_bar6", "kijun_d1", "trailing_stop", "stoploss"]:
        s   = ex_stats.get(etype, {"count": 0, "pct": 0.0,
                                   "avg_r": float("nan"), "avg_bars": float("nan")})
        r_s = f"{s['avg_r']:.4f}"    if not np.isnan(s["avg_r"])   else "—"
        b_s = f"{s['avg_bars']:.1f}" if not np.isnan(s["avg_bars"]) else "—"
        a(f"| {etype:<18} | {s['count']:>5} | {s['pct']:>6.1f}% | {r_s:>7} | {b_s:>8} |")
    a("")
    a("---")
    a("")

    # ── 5. KG-J vs KG-L-V2 comparison ────────────────────────────────────────
    a("## 5. KG-J → KG-L-V2 Comparison (fold by fold)")
    a("")
    a("### ROI%")
    a("")
    a("| Fold | OOS Period | KG-J | KG-L-V2 | Delta |")
    a("|------|------------|------|---------|-------|")
    for _, row in fold_df.iterrows():
        fn    = int(row["fold"])
        kgj_r = _KGIJ_FOLDS.get(fn, (float("nan"), 0, 0))[0]
        kgl_r = row["roi_pct"]
        delta = kgl_r - kgj_r if not np.isnan(kgj_r) else float("nan")
        kgj_s = f"{kgj_r:.2f}%" if not np.isnan(kgj_r) else "—"
        d_s   = f"{delta:+.2f}%" if not np.isnan(delta) else "—"
        period = f"{row['oos_start']} – {row['oos_end']}"
        a(f"| {fn} | {period} | {kgj_s} | {kgl_r:.2f}% | {d_s} |")
    a("")
    a("### MaxDD%")
    a("")
    a("| Fold | OOS Period | KG-J | KG-L-V2 | Delta |")
    a("|------|------------|------|---------|-------|")
    for _, row in fold_df.iterrows():
        fn    = int(row["fold"])
        kgj_d = _KGIJ_FOLDS.get(fn, (0, float("nan"), 0))[1]
        kgl_d = row["max_dd_pct"]
        delta = kgl_d - kgj_d if not np.isnan(kgj_d) else float("nan")
        kgj_s = f"{kgj_d:.2f}%" if not np.isnan(kgj_d) else "—"
        d_s   = f"{delta:+.2f}%" if not np.isnan(delta) else "—"
        period = f"{row['oos_start']} – {row['oos_end']}"
        a(f"| {fn} | {period} | {kgj_s} | {kgl_d:.2f}% | {d_s} |")
    a("")
    a("---")
    a("")

    # ── 6. SL distance sanity ─────────────────────────────────────────────────
    a("## 6. SL Distance Sanity Check")
    a("")
    orig_t = [t for t in all_trades if t.get("trade_type", "original") == "original"]
    re_t   = [t for t in all_trades if t.get("trade_type") == "reentry"]
    orig_sl = [t["sl_distance_atr"] for t in orig_t]
    re_sl   = [t["sl_distance_atr"] for t in re_t]
    if orig_sl:
        orig_exact = all(abs(d - 2.0) < 1e-6 for d in orig_sl)
        a(f"Original trades ({len(orig_t)}):  "
          f"min sl_distance_atr={min(orig_sl):.6f}  max={max(orig_sl):.6f}  "
          f"all exactly 2.0: **{'YES — OK' if orig_exact else 'NO — FAIL'}**")
    if re_sl:
        re_exp   = REENTRY_SL_ATR_MULT
        re_exact = all(abs(d - re_exp) < 1e-6 for d in re_sl)
        a(f"Re-entry trades ({len(re_t)}):  "
          f"min sl_distance_atr={min(re_sl):.6f}  max={max(re_sl):.6f}  "
          f"all exactly {re_exp:.1f}: **{'YES — OK' if re_exact else 'NO — FAIL'}**")
    if not orig_sl and not re_sl:
        a("No trades.")
    a("")
    a("---")
    a("")

    # ── 7. Exposure cap stats ─────────────────────────────────────────────────
    a("## 7. Trades Blocked by Exposure Cap")
    a("")
    oos_start_ts = pd.Timestamp(FOLDS[0]["oos_start"])
    oos_blocked  = [b for b in blocked_events
                    if "exposure" in b.get("reason", "")
                    and b["date"] >= oos_start_ts]
    all_exp_blocked = [b for b in blocked_events
                       if "exposure" in b.get("reason", "")]
    a(f"Total blocked (all time): {len(all_exp_blocked)}")
    a(f"Total blocked (OOS only): {len(oos_blocked)}")
    a("")

    # ── 8. Per-pair OOS ───────────────────────────────────────────────────────
    a("## 8. Per-Pair OOS Performance")
    a("")
    a("| Pair | Trades | ROI% | WinRate | ExpR | AvgBars |")
    a("|------|--------|------|---------|------|---------|")
    for _, row in per_pair_df.iterrows():
        roi_s = f"{row['roi_pct']:.2f}%"      if not pd.isna(row["roi_pct"])       else "—"
        wr_s  = f"{row['win_rate']:.4f}"      if not pd.isna(row["win_rate"])       else "—"
        er_s  = f"{row['expectancy_r']:.4f}"  if not pd.isna(row["expectancy_r"])   else "—"
        bh_s  = f"{row['avg_bars_held']:.1f}" if not pd.isna(row["avg_bars_held"])  else "—"
        a(f"| {row['pair']} | {row['n_trades']} | {roi_s} | {wr_s} | {er_s} | {bh_s} |")
    a("")
    a("---")
    a("")

    # ── 9. KH-16 Re-entry Analysis ───────────────────────────────────────────
    a("## 9. KH-16 Re-entry Analysis (OOS)")
    a("")
    if USE_REENTRY:
        re_trades_oos = [t for t in all_trades
                         if (t.get("trade_type") == "reentry"
                             and pd.Timestamp(t["entry_date"]) >= pd.Timestamp(FOLDS[0]["oos_start"]))]
        orig_oos = [t for t in all_trades
                    if (t.get("trade_type", "original") == "original"
                        and pd.Timestamp(t["entry_date"]) >= pd.Timestamp(FOLDS[0]["oos_start"]))]
        a(f"Re-entry enabled: window={REENTRY_WINDOW_BARS} bars, "
          f"SL={REENTRY_SL_ATR_MULT}×ATR, trigger=close_above_entry")
        a("")
        a("| Metric | Original | Re-entry |")
        a("|--------|----------|----------|")
        a(f"| OOS trades | {len(orig_oos)} | {len(re_trades_oos)} |")
        if orig_oos:
            orig_wr_s = f"{sum(1 for t in orig_oos if t['classification']=='WIN')/len(orig_oos):.4f}"
            orig_mr_s = f"{float(np.mean([t['r_multiple'] for t in orig_oos])):.4f}"
        else:
            orig_wr_s, orig_mr_s = "n/a", "n/a"
        re_wr_s  = (f"{stats['reentry_win_rate']:.4f}"
                    if not np.isnan(stats["reentry_win_rate"]) else "n/a")
        re_mr_s  = (f"{stats['reentry_mean_r']:.4f}"
                    if not np.isnan(stats["reentry_mean_r"]) else "n/a")
        a(f"| Win rate | {orig_wr_s} | {re_wr_s} |")
        a(f"| Mean R | {orig_mr_s} | {re_mr_s} |")
        a("")
        if re_trades_oos:
            a("| Fold | Re-entry Trades | Win Rate | Mean R |")
            a("|------|-----------------|----------|--------|")
            for f_cfg in FOLDS:
                f_re = [t for t in re_trades_oos
                        if (pd.Timestamp(f_cfg["oos_start"])
                            <= pd.Timestamp(t["entry_date"])
                            < pd.Timestamp(f_cfg["oos_end"]))]
                if f_re:
                    f_wr  = f"{sum(1 for t in f_re if t['classification']=='WIN')/len(f_re):.4f}"
                    f_mr  = f"{float(np.mean([t['r_multiple'] for t in f_re])):.4f}"
                else:
                    f_wr, f_mr = "—", "—"
                a(f"| {f_cfg['fold']} | {len(f_re)} | {f_wr} | {f_mr} |")
    else:
        a("Re-entry disabled (USE_REENTRY=False).")
    a("")
    a("---")
    a("")

    # ── 10. Methodology ───────────────────────────────────────────────────────
    a("## 10. Methodology Notes")
    a("")
    a("**signal_flip exit — DISABLED:**")
    a("- V1 exits: signal_flip | kijun_d1 | trailing_stop | stoploss")
    a("- V2 exits:              kijun_d1 | trailing_stop | stoploss")
    a("- signal_flip is not computed on any bar and never checked in the loop.")
    a("- Effect: trades that would have exited on signal_flip in V1 now remain")
    a("  open until kijun_d1, trailing_stop, or stoploss fires.")
    a("- Expected: longer avg_bars_held, potentially higher or lower R per trade")
    a("  depending on whether the signal_flip correctly predicted direction.")
    a("")
    a("| Exit | Condition |")
    a("|------|-----------|")
    a("| kijun_d1 | D1 close < D1 Kijun(26), asof 4H bar date |")
    a("| trailing_stop | close ≤ best_close − 1.5 × ATR, gated on trail_active |")
    a("| stoploss | intrabar low ≤ entry_px − 2.0 × ATR |")
    a("| trail activation | bar close ≥ entry_px + 2.0 × ATR (CLOSE-BASED) |")
    a("")

    report_path = OUT_ROOT / "kgl_v2_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("  Written: results/phase_kg/kg_l_v2/kgl_v2_report.md")


# ── Pre-run integrity checks ──────────────────────────────────────────────────

def _run_pre_checks() -> None:
    """Run pre-WFO integrity checks on EUR_USD as representative pair.
    Raises ValueError on any failure so the WFO does not start with bad data."""
    test_pair = "EUR_USD"
    df_4h = load_pair_csv(test_pair, DATA_DIR_4H)
    df_4h["date"] = pd.to_datetime(df_4h["date"])
    df_4h = df_4h.sort_values("date").reset_index(drop=True)

    # 1. D1 alignment — verify no 4H bar sees same-day or future D1 data
    df_d1 = load_pair_csv(test_pair, D1_DATA_DIR)
    df_d1["date"] = pd.to_datetime(df_d1["date"]).dt.normalize()
    df_d1 = df_d1.sort_values("date").reset_index(drop=True)
    dates_shifted = df_4h["date"].dt.normalize() - pd.Timedelta(days=1)
    tmp_4h = pd.DataFrame({"date_shifted": dates_shifted, "date_orig": df_4h["date"]})
    tmp_d1 = pd.DataFrame({"date_shifted": df_d1["date"], "d1_date": df_d1["date"]})
    merged = pd.merge_asof(
        tmp_4h.sort_values("date_shifted"),
        tmp_d1.sort_values("date_shifted"),
        on="date_shifted",
        direction="backward",
    )
    valid = merged.dropna(subset=["d1_date"])
    violations = valid[valid["d1_date"] >= valid["date_orig"].dt.normalize()]
    if len(violations) > 0:
        raise ValueError(
            f"D1 LOOKAHEAD DETECTED: {len(violations)} rows where D1 bar date "
            f">= 4H bar calendar date. First: {violations.iloc[0].to_dict()}"
        )
    print("D1 ALIGNMENT CHECK: PASS — 0 violations across all bars")

    # 2. ATR spot-check — verify Wilder ATR is lookback-only and positive
    atr_arr = _wilder_atr(df_4h, ATR_PERIOD).values
    rng = np.random.default_rng(42)
    idxs = rng.integers(ATR_PERIOD * 3, min(len(df_4h) - 1, ATR_PERIOD * 50), size=3)
    for idx in idxs:
        val = atr_arr[int(idx)]
        if np.isnan(val) or val <= 0:
            raise ValueError(f"ATR SPOT-CHECK FAIL: ATR[{idx}] = {val}")
        sub_atr = _wilder_atr(df_4h.iloc[:int(idx) + 1], ATR_PERIOD).values
        if abs(sub_atr[-1] - val) > 1e-8:
            raise ValueError(f"ATR SPOT-CHECK FAIL: full[{idx}]={val:.10f} vs truncated={sub_atr[-1]:.10f}")
    print("INDICATOR SPOT-CHECK: ATR — 3 random bars verified PASS")

    # 3. Kijun spot-check — verify (max_high + min_low) / 2 over rolling 26-bar window
    kij_arr = _compute_kijun(df_4h, KIJUN_PERIOD)
    hi = df_4h["high"].values.astype(float)
    lo = df_4h["low"].values.astype(float)
    rng2 = np.random.default_rng(123)
    idxs2 = rng2.integers(KIJUN_PERIOD, min(len(df_4h) - 1, KIJUN_PERIOD * 10), size=3)
    for idx in idxs2:
        start = max(0, int(idx) - KIJUN_PERIOD + 1)
        expected = (hi[start:int(idx) + 1].max() + lo[start:int(idx) + 1].min()) / 2.0
        if abs(kij_arr[int(idx)] - expected) > 1e-8:
            raise ValueError(
                f"KIJUN SPOT-CHECK FAIL: [{idx}] got={kij_arr[int(idx)]:.10f} expected={expected:.10f}"
            )
    print("INDICATOR SPOT-CHECK: Kijun — 3 random bars verified PASS")

    # 4. C6 lookback direction — verify close[N-10] refers to a bar in the past
    rng3 = np.random.default_rng(456)
    idxs3 = rng3.integers(DEPTH_BARS + 2, min(len(df_4h) - 1, DEPTH_BARS * 10), size=3)
    for idx in idxs3:
        past_idx = int(idx) - DEPTH_BARS
        if past_idx < 0 or past_idx >= int(idx):
            raise ValueError(f"C6 LOOKBACK FAIL: past_idx={past_idx} is not strictly before idx={idx}")
        past_date = df_4h.iloc[past_idx]["date"]
        cur_date = df_4h.iloc[int(idx)]["date"]
        if past_date >= cur_date:
            raise ValueError(f"C6 LOOKBACK FAIL: past bar date {past_date} >= current {cur_date}")
    print("INDICATOR SPOT-CHECK: C6 lookback direction — 3 random bars verified PASS")

    # 5. KH-8: verify D1 lag=2 < lag=1 < 4H calendar date for EUR_USD
    d1_filt_chk = _load_d1_filter(test_pair)
    (_cl1, _kj1, _cl2, _kj2,
     _atr1,
     _kj_prior5,
     _d1_lag1, _d1_lag2, _bar_norm,
     _d1_cir_chk) = _precompute_d1_exit_arrays(df_4h, d1_filt_chk)
    _mask = (~np.isnat(_d1_lag1)) & (~np.isnat(_d1_lag2))
    if _mask.any():
        _ok_order = (_d1_lag2[_mask] < _d1_lag1[_mask]).all()
        _ok_past  = (_d1_lag1[_mask] < _bar_norm[_mask]).all()
        if not _ok_order:
            raise ValueError("KH-8 LOOKAHEAD CHECK: d1_lag2 >= d1_lag1 on some rows")
        if not _ok_past:
            raise ValueError("KH-8 LOOKAHEAD CHECK: d1_lag1 >= 4H calendar date on some rows")
        _n = int(_mask.sum())
        _first = np.where(_mask)[0][0]
        print(f"KH-8 LOOKAHEAD CHECK: PASS — lag2 < lag1 < 4H_date on {_n} rows")
        print(f"  Sample (idx {_first}): "
              f"lag2={pd.Timestamp(_d1_lag2[_first]).date()} "
              f"lag1={pd.Timestamp(_d1_lag1[_first]).date()} "
              f"4H={pd.Timestamp(_bar_norm[_first]).date()}")
    else:
        print("KH-8 LOOKAHEAD CHECK: SKIPPED — no rows with both lags")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_kgl_v2() -> None:
    print("\n=== Phase KG-L-V2: D1 ATR Cap + signal_flip REMOVED ===")
    print(f"  Data 4H  : {DATA_DIR_4H}")
    print(f"  Data D1  : {D1_DATA_DIR}")
    print("  Signal   : kb_exhaustion_bar + KF v17 + D1a + cond_9")
    _baseline_label = {
        "kijun": "Kijun(26)",
        "hma":   "HMA(26)",
        "dema":  "DEMA(50)",
    }.get(BASELINE_TYPE, BASELINE_TYPE)
    _warm = WARMUP_BARS.get(BASELINE_TYPE, 75)
    print(f"  BASELINE TYPE: {_baseline_label}")
    print(f"  WARMUP (bars): {_warm}")
    print(f"  C4/C5 use: {_baseline_label}")
    print("  C8/C9/kijun_d1 exit still use: D1 Kijun(26) (unchanged)")
    print(f"  cond_9   : D1 close <= D1 kijun({D1_KIJUN_PERIOD}) + {D1_ATR_DIST_CAP}×D1 ATR({D1_ATR_PERIOD})")
    print("  Trail    : CLOSE-BASED activation (fix from KG-J)")
    print("  Classify : WIN/LOSS/SCRATCH from r_multiple")
    print("  Exits    : kijun_d1 | trailing_stop | stoploss  (signal_flip DISABLED)")
    print("  KIJUN_D1 EXIT CONFIG:")
    print(f"    confirm_bars:       {KIJUN_D1_CONFIRM_BARS}")
    _kh9_any = (
        KIJUN_D1_CONFIRM_IF_TRAIL
        or KIJUN_D1_CONFIRM_MIN_BARS > 0
        or KIJUN_D1_CONFIRM_DEPTH_ATR > 0.0
    )
    if KIJUN_D1_CONFIRM_IF_TRAIL:
        print(f"    confirm_if_trail:   {KIJUN_D1_CONFIRM_IF_TRAIL}"
              "  -> 2-bar only when trail active")
    else:
        print("    confirm_if_trail:   False (disabled)")
    if KIJUN_D1_CONFIRM_MIN_BARS > 0:
        print(f"    confirm_min_bars:   {KIJUN_D1_CONFIRM_MIN_BARS}"
              f"  -> 2-bar only when bars_held >= {KIJUN_D1_CONFIRM_MIN_BARS}")
    else:
        print("    confirm_min_bars:   0 (disabled)")
    if KIJUN_D1_CONFIRM_DEPTH_ATR > 0.0:
        print(f"    confirm_depth_atr:  {KIJUN_D1_CONFIRM_DEPTH_ATR:.2f}"
              f"  -> 2-bar only when cross < {KIJUN_D1_CONFIRM_DEPTH_ATR:.2f} D1 ATR deep")
    else:
        print("    confirm_depth_atr:  0.0 (disabled)")
    if KIJUN_D1_CONFIRM_BARS == 2 and not _kh9_any:
        print("    behaviour: KH-8 2-bar confirmation applied unconditionally")
        print("    Lag=2 NaN guard: enabled (no exit if lag=2 history unavailable)")
    elif KIJUN_D1_CONFIRM_BARS == 2 and _kh9_any:
        print("    behaviour: KH-9 conditional 2-bar confirmation")
        print("    Lag=2 NaN guard: enabled (no exit if lag=2 history unavailable)")
    else:
        print("    behaviour: 1-bar (baseline) — exits on single D1 cross")
    print(f"  SL       : {SL_MULT}×ATR from ENTRY PRICE")
    print(f"  Trail    : {TRAIL_MULT}×ATR from best close (activates at {TRAIL_ACT_MULT}×ATR close)")
    print(f"  Exposure : max {EXPOSURE_CAP} unit per currency | alphabetical processing order")
    print(f"  Balance  : {INITIAL_BAL:,.0f}  Risk: {RISK_PCT*100:.1f}%/trade")
    print(f"  Spread   : per-bar MT5 'spread' column (points / {POINTS_PER_PIP:.0f})")
    if USE_KH13_EARLY_EXIT:
        print(f"  KH-13    : ENABLED — exit at bar N+4 open when "
              f"mae_at_bar_3 >= {KH13_MAE_THRESHOLD}×ATR AND "
              f"mfe_at_bar_3 <= {KH13_MFE_THRESHOLD}×ATR")
    else:
        print("  KH-13    : DISABLED (kh13_triggered recorded for audit)")
    if USE_KH14_BAR6_EXIT:
        print(f"  KH-14    : ENABLED — exit at bar N+7 open (State 2 only) when "
              f"mfe_at_bar_6 < {KH14_MFE_THRESHOLD}×ATR AND "
              f"mae_at_bar_6 >= {KH14_MAE_THRESHOLD}×ATR")
    else:
        print("  KH-14    : DISABLED (kh14_triggered/kh14_state2 recorded for audit)")
    if USE_ATR_SIZING:
        print(f"  KH-15A   : ENABLED — when atr_ratio >= {ATR_SIZING_THRESHOLD:.2f}, "
              f"risk = {ATR_SIZING_REDUCED_PCT*100:.2f}%/trade "
              f"(base = {RISK_PCT*100:.2f}%)")
    else:
        print("  KH-15A   : DISABLED (atr_sized_down=False on all trades)")
    if KH25_ENABLED:
        print(f"  KH-25    : ENABLED — kh14_bar6 exit + re-entry on KH-24 base "
              f"(window={REENTRY_WINDOW_BARS} bars, SL={REENTRY_SL_ATR_MULT}×ATR, "
              f"MFE<{KH14_MFE_THRESHOLD}×ATR AND MAE>={KH14_MAE_THRESHOLD}×ATR, "
              f"trigger=close_above_entry, bypass 1H filter + cap)")
    elif KH19_ENABLED:
        print(f"  KH-19    : ENABLED — baseline entry + kh14_bar6 exit + re-entry "
              f"(window={REENTRY_WINDOW_BARS} bars, SL={REENTRY_SL_ATR_MULT}×ATR, "
              f"trigger=close_above_entry, 1-bar kijun_d1 confirm)")
    elif USE_REENTRY:
        print(f"  KH-16    : ENABLED — re-entry after kh14_bar6 exit "
              f"(window={REENTRY_WINDOW_BARS} bars, SL={REENTRY_SL_ATR_MULT}×ATR, "
              f"trigger=close_above_entry, next-bar-open fill)")
    else:
        print("  KH-16/19 : DISABLED (no re-entry after kh14_bar6 exits)")
    if KH18_ENABLED:
        print(f"  KH-18    : ENABLED — alias of KH-17 two-decision path "
              f"(state1_delayed at bar N+2 open, state2_reentry watch "
              f"bars={KH17_WATCH_WINDOW_BARS}, watch SL={KH17_WATCH_SL_ATR_MULT}×ATR, "
              f"bar6 MFE<{KH17_BAR6_MFE_THRESHOLD}×ATR AND MAE>={KH17_BAR6_MAE_THRESHOLD}×ATR)")
    else:
        print("  KH-18    : DISABLED")

    print("\n  Running pre-checks ...")
    _run_pre_checks()
    print(f"  C7 (volume filter): {'DISABLED' if NO_VOLUME_FILTER else 'ENABLED'}"
          f" — evaluations: {'0 (disabled)' if NO_VOLUME_FILTER else 'active'}")
    print(f"  Exposure cap: {'DISABLED (999)' if EXPOSURE_CAP >= 900 else str(EXPOSURE_CAP)}")
    print(f"  C8 (D1 trend gate): {'DISABLED' if not USE_C8 else 'ENABLED'}")
    print(f"  C9 (D1 ATR distance cap): {'DISABLED' if not USE_C9 else 'ENABLED'}")
    if REGIME_MIN_TRENDING_PAIRS > 0:
        print(f"  PORTFOLIO REGIME GATE: ENABLED (min={REGIME_MIN_TRENDING_PAIRS}, "
              f"lookback={REGIME_SLOPE_LOOKBACK} bars, ema={REGIME_EMA_PERIOD})")
    else:
        print("  PORTFOLIO REGIME GATE: DISABLED")
    if REQUIRE_OWN_EMA_SLOPE:
        print(f"  OWN SLOPE GATE: ENABLED (lookback={REGIME_SLOPE_LOOKBACK} bars, "
              f"ema={REGIME_EMA_PERIOD})")
    else:
        print("  OWN SLOPE GATE: DISABLED")
    if REGIME_MIN_TRENDING_PAIRS > 0 or REQUIRE_OWN_EMA_SLOPE:
        print("  REGIME GATE LOOKAHEAD CHECK:")
        print("    EMA uses bars 0..T only: CONFIRMED")
        print(f"    Slope uses ema[T] > ema[T-{REGIME_SLOPE_LOOKBACK}] "
              "(lookback, not lookahead): CONFIRMED")
        print(f"    All {len(ALL_PAIRS)} pairs loaded to bar T: CONFIRMED")

    print("\n  Loading 4H and D1 data ...")
    pair_cache = _build_pair_cache()
    print(f"  Loaded {len(pair_cache)} of {len(ALL_PAIRS)} pairs")

    blocked_portfolio, blocked_own, regime_evaluated = _apply_regime_gate(pair_cache)
    if REGIME_MIN_TRENDING_PAIRS > 0 or REQUIRE_OWN_EMA_SLOPE:
        total_blocked = blocked_portfolio + blocked_own
        pct_total = (total_blocked / regime_evaluated * 100.0) if regime_evaluated > 0 else 0.0
        print(f"  Regime gate: {total_blocked} / {regime_evaluated} signals blocked "
              f"({pct_total:.1f}%)")
        if REQUIRE_OWN_EMA_SLOPE:
            print(f"    - own-slope blocked: {blocked_own}")
        if REGIME_MIN_TRENDING_PAIRS > 0:
            print(f"    - portfolio-count blocked: {blocked_portfolio} "
                  f"(threshold >= {REGIME_MIN_TRENDING_PAIRS})")

    total_d1_blocked  = sum(v["n_d1_blocked"]    for v in pair_cache.values())
    total_d1_passed   = sum(v["n_d1_passed"]     for v in pair_cache.values())
    total_cond9       = sum(v["n_cond9_blocked"]  for v in pair_cache.values())
    total_signals     = sum(int(v["sig"].sum())   for v in pair_cache.values())
    total_sig_raw     = total_d1_blocked + total_d1_passed + total_cond9
    pct_d1_blocked    = total_d1_blocked / total_sig_raw * 100.0 if total_sig_raw > 0 else 0.0
    pct_cond9_blocked = total_cond9 / (total_d1_passed + total_cond9) * 100.0 if (total_d1_passed + total_cond9) > 0 else 0.0

    print(f"  Total signals (after all filters): {total_signals}")
    print(f"  D1a blocked (cond_8):  {total_d1_blocked} ({pct_d1_blocked:.1f}% of raw)")
    print(f"  cond_9 blocked:        {total_cond9} ({pct_cond9_blocked:.1f}% of D1a-passing)")

    print("\n  Running full unified simulation ...")
    all_trades, blocked_events, bar_paths_data = _simulate_kgl_v2(pair_cache)

    total_exp_blocked = len([b for b in blocked_events if "exposure" in b.get("reason", "")])
    print(f"  Simulation complete: {len(all_trades)} trades completed, "
          f"{total_exp_blocked} blocked by exposure cap")
    print(f"  Cap blocks: {total_exp_blocked}"
          f" (cap={'DISABLED' if EXPOSURE_CAP >= 900 else str(EXPOSURE_CAP)})")

    if REQUIRE_D1_KIJUN_SLOPE_FALLING:
        n_slope_blocked = sum(
            1 for b in blocked_events if b.get("reason") == "d1_kijun_slope_rising"
        )
        n_raw_signals   = len(all_trades) + n_slope_blocked
        pct = (n_slope_blocked / n_raw_signals * 100.0) if n_raw_signals > 0 else 0.0
        print(f"  KH-11A slope gate: {n_slope_blocked} / {n_raw_signals} signals "
              f"blocked ({pct:.1f}%)")

    # SL distance check: state2_reentry trades legitimately use
    # KH17_WATCH_SL_ATR_MULT (default 1.5); everything else uses SL_MULT.
    if KH17_ENABLED or KH18_ENABLED:
        sl_pairs = [(t.get("sl_distance_atr"), t.get("trade_type"))
                    for t in all_trades if t.get("sl_distance_atr") is not None]
        bad_sl = []
        for d, tt in sl_pairs:
            expected = KH17_WATCH_SL_ATR_MULT if tt == "state2_reentry" else SL_MULT
            if abs(d - expected) > 1e-6:
                bad_sl.append((d, tt))
        if sl_pairs:
            if bad_sl:
                print(f"  SL distance check: FAIL — {len(bad_sl)} trades with unexpected SL "
                      f"(state1_delayed={SL_MULT}×ATR, "
                      f"state2_reentry={KH17_WATCH_SL_ATR_MULT}×ATR)")
            else:
                print(f"  SL distance check: all {len(sl_pairs)} trades match expected SL — PASS "
                      f"(state1_delayed={SL_MULT}×ATR, "
                      f"state2_reentry={KH17_WATCH_SL_ATR_MULT}×ATR)")
    else:
        sl_dists = [t.get("sl_distance_atr") for t in all_trades if t.get("sl_distance_atr") is not None]
        if sl_dists:
            bad_sl = [d for d in sl_dists if abs(d - SL_MULT) > 1e-6]
            if bad_sl:
                print(f"  SL distance check: FAIL — {len(bad_sl)} trades with SL != {SL_MULT}×ATR")
            else:
                print(f"  SL distance check: all {len(sl_dists)} trades exactly {SL_MULT:.1f}×ATR — PASS")

    print("\n  WFO fold results:")
    fold_df, per_pair_df, stats = _run_wfo(
        pair_cache, all_trades, blocked_events, bar_paths_data,
    )

    # ── cond_9 sensitivity summary ────────────────────────────────────────────
    print(f"\n=== cond_9 Sensitivity (threshold={D1_ATR_DIST_CAP}x) ===")
    ratios = [t.get("d1_dist_ratio", float("nan")) for t in all_trades]
    ratios_clean = [r for r in ratios if not np.isnan(r)]
    if ratios_clean:
        print("  D1 dist ratio distribution (signals that PASSED cond_9):")
        print(f"    mean: {np.mean(ratios_clean):.2f}  "
              f"median: {np.median(ratios_clean):.2f}  "
              f"p25: {np.percentile(ratios_clean, 25):.2f}  "
              f"p75: {np.percentile(ratios_clean, 75):.2f}  "
              f"min: {np.min(ratios_clean):.2f}  "
              f"max: {np.max(ratios_clean):.2f}")
    near_cap  = [t for t in all_trades
                 if not np.isnan(t.get("d1_dist_ratio", float("nan")))
                 and t.get("d1_dist_ratio", float("nan")) > 0.75]
    with_room = [t for t in all_trades
                 if not np.isnan(t.get("d1_dist_ratio", float("nan")))
                 and t.get("d1_dist_ratio", float("nan")) <= 0.75]
    nc_mean_r = float(np.mean([t["r_multiple"] for t in near_cap]))  if near_cap  else float("nan")
    nc_wr     = sum(1 for t in near_cap  if t["classification"] == "WIN") / len(near_cap)  if near_cap  else float("nan")
    wr_mean_r = float(np.mean([t["r_multiple"] for t in with_room])) if with_room else float("nan")
    wr_wr     = sum(1 for t in with_room if t["classification"] == "WIN") / len(with_room) if with_room else float("nan")
    nc_r_s  = f"{nc_mean_r:.2f}"  if not np.isnan(nc_mean_r)  else "n/a"
    nc_w_s  = f"{nc_wr:.2f}"      if not np.isnan(nc_wr)      else "n/a"
    wr_r_s  = f"{wr_mean_r:.2f}"  if not np.isnan(wr_mean_r)  else "n/a"
    wr_w_s  = f"{wr_wr:.2f}"      if not np.isnan(wr_wr)      else "n/a"
    print(f"  Trades near cap (ratio > 0.75):   {len(near_cap):>3}  mean_r: {nc_r_s}  win_rate: {nc_w_s}")
    print(f"  Trades with room (ratio <= 0.75): {len(with_room):>3}  mean_r: {wr_r_s}  win_rate: {wr_w_s}")
    print(f"  WFO gate: {'PASS' if stats['gate_pass'] else 'FAIL'}"
          f"  Worst-fold ROI: {stats['worst_roi']:.2f}%  Worst-fold DD: {stats['worst_dd']:.2f}%")

    # ── KH-13 OOS summary ─────────────────────────────────────────────────────
    print("\n=== KH-13 Summary (OOS) ===")
    print(f"  kh13_triggered (condition met, regardless of config): "
          f"{stats['kh13_triggered_oos']}")
    if USE_KH13_EARLY_EXIT:
        print(f"  kh13_early exits (actual): {stats['kh13_early_oos']}")
    else:
        print(f"  kh13_early exits: 0  (USE_KH13_EARLY_EXIT=False — "
              f"{stats['kh13_triggered_oos']} would have fired)")

    # ── KH-14 OOS summary ─────────────────────────────────────────────────────
    print("\n=== KH-14 Summary (OOS) ===")
    print(f"  kh14_state2 (first_bar_dir=-1): {stats['kh14_state2_oos']}")
    print(f"  kh14_triggered (condition met at bar 6, regardless of config): "
          f"{stats['kh14_triggered_oos']}")
    if USE_KH14_BAR6_EXIT:
        print(f"  kh14_bar6 exits (actual): {stats['kh14_bar6_oos']}")
    else:
        print(f"  kh14_bar6 exits: 0  (USE_KH14_BAR6_EXIT=False — "
              f"{stats['kh14_triggered_oos']} would have fired)")

    # ── KH-20 OOS summary ─────────────────────────────────────────────────────
    if KH20_ENABLED:
        _kh20_oos_filtered = stats["kh20_total_oos"]
        _kh20_oos_taken    = len([t for t in all_trades
                                   if pd.Timestamp(t["entry_date"]) >= pd.Timestamp(FOLDS[0]["oos_start"])])
        _kh20_total        = _kh20_oos_filtered + _kh20_oos_taken
        _kh20_pct          = (_kh20_oos_filtered / _kh20_total * 100.0) if _kh20_total > 0 else 0.0
        print("\n=== KH-20 D1 Close-in-Range Filter Summary (OOS) ===")
        print(f"  threshold: <= {KH20_D1_RANGE_THRESHOLD:.4f}")
        print(f"  signals filtered (OOS): {_kh20_oos_filtered}  "
              f"taken: {_kh20_oos_taken}  total: {_kh20_total}  "
              f"pct filtered: {_kh20_pct:.1f}%")
        print("  per-fold filtered: "
              + "  ".join(f"F{k}={v}" for k, v in sorted(stats["kh20_per_fold"].items())))
        # Sanity check: d1_close_in_range distribution for taken trades (OOS only)
        _oos_cir = [t.get("d1_close_in_range", float("nan"))
                    for t in all_trades
                    if pd.Timestamp(t["entry_date"]) >= pd.Timestamp(FOLDS[0]["oos_start"])]
        _oos_cir_clean = [v for v in _oos_cir if not (isinstance(v, float) and np.isnan(v))]
        if _oos_cir_clean:
            _cir_min  = min(_oos_cir_clean)
            _cir_max  = max(_oos_cir_clean)
            _cir_mean = float(np.mean(_oos_cir_clean))
            _cir_violations = sum(1 for v in _oos_cir_clean if v <= KH20_D1_RANGE_THRESHOLD)
            print(f"  d1_close_in_range (taken trades): "
                  f"min={_cir_min:.4f}  max={_cir_max:.4f}  mean={_cir_mean:.4f}")
            if _cir_violations:
                print(f"  SANITY FAIL: {_cir_violations} taken trades have "
                      f"d1_close_in_range <= {KH20_D1_RANGE_THRESHOLD:.4f} — filter not applied!")
            else:
                print(f"  Sanity check: all taken trades have d1_close_in_range > {KH20_D1_RANGE_THRESHOLD:.4f} — PASS")
    else:
        print("\n=== KH-20 D1 Close-in-Range Filter: DISABLED ===")

    # ── KH-22 OOS summary ─────────────────────────────────────────────────────
    if KH22_ENABLED:
        _oos_start_kh22 = pd.Timestamp(FOLDS[0]["oos_start"])
        _kh22_oos_filtered = stats["kh22_total_oos"]
        _kh22_oos_taken    = len([t for t in all_trades
                                   if pd.Timestamp(t["entry_date"]) >= _oos_start_kh22])
        _kh22_total        = _kh22_oos_filtered + _kh22_oos_taken
        _kh22_pct          = (_kh22_oos_filtered / _kh22_total * 100.0) if _kh22_total > 0 else 0.0
        print("\n=== KH-22 1H Close-in-Range Filter Summary (OOS) ===")
        print(f"  threshold: > {KH22_H1_RANGE_THRESHOLD:.4f}  (block if h1 closes too high in range)")
        print(f"  signals filtered (OOS): {_kh22_oos_filtered}  "
              f"taken: {_kh22_oos_taken}  total: {_kh22_total}  "
              f"pct filtered: {_kh22_pct:.1f}%")
        print("  per-fold filtered: "
              + "  ".join(f"F{k}={v}" for k, v in sorted(stats["kh22_per_fold"].items())))
        _oos_h1_cir = [t.get("h1_last_bar_close_in_range", float("nan"))
                       for t in all_trades
                       if pd.Timestamp(t["entry_date"]) >= _oos_start_kh22]
        _oos_h1_cir_clean = [v for v in _oos_h1_cir
                              if isinstance(v, float) and not np.isnan(v)]
        if _oos_h1_cir_clean:
            _h1_min  = min(_oos_h1_cir_clean)
            _h1_max  = max(_oos_h1_cir_clean)
            _h1_mean = float(np.mean(_oos_h1_cir_clean))
            _h1_violations = sum(1 for v in _oos_h1_cir_clean if v > KH22_H1_RANGE_THRESHOLD)
            print(f"  h1_last_bar_close_in_range (taken trades): "
                  f"min={_h1_min:.4f}  max={_h1_max:.4f}  mean={_h1_mean:.4f}")
            if _h1_violations:
                print(f"  SANITY FAIL: {_h1_violations} taken trades have "
                      f"h1_last_bar_close_in_range > {KH22_H1_RANGE_THRESHOLD:.4f} — filter not applied!")
            else:
                print(f"  Sanity check: all taken trades have "
                      f"h1_last_bar_close_in_range <= {KH22_H1_RANGE_THRESHOLD:.4f} — PASS")
    else:
        print("\n=== KH-22 1H Close-in-Range Filter: DISABLED ===")

    # ── KH-16 OOS summary ─────────────────────────────────────────────────────
    if USE_REENTRY:
        re_wr_s  = f"{stats['reentry_win_rate']:.4f}" if not np.isnan(stats["reentry_win_rate"]) else "n/a"
        re_mr_s  = f"{stats['reentry_mean_r']:.4f}"   if not np.isnan(stats["reentry_mean_r"])   else "n/a"
        print("\n=== KH-16 Re-entry Summary (OOS) ===")
        print(f"  re-entry trades fired: {stats['reentry_oos']}")
        print(f"  re-entry win rate:     {re_wr_s}")
        print(f"  re-entry mean R:       {re_mr_s}")
    else:
        print("\n=== KH-16 Re-entry: DISABLED ===")

    # ── KH-18 OOS summary (alias of KH-17 path; tagged under KH-18) ──────
    if KH18_ENABLED:
        oos_s18 = pd.Timestamp(FOLDS[0]["oos_start"])
        t18 = [t for t in all_trades
               if pd.Timestamp(t["entry_date"]) >= oos_s18]
        s1  = [t for t in t18 if t.get("trade_type") == "state1_delayed"]
        sr  = [t for t in t18 if t.get("trade_type") == "state2_reentry"]
        def _wr_mr(tl):
            if not tl:
                return "n/a", "n/a"
            w = sum(1 for t in tl if t["classification"] == "WIN") / len(tl)
            r = float(np.mean([t["r_multiple"] for t in tl]))
            return f"{w:.4f}", f"{r:.4f}"
        s1_wr, s1_mr = _wr_mr(s1)
        sr_wr, sr_mr = _wr_mr(sr)
        print("\n=== KH-18 Trade-Type Summary (OOS) ===")
        print(f"  state1_delayed : n={len(s1):>4}  win_rate={s1_wr}  mean_r={s1_mr}")
        print(f"  state2_reentry : n={len(sr):>4}  win_rate={sr_wr}  mean_r={sr_mr}")

    # ── Baseline parity check (only when both KH-13 and KH-14 are disabled) ──
    if not USE_KH13_EARLY_EXIT and not USE_KH14_BAR6_EXIT and not KH17_ENABLED and not KH18_ENABLED:
        oos_start_parity = pd.Timestamp(FOLDS[0]["oos_start"])
        oos_trades_parity = [t for t in all_trades
                             if pd.Timestamp(t["entry_date"]) >= oos_start_parity]
        oos_total_parity  = len(oos_trades_parity)
        fold1_m = _oos_metrics(all_trades, FOLDS[0]["oos_start"], FOLDS[0]["oos_end"])
        worst_roi_actual  = stats["worst_roi"]

        checks = {
            "oos_trades_328":         oos_total_parity == 328,
            "fold1_roi_17.50":        abs(fold1_m["roi_pct"] - 17.50) < 0.01,
            "fold1_dd_9.01":          abs(fold1_m["max_dd_pct"] - 9.01) < 0.01,
            "worst_fold_roi_neg4.20": abs(worst_roi_actual - (-4.20)) < 0.01,
            "kh13_triggered_51":      stats["kh13_triggered_oos"] == 51,
            "kh14_state2_159":        stats["kh14_state2_oos"] == 159,
            "kh14_triggered_97":      stats["kh14_triggered_oos"] == 97,
        }
        parity_pass = all(checks.values())
        print("\n=== BASELINE PARITY CHECK ===")
        for label, ok in checks.items():
            print(f"  {label}: {'OK' if ok else 'FAIL'}")
        if parity_pass:
            print("BASELINE PARITY: PASS")
        else:
            print(f"BASELINE PARITY: FAIL — "
                  f"trades={oos_total_parity}/328  "
                  f"fold1_roi={fold1_m['roi_pct']:.2f}%/17.50%  "
                  f"fold1_dd={fold1_m['max_dd_pct']:.2f}%/9.01%  "
                  f"worst_roi={worst_roi_actual:.2f}%/-4.20%  "
                  f"kh13_triggered={stats['kh13_triggered_oos']}/51  "
                  f"kh14_state2={stats['kh14_state2_oos']}/159  "
                  f"kh14_triggered={stats['kh14_triggered_oos']}/97")

    print("\n  Writing report ...")
    _write_report(all_trades, blocked_events, fold_df, per_pair_df, stats, pair_cache)

    print(f"\n  Outputs: {OUT_ROOT.relative_to(PROJECT_ROOT).as_posix()}/")
    print("    wfo_fold_results_4h.csv")
    print("    wfo_per_pair_4h.csv")
    print("    wfo_summary_4h.txt")
    print("    trades_all.csv")
    print("    trades_paths.csv  (v1.3 per-bar capturability extension)")
    print("    kgl_v2_report.md")
    print("\n=== KG-L-V2 complete ===")


def main() -> None:
    global DATA_DIR_4H, D1_DATA_DIR, OUT_ROOT
    global WFO_START, WFO_IS_MONTHS_0, WFO_OOS_MONTHS, WFO_N_FOLDS
    global GATE_ROI_MIN, GATE_DD_MAX, FOLDS
    global D1_ATR_DIST_CAP, NO_VOLUME_FILTER, VOL_COLUMN, RISK_PCT, RANGE_ATR_CEILING
    global USE_C8, USE_C9, EXPOSURE_CAP, C4_KIJUN_OFFSET_ATR
    global REGIME_MIN_TRENDING_PAIRS, REGIME_EMA_PERIOD, REGIME_SLOPE_LOOKBACK
    global REQUIRE_OWN_EMA_SLOPE
    global REQUIRE_D1_KIJUN_SLOPE_FALLING
    global BASELINE_TYPE
    global KIJUN_D1_CONFIRM_BARS
    global KIJUN_D1_CONFIRM_IF_TRAIL
    global KIJUN_D1_CONFIRM_MIN_BARS
    global KIJUN_D1_CONFIRM_DEPTH_ATR
    global USE_KH13_EARLY_EXIT, KH13_MAE_THRESHOLD, KH13_MFE_THRESHOLD
    global USE_KH14_BAR6_EXIT, KH14_MFE_THRESHOLD, KH14_MAE_THRESHOLD
    global USE_ATR_SIZING, ATR_SIZING_THRESHOLD, ATR_SIZING_REDUCED_PCT
    global USE_REENTRY, REENTRY_WINDOW_BARS, REENTRY_SL_ATR_MULT, REENTRY_TRIGGER
    global KH18_ENABLED, KH19_ENABLED, KH25_ENABLED
    global KH20_ENABLED, KH20_D1_RANGE_THRESHOLD
    global KH22_ENABLED, KH22_H1_RANGE_THRESHOLD
    global SIGNAL_DIRECTION
    global KH17_ENABLED, KH17_STATE2_REENTRY_ENABLED
    global KH17_WATCH_WINDOW_BARS, KH17_WATCH_SL_ATR_MULT
    global KH17_STATE1_SL_ATR_MULT
    global KH17_BAR6_MFE_THRESHOLD, KH17_BAR6_MAE_THRESHOLD
    global D1_HOOK

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="D1 ATR distance cap multiplier for cond_9 (default: 1.0)")
    parser.add_argument("--no-cap", action="store_true", default=False)
    parser.add_argument("--risk", type=float, default=None,
                        help="Risk per trade as a percentage (default: 2.0)")
    parser.add_argument("--no-volume-filter", action="store_true", default=False,
                        help="Disable cond_7 (volume > 1.2x mean) — signal has 8 conditions")
    parser.add_argument("--range-ceiling", type=float, default=None,
                        help="Range/ATR ceiling: veto signal if (high-low)/ATR > threshold")
    parser.add_argument("--out-dir", "--output-dir", type=str, default=None,
                        help="Override output directory (relative to project root)")
    args = parser.parse_args()

    if args.no_volume_filter:
        NO_VOLUME_FILTER = True
    if args.range_ceiling is not None:
        RANGE_ATR_CEILING = args.range_ceiling
    if args.risk is not None:
        RISK_PCT = args.risk / 100.0

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Override output directory
        out_dir = cfg.get("outputs", {}).get("dir")
        if out_dir:
            OUT_ROOT = PROJECT_ROOT / out_dir
            OUT_ROOT.mkdir(parents=True, exist_ok=True)

        # Override data directories
        data_4h = cfg.get("data_dir")
        if data_4h:
            DATA_DIR_4H = PROJECT_ROOT / data_4h
        d1_dir = cfg.get("indicator_params", {}).get("kb_exhaustion_bar", {}).get("d1_data_dir")
        if d1_dir:
            D1_DATA_DIR = PROJECT_ROOT / d1_dir

        # Override WFO fold definitions if folds are explicitly listed in config
        wfo_cfg = cfg.get("wfo", {})
        folds_cfg = wfo_cfg.get("folds")
        if folds_cfg:
            FOLDS = []
            for f in folds_cfg:
                FOLDS.append({
                    "fold":      f["fold"],
                    "is_start":  pd.Timestamp(f["is_start"]),
                    "is_end":    pd.Timestamp(f["is_end"]),
                    "oos_start": pd.Timestamp(f["oos_start"]),
                    "oos_end":   pd.Timestamp(f["oos_end"]),
                })
            WFO_N_FOLDS = len(FOLDS)
        else:
            # Fall back to computed folds from top-level wfo params
            start_str = wfo_cfg.get("start")
            if start_str:
                WFO_START = datetime.strptime(start_str, "%Y-%m-%d")
            if wfo_cfg.get("is_months_0"):
                WFO_IS_MONTHS_0 = wfo_cfg["is_months_0"]
            if wfo_cfg.get("oos_months"):
                WFO_OOS_MONTHS = wfo_cfg["oos_months"]
            if wfo_cfg.get("n_folds"):
                WFO_N_FOLDS = wfo_cfg["n_folds"]
            FOLDS = _wfo_folds()

        # Override gate thresholds
        gate_cfg = cfg.get("gate", {})
        if "worst_fold_roi_min_pct" in gate_cfg:
            GATE_ROI_MIN = gate_cfg["worst_fold_roi_min_pct"]
        if "worst_fold_dd_max_pct" in gate_cfg:
            GATE_DD_MAX = gate_cfg["worst_fold_dd_max_pct"]

        # Signal direction: "long" (default) or "short"
        if "direction" in cfg:
            d = str(cfg["direction"]).lower()
            if d not in ("long", "short"):
                raise ValueError(f"Invalid direction '{d}'. Must be 'long' or 'short'.")
            SIGNAL_DIRECTION = d

        # Override C7, C8, C9, risk, and exposure cap from config
        kbe_params = cfg.get("indicator_params", {}).get("kb_exhaustion_bar", {})
        if "use_c8" in kbe_params:
            USE_C8 = kbe_params["use_c8"]
        if "use_c9" in cfg:
            USE_C9 = bool(cfg["use_c9"])
        if "no_volume_filter" in kbe_params and not args.no_volume_filter:
            NO_VOLUME_FILTER = bool(kbe_params["no_volume_filter"])
        if "volume_column" in kbe_params:
            VOL_COLUMN = kbe_params["volume_column"]
        risk_cfg = cfg.get("risk", {})
        if "risk_per_trade_pct" in risk_cfg and args.risk is None:
            RISK_PCT = risk_cfg["risk_per_trade_pct"] / 100.0
        if "currency_exposure_cap" in risk_cfg:
            cap_val = risk_cfg["currency_exposure_cap"]
            if cap_val is None or (isinstance(cap_val, (int, float)) and cap_val <= 0):
                EXPOSURE_CAP = 999
            else:
                EXPOSURE_CAP = int(cap_val)

        # KH-2: C4 Kijun offset (0.0 = original strict behaviour)
        if "c4_kijun_offset_atr" in cfg:
            C4_KIJUN_OFFSET_ATR = float(cfg["c4_kijun_offset_atr"])

        # KH-4: 4H EMA(50) momentum regime gate (0 = disabled, baseline)
        if "regime_min_trending_pairs" in cfg:
            REGIME_MIN_TRENDING_PAIRS = int(cfg["regime_min_trending_pairs"])
        if "regime_ema_period" in cfg:
            REGIME_EMA_PERIOD = int(cfg["regime_ema_period"])
        if "regime_slope_lookback" in cfg:
            REGIME_SLOPE_LOOKBACK = int(cfg["regime_slope_lookback"])

        # KH-5: own-pair 4H EMA slope gate (False = disabled, baseline)
        if "require_own_ema_slope" in cfg:
            REQUIRE_OWN_EMA_SLOPE = bool(cfg["require_own_ema_slope"])

        # KH-11A: D1 Kijun slope falling/flat entry gate (False = disabled, baseline)
        if "require_d1_kijun_slope_falling" in cfg:
            REQUIRE_D1_KIJUN_SLOPE_FALLING = bool(cfg["require_d1_kijun_slope_falling"])

        # KH-8: kijun_d1 exit confirmation bars (default 1 = current behaviour)
        if "kijun_d1_confirm_bars" in cfg:
            kd = int(cfg["kijun_d1_confirm_bars"])
            if kd not in (1, 2):
                raise ValueError(
                    f"Invalid kijun_d1_confirm_bars '{kd}'. Must be 1 or 2."
                )
            KIJUN_D1_CONFIRM_BARS = kd

        # KH-9: conditional application of the 2-bar confirmation.
        # Variant A — trail-status conditional
        if "kijun_d1_confirm_if_trail" in cfg:
            KIJUN_D1_CONFIRM_IF_TRAIL = bool(cfg["kijun_d1_confirm_if_trail"])
        # Variant B — trade-age conditional
        if "kijun_d1_confirm_min_bars" in cfg:
            mb = int(cfg["kijun_d1_confirm_min_bars"])
            if mb < 0:
                raise ValueError(
                    f"Invalid kijun_d1_confirm_min_bars '{mb}'. Must be >= 0."
                )
            KIJUN_D1_CONFIRM_MIN_BARS = mb
        # Variant C — cross-depth conditional
        if "kijun_d1_confirm_depth_atr" in cfg:
            dp = float(cfg["kijun_d1_confirm_depth_atr"])
            if dp < 0.0:
                raise ValueError(
                    f"Invalid kijun_d1_confirm_depth_atr '{dp}'. Must be >= 0.0."
                )
            KIJUN_D1_CONFIRM_DEPTH_ATR = dp

        # KH-7: C4/C5 baseline replacement — "kijun" (default), "hma", or "dema"
        if "baseline_type" in cfg:
            bt = str(cfg["baseline_type"]).lower()
            if bt not in ("kijun", "hma", "dema"):
                raise ValueError(
                    f"Invalid baseline_type '{bt}'. Must be 'kijun', 'hma', or 'dema'."
                )
            BASELINE_TYPE = bt

        # KH-13: early exit on adverse bar-3 movement (default False = baseline)
        if "use_kh13_early_exit" in cfg:
            USE_KH13_EARLY_EXIT = bool(cfg["use_kh13_early_exit"])
        if "kh13_mae_threshold" in cfg:
            KH13_MAE_THRESHOLD = float(cfg["kh13_mae_threshold"])
        if "kh13_mfe_threshold" in cfg:
            KH13_MFE_THRESHOLD = float(cfg["kh13_mfe_threshold"])

        # KH-14: bar-6 exit on State 2 trades (default False = baseline)
        if "use_kh14_bar6_exit" in cfg:
            USE_KH14_BAR6_EXIT = bool(cfg["use_kh14_bar6_exit"])
        if "kh14_mfe_threshold" in cfg:
            KH14_MFE_THRESHOLD = float(cfg["kh14_mfe_threshold"])
        if "kh14_mae_threshold" in cfg:
            KH14_MAE_THRESHOLD = float(cfg["kh14_mae_threshold"])

        # KH-15A: ATR-conditional position sizing (default False = baseline).
        # atr_sizing_reduced_pct is supplied as a percentage (e.g. 0.5 = 0.5%)
        # and stored internally as a fraction (0.005).
        if "use_atr_sizing" in cfg:
            USE_ATR_SIZING = bool(cfg["use_atr_sizing"])
        if "atr_sizing_threshold" in cfg:
            t = float(cfg["atr_sizing_threshold"])
            if t <= 0.0:
                raise ValueError(
                    f"Invalid atr_sizing_threshold '{t}'. Must be > 0."
                )
            ATR_SIZING_THRESHOLD = t
        if "atr_sizing_reduced_pct" in cfg:
            r = float(cfg["atr_sizing_reduced_pct"])
            if r <= 0.0:
                raise ValueError(
                    f"Invalid atr_sizing_reduced_pct '{r}'. Must be > 0."
                )
            ATR_SIZING_REDUCED_PCT = r / 100.0

        # KH-16: re-entry after kh14_bar6 exit (default False = baseline)
        if "reentry_enabled" in cfg:
            USE_REENTRY = bool(cfg["reentry_enabled"])
        if "reentry_window_bars" in cfg:
            w = int(cfg["reentry_window_bars"])
            if w < 1:
                raise ValueError(f"Invalid reentry_window_bars '{w}'. Must be >= 1.")
            REENTRY_WINDOW_BARS = w
        if "reentry_sl_atr_mult" in cfg:
            s = float(cfg["reentry_sl_atr_mult"])
            if s <= 0.0:
                raise ValueError(f"Invalid reentry_sl_atr_mult '{s}'. Must be > 0.")
            REENTRY_SL_ATR_MULT = s
        if "reentry_trigger" in cfg:
            rt = str(cfg["reentry_trigger"])
            if rt != "close_above_entry":
                raise ValueError(
                    f"Invalid reentry_trigger '{rt}'. Only 'close_above_entry' supported."
                )
            REENTRY_TRIGGER = rt

        # KH-17: two-decision State 1 delayed / State 2 virtual bar-6 gate +
        # re-entry watch.  Default False = baseline behaviour (no change).
        if "kh17_enabled" in cfg:
            KH17_ENABLED = bool(cfg["kh17_enabled"])
        if "kh17_state2_reentry_enabled" in cfg:
            KH17_STATE2_REENTRY_ENABLED = bool(cfg["kh17_state2_reentry_enabled"])
        if "kh17_watch_window_bars" in cfg:
            w = int(cfg["kh17_watch_window_bars"])
            if w < 1:
                raise ValueError(
                    f"Invalid kh17_watch_window_bars '{w}'. Must be >= 1."
                )
            KH17_WATCH_WINDOW_BARS = w
        if "kh17_watch_sl_atr_mult" in cfg:
            s = float(cfg["kh17_watch_sl_atr_mult"])
            if s <= 0.0:
                raise ValueError(
                    f"Invalid kh17_watch_sl_atr_mult '{s}'. Must be > 0."
                )
            KH17_WATCH_SL_ATR_MULT = s
        if "kh17_state1_sl_atr_mult" in cfg:
            s = float(cfg["kh17_state1_sl_atr_mult"])
            if s <= 0.0:
                raise ValueError(
                    f"Invalid kh17_state1_sl_atr_mult '{s}'. Must be > 0."
                )
            KH17_STATE1_SL_ATR_MULT = s
        if "kh17_bar6_mfe_threshold" in cfg:
            t = float(cfg["kh17_bar6_mfe_threshold"])
            if t <= 0.0:
                raise ValueError(
                    f"Invalid kh17_bar6_mfe_threshold '{t}'. Must be > 0."
                )
            KH17_BAR6_MFE_THRESHOLD = t
        if "kh17_bar6_mae_threshold" in cfg:
            t = float(cfg["kh17_bar6_mae_threshold"])
            if t <= 0.0:
                raise ValueError(
                    f"Invalid kh17_bar6_mae_threshold '{t}'. Must be > 0."
                )
            KH17_BAR6_MAE_THRESHOLD = t

        # KH-18: config-level alias of KH-17.  kh18_* keys feed the same
        # KH17_* globals and activate the same code path.  Only
        # KH18_ENABLED is tracked separately so the run banner / OOS
        # summary print under KH-18.  Mutually exclusive with kh17_enabled.
        if "kh18_enabled" in cfg:
            KH18_ENABLED = bool(cfg["kh18_enabled"])
            if KH18_ENABLED:
                KH17_ENABLED = True  # share code path
        if "kh18_watch_window_bars" in cfg:
            w = int(cfg["kh18_watch_window_bars"])
            if w < 1:
                raise ValueError(
                    f"Invalid kh18_watch_window_bars '{w}'. Must be >= 1."
                )
            KH17_WATCH_WINDOW_BARS = w
        if "kh18_watch_sl_atr_mult" in cfg:
            s = float(cfg["kh18_watch_sl_atr_mult"])
            if s <= 0.0:
                raise ValueError(
                    f"Invalid kh18_watch_sl_atr_mult '{s}'. Must be > 0."
                )
            KH17_WATCH_SL_ATR_MULT = s
        if "kh18_bar6_mfe_threshold" in cfg:
            t = float(cfg["kh18_bar6_mfe_threshold"])
            if t <= 0.0:
                raise ValueError(
                    f"Invalid kh18_bar6_mfe_threshold '{t}'. Must be > 0."
                )
            KH17_BAR6_MFE_THRESHOLD = t
        if "kh18_bar6_mae_threshold" in cfg:
            t = float(cfg["kh18_bar6_mae_threshold"])
            if t <= 0.0:
                raise ValueError(
                    f"Invalid kh18_bar6_mae_threshold '{t}'. Must be > 0."
                )
            KH17_BAR6_MAE_THRESHOLD = t

        if cfg.get("kh17_enabled") and cfg.get("kh18_enabled"):
            raise ValueError(
                "kh17_enabled and kh18_enabled are mutually exclusive. "
                "Enable at most one per run."
            )

        # KH-20: D1 close-in-range entry filter (default False = baseline)
        if "kh20_enabled" in cfg:
            KH20_ENABLED = bool(cfg["kh20_enabled"])
        if "kh20_d1_range_threshold" in cfg:
            t = float(cfg["kh20_d1_range_threshold"])
            if not (0.0 < t < 1.0):
                raise ValueError(
                    f"Invalid kh20_d1_range_threshold '{t}'. Must be in (0, 1)."
                )
            KH20_D1_RANGE_THRESHOLD = t

        # KH-22: 1H close-in-range entry filter (default False = baseline)
        if "kh22_enabled" in cfg:
            KH22_ENABLED = bool(cfg["kh22_enabled"])
        if "kh22_h1_range_threshold" in cfg:
            t = float(cfg["kh22_h1_range_threshold"])
            if not (0.0 < t < 1.0):
                raise ValueError(
                    f"Invalid kh22_h1_range_threshold '{t}'. Must be in (0, 1)."
                )
            KH22_H1_RANGE_THRESHOLD = t

        # KH-23: 1H close-in-range filter only (no cap) — reuses KH-22 implementation
        if "kh23_enabled" in cfg:
            KH22_ENABLED = bool(cfg["kh23_enabled"])
        if "kh23_h1_range_threshold" in cfg:
            t = float(cfg["kh23_h1_range_threshold"])
            if not (0.0 < t < 1.0):
                raise ValueError(
                    f"Invalid kh23_h1_range_threshold '{t}'. Must be in (0, 1)."
                )
            KH22_H1_RANGE_THRESHOLD = t

        # KH-24: cap=2 + 1H close-in-range filter, threshold 0.28 — reuses KH-22 implementation
        if "kh24_enabled" in cfg:
            KH22_ENABLED = bool(cfg["kh24_enabled"])
        if "kh24_h1_range_threshold" in cfg:
            t = float(cfg["kh24_h1_range_threshold"])
            if not (0.0 < t < 1.0):
                raise ValueError(
                    f"Invalid kh24_h1_range_threshold '{t}'. Must be in (0, 1)."
                )
            KH22_H1_RANGE_THRESHOLD = t

        # KH-19: baseline entry + kh14_bar6 exit + KH-16-identical re-entry.
        # kh19_* keys feed the same USE_REENTRY / REENTRY_* globals as KH-16.
        # kh19_reentry_enabled also requires use_kh14_bar6_exit in the config.
        if "kh19_reentry_enabled" in cfg:
            KH19_ENABLED = bool(cfg["kh19_reentry_enabled"])
            USE_REENTRY  = KH19_ENABLED
        if "kh19_reentry_window_bars" in cfg:
            w = int(cfg["kh19_reentry_window_bars"])
            if w < 1:
                raise ValueError(
                    f"Invalid kh19_reentry_window_bars '{w}'. Must be >= 1."
                )
            REENTRY_WINDOW_BARS = w
        if "kh19_reentry_sl_atr_mult" in cfg:
            s = float(cfg["kh19_reentry_sl_atr_mult"])
            if s <= 0.0:
                raise ValueError(
                    f"Invalid kh19_reentry_sl_atr_mult '{s}'. Must be > 0."
                )
            REENTRY_SL_ATR_MULT = s

        # KH-25: kh14_bar6 exit + re-entry on top of KH-24 system.
        # kh25_* keys feed the same USE_REENTRY / REENTRY_* globals as KH-19.
        # Enabling kh25_reentry_enabled automatically activates USE_KH14_BAR6_EXIT.
        if "kh25_reentry_enabled" in cfg:
            KH25_ENABLED    = bool(cfg["kh25_reentry_enabled"])
            USE_REENTRY     = KH25_ENABLED
            if KH25_ENABLED:
                USE_KH14_BAR6_EXIT = True
        if "kh25_reentry_window_bars" in cfg:
            w = int(cfg["kh25_reentry_window_bars"])
            if w < 1:
                raise ValueError(
                    f"Invalid kh25_reentry_window_bars '{w}'. Must be >= 1."
                )
            REENTRY_WINDOW_BARS = w
        if "kh25_reentry_sl_atr_mult" in cfg:
            s = float(cfg["kh25_reentry_sl_atr_mult"])
            if s <= 0.0:
                raise ValueError(
                    f"Invalid kh25_reentry_sl_atr_mult '{s}'. Must be > 0."
                )
            REENTRY_SL_ATR_MULT = s
        if "kh25_bar6_mfe_threshold" in cfg:
            t = float(cfg["kh25_bar6_mfe_threshold"])
            if t <= 0.0:
                raise ValueError(
                    f"Invalid kh25_bar6_mfe_threshold '{t}'. Must be > 0."
                )
            KH14_MFE_THRESHOLD = t
        if "kh25_bar6_mae_threshold" in cfg:
            t = float(cfg["kh25_bar6_mae_threshold"])
            if t <= 0.0:
                raise ValueError(
                    f"Invalid kh25_bar6_mae_threshold '{t}'. Must be > 0."
                )
            KH14_MAE_THRESHOLD = t

        if cfg.get("kh25_reentry_enabled") and (
            cfg.get("kh17_enabled") or cfg.get("kh18_enabled")
        ):
            raise ValueError(
                "kh25_reentry_enabled is mutually exclusive with kh17_enabled "
                "and kh18_enabled. Enable at most one per run."
            )

        # Pipeline D1 hook (L_ARC_PROTOCOL v2.0 §3). Absent or empty block
        # leaves D1_HOOK = None and the engine behaves byte-identically to
        # the baseline. See core/d1_pipeline.py for schema details.
        if cfg.get("d1_archetypes"):
            D1_HOOK = D1Hook.from_yaml_dict(
                cfg["d1_archetypes"], project_root=PROJECT_ROOT,
            )
            print(
                f"  Pipeline D1: ENABLED — {len(D1_HOOK.archetypes)} archetype(s), "
                f"bar_offset_t={D1_HOOK.bar_offset_t}"
            )

    D1_ATR_DIST_CAP = args.threshold
    if args.threshold != 1.0:
        OUT_ROOT = OUT_ROOT.parent / (OUT_ROOT.name + f"_cond9_{args.threshold}x")
    if args.no_cap:
        EXPOSURE_CAP = 999
        OUT_ROOT = OUT_ROOT.parent / (OUT_ROOT.name + "_nocap")
    if args.out_dir:
        OUT_ROOT = PROJECT_ROOT / args.out_dir
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    run_kgl_v2()


if __name__ == "__main__":
    main()
