# =============================================
# signal_logic.py  (indicator-ready, YAML-driven exits, no clobber)
# * Reads columns produced by apply_indicators():
#     - c1_signal, c2_signal, baseline_signal, volume_signal
#     - optional: baseline (price series for pullback/cross), atr
# * Outputs: entry_signal (±1), exit_signal (0/1)
# * No lookahead: decisions use yesterday's signals unless config allows same-day baseline catalyst.
# * Exit signals: engine logic OR Exit indicator (config-controlled).
# =============================================
# signal_logic.py — v1.9.8

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any

# Project imports
from utils import calculate_atr
from backtester_helpers import apply_indicators, coerce_entry_exit_signals

# -----------------------------
# Helpers
# -----------------------------
def _series(df: pd.DataFrame, col: str, default=0):
    if col in df.columns:
        return df[col]
    # default scalar -> series
    return pd.Series(default, index=df.index)

def _has_nonnull(df: pd.DataFrame, col: str) -> bool:
    return (col in df.columns) and (df[col].notna().any())

# -----------------------------
# Filter checks
# -----------------------------
def _get_filter_status(df, i, direction, atr_i, use_c2, use_volume, use_baseline,
                       has_baseline_value, baseline_value_col, baseline_sig_col,
                       use_pullback) -> tuple[bool, list]:
    """
    Evaluate confirmation filters on row i for a proposed trade 'direction'.
    Returns (passed: bool, failed_filters: list[str]).
    """
    failed = []

    # C2 direction must match
    if use_c2:
        if df.loc[i, 'c2_signal'] != direction:
            failed.append('c2')

    # Volume must be pass (=1)
    if use_volume:
        if df.loc[i, 'volume_signal'] != 1:
            failed.append('volume')

    # Baseline directional agreement (only if baseline is enabled)
    if use_baseline:
        baseline_dir_ok = True
        if has_baseline_value:
            price = df.loc[i, 'close']
            base  = df.loc[i, baseline_value_col]
            baseline_dir_ok = ((direction == 1 and price > base) or
                               (direction == -1 and price < base))
        elif baseline_sig_col in df.columns:
            baseline_dir_ok = (df.loc[i, baseline_sig_col] == direction)

        if not baseline_dir_ok:
            failed.append('baseline_dir')

    # Pullback rule (requires baseline value)
    if use_pullback and has_baseline_value:
        price = df.loc[i, 'close']
        base  = df.loc[i, baseline_value_col]
        too_far = ((direction == 1 and price > base + atr_i) or
                   (direction == -1 and price < base - atr_i))

        # one-bar lookback "was also too far yesterday?"
        prev_price = df.loc[i - 1, 'close']
        prev_base  = df.loc[i - 1, baseline_value_col]
        prev_too_far = ((direction == 1 and prev_price > prev_base + atr_i) or
                        (direction == -1 and prev_price < prev_base - atr_i))

        if too_far and prev_too_far:
            failed.append('pullback')

    return (len(failed) == 0), failed

def _did_filter_pass(df, i, filter_name, direction, atr_i,
                     has_baseline_value, baseline_value_col, baseline_sig_col):
    """Recovery check for One-Candle / Pullback on bar i."""
    if filter_name == 'c2':
        return df.loc[i, 'c2_signal'] == direction
    elif filter_name == 'volume':
        return df.loc[i, 'volume_signal'] == 1
    elif filter_name == 'baseline_dir':
        if has_baseline_value:
            base = df.loc[i, baseline_value_col]
            price = df.loc[i, 'close']
            return ((direction == 1 and price > base) or
                    (direction == -1 and price < base))
        elif baseline_sig_col in df.columns:
            return df.loc[i, baseline_sig_col] == direction
        return True
    elif filter_name == 'pullback':
        if has_baseline_value:
            base = df.loc[i, baseline_value_col]
            price = df.loc[i, 'close']
            return not ((direction == 1 and price > base + atr_i) or
                        (direction == -1 and price < base - atr_i))
        return True
    return True

# -----------------------------
# Main signal logic
# -----------------------------
def apply_signal_logic(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Generate entry/exit signals using ONLY precomputed indicator columns.
    - C1 is the trigger (yesterday's confirmed value) unless baseline catalyst is allowed.
    - ONE-BAR grace total:
        * C2/Volume 'one-candle' (only if enabled), AND/OR
        * Pullback (price must be within 1×ATR of baseline next bar),
      but never two separate bars.
    - Exits are YAML-driven and OR'ed with the dedicated Exit indicator if enabled.
    - Quiet by default; set tracking.verbose_logs: true to print debug lines.

    Invariants:
      - entry_signal / exit_signal are ALWAYS integers in {-1, 0, +1}.
      - Robust to NaN in source columns.
    """
    VERBOSE = bool(config.get("tracking", {}).get("verbose_logs", False))

    # --- NEW: compute indicators via centralized helper (handles ATR + params) ---
    # Ensures c1_signal / baseline(_signal) / c2_signal / volume_signal / exit_signal are in place if enabled.
    df = apply_indicators(df, config)

    # Defensive: ensure ATR exists (apply_indicators normally adds it; this is a safety net)
    if "atr" not in df.columns:
        try:
            df = calculate_atr(df)
        except Exception:
            df["atr"] = 0.0

    out = df.copy()

    # Preserve any dedicated Exit indicator that may already be present
    if "exit_signal" in out.columns:
        exit_indicator_series = pd.to_numeric(out["exit_signal"], errors="coerce").fillna(0).astype(int)
    else:
        exit_indicator_series = pd.Series([0]*len(out), index=out.index, dtype="int8")

    # Build fresh, integer signals (initialize as zeros)
    out["entry_signal"] = 0
    out["exit_signal"]  = 0

    # Config flags
    ind_cfg  = (config.get('indicators') or {})
    rules    = (config.get('rules') or {})
    exit_cfg = (config.get('exit') or {})

    USE_C2       = bool(ind_cfg.get('use_c2', False))
    USE_VOLUME   = bool(ind_cfg.get('use_volume', False))
    USE_BASELINE = bool(ind_cfg.get('use_baseline', False))
    USE_EXIT_IND = bool(ind_cfg.get('use_exit', False))

    USE_BASELINE_TRIGGER = bool(rules.get('allow_baseline_as_catalyst', False))
    USE_PULLBACK         = bool(rules.get('pullback_rule', False))
    USE_ONE_CANDLE       = bool(rules.get('one_candle_rule', False))
    BRIDGE_TOO_FAR_DAYS  = int(rules.get('bridge_too_far_days', 7))
    PULLBACK_ATR_MULT    = float(rules.get('pullback_atr_mult', 1.0))

    EXIT_ON_C1_REV       = bool(exit_cfg.get("exit_on_c1_reversal", True))
    EXIT_ON_BASE_CROSS   = bool(exit_cfg.get("exit_on_baseline_cross", False))
    EXIT_ON_EXIT_INDIC   = bool(exit_cfg.get("exit_on_exit_signal", False)) and USE_EXIT_IND

    # Baseline availability
    has_baseline_value = ("baseline" in out.columns) and (out["baseline"].notna().any())
    baseline_value_col = 'baseline' if has_baseline_value else None
    baseline_sig_col   = 'baseline_signal'  # may or may not be present

    # Attr for one-bar pending
    if 'pending_trade' not in out.attrs:
        out.attrs['pending_trade'] = None

    def _c1_at(idx: int) -> int:
        if 'c1_signal' not in out.columns:
            return 0
        val = pd.to_numeric(out.loc[idx, 'c1_signal'], errors='coerce')
        if pd.isna(val):
            return 0
        # Expecting -1/0/+1 already; coerce safely
        return int(np.sign(val)) if val not in (-1, 0, 1) else int(val)

    # Iterate (need 2 bars back to detect C1 flip)
    for i in range(2, len(out)):
        atr_i   = float(pd.to_numeric(out.loc[i, 'atr'], errors='coerce')) if pd.notna(out.loc[i, 'atr']) else 0.0
        close_i = float(pd.to_numeric(out.loc[i, 'close'], errors='coerce'))

        # ------------- Exits (engine + optional indicator) -------------
        c1_yday  = _c1_at(i - 1)
        c1_prev2 = _c1_at(i - 2)

        engine_exit_flag = 0

        # Exit on C1 reversal?
        if EXIT_ON_C1_REV and ('c1_signal' in out.columns):
            if (c1_yday != 0) and (c1_prev2 != 0) and (c1_yday == -1 * c1_prev2):
                engine_exit_flag = 1

        # Exit on Baseline cross?
        if EXIT_ON_BASE_CROSS and USE_BASELINE:
            if has_baseline_value:
                prev_close = float(out.loc[i - 1, 'close'])
                prev_base  = float(out.loc[i - 1, baseline_value_col])
                base_now   = float(out.loc[i,   baseline_value_col])
                crossed = ((close_i > base_now and prev_close < prev_base) or
                           (close_i < base_now and prev_close > prev_base))
                if crossed:
                    engine_exit_flag = 1
            elif 'baseline_signal' in out.columns:
                base_y  = int(pd.to_numeric(out.loc[i - 1, 'baseline_signal'], errors='coerce') if pd.notna(out.loc[i - 1, 'baseline_signal']) else 0)
                base_y2 = int(pd.to_numeric(out.loc[i - 2, 'baseline_signal'], errors='coerce') if pd.notna(out.loc[i - 2, 'baseline_signal']) else 0)
                if (base_y != 0) and (base_y2 != 0) and (base_y != base_y2):
                    engine_exit_flag = 1

        final_exit = (engine_exit_flag == 1)
        if EXIT_ON_EXIT_INDIC:
            final_exit = final_exit or (int(exit_indicator_series.iloc[i]) == 1)

        out.loc[i, 'exit_signal'] = 1 if final_exit else 0

        # ------------- Resolve ONE-BAR pending from yesterday -------------
        pending = out.attrs.get('pending_trade')
        if pending and pending.get('index') == i - 1:
            direction = int(pending['direction'])

            # Fresh filter check for today
            passed_today, failed_today = _get_filter_status(
                out, i, direction, atr_i, USE_C2, USE_VOLUME, USE_BASELINE,
                has_baseline_value, baseline_value_col, baseline_sig_col,
                USE_PULLBACK
            )

            # If we were waiting on pullback: require "within 1×ATR * mult of baseline" TODAY
            if pending.get('needs_pullback', False) and has_baseline_value:
                base_now = float(out.loc[i, baseline_value_col])
                within = ((direction == 1 and close_i <= base_now + PULLBACK_ATR_MULT * atr_i) or
                          (direction == -1 and close_i >= base_now - PULLBACK_ATR_MULT * atr_i))
                if not within:
                    passed_today = False
                    if VERBOSE:
                        print(f"[PENDING-FAIL] {out.loc[i, 'date']} pullback not within {PULLBACK_ATR_MULT:.2f}×ATR")

            # If we were waiting on C2/Volume: forbid a "close in favor" TODAY
            if pending.get('needs_c2vol', False):
                close_prev = float(pending['close_ref'])
                closed_in_favor = ((direction == 1 and close_i > close_prev) or
                                   (direction == -1 and close_i < close_prev))
                if closed_in_favor:
                    passed_today = False
                    if VERBOSE:
                        print(f"[PENDING-FAIL] {out.loc[i, 'date']} one-candle: CLOSED in favor")

            if passed_today:
                out.loc[i, 'entry_signal'] = direction
                if VERBOSE:
                    print(f"[ENTRY-DEF] {out.loc[i, 'date']} one-bar deferral satisfied")
            else:
                if VERBOSE:
                    print(f"[DROP] {out.loc[i, 'date']} one-bar deferral not satisfied; no trade")

            # Always clear — one bar max
            out.attrs['pending_trade'] = None
            # (no continue; allow baseline catalyst below if no entry was placed)

        # ------------- Entries -------------
        # C1 trigger: yesterday's confirmed C1 flip
        flip = (c1_yday != 0) and (c1_prev2 != 0) and (c1_yday != c1_prev2)
        if flip:
            direction = int(c1_yday)
            passed, failed = _get_filter_status(
                out, i, direction, atr_i, USE_C2, USE_VOLUME, USE_BASELINE,
                has_baseline_value, baseline_value_col, baseline_sig_col,
                USE_PULLBACK
            )

            # Enforce "too far today" regardless of _get_filter_status internals
            pullback_too_far_today = False
            if USE_PULLBACK and has_baseline_value:
                base_now = float(out.loc[i, baseline_value_col])
                pullback_too_far_today = (
                    (direction == 1 and close_i > base_now + PULLBACK_ATR_MULT * atr_i) or
                    (direction == -1 and close_i < base_now - PULLBACK_ATR_MULT * atr_i)
                )
                if pullback_too_far_today:
                    passed = False
                    if 'pullback' not in failed:
                        failed.append('pullback')

            if passed:
                out.loc[i, 'entry_signal'] = direction
                if VERBOSE:
                    print(f"[ENTRY] {out.loc[i, 'date']} dir={direction} c1_yday={c1_yday} c1_prev2={c1_prev2}")
            else:
                c2_failed       = ('c2' in failed)
                vol_failed      = ('volume' in failed)
                base_failed     = ('baseline_dir' in failed)   # no grace if baseline direction failed
                pullback_failed = ('pullback' in failed) or pullback_too_far_today

                if base_failed:
                    if VERBOSE:
                        print(f"[SKIP] {out.loc[i, 'date']} baseline_dir failed — no deferral")
                else:
                    # ONE-BAR pending that can be satisfied tomorrow
                    needs_c2vol = (c2_failed or vol_failed) and USE_ONE_CANDLE
                    needs_pull  = pullback_failed and USE_PULLBACK
                    if needs_c2vol or needs_pull:
                        out.attrs['pending_trade'] = {
                            'index': i,
                            'direction': int(direction),
                            'needs_c2vol': bool(needs_c2vol),
                            'needs_pullback': bool(needs_pull),
                            'close_ref': float(out.loc[i, 'close']),
                        }
                        if VERBOSE:
                            print(f"[PENDING] {out.loc[i, 'date']} needs_c2vol={needs_c2vol} needs_pullback={needs_pull}")
                    else:
                        if VERBOSE:
                            print(f"[SKIP] {out.loc[i, 'date']} filters failed (no eligible deferral): {failed}")

        # Baseline catalyst (same-day) if allowed — NO deferral here
        elif USE_BASELINE_TRIGGER and (USE_BASELINE or has_baseline_value or ('baseline_signal' in out.columns)):
            crossed = False
            direction = 0

            if has_baseline_value:
                base_prev  = float(out.loc[i - 1, baseline_value_col])
                base_now   = float(out.loc[i,     baseline_value_col])
                close_prev = float(out.loc[i - 1, 'close'])
                close_now  = float(out.loc[i,     'close'])

                crossed_up = (close_prev <= base_prev) and (close_now > base_now)
                crossed_dn = (close_prev >= base_prev) and (close_now < base_now)
                if crossed_up or crossed_dn:
                    crossed = True
                    direction = 1 if crossed_up else -1
            else:
                # Fallback: detect cross via baseline_signal flip
                if baseline_sig_col in out.columns:
                    bsig_y   = int(pd.to_numeric(out.loc[i - 1, baseline_sig_col], errors='coerce') if pd.notna(out.loc[i - 1, baseline_sig_col]) else 0)
                    bsig_now = int(pd.to_numeric(out.loc[i,     baseline_sig_col], errors='coerce') if pd.notna(out.loc[i,     baseline_sig_col]) else 0)
                    if (bsig_y != 0) and (bsig_now != 0) and (bsig_now != bsig_y):
                        crossed = True
                        direction = bsig_now

            if crossed:
                # All other filters must pass on this bar (no deferral)
                passed, failed = _get_filter_status(
                    out, i, direction, atr_i, USE_C2, USE_VOLUME, USE_BASELINE,
                    has_baseline_value, baseline_value_col, baseline_sig_col,
                    USE_PULLBACK
                )

                # Bridge-Too-Far: last C1 of the SAME direction
                last_same_dir = None
                for j in range(i - 1, -1, -1):   # scan full history
                    csig = _c1_at(j)
                    if csig == direction:
                        last_same_dir = j
                        break
                days_since_c1_same = (i - last_same_dir) if last_same_dir is not None else 999

                if passed and (days_since_c1_same < BRIDGE_TOO_FAR_DAYS):
                    out.loc[i, 'entry_signal'] = direction
                    if VERBOSE:
                        print(f"[ENTRY-BASE] {out.loc[i, 'date']} dir={direction} days_since_c1_same={days_since_c1_same}")
                else:
                    if VERBOSE:
                        why = (failed if not passed else f"bridge_too_far ({days_since_c1_same}d)")
                        print(f"[SKIP-BASE] {out.loc[i, 'date']} {why}")
            # else: no baseline-catalyst attempt

    # Clear any leftover pending (should be none)
    out.attrs['pending_trade'] = None

    # ---- Final normalization: ensure clean {-1,0,1} ints, never None ----
    for col in ["entry_signal", "exit_signal"]:
        out[col] = (
            pd.to_numeric(out.get(col, 0), errors="coerce")
              .replace([np.inf, -np.inf], np.nan)
              .fillna(0)
              .clip(-1, 1)
              .astype(int)
        )

    # Also run the shared coercer (no-op if already clean; keeps parity with other modules)
    out = coerce_entry_exit_signals(out)

    return out
