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

# Note: helpers imported elsewhere; not needed here for exit logic only


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
def _get_filter_status(
    df,
    i,
    direction,
    atr_i,
    use_c2,
    use_volume,
    use_baseline,
    has_baseline_value,
    baseline_value_col,
    baseline_sig_col,
    use_pullback,
) -> tuple[bool, list]:
    """
    Evaluate confirmation filters on row i for a proposed trade 'direction'.
    Returns (passed: bool, failed_filters: list[str]).
    """
    failed = []

    # C2 direction must match
    if use_c2:
        if df.loc[i, "c2_signal"] != direction:
            failed.append("c2")

    # Volume must be pass (=1)
    if use_volume:
        if df.loc[i, "volume_signal"] != 1:
            failed.append("volume")

    # Baseline directional agreement (only if baseline is enabled)
    if use_baseline:
        baseline_dir_ok = True
        if has_baseline_value:
            price = df.loc[i, "close"]
            base = df.loc[i, baseline_value_col]
            baseline_dir_ok = (direction == 1 and price > base) or (
                direction == -1 and price < base
            )
        elif baseline_sig_col in df.columns:
            baseline_dir_ok = df.loc[i, baseline_sig_col] == direction

        if not baseline_dir_ok:
            failed.append("baseline_dir")

    # Pullback rule (requires baseline value)
    if use_pullback and has_baseline_value:
        price = df.loc[i, "close"]
        base = df.loc[i, baseline_value_col]
        too_far = (direction == 1 and price > base + atr_i) or (
            direction == -1 and price < base - atr_i
        )

        # one-bar lookback "was also too far yesterday?"
        prev_price = df.loc[i - 1, "close"]
        prev_base = df.loc[i - 1, baseline_value_col]
        prev_too_far = (direction == 1 and prev_price > prev_base + atr_i) or (
            direction == -1 and prev_price < prev_base - atr_i
        )

        if too_far and prev_too_far:
            failed.append("pullback")

    return (len(failed) == 0), failed


def _did_filter_pass(
    df, i, filter_name, direction, atr_i, has_baseline_value, baseline_value_col, baseline_sig_col
):
    """Recovery check for One-Candle / Pullback on bar i."""
    if filter_name == "c2":
        return df.loc[i, "c2_signal"] == direction
    elif filter_name == "volume":
        return df.loc[i, "volume_signal"] == 1
    elif filter_name == "baseline_dir":
        if has_baseline_value:
            base = df.loc[i, baseline_value_col]
            price = df.loc[i, "close"]
            return (direction == 1 and price > base) or (direction == -1 and price < base)
        elif baseline_sig_col in df.columns:
            return df.loc[i, baseline_sig_col] == direction
        return True
    elif filter_name == "pullback":
        if has_baseline_value:
            base = df.loc[i, baseline_value_col]
            price = df.loc[i, "close"]
            return not (
                (direction == 1 and price > base + atr_i)
                or (direction == -1 and price < base - atr_i)
            )
        return True
    return True


# -----------------------------
# Main signal logic
# -----------------------------
def _detect_c1_col(df: pd.DataFrame) -> str | None:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    for name in ("c1_signal", "c1", "signal_c1"):
        if name in lower:
            return lower[name]

    for c in cols:
        cl = c.lower()
        if "c1" in cl and "signal" in cl:
            return c

    signal_like = [c for c in cols if "signal" in c.lower()]
    if len(signal_like) == 1:
        return signal_like[0]

    target = {-1, 0, 1}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        uniq = set(s.dropna().unique().tolist())
        if 0 < len(uniq) <= 3 and uniq.issubset(target):
            return c
    return None


def apply_signal_logic(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    if "exit_signal" not in out.columns:
        out["exit_signal"] = np.nan

    exit_cfg = cfg.get("exit") or {}
    if bool(exit_cfg.get("exit_on_c1_reversal", False)):
        c1_col = _detect_c1_col(out)
        if c1_col is not None and c1_col in out.columns:
            c1 = pd.to_numeric(out[c1_col], errors="coerce")
            flips = c1.ne(c1.shift(1))
            out.loc[flips, "exit_signal"] = 1.0
    return out
