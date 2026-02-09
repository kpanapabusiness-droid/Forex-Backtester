# indicators/confirmation_funcs.py — C1 confirmation indicators (Phase B.1 archetypes + discovery).
# All C1 functions: (df, *, signal_col="c1_signal", **kwargs) -> df; write {-1,0,+1} to signal_col.

from __future__ import annotations

import numpy as np
import pandas as pd

from core.utils import calculate_atr


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Causal ATR: rolling mean of True Range. period >= 1."""
    period = max(1, int(period))
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _ema_series(series: pd.Series, span: int) -> pd.Series:
    """Causal EMA with span."""
    span = max(1, int(span))
    return series.astype(float).ewm(span=span, adjust=False).mean()


# ---------------------------------------------------------------------------
# Canonical legacy C1s that remain active
# ---------------------------------------------------------------------------


def c1_coral(df: pd.DataFrame, period: int = 21, signal_col: str = "c1_signal", **kwargs) -> pd.DataFrame:
    """
    Coral trend confirmation indicator.

    Original behavior preserved from the pre-Phase-B legacy implementation.
    """
    ema1 = df["close"].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    df[signal_col] = 0
    df.loc[df["close"] > ema2, signal_col] = 1
    df.loc[df["close"] < ema2, signal_col] = -1
    return df


# ---------------------------------------------------------------------------
# Archetype 1: Regime State Machine (hysteresis)
# ---------------------------------------------------------------------------


def c1_regime_sm__binary(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    fast: int = 20,
    slow: int = 50,
    atr_period: int = 14,
    strength_upper: float = 0.30,
    strength_lower: float = 0.10,
    **kwargs,
) -> pd.DataFrame:
    """Regime state machine, binary output: 0 -> hold last direction; seed from sign(strength)."""
    df = df.copy()
    upper = max(strength_upper, strength_lower + 1e-9)
    lower = strength_lower
    close = df["close"].astype(float)
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    atr = _atr_series(df, atr_period)
    raw_strength = ema_fast - ema_slow
    strength = raw_strength / np.clip(atr.replace(0, np.nan).ffill().bfill(), 1e-10, None)

    state = pd.Series(0, index=df.index, dtype=np.float64)
    warmup = max(slow, atr_period)
    for i in range(warmup, len(df)):
        s = strength.iloc[i]
        if pd.isna(s):
            state.iloc[i] = state.iloc[i - 1]
            continue
        if s >= upper:
            state.iloc[i] = 1
        elif s <= -upper:
            state.iloc[i] = -1
        elif abs(s) <= lower:
            state.iloc[i] = 0
        else:
            state.iloc[i] = state.iloc[i - 1]

    out = pd.Series(np.nan, index=df.index, dtype=np.float64)
    last = 1.0 if (strength.iloc[warmup] >= 0) else -1.0
    for i in range(warmup, len(df)):
        st = state.iloc[i]
        if st != 0:
            last = int(st)
        out.iloc[i] = last
    df[signal_col] = out
    return df


def c1_regime_sm__neutral_gate(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    fast: int = 20,
    slow: int = 50,
    atr_period: int = 14,
    strength_upper: float = 0.30,
    strength_lower: float = 0.10,
    **kwargs,
) -> pd.DataFrame:
    """Regime state machine, neutral gate: output internal ternary state {-1, 0, +1}."""
    df = df.copy()
    upper = max(strength_upper, strength_lower + 1e-9)
    lower = strength_lower
    close = df["close"].astype(float)
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    atr = _atr_series(df, atr_period)
    raw_strength = ema_fast - ema_slow
    strength = raw_strength / np.clip(atr.replace(0, np.nan).ffill().bfill(), 1e-10, None)

    state = pd.Series(0, index=df.index, dtype=np.float64)
    warmup = max(slow, atr_period)
    for i in range(warmup, len(df)):
        s = strength.iloc[i]
        if pd.isna(s):
            state.iloc[i] = state.iloc[i - 1]
            continue
        if s >= upper:
            state.iloc[i] = 1
        elif s <= -upper:
            state.iloc[i] = -1
        elif abs(s) <= lower:
            state.iloc[i] = 0
        else:
            state.iloc[i] = state.iloc[i - 1]
    df[signal_col] = state.astype(int)
    return df


# ---------------------------------------------------------------------------
# Archetype 2: Volatility-Conditioned Direction
# ---------------------------------------------------------------------------


def c1_vol_dir__binary(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    fast: int = 20,
    slow: int = 50,
    vol_ema: int = 50,
    vol_mult: float = 1.05,
    atr_period: int = 14,
    **kwargs,
) -> pd.DataFrame:
    """Vol-conditioned direction; binary: state==0 -> hold last; seed from dir_raw."""
    df = df.copy()
    close = df["close"].astype(float)
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    dir_raw = np.sign(ema_fast - ema_slow)
    atr = _atr_series(df, atr_period)
    atr_smooth = _ema_series(atr, vol_ema)
    vol_ok = atr >= (atr_smooth * vol_mult)
    state = np.where(vol_ok, dir_raw, 0)

    out = pd.Series(np.nan, index=df.index, dtype=np.float64)
    warmup = max(slow, vol_ema, atr_period)
    last = 1 if (dir_raw.iloc[warmup] >= 0) else -1
    for i in range(warmup, len(df)):
        st = state[i]
        if st != 0:
            last = int(st)
        out.iloc[i] = last
    df[signal_col] = out
    return df


def c1_vol_dir__neutral_gate(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    fast: int = 20,
    slow: int = 50,
    vol_ema: int = 50,
    vol_mult: float = 1.05,
    atr_period: int = 14,
    **kwargs,
) -> pd.DataFrame:
    """Vol-conditioned direction; neutral gate: output ternary (0 when vol not ok)."""
    df = df.copy()
    close = df["close"].astype(float)
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    dir_raw = np.sign(ema_fast - ema_slow)
    atr = _atr_series(df, atr_period)
    atr_smooth = _ema_series(atr, vol_ema)
    vol_ok = atr >= (atr_smooth * vol_mult)
    state = np.where(vol_ok, dir_raw, 0)
    df[signal_col] = pd.Series(state.astype(int), index=df.index)
    return df


# ---------------------------------------------------------------------------
# Archetype 3: Persistence-Filtered Momentum
# ---------------------------------------------------------------------------


def c1_persist_momo__binary(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    fast: int = 20,
    slow: int = 50,
    confirm_bars: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """Persistence-filtered momentum; binary: hold last non-zero confirmed_dir; seed from dir_raw."""
    df = df.copy()
    close = df["close"].astype(float)
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    m = ema_fast - ema_slow
    dir_raw = np.sign(m)

    confirmed = np.zeros(len(df), dtype=np.float64)
    streak = 0
    streak_sign = 0
    N = max(1, int(confirm_bars))
    warmup = slow
    for i in range(warmup, len(df)):
        d = dir_raw.iloc[i]
        if d == 0:
            confirmed[i] = 0
            streak = 0
            streak_sign = 0
            continue
        if d == streak_sign:
            streak += 1
            if streak >= N:
                confirmed[i] = d
            else:
                confirmed[i] = 0
        else:
            streak = 1
            streak_sign = int(d)
            confirmed[i] = 0

    out = pd.Series(np.nan, index=df.index, dtype=np.float64)
    last = 1 if (dir_raw.iloc[warmup] >= 0) else -1
    for i in range(warmup, len(df)):
        c = confirmed[i]
        if c != 0:
            last = int(c)
        out.iloc[i] = last
    df[signal_col] = out
    return df


def c1_persist_momo__neutral_gate(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    fast: int = 20,
    slow: int = 50,
    confirm_bars: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """Persistence-filtered momentum; neutral gate: output confirmed_dir (can be 0)."""
    df = df.copy()
    close = df["close"].astype(float)
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    m = ema_fast - ema_slow
    dir_raw = np.sign(m)

    confirmed = np.zeros(len(df), dtype=np.float64)
    streak = 0
    streak_sign = 0
    N = max(1, int(confirm_bars))
    warmup = slow
    for i in range(warmup, len(df)):
        d = dir_raw.iloc[i]
        if d == 0:
            confirmed[i] = 0
            streak = 0
            streak_sign = 0
            continue
        if d == streak_sign:
            streak += 1
            if streak >= N:
                confirmed[i] = d
            else:
                confirmed[i] = 0
        else:
            streak = 1
            streak_sign = int(d)
            confirmed[i] = 0
    df[signal_col] = pd.Series(confirmed.astype(int), index=df.index)
    return df


# ---------------------------------------------------------------------------
# Supertrend (canonical C1, independent of legacy_rejected)
# ---------------------------------------------------------------------------


def c1_supertrend(
    df: pd.DataFrame,
    atr_period: int = 10,
    multiplier: float = 3.0,
    signal_col: str = "c1_signal",
    **kwargs,
) -> pd.DataFrame:
    """
    Proper Supertrend indicator implementation.

    Writes df["atr"] and df[signal_col] in {-1, +1}.
    """
    if "close" not in df.columns or "high" not in df.columns or "low" not in df.columns:
        raise ValueError("Expected 'high', 'low', 'close' columns in df")

    # Ensure we have ATR
    if "atr" not in df.columns:
        df = calculate_atr(df.copy(), period=atr_period)

    # Calculate basic upper and lower bands
    hl2 = (df["high"] + df["low"]) / 2.0
    basic_upper = hl2 + multiplier * df["atr"]
    basic_lower = hl2 - multiplier * df["atr"]

    # Initialize final bands and trend
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    trend = pd.Series(1, index=df.index, dtype=int)  # Start bullish

    # Calculate final bands with carry-forward rules
    for i in range(1, len(df)):
        # Upper band: use basic upper unless it's higher than previous and close was above previous upper
        if (
            basic_upper.iloc[i] < final_upper.iloc[i - 1]
            or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # Lower band: use basic lower unless it's lower than previous and close was below previous lower
        if (
            basic_lower.iloc[i] > final_lower.iloc[i - 1]
            or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

    # Determine trend direction
    for i in range(1, len(df)):
        if df["close"].iloc[i] <= final_lower.iloc[i]:
            trend.iloc[i] = -1  # Bearish
        elif df["close"].iloc[i] >= final_upper.iloc[i]:
            trend.iloc[i] = 1  # Bullish
        else:
            trend.iloc[i] = trend.iloc[i - 1]  # Continue previous trend

    # Set signal column - Supertrend follows the trend
    df[signal_col] = trend

    # Ensure we have the atr column for output
    if "atr" not in df.columns:
        df["atr"] = calculate_atr(df.copy(), period=atr_period)["atr"]

    return df


def supertrend(
    df: pd.DataFrame,
    atr_period: int = 10,
    multiplier: float = 3.0,
    signal_col: str = "c1_signal",
    **kwargs,
) -> pd.DataFrame:
    """Alias for c1_supertrend to support short name resolution."""
    return c1_supertrend(
        df,
        atr_period=atr_period,
        multiplier=multiplier,
        signal_col=signal_col,
        **kwargs,
    )

