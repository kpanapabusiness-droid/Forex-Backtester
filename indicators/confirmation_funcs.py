# indicators/confirmation_funcs.py — C1 confirmation indicators (Phase B.1 archetypes + discovery).
# All C1 functions: (df, *, signal_col="c1_signal", **kwargs) -> df; write {-1,0,+1} to signal_col.

from __future__ import annotations

import numpy as np
import pandas as pd


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
