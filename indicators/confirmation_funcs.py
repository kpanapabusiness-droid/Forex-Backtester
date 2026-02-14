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
    n = len(df)
    warmup = max(slow, atr_period)
    if n <= warmup:
        df[signal_col] = 0
        return df

    upper = max(strength_upper, strength_lower + 1e-9)
    lower = strength_lower
    close = df["close"].astype(float)
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    atr = _atr_series(df, atr_period)
    raw_strength = ema_fast - ema_slow
    strength = raw_strength / np.clip(atr.replace(0, np.nan).ffill().bfill(), 1e-10, None)

    state = pd.Series(0, index=df.index, dtype=np.float64)
    for i in range(warmup, n):
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

    out = pd.Series(0.0, index=df.index, dtype=np.float64)
    last = 1.0 if (strength.iloc[warmup] >= 0) else -1.0
    for i in range(warmup, n):
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
    n = len(df)
    warmup = max(slow, vol_ema, atr_period)
    if n <= warmup:
        df[signal_col] = 0
        return df

    close = df["close"].astype(float)
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    dir_raw = np.sign(ema_fast - ema_slow)
    atr = _atr_series(df, atr_period)
    atr_smooth = _ema_series(atr, vol_ema)
    vol_ok = atr >= (atr_smooth * vol_mult)
    state = np.where(vol_ok, dir_raw, 0)

    out = pd.Series(0.0, index=df.index, dtype=np.float64)
    last = 1 if (dir_raw.iloc[warmup] >= 0) else -1
    for i in range(warmup, n):
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
    n = len(df)
    warmup = slow
    if n <= warmup:
        df[signal_col] = 0
        return df

    close = df["close"].astype(float)
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    m = ema_fast - ema_slow
    dir_raw = np.sign(m)

    confirmed = np.zeros(n, dtype=np.float64)
    streak = 0
    streak_sign = 0
    N = max(1, int(confirm_bars))
    for i in range(warmup, n):
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

    out = pd.Series(0.0, index=df.index, dtype=np.float64)
    last = 1 if (dir_raw.iloc[warmup] >= 0) else -1
    for i in range(warmup, n):
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


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Causal percentile rank (0..1) of each value within its trailing window."""
    if window < 2 or len(series) < window:
        return pd.Series(np.nan, index=series.index)

    def _pctile(x: np.ndarray) -> float:
        if len(x) < window or not np.any(np.isfinite(x)):
            return np.nan
        val = x[-1]
        if not np.isfinite(val):
            return np.nan
        finite = x[np.isfinite(x)]
        n = len(finite)
        if n == 0:
            return np.nan
        rank = np.sum(finite <= val)
        return (rank - 1) / max(n - 1, 1) if n > 1 else 0.0

    return series.rolling(window=window, min_periods=window).apply(_pctile, raw=True)


def c1_compression_escape_state_machine(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    L_p: int = 252,
    L_box: int = 20,
    q_atr: float = 0.40,
    q_rng: float = 0.40,
    q_box: float = 0.35,
    W_enter: int = 10,
    N_enter: int = 6,
    W_exit: int = 10,
    N_exit: int = 7,
    K_max: int = 30,
    alpha: float = 0.9,
    beta: float = 1.10,
    L_slow: int = 50,
    gamma: float = 0.55,
    cooldown_bars: int = 15,
    atr_period: int = 14,
    **kwargs,
) -> pd.DataFrame:
    """
    C1 Family A v2: Compression → Escape state machine (CEB v2).

    State machine: compression box (low ATR/Range/box-width percentiles), enter when
    N_enter of W_enter bars compressed, exit when N_exit of W_exit bars not compressed.
    Escape: close breaks prior box bounds with expansion confirmation (2 of 3 conditions).
    """
    df = df.copy()
    n = len(df)
    atr = _atr_series(df, atr_period)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    rng = high - low

    atr_pct = _rolling_percentile_rank(atr, L_p)
    rng_pct = _rolling_percentile_rank(rng, L_p)
    box_hi = high.rolling(L_box, min_periods=L_box).max()
    box_lo = low.rolling(L_box, min_periods=L_box).min()
    box_width = box_hi - box_lo
    box_width_pct = _rolling_percentile_rank(box_width, L_p)

    comp = ((atr_pct <= q_atr) & (rng_pct <= q_rng) & (box_width_pct <= q_box)).fillna(False)

    comp_rolling = comp.rolling(W_enter, min_periods=W_enter).sum()
    enter_condition = comp_rolling >= N_enter
    exit_comp_false = (~comp).rolling(W_exit, min_periods=W_exit).sum()
    exit_condition = exit_comp_false >= N_exit

    prior_box_hi = box_hi.shift(1)
    prior_box_lo = box_lo.shift(1)

    # --- IGNITION ISOLATION TEST ---
    # Temporary: disable expansion confirmation filters
    escape_long = (close > prior_box_hi).fillna(False)
    escape_short = (close < prior_box_lo).fillna(False)

    warmup = max(L_p, L_box, W_enter, W_exit, L_slow, atr_period)
    signal = np.zeros(n, dtype=np.int8)

    in_comp = False
    comp_start_idx = 0
    cooldown = 0

    for t in range(n):
        if cooldown > 0:
            cooldown -= 1
            signal[t] = 0
            continue

        if not in_comp:
            if t >= warmup and enter_condition.iloc[t]:
                in_comp = True
                comp_start_idx = t
            signal[t] = 0
            continue

        if (t - comp_start_idx) > K_max:
            in_comp = False
            signal[t] = 0
            continue

        le = escape_long.iloc[t] if t > 0 else False
        se = escape_short.iloc[t] if t > 0 else False

        if le:
            signal[t] = 1
            in_comp = False
            cooldown = cooldown_bars
            continue
        if se:
            signal[t] = -1
            in_comp = False
            cooldown = cooldown_bars
            continue
        if exit_condition.iloc[t]:
            in_comp = False
        signal[t] = 0

    df[signal_col] = pd.Series(signal, index=df.index)
    return df


def c1_compression_escape_ratio_state_machine(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    L_slow: int = 50,
    r_atr: float = 0.85,
    r_rng: float = 0.85,
    r_box: float = 1.10,
    M_enter: int = 4,
    M_exit: int = 3,
    K_max: int = 40,
    cooldown_bars: int = 10,
    atr_period: int = 14,
    **kwargs,
) -> pd.DataFrame:
    """
    C1 CEB v3: Compression regime via ratio (not percentile). Episode box while in compression.
    Escape: close breaks prior box bounds (simple breakout). Cooldown after emission.
    """
    df = df.copy()
    n = len(df)
    atr = _atr_series(df, atr_period)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    rng = high - low

    atr_slow = atr.rolling(L_slow, min_periods=L_slow).mean()
    rng_slow = rng.rolling(L_slow, min_periods=L_slow).mean()
    denom_atr = np.clip(atr_slow.replace(0, np.nan).ffill().bfill(), 1e-12, None)
    denom_rng = np.clip(rng_slow.replace(0, np.nan).ffill().bfill(), 1e-12, None)

    atr_ratio = atr / denom_atr
    range_ratio = rng / denom_rng
    comp_t = (atr_ratio <= r_atr) & (range_ratio <= r_rng)
    comp_t = comp_t.fillna(False)

    warmup = max(L_slow, atr_period)
    signal = np.zeros(n, dtype=np.int8)

    in_comp = False
    comp_start_idx = 0
    comp_streak = 0
    box_hi = 0.0
    box_lo = 0.0
    box_exceed_streak = 0
    cooldown = 0

    for t in range(n):
        if cooldown > 0:
            cooldown -= 1
            signal[t] = 0
            continue

        if not in_comp:
            if comp_t.iloc[t]:
                comp_streak += 1
                if comp_streak >= M_enter and t >= warmup:
                    in_comp = True
                    comp_start_idx = t
                    box_hi = high.iloc[t]
                    box_lo = low.iloc[t]
                    comp_streak = 0
                    box_exceed_streak = 0
            else:
                comp_streak = 0
            signal[t] = 0
            continue

        box_width = box_hi - box_lo
        rng_slow_t = denom_rng.iloc[t]
        box_ratio_t = box_width / np.clip(rng_slow_t, 1e-12, None)

        if box_ratio_t > r_box:
            box_exceed_streak += 1
            if box_exceed_streak >= M_exit:
                in_comp = False
                comp_streak = 0
                box_exceed_streak = 0
                signal[t] = 0
                continue
        else:
            box_exceed_streak = 0

        if (t - comp_start_idx) > K_max:
            in_comp = False
            comp_streak = 0
            box_exceed_streak = 0
            signal[t] = 0
            continue

        box_hi_prev = box_hi
        box_lo_prev = box_lo

        escape_long = close.iloc[t] > box_hi_prev
        escape_short = close.iloc[t] < box_lo_prev

        if escape_long:
            signal[t] = 1
            in_comp = False
            comp_streak = 0
            box_exceed_streak = 0
            cooldown = cooldown_bars
            continue
        if escape_short:
            signal[t] = -1
            in_comp = False
            comp_streak = 0
            box_exceed_streak = 0
            cooldown = cooldown_bars
            continue

        box_hi = max(box_hi, high.iloc[t])
        box_lo = min(box_lo, low.iloc[t])
        signal[t] = 0

    df[signal_col] = pd.Series(signal, index=df.index)
    return df


def c1_compression_expansion_breakout(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    L_p: int = 120,
    L_bb: int = 40,
    bb_k: float = 2.0,
    M: int = 12,
    N_c: int = 8,
    q_atr: float = 0.2,
    q_rng: float = 0.2,
    q_bw: float = 0.2,
    alpha: float = 1.2,
    L_atr_slow: int = 120,
    beta: float = 1.15,
    gamma: float = 0.65,
    L_brk: int = 20,
    atr_period: int = 14,
    **kwargs,
) -> pd.DataFrame:
    """
    C1 Family A: Compression → Expansion Breakout (CEB).

    Compression: ATR, Range, and BB width at low percentiles; at least N_c of last M bars compressed.
    Ignition: Range >= alpha*ATR, ATR ratio >= beta, directional filter (close location + breakout).
    """
    df = df.copy()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    rng = high - low

    atr = _atr_series(df, atr_period)
    atr_slow = atr.rolling(L_atr_slow, min_periods=L_atr_slow).mean()

    atr_pct = _rolling_percentile_rank(atr, L_p)
    rng_pct = _rolling_percentile_rank(rng, L_p)

    sma_close = close.rolling(L_bb, min_periods=L_bb).mean()
    std_close = close.rolling(L_bb, min_periods=L_bb).std(ddof=0)
    bb_upper = sma_close + bb_k * std_close
    bb_lower = sma_close - bb_k * std_close
    bb_width = (bb_upper - bb_lower) / np.clip(sma_close.replace(0, np.nan).ffill().bfill(), 1e-12, None)
    bw_pct = _rolling_percentile_rank(bb_width, L_p)

    compressed = (
        (atr_pct <= q_atr) & (rng_pct <= q_rng) & (bw_pct <= q_bw)
    ).fillna(False)

    comp_count = compressed.rolling(M, min_periods=M).sum()
    in_compression = comp_count >= N_c

    atr_ratio = atr / np.clip(atr_slow.replace(0, np.nan).ffill().bfill(), 1e-12, None)
    range_vs_atr = rng >= (alpha * atr)
    ignition_base = range_vs_atr & (atr_ratio >= beta)

    prior_hhv = high.shift(1).rolling(L_brk, min_periods=L_brk).max()
    prior_llv = low.shift(1).rolling(L_brk, min_periods=L_brk).min()
    denom = np.clip(rng.replace(0, np.nan), 1e-12, None)
    close_loc_long = (close - low) / denom
    close_loc_short = (high - close) / denom

    long_trigger = (
        in_compression
        & ignition_base
        & (close_loc_long >= gamma)
        & (close > prior_hhv)
    )
    short_trigger = (
        in_compression
        & ignition_base
        & (close_loc_short >= gamma)
        & (close < prior_llv)
    )

    out = pd.Series(0, index=df.index, dtype=np.int8)
    out.loc[long_trigger] = 1
    out.loc[short_trigger] = -1
    df[signal_col] = out
    return df


def c1_volatility_regime_impulse(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    L_rv: int = 20,
    L_slow: int = 120,
    L_f: int = 10,
    L_s: int = 80,
    theta_rv: float = 1.25,
    theta_atr: float = 1.15,
    K: int = 20,
    b_min: float = 0.5,
    gamma: float = 0.7,
    L_dc: int = 20,
    **kwargs,
) -> pd.DataFrame:
    """
    C1 Family B: Volatility Regime Shift + Directional Impulse (VRSI).

    Regime shift: RV_ratio or ATR VR spikes above threshold; prior K bars had low vol (median < 1).
    Impulse: body fraction, close location, and breakout confirmation.
    """
    df = df.copy()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    rng = high - low

    prev_close = close.shift(1)
    safe_prev = np.clip(prev_close.replace(0, np.nan).ffill().bfill(), 1e-12, None)
    log_ret = np.log(close / safe_prev)

    r_sq = log_ret**2
    rv = np.sqrt(
        r_sq.rolling(L_rv, min_periods=L_rv).mean().replace(0, np.nan).ffill().bfill()
    )
    rv_slow = rv.rolling(L_slow, min_periods=L_slow).mean()
    rv_ratio = rv / np.clip(rv_slow.replace(0, np.nan).ffill().bfill(), 1e-12, None)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_fast = tr.rolling(L_f, min_periods=L_f).mean()
    atr_slow = tr.rolling(L_s, min_periods=L_s).mean()
    vr = atr_fast / np.clip(atr_slow.replace(0, np.nan).ffill().bfill(), 1e-12, None)

    vol_spike = (rv_ratio >= theta_rv) | (vr >= theta_atr)
    rv_med = rv_ratio.rolling(K, min_periods=K).median()
    vr_med = vr.rolling(K, min_periods=K).median()
    prior_low_vol = (rv_med < 1) & (vr_med < 1)
    regime_shift = vol_spike & prior_low_vol.fillna(False)

    body_frac = np.abs(close - open_) / np.clip(rng.replace(0, np.nan), 1e-12, None)
    impulse_body = body_frac >= b_min
    denom = np.clip(rng.replace(0, np.nan), 1e-12, None)
    close_loc_long = (close - low) / denom
    close_loc_short = (high - close) / denom

    prior_hhv = high.shift(1).rolling(L_dc, min_periods=L_dc).max()
    prior_llv = low.shift(1).rolling(L_dc, min_periods=L_dc).min()

    long_impulse = impulse_body & (close_loc_long >= gamma) & (close > prior_hhv)
    short_impulse = impulse_body & (close_loc_short >= gamma) & (close < prior_llv)

    long_trigger = regime_shift & long_impulse
    short_trigger = regime_shift & short_impulse

    out = pd.Series(0, index=df.index, dtype=np.int8)
    out.loc[long_trigger] = 1
    out.loc[short_trigger] = -1
    df[signal_col] = out
    return df


def c1_lsr_v2(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    variant: str = "A_pin",
    lookback_n: int = 20,
    sweep_atr: float = 0.2,
    wick_min_frac: float = 0.65,
    body_max_frac: float = 0.35,
    close_pos_min: float = 0.75,
    reclaim_frac: float = 0.1,
    cooldown_bars: int = 5,
    confirm_mode: str = "break_high",
    no_resweep: int = 1,
    cluster_k: int = 2,
    vol_expand_atr: float = 1.2,
    range_cap_atr: float = 3.0,
    min_range_atr: float = 0.0,
    atr_period: int = 14,
    **kwargs,
) -> pd.DataFrame:
    """
    C1 LSR v2: Liquidity Sweep Rejection with four variants.

    A_pin: single-bar sweep + rejection on same bar close.
    B_confirm: sweep/rejection bar then 1-bar follow-through; signal at confirm bar.
    C_cluster: allow cluster_k bars to probe beyond prior extreme; signal on latest reclaim.
    D_volexp: like A_pin but requires range >= vol_expand_atr * ATR(prev).
    """
    df = df.copy()
    n = len(df)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    rng = high - low

    atr = _atr_series(df, atr_period)
    atr_prev = atr.shift(1)
    atr_prev = np.clip(atr_prev.replace(0, np.nan).ffill().bfill(), 1e-12, None)

    prior_swing_high = high.shift(1).rolling(lookback_n, min_periods=lookback_n).max()
    prior_swing_low = low.shift(1).rolling(lookback_n, min_periods=lookback_n).min()

    denom_rng = np.clip(rng.replace(0, np.nan), 1e-12, None)
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)
    body = np.abs(close - open_)
    close_loc_long = (close - low) / denom_rng
    close_loc_short = (high - close) / denom_rng

    wick_ok_long = (lower_wick / denom_rng) >= wick_min_frac
    wick_ok_short = (upper_wick / denom_rng) >= wick_min_frac
    body_ok = (body / denom_rng) <= body_max_frac
    close_ok_long = close_loc_long >= close_pos_min
    close_ok_short = close_loc_short >= close_pos_min

    sweep_long_raw = (low < prior_swing_low) & (close > prior_swing_low)
    sweep_short_raw = (high > prior_swing_high) & (close < prior_swing_high)

    sweep_depth_long = (prior_swing_low - low) >= (sweep_atr * atr_prev)
    sweep_depth_short = (high - prior_swing_high) >= (sweep_atr * atr_prev)

    reclaim_long = (close - prior_swing_low) >= (reclaim_frac * rng)
    reclaim_short = (prior_swing_high - close) >= (reclaim_frac * rng)

    if range_cap_atr > 0:
        cap_ok = rng <= (range_cap_atr * atr_prev)
        sweep_long_raw = sweep_long_raw & cap_ok.fillna(True)
        sweep_short_raw = sweep_short_raw & cap_ok.fillna(True)
    if min_range_atr > 0:
        floor_ok = rng >= (min_range_atr * atr_prev)
        sweep_long_raw = sweep_long_raw & floor_ok.fillna(True)
        sweep_short_raw = sweep_short_raw & floor_ok.fillna(True)

    base_long = (
        sweep_long_raw
        & sweep_depth_long.fillna(False)
        & wick_ok_long.fillna(False)
        & body_ok.fillna(False)
        & close_ok_long.fillna(False)
        & reclaim_long.fillna(False)
    )
    base_short = (
        sweep_short_raw
        & sweep_depth_short.fillna(False)
        & wick_ok_short.fillna(False)
        & body_ok.fillna(False)
        & close_ok_short.fillna(False)
        & reclaim_short.fillna(False)
    )

    if variant == "D_volexp":
        vol_ok = rng >= (vol_expand_atr * atr_prev)
        base_long = base_long & vol_ok.fillna(False)
        base_short = base_short & vol_ok.fillna(False)

    warmup = lookback_n + atr_period + 5
    signal = np.zeros(n, dtype=np.int8)
    cd_remaining = 0

    if variant == "A_pin" or variant == "D_volexp":
        for i in range(n):
            if cd_remaining > 0:
                cd_remaining -= 1
                signal[i] = 0
                continue
            if i < warmup:
                signal[i] = 0
                continue
            if base_long.iloc[i]:
                signal[i] = 1
                cd_remaining = cooldown_bars
            elif base_short.iloc[i]:
                signal[i] = -1
                cd_remaining = cooldown_bars
            else:
                signal[i] = 0

    elif variant == "B_confirm":
        for i in range(n):
            if cd_remaining > 0:
                cd_remaining -= 1
                signal[i] = 0
                continue
            if i < warmup or i < 1:
                signal[i] = 0
                continue
            if base_long.iloc[i - 1]:
                if confirm_mode == "break_high":
                    conf_ok = close.iloc[i] > high.iloc[i - 1]
                else:
                    mid = (high.iloc[i - 1] + low.iloc[i - 1]) / 2
                    conf_ok = close.iloc[i] > mid
                if no_resweep and low.iloc[i] < prior_swing_low.iloc[i]:
                    conf_ok = False
                if conf_ok:
                    signal[i] = 1
                    cd_remaining = cooldown_bars
                    continue
            if base_short.iloc[i - 1]:
                if confirm_mode == "break_high":
                    conf_ok = close.iloc[i] < low.iloc[i - 1]
                else:
                    mid = (high.iloc[i - 1] + low.iloc[i - 1]) / 2
                    conf_ok = close.iloc[i] < mid
                if no_resweep and high.iloc[i] > prior_swing_high.iloc[i]:
                    conf_ok = False
                if conf_ok:
                    signal[i] = -1
                    cd_remaining = cooldown_bars
                    continue
            signal[i] = 0

    elif variant == "C_cluster":
        cluster_bars_long = np.zeros(n, dtype=int)
        cluster_bars_short = np.zeros(n, dtype=int)
        for i in range(1, n):
            psl = prior_swing_low.iloc[i]
            psh = prior_swing_high.iloc[i]
            if pd.notna(psl) and low.iloc[i] < psl:
                cluster_bars_long[i] = cluster_bars_long[i - 1] + 1
            else:
                cluster_bars_long[i] = 0
            if pd.notna(psh) and high.iloc[i] > psh:
                cluster_bars_short[i] = cluster_bars_short[i - 1] + 1
            else:
                cluster_bars_short[i] = 0

        for i in range(n):
            if cd_remaining > 0:
                cd_remaining -= 1
                signal[i] = 0
                continue
            if i < warmup:
                signal[i] = 0
                continue
            if base_long.iloc[i] and 1 <= cluster_bars_long[i] <= cluster_k:
                signal[i] = 1
                cd_remaining = cooldown_bars
            elif base_short.iloc[i] and 1 <= cluster_bars_short[i] <= cluster_k:
                signal[i] = -1
                cd_remaining = cooldown_bars
            else:
                signal[i] = 0

    else:
        for i in range(n):
            if cd_remaining > 0:
                cd_remaining -= 1
                signal[i] = 0
                continue
            if i < warmup:
                signal[i] = 0
                continue
            if base_long.iloc[i]:
                signal[i] = 1
                cd_remaining = cooldown_bars
            elif base_short.iloc[i]:
                signal[i] = -1
                cd_remaining = cooldown_bars
            else:
                signal[i] = 0

    df[signal_col] = pd.Series(signal, index=df.index)
    return df


def c1_liquidity_sweep_rejection(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    L_sw: int = 20,
    gamma: float = 0.65,
    alpha: float = 1.1,
    use_expansion_filter: bool = True,
    use_compression_filter: bool = False,
    L_p: int = 120,
    q_atr: float = 0.25,
    q_rng: float = 0.25,
    atr_period: int = 14,
    **kwargs,
) -> pd.DataFrame:
    """
    C1 Family D: Liquidity Sweep + Rejection (LSR).

    Swing detection (causal): swing high/low when bar exceeds prior L_sw bars.
    Long: sweep below prior swing low then close above; rejection strength filter.
    Short: sweep above prior swing high then close below.
    Optional expansion (range >= alpha*ATR) and compression preconditions.
    """
    df = df.copy()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    rng = high - low

    prior_high_max = high.shift(1).rolling(L_sw, min_periods=L_sw).max()
    prior_low_min = low.shift(1).rolling(L_sw, min_periods=L_sw).min()
    is_swing_high = high > prior_high_max
    is_swing_low = low < prior_low_min

    swing_high_val = pd.Series(np.where(is_swing_high, high, np.nan), index=df.index)
    swing_low_val = pd.Series(np.where(is_swing_low, low, np.nan), index=df.index)
    prior_swing_high = swing_high_val.ffill().shift(1)
    prior_swing_low = swing_low_val.ffill().shift(1)

    sweep_long = (low < prior_swing_low) & (close > prior_swing_low)
    sweep_short = (high > prior_swing_high) & (close < prior_swing_high)

    denom = np.clip(rng.replace(0, np.nan), 1e-12, None)
    close_location = (close - low) / denom
    reject_long = close_location >= gamma
    reject_short = close_location <= (1.0 - gamma)

    base_long = sweep_long & reject_long
    base_short = sweep_short & reject_short

    if use_expansion_filter:
        atr = _atr_series(df, atr_period)
        expansion_ok = rng >= (alpha * atr)
        base_long = base_long & expansion_ok
        base_short = base_short & expansion_ok

    if use_compression_filter:
        atr = _atr_series(df, atr_period)
        atr_pct = _rolling_percentile_rank(atr, L_p)
        rng_pct = _rolling_percentile_rank(rng, L_p)
        compressed = (atr_pct <= q_atr) & (rng_pct <= q_rng)
        base_long = base_long & compressed.fillna(False)
        base_short = base_short & compressed.fillna(False)

    long_trigger = base_long
    short_trigger = base_short

    out = pd.Series(0, index=df.index, dtype=np.int8)
    out.loc[long_trigger] = 1
    out.loc[short_trigger] = -1
    df[signal_col] = out
    return df


def c1_pressure_overlap_decay(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    L_o: int = 10,
    o_min: float = 0.6,
    L_p: int = 120,
    q_rng: float = 0.25,
    L_e: int = 5,
    eta: float = 0.65,
    o_drop: float = 0.2,
    alpha: float = 1.1,
    atr_period: int = 14,
    **kwargs,
) -> pd.DataFrame:
    """
    C1 Family C: Pressure Buildup via Overlap Decay (PBO).

    Overlap = shared range with prior bar; overlap_ma measures buildup.
    Compression: high overlap_ma + low range percentile.
    Ignition: overlap drops below o_drop + range expands vs ATR.
    Edge creep bias (CLR_ma) determines long/short.
    """
    df = df.copy()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    rng = high - low

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    overlap_top = np.minimum(high, prev_high)
    overlap_bot = np.maximum(low, prev_low)
    overlap_len = np.maximum(0, overlap_top - overlap_bot)
    denom = np.clip(rng.replace(0, np.nan), 1e-12, None)
    overlap_t = overlap_len / denom
    overlap_t = overlap_t.fillna(0).clip(0, 1)

    overlap_ma = overlap_t.rolling(L_o, min_periods=L_o).mean()
    rng_pct = _rolling_percentile_rank(rng, L_p)
    compression = (overlap_ma >= o_min) & (rng_pct <= q_rng)
    compression = compression.fillna(False)

    clr = (close - low) / denom
    clr = clr.fillna(0.5).clip(0, 1)
    clr_ma = clr.rolling(L_e, min_periods=L_e).mean()
    long_bias = clr_ma >= eta
    short_bias = clr_ma <= (1.0 - eta)

    atr = _atr_series(df, atr_period)
    ignition = (overlap_t <= o_drop) & (rng >= (alpha * atr))
    ignition = ignition.fillna(False)

    comp_prior = compression.shift(1).fillna(False)

    long_trigger = comp_prior & ignition & long_bias
    short_trigger = comp_prior & ignition & short_bias

    out = pd.Series(0, index=df.index, dtype=np.int8)
    out.loc[long_trigger] = 1
    out.loc[short_trigger] = -1
    df[signal_col] = out
    return df


def c1_channel_escape_gradient(
    df: pd.DataFrame,
    *,
    signal_col: str = "c1_signal",
    L_va: int = 60,
    q: float = 0.8,
    L_p: int = 120,
    q_va: float = 0.25,
    g1: int = 3,
    g2: int = 10,
    gamma: float = 0.65,
    **kwargs,
) -> pd.DataFrame:
    """
    C1 Family E: Channel Escape with Expansion Gradient (CEG).

    Value channel from rolling quantiles; compression when width percentile low.
    Expansion gradient: short-range SMA > long-range SMA.
    Escape: close breaks above/below prior channel; close location filter.
    """
    df = df.copy()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    rng = high - low

    va_hi = close.rolling(L_va, min_periods=L_va).quantile(q)
    va_lo = close.rolling(L_va, min_periods=L_va).quantile(1.0 - q)
    va_width = va_hi - va_lo
    va_width_pct = _rolling_percentile_rank(va_width, L_p)
    compression = (va_width_pct <= q_va).fillna(False)

    sma_rng_g1 = rng.rolling(g1, min_periods=g1).mean()
    sma_rng_g2 = rng.rolling(g2, min_periods=g2).mean()
    gradient = sma_rng_g1 > sma_rng_g2
    gradient = gradient.fillna(False)

    prior_va_hi = va_hi.shift(1)
    prior_va_lo = va_lo.shift(1)
    escape_long = close > prior_va_hi
    escape_short = close < prior_va_lo

    denom = np.clip(rng.replace(0, np.nan), 1e-12, None)
    close_loc = (close - low) / denom
    close_loc_long = close_loc >= gamma
    close_loc_short = close_loc <= (1.0 - gamma)

    long_trigger = compression & gradient & escape_long & close_loc_long.fillna(False)
    short_trigger = compression & gradient & escape_short & close_loc_short.fillna(False)

    out = pd.Series(0, index=df.index, dtype=np.int8)
    out.loc[long_trigger] = 1
    out.loc[short_trigger] = -1
    df[signal_col] = out
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

