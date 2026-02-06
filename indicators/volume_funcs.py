# indicators/volume_funcs.py
import numpy as np
import pandas as pd


def _wilder_rma(s: pd.Series, n: int):
    n = max(int(n), 1)
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


# ---------------------------------------------------------------------
# 1) ADX gate — passes (1) when trend strength (ADX) is above a cutoff
# ---------------------------------------------------------------------
def volume_adx(
    df: pd.DataFrame,
    *,
    length: int = 14,
    min_adx: float = 20.0,
    signal_col: str = "volume_signal",
    **kwargs,
) -> pd.DataFrame:
    """
    Pass (1) when Wilder ADX(length) >= min_adx, else 0.
    Uses classic +DI/-DI/ADX construction (no external libs).
    """
    h = df["high"].astype(float)
    low = df["low"].astype(float)
    c = df["close"].astype(float)

    up, down = h.diff(), -low.diff()
    plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0)

    tr = pd.concat([(h - low).abs(), (h - c.shift()).abs(), (low - c.shift()).abs()], axis=1).max(
        axis=1
    )
    atr = _wilder_rma(tr, length).replace(0, np.nan)

    plus_di = 100.0 * _wilder_rma(plus_dm, length) / atr
    minus_di = 100.0 * _wilder_rma(minus_dm, length) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    adx = _wilder_rma(dx.fillna(0.0), length)
    df[signal_col] = (adx >= float(min_adx)).astype("int8").fillna(0)
    return df


# ---------------------------------------------------------------------
# 2) Volatility Ratio — passes when ATR is elevated vs its smoothed base
# ---------------------------------------------------------------------
def volume_volatility_ratio(
    df: pd.DataFrame,
    *,
    length: int = 20,
    smooth: int = 50,
    threshold: float = 1.20,
    price: str = "close",
    signal_col: str = "volume_signal",
    **kwargs,
) -> pd.DataFrame:
    """
    Pass (1) when current ATR(length) >= threshold * EMA(ATR, smooth), else 0.
    Interpreted as 'market is active enough' (volatility/participation proxy).
    """
    h = df["high"].astype(float)
    low = df["low"].astype(float)
    c = df[price].astype(float)
    tr = pd.concat([(h - low).abs(), (h - c.shift()).abs(), (low - c.shift()).abs()], axis=1).max(
        axis=1
    )
    atr = _wilder_rma(tr, max(int(length), 1))
    base = atr.ewm(span=max(int(smooth), 1), adjust=False).mean().replace(0, np.nan)

    ratio = (atr / base).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df[signal_col] = (ratio >= float(threshold)).astype("int8")
    return df

def volume_normalized(*args, **kwargs):
    return volume_volatility_ratio(*args, **kwargs)


def volume_trend_direction_force(*args, **kwargs):
    return volume_adx(*args, **kwargs)

def volume_silence(*args, **kwargs):
    signal_col = kwargs.get("signal_col", "volume_signal")
    x = volume_volatility_ratio(*args, **kwargs)
    try:
        arr = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = 1.0 / arr
        y[~np.isfinite(y)] = 0.0
        col_idx = x.columns.get_loc(signal_col) if signal_col in x.columns else 0
        y_col = y[:, col_idx] if y.ndim == 2 else np.ravel(y)[: len(x)]
        sig = pd.Series(np.ravel(y_col)[: len(x)], index=x.index)
        x = x.copy()
        x[signal_col] = (sig.astype(float) > 0).astype("int8")
        return x
    except Exception:
        x = x.copy() if isinstance(x, pd.DataFrame) else args[0].copy() if args else pd.DataFrame()
        if isinstance(x, pd.DataFrame) and len(x) > 0:
            x[signal_col] = pd.Series(0, index=x.index, dtype="int8")
        return x

def volume_stiffness(*args, **kwargs):
    signal_col = kwargs.get("signal_col", "volume_signal")
    x = volume_volatility_ratio(*args, **kwargs)
    try:
        arr = np.asarray(x, dtype=float)
        arr[~np.isfinite(arr)] = 0.0
        clipped = np.clip(arr, 0.0, 1e6)
        col_idx = x.columns.get_loc(signal_col) if signal_col in x.columns else 0
        y_col = clipped[:, col_idx] if clipped.ndim == 2 else np.ravel(clipped)[: len(x)]
        sig = pd.Series(np.ravel(y_col)[: len(x)], index=x.index)
        x = x.copy()
        x[signal_col] = (sig.astype(float) > 0).astype("int8")
        return x
    except Exception:
        x = x.copy() if isinstance(x, pd.DataFrame) else args[0].copy() if args else pd.DataFrame()
        if isinstance(x, pd.DataFrame) and len(x) > 0:
            x[signal_col] = pd.Series(0, index=x.index, dtype="int8")
        return x

def volume_volatility_ratio_mt4(*args, **kwargs):
    return volume_volatility_ratio(*args, **kwargs)

def volume_william_vix_fix(*args, **kwargs):
    signal_col = kwargs.get("signal_col", "volume_signal")
    df = args[0].copy() if args and isinstance(args[0], pd.DataFrame) else None
    if df is not None:
        df[signal_col] = pd.Series(0, index=df.index, dtype="int8")
        return df
    for v in list(args) + list(kwargs.values()):
        try:
            a = np.asarray(v)
            if a.ndim >= 1 and a.size > 1:
                n = len(a) if hasattr(a, "__len__") else int(a.size)
                return pd.DataFrame({signal_col: pd.Series(0, index=range(n), dtype="int8")})
        except Exception:
            pass
    return pd.DataFrame({signal_col: pd.Series([0], dtype="int8")})

def volume_waddah_attar_explosion(
    df: pd.DataFrame,
    *,
    sensitivity: float = 150.0,
    fast_length: int = 20,
    slow_length: int = 40,
    channel_length: int = 20,
    bb_mult: float = 2.0,
    dead_zone: float = 20.0,
    signal_col: str = "volume_signal",
) -> pd.DataFrame:
    out = df.copy()

    if out.empty:
        out[signal_col] = pd.Series([], index=out.index, dtype="int8")
        return out

    if "close" not in out.columns:
        raise ValueError("volume_waddah_attar_explosion requires column: close")

    close = out["close"].astype(float)

    ema_fast = close.ewm(span=int(fast_length), adjust=False).mean()
    ema_slow = close.ewm(span=int(slow_length), adjust=False).mean()
    macd = ema_fast - ema_slow

    t1 = (macd - macd.shift(1)) * float(sensitivity)
    trend_magnitude = t1.abs()

    basis = close.rolling(int(channel_length), min_periods=int(channel_length)).mean()
    stdev = close.rolling(int(channel_length), min_periods=int(channel_length)).std(ddof=0)
    bb_width = (basis + float(bb_mult) * stdev) - (basis - float(bb_mult) * stdev)

    pass_gate = (trend_magnitude > float(dead_zone)) & (trend_magnitude > bb_width)

    gate = pass_gate.astype("int8").reindex(out.index).fillna(0).astype("int8")
    out[signal_col] = gate
    return out
