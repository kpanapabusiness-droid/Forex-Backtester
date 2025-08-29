# =============================================
# indicators/baseline_funcs.py
# Baseline indicators (NNFX-ready)
#
# Contract (for every function):
#   def baseline_<name>(df, *, signal_col="baseline_signal", **kwargs) -> pd.DataFrame
# Must return:
#   - df["baseline"]            : numeric price-series baseline
#   - df[signal_col] ∈ {+1, 0, -1} : +1 if close > baseline, -1 if close < baseline, else 0
# =============================================

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------- helpers ----------


def set_signal(
    df: pd.DataFrame, baseline_col: str, signal_col: str = "baseline_signal"
) -> pd.DataFrame:
    """Set {+1,0,-1} based on close vs baseline."""
    sig = np.where(
        df["close"] > df[baseline_col],
        1,
        np.where(df["close"] < df[baseline_col], -1, 0),
    )
    df[signal_col] = sig.astype("int8")
    return df


# indicators/baseline_funcs.py

# If you already have this helper elsewhere in this file, keep ONE copy.
# It must accept (series, length).
# indicators/baseline_funcs.py


def ema(
    x,
    length: int | None = None,
    *,
    period: int | None = None,
    signal_col: str = "baseline_signal",
    **kwargs,
):
    """
    Dual-use EMA:

    1) Helper mode (classic):
         ema(series: pd.Series, length: int) -> pd.Series

    2) Indicator mode (legacy-safe):
         ema(df: pd.DataFrame, *, period: int=200, signal_col="baseline_signal", **kwargs) -> pd.DataFrame
       - Writes df["baseline"] and df[signal_col] ∈ {-1,0,+1}

    This shim lets older resolvers that accidentally import 'ema' (instead of 'baseline_ema')
    continue to work, while keeping the helper behavior intact.
    """
    # ---- Helper mode: ema(series, length) ----
    if isinstance(x, pd.Series):
        L = int(length if length is not None else (period if period is not None else 1))
        L = max(L, 1)
        alpha = 2.0 / (L + 1.0)
        return x.astype(float).ewm(alpha=alpha, adjust=False).mean()

    # ---- Indicator mode: ema(df, *, period=..., signal_col=...) ----
    if isinstance(x, pd.DataFrame):
        df = x.copy()
        if "close" not in df.columns:
            raise ValueError("EMA baseline requires a 'close' column.")
        L = int(period if period is not None else (length if length is not None else 200))
        L = max(L, 1)
        alpha = 2.0 / (L + 1.0)
        base = df["close"].astype(float).ewm(alpha=alpha, adjust=False).mean()
        df["baseline"] = base
        df[signal_col] = np.where(
            df["close"] > base, 1, np.where(df["close"] < base, -1, 0)
        ).astype("int8")
        return df

    raise TypeError("ema() expected pd.Series or pd.DataFrame as first argument.")


def baseline_ema(
    df: pd.DataFrame,
    *,
    period: int = 200,
    length: int | None = None,
    price: str = "close",
    signal_col: str = "baseline_signal",
    **kwargs,
) -> pd.DataFrame:
    """
    Baseline = EMA(period) on close.
    Outputs:
      - df['baseline']        : float EMA series
      - df[signal_col] ∈ {-1,0,+1}: direction (close vs baseline)
    """
    # Defensive: ensure required columns exist
    if price not in df.columns:
        raise ValueError("baseline_ema requires a price column on the input DataFrame.")

    p = int(length) if length is not None else int(period)
    p = max(p, 1)

    # IMPORTANT: pass length to the helper
    base = ema(df[price].astype(float), length=p)

    out = df.copy()
    out["baseline"] = base

    # Directional signal: +1 if close > baseline, -1 if close < baseline, else 0
    sig = np.where(
        out["close"] > out["baseline"],
        1,
        np.where(out["close"] < out["baseline"], -1, 0),
    ).astype("int8")
    out[signal_col] = sig

    return out


def wma(s: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    if length <= 1:
        return s.copy()
    w = np.arange(1, length + 1, dtype=float)
    return s.rolling(length).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)


def sma(s: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    return s.rolling(length).mean()


def tma(s: pd.Series, length: int) -> pd.Series:
    """Triangular MA via triangular weights."""
    length = max(int(length), 1)
    if length <= 1:
        return s.copy()
    m = (length + 1) // 2
    up = np.arange(1, m + 1, dtype=float)
    down = up[:-1] if length % 2 == 1 else up
    w = np.concatenate([up, down[::-1]])
    w = w / w.sum()
    return s.rolling(length).apply(lambda x: float(np.dot(x, w)), raw=True)


def gaussian_weights(length: int, sigma: float) -> np.ndarray:
    length = max(int(length), 1)
    sigma = max(float(sigma), 1e-6)
    i = np.arange(length, dtype=float)
    center = (length - 1) / 2.0
    w = np.exp(-0.5 * ((i - center) / sigma) ** 2)
    w /= w.sum()
    return w


def gaussian_filter(s: pd.Series, length: int, sigma: float) -> pd.Series:
    w = gaussian_weights(length, sigma)
    L = len(w)
    if L <= 1:
        return s.copy()
    return s.rolling(L).apply(lambda x: float(np.dot(x, w)), raw=True)


def rolling_linreg_midline(s: pd.Series, length: int) -> pd.Series:
    """Rolling linear regression midline (value at rightmost bar of window)."""
    length = max(int(length), 2)
    x = np.arange(length, dtype=float)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()

    def _last_fit(y: np.ndarray) -> float:
        ym = y.mean()
        cov = ((x - xm) * (y - ym)).sum()
        b = cov / denom if denom != 0 else 0.0
        a = ym - b * xm
        return float(a + b * x[-1])

    return s.rolling(length).apply(_last_fit, raw=True)


def safe_log(x, eps: float = 1e-12) -> np.ndarray | float:
    """Log with clamp to avoid -inf/+inf and NaNs."""
    return np.log(np.clip(x, eps, None))


# ---- helper aliases (to match any internal underscore calls) ----
_set_signal = set_signal
_ema = ema
_wma = wma
_sma = sma
_tma = tma
_safe_log = safe_log
_gaussian_weights = gaussian_weights
_gaussian_filter = gaussian_filter
_rolling_linreg_midline = rolling_linreg_midline

# ---------- baseline implementations ----------


# 📈 fantailvma3.mq4  (approx: variable EMA * 3)
def baseline_fantailvma3(
    df: pd.DataFrame,
    *,
    period: int = 34,
    roc_lookback: int = 5,
    alpha_min: float = 0.05,
    alpha_max: float = 0.5,
    signal_col: str = "baseline_signal",
    price: str = "close",
    **kwargs,
) -> pd.DataFrame:
    """Fantail VMA (approx): triple variable-EMA where alpha adapts to recent ROC magnitude."""
    s = df[price].astype(float)
    roc = s.diff(roc_lookback).abs()
    denom = s.rolling(roc_lookback).mean().abs()
    roc_norm = (roc / np.clip(denom, 1e-12, None)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    base_alpha = 2.0 / max(int(period), 1)

    alpha = (base_alpha * roc_norm).clip(alpha_min, alpha_max).values

    def _vema(x, a):
        y = np.empty_like(x)
        y[:] = np.nan
        acc = np.nan
        for i in range(len(x)):
            xi = x[i]
            ai = a[i] if np.isfinite(a[i]) else (a[i - 1] if i > 0 else base_alpha)
            if not np.isfinite(xi):
                y[i] = acc
                continue
            if np.isnan(acc):
                acc = xi
            else:
                acc = acc + ai * (xi - acc)
            y[i] = acc
        return y

    e1 = _vema(s.values, alpha)
    e2 = _vema(e1, alpha)
    e3 = _vema(e2, alpha)

    df["baseline"] = pd.Series(e3, index=df.index)
    return _set_signal(df, "baseline", signal_col)


# 📈 VIDYA (Chande) using CMO
def baseline_vidya(
    df: pd.DataFrame,
    *,
    length: int = 20,
    cmo_length: int = 9,
    signal_col: str = "baseline_signal",
    price: str = "close",
    **kwargs,
) -> pd.DataFrame:
    """VIDYA: alpha = |CMO| * (2/(length+1))."""
    s = df[price].astype(float)
    diff = s.diff()
    up = diff.clip(lower=0).rolling(cmo_length).sum()
    dn = (-diff.clip(upper=0)).rolling(cmo_length).sum()
    denom = (up + dn).replace(0, np.nan)
    cmo = ((up - dn) / denom).fillna(0.0).clip(-1.0, 1.0)
    k = 2.0 / max(int(length) + 1, 2)
    alpha = (cmo.abs() * k).clip(0.0, 1.0)

    y = np.empty_like(s.values)
    y[:] = np.nan
    acc = np.nan
    for i, xi in enumerate(s.values):
        ai = float(alpha.iloc[i]) if np.isfinite(alpha.iloc[i]) else k
        if np.isnan(acc):
            acc = xi
        else:
            acc = acc + ai * (xi - acc)
        y[i] = acc

    df["baseline"] = pd.Series(y, index=df.index)
    return _set_signal(df, "baseline", signal_col)


# 📈 FRAMA (approx)
def baseline_frama_indicator(
    df: pd.DataFrame,
    *,
    length: int = 16,
    fast: float = 4.0,
    slow: float = 300.0,
    signal_col: str = "baseline_signal",
    **kwargs,
) -> pd.DataFrame:
    """Ehlers FRAMA approximation."""
    h = df["high"].astype(float)
    low = df["low"].astype(float)
    n = max(int(length), 2)
    half = max(n // 2, 1)

    L1 = (h.rolling(half).max() - low.rolling(half).min()).shift(half)
    L2 = h.rolling(half).max() - low.rolling(half).min()
    L3 = h.rolling(n).max() - low.rolling(n).min()

    L1 = L1.replace(0, np.nan)
    L2 = L2.replace(0, np.nan)
    L3 = L3.replace(0, np.nan)

    D = ((_safe_log(L1 + L2) - _safe_log(L3)) / np.log(2)).clip(0.0, 2.0)
    alpha = np.exp(-4.6 * (D - 1.0))
    a_min = 2.0 / (slow + 1.0)
    a_max = 2.0 / (fast + 1.0)
    alpha = alpha.clip(a_min, a_max).fillna(a_min)

    s = df["close"].astype(float)
    y = np.empty_like(s.values)
    y[:] = np.nan
    acc = np.nan
    for i, xi in enumerate(s.values):
        ai = float(alpha.iloc[i]) if np.isfinite(alpha.iloc[i]) else a_min
        if np.isnan(acc):
            acc = xi
        else:
            acc = acc + ai * (xi - acc)
        y[i] = acc

    df["baseline"] = pd.Series(y, index=df.index)
    return _set_signal(df, "baseline", signal_col)


# 📈 ALMA
def baseline_alma_indicator(
    df: pd.DataFrame,
    *,
    length: int = 20,
    offset: float = 0.85,
    sigma: float = 6.0,
    price: str = "close",
    signal_col: str = "baseline_signal",
    **kwargs,
) -> pd.DataFrame:
    """ALMA baseline."""
    s = df[price].astype(float)
    L = max(int(length), 1)
    m = float(offset) * (L - 1)
    w = np.exp(-0.5 * ((np.arange(L) - m) / (L / float(sigma))) ** 2)
    w /= w.sum()
    alma = s.rolling(L).apply(lambda x: float(np.dot(x, w)), raw=True)
    df["baseline"] = alma
    return _set_signal(df, "baseline", signal_col)


# 📈 T3 MA (Tillson)
def baseline_t3_ma(
    df: pd.DataFrame,
    *,
    length: int = 20,
    vfactor: float = 0.7,
    price: str = "close",
    signal_col: str = "baseline_signal",
    **kwargs,
) -> pd.DataFrame:
    """Tillson T3 moving average."""
    s = df[price].astype(float)
    n = max(int(length), 1)
    e1 = _ema(s, n)
    e2 = _ema(e1, n)
    e3 = _ema(e2, n)
    e4 = _ema(e3, n)
    e5 = _ema(e4, n)
    e6 = _ema(e5, n)

    v = float(vfactor)
    c1 = -(v**3)
    c2 = 3 * v**2 + 3 * v**3
    c3 = -6 * v**2 - 3 * v - 3 * v**3
    c4 = 1 + 3 * v + v**3 + 3 * v**2

    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    df["baseline"] = t3
    return _set_signal(df, "baseline", signal_col)


# 📈 Tether Line (Donchian midline)
def baseline_tether_line(
    df: pd.DataFrame, *, length: int = 20, signal_col: str = "baseline_signal", **kwargs
) -> pd.DataFrame:
    """Donchian channel midline."""
    hi = df["high"].rolling(length).max()
    lo = df["low"].rolling(length).min()
    mid = (hi + lo) / 2.0
    df["baseline"] = mid
    return _set_signal(df, "baseline", signal_col)


# 📈 Range Filter (ATR step)
def baseline_range_filter_modified(
    df: pd.DataFrame,
    *,
    length_atr: int = 14,
    atr_mult: float = 1.0,
    price: str = "close",
    signal_col: str = "baseline_signal",
    **kwargs,
) -> pd.DataFrame:
    """Range Filter (approx): baseline steps when price moves beyond +/- ATR*mult."""
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df[price]).abs()
    tr3 = (df["low"] - df[price]).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / max(int(length_atr), 1), adjust=False).mean()

    s = df[price].astype(float).values
    rng = (atr * float(atr_mult)).values

    y = np.empty_like(s)
    y[:] = np.nan
    baseline = np.nan
    for i in range(len(s)):
        x = s[i]
        r = rng[i] if np.isfinite(rng[i]) else 0.0
        if np.isnan(baseline):
            baseline = x
        else:
            if x > baseline + r:
                baseline = x - r
            elif x < baseline - r:
                baseline = x + r
        y[i] = baseline

    df["baseline"] = pd.Series(y, index=df.index)
    return _set_signal(df, "baseline", signal_col)


# 📈 Geometric Mean MA
def baseline_geomin_ma(
    df: pd.DataFrame,
    *,
    length: int = 20,
    price: str = "close",
    signal_col: str = "baseline_signal",
    **kwargs,
) -> pd.DataFrame:
    """Geometric mean over window."""
    s = df[price].astype(float)
    gm = (pd.Series(_safe_log(s), index=s.index).rolling(length).mean()).apply(np.exp)
    df["baseline"] = gm
    return _set_signal(df, "baseline", signal_col)


# 📈 Sine-weighted MA
def baseline_sinewma(
    df: pd.DataFrame,
    *,
    length: int = 20,
    price: str = "close",
    signal_col: str = "baseline_signal",
    **kwargs,
) -> pd.DataFrame:
    """Sine-weighted moving average."""
    s = df[price].astype(float)
    L = max(int(length), 1)
    w = np.sin(np.pi * (np.arange(1, L + 1)) / (L + 1))
    w /= w.sum()
    out = s.rolling(L).apply(lambda x: float(np.dot(x, w)), raw=True)
    df["baseline"] = out
    return _set_signal(df, "baseline", signal_col)


# 📈 Triangular MA (general)
def baseline_trimagen(
    df: pd.DataFrame,
    *,
    length: int = 20,
    price: str = "close",
    signal_col: str = "baseline_signal",
    **kwargs,
) -> pd.DataFrame:
    s = df[price].astype(float)
    df["baseline"] = _tma(s, length)
    return _set_signal(df, "baseline", signal_col)


# 📈 Gaussian filter
def baseline_gd(
    df: pd.DataFrame,
    *,
    length: int = 20,
    sigma: float = 3.0,
    price: str = "close",
    signal_col: str = "baseline_signal",
    **kwargs,
) -> pd.DataFrame:
    s = df[price].astype(float)
    df["baseline"] = _gaussian_filter(s, length, sigma)
    return _set_signal(df, "baseline", signal_col)


# 📈 Regression channel midline
def baseline_gchannel(
    df: pd.DataFrame,
    *,
    length: int = 100,
    price: str = "close",
    signal_col: str = "baseline_signal",
    **kwargs,
) -> pd.DataFrame:
    s = df[price].astype(float)
    df["baseline"] = _rolling_linreg_midline(s, length)
    return _set_signal(df, "baseline", signal_col)


# 📈 HMA (Hull MA)
def baseline_hma(
    df: pd.DataFrame,
    *,
    length: int = 55,
    price: str = "close",
    signal_col: str = "baseline_signal",
    **kwargs,
) -> pd.DataFrame:
    s = df[price].astype(float)
    n = max(int(length), 1)
    wma_n = _wma(s, n)
    wma_half = _wma(s, max(n // 2, 1))
    diff = 2 * wma_half - wma_n
    hma = _wma(diff, max(int(np.sqrt(n)), 1))
    df["baseline"] = hma
    return _set_signal(df, "baseline", signal_col)


# 📈 McGinley Dynamic
def baseline_mcginley_dynamic(
    df: pd.DataFrame,
    *,
    length: int = 14,
    k: float = 0.6,
    price: str = "close",
    signal_col: str = "baseline_signal",
    **kwargs,
) -> pd.DataFrame:
    s = df[price].astype(float).values
    y = np.empty_like(s)
    y[:] = np.nan
    md = np.nan
    L = max(int(length), 1)
    k = float(k)
    for i, x in enumerate(s):
        if np.isnan(md):
            md = x
        else:
            ratio = np.clip(x / (md if md != 0 else 1e-12), 1e-6, 1e6)
            denom = k * L * (ratio**4)
            denom = denom if np.isfinite(denom) and denom != 0 else 1.0
            md = md + (x - md) / denom
        y[i] = md
    df["baseline"] = pd.Series(y, index=range(len(s)))
    return _set_signal(df, "baseline", signal_col)


# 📈 Ehlers Two-Pole Super Smoother
def baseline_ehlers_two_pole_super_smoother_filter(
    df: pd.DataFrame,
    *,
    period: int = 20,
    price: str = "close",
    signal_col: str = "baseline_signal",
    **kwargs,
) -> pd.DataFrame:
    s = df[price].astype(float).values
    per = max(int(period), 2)
    pi = np.pi
    a1 = np.exp(-1.414 * pi / per)
    b1 = 2 * a1 * np.cos(1.414 * pi / per)
    c2 = b1
    c3 = -(a1 * a1)
    c1 = 1 - c2 - c3

    y = np.empty_like(s)
    y[:] = np.nan
    for i in range(len(s)):
        x0 = s[i]
        x1 = s[i - 1] if i > 0 else s[i]
        y1 = y[i - 1] if i > 0 and np.isfinite(y[i - 1]) else x1
        y2 = y[i - 2] if i > 1 and np.isfinite(y[i - 2]) else y1
        y[i] = c1 * (x0 + x1) / 2.0 + c2 * y1 + c3 * y2

    df["baseline"] = pd.Series(y, index=range(len(s)))
    return _set_signal(df, "baseline", signal_col)


# 📈 ATR-based EMA variant
def baseline_atr_based_ema_variant_1(
    df: pd.DataFrame,
    *,
    length: int = 20,
    atr_length: int = 14,
    adapt_power: float = 1.0,
    alpha_min: float = 0.02,
    alpha_max: float = 0.5,
    price: str = "close",
    signal_col: str = "baseline_signal",
    **kwargs,
) -> pd.DataFrame:
    s = df[price].astype(float)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df[price]).abs()
    tr3 = (df["low"] - df[price]).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / max(int(atr_length), 1), adjust=False).mean()
    atr_mean = atr.ewm(span=max(int(atr_length), 1), adjust=False).mean()

    ratio = (atr / atr_mean.replace(0, np.nan)).pow(float(adapt_power))
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    base_alpha = 2.0 / (max(int(length), 1) + 1.0)
    alpha = (base_alpha * ratio).clip(alpha_min, alpha_max).values

    y = np.empty_like(s.values)
    y[:] = np.nan
    acc = np.nan
    for i, xi in enumerate(s.values):
        ai = alpha[i] if np.isfinite(alpha[i]) else base_alpha
        if np.isnan(acc):
            acc = xi
        else:
            acc = acc + ai * (xi - acc)
        y[i] = acc

    df["baseline"] = pd.Series(y, index=df.index)
    return _set_signal(df, "baseline", signal_col)
