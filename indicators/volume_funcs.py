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
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    up, down = h.diff(), -l.diff()
    plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0)

    tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(
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
    l = df["low"].astype(float)
    c = df[price].astype(float)
    tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(
        axis=1
    )
    atr = _wilder_rma(tr, max(int(length), 1))
    base = atr.ewm(span=max(int(smooth), 1), adjust=False).mean().replace(0, np.nan)

    ratio = (atr / base).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df[signal_col] = (ratio >= float(threshold)).astype("int8")
    return df
