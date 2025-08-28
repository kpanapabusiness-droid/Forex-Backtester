# indicators/exit_funcs.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_twiggs_money_flow(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Compute a stable Twiggs Money Flow proxy.
    TMF = EMA(advances, period) / EMA(volume, period), where
      advances = ((close - low) - (high - close)) * volume
    """
    period = max(int(period), 1)

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce").astype(float)

    up_move = (close - low) - (high - close)
    adv = up_move * vol

    # Wilder-style smoothing (EMA with alpha=1/period)
    flow_ema = adv.ewm(alpha=1.0 / period, adjust=False).mean()
    vol_ema = vol.ewm(alpha=1.0 / period, adjust=False).mean().replace(0, np.nan)

    tmf = (flow_ema / vol_ema).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return tmf


def exit_twiggs_money_flow(
    df: pd.DataFrame,
    *,
    period: int = 21,
    threshold: float = 0.0,
    mode: str = "zero_cross",  # "zero_cross" | "threshold_cross"
    signal_col: str = "exit_signal",
    **kwargs,
) -> pd.DataFrame:
    """
    TMF-based exit indicator.
    Writes `df[signal_col]` ∈ {0,1} where 1 = "exit now" (boolean exit).

    Modes:
      - zero_cross       : exit when TMF flips sign (crosses 0)
      - threshold_cross  : exit when TMF crosses between +threshold and -threshold bands

    Parameters:
      period     : TMF smoothing length
      threshold  : band for 'threshold_cross' mode (>=0). Ignored for 'zero_cross'.
      signal_col : output column name (defaults to 'exit_signal')
    """
    out = df.copy()
    tmf = _safe_twiggs_money_flow(out, period=period)

    if mode not in ("zero_cross", "threshold_cross"):
        mode = "zero_cross"  # defensive fallback

    if mode == "zero_cross":
        prev = np.sign(tmf.shift(1))
        curr = np.sign(tmf)
    else:
        thr = float(max(threshold, 0.0))

        # classify to {-1,0,1} using bands
        def classify(x: pd.Series) -> pd.Series:
            return np.where(x > +thr, 1, np.where(x < -thr, -1, 0))

        prev = classify(tmf.shift(1).fillna(0.0))
        curr = classify(tmf)

    # Exit when classification flips between +1 and -1 (and neither is 0)
    exit_now = (prev != 0) & (curr != 0) & (prev != curr)
    out[signal_col] = exit_now.astype("int8")
    return out
