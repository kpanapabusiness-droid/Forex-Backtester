"""Arc 2 redo — signal generator for mtf_alignment.2_down_mixed.kijun.

Thin wrapper that reuses the canonical layer-4 routines from
`scripts.lchar.run_layer4` (`compute_kijun`, `mtf_alignment_states`) verbatim.
Per CLAUDE.md, `scripts/lchar/run_layer4.py` is the canonical L registry signal
source and must not be modified — this module imports from it but does not
patch it.

Public entrypoint:
    compute_signal_mask(df_1h, df_4h, df_d1, kijun_period=26) -> np.ndarray

Returns a boolean mask aligned to df_1h.index, True where the 1H bar is in
state '2_down_mixed' under the kijun trend definition (1H_sign=-1, D1_mr_sign=-1,
4H_mr_sign=+1; lookahead asserted on every firing bar).

No lookahead invariants:
  - kijun(TF) and kijun_sign(TF) computed only from bars ≤ N at TF.
  - 4H sign uses index = floor('4h', T_N) − 1 (strictly prior-completed 4H bar).
  - D1 sign uses index = floor('D',  T_N) − 1 (strictly prior-completed D1 bar
    = one-day lag per v2.0 §1.4).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.lchar.run_layer4 import compute_kijun, mtf_alignment_states  # noqa: E402


def attach_kijun_sign(df: pd.DataFrame, kijun_period: int = 26) -> pd.DataFrame:
    """Add kijun and kijun_sign columns; date column normalised.

    Verbatim mirror of run_layer4.prep_pair_tf's kijun_sign block. The
    full prep_pair_tf adds many columns we don't need; this minimal attach
    keeps the data narrow and CSV-stable.
    """
    df = df.copy()
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["kijun"] = compute_kijun(df, kijun_period)
    df["kijun_sign"] = np.sign((df["close"].astype(float) - df["kijun"]).to_numpy())
    return df


def compute_signal_mask(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_d1: pd.DataFrame,
    *,
    kijun_period: int = 26,
    pair: str = "<pair>",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2_down_mixed.kijun signal mask aligned to df_1h.

    Inputs:
      df_1h, df_4h, df_d1: per-TF OHLCV with date/time column. Either 'date' or
      'time' is accepted; output operates on a copy with 'date' renamed.

    Returns:
      (mask, s_1h, s_4h_mr, s_d1_mr): mask is a bool array of length len(df_1h);
      the three signed arrays are float (NaN where lookup out of range) — useful
      for spot-checks and diagnostics.

    Raises:
      RuntimeError on any runtime lookahead violation (4H_mr ts >= floor('4h', T_N)
      OR D1_mr ts >= floor('D', T_N)) at a signal-firing bar.
    """
    df_1h_k = attach_kijun_sign(df_1h, kijun_period)
    df_4h_k = attach_kijun_sign(df_4h, kijun_period)
    df_d1_k = attach_kijun_sign(df_d1, kijun_period)

    states = mtf_alignment_states(df_1h_k, df_4h_k, df_d1_k, "kijun")
    mask = states == "2_down_mixed"

    # Pre-computed mr indices for diagnostics + lookahead assertion (mirrors
    # canonical mtf_alignment_states logic).
    floor4h = df_1h_k["date"].dt.floor("4h")
    floor_d1 = df_1h_k["date"].dt.normalize()
    idx4 = pd.Series(np.arange(len(df_4h_k), dtype=np.int64), index=df_4h_k["date"])
    idxd = pd.Series(np.arange(len(df_d1_k), dtype=np.int64), index=df_d1_k["date"])
    c4 = floor4h.map(idx4).to_numpy(dtype=float)
    cd = floor_d1.map(idxd).to_numpy(dtype=float)
    val = (~np.isnan(c4)) & (~np.isnan(cd))
    c4i = np.where(val, c4, 0).astype(np.int64)
    cdi = np.where(val, cd, 0).astype(np.int64)
    mr4 = c4i - 1
    mrd = cdi - 1
    val = val & (mr4 >= 0) & (mrd >= 0)

    s_1h_full = df_1h_k["kijun_sign"].to_numpy(dtype=float)
    s_4h_full = df_4h_k["kijun_sign"].to_numpy(dtype=float)
    s_d1_full = df_d1_k["kijun_sign"].to_numpy(dtype=float)

    n = len(df_1h_k)
    s_1h_out = np.full(n, np.nan, dtype=float)
    s_4h_out = np.full(n, np.nan, dtype=float)
    s_d1_out = np.full(n, np.nan, dtype=float)
    pos = np.where(val)[0]
    s_1h_out[pos] = s_1h_full[pos]
    s_4h_out[pos] = s_4h_full[mr4[pos]]
    s_d1_out[pos] = s_d1_full[mrd[pos]]

    # Runtime lookahead assertion on signal-firing bars.
    ts_4h = df_4h_k["date"].to_numpy()
    ts_d1 = df_d1_k["date"].to_numpy()
    fires = np.where(mask)[0]
    if fires.size > 0:
        f4_np = floor4h.to_numpy()
        fd_np = floor_d1.to_numpy()
        bad_4h = ts_4h[mr4[fires]] >= f4_np[fires]
        bad_d1 = ts_d1[mrd[fires]] >= fd_np[fires]
        if bad_4h.any():
            i = int(fires[int(np.argmax(bad_4h))])
            raise RuntimeError(
                f"4H lookahead at {pair} bar {i}: ts_4h_used={ts_4h[mr4[i]]} "
                f">= floor4h(T_N)={f4_np[i]}"
            )
        if bad_d1.any():
            i = int(fires[int(np.argmax(bad_d1))])
            raise RuntimeError(
                f"D1 lookahead at {pair} bar {i}: ts_d1_used={ts_d1[mrd[i]]} "
                f">= floor_d1(T_N)={fd_np[i]}"
            )

    return mask.astype(bool), s_1h_out, s_4h_out, s_d1_out


def wilder_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder ATR(period) on 1H frame.

    Standard Wilder RMA: ATR[period-1] = mean(TR[0:period]); thereafter
    ATR[i] = (ATR[i-1] * (period-1) + TR[i]) / period.
    """
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    prev_close = np.empty(n, dtype=float)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


__all__ = ["attach_kijun_sign", "compute_signal_mask", "wilder_atr"]
