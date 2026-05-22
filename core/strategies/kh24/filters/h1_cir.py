"""H1 Close-In-Range (CIR) filter at T = 0.28.

Matches the deployed MT5 EA ``KH24_EA.mq5`` (see
``reference/kh24_ea/KH24_EA.mq5`` lines 295-313 + 773-785). The filter
evaluates the close-position of the last completed H1 bar at the H4
signal time and requires ``CIR ≤ 0.28``.

CIR formula: ``(close - low) / (high - low)``.

The H1 reference bar is the LAST H1 bar inside the just-closed H4
bar's interval (H4 covers ``[t, t+4h)``; last H1 is at ``[t+3h,
t+4h)``). At signal-evaluation time (first tick of the new H4 bar),
this H1 has just closed; the EA reads it via ``CopyRates(sym,
PERIOD_H1, 1, 1, ...)`` — shift=1, the most-recently-closed H1.

**Bid OHLC**, not mid OHLC. The EA reads MT5's ``CopyRates`` which
returns single-side (bid by broker convention). v3's earlier mid-OHLC
implementation produced a CIR ~0.5-2 percentage points different on
spread-active bars — enough to shift ~40 trades over 5.25 years per
the bisect diagnostic.

See ``docs/dispatches/kh24_fix_diff.md`` Section B.

Causal lineage: clean. The H1 bar referenced is closed strictly before
the H4 bar's close (it ENDS at the H4 close).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class H1CIRParams:
    threshold: float = 0.28  # CIR ≤ this = pass (matches EA H1CirThreshold default)


def evaluate_h1_cir(
    df_h4: pd.DataFrame, df_h1: pd.DataFrame, params: H1CIRParams | None = None
) -> np.ndarray:
    """Boolean array (same length as ``df_h4``) — True = H1 CIR ≤ threshold.

    Uses BID-side single OHLC on the H1 bar (matches the EA's
    single-side ``CopyRates`` convention).
    """
    params = params or H1CIRParams()
    if len(df_h4) == 0:
        return np.zeros(0, dtype=bool)

    h1_high = df_h1["high_bid"].values
    h1_low = df_h1["low_bid"].values
    h1_close = df_h1["close_bid"].values
    h1_range = h1_high - h1_low
    with np.errstate(divide="ignore", invalid="ignore"):
        h1_cir = np.where(h1_range > 0, (h1_close - h1_low) / h1_range, np.nan)

    cir_series = pd.Series(h1_cir, index=df_h1.index, name="h1_cir")
    # For each H4 bar at time t, pick the H1 bar starting at t + 3h
    # (i.e. the last H1 inside the H4 — same physical bar the EA reads
    # via CopyRates(PERIOD_H1, shift=1)).
    target_h1 = df_h4.index + pd.Timedelta(hours=3)
    cir_aligned = cir_series.reindex(target_h1).values

    return np.isfinite(cir_aligned) & (cir_aligned <= params.threshold)
