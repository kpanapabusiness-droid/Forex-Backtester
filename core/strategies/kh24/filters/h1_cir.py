"""H1 Close-In-Range (CIR) filter at T = 0.28.

Per the KH-24 deployment lineage: at each H4 signal bar's close, look up
the coinciding H1 bar's CIR — ``(close − low) / (high − low)`` — and
require ``CIR ≤ T``. T = 0.28 = "close in bottom 28% of the H1 range" =
weakness confirmation for the long entry.

The "coinciding H1 bar" is the H1 bar whose [start, end) interval
contains the H4 bar's close timestamp. Concretely: H4 bar labelled at
``t`` covers ``[t, t+4h)``; its close is at ``t+4h−ε`` → the H1 bar at
``t+3h`` covers ``[t+3h, t+4h)`` which is the last H1 inside the H4.
We pick that H1.

Causal lineage: clean. The H1 bar referenced is closed strictly before
the H4 bar's close (it ENDS at the H4 close), so it's available at
signal time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.features._helpers import mid_close, mid_high, mid_low


@dataclass(frozen=True)
class H1CIRParams:
    threshold: float = 0.28  # CIR ≤ this = pass


def evaluate_h1_cir(
    df_h4: pd.DataFrame, df_h1: pd.DataFrame, params: H1CIRParams | None = None
) -> np.ndarray:
    """Boolean array (same length as ``df_h4``) — True = H1 CIR ≤ threshold.

    The H1 reference bar is the LAST H1 bar inside the H4 bar's
    interval (i.e. the H1 starting 3 hours after the H4's left edge).
    Bars with zero H1 range (``high == low``) are excluded.
    """
    params = params or H1CIRParams()
    if len(df_h4) == 0:
        return np.zeros(0, dtype=bool)

    h1_high = mid_high(df_h1).values
    h1_low = mid_low(df_h1).values
    h1_close = mid_close(df_h1).values
    h1_range = h1_high - h1_low
    with np.errstate(divide="ignore", invalid="ignore"):
        h1_cir = np.where(h1_range > 0, (h1_close - h1_low) / h1_range, np.nan)

    cir_series = pd.Series(h1_cir, index=df_h1.index, name="h1_cir")
    # For each H4 bar at time t, pick the H1 bar starting at t + 3h
    target_h1 = df_h4.index + pd.Timedelta(hours=3)
    cir_aligned = cir_series.reindex(target_h1).values

    return np.isfinite(cir_aligned) & (cir_aligned <= params.threshold)
