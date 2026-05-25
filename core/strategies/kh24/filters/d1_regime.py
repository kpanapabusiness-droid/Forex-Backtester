"""D1 regime filter (one-day-lag rule per L_PROTOCOL §1).

KH-24's deployment lineage: long signals require an "up" D1 regime.
Concretely: at H4 signal bar's calendar day T, the prior calendar
day's D1 close must be above the prior D1 Kijun(26).

This filter is functionally a re-statement of conditions C8 + C9 from
the signal evaluator — kept as a separate, reusable predicate so other
arcs can take just the regime gate without the full KH-24 signal.

Causal lineage: clean. The lag-1 alignment is verified via
``test_d1_regime_lookahead`` — perturbing the same-day D1 close has
zero impact on the filter output for any H4 bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.features._helpers import kijun, mid_close, mid_high, mid_low, wilder_atr
from core.signals.htf_alignment import get_htf_row_at


@dataclass(frozen=True)
class D1RegimeParams:
    kijun_period: int = 26
    atr_period: int = 14
    distance_cap_atr: float = 1.0  # close must be ≤ Kijun + cap × ATR


def evaluate_d1_regime(
    df_h4: pd.DataFrame,
    df_d1: pd.DataFrame,
    params: D1RegimeParams | None = None,
) -> np.ndarray:
    """Boolean array (same length as ``df_h4``) — True = regime allows long.

    Regime gate: prev D1 close > prev D1 Kijun AND prev D1 close ≤ prev
    D1 Kijun + ``distance_cap_atr`` × prev D1 ATR. NaN warmup bars are
    False.
    """
    params = params or D1RegimeParams()
    if len(df_h4) == 0:
        return np.zeros(0, dtype=bool)

    d1_panel = pd.DataFrame(
        {
            "d1_close": mid_close(df_d1).values.astype(float),
            "d1_kijun": kijun(
                mid_high(df_d1), mid_low(df_d1), period=params.kijun_period
            ).values.astype(float),
            "d1_atr": wilder_atr(
                mid_high(df_d1), mid_low(df_d1), mid_close(df_d1), period=params.atr_period
            ).values.astype(float),
        },
        index=df_d1.index,
    )
    rows = get_htf_row_at(df_h4.index, d1_panel, require_fully_closed=True)
    d1c = rows["d1_close"].to_numpy(dtype=float)
    d1k = rows["d1_kijun"].to_numpy(dtype=float)
    d1a = rows["d1_atr"].to_numpy(dtype=float)

    finite = np.isfinite(d1c) & np.isfinite(d1k) & np.isfinite(d1a) & (d1a > 0)
    above_kijun = d1c > d1k
    within_distance = d1c <= (d1k + params.distance_cap_atr * d1a)
    return finite & above_kijun & within_distance
