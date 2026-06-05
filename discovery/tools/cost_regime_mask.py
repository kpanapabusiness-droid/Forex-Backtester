"""Low-COST-regime entry mask — discovery EXPERIMENT tool (BUILT).

Built by arc 2005 (chat 2000s). Attacks the programme's binding constraint —
EDGE < COST — from the **cost side** (every prior arc attacked the edge side).

FundedNext cost = 1.5*spread + slippage + $5/lot RT; the spread term dominates and
varies a lot by bar/session. The cost paid *in R units* on a trade scales with
spread/ATR (a price move of sl_mult*ATR = 1R, so a spread of S costs ~S/(sl_mult*ATR)
in R). So restricting an entry to bars whose spread/ATR is in its own trailing LOW
quantile cuts the per-trade cost — without touching the edge. If a positive-gross-drift
entry (e.g. cross trend, +0.10R gross, arc 1003) is merely cost-bled, low-cost
conditioning should lift its net sign; if it does not, EDGE<COST is binding on EDGE,
not COST (decisive either way).

EXPERIMENT tool: produces a boolean MASK only (geometry/observation); scoring stays
canonical (AND the mask into a SignalModule's per-pair signal_mask, then run through
ArcFoldRunner -> MultiPairBacktester, which still applies the real per-bar cost).

Ex-ante / causal: spread_close[i] is known at bar i close (the signal bar; entry is
next bar). ATR is Wilder(14) on MID, shift(1) (strictly prior bars). The per-bar
threshold is a TRAILING rolling quantile of spread/ATR, shift(1) — so the bar's own
value never enters its own threshold. No future bar is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.sim.panel import Panel
from discovery.tools.trend_entry_signals import _atr_shift1_mid


def make_low_cost_mask(
    panel: Panel,
    pairs: list[str],
    quantile: float = 0.30,
    window: int = 250,
    atr_period: int = 14,
) -> dict[str, pd.Series]:
    """Per-pair boolean Series: True where spread/ATR <= its trailing rolling ``quantile``.

    ``quantile``: keep bars in the cheapest fraction (0.30 = cheapest 30% of recent bars).
    ``window``: trailing window (bars) for the rolling quantile (causal, shift(1)).
    """
    out: dict[str, pd.Series] = {}
    for pair in pairs:
        df = panel.pair_dfs[pair]
        idx = df.index
        spread = df["spread_close"].to_numpy(float)
        atr = _atr_shift1_mid(df, atr_period)
        with np.errstate(invalid="ignore", divide="ignore"):
            cost_ratio = spread / atr  # cost in ~R units (spread per ATR)
        cr = pd.Series(cost_ratio, index=idx)
        # trailing rolling quantile, shifted so bar i's threshold uses only bars < i
        thresh = cr.rolling(window, min_periods=window // 2).quantile(quantile).shift(1)
        mask = (cr <= thresh) & np.isfinite(cost_ratio) & (atr > 0)
        out[pair] = pd.Series(mask.to_numpy(bool), index=idx, name="low_cost_mask")
    return out


__all__ = ("make_low_cost_mask",)
