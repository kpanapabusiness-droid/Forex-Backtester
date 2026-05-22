"""Volatility-regime features.

Both features use ATR computed on prior bars only (shifted 1):

  atr_vs_trailing_100  - current bar's ATR / mean(trailing 100 ATR)
  atr_percentile_100   - percentile rank of current ATR in trailing 100 window

Causal lineage: clean — every input is strictly prior (shift(1) baked in).
"""

from __future__ import annotations

import pandas as pd

from core.features._helpers import (
    mid_close,
    mid_high,
    mid_low,
    percentile_rank_in_window,
    wilder_atr,
)
from core.features.lineage import CausalLineage, FeatureSpec
from core.features.registry import register


def _atr_vs_trailing_100(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    atr = wilder_atr(mid_high(pair_df), mid_low(pair_df), mid_close(pair_df), period=14).shift(1)
    trailing = atr.rolling(window=100, min_periods=100).mean()
    return atr / trailing


register(
    FeatureSpec(
        name="atr_vs_trailing_100",
        producer=_atr_vs_trailing_100,
        lineage=CausalLineage.CLEAN,
        feature_class="vol_regime",
        description=(
            "Wilder ATR(14) at signal bar (computed on prior data) divided by the "
            "mean of the trailing 100-bar ATR series. >1 ⇒ vol expansion."
        ),
        inputs={"atr_period": 14, "trailing_window": 100},
    )
)


def _atr_percentile_100(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    atr = wilder_atr(mid_high(pair_df), mid_low(pair_df), mid_close(pair_df), period=14).shift(1)
    return percentile_rank_in_window(atr, window=100)


register(
    FeatureSpec(
        name="atr_percentile_100",
        producer=_atr_percentile_100,
        lineage=CausalLineage.CLEAN,
        feature_class="vol_regime",
        description=(
            "Trailing-100 percentile rank of Wilder ATR(14). 0..1 (1 = highest "
            "vol in the trailing window). Uses strictly prior bars only."
        ),
        inputs={"atr_period": 14, "trailing_window": 100},
    )
)
