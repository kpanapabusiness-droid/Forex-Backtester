"""Spread-regime features.

PR-A's data layer carries a per-bar ``spread_close = close_ask - close_bid``.
This module exposes:

  spread_vs_trailing_100      — current spread / mean(trailing 100 spread)
  spread_percentile_100       — percentile rank of current spread in trailing 100

Both shift(1) to ensure no lookahead.
"""

from __future__ import annotations

import pandas as pd

from core.features._helpers import percentile_rank_in_window
from core.features.lineage import CausalLineage, FeatureSpec
from core.features.registry import register


def _spread_vs_trailing_100(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    s = pair_df["spread_close"].shift(1)
    trailing = s.rolling(window=100, min_periods=100).mean()
    return s / trailing


register(
    FeatureSpec(
        name="spread_vs_trailing_100",
        producer=_spread_vs_trailing_100,
        lineage=CausalLineage.CLEAN,
        feature_class="spread_regime",
        description=(
            "Prior-bar spread divided by mean of the trailing 100-bar spread. "
            ">1 ⇒ widening spread regime."
        ),
        inputs={"trailing_window": 100},
    )
)


def _spread_percentile_100(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    s = pair_df["spread_close"].shift(1)
    return percentile_rank_in_window(s, window=100)


register(
    FeatureSpec(
        name="spread_percentile_100",
        producer=_spread_percentile_100,
        lineage=CausalLineage.CLEAN,
        feature_class="spread_regime",
        description=("Trailing-100 percentile rank of the prior-bar spread. 0..1."),
        inputs={"trailing_window": 100},
    )
)
