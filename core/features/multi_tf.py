"""Multi-TF features computed from D1 / W1 aggregates.

These features require the caller to supply the higher-TF aggregate
panels (D1 + W1) via the ``panel`` argument's optional ``aux`` dict. The
common pattern is:

    panel = Panel(...)
    aux = {"d1": Panel.from_pairs(pairs, "D1", ...), "w1": ...}
    matrix = compute_feature_matrix(pair, pair_df, panel, aux=aux)

L_PROTOCOL §1 mandates the D1 one-bar lag rule: at any 4H/H1 signal bar
on calendar day T, only D1 bars from day T-1 or earlier are visible.
This module enforces lag via ``merge_asof(direction="backward")`` on a
shifted-by-1-day key.

Causal lineage: clean — strict prior alignment via the lag rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.features._helpers import (
    percentile_rank_in_window,
    wilder_atr,
)
from core.features.lineage import CausalLineage, FeatureSpec
from core.features.registry import register
from core.signals.htf_alignment import get_htf_value_at


def _build_d1_lag1_series(pair_df: pd.DataFrame, d1_df: pd.DataFrame, column: str) -> pd.Series:
    """Return ``d1_df[column]`` aligned to ``pair_df.index`` with one-day lag.

    Per L_PROTOCOL §1: at calendar day T, only D1 bars from T-1 or
    earlier are visible. Timezone-invariant — works correctly under both
    UTC and 5ers EET storage conventions (byte-identical to the legacy
    normalize-and-shift idiom under UTC; correct lag-1 under EET).
    """
    out = get_htf_value_at(pair_df.index, d1_df[[column]], column, require_fully_closed=True)
    return out.rename(column)


def _d1_close_slope_sign(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    """Sign of the D1 close-on-close slope (lag-1 D1 vs lag-2 D1).

    Requires ``panel.aux["d1"]`` — a Panel keyed at D1.
    """
    if panel is None or not hasattr(panel, "aux") or "d1" not in panel.aux:
        return pd.Series(np.nan, index=pair_df.index, name="d1_close_slope_sign")
    pair = pair_df.attrs.get("pair") or panel.aux["d1"].pairs[0]
    d1_df = panel.aux["d1"].pair_dfs[pair]
    d1_close = (d1_df["close_bid"] + d1_df["close_ask"]) / 2.0
    slope = (d1_close - d1_close.shift(1)).rename("slope")
    aligned = _build_d1_lag1_series(pair_df, slope.to_frame(), "slope")
    return np.sign(aligned).astype("float64")


register(
    FeatureSpec(
        name="d1_close_slope_sign",
        producer=_d1_close_slope_sign,
        lineage=CausalLineage.CLEAN,
        feature_class="multi_tf",
        description=(
            "Sign of the prior-D1 close-on-close slope. -1 / 0 / +1. Uses the "
            "L_PROTOCOL §1 one-day-lag rule (D1 bar from calendar day T-1 visible at T)."
        ),
        inputs={"reference_tf": "D1", "lag_rule": "one_day"},
        needs_panel=True,
    )
)


def _d1_close_slope_magnitude(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    if panel is None or not hasattr(panel, "aux") or "d1" not in panel.aux:
        return pd.Series(np.nan, index=pair_df.index, name="d1_close_slope_magnitude")
    pair = pair_df.attrs.get("pair") or panel.aux["d1"].pairs[0]
    d1_df = panel.aux["d1"].pair_dfs[pair]
    d1_close = (d1_df["close_bid"] + d1_df["close_ask"]) / 2.0
    slope = (d1_close - d1_close.shift(1)).abs().rename("mag")
    return _build_d1_lag1_series(pair_df, slope.to_frame(), "mag").astype("float64")


register(
    FeatureSpec(
        name="d1_close_slope_magnitude",
        producer=_d1_close_slope_magnitude,
        lineage=CausalLineage.CLEAN,
        feature_class="multi_tf",
        description=(
            "Absolute D1 close-on-close change (lag-1). Pair with d1_close_slope_sign "
            "for directional + magnitude features."
        ),
        inputs={"reference_tf": "D1", "lag_rule": "one_day"},
        needs_panel=True,
    )
)


def _d1_atr_percentile_100(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    if panel is None or not hasattr(panel, "aux") or "d1" not in panel.aux:
        return pd.Series(np.nan, index=pair_df.index, name="d1_atr_percentile_100")
    pair = pair_df.attrs.get("pair") or panel.aux["d1"].pairs[0]
    d1_df = panel.aux["d1"].pair_dfs[pair]
    h = (d1_df["high_bid"] + d1_df["high_ask"]) / 2.0
    lo = (d1_df["low_bid"] + d1_df["low_ask"]) / 2.0
    c = (d1_df["close_bid"] + d1_df["close_ask"]) / 2.0
    d1_atr = wilder_atr(h, lo, c, period=14)
    pct = percentile_rank_in_window(d1_atr, window=100).rename("d1_atr_pct")
    return _build_d1_lag1_series(pair_df, pct.to_frame(), "d1_atr_pct").astype("float64")


register(
    FeatureSpec(
        name="d1_atr_percentile_100",
        producer=_d1_atr_percentile_100,
        lineage=CausalLineage.CLEAN,
        feature_class="multi_tf",
        description=(
            "Trailing-100 percentile rank of D1 Wilder ATR(14). Computed on the "
            "lag-1 D1 series so it's strictly prior at the signal bar."
        ),
        inputs={
            "reference_tf": "D1",
            "atr_period": 14,
            "trailing_window": 100,
            "lag_rule": "one_day",
        },
        needs_panel=True,
    )
)


def _w1_close_slope_sign(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    """Sign of the W1 close-on-close slope using the most recent fully-closed
    weekly bar (week N-1 visible during week N).

    Requires ``panel.aux["w1"]`` — a Panel keyed at W1.

    Uses the canonical ``get_htf_value_at(..., require_fully_closed=True)``
    pattern that the D1 producers in this module already follow. The slope
    is pre-computed on the W1 series via ``shift(1)`` then looked up at
    the most recently fully-closed W1 bar at each LTF timestamp — i.e. at
    an H4 timestamp inside week N, the result is ``sign(W1_close[N-1] -
    W1_close[N-2])``, never involving week N's own (not-yet-closed) close.
    """
    if panel is None or not hasattr(panel, "aux") or "w1" not in panel.aux:
        return pd.Series(np.nan, index=pair_df.index, name="w1_close_slope_sign")
    pair = pair_df.attrs.get("pair") or panel.aux["w1"].pairs[0]
    w1_df = panel.aux["w1"].pair_dfs[pair]
    w1_close = (w1_df["close_bid"] + w1_df["close_ask"]) / 2.0
    slope = (w1_close - w1_close.shift(1)).rename("slope")
    aligned = get_htf_value_at(
        pair_df.index, slope.to_frame(), "slope", require_fully_closed=True
    )
    return np.sign(aligned).astype("float64")


register(
    FeatureSpec(
        name="w1_close_slope_sign",
        producer=_w1_close_slope_sign,
        lineage=CausalLineage.CLEAN,
        feature_class="multi_tf",
        description=(
            "Sign of the prior-W1 close-on-close slope. -1 / 0 / +1. Uses the "
            "L_PROTOCOL §1 strict-prior rule (most recently fully-closed W1 "
            "bar visible at the LTF timestamp; week N's W1 not visible "
            "until week N+1 starts)."
        ),
        inputs={"reference_tf": "W1", "alignment": "fully_closed_prior"},
        needs_panel=True,
    )
)
