"""Price-geometry features (retained from KH-era + v3 extensions).

All features use mid-close / mid-high / mid-low per bar (averaged
bid+ask). Causal lineage: clean — computed from bars closed strictly
before the signal bar via ``shift(1)`` on rolling windows.
"""

from __future__ import annotations

import pandas as pd

from core.features._helpers import kijun, mid_close, mid_high, mid_low, wilder_atr
from core.features.lineage import CausalLineage, FeatureSpec
from core.features.registry import register

# ── ATR(14) ───────────────────────────────────────────────────────────


def _atr_14(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    return wilder_atr(mid_high(pair_df), mid_low(pair_df), mid_close(pair_df), period=14).shift(1)


register(
    FeatureSpec(
        name="atr_14",
        producer=_atr_14,
        lineage=CausalLineage.CLEAN,
        feature_class="price_geometry",
        description="Wilder ATR(14) on mid-OHLC, shifted 1 to use bars closed strictly before signal.",
        inputs={"period": 14, "window_input": "mid_high+mid_low+mid_close"},
    )
)


# ── Kijun(26) distance ────────────────────────────────────────────────


def _kijun_26_distance(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    k = kijun(mid_high(pair_df), mid_low(pair_df), period=26).shift(1)
    return mid_close(pair_df).shift(1) - k


register(
    FeatureSpec(
        name="kijun_26_distance",
        producer=_kijun_26_distance,
        lineage=CausalLineage.CLEAN,
        feature_class="price_geometry",
        description=(
            "Distance from prior mid-close to prior Kijun-sen(26). Positive = "
            "price above Kijun. Shifted 1 — uses strictly prior bars only."
        ),
        inputs={"period": 26},
    )
)


# ── Swing-high / swing-low distances (N=14) ───────────────────────────


def _swing_high_distance_14(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    swing_high = mid_high(pair_df).rolling(window=14, min_periods=14).max().shift(1)
    return swing_high - mid_close(pair_df).shift(1)


register(
    FeatureSpec(
        name="swing_high_distance_14",
        producer=_swing_high_distance_14,
        lineage=CausalLineage.CLEAN,
        feature_class="price_geometry",
        description=(
            "Distance from prior mid-close to the 14-bar trailing swing high "
            "(also on prior bars). Positive — swing high is above current close."
        ),
        inputs={"period": 14},
    )
)


def _swing_low_distance_14(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    swing_low = mid_low(pair_df).rolling(window=14, min_periods=14).min().shift(1)
    return mid_close(pair_df).shift(1) - swing_low


register(
    FeatureSpec(
        name="swing_low_distance_14",
        producer=_swing_low_distance_14,
        lineage=CausalLineage.CLEAN,
        feature_class="price_geometry",
        description=(
            "Distance from prior mid-close to the 14-bar trailing swing low. "
            "Positive — current close is above the swing low."
        ),
        inputs={"period": 14},
    )
)


# ── Range / close ratio ───────────────────────────────────────────────


def _range_close_ratio(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    h = mid_high(pair_df).shift(1)
    lo = mid_low(pair_df).shift(1)
    c = mid_close(pair_df).shift(1)
    return (h - lo) / c


register(
    FeatureSpec(
        name="range_close_ratio",
        producer=_range_close_ratio,
        lineage=CausalLineage.CLEAN,
        feature_class="price_geometry",
        description="Prior-bar (high - low) / close. Bar-1 volatility proxy.",
        inputs={},
    )
)
