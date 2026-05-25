"""Distance features — prior-session HL, round-number distance.

Both features measure where the current price sits in absolute terms
relative to a reference level. Causal lineage: clean — references are
computed from bars closed strictly before the signal bar (shift(1)).

Session boundary: "prior session" maps to the trading-day boundary
identified by ``panel.boundary_convention`` (default ``"utc"`` for
legacy KH-24 byte-identical safety when no panel is provided). Under
``"5ers_eet"`` (post-PR-189 engine default) sessions are EET trading
days; see ``core.utils.session_boundary.utc_to_eet_trading_day`` for
DST handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.features._helpers import mid_close, mid_high, mid_low, pip_size_for_pair
from core.features.lineage import CausalLineage, FeatureSpec
from core.features.registry import register
from core.utils.session_boundary import utc_to_eet_trading_day


def _convention_from_panel(panel) -> str:
    """Read ``boundary_convention`` from panel; UTC fallback if absent."""
    if panel is None:
        return "utc"
    return getattr(panel, "boundary_convention", "utc")


def _prior_session_high(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    """Distance from current mid-close to the prior trading-day's mid session high.

    Trading-day boundary follows ``panel.boundary_convention`` (UTC or
    EET). Implementation: bucket bars by their trading-day key, take
    max(mid_high) per day, then for each bar look up the prior day's
    high. Strictly prior by construction.
    """
    convention = _convention_from_panel(panel)
    df = pair_df.copy()
    df["_date"] = utc_to_eet_trading_day(df.index, convention=convention)
    df["_mid_high"] = mid_high(pair_df).values
    daily_high = df.groupby("_date")["_mid_high"].max().rename("prior_day_high_mid")
    prev_date_high = pd.Series(
        df["_date"].map(lambda d: daily_high.get(d - pd.Timedelta(days=1), np.nan)).values,
        index=df.index,
        dtype="float64",
    )
    return mid_close(pair_df).shift(1) - prev_date_high


register(
    FeatureSpec(
        name="prior_session_high_distance",
        producer=_prior_session_high,
        lineage=CausalLineage.CLEAN,
        feature_class="distance",
        description=(
            "Distance from the prior-bar mid-close to the prior trading-day's "
            "mid-price session high. Positive ⇒ above prior day's high. "
            "Trading-day boundary per Panel.boundary_convention (UTC or 5ers_eet)."
        ),
        inputs={"reference": "prior_trading_day_high_mid"},
    )
)


def _prior_session_low(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    convention = _convention_from_panel(panel)
    df = pair_df.copy()
    df["_date"] = utc_to_eet_trading_day(df.index, convention=convention)
    df["_mid_low"] = mid_low(pair_df).values
    daily_low = df.groupby("_date")["_mid_low"].min().rename("prior_day_low_mid")
    prev_date_low = pd.Series(
        df["_date"].map(lambda d: daily_low.get(d - pd.Timedelta(days=1), np.nan)).values,
        index=df.index,
        dtype="float64",
    )
    return mid_close(pair_df).shift(1) - prev_date_low


register(
    FeatureSpec(
        name="prior_session_low_distance",
        producer=_prior_session_low,
        lineage=CausalLineage.CLEAN,
        feature_class="distance",
        description=(
            "Distance from the prior-bar mid-close to the prior trading-day's "
            "mid-price session low. Positive ⇒ above prior day's low. "
            "Trading-day boundary per Panel.boundary_convention (UTC or 5ers_eet)."
        ),
        inputs={"reference": "prior_trading_day_low_mid"},
    )
)


def _distance_to_round_number(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    """Pip distance from prior mid-close to the nearest round-number band.

    Round-number bands per CC_06 §5: 0.0050, 0.0100, 0.0500 (in price
    units). For JPY pairs the 0.0050/0.0100/0.0500 grid is interpreted in
    price units too (so 0.05 is 5 yen) — the feature simply finds the
    smallest distance to *any* of the three grids.

    Returned in pips (price / pip_size_for_pair).
    """
    pair = getattr(pair_df, "attrs", {}).get("pair", "EURUSD")
    pip = pip_size_for_pair(pair)
    grids = (0.0050, 0.0100, 0.0500)
    close_prior = mid_close(pair_df).shift(1)
    distances = pd.DataFrame(index=close_prior.index)
    for g in grids:
        nearest = (close_prior / g).round() * g
        distances[f"d_{g}"] = (close_prior - nearest).abs()
    min_dist = distances.min(axis=1)
    return min_dist / pip


register(
    FeatureSpec(
        name="distance_to_round_number",
        producer=_distance_to_round_number,
        lineage=CausalLineage.CLEAN,
        feature_class="distance",
        description=(
            "Minimum pip distance from prior mid-close to the nearest of three "
            "round-number grids (0.0050 / 0.0100 / 0.0500 price units)."
        ),
        inputs={"grids_price_units": [0.0050, 0.0100, 0.0500]},
    )
)
