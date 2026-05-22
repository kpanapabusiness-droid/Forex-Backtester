"""Session / time-of-day / day-of-week features.

UTC session windows (FX market convention):
    Tokyo:    00:00 – 09:00
    London:   07:00 – 16:00
    NY:       12:00 – 21:00
    LDN-NY overlap: 12:00 – 16:00
    Dead zone (Asian post-close to London pre-open): 21:00 – 07:00 next day

A bar can be in multiple sessions (overlap). The session features below
emit boolean (0/1) flags — independent indicators, not a one-hot. The
``session_label`` feature emits a single canonical category for the bar.

Causal lineage: clean — pure function of the bar's timestamp.
"""

from __future__ import annotations

import pandas as pd

from core.features.lineage import CausalLineage, FeatureSpec
from core.features.registry import register


def _hour_of_day(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    return pd.Series(pair_df.index.hour, index=pair_df.index, name="hour_of_day").astype("int64")


register(
    FeatureSpec(
        name="hour_of_day",
        producer=_hour_of_day,
        lineage=CausalLineage.CLEAN,
        feature_class="session",
        description="UTC hour of the bar's left edge (0-23).",
        inputs={},
    )
)


def _day_of_week(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    """Monday = 0, Sunday = 6 (pandas convention)."""
    return pd.Series(pair_df.index.dayofweek, index=pair_df.index, name="day_of_week").astype(
        "int64"
    )


register(
    FeatureSpec(
        name="day_of_week",
        producer=_day_of_week,
        lineage=CausalLineage.CLEAN,
        feature_class="session",
        description="UTC weekday (Mon=0..Sun=6).",
        inputs={},
    )
)


def _in_window(hours: pd.Index, lo: int, hi: int) -> pd.Series:
    """Hours [lo, hi) — bool Series."""
    return pd.Series(((hours.hour >= lo) & (hours.hour < hi)), index=hours).astype("int64")


def _session_tokyo(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    return _in_window(pair_df.index, 0, 9)


def _session_london(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    return _in_window(pair_df.index, 7, 16)


def _session_ny(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    return _in_window(pair_df.index, 12, 21)


def _session_ldn_ny_overlap(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    return _in_window(pair_df.index, 12, 16)


def _session_dead(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    """Asian post-close → London pre-open: 21:00 – 07:00 (wrapping)."""
    h = pair_df.index.hour
    return pd.Series(((h >= 21) | (h < 7)), index=pair_df.index).astype("int64")


for name, producer, desc in [
    ("session_tokyo", _session_tokyo, "Tokyo session flag (UTC 00-09)."),
    ("session_london", _session_london, "London session flag (UTC 07-16)."),
    ("session_ny", _session_ny, "NY session flag (UTC 12-21)."),
    ("session_ldn_ny_overlap", _session_ldn_ny_overlap, "London/NY overlap flag (UTC 12-16)."),
    ("session_dead", _session_dead, "Dead-zone flag — UTC 21-07 (wraps midnight)."),
]:
    register(
        FeatureSpec(
            name=name,
            producer=producer,
            lineage=CausalLineage.CLEAN,
            feature_class="session",
            description=desc,
            inputs={},
        )
    )
