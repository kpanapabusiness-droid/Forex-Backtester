"""Single-feature degeneracy check per §6.

A feature is flagged if > 80% of trades sit at the single most-common value
(after rounding to 4 decimal places). Two+ flagged features halts the arc.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from scripts.arc_kh24_v2.step2._features import FEATURE_COLUMNS

DEGENERACY_THRESHOLD = 0.80
ROUND_DECIMALS = 4


class DegeneracyResult(NamedTuple):
    flags: dict[str, float]  # feature_name -> mode_fraction (only when > THRESHOLD)
    mode_fractions: dict[str, float]  # all features
    halt: bool  # True iff len(flags) >= 2


def check_degeneracy(features_df: pd.DataFrame) -> DegeneracyResult:
    flags: dict[str, float] = {}
    mode_fractions: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        s = features_df[col].round(ROUND_DECIMALS)
        if len(s) == 0:
            mode_fractions[col] = 0.0
            continue
        counts = s.value_counts(normalize=True)
        top_frac = float(counts.iloc[0])
        mode_fractions[col] = top_frac
        if top_frac > DEGENERACY_THRESHOLD:
            flags[col] = top_frac
    return DegeneracyResult(
        flags=flags,
        mode_fractions=mode_fractions,
        halt=len(flags) >= 2,
    )
