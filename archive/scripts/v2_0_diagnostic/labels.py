"""v2.0 archetype diagnostic — descriptive shape labels.

Labels are assigned from path-shape centroid values. Mechanical mapping;
no speculation. See spec §7 §4.

  Monotone ascent  high monotonicity, low local_peaks, low pullback
  Stepwise climber  medium peaks + medium pullback, generally rising
  Inverted-V        peak early (low time_to_peak_relative) with notable
                    pullback (>= 1R)
  Random walk       high local_peaks, high pullback, low monotonicity
  Flat-line         all features near zero
  Unlabelled        none of the above match
"""
from __future__ import annotations

import pandas as pd


def label_archetype(row: pd.Series) -> str:
    mono     = float(row["monotonicity_centroid"])
    peaks    = float(row["local_peaks_centroid"])
    pull     = float(row["pullback_magnitude_centroid"])
    ttp_rel  = float(row["time_to_peak_relative_centroid"])

    # Flat-line: minimal path activity overall (few peaks, low mono).
    if peaks <= 2.0 and mono < 0.30:
        return "Flat-line"

    # Random walk: lots of peaks, low monotonicity. Pullback assist when available.
    if peaks >= 10 and mono < 0.50 and pull >= 1.0:
        return "Random walk"

    # Inverted-V: peak early (low ttp_rel), notable pullback.
    if ttp_rel < 0.35 and pull >= 1.0:
        return "Inverted-V"

    # Monotone ascent: high monotonicity, few-to-moderate peaks, low pullback.
    if mono >= 0.55 and peaks <= 6.0 and pull <= 0.8:
        return "Monotone ascent"

    # Stepwise climber: many peaks, still rising on net (medium-high monotonicity).
    if peaks > 6.0 and mono >= 0.40:
        return "Stepwise climber"

    return "Unlabelled"


def label_all(summary: pd.DataFrame) -> pd.Series:
    return summary.apply(label_archetype, axis=1)
