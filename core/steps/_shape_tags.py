"""Shape-tag assignment per L_PROTOCOL §2 Step 2.

A cluster's archetype label is derived from quartile rules on its
centroid in path-feature space:

  - monotonicity (∈ [0, 1])
  - local peaks (count, normalised by bars_held)
  - mfe_p50 (R)
  - time_to_peak_rel (∈ [0, 1] — ttp / bars_held)
  - wrong_way_pp (∈ [0, 1] — fraction hitting -1R MAE before +1R MFE)

Tags assigned:

  - V-shape recovery:     low monotonicity, low ttp_rel, mfe_p50 > 0, ww_pp ≥ 0.20
  - Stepwise climber:     high monotonicity, mfe_p50 > 1.5R, ww_pp < 0.20
  - Bimodal:              moderate monotonicity, 2+ local peaks, mfe_p50 > 1R
  - Monotonic up:         very high monotonicity, mfe_p50 > 2R
  - Monotonic down:       very low monotonicity, mfe_p50 < 0.5R
  - Choppy:               low monotonicity, high local peaks, mfe_p50 < 1R
  - Unclassified:         doesn't fit any above

The exact thresholds are defined here (locked at v3.0). Per
L_PROTOCOL §1, within-arc thresholds are immutable; cross-arc
recalibration only.
"""

from __future__ import annotations

from dataclasses import dataclass

V_SHAPE = "v_shape_recovery"
STEPWISE = "stepwise_climber"
BIMODAL = "bimodal"
MONOTONIC_UP = "monotonic_up"
MONOTONIC_DOWN = "monotonic_down"
CHOPPY = "choppy"
UNCLASSIFIED = "unclassified"

TAG_ORDER = (
    V_SHAPE,
    STEPWISE,
    BIMODAL,
    MONOTONIC_UP,
    MONOTONIC_DOWN,
    CHOPPY,
    UNCLASSIFIED,
)


@dataclass(frozen=True)
class ClusterCentroid:
    """Centroid representation an archetype classifier consumes."""

    cluster_id: int
    monotonicity: float
    local_peaks: float  # normalised count or raw — caller's choice, used relative
    mfe_p50: float
    time_to_peak_rel: float
    wrong_way_pp: float


def assign_shape_tag(centroid: ClusterCentroid) -> str:
    """Return one of :data:`TAG_ORDER` for ``centroid``.

    Rules applied in order; the first matching rule wins. Unmatched
    centroids return :data:`UNCLASSIFIED`.
    """
    m = centroid.monotonicity
    p = centroid.local_peaks
    mfe = centroid.mfe_p50
    ttp = centroid.time_to_peak_rel
    ww = centroid.wrong_way_pp

    # Monotonic up: very monotone + strong MFE
    if m >= 0.85 and mfe >= 2.0:
        return MONOTONIC_UP
    # Monotonic down: barely any in-profit bars + anaemic MFE
    if m <= 0.15 and mfe <= 0.5:
        return MONOTONIC_DOWN
    # Stepwise climber: monotone + decent MFE + low ww
    if m >= 0.60 and mfe >= 1.5 and ww < 0.20:
        return STEPWISE
    # V-shape: low monotonicity, early peak, decent MFE, meaningful ww
    if m < 0.50 and ttp < 0.40 and mfe > 0.5 and ww >= 0.20:
        return V_SHAPE
    # Bimodal: moderate monotonicity, multiple peaks, decent MFE
    if 0.30 <= m < 0.70 and p >= 2.0 and mfe > 1.0:
        return BIMODAL
    # Choppy: low monotonicity + many peaks + anaemic MFE
    if m < 0.40 and p >= 1.5 and mfe < 1.0:
        return CHOPPY
    return UNCLASSIFIED


__all__ = (
    "V_SHAPE", "STEPWISE", "BIMODAL", "MONOTONIC_UP", "MONOTONIC_DOWN",
    "CHOPPY", "UNCLASSIFIED", "TAG_ORDER",
    "ClusterCentroid", "assign_shape_tag",
)
