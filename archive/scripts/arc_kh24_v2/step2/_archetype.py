"""Archetype labelling for KH-24 v2.0 Step 2 — §11 centroid pattern match.

Each cluster's centroid (in unstandardized feature space) is matched against
the archetype rules in protocol §11. Rules that require forward-geometry
(`pct_peak_and_collapse`) — Early-peak hold vs Peak-and-collapse — are NOT
resolvable at Step 2; clusters matching the shared prefix
`time_to_peak_rel <= 0.30` are labelled `early_peak_family` provisionally
with `labelling_method = provisional_step3`.

V-shape recovery and Split-exit variant require Step 3 distribution shape —
not assignable here; flagged for Step 3.

Random walk: `local_peaks >= 8 AND monotonicity <= 0.30 AND pullback >= 1R`.

When a centroid matches no archetype unambiguously, the label is
`unresolved_<8hexchars>` where the hex is a short deterministic hash of the
centroid (so the label is reproducible across runs).
"""

from __future__ import annotations

import hashlib
from typing import NamedTuple


class ArchetypeLabel(NamedTuple):
    label: str
    method: str  # "centroid_match" | "provisional_step3" | "unresolved"
    notes: str  # which rules were satisfied / why


def _short_hash(values: tuple[float, ...]) -> str:
    """Deterministic 8-char hex from the centroid tuple (rounded to 4dp)."""
    rounded = tuple(round(float(v), 4) for v in values)
    h = hashlib.sha256(repr(rounded).encode("utf-8")).hexdigest()
    return h[:8]


def match_archetype(
    monotonicity: float,
    local_peaks: float,
    pullback: float,
    time_to_peak_rel: float,
) -> ArchetypeLabel:
    """Match a centroid against §11 archetype rules.

    Returns the strongest unambiguous label, or an `unresolved_*` label if
    the centroid satisfies multiple disjoint patterns or none of them.
    """
    matches: list[tuple[str, str, str]] = []  # (label, method, rule_text)

    # Monotone ascent: monotonicity >= 0.55 AND local_peaks <= 4 AND time_to_peak_rel >= 0.50
    if monotonicity >= 0.55 and local_peaks <= 4 and time_to_peak_rel >= 0.50:
        matches.append(
            (
                "monotone_ascent",
                "centroid_match",
                "monotonicity>=0.55, local_peaks<=4, time_to_peak_rel>=0.50",
            )
        )

    # Stepwise climber: monotonicity >= 0.50 AND 5 <= local_peaks <= 30 AND pullback <= 0.5R AND time_to_peak_rel >= 0.50
    if (
        monotonicity >= 0.50
        and 5 <= local_peaks <= 30
        and pullback <= 0.5
        and time_to_peak_rel >= 0.50
    ):
        matches.append(
            (
                "stepwise_climber",
                "centroid_match",
                "monotonicity>=0.50, 5<=local_peaks<=30, pullback<=0.5R, time_to_peak_rel>=0.50",
            )
        )

    # Random walk: local_peaks >= 8 AND monotonicity <= 0.30 AND pullback >= 1R
    if local_peaks >= 8 and monotonicity <= 0.30 and pullback >= 1.0:
        matches.append(
            (
                "random_walk",
                "centroid_match",
                "local_peaks>=8, monotonicity<=0.30, pullback>=1R",
            )
        )

    # Early-peak family (Early-peak hold vs Peak-and-collapse split deferred
    # to Step 3 — requires pct_peak_and_collapse from forward geometry).
    if time_to_peak_rel <= 0.30:
        matches.append(
            (
                "early_peak_family",
                "provisional_step3",
                "time_to_peak_rel<=0.30; split (Early-peak hold vs Peak-and-collapse) "
                "requires pct_peak_and_collapse at Step 3",
            )
        )

    if len(matches) == 1:
        label, method, rule = matches[0]
        return ArchetypeLabel(label=label, method=method, notes=f"matched: {rule}")

    if len(matches) > 1:
        all_labels = ", ".join(m[0] for m in matches)
        h = _short_hash((monotonicity, local_peaks, pullback, time_to_peak_rel))
        return ArchetypeLabel(
            label=f"unresolved_{h}",
            method="unresolved",
            notes=(
                f"multi-match ({all_labels}); resolve at Step 3 per §11 boundary rule "
                "(empirical per-fold capture-ratio test)"
            ),
        )

    # No match — boundary/unclassified.
    h = _short_hash((monotonicity, local_peaks, pullback, time_to_peak_rel))
    return ArchetypeLabel(
        label=f"unresolved_{h}",
        method="unresolved",
        notes=(
            "no archetype rule matched; resolve at Step 3 per §11 boundary rule "
            f"(centroid: mono={monotonicity:.3f}, peaks={local_peaks:.2f}, "
            f"pullback={pullback:.3f}, ttp={time_to_peak_rel:.3f})"
        ),
    )
