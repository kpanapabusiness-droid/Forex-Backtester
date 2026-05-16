"""Finalise per-cluster archetype labels at Step 3.

Inputs from Step 2 + Step 3:
  - provisional label from Step 2 (centroid_match / provisional_step3 / unresolved)
  - pct_peak_and_collapse (per cluster, computed at Step 3)
  - §2 capturability pass/fail per cluster (computed at Step 3)

Resolution rules (per §11 + the Step 3 prompt):

  Provisional `early_peak_family` (Step 2 sentinel):
      pct_peak_and_collapse < 0.30  → "early_peak_hold"
      pct_peak_and_collapse >= 0.50 → "peak_and_collapse"
      otherwise                     → kept as "early_peak_family" with note
                                       (resolution by §11 empirical test
                                        deferred to Step 5+ folds — at
                                        Step 3 we have no per-fold inputs)

  Provisional `unresolved_*` (Step 2 boundary):
      Failed §2 → kept as `unresolved_*` (out of further analysis;
                  labelling-method = "centroid_best_match_no_test")
      Passed §2 → deferred to Step 5/6 per-fold capture-ratio test
                  (per §11 boundary rule). At Step 3 we record the closest
                  §11 archetype by centroid distance and flag for empirical
                  resolution. labelling-method = "deferred_empirical_step5".

  centroid_match labels: no change.
"""

from __future__ import annotations

from typing import NamedTuple

PCT_PC_LOW = 0.30
PCT_PC_HIGH = 0.50


class FinalLabel(NamedTuple):
    label: str
    method: str
    notes: str


def finalise(
    provisional_label: str,
    provisional_method: str,
    pct_peak_and_collapse: float,
    survives_step2_floors: bool,
) -> FinalLabel:
    """Apply Step 3 label finalisation.

    `provisional_method` is one of {centroid_match, provisional_step3, unresolved}.
    """
    if provisional_method == "centroid_match":
        return FinalLabel(
            label=provisional_label,
            method="centroid_match",
            notes="label finalised at Step 2; no Step 3 change",
        )

    if provisional_method == "provisional_step3":
        # Was `early_peak_family` at Step 2 — split now via pct_peak_and_collapse.
        if pct_peak_and_collapse < PCT_PC_LOW:
            return FinalLabel(
                label="early_peak_hold",
                method="pct_peak_and_collapse_split",
                notes=f"pct_peak_and_collapse={pct_peak_and_collapse:.3f} < {PCT_PC_LOW:.2f}",
            )
        if pct_peak_and_collapse >= PCT_PC_HIGH:
            return FinalLabel(
                label="peak_and_collapse",
                method="pct_peak_and_collapse_split",
                notes=f"pct_peak_and_collapse={pct_peak_and_collapse:.3f} >= {PCT_PC_HIGH:.2f}",
            )
        return FinalLabel(
            label="early_peak_family",
            method="ambiguous_pct_pc",
            notes=(
                f"pct_peak_and_collapse={pct_peak_and_collapse:.3f} in [{PCT_PC_LOW:.2f}, {PCT_PC_HIGH:.2f}); "
                "§11 empirical per-fold test deferred to Step 5+ (no folds at Step 3)"
            ),
        )

    if provisional_method == "unresolved":
        if survives_step2_floors:
            return FinalLabel(
                label=provisional_label,  # kept; awaits empirical resolution
                method="deferred_empirical_step5",
                notes=(
                    "survived §2 floors but boundary; §11 empirical "
                    "per-fold capture-ratio test deferred to Step 5+"
                ),
            )
        return FinalLabel(
            label=provisional_label,
            method="centroid_best_match_no_test",
            notes="failed §2 floors; no empirical resolution needed (out)",
        )

    # Fallback for unknown methods.
    return FinalLabel(
        label=provisional_label,
        method=provisional_method,
        notes="unknown provisional method; label unchanged",
    )
