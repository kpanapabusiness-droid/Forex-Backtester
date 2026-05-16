"""§2 capturability gate evaluation (conjunctive).

Six criteria, all must pass per §1.8 / §2:

  1. Clean shape          monotonicity_centroid >= 0.55
  2. Limited oscillation  local_peaks_centroid within shape-appropriate
                          ceiling (per §11 archetype row)
  3. Magnitude            fwd_mfe_h240_p50 >= 1.5R
  4. Direction            frac_reach_1R >= 0.70 AND frac_wrong_way <= 0.30
  5. Internal consistency shape_tag ∈ {tight_unimodal, heavy_right_tail}
  6. Size viability       size_fraction_of_pool >= 0.10

For criterion 4, this module uses Definition B of `frac_wrong_way`
("wrong from outset") per the §17 intent statement. Definition A is
reported alongside but does not gate.

For criterion 2, the archetype-specific local_peaks ceiling is:

  monotone_ascent    : <= 4         (per §11)
  stepwise_climber   : 5..30        (per §11)
  early_peak_family  : no explicit ceiling (per §11 — pairs with low ttp)
  peak_and_collapse  : no explicit ceiling
  random_walk        : >= 8 (lower bound only)
  unresolved_*       : check both bracket halves; pass if either fits

The 6-tuple of booleans is recorded individually so the report shows
which criterion specifically killed each cluster.
"""

from __future__ import annotations

from typing import NamedTuple

CAPTURABILITY_THRESHOLDS = {
    "min_monotonicity": 0.55,
    "min_fwd_mfe_p50": 1.5,
    "min_frac_reach_1R": 0.70,
    "max_frac_wrong_way": 0.30,
    "min_size_fraction": 0.10,
}

CLEAN_SHAPE_TAGS = ("tight_unimodal", "heavy_right_tail")


def _local_peaks_in_archetype_range(local_peaks: float, archetype_label: str) -> tuple[bool, str]:
    """Return (in_range, rule_text). Returns True for archetypes that don't
    constrain local_peaks (early-peak family / unresolved).
    """
    lbl = archetype_label.lower()
    if lbl == "monotone_ascent":
        return (local_peaks <= 4.0, "local_peaks<=4 (§11 Monotone ascent)")
    if lbl == "stepwise_climber":
        return (5.0 <= local_peaks <= 30.0, "5<=local_peaks<=30 (§11 Stepwise climber)")
    if lbl in ("early_peak_family", "early_peak_hold", "peak_and_collapse"):
        return (True, "no explicit ceiling (§11 Early-peak family pairs with ttp<=0.30)")
    if lbl == "random_walk":
        return (local_peaks >= 8.0, "local_peaks>=8 (§11 Random walk)")
    if lbl == "v_shape_recovery":
        return (True, "no explicit ceiling (§11 V-shape recovery — defer to Step 3 emp.)")
    if lbl.startswith("unresolved_"):
        # Boundary cluster — don't impose a ceiling, let the other criteria
        # decide. If it survives, §11 empirical test resolves the label.
        return (True, "unresolved (boundary) — no archetype-specific ceiling at §2")
    return (True, f"unknown archetype {archetype_label!r} — no ceiling imposed")


class CapturabilityEvaluation(NamedTuple):
    cluster_id: int
    clean_shape_pass: bool
    limited_oscillation_pass: bool
    magnitude_pass: bool
    direction_pass: bool
    shape_pass: bool
    size_pass: bool
    overall_pass: bool
    fail_reasons: list[str]
    notes: dict


def evaluate(
    *,
    cluster_id: int,
    archetype_label: str,
    centroid_monotonicity: float,
    centroid_local_peaks: float,
    fwd_mfe_p50: float,
    frac_reach_1R: float,
    frac_wrong_way: float,  # Definition B per §2 intent
    shape_tag: str,
    size_fraction_of_pool: float,
) -> CapturabilityEvaluation:
    thr = CAPTURABILITY_THRESHOLDS
    notes: dict = {}
    fails: list[str] = []

    clean_shape = centroid_monotonicity >= thr["min_monotonicity"]
    notes["clean_shape"] = {"value": centroid_monotonicity, "threshold": thr["min_monotonicity"]}
    if not clean_shape:
        fails.append(
            f"clean_shape: monotonicity_centroid={centroid_monotonicity:.4f} < {thr['min_monotonicity']:.2f}"
        )

    lp_ok, lp_rule = _local_peaks_in_archetype_range(centroid_local_peaks, archetype_label)
    notes["limited_oscillation"] = {
        "value": centroid_local_peaks,
        "rule": lp_rule,
    }
    if not lp_ok:
        fails.append(
            f"limited_oscillation: local_peaks_centroid={centroid_local_peaks:.2f}; rule={lp_rule}"
        )

    magnitude = fwd_mfe_p50 >= thr["min_fwd_mfe_p50"]
    notes["magnitude"] = {"value": fwd_mfe_p50, "threshold": thr["min_fwd_mfe_p50"]}
    if not magnitude:
        fails.append(f"magnitude: fwd_mfe_p50={fwd_mfe_p50:.3f}R < {thr['min_fwd_mfe_p50']:.2f}R")

    dir_ok_1r = frac_reach_1R >= thr["min_frac_reach_1R"]
    dir_ok_wrong = frac_wrong_way <= thr["max_frac_wrong_way"]
    direction = dir_ok_1r and dir_ok_wrong
    notes["direction"] = {
        "frac_reach_1R": frac_reach_1R,
        "frac_wrong_way": frac_wrong_way,
        "thresholds": {
            "min_frac_reach_1R": thr["min_frac_reach_1R"],
            "max_frac_wrong_way": thr["max_frac_wrong_way"],
        },
    }
    if not dir_ok_1r:
        fails.append(
            f"direction: frac_reach_1R={frac_reach_1R:.3f} < {thr['min_frac_reach_1R']:.2f}"
        )
    if not dir_ok_wrong:
        fails.append(
            f"direction: frac_wrong_way={frac_wrong_way:.3f} > {thr['max_frac_wrong_way']:.2f}"
        )

    shape_ok = shape_tag in CLEAN_SHAPE_TAGS
    notes["shape"] = {"value": shape_tag, "allowed": CLEAN_SHAPE_TAGS}
    if not shape_ok:
        fails.append(f"shape: shape_tag={shape_tag!r} not in {set(CLEAN_SHAPE_TAGS)}")

    size_ok = size_fraction_of_pool >= thr["min_size_fraction"]
    notes["size"] = {"value": size_fraction_of_pool, "threshold": thr["min_size_fraction"]}
    if not size_ok:
        fails.append(
            f"size: size_fraction={size_fraction_of_pool:.3f} < {thr['min_size_fraction']:.2f}"
        )

    overall = clean_shape and lp_ok and magnitude and direction and shape_ok and size_ok

    return CapturabilityEvaluation(
        cluster_id=cluster_id,
        clean_shape_pass=clean_shape,
        limited_oscillation_pass=lp_ok,
        magnitude_pass=magnitude,
        direction_pass=direction,
        shape_pass=shape_ok,
        size_pass=size_ok,
        overall_pass=overall,
        fail_reasons=fails,
        notes=notes,
    )
