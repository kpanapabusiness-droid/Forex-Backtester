"""§2 capturability conjunctive gate — KH-24 v2.0 Step 3."""

from __future__ import annotations

import pytest

from scripts.arc_kh24_v2.step3._archetype_finalize import finalise
from scripts.arc_kh24_v2.step3._capturability import evaluate


def _passing_args(**overrides):
    base = dict(
        cluster_id=0,
        archetype_label="stepwise_climber",
        centroid_monotonicity=0.60,
        centroid_local_peaks=10.0,
        fwd_mfe_p50=2.0,
        frac_reach_1R=0.85,
        frac_wrong_way=0.10,
        shape_tag="heavy_right_tail",
        size_fraction_of_pool=0.20,
    )
    base.update(overrides)
    return base


def test_all_pass():
    e = evaluate(**_passing_args())
    assert e.overall_pass


@pytest.mark.parametrize(
    "kw",
    [
        {"centroid_monotonicity": 0.54},  # clean_shape fails
        {"fwd_mfe_p50": 1.49},  # magnitude fails
        {"frac_reach_1R": 0.69},  # direction (1R)
        {"frac_wrong_way": 0.31},  # direction (wrong-way)
        {"shape_tag": "bimodal"},  # internal consistency
        {"shape_tag": "scattered"},
        {"shape_tag": "no_magnitude"},
        {"size_fraction_of_pool": 0.09},  # size
    ],
)
def test_conjunctive_one_criterion_breaks_overall(kw):
    """Failing exactly one criterion must FAIL the overall AND (not OR)."""
    e = evaluate(**_passing_args(**kw))
    assert not e.overall_pass, f"Failing {list(kw.keys())[0]} should fail overall"
    assert e.fail_reasons


def test_limited_oscillation_stepwise_range():
    # 31 > 30 ceiling for stepwise_climber → limited_oscillation fails.
    e = evaluate(**_passing_args(centroid_local_peaks=31.0))
    assert not e.limited_oscillation_pass
    assert not e.overall_pass


def test_limited_oscillation_monotone_ascent_ceiling():
    # mono_ascent ceiling is 4; 5 violates.
    e = evaluate(
        **_passing_args(
            archetype_label="monotone_ascent",
            centroid_local_peaks=5.0,
            # Hold other §2 criteria PASS:
            centroid_monotonicity=0.60,
        )
    )
    assert not e.limited_oscillation_pass


def test_limited_oscillation_early_peak_family_unconstrained():
    # Early-peak family has no explicit ceiling — high local_peaks is allowed.
    e = evaluate(**_passing_args(archetype_label="early_peak_family", centroid_local_peaks=50.0))
    assert e.limited_oscillation_pass


def test_limited_oscillation_unresolved_unconstrained():
    e = evaluate(**_passing_args(archetype_label="unresolved_abc12345", centroid_local_peaks=50.0))
    assert e.limited_oscillation_pass


# ----- Archetype finalisation -----


def test_finalise_early_peak_split_to_hold():
    out = finalise(
        provisional_label="early_peak_family",
        provisional_method="provisional_step3",
        pct_peak_and_collapse=0.10,
        survives_step2_floors=True,
    )
    assert out.label == "early_peak_hold"
    assert out.method == "pct_peak_and_collapse_split"


def test_finalise_early_peak_split_to_collapse():
    out = finalise(
        provisional_label="early_peak_family",
        provisional_method="provisional_step3",
        pct_peak_and_collapse=0.70,
        survives_step2_floors=True,
    )
    assert out.label == "peak_and_collapse"
    assert out.method == "pct_peak_and_collapse_split"


def test_finalise_early_peak_ambiguous_kept_family():
    out = finalise(
        provisional_label="early_peak_family",
        provisional_method="provisional_step3",
        pct_peak_and_collapse=0.35,
        survives_step2_floors=True,
    )
    assert out.label == "early_peak_family"
    assert out.method == "ambiguous_pct_pc"


def test_finalise_unresolved_passing_step2_deferred_to_step5():
    out = finalise(
        provisional_label="unresolved_deadbeef",
        provisional_method="unresolved",
        pct_peak_and_collapse=0.0,
        survives_step2_floors=True,
    )
    assert out.label == "unresolved_deadbeef"
    assert out.method == "deferred_empirical_step5"


def test_finalise_unresolved_failing_step2_no_test():
    out = finalise(
        provisional_label="unresolved_deadbeef",
        provisional_method="unresolved",
        pct_peak_and_collapse=0.0,
        survives_step2_floors=False,
    )
    assert out.label == "unresolved_deadbeef"
    assert out.method == "centroid_best_match_no_test"


def test_finalise_centroid_match_unchanged():
    out = finalise(
        provisional_label="stepwise_climber",
        provisional_method="centroid_match",
        pct_peak_and_collapse=0.4,
        survives_step2_floors=True,
    )
    assert out.label == "stepwise_climber"
    assert out.method == "centroid_match"
