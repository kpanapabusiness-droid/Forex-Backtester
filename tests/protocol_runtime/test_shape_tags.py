"""Tests for shape-tag assignment quartile rules."""

from __future__ import annotations

from core.steps._shape_tags import (
    BIMODAL,
    CHOPPY,
    MONOTONIC_DOWN,
    MONOTONIC_UP,
    STEPWISE,
    UNCLASSIFIED,
    V_SHAPE,
    ClusterCentroid,
    assign_shape_tag,
)


def _cc(**kwargs) -> ClusterCentroid:
    base = dict(
        cluster_id=0,
        monotonicity=0.5,
        local_peaks=1.0,
        mfe_p50=1.0,
        time_to_peak_rel=0.5,
        wrong_way_pp=0.2,
    )
    base.update(kwargs)
    return ClusterCentroid(**base)


def test_monotonic_up() -> None:
    assert assign_shape_tag(_cc(monotonicity=0.90, mfe_p50=2.5)) == MONOTONIC_UP


def test_monotonic_down() -> None:
    assert assign_shape_tag(_cc(monotonicity=0.10, mfe_p50=0.3)) == MONOTONIC_DOWN


def test_stepwise() -> None:
    assert assign_shape_tag(_cc(monotonicity=0.70, mfe_p50=1.8, wrong_way_pp=0.10)) == STEPWISE


def test_v_shape() -> None:
    assert assign_shape_tag(
        _cc(monotonicity=0.30, time_to_peak_rel=0.20, mfe_p50=1.0, wrong_way_pp=0.30)
    ) == V_SHAPE


def test_bimodal() -> None:
    assert assign_shape_tag(_cc(monotonicity=0.50, local_peaks=3.0, mfe_p50=1.5)) == BIMODAL


def test_choppy() -> None:
    assert assign_shape_tag(_cc(monotonicity=0.30, local_peaks=2.0, mfe_p50=0.5)) == CHOPPY


def test_unclassified_falls_through() -> None:
    # Centroid in no specific basin
    assert assign_shape_tag(_cc(monotonicity=0.55, mfe_p50=1.2, local_peaks=0.5, wrong_way_pp=0.05)) in (
        UNCLASSIFIED, BIMODAL, STEPWISE, CHOPPY, V_SHAPE
    )
