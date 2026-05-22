"""Tests for core.discovery.bonferroni — accounting + top-K ranking + survivors."""

from __future__ import annotations

from core.discovery.bonferroni import (
    bonferroni_survivors,
    build_bonferroni_report,
    rank_top_k,
)


def test_build_report_primary_uses_n_evaluated():
    report = build_bonferroni_report(
        n_generated=10000,
        n_evaluated=7000,
        n_causal_rejected=2500,
        n_pool_floor_rejected=500,
        alpha=0.05,
    )
    assert abs(report.threshold_primary - 0.05 / 7000) < 1e-12
    assert abs(report.threshold_budget - 0.05 / 10000) < 1e-12
    assert report.n_other_rejected == 0


def test_build_report_zero_evaluated():
    report = build_bonferroni_report(
        n_generated=100,
        n_evaluated=0,
        n_causal_rejected=100,
        n_pool_floor_rejected=0,
    )
    # NaN -- no rules survived to evaluation.
    assert report.threshold_primary != report.threshold_primary  # NaN check


def test_rank_top_k_by_mean_r_desc():
    rows = [
        {"rule_id": 1, "pool_size": 300, "mean_r": 0.10, "p_value": 0.04},
        {"rule_id": 2, "pool_size": 300, "mean_r": 0.50, "p_value": 0.001},
        {"rule_id": 3, "pool_size": 300, "mean_r": 0.30, "p_value": 0.01},
        {"rule_id": 4, "pool_size": 0,   "mean_r": float("nan"), "p_value": None},  # rejected
        {"rule_id": 5, "pool_size": 300, "mean_r": 0.20, "p_value": 0.005},
    ]
    report = build_bonferroni_report(
        n_generated=10, n_evaluated=4, n_causal_rejected=0, n_pool_floor_rejected=1
    )
    ranked = rank_top_k(rows, k=3, report=report, follow_up_top_k=2)
    assert [r.rule_id for r in ranked] == [2, 3, 5]
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert ranked[0].follow_up_eligible is True   # rank 1
    assert ranked[1].follow_up_eligible is True   # rank 2
    assert ranked[2].follow_up_eligible is False  # rank 3 — analysis-only


def test_rank_excludes_rejected_rules():
    rows = [
        {"rule_id": 1, "pool_size": 0, "mean_r": float("nan"), "p_value": None},
        {"rule_id": 2, "pool_size": 100, "mean_r": 0.5, "p_value": 0.001},
    ]
    report = build_bonferroni_report(
        n_generated=2, n_evaluated=1, n_causal_rejected=1, n_pool_floor_rejected=0
    )
    ranked = rank_top_k(rows, k=10, report=report)
    assert len(ranked) == 1
    assert ranked[0].rule_id == 2


def test_bonferroni_survivors_strict():
    rows = [
        {"rule_id": 1, "pool_size": 300, "p_value": 1e-7},   # passes 5e-6
        {"rule_id": 2, "pool_size": 300, "p_value": 1e-5},   # fails 5e-6 (primary at 10000=5e-6)
        {"rule_id": 3, "pool_size": 0,   "p_value": None},   # rejected
        {"rule_id": 4, "pool_size": 300, "p_value": 2e-7},
    ]
    report = build_bonferroni_report(
        n_generated=10000, n_evaluated=10000, n_causal_rejected=0, n_pool_floor_rejected=0
    )
    survivors = bonferroni_survivors(rows, report)
    # primary threshold = 0.05/10000 = 5e-6 -> rule_id 1 and 4 survive.
    assert survivors == (1, 4)


def test_bonferroni_pass_flags_on_ranked():
    rows = [
        {"rule_id": 1, "pool_size": 300, "mean_r": 0.5, "p_value": 1e-7},
        {"rule_id": 2, "pool_size": 300, "mean_r": 0.4, "p_value": 1e-3},
    ]
    report = build_bonferroni_report(
        n_generated=10000, n_evaluated=10000, n_causal_rejected=0, n_pool_floor_rejected=0
    )
    ranked = rank_top_k(rows, k=2, report=report)
    assert ranked[0].bonferroni_pass_primary is True
    assert ranked[0].bonferroni_pass_budget is True
    assert ranked[1].bonferroni_pass_primary is False
    assert ranked[1].bonferroni_pass_budget is False
