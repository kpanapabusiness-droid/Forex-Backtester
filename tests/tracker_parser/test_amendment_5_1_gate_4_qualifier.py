"""L_PROTOCOL Amendment 5.1 (2026-05-25) — Gate 4 PASS-tier-constituent qualifier.

Coverage:
- ``AMENDMENT_5_1_CUTOFF_ISO`` constant pinned at PR #201 merge timestamp
- Schema accepts the reason string ``a5_gate_4_admission_blocked_by_no_pass_tier_constituent``
  as an entry in ``architectures_skipped_by_amendment_5`` (alongside architecture IDs)
- Unknown reason strings still rejected
- ``_is_post_amendment_5_1_cutoff`` cutoff semantics
- ``_validate_amendment_5_1_gate_4_qualifier`` Phase 1 WARNING-level behaviour:
    - Post-cutoff PASS, ≥2 candidate clusters, A5 not admitted, no reason string → WARNING emitted
    - Post-cutoff PASS, ≥2 candidate clusters, A5 not admitted, reason string present → no warning
    - Post-cutoff PASS, ≥2 candidate clusters, A5 admitted → no warning
    - Pre-cutoff PASS, ≥2 candidate clusters, no qualifier string → no warning (caller gates)
    - Post-cutoff FAIL closure → no warning (caller gates on PASS)
    - Post-cutoff PASS, <2 candidate clusters → no warning
- Phase 1 contract: WARNING-level never HALTs (returns 0)
"""

from __future__ import annotations

import logging

import pytest

from scripts.tracker_parser.schema import (
    AMENDMENT_5_1_CUTOFF_ISO,
    parse_payload,
)
from scripts.update_tracker_from_closure import (
    _A5_1_GATE_4_REASON,
    _count_candidate_clusters_surviving_step_3,
    _is_post_amendment_5_1_cutoff,
    _validate_amendment_5_1_gate_4_qualifier,
)

_BASE_PAYLOAD: dict = {
    "template_version": "v1.3",
    "arc_name": "test_arc_5_1",
    "signal": "x",
    "tf": "H4",
    "sub_protocol": "vanilla",
    "closed_timestamp": "2026-05-26T12:00:00Z",
    "closure_doc_link": "results/test_arc_5_1/ARC_CLOSURE.md",
    "verdict": "FAIL",
    "one_line": "x",
    "failed_at_step": "5",
    "primary_failure_mode": "step5_not_scalable",
    "pool_metadata": {
        "total_n": 500,
        "window_start": "2010",
        "window_end": "2020",
        "configs_evaluated_step5": 80,
        "search_scope_flag": "normal",
    },
    "clusters": {},
    "architectures_tested": ["A1"],
    "architecture_results": {"A1": {"tested": True, "won": False, "worst_fold_ratio": 1.0}},
    "archetypes_observed": [],
}


def _candidate_cluster(extra: dict | None = None) -> dict:
    """Build a cluster dict that satisfies §3 candidate criteria
    (reach_1r >= 0.50, ww_pp <= 0.30, mfe_p50_r >= 1.5).
    """
    base = {
        "n": 100,
        "archetype": "V-shape",
        "sl_atr": 2.0,
        "step3_composite": 0.8,
        "mfe_p50_r": 2.0,
        "ww_pp": 0.2,
        "reach_1r": 0.7,
        "outcome": "passed_step3",
    }
    if extra:
        base.update(extra)
    return base


def _non_candidate_cluster() -> dict:
    """Cluster that fails the §3 candidate criteria (ww_pp too high)."""
    return {
        "n": 100,
        "archetype": "Monotonic_down",
        "sl_atr": 2.0,
        "step3_composite": 0.2,
        "mfe_p50_r": 0.8,
        "ww_pp": 0.6,
        "reach_1r": 0.3,
        "outcome": "dies_step3",
    }


# ── Schema vocabulary tests ────────────────────────────────────────────────


def test_amendment_5_1_cutoff_locked():
    """Cutoff backfilled with PR #201 merge timestamp."""
    assert AMENDMENT_5_1_CUTOFF_ISO == "2026-05-25T05:29:01Z"


def test_schema_accepts_amendment_5_1_reason_string():
    """The Gate 4 PASS-tier-constituent reason string is an accepted entry."""
    payload = dict(_BASE_PAYLOAD)
    payload["architectures_skipped_by_amendment_5"] = [
        "a5_gate_4_admission_blocked_by_no_pass_tier_constituent",
    ]
    norm = parse_payload(payload)
    assert norm["architectures_skipped_by_amendment_5"] == [
        "a5_gate_4_admission_blocked_by_no_pass_tier_constituent",
    ]


def test_schema_accepts_mixed_architecture_ids_and_reason_strings():
    """Entries may mix {A1..A6} architecture IDs and known reason strings."""
    payload = dict(_BASE_PAYLOAD)
    payload["architectures_skipped_by_amendment_5"] = [
        "A2",
        "a5_gate_4_admission_blocked_by_no_pass_tier_constituent",
    ]
    norm = parse_payload(payload)
    assert norm["architectures_skipped_by_amendment_5"] == [
        "A2",
        "a5_gate_4_admission_blocked_by_no_pass_tier_constituent",
    ]


def test_schema_rejects_unknown_reason_string():
    """Unknown reason strings still rejected — closed-vocabulary discipline preserved."""
    payload = dict(_BASE_PAYLOAD)
    payload["architectures_skipped_by_amendment_5"] = ["arbitrary_unknown_reason_string"]
    with pytest.raises(ValueError, match="architectures_skipped_by_amendment_5"):
        parse_payload(payload)


# ── Cutoff helper tests ────────────────────────────────────────────────────


def test_is_post_amendment_5_1_cutoff_true_when_after():
    """Closures strictly after the cutoff are post-cutoff."""
    assert _is_post_amendment_5_1_cutoff("2026-05-26T00:00:00Z") is True
    assert _is_post_amendment_5_1_cutoff("2026-05-25T12:00:00Z") is True


def test_is_post_amendment_5_1_cutoff_false_when_at_or_before():
    """Equality and strictly-before are pre-cutoff (grandfathered)."""
    assert _is_post_amendment_5_1_cutoff("2026-05-25T05:29:01Z") is False
    assert _is_post_amendment_5_1_cutoff("2026-05-25T05:29:00Z") is False


def test_is_post_amendment_5_1_cutoff_missing_timestamp_grandfathered():
    """Missing / malformed timestamps grandfathered — never trip the gate."""
    assert _is_post_amendment_5_1_cutoff(None) is False
    assert _is_post_amendment_5_1_cutoff("not-an-iso-string") is False


# ── Candidate-cluster counting tests ───────────────────────────────────────


def test_count_candidate_clusters_zero_when_no_clusters():
    assert _count_candidate_clusters_surviving_step_3({"clusters": {}}) == 0
    assert _count_candidate_clusters_surviving_step_3({}) == 0


def test_count_candidate_clusters_excludes_non_candidates():
    payload = {
        "clusters": {
            "c0": _candidate_cluster(),
            "c1": _non_candidate_cluster(),
        }
    }
    assert _count_candidate_clusters_surviving_step_3(payload) == 1


def test_count_candidate_clusters_counts_two():
    payload = {
        "clusters": {
            "c0": _candidate_cluster(),
            "c1": _candidate_cluster(),
        }
    }
    assert _count_candidate_clusters_surviving_step_3(payload) == 2


# ── CLI validator tests ────────────────────────────────────────────────────


def test_validator_warns_when_two_candidates_no_a5_no_reason(tmp_path, caplog):
    """Post-cutoff PASS, ≥2 candidate clusters, A5 not admitted, no reason string → WARNING."""
    payload = {
        "closed_timestamp": "2026-06-01T00:00:00Z",
        "verdict": "PASS-DEPLOYABLE",
        "clusters": {"c0": _candidate_cluster(), "c1": _candidate_cluster()},
        "architectures_tested": ["A1", "A2"],
        "architecture_results": {
            "A1": {"tested": True, "won": False, "worst_fold_ratio": 1.0},
            "A2": {"tested": True, "won": True, "worst_fold_ratio": 2.5},
        },
        "architectures_skipped_by_amendment_5": [],
    }
    with caplog.at_level(logging.WARNING):
        rc = _validate_amendment_5_1_gate_4_qualifier(payload, tmp_path / "ARC_CLOSURE.md")
    assert rc == 0  # Phase 1 never HALTs
    assert any(_A5_1_GATE_4_REASON in r.message for r in caplog.records)
    assert any("Amendment 5.1" in r.message for r in caplog.records)


def test_validator_silent_when_reason_string_cited(tmp_path, caplog):
    """Post-cutoff PASS, ≥2 candidate clusters, A5 not admitted, reason string present → no warning."""
    payload = {
        "closed_timestamp": "2026-06-01T00:00:00Z",
        "verdict": "PASS-DEPLOYABLE",
        "clusters": {"c0": _candidate_cluster(), "c1": _candidate_cluster()},
        "architectures_tested": ["A1", "A2"],
        "architecture_results": {
            "A1": {"tested": True, "won": False, "worst_fold_ratio": 1.0},
            "A2": {"tested": True, "won": True, "worst_fold_ratio": 2.5},
        },
        "architectures_skipped_by_amendment_5": [_A5_1_GATE_4_REASON],
    }
    with caplog.at_level(logging.WARNING):
        rc = _validate_amendment_5_1_gate_4_qualifier(payload, tmp_path / "ARC_CLOSURE.md")
    assert rc == 0
    assert not any("Amendment 5.1" in r.message for r in caplog.records)


def test_validator_silent_when_a5_admitted(tmp_path, caplog):
    """Post-cutoff PASS, ≥2 candidate clusters, A5 admitted and ran → no warning."""
    payload = {
        "closed_timestamp": "2026-06-01T00:00:00Z",
        "verdict": "PASS-DEPLOYABLE",
        "clusters": {"c0": _candidate_cluster(), "c1": _candidate_cluster()},
        "architectures_tested": ["A1", "A5"],
        "architecture_results": {
            "A1": {"tested": True, "won": False, "worst_fold_ratio": 1.0},
            "A5": {"tested": True, "won": True, "worst_fold_ratio": 2.8},
        },
        "architectures_skipped_by_amendment_5": [],
    }
    with caplog.at_level(logging.WARNING):
        rc = _validate_amendment_5_1_gate_4_qualifier(payload, tmp_path / "ARC_CLOSURE.md")
    assert rc == 0
    assert not any("Amendment 5.1" in r.message for r in caplog.records)


def test_validator_silent_when_fewer_than_two_candidates(tmp_path, caplog):
    """Post-cutoff PASS with <2 candidate clusters → no warning (Gate 4 condition (a) absent)."""
    payload = {
        "closed_timestamp": "2026-06-01T00:00:00Z",
        "verdict": "PASS-DEPLOYABLE",
        "clusters": {"c0": _candidate_cluster(), "c1": _non_candidate_cluster()},
        "architectures_tested": ["A1", "A2"],
        "architecture_results": {
            "A1": {"tested": True, "won": False, "worst_fold_ratio": 1.0},
            "A2": {"tested": True, "won": True, "worst_fold_ratio": 2.5},
        },
        "architectures_skipped_by_amendment_5": [],
    }
    with caplog.at_level(logging.WARNING):
        rc = _validate_amendment_5_1_gate_4_qualifier(payload, tmp_path / "ARC_CLOSURE.md")
    assert rc == 0
    assert not any("Amendment 5.1" in r.message for r in caplog.records)
