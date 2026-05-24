"""Parser v1.3 schema detection + Step 6 block validation + Phase 2 tightening."""

from __future__ import annotations

import pytest

from scripts.tracker_parser.schema import (
    PHASE_2_CUTOFF_ISO,
    Step6Block,
    detect_schema_version,
    parse_payload,
)


def test_phase_2_cutoff_locked():
    assert PHASE_2_CUTOFF_ISO == "2026-05-23T06:20:59Z"


def test_detect_v13_via_template_version():
    assert detect_schema_version({"template_version": "v1.3"}) == "1.3"
    assert detect_schema_version({"template_version": "1.3"}) == "1.3"


def test_detect_v13_via_top_level_step_6():
    assert detect_schema_version({"step_6": {"ran": True, "trigger": "auto_pass"}}) == "1.3"


def test_detect_v12_unaffected_by_v13_detection():
    assert detect_schema_version({"template_version": "v1.2"}) == "1.2"
    assert (
        detect_schema_version({"best_architecture": {"config_artefact_path": "x"}})
        == "1.2"
    )


def test_step_6_block_required_fields():
    b = Step6Block(ran=True, trigger="auto_pass", overall_passed=True)
    assert b.ran is True
    assert b.trigger == "auto_pass"
    assert b.overall_passed is True
    assert b.warnings_count == 0


def test_invalid_trigger_raises():
    payload = {
        "template_version": "v1.3",
        "arc_name": "x", "signal": "s", "tf": "H4", "sub_protocol": "vanilla",
        "closed_timestamp": "2026-05-24T00:00:00Z",
        "closure_doc_link": "p",
        "verdict": "FAIL", "one_line": "x", "failed_at_step": "5",
        "primary_failure_mode": "step6_causal_audit_fail",
        "pool_metadata": {
            "total_n": 0, "window_start": "2010", "window_end": "2020",
            "configs_evaluated_step5": 0, "search_scope_flag": "thin",
        },
        "clusters": {}, "architectures_tested": [],
        "architecture_results": {}, "archetypes_observed": [],
        "step_6": {
            "ran": True, "trigger": "INVALID_TRIGGER", "overall_passed": True,
        },
    }
    with pytest.raises(ValueError, match="step_6.trigger"):
        parse_payload(payload)


def test_invalid_verdict_impact_raises():
    payload = {
        "template_version": "v1.3",
        "arc_name": "x", "signal": "s", "tf": "H4", "sub_protocol": "vanilla",
        "closed_timestamp": "2026-05-24T00:00:00Z",
        "closure_doc_link": "p",
        "verdict": "FAIL", "one_line": "x", "failed_at_step": "5",
        "primary_failure_mode": "step6_causal_audit_fail",
        "pool_metadata": {
            "total_n": 0, "window_start": "2010", "window_end": "2020",
            "configs_evaluated_step5": 0, "search_scope_flag": "thin",
        },
        "clusters": {}, "architectures_tested": [],
        "architecture_results": {}, "archetypes_observed": [],
        "step_6": {
            "ran": True, "trigger": "auto_pass",
            "verdict_impact": "INVALID_IMPACT",
        },
    }
    with pytest.raises(ValueError, match="verdict_impact"):
        parse_payload(payload)


def test_v13_payload_round_trip_with_step_6():
    payload = {
        "template_version": "v1.3",
        "arc_name": "test_arc", "signal": "x", "tf": "H4", "sub_protocol": "vanilla",
        "closed_timestamp": "2026-05-24T12:00:00Z",
        "closure_doc_link": "results/test_arc/ARC_CLOSURE.md",
        "verdict": "PASS-DEPLOYABLE", "one_line": "x", "failed_at_step": "N/A",
        "primary_failure_mode": "N/A",
        "pool_metadata": {
            "total_n": 500, "window_start": "2010", "window_end": "2020",
            "configs_evaluated_step5": 80, "search_scope_flag": "normal",
        },
        "clusters": {}, "architectures_tested": ["A1"],
        "architecture_results": {"A1": {"tested": True, "won": True, "worst_fold_ratio": 2.5}},
        "archetypes_observed": [],
        "step_6": {
            "ran": True, "trigger": "auto_pass", "overall_passed": True,
            "manifest_path": "results/test_arc/step_6/manifest.json",
            "categories": {
                "lookahead": True, "selection_bias": True,
                "execution_realism": True, "statistical": True,
                "determinism": True, "deployment_readiness": True,
            },
            "critical_failures": [], "warnings_count": 0,
            "verdict_impact": "none",
        },
    }
    norm = parse_payload(payload)
    assert norm["template_version"] == "1.3"
    assert norm["step_6"]["ran"] is True
    assert norm["step_6"]["overall_passed"] is True
