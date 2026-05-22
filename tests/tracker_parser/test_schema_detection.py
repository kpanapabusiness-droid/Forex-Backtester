"""Schema version detection + v1.0 → v1.1 normalisation."""

from __future__ import annotations

import pytest

from scripts.tracker_parser import schema


def _minimal_v10_payload() -> dict:
    return {
        "arc_name": "l_arc_x",
        "signal": "sig",
        "tf": "H4",
        "sub_protocol": "vanilla",
        "closed_timestamp": "2026-05-22T00:00:00Z",
        "closure_doc_link": "results/l_arc_x/ARC_CLOSURE.md",
        "verdict": "FAIL",
        "one_line": "stub",
        "failed_at_step": 5,
        "primary_failure_mode": "step5_dd_above_gate",
        "pool_metadata": {
            "total_n": 100,
            "window_start": "2010-01-01",
            "window_end": "2026-04-30",
            "kh24_co_fire_pct": None,
            "configs_evaluated_step5": 1,
            "search_scope_flag": "thin",
        },
        "best_architecture": {
            "name": "A1 system_level_filter",
            "worst_fold_ratio": 1.234,
            "worst_fold_roi_pct": 5.0,
            "worst_fold_dd_pct": 4.0,
            "features_in_winning_config": ["f1"],
        },
        "cost_decomposition": None,
        "clusters": {
            "c0": {
                "n": 50,
                "archetype": "Bimodal",
                "sl_atr": 1.5,
                "step3_composite": 1.0,
                "mfe_p50_r": 2.0,
                "ww_pp": 0.1,
                "reach_1r": 0.9,
                "step4_e_auc": None,
                "step4_d1_auc": None,
                "outcome": "dies_step3",
            }
        },
        "architectures_tested": ["A1"],
        "architecture_results": {
            "A1": {"tested": True, "won": False, "worst_fold_ratio": 1.234}
        },
        "archetypes_observed": ["Bimodal"],
        "cross_arc_tags": ["t1"],
    }


def test_detect_v10_by_absence_of_marker():
    payload = _minimal_v10_payload()
    assert schema.detect_schema_version(payload) == "1.0"


def test_detect_v11_by_template_version_string():
    payload = _minimal_v10_payload()
    payload["template_version"] = "v1.1"
    assert schema.detect_schema_version(payload) == "1.1"


def test_detect_v10_by_template_version_string():
    payload = _minimal_v10_payload()
    payload["template_version"] = "v1.0"
    assert schema.detect_schema_version(payload) == "1.0"


def test_detect_v11_by_numeric_form():
    payload = _minimal_v10_payload()
    payload["template_version"] = "1.1"
    assert schema.detect_schema_version(payload) == "1.1"


def test_detect_v11_by_exclusive_field():
    payload = _minimal_v10_payload()
    payload["best_architecture"]["k_safe"] = 0.5
    assert schema.detect_schema_version(payload) == "1.1"


def test_detect_unknown_version_raises():
    payload = _minimal_v10_payload()
    payload["template_version"] = "v2.0"
    with pytest.raises(ValueError, match="Unknown template_version"):
        schema.detect_schema_version(payload)


def test_v10_normalises_field_renames():
    payload = _minimal_v10_payload()
    normalised = schema.parse_payload(payload)
    ba = normalised["best_architecture"]
    assert "worst_fold_roi_pct" not in ba
    assert "worst_fold_dd_pct" not in ba
    assert ba["worst_fold_roi_base_pct"] == 5.0
    assert ba["worst_fold_dd_base_pct"] == 4.0
    # v1.1-exclusive fields default to None
    assert ba["k_safe"] is None
    assert ba["chained_max_dd_base_pct"] is None
    assert normalised["template_version"] == "1.0"


def test_v11_payload_preserves_amendment_fields():
    payload = _minimal_v10_payload()
    payload["template_version"] = "v1.1"
    payload["best_architecture"]["worst_fold_roi_base_pct"] = payload[
        "best_architecture"
    ].pop("worst_fold_roi_pct")
    payload["best_architecture"]["worst_fold_dd_base_pct"] = payload[
        "best_architecture"
    ].pop("worst_fold_dd_pct")
    payload["best_architecture"]["k_safe"] = 0.5
    payload["best_architecture"]["scalable_to_safe"] = True
    normalised = schema.parse_payload(payload)
    assert normalised["best_architecture"]["k_safe"] == 0.5
    assert normalised["best_architecture"]["scalable_to_safe"] is True


def test_unknown_failure_mode_raises():
    payload = _minimal_v10_payload()
    payload["primary_failure_mode"] = "made_up_mode"
    with pytest.raises(ValueError, match="primary_failure_mode"):
        schema.parse_payload(payload)


def test_unknown_verdict_raises():
    payload = _minimal_v10_payload()
    payload["verdict"] = "MAYBE"
    with pytest.raises(ValueError, match="verdict"):
        schema.parse_payload(payload)


def test_unknown_architecture_raises():
    payload = _minimal_v10_payload()
    payload["architectures_tested"] = ["A7"]
    with pytest.raises(ValueError, match="architecture"):
        schema.parse_payload(payload)
