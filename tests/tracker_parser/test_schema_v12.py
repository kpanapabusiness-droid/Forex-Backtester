"""v1.2 schema detection + normalisation."""

from __future__ import annotations

import pytest

from scripts.tracker_parser import schema


def _minimal_v11_payload_passlike() -> dict:
    return {
        "arc_name": "l_arc_y",
        "signal": "sig",
        "tf": "H4",
        "sub_protocol": "vanilla",
        "closed_timestamp": "2026-05-23T00:00:00Z",
        "closure_doc_link": "results/l_arc_y/ARC_CLOSURE.md",
        "verdict": "PASS-VIABLE",
        "one_line": "stub",
        "failed_at_step": "N/A",
        "primary_failure_mode": "N/A",
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
            "worst_fold_ratio": 5.0,
            "worst_fold_roi_base_pct": 20.0,
            "worst_fold_dd_base_pct": 4.0,
            "features_in_winning_config": [],
            "k_safe": 2.0,
        },
        "cost_decomposition": None,
        "clusters": {
            "c0": {
                "n": 50,
                "archetype": "V-shape",
                "sl_atr": 3.5,
                "step3_composite": 1.0,
                "mfe_p50_r": 5.0,
                "ww_pp": 0.5,
                "reach_1r": 0.95,
                "step4_e_auc": None,
                "step4_d1_auc": None,
                "outcome": "wins_step5",
            }
        },
        "architectures_tested": ["A1"],
        "architecture_results": {
            "A1": {"tested": True, "won": True, "worst_fold_ratio": 5.0}
        },
        "archetypes_observed": ["V-shape"],
        "cross_arc_tags": [],
    }


def test_detect_v12_by_template_version_string():
    payload = _minimal_v11_payload_passlike()
    payload["template_version"] = "v1.2"
    assert schema.detect_schema_version(payload) == "1.2"


def test_detect_v12_by_numeric_template_version():
    payload = _minimal_v11_payload_passlike()
    payload["template_version"] = "1.2"
    assert schema.detect_schema_version(payload) == "1.2"


def test_detect_v12_by_exclusive_field_config_path():
    """`config_artefact_path` presence (even null) infers v1.2 when template_version absent."""
    payload = _minimal_v11_payload_passlike()
    payload["best_architecture"]["config_artefact_path"] = "configs/l_arc_y/winning_config.yaml"
    assert schema.detect_schema_version(payload) == "1.2"


def test_detect_v12_by_exclusive_field_section_flag():
    payload = _minimal_v11_payload_passlike()
    payload["best_architecture"]["deployment_spec_section_present"] = True
    assert schema.detect_schema_version(payload) == "1.2"


def test_detect_v11_when_only_amendment3_fields_present():
    """Sanity — v1.1 detection unchanged when v1.2 fields absent."""
    payload = _minimal_v11_payload_passlike()
    assert schema.detect_schema_version(payload) == "1.1"


def test_v12_payload_preserves_deployment_spec_fields():
    payload = _minimal_v11_payload_passlike()
    payload["template_version"] = "v1.2"
    payload["best_architecture"]["config_artefact_path"] = "configs/l_arc_y/winning_config.yaml"
    payload["best_architecture"]["deployment_spec_section_present"] = True
    normalised = schema.parse_payload(payload)
    assert normalised["template_version"] == "1.2"
    assert (
        normalised["best_architecture"]["config_artefact_path"]
        == "configs/l_arc_y/winning_config.yaml"
    )
    assert normalised["best_architecture"]["deployment_spec_section_present"] is True


def test_v10_v11_payloads_still_normalise():
    """Backwards-compat — v1.0 and v1.1 closures parse unchanged."""
    payload = _minimal_v11_payload_passlike()
    payload["template_version"] = "v1.1"
    norm = schema.parse_payload(payload)
    assert norm["template_version"] == "1.1"


def test_unknown_template_version_v2_raises():
    payload = _minimal_v11_payload_passlike()
    payload["template_version"] = "v2.0"
    with pytest.raises(ValueError, match="Unknown template_version"):
        schema.detect_schema_version(payload)


def test_v12_verdict_pass_viable_provisional_accepted():
    """PASS-VIABLE-PROVISIONAL added to the verdict set in v1.2."""
    payload = _minimal_v11_payload_passlike()
    payload["template_version"] = "v1.2"
    payload["verdict"] = "PASS-VIABLE-PROVISIONAL"
    payload["best_architecture"]["config_artefact_path"] = "configs/x.yaml"
    payload["best_architecture"]["deployment_spec_section_present"] = True
    norm = schema.parse_payload(payload)
    assert norm["verdict"] == "PASS-VIABLE-PROVISIONAL"
