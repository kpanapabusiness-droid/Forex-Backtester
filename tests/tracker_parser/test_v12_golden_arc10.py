"""Golden test for Arc 10 retrofit — parser is idempotent across v1.1→v1.2 retrofit.

The §1 metric data is unchanged by the retrofit (only template_version, config_artefact_path,
deployment_spec_section_present are added). All tracker A-J mapping inputs are unchanged, so
parsing the retrofitted closure must produce the same v1.2-normalised payload up to those three
new fields.

This test runs against the live closure post-retrofit (Task 4). It complements the existing
`test_golden_arc10.py` which exercises the v1.0/v1.1 paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.tracker_parser import extract, schema


@pytest.fixture
def closure_arc_10(request: pytest.FixtureRequest) -> Path:
    return request.config.rootpath / "results" / "l_arc_10" / "ARC_CLOSURE.md"


def test_v12_payload_parses(closure_arc_10: Path) -> None:
    """Arc 10 retrofit closure parses as v1.2 with deployment-spec fields populated."""
    payload = extract.extract_payload(closure_arc_10)
    normalised = schema.parse_payload(payload)
    assert normalised["template_version"] == "1.2"
    ba = normalised["best_architecture"]
    assert ba["config_artefact_path"] == "configs/l_arc_10/winning_config.yaml"
    assert ba["deployment_spec_section_present"] is True


def test_v12_deployment_spec_heading_present(closure_arc_10: Path) -> None:
    """`## §4 deployment_spec` heading present on Arc 10's retrofit."""
    assert extract.has_deployment_spec_heading(closure_arc_10) is True


def test_v12_metric_fields_unchanged_from_v11(closure_arc_10: Path) -> None:
    """The §1 metric data (verdict, ratios, ROI/DD, cluster outcomes) is identical to the
    pre-retrofit closure. Asserts the retrofit was purely additive — no metric mutations.

    Anchored to the documented Arc 10 winning numbers from the closure §1 (PASS-VIABLE,
    worst_fold_ratio 5.4185, A1 system_level_filter winner).
    """
    payload = extract.extract_payload(closure_arc_10)
    normalised = schema.parse_payload(payload)
    assert normalised["verdict"] == "PASS-VIABLE"
    ba = normalised["best_architecture"]
    assert ba["name"] == "A1 system_level_filter"
    assert abs(ba["worst_fold_ratio"] - 5.4185) < 1e-4
    assert abs(ba["worst_fold_roi_base_pct"] - 26.49) < 1e-3
    assert abs(ba["worst_fold_dd_base_pct"] - 9.22) < 1e-3
    assert ba["sign_pos_folds"] == "11/11"


def test_v12_pass_verdict_validation_passes_against_live_repo(
    closure_arc_10: Path, request: pytest.FixtureRequest
) -> None:
    """Full CLI validation gate runs clean — config file exists, §4 present, flag true."""
    from scripts.update_tracker_from_closure import _validate_v12_pass_verdict
    import scripts.update_tracker_from_closure as cli

    payload = extract.extract_payload(closure_arc_10)
    normalised = schema.parse_payload(payload)
    repo_root = Path(request.config.rootpath)
    original_root = cli._REPO_ROOT
    cli._REPO_ROOT = repo_root
    try:
        rc = _validate_v12_pass_verdict(normalised, closure_arc_10)
    finally:
        cli._REPO_ROOT = original_root
    assert rc == 0
