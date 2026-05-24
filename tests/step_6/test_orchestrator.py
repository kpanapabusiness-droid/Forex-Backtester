"""End-to-end Step 6 orchestrator tests covering dispatch's Task 10 cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.step_6 import AUDIT_CATEGORY_NAMES, AuditConfig, run_step_6
from core.step_6.artefacts import write_step_6_artefacts
from core.step_6.manifest import (
    TriggerSource,
    VerdictImpact,
)
from tests.step_6._fixtures import build_clean_inputs, build_leaky_inputs


def test_case_1_clean_inputs_auto_dispatch_high_pass_rate(tmp_path: Path):
    """Synthetic clean arc, auto-dispatch — lookahead passes on clean lineage."""
    inputs = build_clean_inputs(tmp_path)
    result = run_step_6(inputs, trigger=TriggerSource.AUTO_PASS)
    by_name = {c.category: c for c in result.categories}
    # Lookahead is the load-bearing category for cleanliness — should pass on clean lineage.
    assert by_name["lookahead"].passed
    # All six categories ran.
    assert len(result.categories) == 6


def test_case_2_leaky_feature_downgrades_verdict(tmp_path: Path):
    """Leaky path-feature in entry decision → Step 6 FAIL → verdict_impact downgrade."""
    inputs = build_leaky_inputs(tmp_path)
    result = run_step_6(inputs, trigger=TriggerSource.AUTO_PASS)
    assert result.overall_passed is False
    assert result.verdict_impact == VerdictImpact.DOWNGRADED_TO_FAIL
    crit = result.critical_failures()
    assert any("no_path_features_in_entry" in n for n in crit)


def test_case_3_manual_invocation_never_downgrades(tmp_path: Path):
    """Manual CLI on a leaky arc — verdict_impact stays NONE (chat Q6)."""
    inputs = build_leaky_inputs(tmp_path)
    result = run_step_6(inputs, trigger=TriggerSource.MANUAL)
    assert result.overall_passed is False  # leak still surfaces
    assert result.verdict_impact == VerdictImpact.NONE  # but no verdict impact


def test_case_4_no_block_flag_suppresses_downgrade(tmp_path: Path):
    inputs = build_leaky_inputs(tmp_path)
    result = run_step_6(
        inputs, trigger=TriggerSource.AUTO_PASS,
        audit_config=AuditConfig(no_block=True),
    )
    assert result.verdict_impact == VerdictImpact.NONE


def test_case_5_two_run_determinism(tmp_path: Path):
    """Same inputs run twice → identical category pass/fail + critical_failures."""
    inputs1 = build_clean_inputs(tmp_path / "run1")
    inputs2 = build_clean_inputs(tmp_path / "run2")
    r1 = run_step_6(inputs1, trigger=TriggerSource.MANUAL)
    r2 = run_step_6(inputs2, trigger=TriggerSource.MANUAL)
    # Compare per-category pass states + critical_failures
    cats1 = {c.category: (c.passed, c.critical_failures()) for c in r1.categories}
    cats2 = {c.category: (c.passed, c.critical_failures()) for c in r2.categories}
    assert cats1 == cats2


def test_case_6_unknown_category_raises(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path)
    with pytest.raises(ValueError, match="Unknown audit categories"):
        run_step_6(
            inputs, audit_config=AuditConfig(categories_to_run=("not_a_category",)),
        )


def test_categories_to_run_restricts_dispatch(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path)
    result = run_step_6(
        inputs, audit_config=AuditConfig(categories_to_run=("lookahead",)),
    )
    assert len(result.categories) == 1
    assert result.categories[0].category == "lookahead"


def test_all_six_categories_dispatch_by_default(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path)
    result = run_step_6(inputs)
    assert tuple(c.category for c in result.categories) == AUDIT_CATEGORY_NAMES


def test_artefacts_written_to_disk(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path)
    result = run_step_6(inputs)
    out_dir = tmp_path / "step_6"
    manifest = write_step_6_artefacts(result, out_dir)
    # Manifest + six category reports + summary + sha256
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "sha256_manifest.json").exists()
    for cat in AUDIT_CATEGORY_NAMES:
        assert (out_dir / f"{cat}_report.md").exists()
    data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert data["arc_name"] == manifest.arc_name


def test_sha256_manifest_excludes_itself(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path)
    result = run_step_6(inputs)
    out_dir = tmp_path / "step_6"
    write_step_6_artefacts(result, out_dir)
    sha_manifest = json.loads((out_dir / "sha256_manifest.json").read_text(encoding="utf-8"))
    # sha256_manifest.json must not list itself
    assert "sha256_manifest.json" not in sha_manifest["files"]
    assert "manifest.json" in sha_manifest["files"]
