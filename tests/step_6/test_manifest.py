"""Step 6 manifest/dataclass smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

from core.step_6.manifest import (
    AUDIT_CATEGORY_NAMES,
    AuditConfig,
    CategoryAuditResult,
    CheckResult,
    Severity,
    Step6Manifest,
    Step6Result,
    TriggerSource,
    VerdictImpact,
    write_step_6_manifest,
)


def _make_check(name: str, passed: bool, severity: Severity = Severity.CRITICAL) -> CheckResult:
    return CheckResult(name=name, passed=passed, severity=severity, message=name)


def test_category_passes_only_with_all_critical_clean():
    checks = (
        _make_check("c1", True),
        _make_check("c2", True),
        _make_check("c3", False, Severity.WARNING),  # warning does NOT block
        _make_check("c4", False, Severity.INFO),     # info does NOT block
    )
    cat = CategoryAuditResult(category="lookahead", checks=checks)
    assert cat.passed is True
    assert cat.n_critical == 2
    assert cat.n_critical_fails == 0
    assert cat.n_warnings == 1
    assert cat.n_info == 1


def test_category_fails_when_critical_fails():
    checks = (
        _make_check("c1", True),
        _make_check("c2", False, Severity.CRITICAL),
    )
    cat = CategoryAuditResult(category="lookahead", checks=checks)
    assert cat.passed is False
    assert cat.critical_failures() == ("c2",)


def test_step_6_result_overall_passed_aggregates_categories():
    cat_a = CategoryAuditResult(category="lookahead", checks=(_make_check("c1", True),))
    cat_b = CategoryAuditResult(category="determinism", checks=(_make_check("c2", False),))
    result = Step6Result(
        arc_name="t", trigger=TriggerSource.AUTO_PASS,
        categories=(cat_a, cat_b), audit_config=AuditConfig(),
    )
    assert result.overall_passed is False
    assert result.critical_failures() == ("determinism.c2",)


def test_manual_invocation_never_downgrades():
    cat = CategoryAuditResult(
        category="lookahead", checks=(_make_check("c1", False),),
    )
    result = Step6Result(
        arc_name="t", trigger=TriggerSource.MANUAL,
        categories=(cat,), audit_config=AuditConfig(),
    )
    assert result.overall_passed is False
    assert result.verdict_impact == VerdictImpact.NONE  # manual NEVER downgrades


def test_no_block_suppresses_verdict_downgrade():
    cat = CategoryAuditResult(
        category="lookahead", checks=(_make_check("c1", False),),
    )
    result = Step6Result(
        arc_name="t", trigger=TriggerSource.AUTO_PASS,
        categories=(cat,), audit_config=AuditConfig(no_block=True),
    )
    assert result.verdict_impact == VerdictImpact.NONE


def test_auto_dispatch_downgrades_on_critical_fail():
    cat = CategoryAuditResult(
        category="lookahead", checks=(_make_check("c1", False),),
    )
    result = Step6Result(
        arc_name="t", trigger=TriggerSource.AUTO_PASS,
        categories=(cat,), audit_config=AuditConfig(),
    )
    assert result.verdict_impact == VerdictImpact.DOWNGRADED_TO_FAIL


def test_manifest_serde_roundtrip(tmp_path: Path):
    cat = CategoryAuditResult(
        category="lookahead", checks=(_make_check("c1", True),),
    )
    result = Step6Result(
        arc_name="t", trigger=TriggerSource.AUTO_PASS,
        categories=(cat,), audit_config=AuditConfig(),
    )
    manifest = Step6Manifest.from_result(
        result, report_paths={"lookahead": "lookahead_report.md"},
        summary_path="summary.md", ran_at="2026-05-24T12:00:00Z",
    )
    out = tmp_path / "manifest.json"
    write_step_6_manifest(manifest, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["arc_name"] == "t"
    assert data["trigger"] == "auto_pass"
    assert data["overall_passed"] is True
    assert "lookahead" in data["categories"]


def test_audit_category_names_locked():
    assert AUDIT_CATEGORY_NAMES == (
        "lookahead", "selection_bias", "execution_realism",
        "statistical", "determinism", "deployment_readiness",
    )
