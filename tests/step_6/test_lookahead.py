"""§6.1 lookahead audit tests."""

from __future__ import annotations

from pathlib import Path

from core.step_6.lookahead import audit
from core.step_6.manifest import AuditConfig, Severity
from tests.step_6._fixtures import build_clean_inputs, build_leaky_inputs


def test_clean_inputs_pass_lineage_check(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path)
    result = audit(inputs, AuditConfig())
    lineage_check = next(c for c in result.checks if c.name == "per_feature_lineage_clean")
    assert lineage_check.passed is True
    assert lineage_check.severity == Severity.CRITICAL


def test_leaky_feature_fails_no_path_features_check(tmp_path: Path):
    inputs = build_leaky_inputs(tmp_path)
    result = audit(inputs, AuditConfig())
    leak_check = next(c for c in result.checks if c.name == "no_path_features_in_entry")
    assert leak_check.passed is False
    assert "mfe_p50_r" in leak_check.evidence["leaking_features"]


def test_d1_lag_rule_helper_present_in_multi_tf():
    # The check is static — inspects core.features.multi_tf source. Should always pass.
    inputs = build_clean_inputs(Path("/tmp/_unused"))  # arc_root not used by this check
    result = audit(inputs, AuditConfig())
    d1_check = next(c for c in result.checks if c.name == "d1_lag_rule_enforced")
    assert d1_check.passed is True


def test_byte_compare_skipped_without_panels(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path)
    # No panels provided → byte-compare records info-level skip
    result = audit(inputs, AuditConfig())
    bc_check = next(c for c in result.checks if c.name == "byte_compare_no_drift")
    assert bc_check.severity == Severity.INFO  # demoted when can't run
    assert "panels" in bc_check.message.lower() or "skip" in bc_check.message.lower()


def test_lineage_table_info_check(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path)
    result = audit(inputs, AuditConfig())
    lt = next(c for c in result.checks if c.name == "feature_lineage_table_exists")
    assert lt.passed is True
    assert lt.severity == Severity.INFO


def test_category_is_lookahead(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path)
    result = audit(inputs, AuditConfig())
    assert result.category == "lookahead"
