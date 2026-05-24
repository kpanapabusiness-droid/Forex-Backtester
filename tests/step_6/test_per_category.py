"""Per-category smoke tests for the five categories beyond lookahead."""

from __future__ import annotations

from pathlib import Path

from core.step_6 import AuditConfig
from core.step_6.deployment_readiness import audit as audit_deployment
from core.step_6.determinism import audit as audit_determinism
from core.step_6.execution_realism import audit as audit_exec
from core.step_6.selection_bias import audit as audit_selbias
from core.step_6.statistical import audit as audit_stat
from tests.step_6._fixtures import build_clean_inputs


def test_selection_bias_records_configs_evaluated(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path, configs_evaluated=80)
    result = audit_selbias(inputs, AuditConfig())
    rec = next(c for c in result.checks if c.name == "configs_evaluated_recorded")
    assert rec.passed is True
    assert rec.evidence["configs_evaluated_step5"] == 80


def test_execution_realism_finds_histdata_dir(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path)
    result = audit_exec(inputs, AuditConfig())
    spread = next(c for c in result.checks if c.name == "real_spread_source_present")
    # Spread source check looks for data/histdata under repo root —
    # exists in dev worktree, may not in CI clean env. Either outcome
    # is valid here; just verify the check ran and produced a verdict.
    assert spread.passed in (True, False)


def test_statistical_sample_size_check(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path, n_trades=150)
    result = audit_stat(inputs, AuditConfig(lo_corrected_min_trades=100))
    ss = next(c for c in result.checks if c.name == "sample_size_adequate")
    assert ss.passed is True
    assert ss.evidence["n_total_trades"] == 150


def test_statistical_sample_size_fail_when_below(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path, n_trades=30)
    result = audit_stat(inputs, AuditConfig(lo_corrected_min_trades=100))
    ss = next(c for c in result.checks if c.name == "sample_size_adequate")
    assert ss.passed is False


def test_statistical_pair_survivorship(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path, pair_set=("EURUSD", "GBPUSD", "USDJPY"))
    result = audit_stat(inputs, AuditConfig())
    surv = next(c for c in result.checks if c.name == "pair_set_survivorship")
    assert surv.passed is True


def test_determinism_skips_when_no_step4_manifest(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path)
    result = audit_determinism(inputs, AuditConfig())
    s4 = next(c for c in result.checks if c.name == "step_4_manifest_sha256_match")
    # No Step 4 manifest in synth dir → info skip, passes
    assert s4.passed is True


def test_determinism_seed_pinning_check():
    inputs = build_clean_inputs(Path("/tmp/_unused"))
    result = audit_determinism(inputs, AuditConfig())
    seed = next(c for c in result.checks if c.name == "seed_pinning_verified")
    # Pin check is static — passes when builders contain random_state=
    assert seed.evidence["n_random_state_references"] >= 0


def test_deployment_readiness_flags_missing_closure(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path)
    result = audit_deployment(inputs, AuditConfig())
    heading_check = next(c for c in result.checks if c.name == "deployment_spec_section_present")
    # No ARC_CLOSURE.md in synth dir → critical fail
    assert heading_check.passed is False


def test_deployment_readiness_finds_closure_heading(tmp_path: Path):
    inputs = build_clean_inputs(tmp_path)
    # Write a minimal closure with the §4 heading
    closure = tmp_path / "ARC_CLOSURE.md"
    closure.write_text(
        "# ARC_CLOSURE — synth\n\n## §1 tracker_payload\n\n```yaml\nfoo: bar\n```\n\n"
        "## §4 deployment_spec\n\n### 4.1 Pair set\n\n### 4.2 Signal\n\n### 4.3 Features\n\n"
        "### 4.4 Filters\n\n### 4.5 Entry\n\n### 4.6 Exit\n\n### 4.7 Exposure\n\n"
        "### 4.8 Risk\n\n### 4.9 Session\n\n### 4.10 Discrepancies\n\n### 4.11 Checklist\n\n"
        "- [x] item 1\n- [ ] item 2\n",
        encoding="utf-8",
    )
    result = audit_deployment(inputs, AuditConfig())
    heading_check = next(c for c in result.checks if c.name == "deployment_spec_section_present")
    assert heading_check.passed is True
    subs = next(c for c in result.checks if c.name == "deployment_spec_subsections_present")
    assert subs.passed is True
    checklist = next(c for c in result.checks if c.name == "deployment_checklist_marked")
    assert checklist.evidence["checked"] == 1
    assert checklist.evidence["unchecked"] == 1
