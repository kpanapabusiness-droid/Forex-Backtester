"""v1.2 PASS-verdict validation — template Section 4-L.

Exercises `_validate_v12_pass_verdict` in `scripts/update_tracker_from_closure.py` directly,
plus the full CLI on synthetic closures, to confirm HALT behaviour on each failure mode.
"""

from __future__ import annotations

from pathlib import Path

from scripts.update_tracker_from_closure import _validate_v12_pass_verdict, main as cli_main


def _valid_closure_text(
    *,
    arc_name: str = "l_arc_z",
    verdict: str = "PASS-VIABLE",
    template_version: str | None = "v1.2",
    config_path: str | None = "configs/l_arc_z/winning_config.yaml",
    section_present: bool | None = True,
    include_section_4: bool = True,
) -> str:
    tv_line = f"  template_version: {template_version}\n" if template_version else ""
    config_line = (
        f"    config_artefact_path: {config_path}\n"
        if config_path is not None
        else "    config_artefact_path: null\n"
    )
    flag_line = (
        f"    deployment_spec_section_present: {'true' if section_present else 'false'}\n"
        if section_present is not None
        else "    deployment_spec_section_present: null\n"
    )
    body = f"""# ARC_Z_CLOSURE — {arc_name}

---

## §1 tracker_payload

```yaml
tracker_payload:
{tv_line}  arc_name: {arc_name}
  signal: stub signal
  tf: H4
  sub_protocol: vanilla
  closed_timestamp: 2026-05-23T00:00:00Z
  closure_doc_link: results/{arc_name}/ARC_CLOSURE.md
  verdict: {verdict}
  one_line: stub
  failed_at_step: N/A
  primary_failure_mode: N/A
  pool_metadata:
    total_n: 100
    window_start: 2010-01-01
    window_end: 2026-04-30
    kh24_co_fire_pct: null
    configs_evaluated_step5: 1
    search_scope_flag: thin
  best_architecture:
    name: A1 system_level_filter
    worst_fold_ratio: 5.0
    worst_fold_roi_base_pct: 20.0
    worst_fold_dd_base_pct: 4.0
    features_in_winning_config: []
{config_line}{flag_line}  cost_decomposition: null
  clusters:
    c0:
      n: 50
      archetype: V-shape
      sl_atr: 3.5
      step3_composite: 1.0
      mfe_p50_r: 5.0
      ww_pp: 0.5
      reach_1r: 0.95
      step4_e_auc: null
      step4_d1_auc: null
      outcome: wins_step5
  architectures_tested: [A1]
  architecture_results:
    A1: {{tested: true, won: true, worst_fold_ratio: 5.0}}
  archetypes_observed: [V-shape]
  cross_arc_tags: []
```

---

## §2 Why succeeded

Stub.

---

## §3 Cross-arc observations

- stub
"""
    if include_section_4:
        body += """
---

## §4 deployment_spec

Stub.
"""
    return body


def test_validation_pass(tmp_path: Path) -> None:
    """All four checks pass — returns 0."""
    closure = tmp_path / "ARC_CLOSURE.md"
    closure.write_text(_valid_closure_text(), encoding="utf-8")
    config = tmp_path / "configs" / "l_arc_z" / "winning_config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("# stub", encoding="utf-8")

    payload = {
        "verdict": "PASS-VIABLE",
        "best_architecture": {
            "config_artefact_path": "configs/l_arc_z/winning_config.yaml",
            "deployment_spec_section_present": True,
        },
    }

    import scripts.update_tracker_from_closure as cli
    original_root = cli._REPO_ROOT
    cli._REPO_ROOT = tmp_path
    try:
        rc = _validate_v12_pass_verdict(payload, closure)
    finally:
        cli._REPO_ROOT = original_root
    assert rc == 0


def test_validation_halt_on_null_config_path(tmp_path: Path) -> None:
    """Item 1 — config_artefact_path null → HALT."""
    closure = tmp_path / "ARC_CLOSURE.md"
    closure.write_text(_valid_closure_text(config_path=None), encoding="utf-8")
    payload = {
        "verdict": "PASS-VIABLE",
        "best_architecture": {
            "config_artefact_path": None,
            "deployment_spec_section_present": True,
        },
    }
    rc = _validate_v12_pass_verdict(payload, closure)
    assert rc == 1


def test_validation_halt_on_missing_config_file(tmp_path: Path) -> None:
    """Item 2 — file at config_artefact_path does not exist → HALT."""
    closure = tmp_path / "ARC_CLOSURE.md"
    closure.write_text(_valid_closure_text(), encoding="utf-8")
    payload = {
        "verdict": "PASS-VIABLE",
        "best_architecture": {
            "config_artefact_path": "configs/does_not_exist/x.yaml",
            "deployment_spec_section_present": True,
        },
    }
    import scripts.update_tracker_from_closure as cli
    original_root = cli._REPO_ROOT
    cli._REPO_ROOT = tmp_path
    try:
        rc = _validate_v12_pass_verdict(payload, closure)
    finally:
        cli._REPO_ROOT = original_root
    assert rc == 1


def test_validation_halt_on_missing_section_4(tmp_path: Path) -> None:
    """Item 3 — closure lacks `## §4 deployment_spec` heading → HALT."""
    closure = tmp_path / "ARC_CLOSURE.md"
    closure.write_text(_valid_closure_text(include_section_4=False), encoding="utf-8")
    config = tmp_path / "configs" / "l_arc_z" / "winning_config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("# stub", encoding="utf-8")
    payload = {
        "verdict": "PASS-VIABLE",
        "best_architecture": {
            "config_artefact_path": "configs/l_arc_z/winning_config.yaml",
            "deployment_spec_section_present": True,
        },
    }
    import scripts.update_tracker_from_closure as cli
    original_root = cli._REPO_ROOT
    cli._REPO_ROOT = tmp_path
    try:
        rc = _validate_v12_pass_verdict(payload, closure)
    finally:
        cli._REPO_ROOT = original_root
    assert rc == 1


def test_validation_halt_on_flag_false(tmp_path: Path) -> None:
    """Item 4 — deployment_spec_section_present false → HALT."""
    closure = tmp_path / "ARC_CLOSURE.md"
    closure.write_text(_valid_closure_text(section_present=False), encoding="utf-8")
    config = tmp_path / "configs" / "l_arc_z" / "winning_config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("# stub", encoding="utf-8")
    payload = {
        "verdict": "PASS-VIABLE",
        "best_architecture": {
            "config_artefact_path": "configs/l_arc_z/winning_config.yaml",
            "deployment_spec_section_present": False,
        },
    }
    import scripts.update_tracker_from_closure as cli
    original_root = cli._REPO_ROOT
    cli._REPO_ROOT = tmp_path
    try:
        rc = _validate_v12_pass_verdict(payload, closure)
    finally:
        cli._REPO_ROOT = original_root
    assert rc == 1


def test_v12_fail_verdict_skips_validation(tmp_path: Path) -> None:
    """FAIL verdicts in v1.2 skip the PASS-verdict block — validation never runs."""
    closure = tmp_path / "ARC_CLOSURE.md"
    closure.write_text(
        _valid_closure_text(
            verdict="FAIL", config_path=None, section_present=None, include_section_4=False
        ),
        encoding="utf-8",
    )
    # The validation helper itself is only called for PASS-* verdicts in the CLI;
    # this test confirms the closure parses without invoking validation. We invoke the
    # full CLI in dry-run mode against a synthetic tracker to exercise the gate.

    tracker = tmp_path / "ARC_TRACKER.md"
    # Minimal tracker the parser can read without touching the v1.2 path.
    tracker.write_text(
        "# ARC_TRACKER\n\nLast auto-update: never\n",
        encoding="utf-8",
    )
    # CLI returns 1 here because the synthetic tracker doesn't satisfy the mapping layer's
    # row-existence checks, but the v1.2 validation block must NOT be reached for a FAIL
    # verdict — proven by absence of the v1.2 error string. We assert exit-non-zero is
    # caused by tracker-shape, not v1.2 validation, by checking for the v1.2 error pattern.
    import logging
    records: list[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(self.format(record))

    handler = _CaptureHandler()
    logging.getLogger().addHandler(handler)
    try:
        cli_main(
            [
                str(closure),
                "--dry-run",
                "--tracker-path",
                str(tracker),
                "--rolling-state",
                str(tmp_path / "rolling_state.json"),
                "--registry",
                str(tmp_path / "parsed.log"),
            ]
        )
    finally:
        logging.getLogger().removeHandler(handler)

    # The v1.2 validation strings (which would be emitted on PASS-verdict failure) must not appear.
    joined = "\n".join(records)
    assert "v1.2 PASS-verdict validation" not in joined
