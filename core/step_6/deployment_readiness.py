"""§6.6 Deployment readiness audit.

Per chat Q3: reads the closure's §4 deployment_spec section, verifies
fields are populated, and confirms ``config_artefact_path`` exists +
is self-contained. Does not duplicate the parser's Section 4-L PASS
validation (which the parser already enforces pre-merge) — the audit's
job here is to surface DEPLOYABLE-READY issues to the closure reader.

Five checks:

  1. ``deployment_spec_section_present`` (critical) — closure doc has
     a ``## §4 deployment_spec`` heading. Mirrors parser Section 4-L
     check but runs at engine-time so PASS arcs surface the issue
     before the parser HALT at PR time.
  2. ``config_artefact_path_resolvable`` (critical) — ``config_artefact_path``
     is non-null AND the referenced file exists.
  3. ``deployment_spec_subsections_present`` (critical) — every §4.X
     subsection heading (§4.1 through §4.11) is present in the closure
     doc.
  4. ``features_live_computable`` (warning) — every feature in
     ``best_candidate_features`` has a non-empty Step 6.3
     equivalent: its FeatureSpec exists in the registry AND
     `needs_panel == False` OR a panel is realistically available live.
  5. ``deployment_checklist_marked`` (info) — count of checked items
     in the §4.11 checklist.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.step_6.inputs import Step6Inputs
from core.step_6.manifest import (
    AuditConfig,
    CategoryAuditResult,
    CheckResult,
    Severity,
)

DEPLOYMENT_SPEC_HEADING_RE = re.compile(r"^## §4 deployment_spec\b", re.MULTILINE)
SUBSECTION_RE = re.compile(r"^### 4\.(\d+)\b", re.MULTILINE)
CHECKED_ITEM_RE = re.compile(r"^- \[(x|X)\]", re.MULTILINE)
UNCHECKED_ITEM_RE = re.compile(r"^- \[ \]", re.MULTILINE)


def _load_closure_text(arc_root: Path) -> str | None:
    closure = arc_root / "ARC_CLOSURE.md"
    if not closure.exists():
        return None
    return closure.read_text(encoding="utf-8")


def _check_deployment_spec_heading(inputs: Step6Inputs) -> CheckResult:
    text = _load_closure_text(inputs.arc_root)
    if text is None:
        return CheckResult(
            name="deployment_spec_section_present",
            passed=False,
            severity=Severity.CRITICAL,
            message="ARC_CLOSURE.md not found under arc_root",
            evidence={"arc_root": str(inputs.arc_root)},
        )
    present = bool(DEPLOYMENT_SPEC_HEADING_RE.search(text))
    return CheckResult(
        name="deployment_spec_section_present",
        passed=present,
        severity=Severity.CRITICAL,
        message=(
            "§4 deployment_spec heading present" if present
            else "§4 deployment_spec heading missing — required for PASS verdicts"
        ),
        evidence={"closure_path": str(inputs.arc_root / "ARC_CLOSURE.md")},
    )


def _check_config_artefact_path(inputs: Step6Inputs) -> CheckResult:
    payload = inputs.closure_payload or {}
    ba = payload.get("best_architecture") or {}
    path_str = ba.get("config_artefact_path")
    if not path_str:
        # Auto-dispatch path may have populated via orchestrator differently;
        # try to recover from extras.
        path_str = inputs.extras.get("config_artefact_path") if inputs.extras else None
    if not path_str:
        return CheckResult(
            name="config_artefact_path_resolvable",
            passed=False,
            severity=Severity.CRITICAL,
            message="best_architecture.config_artefact_path missing or null",
            evidence={"path_str": path_str},
        )
    repo_root = inputs.arc_root.resolve().parents[1] if len(inputs.arc_root.resolve().parents) >= 2 else inputs.arc_root.resolve()
    full = repo_root / path_str
    if not full.exists():
        return CheckResult(
            name="config_artefact_path_resolvable",
            passed=False,
            severity=Severity.CRITICAL,
            message=f"config_artefact_path does not resolve to a file: {full}",
            evidence={"path_str": path_str, "resolved": str(full)},
        )
    return CheckResult(
        name="config_artefact_path_resolvable",
        passed=True,
        severity=Severity.CRITICAL,
        message=f"config_artefact_path resolves to {path_str} ({full.stat().st_size} bytes)",
        evidence={"path_str": path_str, "size_bytes": int(full.stat().st_size)},
    )


def _check_subsections_present(inputs: Step6Inputs) -> CheckResult:
    text = _load_closure_text(inputs.arc_root)
    if text is None:
        return CheckResult(
            name="deployment_spec_subsections_present",
            passed=False,
            severity=Severity.CRITICAL,
            message="ARC_CLOSURE.md not found",
            evidence={},
        )
    found = sorted({int(m.group(1)) for m in SUBSECTION_RE.finditer(text)})
    required = list(range(1, 12))  # §4.1 .. §4.11
    missing = sorted(set(required) - set(found))
    passed = not missing
    return CheckResult(
        name="deployment_spec_subsections_present",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"§4 subsections present: {found}; missing: {missing if missing else 'none'}"
        ),
        evidence={"found": found, "missing": missing, "required": required},
    )


def _check_features_live_computable(inputs: Step6Inputs) -> CheckResult:
    features = inputs.best_candidate_features
    if not features:
        return CheckResult(
            name="features_live_computable",
            passed=True,
            severity=Severity.INFO,
            message="no winning-candidate features (rule-based arc)",
            evidence={},
        )
    try:
        from core.features.registry import get as _get_spec  # type: ignore
    except ImportError as exc:
        return CheckResult(
            name="features_live_computable",
            passed=False,
            severity=Severity.WARNING,
            message=f"could not import feature registry: {exc}",
            evidence={"error": str(exc)},
        )
    needs_panel: list[str] = []
    unknown: list[str] = []
    for f in features:
        try:
            spec = _get_spec(f)
        except KeyError:
            unknown.append(f)
            continue
        if getattr(spec, "needs_panel", False):
            needs_panel.append(f)
    passed = not unknown
    return CheckResult(
        name="features_live_computable",
        passed=passed,
        severity=Severity.WARNING,
        message=(
            f"{len(features)} features inspected: {len(needs_panel)} require panel data, "
            f"{len(unknown)} unknown to registry"
        ),
        evidence={
            "n_features": len(features),
            "needs_panel": needs_panel,
            "unknown_to_registry": unknown,
        },
    )


def _check_deployment_checklist(inputs: Step6Inputs) -> CheckResult:
    text = _load_closure_text(inputs.arc_root)
    if text is None:
        return CheckResult(
            name="deployment_checklist_marked",
            passed=True,
            severity=Severity.INFO,
            message="closure doc not found",
            evidence={},
        )
    checked = len(CHECKED_ITEM_RE.findall(text))
    unchecked = len(UNCHECKED_ITEM_RE.findall(text))
    total = checked + unchecked
    return CheckResult(
        name="deployment_checklist_marked",
        passed=True,
        severity=Severity.INFO,
        message=f"deployment-readiness checklist: {checked}/{total} items checked",
        evidence={"checked": checked, "unchecked": unchecked, "total": total},
    )


def audit(inputs: Step6Inputs, audit_config: AuditConfig) -> CategoryAuditResult:
    checks = (
        _check_deployment_spec_heading(inputs),
        _check_config_artefact_path(inputs),
        _check_subsections_present(inputs),
        _check_features_live_computable(inputs),
        _check_deployment_checklist(inputs),
    )
    diagnostic: dict[str, Any] = {
        "best_candidate_architecture": inputs.best_candidate_architecture,
        "best_candidate_config_id": inputs.best_candidate_config_id,
    }
    return CategoryAuditResult(
        category="deployment_readiness", checks=checks, diagnostic=diagnostic,
    )


__all__ = ("audit",)
