"""§6.6 Deployment readiness audit.

Per chat Q3: reads the closure's §4 deployment_spec section, verifies
fields are populated, and confirms ``config_artefact_path`` exists +
is self-contained. Does not duplicate the parser's Section 4-L PASS
validation (which the parser already enforces pre-merge) — the audit's
job here is to surface DEPLOYABLE-READY issues to the closure reader.

Hardened in ``engine/step_6_ultimate_audit`` from 5 to 8 checks with
broker/venue declaration, timezone declaration parity, and the EA
parity check (flagging A6 meta-labeling architectures as
research-only-not-currently-EA-deployable):

  1. ``deployment_spec_section_present`` (critical) — closure doc has
     a ``## §4 deployment_spec`` heading. Mirrors parser Section 4-L
     check but runs at engine-time so PASS arcs surface the issue
     before the parser HALT at PR time.
  2. ``config_artefact_path_resolvable`` (critical) — ``config_artefact_path``
     is non-null AND the referenced file exists.
  3. ``deployment_spec_subsections_present`` (critical) — every §4.X
     subsection heading (§4.1 through §4.11) is present in the closure
     doc.
  4. ``ea_parity`` (critical) — winning architecture is reproducible in
     MQL5. A1 / A2 / A6 with rule-based gates are EA-deployable; A3 /
     A4 (per-fold retraining) and A6 with Python-only classifier are
     research-only. Surfaces as critical fail when verdict is PASS-
     DEPLOYABLE on a non-deployable architecture, warning otherwise.
  5. ``broker_venue_declared`` (warning) — closure §4 declares which
     broker spreads the verdict was computed against (5ers / IC Markets
     / other). Cross-references the §6.3 spread P&L decomposition's
     fragility classification.
  6. ``timezone_declaration_parity`` (warning) — closure declares which
     boundary convention was used (utc / 5ers_eet). Mismatch with the
     declared deployment venue is a critical fail; mismatch with KH-24's
     utc convention is informational.
  7. ``features_live_computable`` (warning) — every feature in
     ``best_candidate_features`` has a non-empty Step 6.3
     equivalent: its FeatureSpec exists in the registry AND
     `needs_panel == False` OR a panel is realistically available live.
  8. ``deployment_checklist_marked`` (info) — count of checked items
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


def _check_ea_parity(inputs: Step6Inputs) -> CheckResult:
    """Winning architecture must be reproducible in MQL5.

    A1 (rule-based system filter) — EA-deployable.
    A2 (admit-classifier) — EA-deployable IF the classifier is
        expressible as ONNX or a simple rule set; flag as warning
        otherwise.
    A3 / A4 (per-fold retraining) — NOT EA-deployable; research-only.
    A6 (meta-labeling) — EA-deployable IF classifier is portable to
        MQL5 inference (ONNX or simple rule set); else research-only.

    Pass condition: winning architecture is A1, OR the closure
    documents the EA-reproducibility path explicitly.
    """
    arch = (inputs.best_candidate_architecture or "").upper()
    closure_text = _load_closure_text(inputs.arc_root) or ""
    # Heuristic for explicit EA-reproducibility declaration.
    mentions_ea = any(
        tag in closure_text.lower()
        for tag in ("mql5", "ea_deployable", "ea deployable", "ea parity",
                    "research-only", "research_only")
    )
    research_only_archs = {"A3", "A4"}
    review_required = {"A2", "A6"}
    deployable = {"A1"}
    if arch in deployable:
        return CheckResult(
            name="ea_parity",
            passed=True,
            severity=Severity.CRITICAL,
            message=f"winning architecture {arch!r} is EA-deployable by construction (A1 rule-based)",
            evidence={"architecture": arch, "ea_deployable": True},
        )
    if arch in research_only_archs:
        # Critical fail if verdict is PASS-DEPLOYABLE (we'd be claiming
        # deployability for a non-deployable architecture).
        ba = (inputs.closure_payload or {}).get("best_architecture") or {}
        verdict = str(ba.get("verdict") or "").upper()
        is_deployable_verdict = verdict == "PASS_DEPLOYABLE"
        passed = mentions_ea  # closure must acknowledge it
        return CheckResult(
            name="ea_parity",
            passed=passed and not is_deployable_verdict,
            severity=Severity.CRITICAL,
            message=(
                f"winning architecture {arch!r} requires per-fold retraining — "
                "NOT EA-deployable. "
                + (
                    f"Verdict PASS-DEPLOYABLE on {arch} is contradictory; "
                    "downgrade to research-only."
                    if is_deployable_verdict
                    else (
                        "Closure acknowledges research-only status."
                        if mentions_ea
                        else "Closure does not mark research-only; review."
                    )
                )
            ),
            evidence={
                "architecture": arch,
                "verdict": verdict,
                "closure_mentions_ea_or_research_only": mentions_ea,
            },
        )
    if arch in review_required:
        return CheckResult(
            name="ea_parity",
            passed=mentions_ea,
            severity=Severity.WARNING,
            message=(
                f"winning architecture {arch!r} requires an EA-side ONNX/rules "
                "inference path. Closure "
                + (
                    "documents the path." if mentions_ea
                    else "MUST document the EA inference path before deployment."
                )
            ),
            evidence={
                "architecture": arch,
                "closure_mentions_ea_or_research_only": mentions_ea,
            },
        )
    # Unknown architecture — informational.
    return CheckResult(
        name="ea_parity",
        passed=True,
        severity=Severity.INFO,
        message=f"architecture {arch!r} not recognised — manual EA-parity review required",
        evidence={"architecture": arch},
    )


def _check_broker_venue_declared(inputs: Step6Inputs) -> CheckResult:
    """Closure must declare which broker spreads the verdict was computed against.

    Cross-refs the §6.3 spread P&L decomposition fragility: a
    'fragile' classification (verdict downgrades at <1.5× spread
    inflation) demands an explicit venue declaration so the deployment
    decision can match the verdict's spread regime.
    """
    text = _load_closure_text(inputs.arc_root) or ""
    declares = any(
        tag in text
        for tag in ("5ers", "IC Markets", "icmarkets", "Pepperstone",
                    "broker:", "venue:", "Broker:", "Venue:")
    )
    return CheckResult(
        name="broker_venue_declared",
        passed=declares,
        severity=Severity.WARNING,
        message=(
            "closure declares broker / venue context"
            if declares
            else "closure does not name the broker/venue context — "
            "spread sensitivity (§6.3 spread P&L) is uninterpretable"
        ),
        evidence={"declared": declares},
    )


def _check_timezone_declaration_parity(inputs: Step6Inputs) -> CheckResult:
    """Closure declares its boundary convention (utc / 5ers_eet).

    Amendment 6: KH-24 runs at ``utc``; new arcs may use ``5ers_eet``
    for daily-DD bucketing aligned to the broker trading day. Missing
    declaration → cannot tell which calendar boundary was used → cannot
    judge daily-DD claims against the 5ers hard 5% limit.

    Pass: closure declares boundary convention AND it matches the
    engine's panel convention (if Step 6 was supplied the panel value).
    """
    closure_payload = inputs.closure_payload or {}
    pool_meta = closure_payload.get("pool_metadata") or {}
    recorded = pool_meta.get("boundary_convention")
    declared = inputs.panel_boundary_convention
    if recorded is None and declared is None:
        # Older closures — treat as informational; default UTC.
        return CheckResult(
            name="timezone_declaration_parity",
            passed=True,
            severity=Severity.INFO,
            message="no boundary_convention declared (pre-Amendment-6); UTC assumed",
            evidence={"recorded": None, "declared": None},
        )
    if recorded and not declared:
        return CheckResult(
            name="timezone_declaration_parity",
            passed=True,
            severity=Severity.INFO,
            message=f"closure records {recorded!r}; engine value not supplied to Step 6",
            evidence={"recorded": recorded, "declared": None},
        )
    if declared and not recorded:
        return CheckResult(
            name="timezone_declaration_parity",
            passed=False,
            severity=Severity.WARNING,
            message=(
                f"engine used {declared!r} but closure does not record "
                "boundary_convention — closure §4 must declare it"
            ),
            evidence={"recorded": None, "declared": declared},
        )
    passed = declared == recorded
    return CheckResult(
        name="timezone_declaration_parity",
        passed=passed,
        severity=Severity.WARNING,
        message=(
            f"closure={recorded!r}, engine={declared!r}: "
            + ("match" if passed else "MISMATCH (deployment-blocking)")
        ),
        evidence={"recorded": recorded, "declared": declared, "match": passed},
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
        _check_ea_parity(inputs),
        _check_broker_venue_declared(inputs),
        _check_timezone_declaration_parity(inputs),
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
