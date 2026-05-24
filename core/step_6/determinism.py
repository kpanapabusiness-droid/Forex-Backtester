"""§6.5 Determinism audit.

Per chat Q5: sha256-manifest verify ONLY — does NOT re-run sims. The
two-run sha256 guarantee is provided by ``tests/test_determinism.py``
in CI. Step 6's job is to confirm the on-disk artefacts match the
recorded manifest, catching post-emission tampering / corruption.

Four checks:

  1. ``step_4_manifest_sha256_match`` (critical) — every entry in
     ``step_4/classifiers/manifest.json`` matches the recomputed sha256
     of its file on disk.
  2. ``arc_artefacts_present`` (critical) — required artefacts
     (ARC_CLOSURE.md, step_1/pool.parquet, step_5/wfo_results.csv if
     present, etc.) are on disk under arc_root.
  3. ``seed_pinning_verified`` (warning) — grep producer source files
     for ``random_state=`` references; flag any classifier builder that
     doesn't pin the seed.
  4. ``line_terminator_lf`` (info) — sample text artefacts use LF line
     endings.
"""

from __future__ import annotations

import hashlib
import inspect
import json
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

_RANDOM_STATE_RE = re.compile(r"random_state\s*=\s*(\d+|RANDOM_STATE)")
_REQUIRED_ARTEFACTS = (
    Path("ARC_CLOSURE.md"),
    Path("step_1/pool.parquet"),
    Path("step_3/capturability.csv"),
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_step_4_manifest(inputs: Step6Inputs) -> CheckResult:
    manifest_path = inputs.arc_root / "step_4" / "classifiers" / "manifest.json"
    if not manifest_path.exists():
        return CheckResult(
            name="step_4_manifest_sha256_match",
            passed=True,
            severity=Severity.INFO,
            message="no Step 4 classifier manifest — arc is rule-based",
            evidence={"manifest_path": str(manifest_path)},
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    classifiers = data.get("classifiers") or {}
    mismatches: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    for cid, entry in classifiers.items():
        rel_path = entry.get("path")
        expected = entry.get("sha256")
        if rel_path is None or expected is None:
            mismatches.append({"cluster_id": cid, "issue": "missing path or sha256"})
            continue
        full = manifest_path.parent / rel_path
        if not full.exists():
            mismatches.append({"cluster_id": cid, "issue": "file missing", "path": str(full)})
            continue
        actual = _sha256_file(full)
        if actual != expected:
            mismatches.append({
                "cluster_id": cid,
                "issue": "sha256 mismatch",
                "expected": expected,
                "actual": actual,
            })
        else:
            verified.append({"cluster_id": cid, "sha256": actual[:16] + "..."})
    passed = len(mismatches) == 0
    return CheckResult(
        name="step_4_manifest_sha256_match",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"{len(verified)} classifier(s) verified, {len(mismatches)} mismatch(es)"
        ),
        evidence={
            "n_verified": len(verified),
            "n_mismatched": len(mismatches),
            "mismatches": mismatches,
            "verified": verified,
        },
    )


def _check_arc_artefacts_present(inputs: Step6Inputs) -> CheckResult:
    missing: list[str] = []
    present: list[str] = []
    for rel in _REQUIRED_ARTEFACTS:
        full = inputs.arc_root / rel
        if full.exists():
            present.append(str(rel))
        else:
            missing.append(str(rel))
    passed = len(missing) == 0
    return CheckResult(
        name="arc_artefacts_present",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"{len(present)}/{len(_REQUIRED_ARTEFACTS)} required artefact(s) on disk; "
            f"missing: {missing if missing else 'none'}"
        ),
        evidence={"present": present, "missing": missing, "arc_root": str(inputs.arc_root)},
    )


def _check_seed_pinning() -> CheckResult:
    try:
        from core.steps import _classifier_defaults  # type: ignore
    except ImportError as exc:
        return CheckResult(
            name="seed_pinning_verified",
            passed=False,
            severity=Severity.WARNING,
            message=f"could not import _classifier_defaults: {exc}",
            evidence={"error": str(exc)},
        )
    try:
        source = inspect.getsource(_classifier_defaults)
    except OSError as exc:
        return CheckResult(
            name="seed_pinning_verified",
            passed=False,
            severity=Severity.WARNING,
            message=f"could not read source: {exc}",
            evidence={"error": str(exc)},
        )
    matches = _RANDOM_STATE_RE.findall(source)
    builders = source.count("def build_")
    passed = len(matches) >= builders and builders > 0
    return CheckResult(
        name="seed_pinning_verified",
        passed=passed,
        severity=Severity.WARNING,
        message=(
            f"{len(matches)} random_state= reference(s) across {builders} builder(s)"
        ),
        evidence={
            "n_random_state_references": len(matches),
            "n_builders": builders,
            "seeds_seen": matches,
        },
    )


def _check_line_terminator(inputs: Step6Inputs) -> CheckResult:
    # Sample ARC_CLOSURE.md + capturability summary if present.
    samples = (
        inputs.arc_root / "ARC_CLOSURE.md",
        inputs.arc_root / "step_3" / "capturability_summary.md",
    )
    findings: list[dict[str, Any]] = []
    for p in samples:
        if not p.exists():
            continue
        data = p.read_bytes()
        has_crlf = b"\r\n" in data
        findings.append({"path": str(p.relative_to(inputs.arc_root)), "has_crlf": has_crlf})
    if not findings:
        return CheckResult(
            name="line_terminator_lf",
            passed=True,
            severity=Severity.INFO,
            message="no sample text artefacts available",
            evidence={},
        )
    any_crlf = any(f["has_crlf"] for f in findings)
    return CheckResult(
        name="line_terminator_lf",
        passed=not any_crlf,
        severity=Severity.INFO,
        message=(
            "all sampled text artefacts use LF" if not any_crlf
            else f"{sum(1 for f in findings if f['has_crlf'])} artefact(s) carry CRLF"
        ),
        evidence={"samples": findings},
    )


def audit(inputs: Step6Inputs, audit_config: AuditConfig) -> CategoryAuditResult:
    checks = (
        _check_step_4_manifest(inputs),
        _check_arc_artefacts_present(inputs),
        _check_seed_pinning(),
        _check_line_terminator(inputs),
    )
    diagnostic: dict[str, Any] = {"arc_root": str(inputs.arc_root)}
    return CategoryAuditResult(
        category="determinism", checks=checks, diagnostic=diagnostic,
    )


__all__ = ("audit",)
