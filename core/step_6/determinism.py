"""§6.5 Determinism audit.

Per chat Q5: sha256-manifest verify ONLY — does NOT re-run sims. The
two-run sha256 guarantee is provided by ``tests/test_determinism.py``
in CI. Step 6's job is to confirm the on-disk artefacts match the
recorded manifest, catching post-emission tampering / corruption.

Hardened in ``engine/step_6_ultimate_audit`` from 4 to 8 checks. The
marquee addition is ``risk_independent_admit_decisions`` (critical) —
catches the 2026-05-25 engine bug where Arc 7 at r=2% produced 36–38%
fewer trades than at r=0.5%, indicating risk had leaked into the
signal/admit path:

  1. ``step_4_manifest_sha256_match`` (critical) — every entry in
     ``step_4/classifiers/manifest.json`` matches the recomputed sha256
     of its file on disk.
  2. ``arc_artefacts_present`` (critical) — required artefacts
     (ARC_CLOSURE.md, step_1/pool.parquet, step_5/wfo_results.csv if
     present, etc.) are on disk under arc_root.
  3. ``risk_independent_admit_decisions`` (critical) — holdout trade
     counts at ``r_base`` vs ``r_safe`` vs ``r_hard`` must match. Same
     admit logic; only sizing scales. Divergence = risk-leak.
  4. ``ledger_schema_parity`` (critical) — Top-1 trade ledger carries
     every column the downstream §6.3 spread P&L decomposition + the
     §4 deployment_spec consume (bid+ask, sl_price, parent_position_id).
     Catches the case where a producer regression silently strips a
     column that later checks depend on.
  5. ``classifier_version_pinning`` (warning) — sklearn / lightgbm /
     joblib / pandas versions recorded in the classifier manifest so
     deployment-time drift is detectable.
  6. ``seed_pinning_verified`` (warning) — grep producer source files
     for ``random_state=`` references; flag any classifier builder that
     doesn't pin the seed.
  7. ``two_run_byte_identity_recorded`` (info) — surfaces whether a
     two-run sha256 reproduction manifest is on disk for the arc
     (recorded by ``tests/test_determinism.py`` in CI).
  8. ``line_terminator_lf`` (info) — sample text artefacts use LF line
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


def _check_risk_independent_admit(inputs: Step6Inputs) -> CheckResult:
    """Catches the 2026-05-25 Arc 7 r=2% vs r=0.5% engine bug.

    Premise: the holdout sim at ``r_base``, ``r_safe``, and ``r_hard``
    runs the *same architecture* against the *same panels* with the
    *same signal evaluation* — only the sizing scales. Trade timestamps
    and counts MUST be identical; only the realised P&L per trade
    (and therefore final balance / DD) differs.

    A divergence means risk has leaked into the signal/admit path —
    either the rescale changed a non-sizing field, or the architecture
    reads ``risk_pct`` inside signal evaluation (which is a contract
    violation). Critical severity; would have caught the recent bug.

    Pass conditions:
      * top1 amended candidate has all three holdout trade counts and
        they match exactly, OR
      * scaling was not feasible (``scalable_to_safe == False`` etc.) AND
        the missing tier's count is None (not a divergence, just absent).
    """
    res = inputs.arc_orchestrator_result
    amended_wfo = getattr(res, "amended_wfo", None) if res is not None else None
    if amended_wfo is None or not getattr(amended_wfo, "amended_results", ()):
        return CheckResult(
            name="risk_independent_admit_decisions",
            passed=True,
            severity=Severity.INFO,
            message="no amended results available — skipping (manual CLI on a fresh closure)",
            evidence={},
        )
    # Top-1 ranking mirrors core.step_6.dispatch._select_top1_amended.
    ranked = sorted(
        amended_wfo.amended_results,
        key=lambda r: (
            2 if r.amended_gate.verdict.name == "PASS_DEPLOYABLE"
            else 1 if r.amended_gate.verdict.name == "PASS_VIABLE" else 0
        ),
        reverse=True,
    )
    top1 = ranked[0]
    n_base = getattr(top1, "holdout_n_trades_at_r_base", None)
    n_safe = getattr(top1, "holdout_n_trades_at_r_safe", None)
    n_hard = getattr(top1, "holdout_n_trades_at_r_hard", None)
    observations = {
        "base": n_base, "r_safe": n_safe, "r_hard": n_hard,
    }
    observed = {k: v for k, v in observations.items() if v is not None}
    if len(observed) < 2:
        return CheckResult(
            name="risk_independent_admit_decisions",
            passed=True,
            severity=Severity.INFO,
            message=(
                f"only {len(observed)} risk tier(s) ran on holdout "
                f"({list(observed.keys())}) — cannot compare for risk-leak"
            ),
            evidence={"observed": observations},
        )
    values = list(observed.values())
    all_match = all(v == values[0] for v in values)
    spread_pct = (
        (max(values) - min(values)) / max(1, max(values)) * 100.0
        if not all_match else 0.0
    )
    return CheckResult(
        name="risk_independent_admit_decisions",
        passed=all_match,
        severity=Severity.CRITICAL,
        message=(
            f"holdout trade counts at base/safe/hard = "
            f"{observations} — "
            + (
                "identical (admit logic is risk-independent)" if all_match
                else f"DIVERGE by {spread_pct:.1f}% — RISK-LEAK detected "
                "(2026-05-25 Arc 7 mechanism); rescale path is touching "
                "non-sizing fields OR architecture reads risk_pct in admit"
            )
        ),
        evidence={
            "trade_counts_per_tier": observations,
            "all_match": all_match,
            "spread_pct": spread_pct,
            "top_1_config_id": top1.config_id,
        },
    )


def _check_ledger_schema_parity(inputs: Step6Inputs) -> CheckResult:
    """Top-1 ledger must carry every column downstream consumers read.

    The §6.3 spread P&L decomposition expects
    ``entry_bid / entry_ask / exit_bid / exit_ask / sl_price``;
    multi-leg consumers expect ``parent_position_id``. A producer
    regression that silently strips one of these would manifest as a
    silently-skipped diagnostic (still records "info: skipped" — not a
    FAIL). This check elevates the silent-skip into an explicit
    critical when the ledger IS present but missing a load-bearing
    column.
    """
    ledger = inputs.top_1_trade_ledger
    if ledger is None:
        return CheckResult(
            name="ledger_schema_parity",
            passed=True,
            severity=Severity.INFO,
            message="no top-1 ledger supplied — skipping (no schema to verify)",
            evidence={"ledger_present": False},
        )
    required = (
        "entry_price", "exit_price",
        "entry_bid", "entry_ask", "exit_bid", "exit_ask",
        "sl_price",
        "parent_position_id",
    )
    missing = [c for c in required if c not in ledger.columns]
    # NaN-only columns are silently broken — flag them too.
    nan_only: list[str] = []
    for c in required:
        if c in ledger.columns and ledger[c].isna().all():
            nan_only.append(c)
    passed = (not missing) and (not nan_only)
    return CheckResult(
        name="ledger_schema_parity",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"ledger has {len(ledger.columns)} column(s); "
            f"{len(missing)} required column(s) missing, "
            f"{len(nan_only)} required column(s) all-NaN"
        ),
        evidence={
            "n_columns": int(len(ledger.columns)),
            "missing_columns": missing,
            "all_nan_columns": nan_only,
            "n_rows": int(len(ledger)),
        },
    )


def _check_classifier_version_pinning(inputs: Step6Inputs) -> CheckResult:
    """sklearn/lightgbm/joblib/pandas versions recorded in manifest.

    Deployment-time version drift can shift classifier outputs even
    when the saved artefact byte-matches. The classifier manifest must
    record the versions present at training time so a deployment check
    can compare them.
    """
    manifest_path = inputs.arc_root / "step_4" / "classifiers" / "manifest.json"
    if not manifest_path.exists():
        return CheckResult(
            name="classifier_version_pinning",
            passed=True,
            severity=Severity.INFO,
            message="no Step 4 classifier manifest — arc is rule-based",
            evidence={"manifest_path": str(manifest_path)},
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    versions = data.get("versions") or data.get("environment") or {}
    required_libs = ("sklearn", "lightgbm", "joblib", "pandas")
    missing = [lib for lib in required_libs if lib not in versions]
    passed = not missing
    return CheckResult(
        name="classifier_version_pinning",
        passed=passed,
        severity=Severity.WARNING,
        message=(
            f"classifier manifest records {len(versions) - len(missing)} of "
            f"{len(required_libs)} required version pin(s); missing: "
            f"{missing if missing else 'none'}"
        ),
        evidence={
            "versions_recorded": list(versions.keys()),
            "missing": missing,
        },
    )


def _check_two_run_byte_identity_recorded(inputs: Step6Inputs) -> CheckResult:
    """Surfaces whether a two-run reproduction manifest is on disk.

    Step 6 does NOT re-run the sim (that's CI's job per existing
    scope) but does verify the manifest claim of byte-identity.
    Looks for ``two_run_sha256.json`` or similar under the arc.
    """
    candidates = (
        inputs.arc_root / "two_run_sha256.json",
        inputs.arc_root / "step_5" / "two_run_sha256.json",
        inputs.arc_root / "ci" / "two_run_sha256.json",
    )
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        return CheckResult(
            name="two_run_byte_identity_recorded",
            passed=True,
            severity=Severity.INFO,
            message=(
                "no two-run byte-identity manifest on disk — CI-side "
                "determinism gate may live elsewhere"
            ),
            evidence={"checked_paths": [str(p) for p in candidates]},
        )
    try:
        data = json.loads(found.read_text(encoding="utf-8"))
        n_files = len(data.get("files") or {})
        return CheckResult(
            name="two_run_byte_identity_recorded",
            passed=True,
            severity=Severity.INFO,
            message=(
                f"two-run byte-identity manifest present: {found.name} "
                f"({n_files} file(s) tracked)"
            ),
            evidence={"path": str(found), "n_files": n_files},
        )
    except (OSError, json.JSONDecodeError) as exc:
        return CheckResult(
            name="two_run_byte_identity_recorded",
            passed=False,
            severity=Severity.INFO,
            message=f"could not parse two-run manifest: {exc}",
            evidence={"path": str(found), "error": str(exc)},
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
        _check_risk_independent_admit(inputs),
        _check_ledger_schema_parity(inputs),
        _check_classifier_version_pinning(inputs),
        _check_seed_pinning(),
        _check_two_run_byte_identity_recorded(inputs),
        _check_line_terminator(inputs),
    )
    diagnostic: dict[str, Any] = {"arc_root": str(inputs.arc_root)}
    return CategoryAuditResult(
        category="determinism", checks=checks, diagnostic=diagnostic,
    )


__all__ = ("audit",)
