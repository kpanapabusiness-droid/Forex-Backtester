"""Per-category markdown reports + summary + sha256_manifest.

Layout written under ``results/<arc>/step_6/``:

  manifest.json               # Step6Manifest
  summary.md                  # one-page roll-up
  lookahead_report.md         # §6.1 detail
  selection_bias_report.md    # §6.2
  execution_realism_report.md # §6.3
  statistical_report.md       # §6.4
  determinism_report.md       # §6.5
  deployment_readiness_report.md  # §6.6
  sha256_manifest.json        # per-file sha256 for two-run determinism

Manual CLI variant writes under ``results/<arc>/step_6_manual_<timestamp>/``
per chat Q6 — does not clobber the auto-dispatched run.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from core.determinism import LINE_TERMINATOR, TEXT_ENCODING
from core.step_6.manifest import (
    CategoryAuditResult,
    Severity,
    Step6Manifest,
    Step6Result,
    write_step_6_manifest,
)

_REPORT_FILENAME = {
    "lookahead": "lookahead_report.md",
    "selection_bias": "selection_bias_report.md",
    "execution_realism": "execution_realism_report.md",
    "statistical": "statistical_report.md",
    "determinism": "determinism_report.md",
    "deployment_readiness": "deployment_readiness_report.md",
}

# Sentinel key honoured in ``CategoryAuditResult.diagnostic``. When
# present, its value (a markdown string) is appended raw to the rendered
# category report AFTER the standard checks/evidence sections. Used by
# §6.3 to embed the spread P&L decomposition subsection's table without
# JSON-escaping. Keep in sync with
# ``core.step_6.execution_realism.APPENDED_MARKDOWN_KEY``.
_APPENDED_MARKDOWN_KEY = "__appended_markdown__"


def _severity_marker(severity: Severity, passed: bool) -> str:
    if passed:
        return "PASS"
    if severity == Severity.CRITICAL:
        return "FAIL-CRITICAL"
    if severity == Severity.WARNING:
        return "WARNING"
    return "INFO"


def render_category_report(
    category: CategoryAuditResult, *, no_block: bool = False
) -> str:
    """Render one category's detailed markdown report.

    ``no_block`` per chat Q4 + manual CLI ``--no-block``: every critical
    failure is rendered as ``DEMOTED-WARNING`` instead of ``FAIL-CRITICAL``
    so the report reads as diagnostic, not blocking.
    """
    lines: list[str] = []
    lines.append(f"# Step 6 — §6 {category.category} report")
    lines.append("")
    lines.append(f"- **Category:** `{category.category}`")
    lines.append(f"- **Passed:** {bool(category.passed)}  "
                 f"(critical: {category.n_critical_fails}/{category.n_critical} fails, "
                 f"warnings: {category.n_warnings}, info: {category.n_info})")
    if no_block:
        lines.append("- **--no-block:** critical failures rendered as DEMOTED-WARNING")
    lines.append("")
    if category.diagnostic:
        # The appended-markdown sentinel is rendered LAST, not in the
        # JSON diagnostic block. Strip it before serialising the rest.
        diagnostic_for_json = {
            k: v for k, v in dict(category.diagnostic).items()
            if k != _APPENDED_MARKDOWN_KEY
        }
        if diagnostic_for_json:
            lines.append("## Diagnostic")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(diagnostic_for_json, indent=2, sort_keys=True, default=str))
            lines.append("```")
            lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| # | Name | Status | Message |")
    lines.append("|---:|---|---|---|")
    for i, c in enumerate(category.checks, start=1):
        marker = _severity_marker(c.severity, c.passed)
        if no_block and marker == "FAIL-CRITICAL":
            marker = "DEMOTED-WARNING"
        msg = c.message.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {i} | `{c.name}` | {marker} | {msg} |")
    lines.append("")
    # Evidence — append per-check JSON for traceability.
    lines.append("## Evidence")
    lines.append("")
    for c in category.checks:
        lines.append(f"### `{c.name}` ({_severity_marker(c.severity, c.passed)})")
        lines.append("")
        if c.evidence:
            lines.append("```json")
            lines.append(json.dumps(dict(c.evidence), indent=2, sort_keys=True, default=str))
            lines.append("```")
        else:
            lines.append("_(no structured evidence)_")
        lines.append("")
    # Append any category-supplied raw markdown last (e.g. §6.3 spread
    # P&L decomposition subsection — needs to render as a markdown table,
    # not as JSON-escaped text in the diagnostic block).
    appended = (
        category.diagnostic.get(_APPENDED_MARKDOWN_KEY)
        if category.diagnostic else None
    )
    if appended:
        lines.append(str(appended).rstrip("\n"))
        lines.append("")
    return LINE_TERMINATOR.join(lines) + LINE_TERMINATOR


def render_summary(result: Step6Result) -> str:
    lines: list[str] = []
    lines.append(f"# Step 6 — `{result.arc_name}` summary")
    lines.append("")
    lines.append(f"- **Trigger:** `{result.trigger.value}`")
    lines.append(f"- **Overall passed:** {bool(result.overall_passed)}")
    lines.append(f"- **Verdict impact:** `{result.verdict_impact.value}`")
    lines.append(f"- **Warnings total:** {result.n_warnings_total}")
    if result.audit_config.no_block:
        lines.append("- **--no-block:** critical failures suppressed for verdict purposes")
    crit = result.critical_failures()
    if crit:
        lines.append("")
        lines.append("**Critical failures:**")
        for name in crit:
            lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## Categories")
    lines.append("")
    lines.append("| Category | Passed | Critical | Warnings | Info | Report |")
    lines.append("|---|---|---:|---:|---:|---|")
    for c in result.categories:
        report = _REPORT_FILENAME.get(c.category, f"{c.category}_report.md")
        lines.append(
            f"| {c.category} | {bool(c.passed)} | "
            f"{c.n_critical_fails}/{c.n_critical} | {c.n_warnings} | {c.n_info} | "
            f"[{report}]({report}) |"
        )
    lines.append("")
    return LINE_TERMINATOR.join(lines) + LINE_TERMINATOR


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha256_manifest(out_dir: Path, files: list[Path]) -> Path:
    """Write ``sha256_manifest.json`` for two-run determinism comparison.

    Lists every Step 6 artefact (manifest.json + per-category .md + summary.md)
    with its sha256. Used by §6.5 determinism category and by CI tests.
    """
    entries = {}
    for p in sorted(files):
        if not p.exists():
            continue
        rel = p.relative_to(out_dir)
        entries[str(rel).replace("\\", "/")] = _sha256_file(p)
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": entries,
    }
    out = out_dir / "sha256_manifest.json"
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + LINE_TERMINATOR,
        encoding=TEXT_ENCODING, newline=LINE_TERMINATOR,
    )
    return out


def write_step_6_artefacts(
    result: Step6Result,
    out_dir: Path,
    *,
    ran_at: str | None = None,
) -> Step6Manifest:
    """Write every Step 6 artefact to ``out_dir`` and return the manifest.

    Idempotent — running twice on the same Step6Result writes byte-identical
    files modulo ``ran_at`` (which is recorded in manifest.json and the
    sha256_manifest.json header).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    no_block = result.audit_config.no_block
    report_paths: dict[str, str] = {}
    written: list[Path] = []

    for category in result.categories:
        fname = _REPORT_FILENAME.get(category.category, f"{category.category}_report.md")
        path = out_dir / fname
        path.write_text(
            render_category_report(category, no_block=no_block),
            encoding=TEXT_ENCODING, newline=LINE_TERMINATOR,
        )
        report_paths[category.category] = fname
        written.append(path)

    summary_path = out_dir / "summary.md"
    summary_path.write_text(
        render_summary(result), encoding=TEXT_ENCODING, newline=LINE_TERMINATOR,
    )
    written.append(summary_path)

    manifest = Step6Manifest.from_result(
        result, report_paths=report_paths, summary_path="summary.md", ran_at=ran_at,
    )
    manifest_path = out_dir / "manifest.json"
    write_step_6_manifest(manifest, manifest_path)
    written.append(manifest_path)

    # sha256 manifest written LAST and explicitly excludes itself.
    write_sha256_manifest(out_dir, written)

    return manifest


__all__ = (
    "render_category_report",
    "render_summary",
    "write_sha256_manifest",
    "write_step_6_artefacts",
)
