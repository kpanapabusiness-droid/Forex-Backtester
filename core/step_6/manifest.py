"""Step 6 dataclasses + manifest serde.

`CheckResult` is one check (one row in a category report). `CategoryAuditResult`
bundles all checks for one category. `Step6Result` bundles all six categories.
`Step6Manifest` is the on-disk JSON artefact written to
``results/<arc>/step_6/manifest.json``.

Severity rules per chat resolution Q4:
- ``critical`` failure → category FAIL → Step 6 FAIL → verdict downgrade
- ``warning`` failure → category PASS but flagged
- ``info`` failure → recorded only; no effect on category outcome

``CategoryAuditResult.passed`` is the AND over critical-severity checks only.
Warnings tracked separately as ``n_warnings``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from core.determinism import LINE_TERMINATOR, TEXT_ENCODING


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class VerdictImpact(str, Enum):
    NONE = "none"
    DOWNGRADED_TO_FAIL = "downgraded_to_fail"


class TriggerSource(str, Enum):
    AUTO_PASS = "auto_pass"
    MANUAL = "manual"
    NOT_APPLICABLE = "not_applicable"


AUDIT_CATEGORY_NAMES: tuple[str, ...] = (
    "lookahead",
    "selection_bias",
    "execution_realism",
    "statistical",
    "determinism",
    "deployment_readiness",
)


@dataclass(frozen=True)
class CheckResult:
    """One audit check outcome."""

    name: str
    passed: bool
    severity: Severity
    message: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "severity": self.severity.value,
            "message": self.message,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class CategoryAuditResult:
    """All checks for one audit category.

    ``passed`` per chat Q4 = AND over critical-severity check results.
    Warnings flag but do not block; info records only.
    """

    category: str
    checks: tuple[CheckResult, ...]
    diagnostic: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == Severity.CRITICAL)

    @property
    def n_critical(self) -> int:
        return sum(1 for c in self.checks if c.severity == Severity.CRITICAL)

    @property
    def n_critical_fails(self) -> int:
        return sum(
            1 for c in self.checks
            if c.severity == Severity.CRITICAL and not c.passed
        )

    @property
    def n_warnings(self) -> int:
        return sum(
            1 for c in self.checks
            if c.severity == Severity.WARNING and not c.passed
        )

    @property
    def n_info(self) -> int:
        return sum(
            1 for c in self.checks
            if c.severity == Severity.INFO and not c.passed
        )

    def critical_failures(self) -> tuple[str, ...]:
        return tuple(
            c.name for c in self.checks
            if c.severity == Severity.CRITICAL and not c.passed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "passed": bool(self.passed),
            "n_checks": len(self.checks),
            "n_critical": self.n_critical,
            "n_critical_fails": self.n_critical_fails,
            "n_warnings": self.n_warnings,
            "n_info": self.n_info,
            "critical_failures": list(self.critical_failures()),
            "checks": [c.to_dict() for c in self.checks],
            "diagnostic": dict(self.diagnostic),
        }


@dataclass(frozen=True)
class AuditConfig:
    """Per-run configuration knobs.

    ``no_block`` (manual CLI ``--no-block`` flag) demotes every critical
    failure to a warning in the rendered report and prevents the
    orchestrator from downgrading the verdict — diagnostic only.

    ``categories_to_run`` restricts the dispatch to a subset; ``None``
    runs all six.

    ``byte_compare_n_samples`` controls how many random trades are
    re-computed from raw OHLC for the lookahead category.

    ``spread_delta_warn_pct`` / ``spread_delta_critical_pct`` configure
    execution-realism §6.3 thresholds.
    """

    no_block: bool = False
    categories_to_run: tuple[str, ...] | None = None
    byte_compare_n_samples: int = 5
    byte_compare_seed: int = 42
    spread_delta_warn_pct: float = 0.20
    spread_delta_critical_pct: float = 0.50
    lo_corrected_min_trades: int = 100
    correlation_warn_threshold: float = 0.70


@dataclass(frozen=True)
class Step6Result:
    """Bundle of all six category results + overall verdict + trigger."""

    arc_name: str
    trigger: TriggerSource
    categories: tuple[CategoryAuditResult, ...]
    audit_config: AuditConfig

    @property
    def overall_passed(self) -> bool:
        return all(c.passed for c in self.categories)

    @property
    def verdict_impact(self) -> VerdictImpact:
        # Manual invocations NEVER downgrade per Q6 + dispatch
        # "Discipline rules" line "Manual invocations don't change verdicts".
        # --no-block also short-circuits per AuditConfig docstring.
        if self.trigger == TriggerSource.MANUAL or self.audit_config.no_block:
            return VerdictImpact.NONE
        if self.overall_passed:
            return VerdictImpact.NONE
        return VerdictImpact.DOWNGRADED_TO_FAIL

    @property
    def n_warnings_total(self) -> int:
        return sum(c.n_warnings for c in self.categories)

    def critical_failures(self) -> tuple[str, ...]:
        out: list[str] = []
        for c in self.categories:
            for name in c.critical_failures():
                out.append(f"{c.category}.{name}")
        return tuple(out)


@dataclass(frozen=True)
class Step6Manifest:
    """On-disk manifest schema for ``results/<arc>/step_6/manifest.json``.

    Field set tracks the dispatch's Task 4 manifest shape verbatim so
    closure-template v1.3 `§1 tracker_payload.step_6` consumers can mirror
    these field names 1:1.
    """

    arc_name: str
    ran_at: str  # ISO-8601 UTC, second precision
    trigger: TriggerSource
    overall_passed: bool
    verdict_impact: VerdictImpact
    categories: tuple[CategoryAuditResult, ...]
    report_paths: Mapping[str, str]  # category -> "step_6/<category>_report.md"
    summary_path: str
    critical_failures: tuple[str, ...]
    n_warnings: int

    @classmethod
    def from_result(
        cls,
        result: Step6Result,
        *,
        report_paths: Mapping[str, str],
        summary_path: str,
        ran_at: str | None = None,
    ) -> "Step6Manifest":
        return cls(
            arc_name=result.arc_name,
            ran_at=ran_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            trigger=result.trigger,
            overall_passed=result.overall_passed,
            verdict_impact=result.verdict_impact,
            categories=result.categories,
            report_paths=dict(report_paths),
            summary_path=summary_path,
            critical_failures=result.critical_failures(),
            n_warnings=result.n_warnings_total,
        )

    def to_dict(self) -> dict[str, Any]:
        # Per-category summary (compact) — full check detail lives in the
        # markdown reports under report_paths[category].
        categories_summary: dict[str, dict[str, Any]] = {}
        for c in self.categories:
            categories_summary[c.category] = {
                "passed": bool(c.passed),
                "n_checks": len(c.checks),
                "n_critical": c.n_critical,
                "n_critical_fails": c.n_critical_fails,
                "n_warnings": c.n_warnings,
                "n_info": c.n_info,
            }
        return {
            "arc_name": self.arc_name,
            "ran_at": self.ran_at,
            "trigger": self.trigger.value,
            "overall_passed": bool(self.overall_passed),
            "verdict_impact": self.verdict_impact.value,
            "categories": categories_summary,
            "report_paths": dict(self.report_paths),
            "summary_path": self.summary_path,
            "critical_failures": list(self.critical_failures),
            "n_warnings": int(self.n_warnings),
        }


def write_step_6_manifest(manifest: Step6Manifest, out_path: Path) -> Path:
    """Serialise manifest to ``out_path`` (UTF-8, LF, sort_keys=True)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + LINE_TERMINATOR
    out_path.write_text(body, encoding=TEXT_ENCODING, newline=LINE_TERMINATOR)
    return out_path


__all__ = (
    "AUDIT_CATEGORY_NAMES",
    "AuditConfig",
    "CategoryAuditResult",
    "CheckResult",
    "Severity",
    "Step6Manifest",
    "Step6Result",
    "TriggerSource",
    "VerdictImpact",
    "write_step_6_manifest",
)
