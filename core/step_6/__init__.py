"""L_PROTOCOL §2 Step 6 — causal audit framework (Amendment 4).

Six audit categories run on any candidate that clears §3 constraints
#1-#9. Step 6 is the last gate per §3 "Evaluation order" — its failure
downgrades a candidate to FAIL with
``primary_failure_mode = step6_causal_audit_fail``.

Public API:

    run_step_6              # dispatch the six categories on one candidate
    Step6Result             # bundle: per-category results + overall verdict
    Step6Manifest           # serialisable artefact for the tracker payload
    CategoryAuditResult     # one category's check bundle
    CheckResult             # one check's outcome (passed + severity + evidence)
    Severity                # critical | warning | info
    AuditConfig             # per-run configuration knobs

    AUDIT_CATEGORY_NAMES    # ordered tuple of the six category names

Per chat resolution Q1 + Q2: the orchestrator calls ``run_step_6`` on
the Top-1 verdict-carrying candidate AFTER the amended gate clears
constraints #1-#9, then re-calls the amended gate with the real
``causal_audit_clean`` value to finalise the verdict.
"""

from core.step_6.manifest import (
    AUDIT_CATEGORY_NAMES,
    AuditConfig,
    CategoryAuditResult,
    CheckResult,
    Severity,
    Step6Manifest,
    Step6Result,
    VerdictImpact,
    write_step_6_manifest,
)
from core.step_6.orchestrator import run_step_6

__all__ = (
    "AUDIT_CATEGORY_NAMES",
    "AuditConfig",
    "CategoryAuditResult",
    "CheckResult",
    "Severity",
    "Step6Manifest",
    "Step6Result",
    "VerdictImpact",
    "run_step_6",
    "write_step_6_manifest",
)
