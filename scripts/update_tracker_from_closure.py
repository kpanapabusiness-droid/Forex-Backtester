"""Tracker parser CLI — ingests one ARC_CLOSURE.md, applies §4 A-K mappings to ARC_TRACKER.md.

Usage:
    python scripts/update_tracker_from_closure.py results/<arc>/ARC_CLOSURE.md [options]

Options:
    --dry-run               Parse + validate, print intended mutations, do not write.
    --tracker-path PATH     Override ARC_TRACKER.md location (default: repo root).
    --rolling-state PATH    Override rolling_state.json location.
    --registry PATH         Override parsed.log location.
    --verbose               Print the parsed payload + planned mutations.

Exit codes:
    0  success or no-op (already-parsed sha256)
    1  schema / payload error
    2  tracker IO error
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is importable when invoked as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tracker_parser import (  # noqa: E402
    extract,
    mapping,
    registry,
    rolling_state,
    schema,
    tracker_io,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Apply ARC_CLOSURE.md §1 tracker_payload to ARC_TRACKER.md."
    )
    p.add_argument(
        "closure_path",
        type=Path,
        help="Path to results/<arc>/ARC_CLOSURE.md",
    )
    p.add_argument("--dry-run", action="store_true", help="Validate + plan; do not write.")
    p.add_argument(
        "--tracker-path",
        type=Path,
        default=_REPO_ROOT / "ARC_TRACKER.md",
        help="ARC_TRACKER.md path (default: repo-root ARC_TRACKER.md)",
    )
    p.add_argument(
        "--rolling-state",
        type=Path,
        default=None,
        help="rolling_state.json path (default: scripts/tracker_parser/rolling_state.json)",
    )
    p.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="parsed.log path (default: scripts/tracker_parser/parsed.log)",
    )
    p.add_argument("--verbose", action="store_true", help="Print parsed payload + mutations.")
    return p


def _is_post_cutoff(closed_ts: str | None, cutoff_iso: str) -> bool:
    """Return True iff ``closed_ts`` is strictly after ``cutoff_iso``.

    Closures missing or with malformed timestamps are treated as PRE-cutoff
    (grandfathered) — Phase 2 tightening kicks in only when we can confidently
    determine the closure post-dates the cutoff.
    """
    if not closed_ts:
        return False
    from datetime import datetime, timezone
    s = str(closed_ts).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    cutoff_s = cutoff_iso[:-1] + "+00:00" if cutoff_iso.endswith("Z") else cutoff_iso
    cutoff_dt = datetime.fromisoformat(cutoff_s).astimezone(timezone.utc)
    return dt_utc > cutoff_dt


def _is_deferred_provisional(payload: dict) -> bool:
    """True iff closure ships with explicit Amendment 3 deferral (PROVISIONAL pattern).

    Detection rule: verdict ends in '-PROVISIONAL' AND `amendment_3_evaluation.ran`
    is the literal False. When this returns True, downstream PASS-verdict gates
    (config_artefact_path resolution, §4 deployment_spec heading, Amendment 3
    fields populated, step_6 overall_passed) are downgraded from ERROR to WARNING
    — the closure addendum is the source of truth for those fields and the
    parser permits the deferred-state to commit so the §3 cross-arc work is not
    blocked on the engine fix / addendum sequencing.

    The closure addendum (post-engine-fix-merge or post-deferral-completion) MUST
    backfill the deferred fields; the parser will then re-validate at full
    strictness on re-parse.

    Mirrors the Amendment 5.1 deferral-tolerance pattern in
    ``VALID_ARCHITECTURES_SKIPPED_REASONS`` / ``_validate_amendment_5_1_field``.
    """
    verdict_str = str(payload.get("verdict", ""))
    if not verdict_str.endswith("-PROVISIONAL"):
        return False
    a3 = payload.get("amendment_3_evaluation")
    if not isinstance(a3, dict):
        return False
    return a3.get("ran") is False


def _is_post_phase_2_cutoff(closed_ts: str | None) -> bool:
    """Return True iff ``closed_ts`` is strictly after the PR-186 merge cutoff
    (2026-05-23T06:20:59Z per chat Q7) and therefore subject to Phase 2 tightening.
    """
    return _is_post_cutoff(closed_ts, schema.PHASE_2_CUTOFF_ISO)


def _is_post_amendment_5_cutoff(closed_ts: str | None) -> bool:
    """Return True iff ``closed_ts`` is strictly after the L_PROTOCOL Amendment 5
    cutoff (placeholder pinned at the ratification date; backfilled with this PR's
    actual merge timestamp post-merge — see ``schema.AMENDMENT_5_CUTOFF_ISO``).
    """
    return _is_post_cutoff(closed_ts, schema.AMENDMENT_5_CUTOFF_ISO)


def _is_post_amendment_5_1_cutoff(closed_ts: str | None) -> bool:
    """Return True iff ``closed_ts`` is strictly after the L_PROTOCOL Amendment 5.1
    cutoff (placeholder pinned at the ratification date; backfilled with this PR's
    actual merge timestamp post-merge — see ``schema.AMENDMENT_5_1_CUTOFF_ISO``).
    """
    return _is_post_cutoff(closed_ts, schema.AMENDMENT_5_1_CUTOFF_ISO)


def _validate_phase_2_amendment_3_fields(payload: dict, closure_path: Path) -> int:
    """Phase 2 tightening: for PASS verdicts with closed_timestamp > PR-186 merge,
    REQUIRE the Amendment 3 risk-normalised fields on `best_architecture`
    (chat Q7 / template v1.3 Schema versioning row).

    Pre-cutoff closures grandfathered (return 0).
    """
    ba = payload.get("best_architecture") or {}
    required = (
        "chained_max_dd_base_pct",
        "k_safe",
        "k_hard",
        "r_safe_pct",
        "r_hard_pct",
        "scalable_to_safe",
        "scalable_to_hard",
    )
    missing = [k for k in required if ba.get(k) is None]
    if missing:
        logging.error(
            "Phase 2 tightening: PASS verdict at closed_timestamp=%r is post-PR-186-merge "
            "(%s) but best_architecture is missing Amendment 3 fields: %s. Closure %s.",
            payload.get("closed_timestamp"),
            schema.PHASE_2_CUTOFF_ISO,
            missing,
            closure_path,
        )
        return 1
    return 0


def _validate_amendment_5_field(payload: dict, closure_path: Path) -> int:
    """L_PROTOCOL Amendment 5 Phase 2 tightening (template v1.3.1 Schema versioning row):
    PASS verdicts closed strictly after the Amendment-5 cutoff MUST carry the
    `architectures_skipped_by_amendment_5` field (may be `[]`).

    Pre-cutoff closures grandfathered. The field is informational — it captures
    architectures admissible under Amendment 1's archetype-driven rule but skipped
    under Amendment 5's four-gate AUC-driven rule. `[]` indicates the Amendment-5
    set equals or supersets the Amendment-1 set.
    """
    if payload.get("architectures_skipped_by_amendment_5") is None:
        logging.error(
            "L_PROTOCOL Amendment 5: PASS verdict at closed_timestamp=%r is post-Amendment-5-cutoff "
            "(%s) but `architectures_skipped_by_amendment_5` is missing. Closure %s. "
            "Use `[]` when no architectures were skipped under Amendment 5.",
            payload.get("closed_timestamp"),
            schema.AMENDMENT_5_CUTOFF_ISO,
            closure_path,
        )
        return 1
    return 0


_A5_1_GATE_4_REASON = "a5_gate_4_admission_blocked_by_no_pass_tier_constituent"


def _count_candidate_clusters_surviving_step_3(payload: dict) -> int:
    """Re-derive the §3 candidate-cluster count from the closure body.

    L_PROTOCOL §3 (capturability): a cluster is flagged as a candidate cluster
    when ``reach_1r >= 0.50 AND ww_pp <= 0.30 AND mfe_p50_r >= 1.5``. We re-derive
    here rather than persist the flag (the closure schema doesn't carry it
    explicitly; only the underlying metrics).
    """
    clusters = payload.get("clusters") or {}
    if not isinstance(clusters, dict):
        return 0
    count = 0
    for c in clusters.values():
        if not isinstance(c, dict):
            continue
        reach = c.get("reach_1r")
        wwpp = c.get("ww_pp")
        mfe = c.get("mfe_p50_r")
        if reach is None or wwpp is None or mfe is None:
            continue
        try:
            if float(reach) >= 0.50 and float(wwpp) <= 0.30 and float(mfe) >= 1.5:
                count += 1
        except (TypeError, ValueError):
            continue
    return count


def _validate_amendment_5_1_gate_4_qualifier(payload: dict, closure_path: Path) -> int:
    """L_PROTOCOL Amendment 5.1 (2026-05-25) — Gate 4 PASS-tier-constituent qualifier.

    When a post-AMENDMENT_5_1_CUTOFF_ISO PASS closure declares ≥2 candidate
    clusters surviving Step 3, ensure A5 admission state is explicit in
    architectures_skipped_by_amendment_5:
      - If A5 was admitted and ran: no entry needed
      - If A5 was not admitted because no constituent PASSed: MUST list
        "a5_gate_4_admission_blocked_by_no_pass_tier_constituent"
    Pre-cutoff closures grandfathered.

    Phase 1 (this PR): WARNING-level — logs and returns 0 (does not HALT).
    Phase 2 (post-AMENDMENT_5_1_CUTOFF_ISO backfill): upgrade to ERROR-level
    by returning 1 on the missing-reason branch.
    """
    n_candidate = _count_candidate_clusters_surviving_step_3(payload)
    if n_candidate < 2:
        return 0

    arches_tested = payload.get("architectures_tested") or []
    arch_results = payload.get("architecture_results") or {}
    a5_ran = (
        "A5" in arches_tested
        or bool((arch_results.get("A5") or {}).get("tested"))
    )
    if a5_ran:
        return 0

    skipped = payload.get("architectures_skipped_by_amendment_5") or []
    if _A5_1_GATE_4_REASON in skipped:
        return 0

    logging.warning(
        "L_PROTOCOL Amendment 5.1 (Phase 1 WARNING): PASS verdict at closed_timestamp=%r is "
        "post-Amendment-5.1-cutoff (%s) with %d candidate clusters surviving Step 3, but A5 was "
        "not admitted and `architectures_skipped_by_amendment_5` does not include %r. "
        "Per Gate 4 qualifier, either A5 must run or the reason string must be cited. "
        "Closure %s. (Phase 2 will upgrade this to ERROR-level after cutoff backfill.)",
        payload.get("closed_timestamp"),
        schema.AMENDMENT_5_1_CUTOFF_ISO,
        n_candidate,
        _A5_1_GATE_4_REASON,
        closure_path,
    )
    return 0


def _validate_v13_pass_step6(payload: dict, closure_path: Path) -> int:
    """Phase 2 tightening: v1.3 PASS verdicts MUST carry a step_6 block with
    overall_passed=true (chat Q7 / template v1.3 Schema versioning row).

    Manual invocations cannot satisfy this — only auto-dispatch can produce
    overall_passed=true with verdict_impact=none.
    """
    step6 = payload.get("step_6")
    if not isinstance(step6, dict):
        logging.error(
            "v1.3 PASS-verdict validation: §1 tracker_payload.step_6 block missing. "
            "Closure %s has verdict %r — step_6 block REQUIRED for v1.3 PASS verdicts.",
            closure_path,
            payload.get("verdict"),
        )
        return 1
    if not step6.get("ran"):
        logging.error(
            "v1.3 PASS-verdict validation: step_6.ran is false but verdict is %r. "
            "PASS verdicts cannot ship without Step 6 dispatch.",
            payload.get("verdict"),
        )
        return 1
    if step6.get("overall_passed") is not True:
        logging.error(
            "v1.3 PASS-verdict validation: step_6.overall_passed is %r — must be true "
            "for PASS verdicts. (verdict_impact=%r)",
            step6.get("overall_passed"),
            step6.get("verdict_impact"),
        )
        return 1
    return 0


def _validate_v12_pass_verdict(payload: dict, closure_path: Path) -> int:
    """v1.2 PASS-verdict validation per template Section 4-L.

    Checks (all must pass):
      1. `best_architecture.config_artefact_path` is non-null.
      2. The file at that path (relative to repo root) exists.
      3. The closure doc contains a `## §4 deployment_spec` heading.
      4. `best_architecture.deployment_spec_section_present` is true.

    Returns 0 on pass, 1 on any failure (with logged error). No tracker writes happen if any
    check fails — the caller short-circuits before the mapping layer.
    """
    ba = payload.get("best_architecture") or {}
    config_path_str = ba.get("config_artefact_path")
    if not config_path_str:
        logging.error(
            "v1.2 PASS-verdict validation: best_architecture.config_artefact_path is null or missing. "
            "Closure %s has verdict %r — config_artefact_path is REQUIRED for PASS verdicts. "
            "(template Section 4-L, item 1)",
            closure_path,
            payload.get("verdict"),
        )
        return 1

    config_path = (_REPO_ROOT / config_path_str).resolve()
    if not config_path.exists():
        logging.error(
            "v1.2 PASS-verdict validation: config_artefact_path %r does not exist (resolved to %s). "
            "(template Section 4-L, item 2)",
            config_path_str,
            config_path,
        )
        return 1

    if not extract.has_deployment_spec_heading(closure_path):
        logging.error(
            "v1.2 PASS-verdict validation: closure doc %s is missing the `## §4 deployment_spec` "
            "heading. (template Section 4-L, item 3)",
            closure_path,
        )
        return 1

    if ba.get("deployment_spec_section_present") is not True:
        logging.error(
            "v1.2 PASS-verdict validation: best_architecture.deployment_spec_section_present is "
            "%r — must be true for PASS verdicts. (template Section 4-L, item 4)",
            ba.get("deployment_spec_section_present"),
        )
        return 1

    return 0


def _diff_tracker(before: bytes, after: bytes) -> str:
    """Return a human-readable summary of changes between two tracker byte sequences."""
    if before == after:
        return "(no changes)"
    before_lines = before.decode("utf-8").splitlines(keepends=True)
    after_lines = after.decode("utf-8").splitlines(keepends=True)
    import difflib

    diff = difflib.unified_diff(
        before_lines, after_lines, fromfile="ARC_TRACKER.md (before)", tofile="ARC_TRACKER.md (after)"
    )
    return "".join(diff)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    closure_path: Path = args.closure_path
    tracker_path: Path = args.tracker_path

    if not closure_path.exists():
        logging.error("closure path does not exist: %s", closure_path)
        return 1
    if not tracker_path.exists():
        logging.error("tracker path does not exist: %s", tracker_path)
        return 2

    # 1. sha256 + idempotency check
    sha = extract.closure_sha256(closure_path)
    if registry.already_parsed(sha, args.registry):
        logging.info("already parsed (sha256=%s), no-op.", sha)
        return 0

    # 2. Extract + validate payload
    try:
        raw_payload = extract.extract_payload(closure_path)
    except extract.ClosureExtractionError as exc:
        logging.error("extraction failed: %s", exc)
        return 1

    try:
        payload = schema.parse_payload(raw_payload)
    except Exception as exc:  # pydantic.ValidationError, ValueError
        logging.error("schema validation failed: %s", exc)
        return 1

    template_version = payload.get("template_version")
    verdict_str = str(payload.get("verdict", ""))
    is_pass = verdict_str.startswith("PASS-")

    # Detect deferral state — closure ships PROVISIONAL with Amendment 3 deferral
    # block populated. Downstream PASS-verdict gates downgrade ERROR to WARNING
    # since the closure addendum (post-engine-fix-merge or post-deferral) is the
    # source of truth for the deferred fields.
    deferred_provisional = _is_deferred_provisional(payload)

    # v1.2+ PASS-verdict validation (template Section 4-L) — config_artefact_path + §4 heading.
    if template_version in ("1.2", "1.3") and is_pass:
        rc = _validate_v12_pass_verdict(payload, closure_path)
        if rc != 0:
            if deferred_provisional:
                logging.warning(
                    "Section 4-L validation failed but closure is in deferred-PROVISIONAL state; "
                    "downgrading to WARNING. Closure addendum must backfill config_artefact_path + §4."
                )
            else:
                return rc

    # Phase 2 tightening (template v1.3 Schema versioning row, chat Q7):
    # PASS verdict closed after PR-186 merge MUST have Amendment 3 fields.
    if is_pass and _is_post_phase_2_cutoff(payload.get("closed_timestamp")):
        rc = _validate_phase_2_amendment_3_fields(payload, closure_path)
        if rc != 0:
            if deferred_provisional:
                logging.warning(
                    "Phase 2 Amendment 3 field validation failed but closure is in deferred-PROVISIONAL "
                    "state; downgrading to WARNING. Closure addendum must backfill Amendment 3 fields."
                )
            else:
                return rc

    # v1.3 PASS verdicts MUST have a step_6 block with overall_passed=true.
    if template_version == "1.3" and is_pass:
        rc = _validate_v13_pass_step6(payload, closure_path)
        if rc != 0:
            if deferred_provisional:
                logging.warning(
                    "Step 6 overall_passed validation failed but closure is in deferred-PROVISIONAL "
                    "state; downgrading to WARNING. Step 6 gates on Amendment 3 PASS-tier "
                    "classification; closure addendum will populate step_6 block."
                )
            else:
                return rc

    # L_PROTOCOL Amendment 5 (template v1.3.1 Schema versioning row):
    # PASS verdicts closed after Amendment-5 cutoff MUST carry
    # `architectures_skipped_by_amendment_5`.
    if is_pass and _is_post_amendment_5_cutoff(payload.get("closed_timestamp")):
        rc = _validate_amendment_5_field(payload, closure_path)
        if rc != 0:
            return rc

    # L_PROTOCOL Amendment 5.1 — Gate 4 PASS-tier-constituent qualifier.
    # Phase 1: WARNING-level check; never HALTs. Upgrade to ERROR after the
    # AMENDMENT_5_1_CUTOFF_ISO backfill (see standing TODO).
    if is_pass and _is_post_amendment_5_1_cutoff(payload.get("closed_timestamp")):
        _validate_amendment_5_1_gate_4_qualifier(payload, closure_path)

    if args.verbose:
        print("--- Parsed payload (v1.1-normalised) ---")
        import json

        print(json.dumps(payload, indent=2, default=str))

    # 3. Load tracker + rolling state, snapshot before
    try:
        state = tracker_io.read_tracker(tracker_path)
    except Exception as exc:
        logging.error("tracker read failed: %s", exc)
        return 2
    rolling = rolling_state.load_state(args.rolling_state)

    before_bytes = state.to_bytes()

    # 4. Apply mappings
    try:
        mapping.apply_payload(state, payload, rolling)
    except Exception as exc:
        logging.error("mapping application failed: %s", exc)
        return 1

    after_bytes = state.to_bytes()

    if args.verbose or args.dry_run:
        print("\n--- Tracker diff ---")
        print(_diff_tracker(before_bytes, after_bytes))

    # 5. Commit (or skip on dry-run)
    if args.dry_run:
        logging.info("--dry-run: no writes performed.")
        return 0

    try:
        state.write()
    except Exception as exc:
        logging.error("tracker write failed: %s", exc)
        return 2
    rolling_state.save_state(rolling, args.rolling_state)
    registry.record_parse(sha, payload["arc_name"], args.registry)
    logging.info(
        "applied %s (sha256=%s) — tracker updated.", payload["arc_name"], sha
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
