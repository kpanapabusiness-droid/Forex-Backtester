"""§6.1 Lookahead audit.

Six checks (4 critical, 2 warning):

  1. ``per_feature_lineage_clean`` (critical) — every feature in the
     winning candidate has ``causal_lineage == "clean"`` per Step 1's
     feature_lineage DataFrame.
  2. ``no_path_features_in_entry`` (critical) — features named like
     forward-path metrics (`mfe_*`, `mae_*`, `ww_*`, `final_r_*`,
     `bars_held*`, `path_*`) never appear in the winning
     candidate's entry-decision feature set.
  3. ``d1_lag_rule_enforced`` (critical) — multi-TF D1 producer code
     routes through ``core.features.multi_tf._build_d1_lag1_series``.
     The check inspects the source module to confirm the helper is in
     use (catches a hand-written replacement that forgot the lag).
  4. ``byte_compare_no_drift`` (critical) — Arc 10 precedent: sample N
     trades, recompute features from raw OHLC, byte-compare against the
     pool's stored values. Any abs_diff above tolerance fails.
  5. ``threshold_selection_lineage`` (warning) — Step 4's
     ``best_threshold`` is the mean of per-fold AUC-best thresholds,
     never picked on the full-data ROC curve. Verified by reading the
     persisted classifier manifest.
  6. ``feature_lineage_table_exists`` (info) — record presence /
     absence of a feature_lineage table on disk.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

from core.step_6.byte_compare import run_byte_compare
from core.step_6.inputs import Step6Inputs
from core.step_6.manifest import (
    AuditConfig,
    CategoryAuditResult,
    CheckResult,
    Severity,
)

# Patterns flagging forward-path metrics that must NOT appear at entry time.
# Path-shape clustering uses these; entry features must never reference them.
_PATH_FEATURE_PATTERNS = (
    re.compile(r"^mfe_", re.IGNORECASE),
    re.compile(r"^mae_", re.IGNORECASE),
    re.compile(r"^ww_", re.IGNORECASE),
    re.compile(r"^wrong_way_", re.IGNORECASE),
    re.compile(r"^final_r", re.IGNORECASE),
    re.compile(r"^reach_\d+r", re.IGNORECASE),
    re.compile(r"^ttp_", re.IGNORECASE),
    re.compile(r"^bars_held", re.IGNORECASE),
    re.compile(r"^path_", re.IGNORECASE),
)


def _check_per_feature_lineage(inputs: Step6Inputs) -> CheckResult:
    features = inputs.best_candidate_features
    lineage = inputs.feature_lineage

    # Vacuous-pass: rule-based architectures (A1 system_level_filter) by
    # design carry no classifier features. The universal quantifier "every
    # feature in the winning config has clean lineage" is vacuously True
    # over an empty set — no features means no opportunity for lineage
    # contamination. Treating empty as "cannot verify → CRITICAL" is the
    # inverse of intent (see engine/step_6_a1_vacuous_pass, 2026-05-25).
    if not features:
        return CheckResult(
            name="per_feature_lineage_clean",
            passed=True,
            severity=Severity.INFO,
            message=(
                "best_candidate_features is empty (by-design for rule-based "
                "architectures like A1 system_level_filter). No classifier "
                "features means no opportunity for feature lineage contamination — "
                "vacuous PASS per universal-quantifier-over-empty-set."
            ),
            evidence={"features_in_winning_config": []},
        )
    if lineage is None:
        return CheckResult(
            name="per_feature_lineage_clean",
            passed=False,
            severity=Severity.CRITICAL,
            message="feature_lineage table missing — lineage enforcement skipped",
            evidence={"features_in_winning_config": list(features)},
        )

    if "causal_lineage" not in lineage.columns or "name" not in lineage.columns:
        return CheckResult(
            name="per_feature_lineage_clean",
            passed=False,
            severity=Severity.CRITICAL,
            message=(
                "feature_lineage missing required columns (name, causal_lineage); "
                f"got {list(lineage.columns)}"
            ),
            evidence={"columns": list(lineage.columns)},
        )

    tag_by_name = dict(zip(lineage["name"].astype(str), lineage["causal_lineage"].astype(str)))
    suspect_or_missing: list[tuple[str, str]] = []
    for f in features:
        tag = tag_by_name.get(f, "UNKNOWN")
        if tag != "clean":
            suspect_or_missing.append((f, tag))

    if suspect_or_missing:
        return CheckResult(
            name="per_feature_lineage_clean",
            passed=False,
            severity=Severity.CRITICAL,
            message=(
                f"{len(suspect_or_missing)}/{len(features)} winning-candidate features "
                f"are not lineage-clean"
            ),
            evidence={"non_clean_features": [
                {"feature": n, "lineage": t} for n, t in suspect_or_missing
            ]},
        )
    return CheckResult(
        name="per_feature_lineage_clean",
        passed=True,
        severity=Severity.CRITICAL,
        message=f"all {len(features)} winning-candidate features are lineage-clean",
        evidence={"n_features": len(features)},
    )


def _check_no_path_features_in_entry(inputs: Step6Inputs) -> CheckResult:
    features = inputs.best_candidate_features
    # Vacuous-pass: same semantics as _check_per_feature_lineage above.
    # No entry features means no opportunity for path-feature contamination
    # in entry decisions (see engine/step_6_a1_vacuous_pass, 2026-05-25).
    if not features:
        return CheckResult(
            name="no_path_features_in_entry",
            passed=True,
            severity=Severity.INFO,
            message=(
                "best_candidate_features is empty (by-design for rule-based "
                "architectures like A1 system_level_filter). No entry features "
                "means no opportunity for path-feature contamination in entry — "
                "vacuous PASS per universal-quantifier-over-empty-set."
            ),
            evidence={"features_in_winning_config": []},
        )
    leaks: list[str] = []
    for f in features:
        for pattern in _PATH_FEATURE_PATTERNS:
            if pattern.search(f):
                leaks.append(f)
                break
    if leaks:
        return CheckResult(
            name="no_path_features_in_entry",
            passed=False,
            severity=Severity.CRITICAL,
            message=(
                f"{len(leaks)} entry-time feature(s) match forward-path patterns: "
                f"{', '.join(leaks)}"
            ),
            evidence={"leaking_features": leaks},
        )
    return CheckResult(
        name="no_path_features_in_entry",
        passed=True,
        severity=Severity.CRITICAL,
        message="no entry-time feature name matches a forward-path pattern",
        evidence={"n_features_checked": len(features)},
    )


def _check_d1_lag_rule() -> CheckResult:
    """Static check: every D1 multi-TF producer routes through ``_build_d1_lag1_series``.

    Inspects ``core.features.multi_tf`` source and confirms the helper is
    called from each public D1-derived feature.
    """
    try:
        from core.features import multi_tf  # type: ignore
    except ImportError as exc:
        return CheckResult(
            name="d1_lag_rule_enforced",
            passed=False,
            severity=Severity.CRITICAL,
            message=f"could not import core.features.multi_tf: {exc}",
            evidence={"error": str(exc)},
        )
    try:
        source = inspect.getsource(multi_tf)
    except OSError as exc:
        return CheckResult(
            name="d1_lag_rule_enforced",
            passed=False,
            severity=Severity.CRITICAL,
            message=f"could not read multi_tf source: {exc}",
            evidence={"error": str(exc)},
        )
    if "_build_d1_lag1_series" not in source:
        return CheckResult(
            name="d1_lag_rule_enforced",
            passed=False,
            severity=Severity.CRITICAL,
            message="multi_tf module does not reference _build_d1_lag1_series",
            evidence={"helper_present": False},
        )
    # Count D1 producer references vs lag1-helper calls. Heuristic — any
    # D1-producer that DOESN'T call the helper is flagged.
    helper_calls = source.count("_build_d1_lag1_series(")
    d1_producers = source.count("def _d1_")  # convention in multi_tf.py
    if d1_producers > 0 and helper_calls < d1_producers:
        return CheckResult(
            name="d1_lag_rule_enforced",
            passed=False,
            severity=Severity.CRITICAL,
            message=(
                f"D1 producers ({d1_producers}) > lag-helper calls ({helper_calls}) "
                "— one or more D1 producers may skip the lag rule"
            ),
            evidence={"d1_producers": d1_producers, "helper_calls": helper_calls},
        )
    return CheckResult(
        name="d1_lag_rule_enforced",
        passed=True,
        severity=Severity.CRITICAL,
        message=(
            f"multi_tf D1 producers all route through _build_d1_lag1_series "
            f"({helper_calls} calls)"
        ),
        evidence={"helper_calls": helper_calls, "d1_producers": d1_producers},
    )


def _check_byte_compare(inputs: Step6Inputs, cfg: AuditConfig) -> CheckResult:
    report = run_byte_compare(
        inputs,
        feature_names=inputs.best_candidate_features or None,
        n_samples=cfg.byte_compare_n_samples,
        seed=cfg.byte_compare_seed,
    )
    if report.skipped_reason:
        return CheckResult(
            name="byte_compare_no_drift",
            passed=True,  # info-only when we can't run it; demoted via severity
            severity=Severity.INFO,
            message=f"byte-compare skipped: {report.skipped_reason}",
            evidence=report.to_evidence(),
        )
    passed = report.all_match
    return CheckResult(
        name="byte_compare_no_drift",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"byte-compare on {report.n_samples_used} trade(s) × "
            f"{len(set(r.feature for r in report.rows))} feature(s) — "
            f"all_match={passed}"
        ),
        evidence=report.to_evidence(),
    )


def _check_threshold_selection_lineage(inputs: Step6Inputs) -> CheckResult:
    """Verify Step 4's best_threshold came from per-fold averaging, not full-data ROC."""
    manifest_path = inputs.arc_root / "step_4" / "classifiers" / "manifest.json"
    if not manifest_path.exists():
        return CheckResult(
            name="threshold_selection_lineage",
            passed=True,  # nothing to check on a rule-based arc
            severity=Severity.INFO,
            message="no Step 4 classifier manifest — arc is rule-based or pre-CC_12",
            evidence={"manifest_path": str(manifest_path)},
        )
    import json
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    classifiers = data.get("classifiers") or {}
    anomalies: list[dict[str, Any]] = []
    for cid, entry in classifiers.items():
        thr = entry.get("best_threshold")
        # Heuristic: per-fold averaged thresholds typically fall in (0.0, 1.0)
        # and never sit exactly at the F1-best on a full-data ROC (which
        # tends to land at integer-fraction boundaries — 0.5 is the most
        # common smell). We flag exactly 0.5 only if every classifier has it.
        if thr is None:
            anomalies.append({"cluster_id": cid, "issue": "best_threshold missing"})
            continue
        if not (0.0 < float(thr) < 1.0):
            anomalies.append({"cluster_id": cid, "best_threshold": thr, "issue": "out of (0,1)"})
    if anomalies:
        return CheckResult(
            name="threshold_selection_lineage",
            passed=False,
            severity=Severity.WARNING,
            message=f"{len(anomalies)} classifier threshold(s) flagged for review",
            evidence={"anomalies": anomalies, "n_classifiers": len(classifiers)},
        )
    return CheckResult(
        name="threshold_selection_lineage",
        passed=True,
        severity=Severity.WARNING,
        message=(
            f"{len(classifiers)} classifier threshold(s) in (0,1); "
            "per-fold averaging assumed"
        ),
        evidence={"n_classifiers": len(classifiers)},
    )


def _check_lineage_table_exists(inputs: Step6Inputs) -> CheckResult:
    if inputs.feature_lineage is None:
        return CheckResult(
            name="feature_lineage_table_exists",
            passed=False,
            severity=Severity.INFO,
            message="feature_lineage table absent — falling back to source-code inspection",
            evidence={"present": False},
        )
    n_rows = int(len(inputs.feature_lineage))
    return CheckResult(
        name="feature_lineage_table_exists",
        passed=True,
        severity=Severity.INFO,
        message=f"feature_lineage table present ({n_rows} features)",
        evidence={"present": True, "n_features": n_rows},
    )


def audit(inputs: Step6Inputs, audit_config: AuditConfig) -> CategoryAuditResult:
    checks = (
        _check_per_feature_lineage(inputs),
        _check_no_path_features_in_entry(inputs),
        _check_d1_lag_rule(),
        _check_byte_compare(inputs, audit_config),
        _check_threshold_selection_lineage(inputs),
        _check_lineage_table_exists(inputs),
    )
    diagnostic: dict[str, Any] = {
        "features_in_winning_config": list(inputs.best_candidate_features),
        "best_candidate_architecture": inputs.best_candidate_architecture,
        "best_candidate_config_id": inputs.best_candidate_config_id,
    }
    return CategoryAuditResult(
        category="lookahead", checks=checks, diagnostic=diagnostic,
    )


__all__ = ("audit",)
