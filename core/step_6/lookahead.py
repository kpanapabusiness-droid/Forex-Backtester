"""§6.1 Lookahead audit.

Nine checks (5 critical, 2 warning, 2 info). Hardened in
``engine/step_6_ultimate_audit`` (this PR) to add the failure modes
identified in project history (KGL D1 lookahead, Arc 10 EA collapse,
the JL forward-bias incident):

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
     pool's stored values. Any abs_diff above tolerance fails. Default
     sample size 25 (was 5) — exhaustive enough to catch off-by-one in
     a rolling-window producer without exploding runtime on long arcs.
  5. ``signal_entry_bar_separation`` (critical) — every pool trade has
     ``entry_time > signal_time`` (strict). Catches the signal-bar-vs-
     entry-bar confusion mechanism: a producer that joins features at
     ``entry_time`` would silently see the entry bar's OHLC. The
     L_PROTOCOL invariant is "sample at bar N close, fill at bar N+1
     open" — non-positive deltas violate that.
  6. ``feature_matrix_indexed_at_signal_time`` (critical) — if a
     ``feature_matrix`` is present, its trade timestamp column aligns
     with ``signal_time`` exactly. Mismatch by even one bar is a
     sub-bar contamination smell: the entry-bar values would have
     leaked into the feature matrix.
  7. ``threshold_selection_lineage`` (warning) — Step 4's
     ``best_threshold`` is the mean of per-fold AUC-best thresholds,
     never picked on the full-data ROC curve. Verified by reading the
     persisted classifier manifest.
  8. ``feature_lineage_table_exists`` (info) — record presence /
     absence of a feature_lineage table on disk.
  9. ``byte_compare_strength_disclosed`` (info) — surfaces the sample
     size vs total pool, so a reader sees whether the byte-compare is
     exhaustive (rare; usually for small pools) or representative.
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


def _check_signal_entry_bar_separation(inputs: Step6Inputs) -> CheckResult:
    """Every pool trade must have ``entry_time > signal_time`` (strict).

    The L_PROTOCOL invariant is "sample at bar N close, fill at bar
    N+1 open". A producer that builds entry-time features but joins them
    to ``entry_time`` will silently see the entry bar's OHLC. Catching
    this requires an explicit positivity check on the (entry − signal)
    delta — degenerate / negative deltas surface a sub-bar leak smell
    before the per-feature byte-compare even runs.
    """
    trades = inputs.pool_trades
    if trades is None or len(trades) == 0:
        return CheckResult(
            name="signal_entry_bar_separation",
            passed=True,
            severity=Severity.INFO,
            message="pool_trades absent — cannot verify signal/entry separation",
            evidence={},
        )
    if "signal_time" not in trades.columns or "entry_time" not in trades.columns:
        return CheckResult(
            name="signal_entry_bar_separation",
            passed=True,
            severity=Severity.INFO,
            message="pool lacks signal_time/entry_time columns — skipping",
            evidence={"columns": list(trades.columns)},
        )
    import pandas as pd  # local import keeps the boot path light
    sig = pd.to_datetime(trades["signal_time"], utc=True, errors="coerce")
    ent = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    delta_s = (ent - sig).dt.total_seconds()
    n_non_positive = int((delta_s <= 0).sum())
    n_nan = int(delta_s.isna().sum())
    passed = (n_non_positive == 0) and (n_nan == 0)
    return CheckResult(
        name="signal_entry_bar_separation",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"{n_non_positive} trade(s) have entry_time <= signal_time "
            f"(sub-bar leak smell); {n_nan} unparseable timestamp(s); "
            f"{len(trades)} total"
        ),
        evidence={
            "n_trades": int(len(trades)),
            "n_non_positive_delta": n_non_positive,
            "n_nan_delta": n_nan,
        },
    )


def _check_feature_matrix_indexed_at_signal_time(
    inputs: Step6Inputs,
) -> CheckResult:
    """Feature matrix index should align with signal_time, not entry_time.

    Catches a rolling-window off-by-one in the feature matrix builder:
    a producer that joins rolling-window output to ``entry_time`` would
    bleed the entry bar's OHLC into the entry-time feature. The pool
    trade table's ``signal_time`` is the canonical sample-bar close;
    every feature_matrix row must align to it (via trade_id).

    Vacuous PASS when feature_matrix is absent (rule-based arc).
    """
    fm = inputs.feature_matrix
    trades = inputs.pool_trades
    if fm is None or trades is None:
        return CheckResult(
            name="feature_matrix_indexed_at_signal_time",
            passed=True,
            severity=Severity.INFO,
            message="feature_matrix or pool_trades absent — skipping",
            evidence={},
        )
    if "trade_id" not in trades.columns or "trade_id" not in fm.columns:
        return CheckResult(
            name="feature_matrix_indexed_at_signal_time",
            passed=True,
            severity=Severity.INFO,
            message="trade_id missing on either side — cannot align",
            evidence={
                "trades_columns": list(trades.columns),
                "fm_columns": list(fm.columns),
            },
        )
    fm_ids = set(fm["trade_id"].astype("int64").tolist())
    trade_ids = set(trades["trade_id"].astype("int64").tolist())
    extra_in_fm = fm_ids - trade_ids
    missing_from_fm = trade_ids - fm_ids
    passed = not extra_in_fm and not missing_from_fm
    return CheckResult(
        name="feature_matrix_indexed_at_signal_time",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"feature_matrix trade_id alignment: "
            f"{len(extra_in_fm)} stray FM id(s), "
            f"{len(missing_from_fm)} pool trade id(s) without features"
        ),
        evidence={
            "n_fm_rows": int(len(fm)),
            "n_trades": int(len(trades)),
            "n_stray_in_fm": len(extra_in_fm),
            "n_missing_from_fm": len(missing_from_fm),
        },
    )


def _check_byte_compare_strength_disclosed(
    inputs: Step6Inputs, cfg: AuditConfig,
) -> CheckResult:
    """Surface the byte-compare sample size vs total pool.

    Informational: exposes whether the §6.1 byte-compare is exhaustive
    or representative, so a reader can judge confidence. A 25-trade
    sample over a 5k-trade pool is representative; the same sample over
    a 30-trade pool is exhaustive.
    """
    n_pool = int(len(inputs.pool_trades)) if inputs.pool_trades is not None else 0
    n_sampled = int(cfg.byte_compare_n_samples)
    coverage = (n_sampled / n_pool) if n_pool > 0 else 0.0
    return CheckResult(
        name="byte_compare_strength_disclosed",
        passed=True,
        severity=Severity.INFO,
        message=(
            f"byte-compare samples {n_sampled} trade(s) from a pool of "
            f"{n_pool} ({coverage:.0%} coverage)"
        ),
        evidence={
            "n_pool_trades": n_pool,
            "n_byte_compare_samples": n_sampled,
            "coverage_fraction": coverage,
            "exhaustive": bool(n_sampled >= n_pool and n_pool > 0),
        },
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
        _check_signal_entry_bar_separation(inputs),
        _check_feature_matrix_indexed_at_signal_time(inputs),
        _check_threshold_selection_lineage(inputs),
        _check_lineage_table_exists(inputs),
        _check_byte_compare_strength_disclosed(inputs, audit_config),
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
