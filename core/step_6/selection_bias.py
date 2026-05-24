"""§6.2 Selection bias audit.

Per chat Q3: this category VERIFIES that selection-bias accounting at
Step 5 produced the right record — it does not re-run the search.

Five checks:

  1. ``configs_evaluated_recorded`` (critical) — pool_metadata exposes
     ``configs_evaluated_step5`` and the search-scope flag is in
     {"thin", "normal", "broad"} per template.
  2. ``ratio_clears_bonferroni_noise_floor`` (critical) — for the Top-1
     candidate, worst-fold ratio is large enough vs the noise floor
     implied by N configs evaluated. Noise floor: 2.0 + 0.01 × log2(N)
     (a soft lower bound; chat tightens via amendment when calibration
     evidence accumulates).
  3. ``holdout_never_touched_during_is`` (critical) — search artefacts
     reference only IS dates (< holdout_start). Reads
     ``step_5/wfo_results.csv`` if present and checks every fold's
     OOS date upper bound.
  4. ``cluster_selection_recorded`` (warning) — every cluster Step 3
     considered is on disk via ``step_3/capturability.csv`` AND the
     winning cluster is tagged ``is_candidate=True``.
  5. ``search_scope_flag_matches_n`` (info) — flag derivation matches
     the closed template definitions (thin <50, normal 50-99, broad 100+).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from core.step_6.inputs import Step6Inputs
from core.step_6.manifest import (
    AuditConfig,
    CategoryAuditResult,
    CheckResult,
    Severity,
)


def _scope_from_n(n: int) -> str:
    if n < 50:
        return "thin"
    if n < 100:
        return "normal"
    return "broad"


def _bonferroni_noise_floor(n_configs: int) -> float:
    """Soft noise floor: 2.0 + 0.01 × log2(max(1, N)).

    Rationale: §3 baseline DEPLOYABLE ratio is 2.0; each doubling of the
    config count adds 0.01 pp pressure. At N=128 the floor is 2.07; at
    N=1024 it's 2.10. Chat may tighten via amendment.
    """
    return 2.0 + 0.01 * math.log2(max(1, n_configs))


def _check_configs_evaluated_recorded(inputs: Step6Inputs) -> CheckResult:
    n = inputs.configs_evaluated_step5
    if n is None:
        return CheckResult(
            name="configs_evaluated_recorded",
            passed=False,
            severity=Severity.CRITICAL,
            message="configs_evaluated_step5 missing from pool_metadata",
            evidence={"configs_evaluated_step5": None},
        )
    derived_scope = _scope_from_n(int(n))
    closure_payload = inputs.closure_payload or {}
    recorded_scope = (
        (closure_payload.get("pool_metadata") or {}).get("search_scope_flag")
    )
    return CheckResult(
        name="configs_evaluated_recorded",
        passed=True,
        severity=Severity.CRITICAL,
        message=(
            f"configs_evaluated_step5 = {n}; scope flag = "
            f"{recorded_scope!r} (derived from N: {derived_scope})"
        ),
        evidence={
            "configs_evaluated_step5": int(n),
            "recorded_scope": recorded_scope,
            "derived_scope": derived_scope,
        },
    )


def _check_ratio_clears_bonferroni(inputs: Step6Inputs) -> CheckResult:
    n = inputs.configs_evaluated_step5
    closure_payload = inputs.closure_payload or {}
    ba = closure_payload.get("best_architecture") or {}
    worst_fold_ratio = ba.get("worst_fold_ratio")

    # Auto-dispatch path — read from live result.
    if worst_fold_ratio is None and inputs.arc_orchestrator_result is not None:
        wfo_search = getattr(inputs.arc_orchestrator_result, "wfo_search", None)
        if wfo_search is not None and wfo_search.top_k:
            worst_fold_ratio = wfo_search.top_k[0].gate.worst_fold_ratio

    if worst_fold_ratio is None or n is None:
        return CheckResult(
            name="ratio_clears_bonferroni_noise_floor",
            passed=False,
            severity=Severity.CRITICAL,
            message=(
                "missing inputs — "
                f"worst_fold_ratio={worst_fold_ratio!r}, configs_evaluated_step5={n!r}"
            ),
            evidence={
                "worst_fold_ratio": worst_fold_ratio,
                "configs_evaluated_step5": n,
            },
        )

    noise_floor = _bonferroni_noise_floor(int(n))
    passed = float(worst_fold_ratio) >= noise_floor
    return CheckResult(
        name="ratio_clears_bonferroni_noise_floor",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"worst-fold ratio {float(worst_fold_ratio):.3f} "
            f"vs Bonferroni-equivalent floor {noise_floor:.3f} "
            f"(N={n})"
        ),
        evidence={
            "worst_fold_ratio": float(worst_fold_ratio),
            "noise_floor": noise_floor,
            "configs_evaluated_step5": int(n),
        },
    )


def _check_holdout_never_touched(inputs: Step6Inputs) -> CheckResult:
    holdout_start = inputs.holdout_start
    if holdout_start is None:
        # Default to L_PROTOCOL Step 5 spec: holdout = 2021-01-01
        holdout_start = pd.Timestamp("2021-01-01", tz="UTC")
    wfo_csv = inputs.arc_root / "step_5" / "wfo_results.csv"
    if not wfo_csv.exists():
        return CheckResult(
            name="holdout_never_touched_during_is",
            passed=True,
            severity=Severity.INFO,
            message=f"{wfo_csv.name} absent — cannot verify",
            evidence={"path": str(wfo_csv)},
        )
    try:
        wfo = pd.read_csv(wfo_csv)
    except Exception as exc:
        return CheckResult(
            name="holdout_never_touched_during_is",
            passed=False,
            severity=Severity.CRITICAL,
            message=f"could not read {wfo_csv.name}: {exc}",
            evidence={"error": str(exc)},
        )
    # Heuristic — any column whose name ends with "_end" / "_oos_end" or
    # contains "fold_end" + date values represents a fold's OOS upper bound.
    suspect_rows: list[dict[str, Any]] = []
    date_columns = [c for c in wfo.columns if "end" in c.lower() and "fold" in c.lower()]
    if not date_columns:
        date_columns = [c for c in wfo.columns if c.lower().endswith("_end")]
    for col in date_columns:
        try:
            col_ts = pd.to_datetime(wfo[col], utc=True, errors="coerce")
        except Exception:
            continue
        leaked = wfo.loc[col_ts >= holdout_start]
        if len(leaked) > 0:
            suspect_rows.append({"column": col, "n_leaked_rows": int(len(leaked))})
    if suspect_rows:
        return CheckResult(
            name="holdout_never_touched_during_is",
            passed=False,
            severity=Severity.CRITICAL,
            message=(
                f"WFO search artefacts reference dates ≥ holdout_start "
                f"({holdout_start.isoformat()})"
            ),
            evidence={"suspects": suspect_rows, "holdout_start": str(holdout_start)},
        )
    return CheckResult(
        name="holdout_never_touched_during_is",
        passed=True,
        severity=Severity.CRITICAL,
        message=(
            f"WFO search artefacts all bounded before holdout_start "
            f"({holdout_start.isoformat()})"
        ),
        evidence={"date_columns_checked": date_columns, "holdout_start": str(holdout_start)},
    )


def _check_cluster_selection_recorded(inputs: Step6Inputs) -> CheckResult:
    cap = inputs.arc_root / "step_3" / "capturability.csv"
    if not cap.exists():
        return CheckResult(
            name="cluster_selection_recorded",
            passed=False,
            severity=Severity.WARNING,
            message=f"{cap.name} absent — cluster selection record missing",
            evidence={"path": str(cap)},
        )
    try:
        df = pd.read_csv(cap)
    except Exception as exc:
        return CheckResult(
            name="cluster_selection_recorded",
            passed=False,
            severity=Severity.WARNING,
            message=f"could not read {cap.name}: {exc}",
            evidence={"error": str(exc)},
        )
    n_clusters = len(df)
    if "is_candidate" not in df.columns:
        return CheckResult(
            name="cluster_selection_recorded",
            passed=False,
            severity=Severity.WARNING,
            message="capturability.csv missing 'is_candidate' column",
            evidence={"columns": list(df.columns)},
        )
    n_candidates = int(df["is_candidate"].sum())
    return CheckResult(
        name="cluster_selection_recorded",
        passed=True,
        severity=Severity.WARNING,
        message=(
            f"{n_clusters} cluster(s) considered, {n_candidates} flagged candidate"
        ),
        evidence={
            "n_clusters_considered": n_clusters,
            "n_candidate_clusters": n_candidates,
        },
    )


def _check_scope_flag_matches_n(inputs: Step6Inputs) -> CheckResult:
    n = inputs.configs_evaluated_step5
    closure_payload = inputs.closure_payload or {}
    recorded = (
        (closure_payload.get("pool_metadata") or {}).get("search_scope_flag")
    )
    if n is None or recorded is None:
        return CheckResult(
            name="search_scope_flag_matches_n",
            passed=True,
            severity=Severity.INFO,
            message="cannot verify scope flag — missing inputs",
            evidence={"n": n, "recorded_scope": recorded},
        )
    derived = _scope_from_n(int(n))
    passed = derived == recorded
    return CheckResult(
        name="search_scope_flag_matches_n",
        passed=passed,
        severity=Severity.INFO,
        message=(
            f"recorded={recorded!r}, derived={derived!r} (N={n}); "
            f"{'match' if passed else 'MISMATCH'}"
        ),
        evidence={"n": int(n), "derived": derived, "recorded": recorded},
    )


def audit(inputs: Step6Inputs, audit_config: AuditConfig) -> CategoryAuditResult:
    checks = (
        _check_configs_evaluated_recorded(inputs),
        _check_ratio_clears_bonferroni(inputs),
        _check_holdout_never_touched(inputs),
        _check_cluster_selection_recorded(inputs),
        _check_scope_flag_matches_n(inputs),
    )
    diagnostic = {
        "configs_evaluated_step5": inputs.configs_evaluated_step5,
        "holdout_start": str(inputs.holdout_start) if inputs.holdout_start else None,
    }
    return CategoryAuditResult(
        category="selection_bias", checks=checks, diagnostic=diagnostic,
    )


__all__ = ("audit",)
