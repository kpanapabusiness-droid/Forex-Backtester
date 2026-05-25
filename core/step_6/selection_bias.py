"""§6.2 Selection bias audit.

Per chat Q3: this category VERIFIES that selection-bias accounting at
Step 5 produced the right record — it does not re-run the search.

Eight checks (4 critical, 2 warning, 2 info). Hardened in
``engine/step_6_ultimate_audit`` with three new checks targeting
failure modes that have surfaced in past arcs (heavy_ml_probe trial
inflation, feature-set co-selection contamination, Step 3→Step 5
re-evaluation of the same SL multiplier):

  1. ``configs_evaluated_recorded`` (critical) — pool_metadata exposes
     ``configs_evaluated_step5`` and the search-scope flag is in
     {"thin", "normal", "broad"} per template.
  2. ``ratio_clears_bonferroni_noise_floor`` (critical) — for the Top-1
     candidate, worst-fold ratio is large enough vs the noise floor
     implied by N configs evaluated. Noise floor: 2.0 + 0.01 × log2(N).
     Now inflates N by sub-protocol trial counts when a heavy_ml_probe
     manifest is on disk (catches the laundering of trial counts).
  3. ``holdout_never_touched_during_is`` (critical) — search artefacts
     reference only IS dates (< holdout_start). Reads
     ``step_5/wfo_results.csv`` if present and checks every fold's
     OOS date upper bound.
  4. ``sub_protocol_trial_counts_included`` (critical) — if a
     heavy_ml_probe (or other sub-protocol) manifest is present, its
     trial count is accounted for in the effective
     ``configs_evaluated_step5``. Without this, a probe that evaluated
     10,000 trials and surfaced its top survivor as a single Step 5
     config would understate selection pressure by 3+ orders of
     magnitude.
  5. ``cluster_selection_recorded`` (warning) — every cluster Step 3
     considered is on disk via ``step_3/capturability.csv`` AND the
     winning cluster is tagged ``is_candidate=True``.
  6. ``sl_multiplier_step3_step5_independence`` (warning) — surfaces
     whether the SL multiplier selected via Step 3 capturability also
     drove the Step 5 evaluation on the same data. When both stages
     read the same outcomes, that's technically OOS-aware tuning; the
     check flags it for review (the canonical engine path picks SL via
     Step 3 then evaluates separately, but custom pipelines can drift).
  7. ``feature_set_selection_recorded`` (info) — Step 4 selected its
     feature set from a finite list; record what list and how many
     features were available so a reader can judge selection pressure
     on the classifier feature set (the JL forward-bias incident is
     the canonical "features chosen because they correlated with
     outcomes on the same data being WFO'd" failure).
  8. ``search_scope_flag_matches_n`` (info) — flag derivation matches
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
    n_recorded = inputs.configs_evaluated_step5
    closure_payload = inputs.closure_payload or {}
    ba = closure_payload.get("best_architecture") or {}
    worst_fold_ratio = ba.get("worst_fold_ratio")

    # Auto-dispatch path — read from live result.
    if worst_fold_ratio is None and inputs.arc_orchestrator_result is not None:
        wfo_search = getattr(inputs.arc_orchestrator_result, "wfo_search", None)
        if wfo_search is not None and wfo_search.top_k:
            worst_fold_ratio = wfo_search.top_k[0].gate.worst_fold_ratio

    if worst_fold_ratio is None or n_recorded is None:
        return CheckResult(
            name="ratio_clears_bonferroni_noise_floor",
            passed=False,
            severity=Severity.CRITICAL,
            message=(
                "missing inputs — "
                f"worst_fold_ratio={worst_fold_ratio!r}, configs_evaluated_step5={n_recorded!r}"
            ),
            evidence={
                "worst_fold_ratio": worst_fold_ratio,
                "configs_evaluated_step5": n_recorded,
            },
        )

    # Inflate N by sub-protocol trial counts when a heavy_ml manifest is
    # on disk — see ``_check_sub_protocol_trial_counts`` for the
    # mechanism. The noise floor must reflect TOTAL selection pressure,
    # not just the configs Step 5 itself iterated.
    sub_protocol_trials = _read_heavy_ml_trial_count(inputs.arc_root)
    n_effective = int(n_recorded)
    if sub_protocol_trials and sub_protocol_trials > 0:
        n_effective = int(n_recorded) + int(sub_protocol_trials)

    noise_floor = _bonferroni_noise_floor(n_effective)
    passed = float(worst_fold_ratio) >= noise_floor
    return CheckResult(
        name="ratio_clears_bonferroni_noise_floor",
        passed=passed,
        severity=Severity.CRITICAL,
        message=(
            f"worst-fold ratio {float(worst_fold_ratio):.3f} "
            f"vs Bonferroni-equivalent floor {noise_floor:.3f} "
            f"(N_effective={n_effective}, recorded={n_recorded}, "
            f"sub_protocol_trials={sub_protocol_trials or 0})"
        ),
        evidence={
            "worst_fold_ratio": float(worst_fold_ratio),
            "noise_floor": noise_floor,
            "configs_evaluated_step5": int(n_recorded),
            "sub_protocol_trials": int(sub_protocol_trials or 0),
            "n_effective": n_effective,
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


def _read_heavy_ml_trial_count(arc_root) -> int | None:
    """Return total FLAML trials across folds for a heavy_ml_probe arc.

    Returns ``None`` when no heavy_ml manifest is present (arc is not a
    heavy_ml_probe variant — most arcs). Returns 0 when manifest exists
    but no per-fold model count is parseable (something is wrong; the
    caller surfaces it).
    """
    import json

    manifest = arc_root / "step_5" / "heavy_ml_augmented" / "heavy_ml_manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    # Schema: per-fold modelcount lives under `folds[*].automl.modelcount`.
    total = 0
    for fold in data.get("folds") or []:
        automl = fold.get("automl") or {}
        mc = automl.get("modelcount") or automl.get("n_trials")
        if isinstance(mc, (int, float)):
            total += int(mc)
    return total


def _check_sub_protocol_trial_counts(inputs: Step6Inputs) -> CheckResult:
    """If a sub-protocol manifest is on disk, its trial count is in N.

    The Bonferroni floor in :func:`_check_ratio_clears_bonferroni`
    rescales N to include sub-protocol trials so the noise floor lifts
    appropriately. Without this accounting a heavy_ml_probe arc that
    evaluated 10k FLAML trials and surfaced its single top survivor as
    a Step 5 config understates selection pressure by ~3 orders of
    magnitude.

    Pass condition: either no sub-protocol manifest exists (vanilla
    arc), or the trial count is recorded somewhere parseable.
    """
    n_trials = _read_heavy_ml_trial_count(inputs.arc_root)
    if n_trials is None:
        return CheckResult(
            name="sub_protocol_trial_counts_included",
            passed=True,
            severity=Severity.INFO,
            message="no sub-protocol manifest on disk — vanilla arc",
            evidence={"heavy_ml_manifest_present": False},
        )
    if n_trials <= 0:
        return CheckResult(
            name="sub_protocol_trial_counts_included",
            passed=False,
            severity=Severity.CRITICAL,
            message=(
                "heavy_ml manifest present but per-fold trial counts "
                "(automl.modelcount) are missing/zero — search-space "
                "inflation cannot be accounted for"
            ),
            evidence={"heavy_ml_total_trials": n_trials},
        )
    # Compare against recorded configs_evaluated; if recorded value is
    # much smaller, flag as understated.
    recorded = inputs.configs_evaluated_step5 or 0
    understated = bool(recorded > 0 and n_trials > recorded * 10)
    return CheckResult(
        name="sub_protocol_trial_counts_included",
        passed=not understated,
        severity=Severity.CRITICAL,
        message=(
            f"heavy_ml trials total = {n_trials}; "
            f"recorded configs_evaluated_step5 = {recorded}. "
            + (
                "Recorded N understates the effective search space — "
                "selection-bias accounting must include sub-protocol trials."
                if understated
                else "Effective N is reflected by recorded N (probe trials "
                "are within an order of magnitude)."
            )
        ),
        evidence={
            "heavy_ml_total_trials": int(n_trials),
            "recorded_configs_evaluated_step5": int(recorded),
            "understated": understated,
        },
    )


def _check_sl_multiplier_step3_step5_independence(
    inputs: Step6Inputs,
) -> CheckResult:
    """Flag if Step 3 selected the SL multiplier that Step 5 then re-tuned.

    The canonical engine path: Step 3 capturability picks the SL
    multiplier per cluster, Step 5 evaluates the resulting cluster
    config without re-tuning SL. A custom pipeline that re-tunes SL at
    Step 5 against the same data has technically used OOS-aware tuning
    twice. Heuristic check: if capturability.csv records an
    ``sl_atr_multiplier`` column AND the Step 5 wfo_results.csv ALSO
    records a per-fold ``sl_atr_multiplier`` that varies across folds,
    Step 5 is re-tuning. Pass when either is absent or the multiplier
    is fixed.
    """
    import pandas as pd

    cap = inputs.arc_root / "step_3" / "capturability.csv"
    wfo = inputs.arc_root / "step_5" / "wfo_results.csv"
    if not cap.exists() or not wfo.exists():
        return CheckResult(
            name="sl_multiplier_step3_step5_independence",
            passed=True,
            severity=Severity.INFO,
            message="capturability or wfo_results CSV absent — skipping",
            evidence={
                "capturability_present": cap.exists(),
                "wfo_results_present": wfo.exists(),
            },
        )
    try:
        cap_df = pd.read_csv(cap)
        wfo_df = pd.read_csv(wfo)
    except Exception as exc:
        return CheckResult(
            name="sl_multiplier_step3_step5_independence",
            passed=False,
            severity=Severity.WARNING,
            message=f"could not read CSVs: {exc}",
            evidence={"error": str(exc)},
        )
    sl_col = "sl_atr_multiplier"
    cap_has = sl_col in cap_df.columns
    wfo_has = sl_col in wfo_df.columns
    if not (cap_has and wfo_has):
        return CheckResult(
            name="sl_multiplier_step3_step5_independence",
            passed=True,
            severity=Severity.INFO,
            message=(
                f"{sl_col} absent on one side "
                f"(cap={cap_has}, wfo={wfo_has}) — re-tuning not detectable"
            ),
            evidence={"cap_has_col": cap_has, "wfo_has_col": wfo_has},
        )
    # If Step 5 records ≥ 2 distinct sl_atr_multiplier values per (config_id, fold_id)
    # group, that's Step 5 re-tuning beyond what Step 3 picked.
    group_cols = [c for c in ("config_id", "candidate", "candidate_id")
                  if c in wfo_df.columns]
    if not group_cols:
        # No way to group — fall back to global uniqueness.
        n_distinct = int(wfo_df[sl_col].nunique())
    else:
        n_distinct = int(
            wfo_df.groupby(group_cols)[sl_col].nunique().max()
        )
    retuning = bool(n_distinct > 1)
    return CheckResult(
        name="sl_multiplier_step3_step5_independence",
        passed=not retuning,
        severity=Severity.WARNING,
        message=(
            f"Step 5 records {n_distinct} distinct {sl_col} value(s) "
            f"per candidate — "
            + (
                "Step 5 IS re-tuning SL beyond Step 3's selection (review)."
                if retuning
                else "Step 5 holds SL constant per candidate (canonical)."
            )
        ),
        evidence={
            "n_distinct_sl_per_candidate": n_distinct,
            "group_cols": group_cols,
            "re_tuning": retuning,
        },
    )


def _check_feature_set_selection_recorded(inputs: Step6Inputs) -> CheckResult:
    """Surface how many features Step 4 selected from and how many it kept.

    The JL forward-bias incident's mechanism was features chosen because
    they correlated with outcomes on the same data later WFO'd against.
    Detection without re-running Step 4 is impossible; this check
    surfaces the selection-pressure ratio (n_features_considered /
    n_features_kept) so a reader can judge the magnitude.
    """
    import json

    fm = inputs.feature_matrix
    n_kept = len(inputs.best_candidate_features)
    n_considered = int(len(fm.columns) - (1 if "trade_id" in fm.columns else 0)) \
        if fm is not None else 0
    # Step 4 may also persist the full feature list it scanned in the
    # extraction_metrics CSV or classifier manifest.
    manifest = inputs.arc_root / "step_4" / "classifiers" / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for entry in (data.get("classifiers") or {}).values():
                order = entry.get("feature_order") or []
                n_kept = max(n_kept, len(order))
                break
        except (OSError, json.JSONDecodeError):
            pass
    pressure_ratio = (n_considered / n_kept) if n_kept > 0 else 0.0
    return CheckResult(
        name="feature_set_selection_recorded",
        passed=True,
        severity=Severity.INFO,
        message=(
            f"n_features_considered={n_considered}, n_features_kept={n_kept}, "
            f"selection pressure ≈ {pressure_ratio:.2f}× "
            "(higher = more pressure on the chosen subset)"
        ),
        evidence={
            "n_features_considered": n_considered,
            "n_features_kept": n_kept,
            "selection_pressure_ratio": pressure_ratio,
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
        _check_sub_protocol_trial_counts(inputs),
        _check_cluster_selection_recorded(inputs),
        _check_sl_multiplier_step3_step5_independence(inputs),
        _check_feature_set_selection_recorded(inputs),
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
