"""Auto-dispatch helpers used by :class:`core.arc.arc_orchestrator.ArcOrchestrator`.

Per chat Q1: post-gate. Amended gate runs first with the default
``causal_audit_clean=True``. If the Top-1 candidate clears PASS-tier,
Step 6 dispatches; if it FAILs, the orchestrator re-classifies the
Top-1 with ``causal_audit_clean=False`` to downgrade.

Per chat Q2: Top-1 only. Top-2/Top-3 feature-set divergence surfaces
as a ``feature_set_divergence`` warning on the Step 6 manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.step_6.artefacts import write_step_6_artefacts
from core.step_6.io import from_arc_orchestrator_result
from core.step_6.manifest import (
    AuditConfig,
    CategoryAuditResult,
    CheckResult,
    Severity,
    Step6Manifest,
    Step6Result,
    TriggerSource,
)
from core.step_6.orchestrator import run_step_6


@dataclass(frozen=True)
class DispatchOutcome:
    """Result of an auto-dispatch invocation.

    ``step_6_result`` / ``step_6_manifest`` are None when no PASS-tier
    candidate cleared §3 constraints #1-9 (no dispatch warranted).

    ``downgrade_top_1`` is True iff Step 6 surfaced at least one
    critical failure on a PASS-tier candidate; the orchestrator must
    re-call :func:`classify_amended_fold_stats` with
    ``causal_audit_clean=False`` for the Top-1 candidate.

    ``divergence_warning`` is True iff Top-2/Top-3 use a materially
    different feature set from Top-1 (per chat Q2).
    """

    dispatched: bool
    step_6_result: Step6Result | None
    step_6_manifest: Step6Manifest | None
    downgrade_top_1: bool
    divergence_warning: bool
    out_dir: Path | None


def _select_top1_amended(amended_wfo: Any) -> Any | None:
    """Return the Top-1 :class:`CandidateAmendedResult` by verdict rank.

    Mirrors arc_orchestrator._amended_verdict_rank: PASS_DEPLOYABLE > PASS_VIABLE > FAIL.
    """
    if amended_wfo is None or not amended_wfo.amended_results:
        return None
    ranked = sorted(
        amended_wfo.amended_results,
        key=lambda r: _verdict_rank(r.amended_gate.verdict),
        reverse=True,
    )
    return ranked[0]


def _verdict_rank(verdict: Any) -> int:
    name = verdict.name if hasattr(verdict, "name") else str(verdict)
    if name == "PASS_DEPLOYABLE":
        return 2
    if name == "PASS_VIABLE":
        return 1
    return 0


def _is_pass_tier(verdict: Any) -> bool:
    return _verdict_rank(verdict) >= 1


def _feature_sets_diverge(
    top1: Any, others: tuple[Any, ...], strategy_results: Mapping[str, Any] | None,
) -> bool:
    """Heuristic: if Top-2/Top-3 carry a different architecture name from
    Top-1, OR their config_id prefix differs, flag divergence.

    Strategy: arc-orchestrator config_ids look like ``"A2::cfg_xxx"`` — split
    on ``"::"`` and compare prefixes. When the prefixes differ we likely
    have different feature sets (A1 vs A2 vs A6).
    """
    if not others:
        return False
    top1_prefix = top1.config_id.split("::", 1)[0]
    for other in others:
        other_prefix = other.config_id.split("::", 1)[0]
        if other_prefix != top1_prefix:
            return True
    return False


def maybe_dispatch_step_6(
    *,
    arc_orchestrator_result: Any,
    amended_wfo: Any,
    arc_root: Path,
    audit_config: AuditConfig | None = None,
    holdout_start: pd.Timestamp | None = None,
    panels: Mapping[str, Any] | None = None,
    feature_matrix: pd.DataFrame | None = None,
    feature_lineage: pd.DataFrame | None = None,
    signal_module_name: str | None = None,
    primary_tf: str | None = None,
    pair_set: tuple[str, ...] = (),
    strategy_results: Mapping[str, Mapping[int, Any]] | None = None,
    holdout_results: tuple = (),
    holdout_fold_id: int | None = None,
    r_base_pct: float | None = None,
    panel_boundary_convention: str | None = None,
    extras: Mapping[str, Any] | None = None,
) -> DispatchOutcome:
    """If the Top-1 amended candidate is PASS-tier, run Step 6 and return
    the outcome bundle. Otherwise return a no-op outcome.
    """
    top1 = _select_top1_amended(amended_wfo)
    if top1 is None or not _is_pass_tier(top1.amended_gate.verdict):
        return DispatchOutcome(
            dispatched=False, step_6_result=None, step_6_manifest=None,
            downgrade_top_1=False, divergence_warning=False, out_dir=None,
        )

    cfg = audit_config or AuditConfig()

    # Per Amendment 4 + §6.3 spread P&L wiring: extract Top-1's
    # closed-trade ledger + fold assignments from the orchestrator's
    # strategy-results side-channel so the diagnostic runs end-to-end.
    # Returns (None, None, holdout_fold_id) when reconstruction fails;
    # the diagnostic then skips gracefully (info-severity).
    from core.step_6.ledger_extraction import extract_top_1_ledger_bundle
    top_1_trade_ledger, top_1_fold_assignments, _hfid = (
        extract_top_1_ledger_bundle(
            top_1_config_id=top1.config_id,
            strategy_results=strategy_results,
            holdout_results=holdout_results,
            holdout_fold_id=holdout_fold_id,
        )
    )

    # Build Step6Inputs from the live result. Pass through any extras the
    # auto-dispatch path can supply.
    inputs = from_arc_orchestrator_result(
        arc_orchestrator_result, arc_root=arc_root,
        feature_matrix=feature_matrix,
        feature_lineage=feature_lineage,
        panels=panels,
        signal_module_name=signal_module_name,
        primary_tf=primary_tf,
        pair_set=pair_set,
        holdout_start=holdout_start,
        top_1_trade_ledger=top_1_trade_ledger,
        top_1_fold_assignments=top_1_fold_assignments,
        holdout_fold_id=holdout_fold_id,
        r_base_pct=r_base_pct,
        panel_boundary_convention=panel_boundary_convention,
        extras=extras,
    )

    result = run_step_6(inputs, trigger=TriggerSource.AUTO_PASS, audit_config=cfg)

    # Top-K feature-set divergence per chat Q2. Surface as a synthetic check
    # appended to the first category so it shows in the report.
    others = tuple(
        r for r in amended_wfo.amended_results
        if r is not top1 and _is_pass_tier(r.amended_gate.verdict)
    )
    divergence = _feature_sets_diverge(top1, others, None)
    if divergence and result.categories:
        # Append divergence warning to the lookahead category (first in list).
        first_cat = result.categories[0]
        divergence_check = CheckResult(
            name="top_k_feature_set_divergence",
            passed=False,
            severity=Severity.WARNING,
            message=(
                f"Top-1 ({top1.config_id}) has feature set distinct from "
                f"Top-{len(others)+1} candidate(s); chat may want to audit them too"
            ),
            evidence={
                "top1_config_id": top1.config_id,
                "other_config_ids": [r.config_id for r in others],
            },
        )
        first_cat_with_warn = CategoryAuditResult(
            category=first_cat.category,
            checks=first_cat.checks + (divergence_check,),
            diagnostic=first_cat.diagnostic,
        )
        result = Step6Result(
            arc_name=result.arc_name,
            trigger=result.trigger,
            categories=(first_cat_with_warn,) + result.categories[1:],
            audit_config=result.audit_config,
        )

    out_dir = arc_root / "step_6"
    manifest = write_step_6_artefacts(result, out_dir)

    return DispatchOutcome(
        dispatched=True,
        step_6_result=result,
        step_6_manifest=manifest,
        downgrade_top_1=not result.overall_passed,
        divergence_warning=divergence,
        out_dir=out_dir,
    )


def replace_top_1_with_step6_fail(amended_wfo: Any) -> Any:
    """Re-classify the Top-1 amended candidate's gate with
    ``causal_audit_clean=False`` per Amendment 4 §"Verdict downgrade".

    Returns a new :class:`AmendedWfoSearchResult` with the Top-1's
    ``amended_gate`` replaced; other candidates untouched.

    The re-classify path inside ``classify_amended_fold_stats`` needs the
    full input set (folds, chained DD, per-day DD, holdout safe/hard).
    We replay those from the CandidateAmendedResult bundle so no holdout
    sim is repeated. Per-day max-DD DataFrame is re-loaded from the
    parquet path the original eval recorded.
    """
    from core.wfo.amended_gates import (  # local import keeps boot cheap
        classify_amended_fold_stats,
    )

    if amended_wfo is None or not amended_wfo.amended_results:
        return amended_wfo

    top1 = _select_top1_amended(amended_wfo)
    if top1 is None:
        return amended_wfo

    # Reload per-day DF from parquet path the first pass recorded
    per_day_df = None
    if top1.per_day_max_dd_artefact_path is not None and Path(top1.per_day_max_dd_artefact_path).exists():
        try:
            per_day_df = pd.read_parquet(top1.per_day_max_dd_artefact_path)
        except Exception:
            per_day_df = None

    # We don't have direct access to the original holdout FoldStats objects
    # (they live in the orchestrator's holdout_results tuple). Reconstruct
    # minimal FoldStats from the AmendedGateResult fields when present.
    from core.wfo.gates import FoldStats
    safe = None
    if top1.amended_gate.holdout_roi_at_r_safe_pct is not None:
        safe = FoldStats(
            fold_id=-1, n_trades=0,
            roi_pct=top1.amended_gate.holdout_roi_at_r_safe_pct,
            max_dd_pct=top1.amended_gate.holdout_dd_at_r_safe_pct or 0.0,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=0.0,
        )
    hard = None
    if top1.amended_gate.holdout_roi_at_r_hard_pct is not None:
        hard = FoldStats(
            fold_id=-1, n_trades=0,
            roi_pct=top1.amended_gate.holdout_roi_at_r_hard_pct,
            max_dd_pct=top1.amended_gate.holdout_dd_at_r_hard_pct or 0.0,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=0.0,
        )

    # We need the original fold stats — those live on the underlying
    # CandidateSearchResult, accessible via amended_wfo.base.top_k.
    fold_stats = ()
    for cand in amended_wfo.base.top_k:
        if cand.config_id == top1.config_id:
            fold_stats = cand.fold_stats
            break

    new_gate = classify_amended_fold_stats(
        folds=fold_stats,
        chained_max_dd_base_pct=top1.chained_max_dd_base_pct,
        per_day_max_dd_df=per_day_df,
        holdout_stats_at_r_safe=safe,
        holdout_stats_at_r_hard=hard,
        sizing_convention=top1.amended_gate.sizing_convention,
        causal_audit_clean=False,
    )

    new_top1 = replace(top1, amended_gate=new_gate)
    new_results = tuple(
        new_top1 if r is top1 else r for r in amended_wfo.amended_results
    )
    # Same shape as AmendedWfoSearchResult constructor — preserve base.
    return type(amended_wfo)(base=amended_wfo.base, amended_results=new_results)


__all__ = (
    "DispatchOutcome",
    "maybe_dispatch_step_6",
    "replace_top_1_with_step6_fail",
)
