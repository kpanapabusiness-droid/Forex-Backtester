"""Amendment 3 risk-normalised gate logic.

Lands the six Amendment 3 emission items per
``archive/L_PROTOCOL_v3_0_AMENDMENT_3.md`` and
``docs/audits/engine_capability_audit_2026_05.md`` §"Amendment 3 —
Risk-normalised gates":

  - scalability bounds (`r_safe`, `r_hard`, `r_min`, `r_max`)
  - scaled DEPLOYABLE / VIABLE gates evaluated at scaled risk
  - per-day max-DD re-evaluation at scaled risk (NOT count scaling)
  - chained max DD at scaled risk
  - sizing-convention check (FAIL on equity_pct without chat approval)
  - priority-ordered failure-mode taxonomy

This module is pure logic — no engine emission, no holdout re-runs.
Inputs:

  - per-fold ``FoldStats`` (existing; at ``r_base``)
  - ``chained_max_dd_base_pct`` — IS + holdout continuous-equity DD
    (provided by ``core.wfo.chained_dd``)
  - per-day max-DD DataFrame — full per-day series at ``r_base``
    (provided by ``core.runners._fold_stats_helpers.compute_per_day_max_dd``)
  - re-run holdout FoldStats at ``r_safe`` and ``r_hard``
    (provided by ``core.wfo.holdout_rerun``)
  - ``sizing_convention`` + ``accept_equity_pct``
  - ``r_base``

Output: :class:`AmendedGateResult` — verdict + ``primary_failure_mode``
+ every Amendment 3 risk-normalised field that the tracker payload
requires.

Backwards compatibility: legacy ``core.wfo.gates.classify_fold_stats``
is preserved unmodified for pre-Amendment-3 callers; this module's
``classify_amended_fold_stats`` is the new entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import pandas as pd

from core.wfo.gates import (
    DD_MAX_DEPLOYABLE_PCT,
    DD_MAX_VIABLE_PCT,
    MAX_DAYS_BREACHING_DAILY_5PCT,
    MEAN_FOLD_RATIO_MIN_VIABLE,
    MIN_TRADES_PER_FOLD,
    WORST_FOLD_RATIO_MIN_DEPLOYABLE,
    FoldStats,
)

# ── Scalability bounds (locked per Amendment 3 §"Scalability bounds") ─

R_MIN: float = 0.0015   # 0.15% lower bound on r_safe / r_hard
R_MAX: float = 0.0200   # 2.00% upper bound on r_safe / r_hard
CHAINED_DD_MAX_PCT: float = 0.10   # 10% chained DD cap at scaled risk

DEFAULT_R_BASE: float = 0.005   # KH-24 / L arc convention


# ── Verdicts + failure modes ────────────────────────────────────────


class AmendedVerdict(Enum):
    """L_PROTOCOL §3 verdicts under Amendment 3 risk-normalised gates."""

    PASS_DEPLOYABLE = "pass_deployable"
    PASS_VIABLE = "pass_viable"
    FAIL = "fail"


class PrimaryFailureMode(Enum):
    """Amendment 3 §"Failure-mode priority" — ordered enum.

    Listed in priority order: when multiple constraints fail
    simultaneously, ``classify_amended_fold_stats`` returns the FIRST
    matching mode (lower index = higher priority).
    """

    NONE = "none"
    POOL_TOO_SMALL = "pool_too_small"
    STEP5_NOT_SCALABLE = "step5_not_scalable"
    STEP5_DD_ABOVE_GATE = "step5_dd_above_gate"   # defensive — should not occur post-scaling
    STEP5_CHAINED_DD_ABOVE_GATE = "step5_chained_dd_above_gate"
    STEP5_DAILY_DD_BREACH = "step5_daily_dd_breach"
    HOLDOUT_FAIL_AFTER_IS_PASS = "holdout_fail_after_is_pass"
    STEP5_NEGATIVE_FOLDS = "step5_negative_folds"
    STEP5_SIGN_CONSISTENCY_FAIL = "step5_sign_consistency_fail"
    STEP5_TRADE_COUNT_BELOW_GATE = "step5_trade_count_below_gate"
    STEP5_WF_ROI_BELOW_GATE_AFTER_SCALING = "step5_wf_roi_below_gate_after_scaling"
    STEP5_RATIO_BELOW_GATE_AFTER_SCALING = "step5_ratio_below_gate_after_scaling"
    STEP6_CAUSAL_AUDIT_FAIL = "step6_causal_audit_fail"


# ── Result dataclasses ───────────────────────────────────────────────


@dataclass(frozen=True)
class ScalingFactors:
    """Computed scaling factors per Amendment 3 §"Scaling rule"."""

    worst_fold_dd_base_pct: float
    k_safe: float
    k_hard: float
    r_safe_pct: float
    r_hard_pct: float
    scalable_to_safe: bool
    scalable_to_hard: bool


@dataclass(frozen=True)
class AmendedGateResult:
    """Verdict + every Amendment 3 risk-normalised field.

    Mirrors the ``§1 tracker_payload.best_architecture`` schema in
    ``docs/templates/ARC_CLOSURE_TEMPLATE.md`` v1.2 so closure writers
    can drop these fields straight in.

    ``primary_failure_mode`` is ``PrimaryFailureMode.NONE`` on PASS.
    """

    verdict: AmendedVerdict
    primary_failure_mode: PrimaryFailureMode
    reason: str

    # Raw-risk metrics (at r_base)
    worst_fold_roi_base_pct: float
    worst_fold_dd_base_pct: float
    worst_fold_ratio: float
    mean_fold_ratio: float
    n_negative_folds: int
    min_trades_per_fold: int
    chained_max_dd_base_pct: float

    # Scaling factors
    k_safe: float
    k_hard: float
    r_safe_pct: float
    r_hard_pct: float
    scalable_to_safe: bool
    scalable_to_hard: bool

    # Scaled metrics (at r_safe and r_hard)
    worst_fold_roi_at_r_safe_pct: float
    worst_fold_roi_at_r_hard_pct: float
    chained_max_dd_at_r_safe_pct: float
    chained_max_dd_at_r_hard_pct: float
    daily_dd_breaches_at_r_safe: int
    daily_dd_breaches_at_r_hard: int

    # Holdout re-run results (may be None if re-run not performed)
    holdout_roi_at_r_safe_pct: float | None
    holdout_dd_at_r_safe_pct: float | None
    holdout_roi_at_r_hard_pct: float | None
    holdout_dd_at_r_hard_pct: float | None

    # Sizing convention
    sizing_convention: str   # "reset_floor" | "equity_pct"


# ── Scaling factor computation ──────────────────────────────────────


def compute_scaling_factors(
    worst_fold_dd_base_pct: float,
    *,
    r_base: float = DEFAULT_R_BASE,
) -> ScalingFactors:
    """Compute Amendment 3 §"Scaling rule" k / r values from worst-fold DD at r_base.

    ``k_safe = 8.0 / worst_fold_dd_base`` and
    ``k_hard = 10.0 / worst_fold_dd_base`` are in PERCENTAGE-POINT units
    (the source of all the linear-scaling-works arithmetic — DD scales
    linearly with risk under reset-floor sizing). The output
    ``r_safe_pct`` / ``r_hard_pct`` are in DECIMAL FRACTIONS (e.g.
    ``0.0050`` = 0.5%); matches the convention used everywhere else
    in the engine.

    Scalability bounds locked at ``[R_MIN, R_MAX]`` = ``[0.15%, 2.00%]``.

    Edge case: ``worst_fold_dd_base_pct <= 0`` → ``k = ∞`` →
    ``scalable_to_safe = False``, ``scalable_to_hard = False`` per
    Amendment 3 §"Scalability bounds" edge case.
    """
    # worst_fold_dd_base_pct is in DECIMAL FRACTION (e.g. 0.0922 = 9.22%).
    # The Amendment 3 spec writes 8.0 / 10.0 as the threshold; that's
    # percentage-points, so we compare DD in pp too: dd_pp = dd_frac * 100.
    dd_pp = worst_fold_dd_base_pct * 100.0
    if dd_pp <= 0:
        return ScalingFactors(
            worst_fold_dd_base_pct=worst_fold_dd_base_pct,
            k_safe=float("inf"),
            k_hard=float("inf"),
            r_safe_pct=float("inf"),
            r_hard_pct=float("inf"),
            scalable_to_safe=False,
            scalable_to_hard=False,
        )
    k_safe = 8.0 / dd_pp
    k_hard = 10.0 / dd_pp
    r_safe = r_base * k_safe
    r_hard = r_base * k_hard
    scalable_to_safe = R_MIN <= r_safe <= R_MAX
    scalable_to_hard = R_MIN <= r_hard <= R_MAX
    return ScalingFactors(
        worst_fold_dd_base_pct=worst_fold_dd_base_pct,
        k_safe=k_safe,
        k_hard=k_hard,
        r_safe_pct=r_safe,
        r_hard_pct=r_hard,
        scalable_to_safe=scalable_to_safe,
        scalable_to_hard=scalable_to_hard,
    )


# ── Daily DD per-day re-evaluation ──────────────────────────────────


def count_daily_breaches_at_scaled_risk(
    per_day_max_dd_df: pd.DataFrame,
    k: float,
    *,
    threshold_pct: float = 0.05,
) -> int:
    """Count UTC trading days where ``day_max_dd × k ≥ threshold_pct``.

    Per Amendment 3 §"Daily DD measurement": NOT count scaling.
    Each day's max-DD at ``r_base`` is multiplied by ``k`` and
    compared against the 5% threshold; days above threshold are
    counted. Linear-scaling-of-DD assumption holds under reset-floor
    sizing only.

    ``per_day_max_dd_df`` must have column ``day_max_dd_base_pct``
    (decimal fraction; 0.05 = 5%).
    """
    if per_day_max_dd_df is None or per_day_max_dd_df.empty:
        return 0
    scaled = per_day_max_dd_df["day_max_dd_base_pct"].astype(float) * float(k)
    return int((scaled >= float(threshold_pct)).sum())


# ── Main gate logic ─────────────────────────────────────────────────


def classify_amended_fold_stats(
    folds: Sequence[FoldStats],
    *,
    chained_max_dd_base_pct: float,
    per_day_max_dd_df: pd.DataFrame | None,
    holdout_stats_at_r_safe: FoldStats | None,
    holdout_stats_at_r_hard: FoldStats | None,
    sizing_convention: str = "reset_floor",
    accept_equity_pct: bool = False,
    r_base: float = DEFAULT_R_BASE,
    causal_audit_clean: bool = True,
) -> AmendedGateResult:
    """Apply Amendment 3 risk-normalised gate logic.

    Returns an :class:`AmendedGateResult` with verdict, ``primary_failure_mode``
    (assigned per Amendment 3 §"Failure-mode priority"), and every
    risk-normalised field the closure ``§1 tracker_payload.best_architecture``
    schema requires.

    Step 6 is LAZY per protocol §2 — ``causal_audit_clean`` defaults to
    True; orchestrator passes the real value once Step 6 has been
    invoked on a candidate that cleared all other constraints.
    """
    # ── Pre-conditions ────────────────────────────────────────────────
    if not folds:
        return _build_fail_result(
            failure_mode=PrimaryFailureMode.POOL_TOO_SMALL,
            reason="no folds evaluated",
            folds=tuple(folds),
            chained_max_dd_base_pct=chained_max_dd_base_pct,
            scaling=ScalingFactors(
                worst_fold_dd_base_pct=0.0, k_safe=float("inf"), k_hard=float("inf"),
                r_safe_pct=float("inf"), r_hard_pct=float("inf"),
                scalable_to_safe=False, scalable_to_hard=False,
            ),
            per_day_max_dd_df=per_day_max_dd_df,
            holdout_safe=holdout_stats_at_r_safe,
            holdout_hard=holdout_stats_at_r_hard,
            sizing_convention=sizing_convention,
        )

    # Per-fold raw-risk roll-ups
    rois = [f.roi_pct for f in folds]
    dds = [f.max_dd_pct for f in folds]
    ratios = [f.roi_dd_ratio for f in folds]
    trades = [f.n_trades for f in folds]

    worst_fold_roi = min(rois)
    worst_fold_dd = max(dds)
    worst_fold_ratio = min(ratios)
    mean_fold_ratio = sum(ratios) / len(ratios)
    n_negative = sum(1 for r in rois if r < 0)
    min_trades = min(trades)

    # Scaling factors
    scaling = compute_scaling_factors(worst_fold_dd, r_base=r_base)

    # Daily DD evaluation
    daily_breaches_safe = (
        count_daily_breaches_at_scaled_risk(per_day_max_dd_df, scaling.k_safe)
        if per_day_max_dd_df is not None
        else 0
    )
    daily_breaches_hard = (
        count_daily_breaches_at_scaled_risk(per_day_max_dd_df, scaling.k_hard)
        if per_day_max_dd_df is not None
        else 0
    )

    chained_dd_at_safe = chained_max_dd_base_pct * scaling.k_safe
    chained_dd_at_hard = chained_max_dd_base_pct * scaling.k_hard
    wf_roi_at_safe = worst_fold_roi * scaling.k_safe
    wf_roi_at_hard = worst_fold_roi * scaling.k_hard

    def _fail(mode: PrimaryFailureMode, reason: str) -> AmendedGateResult:
        return AmendedGateResult(
            verdict=AmendedVerdict.FAIL,
            primary_failure_mode=mode,
            reason=reason,
            worst_fold_roi_base_pct=worst_fold_roi,
            worst_fold_dd_base_pct=worst_fold_dd,
            worst_fold_ratio=worst_fold_ratio,
            mean_fold_ratio=mean_fold_ratio,
            n_negative_folds=n_negative,
            min_trades_per_fold=min_trades,
            chained_max_dd_base_pct=chained_max_dd_base_pct,
            k_safe=scaling.k_safe,
            k_hard=scaling.k_hard,
            r_safe_pct=scaling.r_safe_pct,
            r_hard_pct=scaling.r_hard_pct,
            scalable_to_safe=scaling.scalable_to_safe,
            scalable_to_hard=scaling.scalable_to_hard,
            worst_fold_roi_at_r_safe_pct=wf_roi_at_safe,
            worst_fold_roi_at_r_hard_pct=wf_roi_at_hard,
            chained_max_dd_at_r_safe_pct=chained_dd_at_safe,
            chained_max_dd_at_r_hard_pct=chained_dd_at_hard,
            daily_dd_breaches_at_r_safe=daily_breaches_safe,
            daily_dd_breaches_at_r_hard=daily_breaches_hard,
            holdout_roi_at_r_safe_pct=(
                holdout_stats_at_r_safe.roi_pct
                if holdout_stats_at_r_safe is not None else None
            ),
            holdout_dd_at_r_safe_pct=(
                holdout_stats_at_r_safe.max_dd_pct
                if holdout_stats_at_r_safe is not None else None
            ),
            holdout_roi_at_r_hard_pct=(
                holdout_stats_at_r_hard.roi_pct
                if holdout_stats_at_r_hard is not None else None
            ),
            holdout_dd_at_r_hard_pct=(
                holdout_stats_at_r_hard.max_dd_pct
                if holdout_stats_at_r_hard is not None else None
            ),
            sizing_convention=sizing_convention,
        )

    # ── Priority-ordered evaluation per Amendment 3 §"Failure-mode priority" ──

    # 2. step5_not_scalable: sizing convention + scalability bounds
    if sizing_convention == "equity_pct" and not accept_equity_pct:
        return _fail(
            PrimaryFailureMode.STEP5_NOT_SCALABLE,
            f"sizing_convention={sizing_convention!r} is not approved for "
            "linear DD scaling; chat must approve via accept_equity_pct=True",
        )
    if not scaling.scalable_to_safe and not scaling.scalable_to_hard:
        return _fail(
            PrimaryFailureMode.STEP5_NOT_SCALABLE,
            f"r_safe={scaling.r_safe_pct:.4%} / r_hard={scaling.r_hard_pct:.4%} "
            f"outside [{R_MIN:.4%}, {R_MAX:.4%}] — config not scalable",
        )

    # 3. step5_dd_above_gate (defensive — should not occur post-scaling)
    if scaling.scalable_to_safe and (worst_fold_dd * scaling.k_safe) > DD_MAX_DEPLOYABLE_PCT + 1e-9:
        return _fail(
            PrimaryFailureMode.STEP5_DD_ABOVE_GATE,
            "defensive trigger: worst-fold DD at r_safe exceeds 8% — "
            "scaling math invariant broken",
        )

    # 4. step5_chained_dd_above_gate
    # Evaluate against r_safe for DEPLOYABLE; if not scalable to safe,
    # fall through to hard for VIABLE.
    if scaling.scalable_to_safe and chained_dd_at_safe > CHAINED_DD_MAX_PCT + 1e-9:
        return _fail(
            PrimaryFailureMode.STEP5_CHAINED_DD_ABOVE_GATE,
            f"chained max DD at r_safe = {chained_dd_at_safe:.4%} > "
            f"{CHAINED_DD_MAX_PCT:.0%}",
        )
    if (not scaling.scalable_to_safe) and scaling.scalable_to_hard and chained_dd_at_hard > CHAINED_DD_MAX_PCT + 1e-9:
        return _fail(
            PrimaryFailureMode.STEP5_CHAINED_DD_ABOVE_GATE,
            f"chained max DD at r_hard = {chained_dd_at_hard:.4%} > "
            f"{CHAINED_DD_MAX_PCT:.0%}",
        )

    # 5. step5_daily_dd_breach (exactly 0 breaches at either tier per
    # Amendment 3 §"Daily DD measurement" tolerance)
    if scaling.scalable_to_safe and daily_breaches_safe > MAX_DAYS_BREACHING_DAILY_5PCT:
        return _fail(
            PrimaryFailureMode.STEP5_DAILY_DD_BREACH,
            f"{daily_breaches_safe} days breach 5% daily-DD at r_safe — "
            "exactly 0 required",
        )
    if (not scaling.scalable_to_safe) and scaling.scalable_to_hard and daily_breaches_hard > MAX_DAYS_BREACHING_DAILY_5PCT:
        return _fail(
            PrimaryFailureMode.STEP5_DAILY_DD_BREACH,
            f"{daily_breaches_hard} days breach 5% daily-DD at r_hard — "
            "exactly 0 required",
        )

    # 7-8. step5_negative_folds / step5_trade_count_below_gate
    if scaling.scalable_to_safe and n_negative > 0:
        # DEPLOYABLE forbids any negative fold — failover to VIABLE check
        pass   # handled below in tier-selection logic
    if min_trades < MIN_TRADES_PER_FOLD:
        return _fail(
            PrimaryFailureMode.STEP5_TRADE_COUNT_BELOW_GATE,
            f"fold with only {min_trades} trades; need ≥ {MIN_TRADES_PER_FOLD}",
        )

    # 9. step5_wf_roi_below_gate_after_scaling
    if scaling.scalable_to_safe and wf_roi_at_safe <= 0:
        return _fail(
            PrimaryFailureMode.STEP5_WF_ROI_BELOW_GATE_AFTER_SCALING,
            f"worst-fold ROI at r_safe = {wf_roi_at_safe:.4%} ≤ 0",
        )

    # 10. step5_ratio_below_gate_after_scaling
    # (ratio is invariant under linear scaling; check at any tier)
    if worst_fold_ratio < WORST_FOLD_RATIO_MIN_DEPLOYABLE:
        # Defer to VIABLE check below
        pass

    # 6. holdout_fail_after_is_pass — checked when holdout re-runs supplied
    holdout_safe_pass = _holdout_passes_deploy_tier(holdout_stats_at_r_safe)
    holdout_hard_pass = _holdout_passes_viable_tier(holdout_stats_at_r_hard)

    # ── Tier selection: PASS-DEPLOYABLE first, then PASS-VIABLE ──────

    deployable_ok = (
        scaling.scalable_to_safe
        and worst_fold_ratio >= WORST_FOLD_RATIO_MIN_DEPLOYABLE
        and n_negative == 0
        and wf_roi_at_safe > 0
        and chained_dd_at_safe <= CHAINED_DD_MAX_PCT + 1e-9
        and daily_breaches_safe == 0
        and causal_audit_clean
        and (holdout_stats_at_r_safe is None or holdout_safe_pass)
    )
    if deployable_ok:
        return _ok_result(
            verdict=AmendedVerdict.PASS_DEPLOYABLE,
            reason=(
                f"PASS-DEPLOYABLE at r_safe={scaling.r_safe_pct:.4%}: "
                f"worst-fold ratio {worst_fold_ratio:.2f}, "
                f"chained DD {chained_dd_at_safe:.4%}, "
                f"0 daily breaches"
            ),
            folds=tuple(folds), scaling=scaling,
            chained_max_dd_base_pct=chained_max_dd_base_pct,
            per_day_max_dd_df=per_day_max_dd_df,
            holdout_safe=holdout_stats_at_r_safe,
            holdout_hard=holdout_stats_at_r_hard,
            sizing_convention=sizing_convention,
        )

    # If DEPLOYABLE failed because of holdout, surface that mode explicitly
    if (
        scaling.scalable_to_safe
        and worst_fold_ratio >= WORST_FOLD_RATIO_MIN_DEPLOYABLE
        and n_negative == 0
        and wf_roi_at_safe > 0
        and chained_dd_at_safe <= CHAINED_DD_MAX_PCT + 1e-9
        and daily_breaches_safe == 0
        and causal_audit_clean
        and holdout_stats_at_r_safe is not None
        and not holdout_safe_pass
    ):
        return _fail(
            PrimaryFailureMode.HOLDOUT_FAIL_AFTER_IS_PASS,
            "IS cleared all DEPLOYABLE constraints; holdout at r_safe failed",
        )

    viable_ok = (
        scaling.scalable_to_hard
        and n_negative <= 1
        and worst_fold_ratio >= WORST_FOLD_RATIO_MIN_DEPLOYABLE
        and mean_fold_ratio >= MEAN_FOLD_RATIO_MIN_VIABLE
        and chained_dd_at_hard <= CHAINED_DD_MAX_PCT + 1e-9
        and daily_breaches_hard == 0
        and causal_audit_clean
        and (holdout_stats_at_r_hard is None or holdout_hard_pass)
    )
    if viable_ok:
        return _ok_result(
            verdict=AmendedVerdict.PASS_VIABLE,
            reason=(
                f"PASS-VIABLE at r_hard={scaling.r_hard_pct:.4%}: "
                f"worst-fold ratio {worst_fold_ratio:.2f}, "
                f"mean-fold ratio {mean_fold_ratio:.2f}, "
                f"{n_negative} negative fold(s) permitted"
            ),
            folds=tuple(folds), scaling=scaling,
            chained_max_dd_base_pct=chained_max_dd_base_pct,
            per_day_max_dd_df=per_day_max_dd_df,
            holdout_safe=holdout_stats_at_r_safe,
            holdout_hard=holdout_stats_at_r_hard,
            sizing_convention=sizing_convention,
        )

    # Pick the most-relevant FAIL reason in priority order
    if n_negative > 0:
        return _fail(
            PrimaryFailureMode.STEP5_NEGATIVE_FOLDS,
            f"{n_negative} negative fold(s); DEPLOYABLE permits 0, VIABLE permits ≤ 1",
        )
    if worst_fold_ratio < WORST_FOLD_RATIO_MIN_DEPLOYABLE:
        return _fail(
            PrimaryFailureMode.STEP5_RATIO_BELOW_GATE_AFTER_SCALING,
            f"worst-fold ratio {worst_fold_ratio:.2f} < {WORST_FOLD_RATIO_MIN_DEPLOYABLE}",
        )
    if not causal_audit_clean:
        return _fail(
            PrimaryFailureMode.STEP6_CAUSAL_AUDIT_FAIL,
            "Step 6 causal audit failed for load-bearing feature",
        )
    return _fail(
        PrimaryFailureMode.STEP5_RATIO_BELOW_GATE_AFTER_SCALING,
        f"neither PASS tier met: ratio {worst_fold_ratio:.2f}, "
        f"mean {mean_fold_ratio:.2f}, {n_negative} negative folds",
    )


# ── Helpers ─────────────────────────────────────────────────────────


def _holdout_passes_deploy_tier(stats: FoldStats | None) -> bool:
    """Holdout one-fold gate per L_PROTOCOL §3 PASS-DEPLOYABLE rule."""
    if stats is None:
        return False
    return (
        stats.roi_pct > 0
        and stats.max_dd_pct <= DD_MAX_DEPLOYABLE_PCT + 1e-9
        and stats.days_breaching_daily_5pct == 0
        and stats.n_trades >= MIN_TRADES_PER_FOLD
    )


def _holdout_passes_viable_tier(stats: FoldStats | None) -> bool:
    """Holdout one-fold gate per L_PROTOCOL §3 PASS-VIABLE rule."""
    if stats is None:
        return False
    return (
        stats.roi_pct > 0
        and stats.max_dd_pct <= DD_MAX_VIABLE_PCT + 1e-9
        and stats.days_breaching_daily_5pct == 0
        and stats.n_trades >= MIN_TRADES_PER_FOLD
    )


def _ok_result(
    *,
    verdict: AmendedVerdict,
    reason: str,
    folds: tuple[FoldStats, ...],
    scaling: ScalingFactors,
    chained_max_dd_base_pct: float,
    per_day_max_dd_df: pd.DataFrame | None,
    holdout_safe: FoldStats | None,
    holdout_hard: FoldStats | None,
    sizing_convention: str,
) -> AmendedGateResult:
    rois = [f.roi_pct for f in folds]
    dds = [f.max_dd_pct for f in folds]
    ratios = [f.roi_dd_ratio for f in folds]
    trades = [f.n_trades for f in folds]
    return AmendedGateResult(
        verdict=verdict,
        primary_failure_mode=PrimaryFailureMode.NONE,
        reason=reason,
        worst_fold_roi_base_pct=min(rois),
        worst_fold_dd_base_pct=max(dds),
        worst_fold_ratio=min(ratios),
        mean_fold_ratio=sum(ratios) / len(ratios),
        n_negative_folds=sum(1 for r in rois if r < 0),
        min_trades_per_fold=min(trades),
        chained_max_dd_base_pct=chained_max_dd_base_pct,
        k_safe=scaling.k_safe,
        k_hard=scaling.k_hard,
        r_safe_pct=scaling.r_safe_pct,
        r_hard_pct=scaling.r_hard_pct,
        scalable_to_safe=scaling.scalable_to_safe,
        scalable_to_hard=scaling.scalable_to_hard,
        worst_fold_roi_at_r_safe_pct=min(rois) * scaling.k_safe,
        worst_fold_roi_at_r_hard_pct=min(rois) * scaling.k_hard,
        chained_max_dd_at_r_safe_pct=chained_max_dd_base_pct * scaling.k_safe,
        chained_max_dd_at_r_hard_pct=chained_max_dd_base_pct * scaling.k_hard,
        daily_dd_breaches_at_r_safe=(
            count_daily_breaches_at_scaled_risk(per_day_max_dd_df, scaling.k_safe)
            if per_day_max_dd_df is not None else 0
        ),
        daily_dd_breaches_at_r_hard=(
            count_daily_breaches_at_scaled_risk(per_day_max_dd_df, scaling.k_hard)
            if per_day_max_dd_df is not None else 0
        ),
        holdout_roi_at_r_safe_pct=(
            holdout_safe.roi_pct if holdout_safe is not None else None
        ),
        holdout_dd_at_r_safe_pct=(
            holdout_safe.max_dd_pct if holdout_safe is not None else None
        ),
        holdout_roi_at_r_hard_pct=(
            holdout_hard.roi_pct if holdout_hard is not None else None
        ),
        holdout_dd_at_r_hard_pct=(
            holdout_hard.max_dd_pct if holdout_hard is not None else None
        ),
        sizing_convention=sizing_convention,
    )


def _build_fail_result(
    *,
    failure_mode: PrimaryFailureMode,
    reason: str,
    folds: tuple[FoldStats, ...],
    scaling: ScalingFactors,
    chained_max_dd_base_pct: float,
    per_day_max_dd_df: pd.DataFrame | None,
    holdout_safe: FoldStats | None,
    holdout_hard: FoldStats | None,
    sizing_convention: str,
) -> AmendedGateResult:
    rois = [f.roi_pct for f in folds] if folds else [0.0]
    dds = [f.max_dd_pct for f in folds] if folds else [0.0]
    ratios = [f.roi_dd_ratio for f in folds] if folds else [0.0]
    trades = [f.n_trades for f in folds] if folds else [0]
    return AmendedGateResult(
        verdict=AmendedVerdict.FAIL,
        primary_failure_mode=failure_mode,
        reason=reason,
        worst_fold_roi_base_pct=min(rois),
        worst_fold_dd_base_pct=max(dds),
        worst_fold_ratio=min(ratios),
        mean_fold_ratio=sum(ratios) / max(1, len(ratios)),
        n_negative_folds=sum(1 for r in rois if r < 0),
        min_trades_per_fold=min(trades),
        chained_max_dd_base_pct=chained_max_dd_base_pct,
        k_safe=scaling.k_safe,
        k_hard=scaling.k_hard,
        r_safe_pct=scaling.r_safe_pct,
        r_hard_pct=scaling.r_hard_pct,
        scalable_to_safe=scaling.scalable_to_safe,
        scalable_to_hard=scaling.scalable_to_hard,
        worst_fold_roi_at_r_safe_pct=min(rois) * scaling.k_safe,
        worst_fold_roi_at_r_hard_pct=min(rois) * scaling.k_hard,
        chained_max_dd_at_r_safe_pct=chained_max_dd_base_pct * scaling.k_safe,
        chained_max_dd_at_r_hard_pct=chained_max_dd_base_pct * scaling.k_hard,
        daily_dd_breaches_at_r_safe=0,
        daily_dd_breaches_at_r_hard=0,
        holdout_roi_at_r_safe_pct=(
            holdout_safe.roi_pct if holdout_safe is not None else None
        ),
        holdout_dd_at_r_safe_pct=(
            holdout_safe.max_dd_pct if holdout_safe is not None else None
        ),
        holdout_roi_at_r_hard_pct=(
            holdout_hard.roi_pct if holdout_hard is not None else None
        ),
        holdout_dd_at_r_hard_pct=(
            holdout_hard.max_dd_pct if holdout_hard is not None else None
        ),
        sizing_convention=sizing_convention,
    )


__all__ = (
    "R_MIN", "R_MAX", "CHAINED_DD_MAX_PCT", "DEFAULT_R_BASE",
    "AmendedVerdict", "PrimaryFailureMode",
    "ScalingFactors", "AmendedGateResult",
    "compute_scaling_factors",
    "count_daily_breaches_at_scaled_risk",
    "classify_amended_fold_stats",
)
