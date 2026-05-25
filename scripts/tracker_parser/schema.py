"""Pydantic models for ARC_CLOSURE.md §1 tracker_payload — v1.0, v1.1, v1.2, v1.3, and v1.3.1 schemas + normalisation.

Schema versions:
- v1.0 — pre-Amendment 3 (legacy field names `worst_fold_roi_pct`, `worst_fold_dd_pct`)
- v1.1 — post-Amendment 3 (renamed to `*_base_pct`, plus risk-normalised fields)
- v1.2 — deployment-spec addition (adds `config_artefact_path`, `deployment_spec_section_present` to
  `best_architecture`). PASS-verdict closures must point to a canonical config YAML; the parser CLI
  enforces the file's existence + §4 heading presence before applying tracker mappings.
- v1.3 — L_PROTOCOL Amendment 4 (Step 6 causal-audit framework). Adds top-level `step_6` block
  to `tracker_payload`. REQUIRED for any v1.3 PASS verdict. Phase 2 tightening: any PASS verdict
  with ``closed_timestamp > 2026-05-23T06:20:59Z`` (PR-186 merge) MUST carry Amendment 3 fields
  in `best_architecture`; v1.3 PASS verdicts MUST additionally have a `step_6` block with
  `overall_passed: true`. Pre-cutoff closures grandfathered.
- v1.3.1 — L_PROTOCOL Amendment 5 (AUC-gated A2/A6 architecture selection). Adds top-level
  OPTIONAL field ``architectures_skipped_by_amendment_5`` (subset of ``{A1..A6}``, may be ``[]``).
  v1.3.1 shares ``template_version: v1.3`` declaration with v1.3 — the field's presence is the
  discriminator. Phase 2 tightening: any PASS verdict with ``closed_timestamp >
  AMENDMENT_5_CUTOFF_ISO`` MUST carry the field. Pre-cutoff closures grandfathered. Cutoff
  placeholder is the ratification date; backfill with this PR's merge timestamp post-merge.

Detection precedence (see `detect_schema_version`):
1. `template_version` field present → use that
2. Any v1.3-exclusive field present (top-level `step_6`) → v1.3
3. Any v1.2-exclusive field present → v1.2
4. Any v1.1-exclusive field present → v1.1
5. Else → v1.0

`normalize_to_v13()` returns a v1.3-shaped dict that mapping logic consumes uniformly. The mapping
layer reads v1.0/v1.1-era fields + the new `step_6` block; v1.2-exclusive fields are validated at
the CLI layer before tracker mutations begin.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SchemaVersion = Literal["1.0", "1.1", "1.2", "1.3"]

# PR-186 merge cutoff for Phase 2 tightening (per chat Q7).
# After this timestamp, PASS verdicts must carry Amendment 3 fields.
PHASE_2_CUTOFF_ISO: str = "2026-05-23T06:20:59Z"

# L_PROTOCOL Amendment 5 cutoff for Phase 2 tightening on
# `architectures_skipped_by_amendment_5`. After this timestamp, PASS verdicts
# MUST carry the field (may be `[]`). Backfilled with PR #194 merge timestamp
# (mirrors PHASE_2_CUTOFF_ISO / Amendment 3 backfill pattern).
AMENDMENT_5_CUTOFF_ISO: str = "2026-05-25T02:03:13Z"

# L_PROTOCOL Amendment 5.1 cutoff. After this timestamp, PASS closures with
# ≥2 candidate clusters surviving Step 3 MUST either run A5 or cite
# "a5_gate_4_admission_blocked_by_no_pass_tier_constituent" in
# architectures_skipped_by_amendment_5. Backfilled with PR #201 merge timestamp.
AMENDMENT_5_1_CUTOFF_ISO: str = "2026-05-25T05:29:01Z"

# L_PROTOCOL Amendment 3.1 cutoff (r_max reframed as deployment cap, not gate).
# Placeholder. Backfill with PR merge timestamp post-merge.
# Closures with closed_timestamp >= AMENDMENT_3_1_CUTOFF_ISO and
# r_safe_capped_at_rmax == true OR r_hard_capped_at_rmax == true
# MUST also have r_safe_intrinsic_pct / r_hard_intrinsic_pct populated.
# Pre-cutoff closures grandfathered.
AMENDMENT_3_1_CUTOFF_ISO: str = "2026-05-25T00:00:00Z"

V11_EXCLUSIVE_FIELDS = {
    "worst_fold_dd_base_pct",
    "worst_fold_roi_base_pct",
    "chained_max_dd_base_pct",
    "k_safe",
    "k_hard",
    "r_safe_pct",
    "r_hard_pct",
    "scalable_to_safe",
    "scalable_to_hard",
}

V12_EXCLUSIVE_FIELDS = {
    "config_artefact_path",
    "deployment_spec_section_present",
}

# v1.3-exclusive payload-level fields (NOT inside best_architecture).
V13_EXCLUSIVE_TOP_LEVEL_FIELDS = {
    "step_6",
}

VALID_FAILURE_MODES = {
    "pool_too_small",
    "no_clusters_separable",
    "no_capturable_cluster",
    "entry_feature_auc_ceiling",
    "step5_not_scalable",
    "step5_wf_roi_below_gate_after_scaling",
    "step5_ratio_below_gate_after_scaling",
    "step5_chained_dd_above_gate",
    "step5_daily_dd_breach",
    "step5_negative_folds",
    "step5_sign_consistency_fail",
    "step5_trade_count_below_gate",
    "step6_causal_audit_fail",
    "selection_bias",
    "holdout_fail_after_is_pass",
    "admit_only_vs_deployment",
    "step5_dd_above_gate",
    "step5_wf_roi_below_gate",
    "other",
    "N/A",
}

VALID_VERDICTS = {
    "PASS-DEPLOYABLE",
    "PASS-VIABLE",
    "FAIL",
    "HALT",
    "DISCOVERY_COMPLETE",
    "PASS-DEPLOYABLE-PROVISIONAL",
    "PASS-VIABLE-PROVISIONAL",
    "PASS-DEPLOYABLE-PENDING-STEP6",
    "PASS-VIABLE-PENDING-STEP6",
}

VALID_ARCHITECTURES = {"A1", "A2", "A3", "A4", "A5", "A6"}

# Reason-string entries accepted in `architectures_skipped_by_amendment_5`
# alongside {A1..A6} architecture IDs. Introduced by L_PROTOCOL Amendment 5.1
# (2026-05-25) to record the case where Gate 4 admission of A5 was blocked
# because no constituent candidate cluster cleared Step 5 PASS-tier under
# Gates 1/2/3. Extend this set as protocol evolves with new amendment-skip
# reason strings.
VALID_ARCHITECTURES_SKIPPED_REASONS = {
    "a5_gate_4_admission_blocked_by_no_pass_tier_constituent",
}


def detect_schema_version(payload: dict[str, Any]) -> SchemaVersion:
    """Return '1.0', '1.1', '1.2', or '1.3' based on the rules in the module docstring.

    Accepts `template_version` values: 'v1.0', 'v1.1', 'v1.2', 'v1.3', '1.0', '1.1', '1.2', '1.3'.
    """
    raw = payload.get("template_version")
    if raw is not None:
        s = str(raw).lstrip("v").lstrip("V").strip()
        if s == "1.3":
            return "1.3"
        if s == "1.2":
            return "1.2"
        if s == "1.1":
            return "1.1"
        if s == "1.0":
            return "1.0"
        raise ValueError(
            f"Unknown template_version {raw!r} — expected one of "
            f"v1.0, v1.1, v1.2, v1.3, 1.0, 1.1, 1.2, 1.3"
        )

    # v1.3 detected via top-level step_6 block presence.
    if any(k in payload for k in V13_EXCLUSIVE_TOP_LEVEL_FIELDS):
        return "1.3"

    best_arch = payload.get("best_architecture") or {}
    if not isinstance(best_arch, dict):
        best_arch = {}
    if any(k in best_arch for k in V12_EXCLUSIVE_FIELDS):
        return "1.2"
    if any(k in best_arch for k in V11_EXCLUSIVE_FIELDS):
        return "1.1"

    return "1.0"


class PoolMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_n: int
    window_start: Any
    window_end: Any
    kh24_co_fire_pct: float | None = None
    configs_evaluated_step5: int
    search_scope_flag: str


class ClusterRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    n: int
    archetype: str
    sl_atr: float
    step3_composite: float
    mfe_p50_r: float
    ww_pp: float
    reach_1r: float
    step4_e_auc: float | None = None
    step4_d1_auc: float | None = None
    outcome: str


class ArchitectureResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    tested: bool
    won: bool
    worst_fold_ratio: float | None = None


class CostPool(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_fraction: float
    mean_r: float


class CostDecomposition(BaseModel):
    model_config = ConfigDict(extra="allow")

    admit_pool: CostPool
    reject_pool: CostPool
    early_exit_pool: CostPool


class BestArchitectureV10(BaseModel):
    """v1.0 best_architecture block — legacy field names."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    cluster: Any = None
    archetype: str | None = None
    config: str | None = None
    sl_atr: float | None = None
    exit_policy: str | None = None
    exposure_cap: Any = None
    worst_fold_ratio: float | None = None
    worst_fold_roi_pct: float | None = None
    worst_fold_dd_pct: float | None = None
    mean_fold_ratio: float | None = None
    mean_fold_roi_pct: float | None = None
    sign_pos_folds: str | None = None
    n_trades_total: int | None = None
    holdout_roi_pct: float | None = None
    holdout_dd_pct: float | None = None
    holdout_passed: bool | None = None
    oracle_worst_ratio: float | None = None
    oracle_real_gap_sharpe: float | None = None
    features_in_winning_config: list[str] = Field(default_factory=list)


class BestArchitectureV11(BaseModel):
    """v1.1 best_architecture block — renamed fields + Amendment 3 risk-normalised fields."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    cluster: Any = None
    archetype: str | None = None
    config: str | None = None
    sl_atr: float | None = None
    exit_policy: str | None = None
    exposure_cap: Any = None
    worst_fold_ratio: float | None = None
    worst_fold_roi_base_pct: float | None = None
    worst_fold_dd_base_pct: float | None = None
    mean_fold_ratio: float | None = None
    mean_fold_roi_pct: float | None = None
    sign_pos_folds: str | None = None
    n_trades_total: int | None = None
    holdout_roi_pct: float | None = None
    holdout_dd_pct: float | None = None
    holdout_passed: bool | None = None
    oracle_worst_ratio: float | None = None
    oracle_real_gap_sharpe: float | None = None
    features_in_winning_config: list[str] = Field(default_factory=list)

    chained_max_dd_base_pct: float | None = None
    per_day_max_dd_artefact_path: str | None = None
    per_day_max_dd_base_summary: dict[str, Any] | None = None
    k_safe: float | None = None
    k_hard: float | None = None
    r_safe_pct: float | None = None
    r_hard_pct: float | None = None
    scalable_to_safe: bool | None = None
    scalable_to_hard: bool | None = None
    # Amendment 3.1 (2026-05-25): r_max reframed as deployment cap.
    # r_safe_pct / r_hard_pct above now record POST-CAP deploy values;
    # the pre-cap intrinsics + cap-activation flags are recorded here.
    # Phase 1 (this PR): accepted as optional on all closures.
    # Phase 2 tightening enforced at the CLI layer for post-cutoff PASS
    # closures with r_safe_capped_at_rmax / r_hard_capped_at_rmax == true.
    r_safe_intrinsic_pct: float | None = None
    r_hard_intrinsic_pct: float | None = None
    r_safe_capped_at_rmax: bool | None = None
    r_hard_capped_at_rmax: bool | None = None
    worst_fold_roi_at_r_safe_pct: float | None = None
    worst_fold_roi_at_r_hard_pct: float | None = None
    chained_max_dd_at_r_safe_pct: float | None = None
    chained_max_dd_at_r_hard_pct: float | None = None
    daily_dd_breaches_at_r_safe: int | None = None
    daily_dd_breaches_at_r_hard: int | None = None
    holdout_roi_at_r_safe_pct: float | None = None
    holdout_dd_at_r_safe_pct: float | None = None
    holdout_roi_at_r_hard_pct: float | None = None
    holdout_dd_at_r_hard_pct: float | None = None
    sizing_convention: str | None = None
    # Chained-DD reconstruction method per PR-186 review item 1.
    # Phase 1 (this PR): accepted as optional.
    # Phase 2 (post-Wave-2 first PASS arc): required for any PASS
    # verdict whose template_version == "1.2" AND closed_timestamp >
    # PR-186 merge date. Old closures grandfathered by closed_timestamp.
    chained_dd_method: str | None = None


class _TrackerPayloadBase(BaseModel):
    model_config = ConfigDict(extra="allow")

    arc_name: str
    signal: str
    tf: str
    sub_protocol: str
    closed_timestamp: Any
    closure_doc_link: str

    verdict: str
    one_line: str
    failed_at_step: Any
    primary_failure_mode: str

    pool_metadata: PoolMetadata

    cost_decomposition: CostDecomposition | None = None

    clusters: dict[str, ClusterRow]
    architectures_tested: list[str]
    architecture_results: dict[str, ArchitectureResult]
    archetypes_observed: list[str]
    cross_arc_tags: list[str] = Field(default_factory=list)

    # L_PROTOCOL Amendment 5 (v1.3.1). Architectures admissible under
    # Amendment 1's archetype-driven rule but skipped under Amendment 5's
    # four-gate AUC-driven rule. OPTIONAL on Phase 1; required for post-cutoff
    # PASS verdicts (enforced at the CLI layer — needs closed_timestamp +
    # cutoff context the model doesn't have). May be `[]` when the
    # Amendment-5 set equals or supersets the Amendment-1 set.
    architectures_skipped_by_amendment_5: list[str] | None = None


class TrackerPayloadV10(_TrackerPayloadBase):
    """v1.0 payload — legacy field names in best_architecture."""

    best_architecture: BestArchitectureV10 | None = None

    def normalize_to_v11(self) -> dict[str, Any]:
        """Return a dict shaped like v1.1: rename two fields, fill v1.1-exclusive fields with None."""
        d = self.model_dump()
        ba = d.get("best_architecture") or {}
        if ba:
            ba["worst_fold_roi_base_pct"] = ba.pop("worst_fold_roi_pct", None)
            ba["worst_fold_dd_base_pct"] = ba.pop("worst_fold_dd_pct", None)
            for f in (
                "chained_max_dd_base_pct",
                "per_day_max_dd_artefact_path",
                "per_day_max_dd_base_summary",
                "k_safe",
                "k_hard",
                "r_safe_pct",
                "r_hard_pct",
                "scalable_to_safe",
                "scalable_to_hard",
                "worst_fold_roi_at_r_safe_pct",
                "worst_fold_roi_at_r_hard_pct",
                "chained_max_dd_at_r_safe_pct",
                "chained_max_dd_at_r_hard_pct",
                "daily_dd_breaches_at_r_safe",
                "daily_dd_breaches_at_r_hard",
                "holdout_roi_at_r_safe_pct",
                "holdout_dd_at_r_safe_pct",
                "holdout_roi_at_r_hard_pct",
                "holdout_dd_at_r_hard_pct",
                "sizing_convention",
                "chained_dd_method",
            ):
                ba.setdefault(f, None)
            d["best_architecture"] = ba
        d["template_version"] = "1.0"
        return d


class TrackerPayloadV11(_TrackerPayloadBase):
    """v1.1 payload — Amendment 3 schema."""

    best_architecture: BestArchitectureV11 | None = None

    def normalize_to_v11(self) -> dict[str, Any]:
        d = self.model_dump()
        d["template_version"] = "1.1"
        return d


class BestArchitectureV12(BestArchitectureV11):
    """v1.2 best_architecture block — adds deployment-spec pointer + flag.

    `config_artefact_path` MUST be non-null for PASS-verdict closures and MUST point to a file
    that exists relative to the repo root. `deployment_spec_section_present` MUST be true for
    PASS-verdict closures. Both validations are enforced at the CLI layer
    (`update_tracker_from_closure.py`), not in Pydantic — the validation needs a closure-doc-path
    + repo-root context that the model doesn't have.
    """

    config_artefact_path: str | None = None
    deployment_spec_section_present: bool | None = None


class TrackerPayloadV12(_TrackerPayloadBase):
    """v1.2 payload — deployment-spec addition."""

    best_architecture: BestArchitectureV12 | None = None

    def normalize_to_v12(self) -> dict[str, Any]:
        d = self.model_dump()
        d["template_version"] = "1.2"
        return d


# ── v1.3 — Step 6 causal-audit framework (Amendment 4) ──


class Step6Categories(BaseModel):
    model_config = ConfigDict(extra="allow")

    lookahead: bool | None = None
    selection_bias: bool | None = None
    execution_realism: bool | None = None
    statistical: bool | None = None
    determinism: bool | None = None
    deployment_readiness: bool | None = None


class Step6Block(BaseModel):
    """v1.3 ``§1 tracker_payload.step_6`` block per ARC_CLOSURE_TEMPLATE v1.3.

    Field semantics:
    - ``ran``: true when Step 6 dispatched (auto OR manual). False = "not run for this closure".
    - ``trigger``: ``auto_pass`` (orchestrator dispatched), ``manual`` (CLI invoked),
      ``not_applicable`` (didn't run).
    - ``overall_passed``: null when ``ran=false``.
    - ``manifest_path``: relative path to `step_6/manifest.json`. Null when ``ran=false``.
    - ``verdict_impact``: ``none`` for PASS audits + manual invocations + ``--no-block``;
      ``downgraded_to_fail`` when auto-dispatch detected a critical failure.
    """

    model_config = ConfigDict(extra="allow")

    ran: bool
    trigger: str  # auto_pass | manual | not_applicable
    overall_passed: bool | None = None
    manifest_path: str | None = None
    categories: Step6Categories = Field(default_factory=Step6Categories)
    critical_failures: list[str] = Field(default_factory=list)
    warnings_count: int = 0
    verdict_impact: str = "none"


VALID_STEP6_TRIGGERS = {"auto_pass", "manual", "not_applicable"}
VALID_STEP6_VERDICT_IMPACTS = {"none", "downgraded_to_fail"}


class BestArchitectureV13(BestArchitectureV12):
    """v1.3 best_architecture block — identical to v1.2 (Amendment 4 added no
    best_architecture-level fields; the new ``step_6`` block is top-level).
    """


class TrackerPayloadV13(_TrackerPayloadBase):
    """v1.3 payload — Amendment 4 Step 6 framework."""

    best_architecture: BestArchitectureV13 | None = None
    step_6: Step6Block | None = None

    def normalize_to_v13(self) -> dict[str, Any]:
        d = self.model_dump()
        d["template_version"] = "1.3"
        return d


def _coerce_legacy_field_names(payload: dict[str, Any]) -> dict[str, Any]:
    """Rename v1.0 best_architecture fields to v1.1 names if present.

    Used when a v1.2 closure retains the legacy `worst_fold_roi_pct` / `worst_fold_dd_pct` names
    (retrofit pattern — §1 changes restricted to additive v1.2 fields). Idempotent and a no-op
    if v1.1 names are already in place.
    """
    ba = payload.get("best_architecture")
    if not isinstance(ba, dict):
        return payload
    if "worst_fold_roi_pct" in ba and "worst_fold_roi_base_pct" not in ba:
        ba["worst_fold_roi_base_pct"] = ba.pop("worst_fold_roi_pct")
    elif "worst_fold_roi_pct" in ba and "worst_fold_roi_base_pct" in ba:
        # Both present — base_pct wins; legacy dropped (rare; retrofit edge case).
        ba.pop("worst_fold_roi_pct")
    if "worst_fold_dd_pct" in ba and "worst_fold_dd_base_pct" not in ba:
        ba["worst_fold_dd_base_pct"] = ba.pop("worst_fold_dd_pct")
    elif "worst_fold_dd_pct" in ba and "worst_fold_dd_base_pct" in ba:
        ba.pop("worst_fold_dd_pct")
    return payload


def parse_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Detect schema version, validate via the matching model, return a normalised dict.

    Returns a v1.1-shaped dict for v1.0/v1.1 closures (legacy compatibility — mapping logic
    consumes a unified v1.1 shape). For v1.2 closures, returns a v1.2-shaped dict (a superset
    of v1.1; mapping logic ignores the two extra fields). For v1.3 closures, returns a
    v1.3-shaped dict (superset of v1.2 + ``step_6`` block).

    For v1.2/v1.3 closures that retain v1.0-style field names in `best_architecture` (the
    retrofit pattern — see `_coerce_legacy_field_names`), legacy names are renamed pre-validation.

    Raises pydantic.ValidationError on schema violations and ValueError on unknown enum values.
    """
    version = detect_schema_version(payload)
    if version == "1.3":
        payload = _coerce_legacy_field_names(payload)
        validated = TrackerPayloadV13.model_validate(payload)
        norm = validated.normalize_to_v13()
    elif version == "1.2":
        payload = _coerce_legacy_field_names(payload)
        validated = TrackerPayloadV12.model_validate(payload)
        norm = validated.normalize_to_v12()
    elif version == "1.1":
        validated = TrackerPayloadV11.model_validate(payload)
        norm = validated.normalize_to_v11()
    else:
        validated = TrackerPayloadV10.model_validate(payload)
        norm = validated.normalize_to_v11()

    failure_mode = norm.get("primary_failure_mode")
    if failure_mode not in VALID_FAILURE_MODES:
        raise ValueError(
            f"primary_failure_mode {failure_mode!r} not in locked enum "
            f"(template §1 primary_failure_mode list)"
        )
    verdict = norm.get("verdict")
    if verdict not in VALID_VERDICTS:
        raise ValueError(
            f"verdict {verdict!r} not in locked enum "
            f"(template §1 verdict list)"
        )
    for arch in norm.get("architectures_tested", []):
        if arch not in VALID_ARCHITECTURES:
            raise ValueError(
                f"architecture {arch!r} not in {{A1…A6}}"
            )

    # Amendment 5 (v1.3.1) field — enum-validate any entries when present.
    # Amendment 5.1 (2026-05-25) extends the accepted vocabulary to include
    # reason strings (see VALID_ARCHITECTURES_SKIPPED_REASONS) alongside the
    # {A1..A6} architecture IDs. Unknown entries are still rejected — preserves
    # the closed-vocabulary discipline of the prior validator.
    skipped = norm.get("architectures_skipped_by_amendment_5")
    if skipped is not None:
        for entry in skipped:
            if (
                entry not in VALID_ARCHITECTURES
                and entry not in VALID_ARCHITECTURES_SKIPPED_REASONS
            ):
                raise ValueError(
                    f"architectures_skipped_by_amendment_5 entry {entry!r} not in "
                    f"{{A1…A6}} ∪ valid reason strings "
                    f"({sorted(VALID_ARCHITECTURES_SKIPPED_REASONS)})"
                )

    # v1.3 step_6 block enum validation
    step6 = norm.get("step_6")
    if isinstance(step6, dict):
        trigger = step6.get("trigger")
        if trigger is not None and trigger not in VALID_STEP6_TRIGGERS:
            raise ValueError(
                f"step_6.trigger {trigger!r} not in {sorted(VALID_STEP6_TRIGGERS)}"
            )
        impact = step6.get("verdict_impact")
        if impact is not None and impact not in VALID_STEP6_VERDICT_IMPACTS:
            raise ValueError(
                f"step_6.verdict_impact {impact!r} not in {sorted(VALID_STEP6_VERDICT_IMPACTS)}"
            )

    return norm
