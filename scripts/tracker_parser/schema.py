"""Pydantic models for ARC_CLOSURE.md §1 tracker_payload — v1.0 and v1.1 schemas + normalisation.

Schema versions:
- v1.0 — pre-Amendment 3 (legacy field names `worst_fold_roi_pct`, `worst_fold_dd_pct`)
- v1.1 — post-Amendment 3 (renamed to `*_base_pct`, plus risk-normalised fields)

Detection precedence (see `detect_schema_version`):
1. `template_version` field present → use that
2. Any v1.1-exclusive field present → v1.1
3. Else → v1.0

`normalize_to_v11()` returns a v1.1-shaped dict that mapping logic consumes uniformly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SchemaVersion = Literal["1.0", "1.1"]

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
}

VALID_ARCHITECTURES = {"A1", "A2", "A3", "A4", "A5", "A6"}


def detect_schema_version(payload: dict[str, Any]) -> SchemaVersion:
    """Return '1.0' or '1.1' based on the rules in the module docstring.

    Accepts `template_version` values: 'v1.0', 'v1.1', '1.0', '1.1'.
    """
    raw = payload.get("template_version")
    if raw is not None:
        s = str(raw).lstrip("v").lstrip("V").strip()
        if s == "1.1":
            return "1.1"
        if s == "1.0":
            return "1.0"
        raise ValueError(
            f"Unknown template_version {raw!r} — expected one of v1.0, v1.1, 1.0, 1.1"
        )

    best_arch = payload.get("best_architecture") or {}
    if not isinstance(best_arch, dict):
        best_arch = {}
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


def parse_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Detect schema version, validate via the matching model, return v1.1-normalised dict.

    Raises pydantic.ValidationError on schema violations and ValueError on unknown enum values.
    """
    version = detect_schema_version(payload)
    model = TrackerPayloadV11 if version == "1.1" else TrackerPayloadV10
    validated = model.model_validate(payload)
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

    return norm
