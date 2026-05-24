"""Orchestration for the heavy_ml_probe sub-protocol.

Owns the end-to-end flow:

  1. Load + validate config
  2. Load Step 1 pool
  3. Apply causal lineage gate (PR-A)
  4. Run AutoML across an 11-fold TimeSeriesSplit (PR-B)
  5. [future] Meta-labeling (PR-C)
  6. [future] Survival (PR-D)
  7. Write artefact set + sha256 manifest

PR-B adds steps 4 — AutoML — and the three new artefacts
(``automl_leaderboard.csv``, ``automl_feature_importance.csv``,
``compute_budget_used.md``). The AutoML stage is auto-skipped when the
input pool lacks the required ``entry_time`` / target columns; this
keeps PR-A's lineage-gate-only tests untouched. Production invocations
always have these columns and AutoML runs.

Skip-reason taxonomy (recorded in the manifest's ``automl`` extras
block so downstream tooling can audit why a run shipped without the
AutoML artefact set):

  * ``ok`` — AutoML ran; artefacts present
  * ``missing_entry_time`` — pool has no ``entry_time`` column; PR-A
    smoke / synthetic-pool path
  * ``missing_target`` — pool has no ``y`` column; same
  * ``no_clean_features`` — lineage gate rejected every column;
    raised as ``AllFeaturesRejected`` from automl.run_automl, recorded
    here as well for the manifest narrative
  * ``holdout_guard_violation`` — pool max ``entry_time`` >= train_end;
    raised as ``HoldoutGuardViolation``; surfaces in the manifest
    before the exception propagates to the CLI

Public surface stable from PR-A:
  - ``PipelineConfig`` (extended fields, additive only)
  - ``PipelineResult`` (extended fields, additive only)
  - ``load_config`` / ``load_pool`` / ``apply_lineage_gate``
  - ``run_pipeline``
  - ``stable_payload_sha256``  — intentional public API for downstream
    determinism verification per PR-A flag-3 disposition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from core.heavy_ml_probe import __version__ as HEAVY_ML_VERSION
from core.heavy_ml_probe.automl import (
    AllFeaturesRejected,
    AutoMLResult,
    HoldoutGuardViolation,
    compute_budget_markdown,
    importance_to_dataframe,
    leaderboard_to_dataframe,
    run_automl,
)
from core.heavy_ml_probe.causal_lineage import (
    LineageGateResult,
    filter_training_columns,
    lineage_summary_markdown,
)
from core.heavy_ml_probe.io import (
    MANIFEST_SCHEMA_VERSION,
    write_csv,
    write_manifest,
    write_text,
)


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved invocation parameters for one pipeline run.

    Mirrors ``configs/heavy_ml_probe/default.yaml``. Fields not used at
    this PR are still parsed so the YAML schema stays locked.
    """

    arc_name: str
    cluster_id: int
    pool_path: Path
    output_root: Path
    config_path: Path
    raw_config: Mapping[str, Any]

    @property
    def step4_dir(self) -> Path:
        sub = str(self.raw_config["output"]["step4_subdir"])
        return self.output_root / sub

    @property
    def step5_dir(self) -> Path:
        sub = str(self.raw_config["output"]["step5_subdir"])
        return self.output_root / sub

    @property
    def train_end(self) -> pd.Timestamp:
        """Holdout cutoff per L_PROTOCOL §1 + dispatch §5 #3."""
        return pd.Timestamp(
            str(self.raw_config["training_window"]["train_end"])
        ).tz_localize("UTC")

    @property
    def automl_max_iter_per_fold(self) -> int:
        return int(self.raw_config["automl"]["max_iter_per_fold"])

    @property
    def automl_n_folds(self) -> int:
        return int(self.raw_config["automl"]["n_folds"])

    @property
    def automl_metric(self) -> str:
        return str(self.raw_config["automl"].get("metric", "roc_auc"))

    @property
    def random_state(self) -> int:
        return int(self.raw_config["determinism"]["random_state"])

    @property
    def n_jobs(self) -> int:
        return int(self.raw_config["determinism"]["n_jobs"])


@dataclass(frozen=True)
class PipelineResult:
    """Aggregated outcome of one pipeline run.

    PR-A: ``lineage_gate``, ``step4_manifest_path``, ``stub_summary_path``.
    PR-B adds: ``automl_result``, ``automl_skip_reason``,
    ``automl_leaderboard_path``, ``automl_importance_path``,
    ``compute_budget_path``.
    """

    cfg: PipelineConfig
    lineage_gate: LineageGateResult
    step4_manifest_path: Path
    stub_summary_path: Path
    # PR-B fields (None when AutoML was skipped or upstream guards fired)
    automl_result: AutoMLResult | None = None
    automl_skip_reason: str = "ok"
    automl_leaderboard_path: Path | None = None
    automl_importance_path: Path | None = None
    compute_budget_path: Path | None = None
    # Reserved for PR-C/D — empty in PR-B.
    meta_label_artefacts: tuple[Path, ...] = field(default_factory=tuple)
    survival_artefacts: tuple[Path, ...] = field(default_factory=tuple)
    step5_manifest_path: Path | None = None


def load_config(
    config_path: Path,
    *,
    arc_name: str,
    cluster_id: int,
    pool_path: Path,
    output_root: Path,
) -> PipelineConfig:
    """Load + minimally-validate a heavy_ml_probe YAML config."""
    cp = Path(config_path)
    if not cp.exists():
        raise FileNotFoundError(f"heavy_ml_probe config not found: {cp}")
    with cp.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"heavy_ml_probe config must be a YAML mapping; got {type(raw).__name__}"
        )
    required_top = {"schema_version", "sub_protocol", "automl",
                    "meta_labeling", "survival", "training_window",
                    "causal_filter", "determinism", "output"}
    missing = required_top - set(raw.keys())
    if missing:
        raise ValueError(
            f"heavy_ml_probe config missing top-level sections: {sorted(missing)}"
        )
    if str(raw["sub_protocol"]["name"]) != "heavy_ml_probe":
        raise ValueError(
            f"config sub_protocol.name must be 'heavy_ml_probe'; got "
            f"{raw['sub_protocol']['name']!r}"
        )
    return PipelineConfig(
        arc_name=arc_name,
        cluster_id=int(cluster_id),
        pool_path=Path(pool_path),
        output_root=Path(output_root),
        config_path=cp,
        raw_config=raw,
    )


def load_pool(pool_path: Path) -> pd.DataFrame:
    """Read the Step 1 pool parquet."""
    p = Path(pool_path)
    if not p.exists():
        raise FileNotFoundError(f"pool parquet not found: {p}")
    return pd.read_parquet(p)


def _build_lineage_dataframe() -> pd.DataFrame:
    """Resolve the canonical lineage table.

    Wraps ``core.features.pipeline.feature_lineage_dataframe`` so tests
    can monkey-patch this thin shim without importing the full v3
    features stack (which triggers panel + cache imports).
    """
    from core.features.pipeline import feature_lineage_dataframe
    return feature_lineage_dataframe()


def apply_lineage_gate(
    pool_columns: list[str],
    cfg: PipelineConfig,
    *,
    lineage_df: pd.DataFrame | None = None,
) -> LineageGateResult:
    """Run the pre-evaluation lineage gate over ``pool_columns``."""
    if lineage_df is None:
        lineage_df = _build_lineage_dataframe()
    cf_cfg = cfg.raw_config["causal_filter"]
    return filter_training_columns(
        pool_columns,
        lineage_df,
        accepted_lineage=tuple(cf_cfg.get("accepted_lineage", ["clean"])),
        exclude_classes=tuple(cf_cfg.get("exclude_classes", []) or []),
    )


# ── PR-B: AutoML stage skip / run decision ───────────────────────────


# Columns the AutoML stage requires. Skip cleanly if either is absent
# rather than crashing — this lets PR-A's lineage-gate-only smoke pools
# coexist with PR-B production pools in the same code path.
AUTOML_REQUIRED_COLUMNS: tuple[str, str] = ("entry_time", "y")


def _should_run_automl(
    pool: pd.DataFrame,
    gate: LineageGateResult,
) -> tuple[bool, str]:
    """Decide whether to run the AutoML stage. Returns (should_run, reason)."""
    if gate.n_accepted == 0:
        return False, "no_clean_features"
    for col in AUTOML_REQUIRED_COLUMNS:
        if col not in pool.columns:
            return False, f"missing_{col}"
    return True, "ok"


def _run_automl_stage(
    cfg: PipelineConfig,
    pool: pd.DataFrame,
    gate: LineageGateResult,
) -> tuple[AutoMLResult | None, str, dict[str, Path]]:
    """Run AutoML + write the three new artefacts. Returns
    (result_or_none, skip_reason, {logical_name: path})."""
    should_run, reason = _should_run_automl(pool, gate)
    if not should_run:
        return None, reason, {}

    try:
        result = run_automl(
            pool=pool,
            used_features=gate.accepted_features,
            train_end=cfg.train_end,
            n_folds=cfg.automl_n_folds,
            max_iter_per_fold=cfg.automl_max_iter_per_fold,
            seed=cfg.random_state,
            metric=cfg.automl_metric,
            n_jobs=cfg.n_jobs,
        )
    except HoldoutGuardViolation:
        # Propagate after recording in the manifest narrative — but the
        # current orchestrator can't write the manifest after raising.
        # So we record the skip reason and re-raise; the CLI surfaces
        # the error via stderr + exit code 1. The manifest's
        # ``automl.skip_reason`` field will reflect ``holdout_guard_violation``
        # only on a SUCCESSFUL guard-violation pre-check path; an
        # actual violation kills the pipeline cleanly.
        raise
    except AllFeaturesRejected:
        # Same: the caller (run_pipeline) catches this and folds it
        # into the manifest. Re-raise here for the orchestrator to
        # handle.
        raise

    paths: dict[str, Path] = {}
    art_cfg = cfg.raw_config["output"]["artefacts"]
    lb_path = cfg.step4_dir / str(art_cfg["automl_leaderboard"])
    imp_path = cfg.step4_dir / str(art_cfg["automl_feature_importance"])
    budget_path = cfg.step4_dir / str(art_cfg["compute_budget_used"])

    write_csv(
        lb_path,
        leaderboard_to_dataframe(result),
        sort_by=["fold", "estimator"],
    )
    write_csv(
        imp_path,
        importance_to_dataframe(result),
        sort_by=["feature"],
    )
    write_text(
        budget_path,
        compute_budget_markdown(result, train_end=cfg.train_end),
    )
    paths["automl_leaderboard"] = lb_path
    paths["automl_feature_importance"] = imp_path
    paths["compute_budget_used"] = budget_path
    return result, "ok", paths


def _stub_summary_md(
    cfg: PipelineConfig,
    gate: LineageGateResult,
    pool_size: int,
    automl_result: AutoMLResult | None,
    automl_skip_reason: str,
) -> str:
    """Render the Step 4 stub summary.

    Stable formatting: no embedded timestamps. PR-A behaviour preserved
    when AutoML is skipped; PR-B appends a stage-status block when
    AutoML ran.
    """
    lines = [
        "# heavy_ml_probe — Step 4 summary",
        "",
        f"- Arc: `{cfg.arc_name}`",
        f"- Cluster ID: `{cfg.cluster_id}`",
        f"- Pool path: `{cfg.pool_path.as_posix()}`",
        f"- Config path: `{cfg.config_path.as_posix()}`",
        f"- heavy_ml_probe version: `{HEAVY_ML_VERSION}`",
        f"- Manifest schema version: `{MANIFEST_SCHEMA_VERSION}`",
        "",
        "## Pool",
        "",
        f"- Trade rows loaded: **{pool_size}**",
        "",
        "## Lineage gate",
        "",
        f"- Columns evaluated: **{gate.n_input_columns}**",
        f"- Accepted (clean): **{gate.n_accepted}**",
        f"- Rejected: **{gate.n_rejected}**",
        "",
        "## AutoML stage",
        "",
        f"- Status: **{automl_skip_reason}**",
    ]
    if automl_result is not None:
        ar = automl_result
        lines += [
            f"- FLAML version: `{ar.flaml_version}`",
            f"- Folds total: **{ar.n_folds_total}** (valid AUC: **{ar.n_folds_valid}**)",
            (
                f"- AUC (nanmean ± nanstd): **{ar.auc_mean:.4f} ± {ar.auc_std:.4f}**"
                if pd.notna(ar.auc_mean)
                else "- AUC (nanmean ± nanstd): **NaN ± NaN** (no valid folds)"
            ),
            f"- Total modelcount across folds: **{ar.total_modelcount}**",
            # Wall-clock intentionally NOT logged here — non-deterministic;
            # available on the in-memory AutoMLResult for log/diagnostic use.
        ]
    lines += [
        "",
        "## Pipeline stages",
        "",
        "- [x] PR-A: scaffolding + causal lineage gate + deterministic IO + sha256 manifest",
        f"- [{'x' if automl_result is not None else ' '}] PR-B: AutoML (FLAML, 11-fold TimeSeriesSplit, 1000 evals/fold cap)",
        "- [ ] PR-C: Meta-labeling target (reach +1R MFE before SL)",
        "- [ ] PR-D: Survival models (Cox PH + RSF; A4 adapter)",
        "- [ ] PR-E: Step 5 augmentation hook",
        "- [ ] PR-F: Docs + polish",
        "",
        "_See `docs/dispatches/heavy_ml_probe_build_intent.md` for the full build plan._",
        "",
    ]
    if gate.n_rejected > 0:
        lines.append(lineage_summary_markdown(gate))
    return "\n".join(lines)


def _gate_extras_for_manifest(gate: LineageGateResult) -> dict:
    """Serialise the gate result into the manifest extras block.

    Sorted lists / dicts so the manifest stays byte-identical across
    two runs.
    """
    from collections import Counter
    reasons = Counter(r["reason"] for r in gate.rejected)
    return {
        "lineage_gate": {
            "n_input_columns": int(gate.n_input_columns),
            "n_accepted": int(gate.n_accepted),
            "n_rejected": int(gate.n_rejected),
            "accepted_features": list(gate.accepted_features),
            "rejection_reasons": dict(sorted(reasons.items())),
        }
    }


def _automl_extras_for_manifest(
    result: AutoMLResult | None,
    skip_reason: str,
) -> dict:
    """Serialise the AutoML stage result for the manifest.

    Floats are coerced to native Python; NaN serialised via ``None`` so
    the JSON is strict-spec compliant (JSON does not have NaN).
    """
    def _f(x: float) -> float | None:
        if x is None:
            return None
        try:
            if not pd.notna(x):
                return None
        except TypeError:
            return None
        return float(x)

    if result is None:
        return {"automl": {"skip_reason": str(skip_reason)}}
    return {
        "automl": {
            "skip_reason": str(skip_reason),
            "flaml_version": str(result.flaml_version),
            "metric": str(result.metric),
            "n_folds_total": int(result.n_folds_total),
            "n_folds_valid": int(result.n_folds_valid),
            "auc_mean": _f(result.auc_mean),
            "auc_std": _f(result.auc_std),
            "total_modelcount": int(result.total_modelcount),
            # Wall-clock intentionally omitted — non-deterministic.
            "n_features_used": int(len(result.used_features)),
        }
    }


def run_pipeline(
    cfg: PipelineConfig,
    *,
    lineage_df: pd.DataFrame | None = None,
) -> PipelineResult:
    """Run the heavy_ml_probe pipeline end-to-end.

    PR-A stages: lineage gate + skeleton manifest + stub summary.
    PR-B stages: AutoML across an 11-fold TimeSeriesSplit (when the
    input pool supports it) + AutoML artefacts in the manifest.

    Determinism: identical config + identical pool → byte-identical
    artefacts and (modulo ``created_at``) byte-identical manifest. The
    determinism tests use :func:`stable_payload_sha256` to verify the
    payload sans timestamp.
    """
    pool = load_pool(cfg.pool_path)
    gate = apply_lineage_gate(list(pool.columns), cfg, lineage_df=lineage_df)
    cfg.step4_dir.mkdir(parents=True, exist_ok=True)

    # ── PR-B AutoML stage ────────────────────────────────────────────
    automl_result: AutoMLResult | None = None
    automl_skip_reason: str = "ok"
    automl_paths: dict[str, Path] = {}
    try:
        automl_result, automl_skip_reason, automl_paths = _run_automl_stage(cfg, pool, gate)
    except HoldoutGuardViolation:
        # Record the skip reason in the manifest, then re-raise.
        automl_skip_reason = "holdout_guard_violation"
        _emit_partial_manifest(
            cfg, gate, pool_size=len(pool),
            automl_result=None, automl_skip_reason=automl_skip_reason,
            automl_paths={},
        )
        raise
    except AllFeaturesRejected:
        automl_skip_reason = "no_clean_features"
        _emit_partial_manifest(
            cfg, gate, pool_size=len(pool),
            automl_result=None, automl_skip_reason=automl_skip_reason,
            automl_paths={},
        )
        raise

    # ── Stub summary (PR-A artefact, PR-B extended content) ──────────
    stub_summary_path = cfg.step4_dir / "stub_summary.md"
    write_text(
        stub_summary_path,
        _stub_summary_md(
            cfg, gate, pool_size=len(pool),
            automl_result=automl_result,
            automl_skip_reason=automl_skip_reason,
        ),
    )

    # ── Manifest ─────────────────────────────────────────────────────
    artefact_paths: dict[str, Path] = {"stub_summary": stub_summary_path, **automl_paths}
    manifest_path = cfg.step4_dir / str(cfg.raw_config["output"]["artefacts"]["step4_manifest"])
    write_manifest(
        manifest_path,
        arc_name=cfg.arc_name,
        step="step_4/heavy_ml",
        artefact_paths=artefact_paths,
        extras={
            "heavy_ml_probe_version": HEAVY_ML_VERSION,
            "cluster_id": int(cfg.cluster_id),
            "pool_path": cfg.pool_path.as_posix(),
            "pool_size": int(len(pool)),
            **_gate_extras_for_manifest(gate),
            **_automl_extras_for_manifest(automl_result, automl_skip_reason),
        },
    )

    return PipelineResult(
        cfg=cfg,
        lineage_gate=gate,
        step4_manifest_path=manifest_path,
        stub_summary_path=stub_summary_path,
        automl_result=automl_result,
        automl_skip_reason=automl_skip_reason,
        automl_leaderboard_path=automl_paths.get("automl_leaderboard"),
        automl_importance_path=automl_paths.get("automl_feature_importance"),
        compute_budget_path=automl_paths.get("compute_budget_used"),
    )


def _emit_partial_manifest(
    cfg: PipelineConfig,
    gate: LineageGateResult,
    *,
    pool_size: int,
    automl_result: AutoMLResult | None,
    automl_skip_reason: str,
    automl_paths: dict[str, Path],
) -> None:
    """Write a manifest reflecting the partial run before re-raising.

    Used when a guard fires after the lineage gate but before the
    AutoML artefacts land — gives downstream tooling something to
    inspect even on the failure path.
    """
    cfg.step4_dir.mkdir(parents=True, exist_ok=True)
    stub_summary_path = cfg.step4_dir / "stub_summary.md"
    write_text(
        stub_summary_path,
        _stub_summary_md(
            cfg, gate, pool_size=pool_size,
            automl_result=automl_result,
            automl_skip_reason=automl_skip_reason,
        ),
    )
    artefact_paths: dict[str, Path] = {"stub_summary": stub_summary_path, **automl_paths}
    manifest_path = cfg.step4_dir / str(cfg.raw_config["output"]["artefacts"]["step4_manifest"])
    write_manifest(
        manifest_path,
        arc_name=cfg.arc_name,
        step="step_4/heavy_ml",
        artefact_paths=artefact_paths,
        extras={
            "heavy_ml_probe_version": HEAVY_ML_VERSION,
            "cluster_id": int(cfg.cluster_id),
            "pool_path": cfg.pool_path.as_posix(),
            "pool_size": int(pool_size),
            **_gate_extras_for_manifest(gate),
            **_automl_extras_for_manifest(automl_result, automl_skip_reason),
        },
    )


def stable_payload_sha256(manifest_path: Path) -> str:
    """SHA256 of the manifest with the ``created_at`` field zeroed out.

    Intentional public API for downstream determinism verification per
    PR-A flag-3 disposition. Two consecutive ``run_pipeline`` calls on
    identical inputs produce manifests whose ``stable_payload_sha256``
    values match — the only varying field is ``created_at``, which is
    excluded here.
    """
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    payload["created_at"] = "<determinism-test-sentinel>"
    import hashlib

    blob = json.dumps(payload, sort_keys=True, indent=2)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


__all__ = (
    "AUTOML_REQUIRED_COLUMNS",
    "PipelineConfig",
    "PipelineResult",
    "load_config",
    "load_pool",
    "apply_lineage_gate",
    "run_pipeline",
    "stable_payload_sha256",
)
