"""Orchestration skeleton for the heavy_ml_probe sub-protocol.

Owns the end-to-end flow: load pool → apply causal lineage gate →
[future] AutoML → [future] meta-labeling → [future] survival → write
artefact set + manifest. Steps not yet implemented HALT cleanly with
``NotImplementedError`` carrying the PR they land in (per the build
plan in ``docs/dispatches/heavy_ml_probe_build_intent.md`` §8).

PR-A scope: load config + apply lineage gate + write skeleton manifest
+ summary. Returns enough context for the CLI to print actionable next
steps.

PR-B (automl), PR-C (meta_labeling), PR-D (survival) extend this
module. The public ``run_pipeline`` surface is stable from PR-A; new PRs
add fields to :class:`PipelineResult` and populate currently-empty
artefact slots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from core.heavy_ml_probe import __version__ as HEAVY_ML_VERSION
from core.heavy_ml_probe.causal_lineage import (
    LineageGateResult,
    filter_training_columns,
    lineage_summary_markdown,
)
from core.heavy_ml_probe.io import (
    MANIFEST_SCHEMA_VERSION,
    write_manifest,
    write_text,
)


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved invocation parameters for one pipeline run.

    Subset of ``configs/heavy_ml_probe/default.yaml``. Fields not used
    by PR-A scaffolding are still parsed so the YAML schema stays
    locked from this PR forward.
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


@dataclass(frozen=True)
class PipelineResult:
    """Aggregated outcome of one pipeline run.

    PR-A populates ``lineage_gate``, ``step4_manifest_path``, and
    ``stub_summary_path``. Later PRs fill the empty slots
    (``automl_artefacts``, ``meta_label_artefacts``, ``survival_artefacts``,
    ``step5_manifest_path``).
    """

    cfg: PipelineConfig
    lineage_gate: LineageGateResult
    step4_manifest_path: Path
    stub_summary_path: Path
    # Reserved for PR-B/C/D — empty in PR-A.
    automl_artefacts: tuple[Path, ...] = field(default_factory=tuple)
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
    """Load + minimally-validate a heavy_ml_probe YAML config.

    Validation is intentionally light at PR-A — full schema enforcement
    lands when PR-B/C/D start consuming the AutoML / meta-labeling /
    survival blocks.
    """
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
    """Read the Step 1 pool parquet.

    PR-A only reads the columns; the body of work happens in PR-B+.
    Surfacing the load here gives PR-A's CLI a non-trivial smoke
    capability — we can fail fast on a bad pool path before later PRs
    add dependencies.
    """
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
    """Run the pre-evaluation lineage gate over ``pool_columns``.

    ``lineage_df`` defaults to the registry-backed table; pass in a
    hand-built DataFrame for tests.
    """
    if lineage_df is None:
        lineage_df = _build_lineage_dataframe()
    cf_cfg = cfg.raw_config["causal_filter"]
    return filter_training_columns(
        pool_columns,
        lineage_df,
        accepted_lineage=tuple(cf_cfg.get("accepted_lineage", ["clean"])),
        exclude_classes=tuple(cf_cfg.get("exclude_classes", []) or []),
    )


def _stub_summary_md(
    cfg: PipelineConfig,
    gate: LineageGateResult,
    pool_size: int,
) -> str:
    """Render a deterministic stub summary for PR-A end-to-end runs.

    No timestamps inside the artefact body (only in the manifest's
    ``created_at`` field, which lives one indirection away) so the
    summary stays byte-identical across two consecutive runs.
    """
    lines = [
        "# heavy_ml_probe — Step 4 stub summary (PR-A)",
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
        "## Pipeline stages",
        "",
        "- [x] PR-A: scaffolding + causal lineage gate + deterministic IO + sha256 manifest",
        "- [ ] PR-B: AutoML (FLAML, 11-fold TimeSeriesSplit, 1000 evals/fold cap)",
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
    two runs. Reasons aggregated for quick scan; full per-feature
    rejection list deferred to the stub summary artefact.
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


def run_pipeline(
    cfg: PipelineConfig,
    *,
    lineage_df: pd.DataFrame | None = None,
) -> PipelineResult:
    """Run the heavy_ml_probe pipeline end-to-end (PR-A: lineage gate
    + skeleton manifest only).

    Side effects:
      * Writes ``<step4_dir>/stub_summary.md`` and
        ``<step4_dir>/manifest.json`` to disk.
      * Creates parent directories as needed.

    The function is idempotent on identical inputs by construction
    (deterministic writes via :mod:`core.heavy_ml_probe.io`). Two
    consecutive invocations produce byte-identical artefacts (modulo
    the manifest's ``created_at`` field, which the determinism tests
    intentionally exclude from the sha256 comparison by hashing the
    payload sans that field).
    """
    pool = load_pool(cfg.pool_path)
    gate = apply_lineage_gate(list(pool.columns), cfg, lineage_df=lineage_df)

    cfg.step4_dir.mkdir(parents=True, exist_ok=True)
    stub_summary_path = cfg.step4_dir / "stub_summary.md"
    write_text(stub_summary_path, _stub_summary_md(cfg, gate, pool_size=len(pool)))

    manifest_path = cfg.step4_dir / str(cfg.raw_config["output"]["artefacts"]["step4_manifest"])
    write_manifest(
        manifest_path,
        arc_name=cfg.arc_name,
        step="step_4/heavy_ml",
        artefact_paths={"stub_summary": stub_summary_path},
        extras={
            "heavy_ml_probe_version": HEAVY_ML_VERSION,
            "cluster_id": int(cfg.cluster_id),
            "pool_path": cfg.pool_path.as_posix(),
            "pool_size": int(len(pool)),
            **_gate_extras_for_manifest(gate),
        },
    )

    return PipelineResult(
        cfg=cfg,
        lineage_gate=gate,
        step4_manifest_path=manifest_path,
        stub_summary_path=stub_summary_path,
    )


def stable_payload_sha256(manifest_path: Path) -> str:
    """SHA256 of the manifest with the ``created_at`` field zeroed out.

    Useful for two-run determinism tests where everything except the
    timestamp must match byte-for-byte. Reads the manifest, replaces
    ``created_at`` with a fixed sentinel, re-serialises, hashes.
    """
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    payload["created_at"] = "<determinism-test-sentinel>"
    import hashlib

    blob = json.dumps(payload, sort_keys=True, indent=2)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


__all__ = (
    "PipelineConfig",
    "PipelineResult",
    "load_config",
    "load_pool",
    "apply_lineage_gate",
    "run_pipeline",
    "stable_payload_sha256",
)
